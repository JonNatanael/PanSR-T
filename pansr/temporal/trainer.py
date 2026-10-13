# ------------------------------------------------------------------------
# TG-PanSR: trainer extension.
#
# TemporalTrainer reuses the original PanSR Trainer (pansr.trainer) and
# overrides exactly the points the temporal protocol changes:
#   1. the train/test data loaders use the temporal mapper;
#   2. the model builder first registers the temporal datasets (the model
#      reads their metadata) and validates the TEMPORAL config;
#   3. explicit parameter groups (backbone multiplier, new-module LR,
#      frozen Pi3 excluded) with a uniqueness assertion;
#   4. deliberate gradient accumulation: one d2 iteration = one optimizer
#      update made of TEMPORAL.OPTIM.ACCUMULATION_STEPS data batches
#      (losses averaged); scheduler / checkpoints / eval count optimizer
#      updates, matching the brief's 10,000-update schedule.
# ------------------------------------------------------------------------
from __future__ import annotations

import itertools
import logging
import time
from typing import Any, Dict, List, Set, Tuple

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    TrainerBase,
    create_ddp_model,
)

from ..trainer import Trainer as PanSRTrainer
from .config import validate_temporal_config
from .dataset import register_temporal_lars
from .mapper import TemporalCOCOPanopticDatasetMapper

logger = logging.getLogger(__name__)

__all__ = ["TemporalTrainer", "build_temporal_parameter_groups"]


# ---------------------------------------------------------------------------
# Parameter groups
# ---------------------------------------------------------------------------
def build_temporal_parameter_groups(cfg, model):
    """Explicit module-ownership parameter groups.

    * Pi3 (``geometry_encoder.*`` frozen parameters): excluded, and
      asserted to require no gradient.
    * New temporal modules (``temporal_fusion.*`` plus any trainable
      ``geometry_encoder.*`` adapters): TEMPORAL.OPTIM.NEW_MODULE_LR.
    * Backbone (``backbone.*``): BASE_LR * BACKBONE_MULTIPLIER.
    * Everything else: BASE_LR.
    * Weight-decay exceptions mirror the original trainer
      (norm modules, embeddings, relative/absolute position biases).

    Asserts that every trainable parameter occurs in exactly one group and
    that no frozen parameter appears at all.
    """
    defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
    norm_module_types = (
        torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm, torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm, torch.nn.LocalResponseNorm,
    )

    new_module_lr = cfg.TEMPORAL.OPTIM.NEW_MODULE_LR
    groups: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    frozen_seen = 0
    counts = {"backbone": 0, "new_temporal": 0, "other": 0}

    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                if module_name.startswith("geometry_encoder"):
                    frozen_seen += 1
                continue
            if value in memo:
                continue
            memo.add(value)

            hyperparams = dict(defaults)
            is_backbone = module_name.startswith("backbone")
            is_new = module_name.startswith("temporal_fusion") or (
                module_name.startswith("geometry_encoder")
            )
            if is_new:
                hyperparams["lr"] = new_module_lr
                counts["new_temporal"] += 1
            elif is_backbone:
                hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                counts["backbone"] += 1
            else:
                counts["other"] += 1

            if ("relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name):
                hyperparams["weight_decay"] = 0.0
            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_NORM
            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_EMBED
            groups.append({"params": [value], **hyperparams})

    # every trainable parameter exactly once; no frozen Pi3 parameter
    trainable = set(p for p in model.parameters() if p.requires_grad)
    grouped: Set[torch.nn.parameter.Parameter] = set()
    for g in groups:
        for p in g["params"]:
            assert p not in grouped, "parameter assigned to two groups"
            grouped.add(p)
    assert grouped == trainable, (
        f"param group mismatch: {len(trainable - grouped)} trainable missing, "
        f"{len(grouped - trainable)} frozen included"
    )
    info = {
        "n_groups": len(groups),
        "n_backbone_params": counts["backbone"],
        "n_new_temporal_params": counts["new_temporal"],
        "n_other_params": counts["other"],
        "n_frozen_pi3_params": frozen_seen,
        "new_module_lr": new_module_lr,
    }
    logger.info(
        "Temporal parameter groups: %d groups | backbone %d (lr=%.3g) | "
        "new temporal %d (lr=%.3g) | other %d (lr=%.3g) | frozen Pi3 params excluded: %d",
        info["n_groups"], counts["backbone"],
        cfg.SOLVER.BASE_LR * cfg.SOLVER.BACKBONE_MULTIPLIER,
        counts["new_temporal"], new_module_lr,
        counts["other"], cfg.SOLVER.BASE_LR, frozen_seen,
    )
    return groups, info


# ---------------------------------------------------------------------------
# Accumulating inner trainers
# ---------------------------------------------------------------------------
class _AccumulatingSimpleTrainer(SimpleTrainer):
    """SimpleTrainer where one d2 iteration = one optimizer update made of
    ``accumulation_steps`` data batches (losses averaged)."""

    def __init__(self, model, data_loader, optimizer, accumulation_steps: int, **kw):
        super().__init__(model, data_loader, optimizer, **kw)
        assert accumulation_steps >= 1
        self.accumulation_steps = int(accumulation_steps)

    def run_step(self):
        assert self.model.training, "[_AccumulatingSimpleTrainer] model in eval mode!"
        start = time.perf_counter()
        n = self.accumulation_steps
        # fresh gradients for this optimizer update (d2's default
        # zero_grad_before_forward=False zeroes right before backward;
        # with accumulation we zero once per update cycle)
        self.optimizer.zero_grad()
        loss_dict_acc: Dict[str, torch.Tensor] = {}
        for _ in range(n):
            data = next(self._data_loader_iter)
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                loss_dict = {"total_loss": loss_dict}
            losses = sum(loss_dict.values()) / n
            losses.backward()
            for k, v in loss_dict.items():
                loss_dict_acc[k] = loss_dict_acc.get(k, torch.tensor(0.0)) + (
                    v.detach() / n
                )
        self.after_backward()
        self._write_metrics(loss_dict_acc, time.perf_counter() - start)
        self.optimizer.step()


class _AccumulatingAMPTrainer(AMPTrainer):
    """AMP variant of :class:`_AccumulatingSimpleTrainer`.

    Ordering per update cycle: zero_grad -> n x (forward under autocast,
    scaled backward) -> after_backward hooks -> unscale+clip (via the
    wrapped optimizer) inside grad_scaler.step -> scaler.update."""

    def __init__(self, model, data_loader, optimizer, accumulation_steps: int, **kw):
        super().__init__(model, data_loader, optimizer, **kw)
        assert accumulation_steps >= 1
        self.accumulation_steps = int(accumulation_steps)

    def run_step(self):
        assert self.model.training, "[_AccumulatingAMPTrainer] model in eval mode!"
        assert torch.cuda.is_available(), "AMP requires CUDA"
        start = time.perf_counter()
        n = self.accumulation_steps
        self.optimizer.zero_grad()
        loss_dict_acc: Dict[str, torch.Tensor] = {}
        with torch.amp.autocast("cuda", dtype=self.precision):
            for _ in range(n):
                data = next(self._data_loader_iter)
                loss_dict = self.model(data)
                if isinstance(loss_dict, torch.Tensor):
                    loss_dict = {"total_loss": loss_dict}
                losses = sum(loss_dict.values()) / n
                self.grad_scaler.scale(losses).backward()
                for k, v in loss_dict.items():
                    loss_dict_acc[k] = loss_dict_acc.get(k, torch.tensor(0.0)) + (
                        v.detach() / n
                    )
        self.after_backward()
        if self.log_grad_scaler:
            from detectron2.utils.events import get_event_storage

            get_event_storage().put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())
        self._write_metrics(loss_dict_acc, time.perf_counter() - start)
        # clipping (when wrapped on the optimizer) runs inside step();
        # GradScaler.step unscales -> steps (with clip) -> updates -> zero_grad
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()


# ---------------------------------------------------------------------------
# TemporalTrainer
# ---------------------------------------------------------------------------
class TemporalTrainer(PanSRTrainer):
    """PanSR Trainer + temporal loaders, parameter groups, accumulation."""

    # ------------------------------------------------------------------
    # model / data
    # ------------------------------------------------------------------
    @classmethod
    def build_model(cls, cfg):
        """Register the temporal datasets (metadata is read by the model)
        and validate the temporal config, then build as usual."""
        if cfg.MODEL.META_ARCHITECTURE == "TemporalPanSR" and cfg.TEMPORAL.ENABLED:
            validate_temporal_config(cfg, is_train=bool(cfg.TEMPORAL.MANIFEST_TRAIN))
            register_temporal_lars(cfg)
        return PanSRTrainer.build_model(cfg)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """AdamW with the explicit temporal parameter groups and the
        original PanSR full-model gradient-clipping semantics."""
        groups, _info = build_temporal_parameter_groups(cfg, model)

        clip_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        clip_enabled = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
            and clip_val > 0.0
        )

        def make_optim(cls_opt):
            if clip_enabled and cls_opt is torch.optim.AdamW:
                class _FullModelClipped(torch.optim.AdamW):
                    def step(self, closure=None):
                        all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                        torch.nn.utils.clip_grad_norm_(all_params, clip_val)
                        super().step(closure)
                return _FullModelClipped
            return cls_opt

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "ADAMW":
            optimizer = make_optim(torch.optim.AdamW)(groups, cfg.SOLVER.BASE_LR)
        elif optimizer_type == "SGD":
            optimizer = make_optim(torch.optim.SGD)(
                groups, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            from detectron2.solver.build import maybe_add_gradient_clipping

            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_train_loader(cls, cfg):
        assert cfg.INPUT.DATASET_MAPPER_NAME == "lars_temporal_panoptic", (
            f"TemporalTrainer expects the 'lars_temporal_panoptic' mapper, "
            f"got '{cfg.INPUT.DATASET_MAPPER_NAME}'."
        )
        mapper = TemporalCOCOPanopticDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """The *test* loader uses the temporal mapper as well (context must
        reach the model at inference; the vanilla mapper has no clip)."""
        mapper = TemporalCOCOPanopticDatasetMapper(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    # ------------------------------------------------------------------
    # distributed / inner trainer
    # ------------------------------------------------------------------
    def _create_ddp_model(self, model):
        # find_unused_parameters: with an explicit per-sample baseline
        # fallback (TEMPORAL.MISSING_POLICY=fallback_baseline) the fusion
        # modules may be skipped for a whole micro-batch; without this,
        # DDP would hang waiting for gradients that never arrive.
        return create_ddp_model(
            model, broadcast_buffers=False, find_unused_parameters=True
        )

    def build_hooks(self):
        hooks = [h for h in super().build_hooks() if h is not None]
        if not self.cfg.TEMPORAL.OPTIM.RUN_FINAL_EVAL:
            from detectron2.engine import hooks as d2_hooks

            # drop the unconditional final-eval hook (see config note)
            hooks = [h for h in hooks if not isinstance(h, d2_hooks.EvalHook)]
        return hooks

    def _build_inner_trainer(self, cfg, model, data_loader, optimizer):
        accum = int(cfg.TEMPORAL.OPTIM.ACCUMULATION_STEPS)
        if accum == 1:
            return super()._build_inner_trainer(cfg, model, data_loader, optimizer)
        if cfg.SOLVER.AMP.ENABLED:
            return _AccumulatingAMPTrainer(
                model, data_loader, optimizer, accumulation_steps=accum
            )
        return _AccumulatingSimpleTrainer(
            model, data_loader, optimizer, accumulation_steps=accum
        )

    # ------------------------------------------------------------------
    def __init__(self, cfg):
        # validate up front (train path: MANIFEST_TRAIN must be set)
        if cfg.MODEL.META_ARCHITECTURE == "TemporalPanSR" and cfg.TEMPORAL.ENABLED:
            validate_temporal_config(cfg, is_train=True)

        # NOTE: we intentionally do NOT call
        # DefaultTrainer.auto_scale_workers here: IMS_PER_BATCH is the
        # global MICRO-batch (d2 convention), and the effective batch
        # accounting below (micro x world x accumulation) is the
        # experiment contract.  (The base PanSR Trainer applies
        # auto_scale_workers; it is a no-op when
        # SOLVER.REFERENCE_WORLD_SIZE == 0, which the LaRS configs use.)
        super().__init__(cfg)

        # effective batch accounting (asserted, not rescaled)
        world = comm.get_world_size()
        micro_global = int(cfg.SOLVER.IMS_PER_BATCH)
        assert micro_global % world == 0, (
            f"IMS_PER_BATCH {micro_global} not divisible by world size {world}"
        )
        accum = int(cfg.TEMPORAL.OPTIM.ACCUMULATION_STEPS)
        effective = micro_global * world * accum
        expected = int(cfg.TEMPORAL.OPTIM.EXPECTED_EFFECTIVE_BATCH)
        if expected:
            assert effective == expected, (
                f"effective target batch {effective} (micro {micro_global} x world "
                f"{world} x accumulation {accum}) != EXPECTED_EFFECTIVE_BATCH {expected}"
            )
        logger.info(
            "Temporal effective target batch: %d (micro %d x world %d x accumulation %d)",
            effective, micro_global, world, accum,
        )

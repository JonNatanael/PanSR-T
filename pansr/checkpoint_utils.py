# ------------------------------------------------------------------------
# TG-PanSR: checkpoint loading helpers (.pth and .safetensors).
#
# detectron2 0.6's DetectionCheckpointer only reads torch .pth files, but
# the canonical PanSR LaRS checkpoint is distributed as .safetensors.  This
# module provides validated loading for both formats, shared by the vanilla
# and temporal trainers and by tools/predictor entry points.
#
# For the temporal model, the *new* module prefixes (geometry_encoder.*,
# temporal_fusion.*) are allowed to be missing when loading an original
# PanSR checkpoint (they are freshly initialized); any other missing or
# shape-mismatched key is fatal.
#
# Legacy PanSR checkpoint compatibility (structural drift, see
# docs/repository_audit.md):
#   * The predictor shares ONE box MLP across all decoder layers
#     (pansr_decoder.py: ``box_embed_layerlist = [_bbox_embed for i in
#     range(num_layers)]`` and ``decoder.bbox_embed = self.bbox_embed``).
#     Checkpoints exported from the original training code therefore store
#     only the six ``predictor._bbox_embed.layers.*`` tensors, while the
#     current code's state_dict contains per-layer alias keys
#     ``predictor.bbox_embed.{i}.layers.*`` and
#     ``predictor.decoder.bbox_embed.{i}.layers.*``.  Both alias families
#     are expanded from the shared source tensor (shape-asserted).
#   * ``predictor.decoder_norm.{weight,bias}`` (final decoder LayerNorm)
#     was added after the checkpoint was exported.  A fresh nn.LayerNorm is
#     initialized to weight=1/bias=0, i.e. the identity map, which exactly
#     reproduces the pre-norm behavior the checkpoint was trained with, so
#     these two keys are allowed to be missing (legacy checkpoints only).
# ------------------------------------------------------------------------
import logging
import os
import re

import torch

logger = logging.getLogger(__name__)

__all__ = ["load_initial_weights", "load_model_weights"]

_NEW_MODULE_PREFIXES = ("geometry_encoder.", "temporal_fusion.")

# predictor.bbox_embed.{i}.layers.{j}.{weight|bias}
# and the aliased family predictor.decoder.bbox_embed.{i}.layers.{j}.{...}
_BBOX_ALIAS_RE = re.compile(
    r"^(?P<base>.+?)\.(?:decoder\.)?bbox_embed\."
    r"(?P<i>\d+)\.layers\.(?P<j>\d+)\.(?P<p>weight|bias)$"
)
# predictor.decoder_norm.{weight|bias}
_DECODER_NORM_RE = re.compile(r"^(?P<base>.+?)\.decoder_norm\.(?P<p>weight|bias)$")


def _strip_wrapper_prefix(state, model_state):
    for prefix in ("module.", "model.", "net."):
        if all(k.startswith(prefix) for k in state):
            stripped = {k[len(prefix):]: v for k, v in state.items()}
            if sum(1 for k in stripped if k in model_state) > max(1, len(stripped) // 2):
                return stripped
    return state


def _expand_bbox_embed_aliases(model_state, state):
    """Expand shared ``_bbox_embed`` checkpoint keys into per-layer aliases.

    Returns ``(expanded_state, n_expanded, legacy_format)``.  Raises
    RuntimeError if an alias key is missing *and* its shared source is not
    present in the checkpoint (a genuine structural mismatch).
    """
    expanded = dict(state)
    n_expanded = 0
    for key in model_state:
        if key in state:
            continue
        m = _BBOX_ALIAS_RE.match(key)
        if m is None:
            continue
        src = f"{m.group('base')}._bbox_embed.layers.{m.group('j')}.{m.group('p')}"
        if src not in state:
            raise RuntimeError(
                f"checkpoint is missing both the per-layer key '{key}' and the "
                f"shared source '{src}'; cannot resolve predictor box-embed alias"
            )
        if state[src].shape != model_state[key].shape:
            raise RuntimeError(
                f"shared source '{src}' has shape {tuple(state[src].shape)}, "
                f"expected {tuple(model_state[key].shape)} for alias '{key}'"
            )
        expanded[key] = state[src]
        n_expanded += 1
    legacy_format = (
        n_expanded > 0
        and not any(_BBOX_ALIAS_RE.match(k) for k in state)
    )
    return expanded, n_expanded, legacy_format


def _legacy_decoder_norm_missing(model_state, state, legacy_format):
    """Keys allowed to be missing because they post-date the checkpoint."""
    allowed = []
    if not legacy_format:
        return allowed
    for key in model_state:
        if key in state:
            continue
        m = _DECODER_NORM_RE.match(key)
        if m is not None:
            allowed.append(key)
    return allowed


def load_initial_weights(model, path: str) -> dict:
    """Load *initial* (not resume) weights with full key validation.

    Returns a report ``{"missing": [...], "allowed_missing": [...],
    "unexpected": [...], "mismatched": [...], "expanded_aliases": int}``.
    Raises RuntimeError on shape mismatches or on missing keys that are not
    new temporal modules or documented legacy-structure allowances.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path, device="cpu")
    else:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "model" in blob and isinstance(blob["model"], dict):
            state = blob["model"]
        elif isinstance(blob, dict) and "state_dict" in blob and isinstance(blob["state_dict"], dict):
            state = blob["state_dict"]
        else:
            state = blob
    if not isinstance(state, dict):
        raise RuntimeError(f"unrecognized checkpoint structure in {path}")

    model = getattr(model, "module", model)  # DDP unwrap
    model_state = model.state_dict()
    state = _strip_wrapper_prefix(state, model_state)
    expanded, n_expanded, legacy_format = _expand_bbox_embed_aliases(model_state, state)

    missing = [k for k in model_state if k not in expanded]
    unexpected = [k for k in expanded if k not in model_state]
    mismatched = [
        k for k in model_state
        if k in expanded and model_state[k].shape != expanded[k].shape
    ]
    if mismatched:
        raise RuntimeError(
            f"checkpoint {path} has shape-mismatched keys, e.g. {mismatched[:5]}"
        )
    allowed_missing = _legacy_decoder_norm_missing(model_state, expanded, legacy_format)
    hard_missing = [
        k
        for k in missing
        if not k.startswith(_NEW_MODULE_PREFIXES)
        and k not in allowed_missing
    ]
    if hard_missing:
        raise RuntimeError(
            f"checkpoint {path} is missing {len(hard_missing)} expected keys, "
            f"e.g. {hard_missing[:5]}"
        )
    model.load_state_dict(
        {k: v for k, v in expanded.items() if k in model_state}, strict=False
    )
    logger.info(
        "Loaded weights from %s: %d model keys, %d missing (%d new modules, "
        "%d legacy-allowed), %d expanded box-embed aliases, %d unexpected",
        os.path.basename(path), len(model_state), len(missing),
        len(missing) - len(allowed_missing) - len(hard_missing),
        len(allowed_missing), n_expanded, len(unexpected),
    )
    return {
        "missing": missing,
        "allowed_missing": allowed_missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
        "expanded_aliases": n_expanded,
    }


def load_model_weights(model, cfg, resume: bool = True):
    """resume_or_load-compatible loader.

    * resume + a local last_checkpoint exists  -> d2 checkpointer resume
    * cfg.MODEL.WEIGHTS is .safetensors        -> validated safetensors load
    * cfg.MODEL.WEIGHTS is .pth / empty        -> d2 checkpointer behavior
    """
    from detectron2.checkpoint import DetectionCheckpointer

    save_dir = cfg.OUTPUT_DIR
    checkpointer = DetectionCheckpointer(model, save_dir=save_dir)
    weights = cfg.MODEL.WEIGHTS
    local_last = os.path.join(save_dir, "last_checkpoint")
    if resume and os.path.isfile(local_last):
        checkpointer.resume_or_load(weights, resume=True)
        return None
    if not weights:
        logger.info("MODEL.WEIGHTS is empty; starting from random initialization")
        return None
    if weights.endswith(".safetensors"):
        return load_initial_weights(model, weights)
    # External .pth: validated model-weights-only load (same reasoning as
    # Trainer.resume_or_load: the stored optimizer state belongs to the old
    # training run and has an incompatible parameter-group layout).
    return load_initial_weights(model, weights)

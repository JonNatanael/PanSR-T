# ------------------------------------------------------------------------
# TG-PanSR: temporal predictor.
#
# A single-clip predictor built on the original PanSR inference protocol
# (deterministic resize-only transform).  Accepts either an explicit
# ordered list of context+target images (with the target index) or a
# manifest record.  Returns the normal PanSR result dict for the target
# frame; diagnostics are available under the optional "temporal_diag"
# key.
# ------------------------------------------------------------------------
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = ["TemporalPredictor"]


class TemporalPredictor:
    def __init__(self, cfg, model: Optional[torch.nn.Module] = None):
        """
        Args:
            cfg: frozen config (temporal keys merged; EVAL-style dataset).
            model: an already-built model; built from cfg when None.
        """
        from detectron2.engine import build_model

        self.cfg = cfg
        self.model = model if model is not None else build_model(cfg)
        self.model.eval()
        self.metadata = None
        if len(cfg.DATASETS.TEST):
            from detectron2.data import MetadataCatalog

            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        from .mapper import TemporalCOCOPanopticDatasetMapper

        self.mapper = TemporalCOCOPanopticDatasetMapper(cfg, False)

    # ------------------------------------------------------------------
    def __call__(
        self,
        target_image: Union[str, np.ndarray],
        context_images: Optional[Sequence[Union[str, np.ndarray]]] = None,
        target_index: Optional[int] = None,
        frame_ids: Optional[List[str]] = None,
        image_id: Any = 0,
        original_hw: Optional[tuple] = None,
        diagnostics: bool = False,
    ) -> Dict[str, Any]:
        """Run inference on one clip.

        Args:
            target_image: path or uint8 array (cfg.INPUT.FORMAT order).
            context_images: paths/arrays of the preceding frames in
                chronological order.  May be empty (target-only call).
            target_index: index of the target inside the full ordered
                clip (context + target).  Defaults to last.
            frame_ids: optional identifiers for the frames.
            original_hw: original (h, w) of the target for output
                restoration; read from the target when omitted.
            diagnostics: when True include "temporal_diag" (frame ids,
                geometry size, fallback flag) in the result.
        """
        cfg = self.cfg
        from detectron2.data import detection_utils as utils

        if isinstance(target_image, (str, os.PathLike)):
            target = utils.read_image(str(target_image), format=cfg.INPUT.FORMAT)
        else:
            target = np.asarray(target_image)

        if original_hw is None:
            original_hw = (target.shape[0], target.shape[1])

        ctx = []
        if context_images:
            for c in context_images:
                if isinstance(c, (str, os.PathLike)):
                    ctx.append(utils.read_image(str(c), format=cfg.INPUT.FORMAT))
                else:
                    ctx.append(np.asarray(c))

        n = len(ctx)
        if target_index is None:
            target_index = n
        assert 0 <= target_index <= n, "target_index out of range"

        ordered = list(ctx[:target_index]) + [target] + list(ctx[target_index:])
        if frame_ids is None:
            frame_ids = [f"ctx_{i}" for i in range(len(ordered))]

        dataset_dict = {
            "target_file": None,
            "clip_frames": None,  # bypass file loading (arrays provided)
            "clip_frame_ids": frame_ids,
            "target_index": target_index,
            "image_id": image_id,
            "height": int(original_hw[0]),
            "width": int(original_hw[1]),
            "split": "predict",
            "sequence_id": "predict",
            "clip_manifest_id": "predict",
            "clip_sampling": cfg.TEMPORAL.SAMPLING,
            "clip_num_frames": cfg.TEMPORAL.NUM_FRAMES,
            "clip_allow_future": False,
            "clip_seed": cfg.TEMPORAL.SAMPLING_SEED,
            "_raw_clip_arrays": ordered,
            "_raw_target_array": target,
        }
        batched = self.mapper(dataset_dict)
        with torch.no_grad():
            out = self.model([batched])
        result = out[0]
        if diagnostics:
            result["temporal_diag"] = {
                "frame_ids": batched["frame_ids"],
                "target_index": batched["target_index"],
                "geo_hw": batched["branch_transforms"]["geo_hw"],
                "fallback": getattr(self.model, "fallback_count", 0),
            }
        return result

    # ------------------------------------------------------------------
    def from_manifest_record(
        self,
        record: Dict[str, Any],
        lars_root: str,
        diagnostics: bool = False,
    ) -> Dict[str, Any]:
        """Run inference for one manifest record (target + its context)."""
        from .manifest import select_frames, validate_record

        rec = validate_record(record, root=os.path.abspath(lars_root), require_files=True)
        chosen, _ = select_frames(
            rec["frames"], rec["target_index"],
            sampling=self.cfg.TEMPORAL.SAMPLING,
            num_frames=self.cfg.TEMPORAL.NUM_FRAMES,
            allow_future=self.cfg.TEMPORAL.ALLOW_FUTURE,
        )
        tidx = rec["target_index"]
        paths = [os.path.join(lars_root, f["file"]) for f in rec["frames"]]
        target_path = paths[tidx]
        ctx = [paths[i] for i in chosen if i != tidx]
        return self(
            target_image=target_path,
            context_images=ctx,
            target_index=tidx,
            frame_ids=[f["frame_id"] for f in rec["frames"]],
            image_id=rec["image_id"],
            original_hw=(rec["original_height"], rec["original_width"]),
            diagnostics=diagnostics,
        )

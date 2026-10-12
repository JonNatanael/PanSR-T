# ------------------------------------------------------------------------
# TG-PanSR: temporal dataset mapper.
#
# The target frame is processed *exactly* like the original PanSR mapper
# (same transform sampling, same instance construction), but the sampled
# transform list is also applied to every context frame so the whole clip
# is spatially consistent.  The clip is then resized to the geometry
# raster (aspect-preserving, dims divisible by the Pi3 patch size, within
# the pixel budget) and handed to the model as a float RGB [0, 1] tensor.
#
# No Pi3 / CUDA work happens here: workers only decode and transform.
# ------------------------------------------------------------------------
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Boxes, Instances

from . import alignment
from .manifest import ManifestError, select_frames

logger = logging.getLogger(__name__)

__all__ = ["TemporalCOCOPanopticDatasetMapper", "transforms_to_affine",
           "build_temporal_transform_gen"]


def build_temporal_transform_gen(cfg, is_train: bool) -> List[T.Augmentation]:
    """Train: identical to the original PanSR mapper (flip + resize +
    fixed crop).  Eval: deterministic resize-only matching the PanSR
    inference protocol (ResizeShortestEdge, see predict.py / DefaultPredictor).
    """
    if is_train:
        from pansr.data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import (
            build_transform_gen,
        )
        return build_transform_gen(cfg, True)
    short = int(cfg.INPUT.MIN_SIZE_TEST)
    return [T.ResizeShortestEdge([short, short], int(cfg.INPUT.MAX_SIZE_TEST))]


def transforms_to_affine(transforms, original_hw: Tuple[int, int]) -> Tuple[alignment.PixelAffine, Tuple[int, int]]:
    """Compose a detectron2 TransformList into a PixelAffine (original ->
    final raster, pixel-center convention) and the final raster size.

    Supports the transform types used by the PanSR LSJ mapper
    (ResizeTransform, CropTransform, PadTransform, H/V flips, NoOp).
    """
    from fvcore.transforms.transform import (
        CropTransform,
        HFlipTransform,
        NoOpTransform,
        PadTransform,
        VFlipTransform,
    )
    from detectron2.data.transforms import ResizeTransform

    affine = alignment.PixelAffine.identity()
    size = tuple(original_hw)  # (h, w)
    for tfm in transforms:
        if isinstance(tfm, NoOpTransform):
            continue
        # NOTE on ordering: PixelAffine.compose applies the receiver FIRST
        # and the argument second (receiver.compose(other) == other @
        # receiver).  The transforms arrive in application order, so the
        # accumulated affine must stay the receiver: affine = affine @ new.
        if isinstance(tfm, ResizeTransform):
            affine = affine.compose(
                alignment.PixelAffine.resize(size, (tfm.new_h, tfm.new_w))
            )
            size = (tfm.new_h, tfm.new_w)
        elif isinstance(tfm, CropTransform):
            affine = affine.compose(alignment.PixelAffine.crop((tfm.x0, tfm.y0)))
            # numpy slice semantics: img[y0:y0+h, x0:x0+w] clips the
            # requested window to the actual image bounds, so the real
            # output can be smaller than (h, w) when the previous step
            # (e.g. a downscale) produced a sub-crop-size raster.
            size = (
                max(0, min(tfm.h, size[0] - tfm.y0)),
                max(0, min(tfm.w, size[1] - tfm.x0)),
            )
        elif isinstance(tfm, PadTransform):
            affine = affine.compose(alignment.PixelAffine.pad((tfm.x0, tfm.y0)))
            size = (size[0] + tfm.y0 + tfm.y1, size[1] + tfm.x0 + tfm.x1)
        elif isinstance(tfm, HFlipTransform):
            if getattr(tfm, "do", True):
                affine = affine.compose(alignment.PixelAffine.hflip(size[1]))
        elif isinstance(tfm, VFlipTransform):
            if getattr(tfm, "do", True):
                affine = affine.compose(alignment.PixelAffine.vflip(size[0]))
        else:
            raise ManifestError(f"unsupported transform for affine tracking: {type(tfm).__name__}")
    return affine, size


class TemporalCOCOPanopticDatasetMapper:
    """Temporal counterpart of COCOPanopticNewBaselineDatasetMapper.

    The target branch (image tensor + instances) is produced with the same
    code path as the original mapper (kept in sync line by line; see
    tests/temporal/test_mapper_parity.py).
    """

    @configurable
    def __init__(self, is_train: bool, *, tfm_gens, image_format: str,
                 pixel_budget: int, patch_size: int = 14,
                 preprocess_mode: str = "shared_view_online",
                 missing_policy: str = "error",
                 frame_dropout: float = 0.0,
                 sampling: str = "all_history",
                 num_frames: int = 10,
                 seed: int = 0):
        self.tfm_gens = tfm_gens
        self.img_format = image_format
        self.is_train = is_train
        self.pixel_budget = int(pixel_budget)
        self.patch_size = int(patch_size)
        self.preprocess_mode = preprocess_mode
        self.missing_policy = missing_policy
        self.frame_dropout = float(frame_dropout)
        self.sampling = sampling
        self.num_frames = int(num_frames)
        self.seed = int(seed)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        t = cfg.TEMPORAL
        if cfg.INPUT.COPY_PASTE.ENABLED:
            raise ManifestError(
                "Copy-paste augmentation is disabled for the temporal model and its "
                "matched controls (see brief section 5.1); set INPUT.COPY_PASTE.ENABLED False"
            )
        return {
            "is_train": is_train,
            "tfm_gens": build_temporal_transform_gen(cfg, is_train),
            "image_format": cfg.INPUT.FORMAT,
            "pixel_budget": t.GEOMETRY.PIXEL_BUDGET,
            "patch_size": 14,
            "preprocess_mode": t.PREPROCESS_MODE,
            "missing_policy": t.MISSING_POLICY,
            "frame_dropout": t.FRAME_DROPOUT if is_train else 0.0,
            "sampling": t.SAMPLING,
            "num_frames": t.NUM_FRAMES,
            "seed": t.SAMPLING_SEED,
        }

    # ------------------------------------------------------------------
    def __call__(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        import copy

        dataset_dict = copy.deepcopy(dataset_dict)

        # ---- target: exactly the original PanSR target path ------------
        raw_target = dataset_dict.get("_raw_target_array")
        if raw_target is not None:
            image = np.asarray(raw_target)
            dataset_dict["target_file"] = dataset_dict.get("target_file") or "in-memory"
            # in-memory arrays define the size (e.g. probe images)
            dataset_dict["height"] = image.shape[0]
            dataset_dict["width"] = image.shape[1]
        else:
            image = utils.read_image(dataset_dict["target_file"], format=self.img_format)
            utils.check_image_size(dataset_dict, image)
        orig_hw = (image.shape[0], image.shape[1])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]
        crop_affine, crop_hw = transforms_to_affine(transforms, orig_hw)
        # defense in depth: the tracked raster size must equal the real one
        if tuple(crop_hw) != tuple(image_shape):
            raise ManifestError(
                f"affine size tracking mismatch: tracked {tuple(crop_hw)} vs "
                f"actual {tuple(image_shape)} for transforms "
                f"{[type(t).__name__ for t in transforms]}"
            )

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "pan_seg_file_name" in dataset_dict and self.is_train:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
            dataset_dict["instances"] = self._build_instances(
                pan_seg_gt, segments_info, image_shape
            )
        else:
            dataset_dict.pop("annotations", None)

        # ---- context frames: the *same* sampled transform ---------------
        clip_paths: Optional[List[str]] = dataset_dict.get("clip_frames")
        clip_frame_ids: List[str] = dataset_dict["clip_frame_ids"]
        target_index: int = int(dataset_dict["target_index"])
        if not (0 <= target_index < len(clip_frame_ids)):
            raise ManifestError("target_index out of range for clip frames")

        # ---- sampling policy --------------------------------------------
        frame_records = []
        # relative ranks are known from the manifest; recompute from frame
        # indices (frame_id "video_NNNNN" gives the absolute frame number)
        from .manifest import parse_frame_name

        ranks = []
        for fid in clip_frame_ids:
            try:
                _, fr = parse_frame_name(fid)
                ranks.append(fr)
            except ManifestError:
                ranks.append(None)
        if any(r is None for r in ranks):
            raise ManifestError(f"unparseable frame id in clip: {clip_frame_ids}")
        target_rank = ranks[target_index]
        for i, r in enumerate(ranks):
            frame_records.append({
                "relative_rank": r - target_rank,
                "is_target": i == target_index,
            })
        chosen, _ = select_frames(
            frame_records, target_index,
            sampling=self.sampling, num_frames=self.num_frames,
        )

        # context dropout (train only): physically remove selected context
        # frames *before* the geometric call; the target is always kept.
        # (No temporal padding mask exists in the frozen encoder, so
        # dropped frames are simply not passed to Pi3.)
        if self.is_train and self.frame_dropout > 0.0 and self.sampling != "repeat_target":
            # per-sample deterministic seed (reproducible, but different
            # samples drop different frames)
            uid = f"{dataset_dict.get('image_id')}|{dataset_dict.get('sequence_id')}"
            import hashlib as _hashlib
            sample_seed = (self.seed + int.from_bytes(
                _hashlib.sha256(uid.encode("utf-8")).digest()[:8], "big"
            )) % (2**32 - 1)
            rng = np.random.RandomState(sample_seed)
            keep = [i for i in chosen if i == target_index or rng.rand() >= self.frame_dropout]
            if target_index not in keep:
                keep.append(target_index)
            chosen = sorted([i for i in keep if i != target_index],
                            key=lambda i: ranks[i]) + [target_index]

        context_images: List[np.ndarray] = []
        missing = []
        raw_clip = dataset_dict.get("_raw_clip_arrays")
        for i in chosen:
            if i == target_index:
                context_images.append(image)  # already transformed
                continue
            if raw_clip is not None:
                ctx_img = np.asarray(raw_clip[i])
            else:
                if clip_paths is None:
                    raise ManifestError("clip_frames paths missing for file loading")
                path = clip_paths[i]
                if not os.path.isfile(path):
                    if self.missing_policy == "error":
                        raise ManifestError(f"context frame missing: {path}")
                    missing.append(path)
                    continue
                ctx_img = utils.read_image(path, format=self.img_format)
            if tuple(ctx_img.shape[:2]) != orig_hw:
                raise ManifestError(
                    f"context frame {clip_frame_ids[i]} has size {ctx_img.shape[:2]} but "
                    f"target has size {orig_hw}; clips must be resolution-consistent"
                )
            context_images.append(transforms.apply_image(ctx_img))

        # chronological ordering of the selected clip (target keeps its
        # verified position; repeated targets collapse to one position)
        ordered_idx: List[int] = []
        if self.sampling == "repeat_target":
            ordered_idx = [target_index] * len(chosen)
        else:
            ordered_idx = chosen  # already chronological (target last for causal clips)

        # ---- geometry raster --------------------------------------------
        geo_hw = alignment.resize_geometry_size(crop_hw, self.pixel_budget, self.patch_size)
        geo_scale = (geo_hw[1] / crop_hw[1], geo_hw[0] / crop_hw[0])  # (sx, sy)
        # original -> crop raster (crop_affine) first, then crop raster ->
        # geo raster (resize); compose applies the receiver first.
        geo_affine = crop_affine.compose(
            alignment.PixelAffine.resize(crop_hw, geo_hw)
        )

        # context_images is indexed by load order, which equals `chosen`
        # order; map each ordered position back to its loaded array.
        load_pos = {i: k for k, i in enumerate(chosen)}
        clip_rgb = np.empty((len(ordered_idx), 3, geo_hw[0], geo_hw[1]), dtype=np.float32)
        frame_ids_out: List[str] = []
        for k, i in enumerate(ordered_idx):
            rgb = self._to_rgb(context_images[load_pos[i]])
            small = cv2.resize(rgb, (geo_hw[1], geo_hw[0]), interpolation=cv2.INTER_LINEAR)
            clip_rgb[k] = small.astype(np.float32).transpose(2, 0, 1) / 255.0
            frame_ids_out.append(clip_frame_ids[i])

        # ---- record everything the model needs ---------------------------
        out = {
            "image": dataset_dict["image"],
            # the official COCOPanopticEvaluator names its outputs from
            # "file_name"; the temporal records keep it as "target_file"
            "file_name": dataset_dict["target_file"],
            "height": dataset_dict["height"],   # original target output size
            "width": dataset_dict["width"],
            "image_id": dataset_dict["image_id"],
            "clip_manifest_id": dataset_dict.get("clip_manifest_id"),
            "sequence_id": dataset_dict.get("sequence_id"),
            "split": dataset_dict.get("split"),
            "clip_rgb": torch.from_numpy(np.ascontiguousarray(clip_rgb)),
            "target_index": int(ordered_idx.index(target_index)),
            "frame_ids": frame_ids_out,
            "clip_full_frame_ids": clip_frame_ids,
            "selected_frame_ids": frame_ids_out,
            "branch_transforms": {
                "original_hw": list(orig_hw),
                "crop_hw": list(crop_hw),
                "geo_hw": list(geo_hw),
                "geo_resize_scales": [geo_scale[0], geo_scale[1]],
                "crop_affine": self._affine_params(crop_affine),
                "geo_affine": self._affine_params(geo_affine),
                "preprocess_mode": self.preprocess_mode,
                "transform_summary": " -> ".join(type(t).__name__ for t in transforms),
            },
        }
        if self.is_train and "instances" in dataset_dict:
            out["instances"] = dataset_dict["instances"]
        if missing:
            out["missing_context_frames"] = missing
        return out

    # ------------------------------------------------------------------
    def _to_rgb(self, arr: np.ndarray) -> np.ndarray:
        """Convert to RGB (Pi3 input convention) exactly once."""
        if self.img_format == "BGR":
            return arr[:, :, ::-1].copy()
        return arr

    @staticmethod
    def _affine_params(a: alignment.PixelAffine) -> Dict[str, float]:
        return {"sx": a.sx, "sy": a.sy, "ox": a.ox, "oy": a.oy}

    # ------------------------------------------------------------------
    def _build_instances(self, pan_seg_gt: np.ndarray, segments_info, image_shape):
        """Byte-for-byte the original mapper's instance construction."""
        from panopticapi.utils import rgb2id

        pan_seg_gt = rgb2id(pan_seg_gt)
        instances = Instances(image_shape)
        classes = []
        masks = []
        isthing = []
        cls_weight = []
        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                mask = pan_seg_gt == segment_info["id"]
                if mask.sum() < 1:
                    continue
                classes.append(class_id)
                masks.append(mask)
                isthing.append(segment_info["isthing"] == 1)
                w = segment_info["cls_weight"] if "cls_weight" in segment_info else 1.0
                cls_weight.append(w)

        classes = np.array(classes)
        isthing = np.array(isthing)
        cls_weight = np.array(cls_weight)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        instances.gt_isthing = torch.tensor(isthing, dtype=torch.bool)
        instances.gt_cls_weight = torch.tensor(cls_weight, dtype=torch.float16)
        if len(masks) == 0:
            instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            instances.gt_boxes = Boxes(torch.zeros((0, 4)))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
            instances.gt_boxes = masks.get_bounding_boxes()
        return instances

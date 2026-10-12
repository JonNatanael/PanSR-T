# ------------------------------------------------------------------------
# TG-PanSR: TemporalPanSR meta-architecture.
#
# Subclass of the original PanSR that adds the temporal-geometric branch
# through the single overridable feature-extraction hook
# (:meth:`PanSR.extract_backbone_features`).  Everything else (target
# preparation, OCP, pixel decoder, criterion, postprocessing) is the
# original code path, unmodified.
#
# Parameter prefixes:
#   backbone.*, sem_seg_head.*      - original PanSR (checkpoint compatible)
#   geometry_encoder.*              - frozen Pi3 (never trained)
#   temporal_fusion.*               - new trainable modules
# ------------------------------------------------------------------------
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList

from ..pansr_model import PanSR
from . import alignment
from .fusion import TemporalFusionModule
from .geometry import GeometryEncoderError, build_geometry_encoder

logger = logging.getLogger(__name__)

__all__ = ["TemporalPanSR"]


@META_ARCH_REGISTRY.register()
class TemporalPanSR(PanSR):
    """PanSR + frozen Pi3 geometric features + residual pyramid fusion."""

    @configurable
    def __init__(
        self,
        *,
        geometry_encoder=None,
        temporal_fusion: Optional[TemporalFusionModule] = None,
        temporal_settings: Optional[Dict[str, Any]] = None,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.geometry_encoder = geometry_encoder  # None when disabled/zero-geo
        self.temporal_fusion = temporal_fusion
        t = (temporal_settings or {})
        self.temporal_enabled = bool(t.get("enabled"))
        self.zero_geometry = bool(t.get("zero_geometry"))
        self.missing_policy = t.get("missing_policy", "error")
        self.fallback_count = 0

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg):
        base_kwargs = PanSR.from_config(cfg)
        t = cfg.TEMPORAL
        f = t.FUSION

        # Backbone feature shapes (names/channels/strides) for adapters.
        backbone = base_kwargs["backbone"]
        shapes = backbone.output_shape()

        zero_geometry = bool(f.ZERO_GEOMETRY)
        geometry_encoder = None
        if t.ENABLED and not zero_geometry:
            geometry_encoder = build_geometry_encoder(cfg)
        geo_channels = int(t.GEOMETRY.FEATURE_WIDTH)
        if t.ENABLED and geometry_encoder is None and not zero_geometry:
            raise GeometryEncoderError(
                "TEMPORAL.ENABLED=True with no geometry encoder and no "
                "ZERO_GEOMETRY control: misconfiguration"
            )

        temporal_fusion = None
        if t.ENABLED:
            temporal_fusion = TemporalFusionModule(cfg, shapes, geo_channels)

        temporal_settings = {
            "enabled": t.ENABLED,
            "zero_geometry": zero_geometry,
            "missing_policy": t.MISSING_POLICY,
            "pixel_budget": int(t.GEOMETRY.PIXEL_BUDGET),
            "patch_size": 14,
            "preprocess_mode": t.PREPROCESS_MODE,
        }
        return {
            **base_kwargs,
            "geometry_encoder": geometry_encoder,
            "temporal_fusion": temporal_fusion,
            "temporal_settings": temporal_settings,
        }

    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        # the frozen geometry encoder must stay in eval
        if self.geometry_encoder is not None:
            self.geometry_encoder.eval()
            for p in self.geometry_encoder.parameters():
                p.requires_grad = False
        return self

    # ------------------------------------------------------------------
    def _level_stride(self, name: str) -> int:
        # derived from the backbone output spec, never hard-coded
        return int(self.backbone.output_shape()[name].stride)

    def extract_backbone_features(self, images: ImageList, batched_inputs: List[dict]):
        features = self.backbone(images.tensor)
        if not self.temporal_enabled or self.temporal_fusion is None:
            return features
        return self._apply_temporal_branch(features, images, batched_inputs)

    # ------------------------------------------------------------------
    def _swin_grids_for_sample(
        self, b: int, features: Dict[str, torch.Tensor], image_size: Tuple[int, int],
        affine: alignment.PixelAffine, names: List[str],
    ) -> Dict[str, alignment.FeatureGrid]:
        h, w = image_size
        grids = {}
        for name in names:
            s = self._level_stride(name)
            feat = features[name][b]
            # stride-s conv over the real (unpadded) raster: floor(r/s) cells
            fh, fw = h // s, w // s
            if not (1 <= fh <= feat.shape[-2] and 1 <= fw <= feat.shape[-1]):
                raise RuntimeError(
                    f"feature {name}: valid region {fh}x{fw} exceeds batch feature "
                    f"size {tuple(feat.shape[-2:])} for image {h}x{w}"
                )
            grids[name] = alignment.FeatureGrid(fh, fw, s, (h, w), affine)
        return grids

    def _apply_temporal_branch(self, features, images, batched_inputs):
        image_sizes = images.image_sizes  # real (pre-pad) sizes
        names = list(self.temporal_fusion.inject_features)
        if self.temporal_fusion.appearance_source not in names:
            names.append(self.temporal_fusion.appearance_source)
        f = self.temporal_fusion

        fused_levels: Dict[int, Dict[str, torch.Tensor]] = {}
        for b in range(len(image_sizes)):
            bi = batched_inputs[b]
            if "clip_rgb" not in bi or bi["clip_rgb"] is None:
                self._fallback(bi, reason="no clip in batch input")
                continue
            try:
                per, grid_info = self._per_sample_fusion(b, features, image_sizes[b], bi)
            except (GeometryEncoderError, ValueError, RuntimeError) as e:
                if self.missing_policy == "fallback_baseline":
                    logger.warning("temporal fallback for sample %s: %s", bi.get("image_id"), e)
                    self._fallback(bi, reason=str(e))
                    continue
                raise
            fused_levels[b] = per

        if not fused_levels:
            return features

        # write the fused levels back into batched tensors (valid region only)
        out = {k: v.clone() if k in names else v for k, v in features.items()}
        for b, per in fused_levels.items():
            h, w = image_sizes[b]
            for name in names:
                s = self._level_stride(name)
                fh, fw = h // s, w // s
                out[name][b, :, :fh, :fw] = per[name]
        return out

    def _per_sample_fusion(self, b, features, image_size, bi):
        h, w = image_size
        # --- Swin grids --------------------------------------------------
        bt = bi["branch_transforms"]
        ca = bt["crop_affine"]
        crop_affine = alignment.PixelAffine(
            sx=ca["sx"], sy=ca["sy"], ox=ca["ox"], oy=ca["oy"]
        )
        names = list(self.temporal_fusion.inject_features)
        if self.temporal_fusion.appearance_source not in names:
            names.append(self.temporal_fusion.appearance_source)
        grids = self._swin_grids_for_sample(b, features, (h, w), crop_affine, names)

        per = {}
        for name in names:
            s = self._level_stride(name)
            fh, fw = h // s, w // s
            per[name] = features[name][b, :, :fh, :fw].contiguous()

        # --- geometry -----------------------------------------------------
        geo_features, geo_grid = self._geometry_for_sample(bi, (h, w), crop_affine)
        if self.zero_geometry:
            dev = next(iter(per.values())).device
            geo_features = torch.zeros(self.temporal_fusion.geo_channels,
                                       geo_grid.height, geo_grid.width, device=dev)
        fused = self.temporal_fusion(per, geo_features, geo_grid, grids)
        return fused, (geo_features.shape[-2:])

    def _geometry_for_sample(self, bi, crop_hw, crop_affine):
        """Return ([C_geo, hg, wg] features, FeatureGrid) or (None, None)
        when the geometry branch is unavailable for this sample."""
        bt = bi["branch_transforms"]
        gh, gw = bt["geo_hw"]
        ga = bt["geo_affine"]
        geo_affine = alignment.PixelAffine(sx=ga["sx"], sy=ga["sy"], ox=ga["ox"], oy=ga["oy"])
        # Pi3's patch embed is a stride-14 conv without padding: the
        # feature grid is floor(geo_dim / 14) (the same convention
        # enforced by alignment.FeatureGrid; geo rasters from
        # resize_geometry_size are always patch-multiples).
        ph = gh // 14
        pw = gw // 14
        geo_grid = alignment.FeatureGrid(
            ph, pw, 14, (gh, gw), geo_affine,
        )

        if self.zero_geometry:
            # capacity control B2: features materialized on the target-
            # derived lattice in _per_sample_fusion (with the correct device).
            return None, geo_grid

        # the train/val loaders deliver CPU batches; move the clip to the
        # model device exactly like the image/instance branches do
        # (the frozen encoder may have been placed on any device)
        clip = bi["clip_rgb"].to(self.device)
        target_index = int(bi["target_index"])
        frame_ids = list(bi["frame_ids"])
        assert clip.shape[-2:] == (gh, gw), (
            f"clip_rgb raster {tuple(clip.shape[-2:])} != branch geo raster {(gh, gw)}"
        )
        assert clip.shape[0] == len(frame_ids), "clip/frame_ids length mismatch"
        packet = self.geometry_encoder.encode(
            clip,
            target_index,
            frame_ids=frame_ids,
            transforms={"original_to_geometry": geo_affine},
        )
        return packet.target_features, geo_grid

    def _fallback(self, bi, reason: str):
        self.fallback_count += 1
        logger.info(
            "temporal: %d sample(s) on the vanilla baseline path (reason: %s)",
            self.fallback_count, reason,
        )

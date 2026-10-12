# ------------------------------------------------------------------------
# TG-PanSR: frozen geometric feature adapter around the GMOS Pi3 encoder.
#
# ``Pi3GeometryEncoder`` wraps the GMOS-compatible ``Pi3Encoder``
# (gmos/core/pi3_encoder.py) and exposes the explicit per-sample packet
# interface required by the temporal model:
#
#   G_all = frozen_pi3(joint_clip)            [1, T, h, w, 1024]
#   G_t   = G_all[0, target_index]            [h, w, 1024]
#
# Design points (see docs/temporal_pansr.md):
#   * The encoder is frozen (requires_grad_(False)), kept in eval() even
#     when the parent model trains, and always run under torch.no_grad().
#   * Weight loading uses a reviewed allowlist for the reconstruction-head
#     keys that exist in the full Pi3 checkpoint but not in the
#     encoder-only wrapper; missing encoder/decoder keys are fatal.
#   * No SAM2, no GMOS proposer, no inference scripts are imported.
#   * The GMOS checkout is imported lazily and only when a geometry
#     backend is actually instantiated.
# ------------------------------------------------------------------------
from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from . import alignment

logger = logging.getLogger(__name__)

__all__ = ["GeometryPacket", "GeometryEncoderError", "Pi3GeometryEncoder", "build_geometry_encoder"]


class GeometryEncoderError(RuntimeError):
    """Raised for invalid geometry inputs or incompatible checkpoints."""


# Full Pi3 checkpoints contain reconstruction heads that the GMOS
# encoder-only wrapper does not define.  These prefixes are the reviewed
# list of allowed extra keys (see docs/repository_audit.md).
_PI3_EXTRA_KEY_PREFIXES = (
    "point_decoder.",
    "point_head.",
    "conf_decoder.",
    "conf_head.",
    "camera_decoder.",
    "camera_head.",
)


def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _import_gmos_pi3_encoder(gmos_root: str):
    """Import ``core.pi3_encoder.Pi3Encoder`` from the GMOS checkout.

    This is the single, documented, controlled entry point that touches
    ``sys.path``: the GMOS root is appended, and the imported module is
    verified to physically live inside that root so an unrelated top-level
    package named ``core`` can never shadow it.
    """
    gmos_root = os.path.abspath(os.path.expanduser(gmos_root))
    encoder_path = os.path.join(gmos_root, "core", "pi3_encoder.py")
    if not os.path.isfile(encoder_path):
        raise GeometryEncoderError(
            f"GMOS Pi3 encoder not found at {encoder_path}; set "
            "TEMPORAL.GEOMETRY.GMOS_ROOT or the GMOS_ROOT environment variable"
        )
    if gmos_root not in sys.path:
        sys.path.insert(0, gmos_root)
    import importlib

    core = importlib.import_module("core")
    core_file = os.path.abspath(core.__file__)
    if not core_file.startswith(gmos_root + os.sep):
        raise GeometryEncoderError(
            f"top-level package 'core' resolves to {core_file}, which is not "
            f"inside the GMOS root {gmos_root}; refusing to import a shadowed module"
        )
    module = importlib.import_module("core.pi3_encoder")
    Pi3Encoder = getattr(module, "Pi3Encoder", None)
    if Pi3Encoder is None:
        raise GeometryEncoderError(f"core.pi3_encoder in {gmos_root} has no Pi3Encoder class")
    return Pi3Encoder


@dataclass
class GeometryPacket:
    """Per-sample output of the frozen geometric encoder.

    Attributes:
        target_features: [C_geo, h_geo, w_geo] detached float32 tensor.
        target_valid: [h_geo, w_geo] bool; all True in the reference
            shared-view path (carried for generality and padding).
        target_index: explicit index of the target frame in the clip.
        frame_ids: ordered frame identifiers of the clip actually encoded.
        geometry_hw: (h_geo, w_geo).
        original_to_geometry: PixelAffine mapping original image
            pixel-center coordinates to the geometry raster.
        metadata: free-form diagnostics (sizes, scales, hashes, ...).
        all_frame_features: [T, C_geo, h, w] if requested, else None.
    """

    target_features: torch.Tensor
    target_valid: torch.Tensor
    target_index: int
    frame_ids: Tuple[str, ...]
    geometry_hw: Tuple[int, int]
    original_to_geometry: alignment.PixelAffine
    metadata: Dict[str, Any] = field(default_factory=dict)
    all_frame_features: Optional[torch.Tensor] = None

    @property
    def feature_width(self) -> int:
        return int(self.target_features.shape[0])

    def check(self, expected_width: Optional[int] = None) -> None:
        if self.target_features.ndim != 3:
            raise GeometryEncoderError(
                f"target_features must be [C,h,w], got {tuple(self.target_features.shape)}"
            )
        if self.target_features.shape[-2:] != self.geometry_hw:
            raise GeometryEncoderError("feature grid inconsistent with geometry_hw")
        if not self.target_features.is_floating_point():
            raise GeometryEncoderError("target_features must be floating point")
        if expected_width is not None and self.feature_width != expected_width:
            raise GeometryEncoderError(
                f"feature width {self.feature_width} != expected {expected_width}"
            )


class Pi3GeometryEncoder(nn.Module):
    """Frozen Pi3 joint clip encoder with explicit target selection."""

    PATCH_SIZE = 14

    def __init__(
        self,
        checkpoint: str,
        gmos_root: str,
        feature_mode: str = "last_decoder",
        tapped_layers: Tuple[int, ...] = (7, 15, 23, 31),
        expected_feature_width: int = 1024,
        dtype: str = "fp32",
        device: Optional[torch.device] = None,
        keep_all_frames: bool = False,
    ):
        super().__init__()
        self.checkpoint = os.path.abspath(checkpoint)
        if not os.path.isfile(self.checkpoint):
            raise GeometryEncoderError(f"Pi3 checkpoint not found: {self.checkpoint}")
        self.checkpoint_sha256 = _sha256_file(self.checkpoint)
        self.gmos_root = os.path.abspath(gmos_root)
        self.feature_mode = feature_mode
        self.expected_feature_width = int(expected_feature_width)
        self.keep_all_frames = keep_all_frames
        self.device = (
            torch.device(device) if device is not None else None
        )

        if feature_mode == "last_decoder":
            self.tapped_layers: Tuple[int, ...] = ()
        elif feature_mode.startswith("tapped:"):
            self.tapped_layers = tuple(int(x) for x in feature_mode.split(":", 1)[1].split(","))
            for l in self.tapped_layers:
                if l % 2 == 0 or l >= 36:
                    raise GeometryEncoderError(
                        f"tapped layers must be odd indices < 36 (cross-frame blocks), got {l}"
                    )
        else:
            raise GeometryEncoderError(f"unknown feature mode {feature_mode!r}")

        Pi3Encoder = _import_gmos_pi3_encoder(self.gmos_root)
        self.encoder = Pi3Encoder()
        self._load_checkpoint(self.encoder, self.checkpoint)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # bf16 needs an SDPA backend that is not available on Turing (sm75);
        # detect at init and fall back to fp32 with a clear log.
        self.requested_dtype = dtype
        self.compute_dtype = torch.float32
        if dtype == "bf16":
            if device is not None and device.type == "cuda" and _bf16_sdpa_supported(device):
                self.compute_dtype = torch.bfloat16
            else:
                logger.warning(
                    "Pi3GeometryEncoder: bf16 SDPA backend not supported on this device; "
                    "running the frozen encoder in fp32 instead."
                )
        if device is not None:
            self.encoder.to(device)

    # ------------------------------------------------------------------
    def _load_checkpoint(self, model: nn.Module, path: str) -> None:
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(path, device="cpu")
        else:
            state = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
        model_state = model.state_dict()

        unexpected = [k for k in state if k not in model_state]
        missing = [k for k in model_state if k not in state]
        allowed_extra = [
            k for k in unexpected if k.startswith(_PI3_EXTRA_KEY_PREFIXES)
        ]
        disallowed_extra = [k for k in unexpected if k not in allowed_extra]
        if disallowed_extra:
            raise GeometryEncoderError(
                "Pi3 checkpoint contains keys not accepted by the encoder-only wrapper "
                f"and not on the reviewed allowlist: {disallowed_extra[:8]} "
                f"({len(disallowed_extra)} total)"
            )
        if missing:
            raise GeometryEncoderError(
                f"Pi3 checkpoint is missing {len(missing)} encoder/decoder keys, e.g. "
                f"{missing[:8]}; refusing to load a partially initialized encoder"
            )
        # strict=False only for the reviewed reconstruction-head keys
        incompatible = model.load_state_dict(
            {k: v for k, v in state.items() if k in model_state}, strict=False
        )
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        n_extra = len(allowed_extra)
        logger.info(
            "Loaded Pi3 checkpoint %s (sha256=%s); %d allowed extra reconstruction-head keys ignored.",
            path, self.checkpoint_sha256[:12], n_extra,
        )

    # ------------------------------------------------------------------
    def train(self, mode: bool = True):  # noqa: D401 - override on purpose
        """The frozen encoder must stay in eval() no matter what the parent
        model does.  (The *trainable* adapters live outside this module.)"""
        super().train(mode)
        self.encoder.eval()
        return self

    # ------------------------------------------------------------------
    def encode(
        self,
        clip_rgb: torch.Tensor,
        target_index: int,
        frame_ids: Optional[List[str]] = None,
        transforms: Optional[Dict[str, Any]] = None,
    ) -> GeometryPacket:
        """Run the joint geometric encoding of one clip and select the
        explicitly identified target frame.

        Args:
            clip_rgb: [T, 3, H, W] RGB; float in [0, 1] (uint8 0..255 is
                accepted and converted).  H, W must be positive multiples
                of the patch size (14).
            target_index: 0-based index of the annotated frame in the clip.
            frame_ids: one identifier per clip frame (for cache keys / logs).
            transforms: metadata describing how this clip raster relates to
                the original images (used for the packet's affine record).
        """
        if clip_rgb.ndim != 4 or clip_rgb.shape[1] != 3:
            raise GeometryEncoderError(f"clip_rgb must be [T,3,H,W], got {tuple(clip_rgb.shape)}")
        T, _, H, W = clip_rgb.shape
        if H % self.PATCH_SIZE or W % self.PATCH_SIZE:
            raise GeometryEncoderError(
                f"geometry raster {H}x{W} not divisible by patch size {self.PATCH_SIZE}"
            )
        if not (0 <= target_index < T):
            raise GeometryEncoderError(
                f"target_index {target_index} out of range for T={T}"
            )
        if frame_ids is None:
            frame_ids = [f"frame_{i}" for i in range(T)]
        if len(frame_ids) != T:
            raise GeometryEncoderError(
                f"frame_ids length {len(frame_ids)} != T={T}"
            )

        if clip_rgb.dtype == torch.uint8:
            clip = clip_rgb.to(torch.float32) / 255.0
        else:
            clip = clip_rgb.to(torch.float32)
            if clip.numel():
                if clip.min() < -1e-6 or clip.max() > 1.0 + 1e-6:
                    raise GeometryEncoderError(
                        "clip_rgb must be float in [0, 1] (or uint8); "
                        f"got range [{clip.min().item():.3f}, {clip.max().item():.3f}]"
                    )

        device = self.device if self.device is not None else clip.device
        clip = clip.to(device)

        h, w = H // self.PATCH_SIZE, W // self.PATCH_SIZE
        # The frozen encoder is *always* run in its verified precision
        # (fp32, or bf16 where the SDPA backend supports it), independent
        # of any surrounding AMP autocast context: autocast would otherwise
        # silently downcast the encoder's conv/linear ops to fp16, which is
        # numerically unsafe on Turing (sm75) and changes the frozen
        # feature distribution.  autocast(enabled=False) also prevents
        # fp32<->fp16 round-trips inside the encoder.
        autocast_ctx = (
            torch.autocast(device_type=device.type, enabled=False)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        if self.tapped_layers:
            with torch.no_grad(), autocast_ctx:
                feats = self._encode_tapped(clip.unsqueeze(0), h, w)  # [L, T, h, w, 1024]
            # combine with fixed equal weights unless learned combination
            # is attached by the model (see TemporalPanSR); the default
            # interface keeps the last-layer contract by returning the
            # per-layer stack in metadata and the last layer in the packet.
            target = feats[-1][target_index]
        else:
            with torch.no_grad(), autocast_ctx:
                out = self.encoder(clip.unsqueeze(0).to(self.compute_dtype))  # [1, T, h, w, C]
            out = out.float()
            target = out[0, target_index]
            all_feats = out[0] if self.keep_all_frames else None

        target = target.permute(2, 0, 1).contiguous().float()  # [C, h, w]
        if target.shape[0] != self.expected_feature_width:
            raise GeometryEncoderError(
                f"Pi3 produced {target.shape[0]} channels, expected "
                f"{self.expected_feature_width}; the extraction contract changed"
            )
        valid = torch.ones(h, w, dtype=torch.bool, device=target.device)

        transforms = transforms or {}
        affine = transforms.get("original_to_geometry")
        if affine is None:
            # Identity is only correct when the clip raster *is* the
            # original image; the mapper always supplies the real affine.
            logger.debug("geometry.encode: no transforms metadata; assuming identity affine")
            affine = alignment.PixelAffine.identity()

        packet = GeometryPacket(
            target_features=target.detach(),
            target_valid=valid,
            target_index=int(target_index),
            frame_ids=tuple(str(f) for f in frame_ids),
            geometry_hw=(h, w),
            original_to_geometry=affine,
            metadata={
                "clip_hw": (int(H), int(W)),
                "num_frames": int(T),
                "feature_mode": self.feature_mode,
                "checkpoint_sha256": self.checkpoint_sha256,
                "compute_dtype": str(self.compute_dtype),
                "tapped_layers": list(self.tapped_layers),
            },
            all_frame_features=(
                all_feats.permute(0, 3, 1, 2).contiguous() if all_feats is not None else None
            ),
        )
        packet.check(self.expected_feature_width)
        return packet

    # ------------------------------------------------------------------
    def _encode_tapped(self, clip: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Run the encoder once, capturing the requested cross-frame blocks.

        Returns [L, T, h, w, C] for the tapped layers (in increasing index
        order).  This is the optional multi-tap extraction path; hooks are
        registered for the duration of the call and removed afterwards.
        """
        assert clip.ndim == 5 and clip.shape[0] == 1, f"expected [1,T,3,H,W], got {tuple(clip.shape)}"
        T = clip.shape[1]
        stored: Dict[int, torch.Tensor] = {}

        def make_hook(idx):
            def hook(_m, _inp, out):
                stored[idx] = out.detach()

            return hook

        handles = [
            self.encoder.decoder[i].register_forward_hook(make_hook(i))
            for i in self.tapped_layers
        ]
        try:
            self.encoder(clip.to(self.compute_dtype))
        finally:
            for hnd in handles:
                hnd.remove()

        # The hook output is the block's *input* layout (B, N*hw, C) for
        # cross-frame (odd) blocks: reshape to [1, T, hw, C], drop the
        # 5 register tokens, and reshape to the spatial grid.
        ps = self.encoder.patch_start_idx
        C = self.encoder.dec_embed_dim
        out = []
        for i in self.tapped_layers:
            x = stored[i]  # [1, T*(ps+h*w), C]
            x = x.reshape(1, T, ps + h * w, C)[:, :, ps:, :]
            out.append(x[0].permute(0, 3, 1, 2))  # [T, C, h, w]
        return torch.stack(out, dim=0)


def _bf16_sdpa_supported(device: torch.device) -> bool:
    """Check that the SDPA backends used by Pi3's bf16 path exist on this
    GPU (FLASH_ATTENTION requires sm80+)."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q = torch.randn(2, 2, 16, 16, dtype=torch.bfloat16, device=device)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
        return True
    except Exception:
        return False


def build_geometry_encoder(cfg, device: Optional[torch.device] = None) -> Optional[Pi3GeometryEncoder]:
    """Instantiate the configured frozen geometry encoder (or None when the
    temporal branch is disabled).  Fails fast on missing checkpoints."""
    t = cfg.TEMPORAL
    if not t.ENABLED:
        return None
    if t.FUSION.ZERO_GEOMETRY:
        # Capacity control B2: no Pi3 call at all; the fusion module
        # consumes a zero tensor on the target-derived lattice.
        logger.info("TEMPORAL.FUSION.ZERO_GEOMETRY set: no Pi3 checkpoint required")
        return None
    gmos_root = t.GEOMETRY.GMOS_ROOT or os.environ.get("GMOS_ROOT", "")
    if not gmos_root:
        raise GeometryEncoderError(
            "TEMPORAL.GEOMETRY.GMOS_ROOT (or the GMOS_ROOT environment "
            "variable) must point at the GMOS checkout"
        )
    return Pi3GeometryEncoder(
        checkpoint=t.GEOMETRY.CHECKPOINT,
        gmos_root=gmos_root,
        feature_mode=t.GEOMETRY.FEATURE_MODE,
        tapped_layers=tuple(t.GEOMETRY.TAPPED_LAYERS),
        expected_feature_width=int(t.GEOMETRY.FEATURE_WIDTH),
        dtype=t.GEOMETRY.DTYPE,
        device=device,
    )

# ------------------------------------------------------------------------
# TG-PanSR: GMOS-style bottleneck fusion + per-level residual injection.
#
# Architecture (reference, see brief section 7 and docs/temporal_pansr.md):
#
#   Z_g = P_g(G_t)                                  # C_geo -> d
#   Z_a = align_swin_to_geo(P_a(S_a))                # C_a   -> d
#   Z_0 = Merge(concat(norm(Z_g), norm(Z_a)))        # 2d    -> d
#   Z   = SpatialTransformer(Z_0 + PE_2D, valid)     # [B,d,hg,wg]
#
#   per level s:
#     A_s = ChannelNorm(Conv1x1_s(S_s))              # C_s -> d_inj
#     M_s = align_geo_to_swin(Conv1x1'_s(Z))         # d   -> d_inj
#     H_s = LocalRefine_s(concat(A_s, M_s))          # 2*d_inj -> d_inj
#     D_s = Out_s(H_s)                               # d_inj -> C_s
#     q_s = sigmoid(Gate_s(concat(A_s, M_s)))        # [B,1,H,W]
#     S'_s = S_s + valid * q_s * D_s
#
# The final ``Out_s`` projections are initialized to zero (gate logits to
# zero => sigmoid 0.5), so the initial augmented pyramid is *exactly* the
# original Swin pyramid while the output projections receive gradients on
# the first step.
#
# Attribution: the bottleneck-fusion pattern (project, concat/merge,
# spatial-transformer refinement, residual injection) is inspired by the
# GMOS proposer (gmos/core/gmos_proposer.py, Apache-2.0 / Meta copyright
# headers preserved in that repo).  This implementation is a clean local
# re-implementation for a 2D sincos-positioned, maskable lattice; it does
# not import GMOS/SAM2 code and does not use SAM RoPE scales.
# ------------------------------------------------------------------------
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from . import alignment

__all__ = ["PerPixelLayerNorm", "PositionEmbeddingSine2D", "SpatialTransformer",
           "BottleneckFusion", "ResidualLevelAdapter", "TemporalFusionModule"]


# ---------------------------------------------------------------------------
# small building blocks
# ---------------------------------------------------------------------------
class PerPixelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of a [B, C, H, W] map (per-pixel)."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


def _conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)


class PositionEmbeddingSine2D(nn.Module):
    """Normalized 2D sine/cosine position encoding for an (h, w) grid.

    Each cell receives an embedding derived from its *normalized*
    (x, y) center coordinates in [0, 1], so the same module works for
    arbitrary rectangular grids (no square-image constant, no flattened
    1-D index).  The embedding width equals the model dimension ``d``;
    d/4 frequencies are used per axis.
    """

    def __init__(self, d_model: int, temperature: float = 10000.0):
        super().__init__()
        assert d_model % 8 == 0, f"d_model {d_model} must be divisible by 8"
        self.d_model = d_model
        self.half = d_model // 2  # embedding width per axis (sin/cos pairs)
        self.n_freq = self.half // 2  # d/4 frequencies per axis
        self.temperature = temperature

    def forward(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return [1, d_model, h, w]."""
        fy = torch.arange(h, device=device, dtype=torch.float32) + 0.5
        fx = torch.arange(w, device=device, dtype=torch.float32) + 0.5
        fy = fy / h  # normalize to (0, 1)
        fx = fx / w
        dim_t = torch.arange(self.n_freq, device=device, dtype=torch.float32)
        div_t = self.temperature ** (2 * (dim_t // 2) / self.n_freq)

        y_t = (fy[:, None] * math.pi * 0.5) / div_t[None, :]   # [h, n_freq]
        x_t = (fx[:, None] * math.pi * 0.5) / div_t[None, :]   # [w, n_freq]

        pos_y = y_t.new_zeros(h, self.half)
        pos_y[:, 0::2] = torch.sin(y_t)
        pos_y[:, 1::2] = torch.cos(y_t)
        pos_x = x_t.new_zeros(w, self.half)
        pos_x[:, 0::2] = torch.sin(x_t)
        pos_x[:, 1::2] = torch.cos(x_t)

        # interleave [y,x] components: [1, 2*half*... ] -> layout
        # [y_sin/cos (half), x_sin/cos (half)] = d_model total
        pe = torch.cat([pos_y[:, None, :].expand(h, w, -1),
                        pos_x[None, :, :].expand(h, w, -1)], dim=-1)  # [h, w, d]
        pe = pe.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)
        return pe


class SpatialTransformer(nn.Module):
    """Pre-norm 2D transformer over a (h, w) lattice with a valid mask.

    Uses torch.nn.functional.scaled_dot_product_attention (memory
    efficient; no attention weights materialized).  Invalid positions
    attend to nothing: their keys are masked out, and their outputs are
    zeroed after every block so invalid tokens never contaminate valid
    ones.  An all-invalid sample produces a zero output (no NaNs).
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True,
            ))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pe: torch.Tensor, valid: Optional[torch.Tensor] = None):
        """x: [B, d, h, w]; pe: [1, d, h, w]; valid: [B, h, w] bool or None."""
        B, d, h, w = x.shape
        z = (x + pe).permute(0, 2, 3, 1).reshape(B, h * w, d)  # [B, N, d]
        if valid is None:
            key_mask = None
            keep = None
        else:
            keep = valid.reshape(B, h * w)  # True = keep
            key_mask = ~keep  # True = ignored
            # A sample whose keys are *all* invalid would produce a softmax
            # over an entirely masked row (NaN).  Unmask one sentinel key
            # for such samples to keep the softmax well-defined; their
            # outputs are zeroed below regardless (torch.where is a select,
            # so any NaNs are discarded, not multiplied away).
            no_valid = ~keep.any(dim=1)  # [B]
            if no_valid.any():
                key_mask = key_mask.clone()
                key_mask[no_valid, 0] = False

        for layer in self.layers:
            z = layer(z, src_key_padding_mask=key_mask)
            if keep is not None:
                z = torch.where(keep.unsqueeze(-1), z, torch.zeros_like(z))

        z = self.norm(z)
        if keep is not None:
            z = torch.where(keep.unsqueeze(-1), z, torch.zeros_like(z))
        return z.reshape(B, h, w, d).permute(0, 3, 1, 2)


class BottleneckFusion(nn.Module):
    """Project geo + appearance to a common width, merge, refine."""

    def __init__(self, cfg_fusion, geo_channels: int, appearance_channels: int):
        super().__init__()
        d = int(cfg_fusion.DIM)
        self.d = d
        self.pos_enc = PositionEmbeddingSine2D(d)
        # P_g: geo projector (applied at the native geo grid)
        self.proj_geo = nn.Sequential(
            _conv1x1(geo_channels, d),
            PerPixelLayerNorm(d),
            nn.GELU(),
        )
        # P_a: appearance projector (applied after alignment to geo grid)
        self.proj_app = nn.Sequential(
            _conv1x1(appearance_channels, d),
            PerPixelLayerNorm(d),
            nn.GELU(),
        )
        # Merge: 2d -> d
        self.merge = nn.Sequential(
            _conv1x1(2 * d, d),
            nn.GELU(),
            PerPixelLayerNorm(d),
        )
        self.refine_layers = int(cfg_fusion.REFINE_LAYERS)
        self.transformer = (
            SpatialTransformer(d, int(cfg_fusion.NUM_HEADS), self.refine_layers,
                              int(cfg_fusion.FFN_DIM), float(cfg_fusion.DROPOUT))
            if self.refine_layers > 0 and cfg_fusion.MODE == "gmos_bottleneck_residual"
            else None
        )

    def forward(self, z_g: torch.Tensor, z_a: torch.Tensor, valid: Optional[torch.Tensor]):
        """z_g: [B, C_geo, hg, wg]; z_a: [B, C_app, hg, wg] (already aligned)."""
        a = self.proj_geo(z_g)
        b = self.proj_app(z_a)
        z = self.merge(torch.cat([a, b], dim=1))
        if self.transformer is not None:
            h, w = z.shape[-2:]
            pe = self.pos_enc(h, w, z.device, z.dtype)
            z = self.transformer(z, pe, valid)
        return z


class ResidualLevelAdapter(nn.Module):
    """Per-Swin-level gated residual adapter (brief section 7.4)."""

    def __init__(self, swin_channels: int, fusion_dim: int, inject_dim: int,
                 learned_gate: bool = True):
        super().__init__()
        self.swin_channels = swin_channels
        self.inject_dim = inject_dim
        # appearance branch A_s
        self.app_proj = nn.Sequential(
            _conv1x1(swin_channels, inject_dim),
            PerPixelLayerNorm(inject_dim),
        )
        # geometry branch M_s (before resampling onto the swin grid)
        self.geo_proj = _conv1x1(fusion_dim, inject_dim)
        # H_s = LocalRefine(concat(A_s, M_s)): 1x1 -> 64, norm, GELU,
        # depthwise 3x3, GELU
        self.local_refine = nn.Sequential(
            _conv1x1(2 * inject_dim, inject_dim),
            PerPixelLayerNorm(inject_dim),
            nn.GELU(),
            nn.Conv2d(inject_dim, inject_dim, kernel_size=3, padding=1, groups=inject_dim),
            nn.GELU(),
        )
        # D_s = Out_s(H_s): final projection, zero-initialized
        self.out = _conv1x1(inject_dim, swin_channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # gate
        self.learned_gate = learned_gate
        if learned_gate:
            self.gate = _conv1x1(2 * inject_dim, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)  # sigmoid(0) = 0.5

    def forward(self, s: torch.Tensor, z_geo: torch.Tensor,
                valid: Optional[torch.Tensor]) -> torch.Tensor:
        """s: [B, C_s, H, W] (Swin level); z_geo: [B, d, hg, wg] (fused geo)."""
        a = self.app_proj(s)                       # [B, k, H, W]
        m = F.interpolate(self.geo_proj(z_geo), size=(s.shape[-2], s.shape[-1]),
                          mode="bilinear", align_corners=False)
        h = self.local_refine(torch.cat([a, m], dim=1))
        d = self.out(h)
        if self.learned_gate:
            q = torch.sigmoid(self.gate(torch.cat([a, m], dim=1)))
        else:
            q = torch.ones_like(d[:, :1])
        if valid is not None:
            assert valid.shape[-2:] == (s.shape[-2], s.shape[-1])
            d = d * valid.unsqueeze(0)
        return d * q


class TemporalFusionModule(nn.Module):
    """Fusion of frozen geometry features with the Swin feature pyramid.

    Operates per sample: given one sample's Swin feature levels (dict of
    [C, H, W] tensors) and one geometry tensor [C_geo, hg, wg] with its
    :class:`alignment.FeatureGrid`, returns a new dict with the same
    names/shapes (residuals injected).
    """

    def __init__(self, cfg, backbone_shapes: Dict[str, "object"], geo_channels: int):
        """
        Args:
            cfg: full detectron2 cfg (uses cfg.TEMPORAL.FUSION).
            backbone_shapes: {name: ShapeSpec} from backbone.output_shape().
            geo_channels: verified feature width of the geometry encoder.
        """
        super().__init__()
        f = cfg.TEMPORAL.FUSION
        self.mode = f.MODE
        self.inject_features: List[str] = list(f.INJECT_FEATURES)
        self.appearance_source = f.APPEARANCE_SOURCE
        self.geo_channels = int(geo_channels)
        self.fusion_dim = int(f.DIM)

        shapes = {k: v for k, v in backbone_shapes.items()}
        if self.appearance_source not in shapes:
            raise ValueError(
                f"appearance source {self.appearance_source!r} not in backbone outputs "
                f"{sorted(shapes)}"
            )
        for name in self.inject_features:
            if name not in shapes:
                raise ValueError(f"inject feature {name!r} not in backbone outputs {sorted(shapes)}")

        self.bottleneck = BottleneckFusion(
            f, self.geo_channels, int(shapes[self.appearance_source].channels)
        )
        self.adapters: Dict[str, ResidualLevelAdapter] = nn.ModuleDict()
        for name in self.inject_features:
            self.adapters[name] = ResidualLevelAdapter(
                int(shapes[name].channels), self.fusion_dim, int(f.INJECT_DIM),
                learned_gate=bool(f.LEARNED_GATE),
            )

    # ------------------------------------------------------------------
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        geo_features: Optional[torch.Tensor],
        geo_grid: Optional[alignment.FeatureGrid],
        swin_grids: Dict[str, alignment.FeatureGrid],
    ) -> Dict[str, torch.Tensor]:
        """Fuse one sample.

        Args:
            features: {name: [C, H, W]} Swin levels (valid region only).
            geo_features: [C_geo, hg, wg] or None (fallback path -> the
                input features are returned unchanged).
            geo_grid: FeatureGrid of the geometry raster.
            swin_grids: {name: FeatureGrid} for every level that is
                injected or used as the appearance source.

        Returns: a NEW dict with identical names and shapes.
        """
        if self.mode == "off" or geo_features is None or geo_grid is None:
            return {k: v for k, v in features.items()}

        device = next(iter(features.values())).device

        # ---- appearance alignment onto the geometry lattice -------------
        app = features[self.appearance_source]
        app_grid = swin_grids[self.appearance_source]
        app_on_geo, app_valid = alignment.resample_features(
            app.unsqueeze(0), app_grid, geo_grid, mode="bilinear"
        )  # [1, C_app, hg, wg]
        valid_geo = geo_grid.valid_mask(device=device) & app_valid

        z = self.bottleneck(geo_features.unsqueeze(0), app_on_geo, valid_geo)

        # per-level validity of the geometry support (nearest)
        lvl_valid = {
            name: alignment.resample_validity(geo_grid, swin_grids[name], device=device)
            for name in self.inject_features
        }

        out: Dict[str, torch.Tensor] = dict(features)
        for name in self.inject_features:
            # adapters operate batched ([1, C, H, W]); input levels are
            # per-sample [C, H, W]
            delta = self.adapters[name](
                features[name].unsqueeze(0), z, lvl_valid[name]
            )
            out[name] = features[name] + delta[0]
        return out



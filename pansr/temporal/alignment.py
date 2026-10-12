# ------------------------------------------------------------------------
# TG-PanSR: pixel-center coordinate bookkeeping and feature resampling.
#
# Every feature map produced by a branch (Swin backbone, frozen Pi3)
# covers a *raster* (an image buffer) that is related to the original
# decoded image by a chain of geometric operations (resize, flip, crop,
# pad).  We record that chain as an axis-aligned affine transform in
# *pixel-center* coordinates and use it to move points (and feature
# values) between branches.
#
# Pixel-center convention
# -----------------------
# For an image of width W, the center of column ``j`` (0-based) has
# coordinate ``j + 0.5``.  A feature grid with stride ``s`` over a raster
# of width W has cell-center coordinate ``(j + 0.5) * s``: cell ``j``
# covers the raster pixel interval ``[j*s, (j+1)*s)`` and is anchored at
# the *center of that region*.  With this anchor, ``grid_sample`` with
# ``align_corners=False`` (see :func:`_to_grid_sample_coords`) returns a
# cell's value *exactly* when queried at the cell center, which is what
# makes identity resampling, stride-aligned crops, and flips of
# stride-divisible rasters exact.  The anchor convention is consistent
# with the half-pixel phase of cv2/PIL bilinear resampling (verified
# empirically in tests/temporal/test_alignment.py).
#
# Transform composition
# ---------------------
# Each transform maps *source* pixel centers to *destination* pixel
# centers as ``dst = a * src + b`` (independent axes; a < 0 encodes a
# flip).  Given two branches with affines ``A_s`` (original -> raster S)
# and ``A_g`` (original -> raster G), a point in raster S maps to raster
# G by ``A_g @ invert(A_s)`` (see :func:`map_points`).
#
# This module is pure torch/numpy math with no dataset or model imports.
# ------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "PixelAffine",
    "FeatureGrid",
    "resample_features",
    "resample_validity",
    "resize_geometry_size",
    "geo_to_swin_grid",
]


# ---------------------------------------------------------------------------
# Affine transforms in pixel-center coordinates
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PixelAffine:
    """Axis-aligned 2D affine in pixel-center coordinates.

    ``dst = (sx * src_x + ox, sy * src_y + oy)`` with ``sx`` / ``sy``
    negative when a flip is involved.  Only flips parallel to an axis
    occur in this pipeline.
    """

    sx: float = 1.0
    sy: float = 1.0
    ox: float = 0.0
    oy: float = 0.0

    @staticmethod
    def identity() -> "PixelAffine":
        return PixelAffine()

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """Map (..., 2) pixel-center points [x, y]."""
        assert pts.shape[-1] == 2
        out = torch.empty_like(pts, dtype=torch.float32)
        out[..., 0] = self.sx * pts[..., 0].float() + self.ox
        out[..., 1] = self.sy * pts[..., 1].float() + self.oy
        return out

    def invert(self) -> "PixelAffine":
        return PixelAffine(
            sx=1.0 / self.sx,
            sy=1.0 / self.sy,
            ox=-self.ox / self.sx,
            oy=-self.oy / self.sy,
        )

    def compose(self, other: "PixelAffine") -> "PixelAffine":
        """Return ``other @ self`` (apply ``self`` first, then ``other``)."""
        return PixelAffine(
            sx=other.sx * self.sx,
            sy=other.sy * self.sy,
            ox=other.sx * self.ox + other.ox,
            oy=other.sy * self.oy + other.oy,
        )

    def to_matrix(self) -> np.ndarray:
        """3x3 homogeneous matrix in pixel-center coordinates."""
        return np.array(
            [[self.sx, 0.0, self.ox], [0.0, self.sy, self.oy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    # -- primitive constructors ------------------------------------------
    @staticmethod
    def resize(src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> "PixelAffine":
        """Bilinear resize, align_corners=False (half-pixel) convention.

        Rasters produced by cv2/PIL bilinear (the PanSR pipeline) satisfy:
        a dst pixel at index ``y`` samples the source field at
        ``(y + 0.5) / s - 0.5`` with ``s = dst/src`` per axis.  Inverting
        for the source->raster direction and writing both endpoints as
        *pixel-center* coordinates gives

            dst_center = s * (src_center + 0.5)

        i.e. ``sx = s`` and ``ox = 0.5 * s`` (verified empirically
        against cv2.INTER_LINEAR and PIL.Image.BILINEAR; note that
        fvcore's ``ResizeTransform.apply_coords`` uses plain scaling
        without the half-pixel correction and must not be used as
        ground truth for this transform).
        """
        (sh, sw), (dh, dw) = src_hw, dst_hw
        sx, sy = dw / sw, dh / sh
        return PixelAffine(sx=sx, sy=sy, ox=0.5 * sx, oy=0.5 * sy)

    @staticmethod
    def hflip(width: int) -> "PixelAffine":
        """Horizontal flip of an image of the given width (in pixels)."""
        return PixelAffine(sx=-1.0, sy=1.0, ox=float(width), oy=0.0)

    @staticmethod
    def vflip(height: int) -> "PixelAffine":
        return PixelAffine(sx=1.0, sy=-1.0, ox=0.0, oy=float(height))

    @staticmethod
    def crop(offset_xy: Tuple[int, int]) -> "PixelAffine":
        """Crop a window starting at (cx, cy); dst = src - (cx, cy)."""
        cx, cy = offset_xy
        return PixelAffine(sx=1.0, sy=1.0, ox=-float(cx), oy=-float(cy))

    @staticmethod
    def pad(offset_xy: Tuple[int, int]) -> "PixelAffine":
        """Pad so that the content starts at (ox, oy)."""
        ox, oy = offset_xy
        return PixelAffine(sx=1.0, sy=1.0, ox=float(ox), oy=float(oy))


def map_points(pts: torch.Tensor, src: PixelAffine, dst: PixelAffine) -> torch.Tensor:
    """Map points from ``src``'s raster to ``dst``'s raster via the
    common original space: ``dst @ invert(src)``."""
    return dst.apply(src.invert().apply(pts))


# ---------------------------------------------------------------------------
# Feature grids
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureGrid:
    """A feature map's lattice inside a branch raster.

    Attributes:
        height/width: feature map size (cells).
        stride: pixel stride of one cell relative to ``raster``.
        raster_hw: size of the raster the grid indexes.
        affine: mapping *original image* pixel centers -> raster pixel
            centers for this branch.
        valid: optional [h, w] bool tensor marking valid cells (default
            all valid).
    """

    height: int
    width: int
    stride: int
    raster_hw: Tuple[int, int]
    affine: PixelAffine
    valid: Optional[torch.Tensor] = None

    def __post_init__(self):
        rh, rw = self.raster_hw
        if self.stride <= 0 or rh <= 0 or rw <= 0:
            raise ValueError(f"invalid grid dims: {self}")
        # A stride-``s`` convolution over a raster of size (rh, rw) produces
        # floor(rh/s) x floor(rw/s) cells (the last row/column may cover a
        # partial receptive field).  Rasters are therefore *not* required to
        # be divisible by the stride; the grid must match the actual
        # convolution output size.
        if (rh // self.stride, rw // self.stride) != (self.height, self.width):
            raise ValueError(
                f"grid {self.height}x{self.width} inconsistent with raster "
                f"{self.raster_hw} at stride {self.stride} "
                f"(expected {(rh // self.stride, rw // self.stride)})"
            )

    def cell_centers_raster(self, device="cpu") -> torch.Tensor:
        """[h*w, 2] pixel-center coordinates of all cell anchors in the
        raster (the center of the raster region each cell covers)."""
        ys = (torch.arange(self.height, dtype=torch.float32, device=device) + 0.5) * self.stride
        xs = (torch.arange(self.width, dtype=torch.float32, device=device) + 0.5) * self.stride
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    def cell_centers_original(self, device="cpu") -> torch.Tensor:
        return self.affine.invert().apply(self.cell_centers_raster(device))

    def valid_mask(self, device="cpu") -> torch.Tensor:
        if self.valid is not None:
            return self.valid.to(device=device, dtype=torch.bool)
        # A cell is valid when its raster center lies inside the raster
        # (pixel j spans [j, j+1); inside when 0 <= p < W).
        c = self.cell_centers_raster(device)
        rh, rw = self.raster_hw
        return ((c[..., 0] >= 0.0) & (c[..., 0] < rw)
                & (c[..., 1] >= 0.0) & (c[..., 1] < rh)).reshape(self.height, self.width)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
def _to_grid_sample_coords(
    raster_pts: torch.Tensor, src_grid: "FeatureGrid"
) -> torch.Tensor:
    """Raster pixel-center coords -> grid_sample normalized coords for a
    feature tensor of ``src_grid`` (align_corners=False).

    A raster point ``p`` maps to feature-field pixel position ``p / s``
    (feature pixel ``k`` is centered at position ``k + 0.5``, the usual
    ``align_corners=False`` layout).  With the region-center cell anchors
    of :meth:`FeatureGrid.cell_centers_raster`, a query at cell ``k``'s
    anchor ``(k + 0.5) * s`` evaluates to ``k + 0.5`` exactly, so the
    cell's own value is reproduced without blending.  Verified in
    tests/temporal/test_alignment.py (identity, flip, aligned-crop, and
    analytic-bilinear cases).
    """
    sh, sw = src_grid.raster_hw
    sfh, sfw = src_grid.height, src_grid.width
    x = 2.0 * raster_pts[..., 0] / (src_grid.stride * sfw) - 1.0
    y = 2.0 * raster_pts[..., 1] / (src_grid.stride * sfh) - 1.0
    if sfw == 1:
        x = torch.zeros_like(x)
    if sfh == 1:
        y = torch.zeros_like(y)
    return torch.stack([x, y], dim=-1)


def resample_features(
    src: torch.Tensor,
    src_grid: FeatureGrid,
    dst_grid: FeatureGrid,
    mode: str = "bilinear",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resample ``src`` [C, h, w] (or [B, C, h, w]) from ``src_grid`` onto
    ``dst_grid`` using bilinear (continuous fields) or nearest (categorical
    fields) interpolation over the common original space.

    Returns:
        (out [C, h', w'] or [B, C, h', w'], valid [h', w'] bool)
    """
    assert mode in ("bilinear", "nearest"), mode
    was_2d = src.ndim == 3
    if was_2d:
        src = src.unsqueeze(0)

    # destination cell centers in destination raster coordinates
    dst_pts = dst_grid.cell_centers_raster(device=src.device)          # [N, 2]
    # destination centers in the common original space
    orig_pts = dst_grid.affine.invert().apply(dst_pts)                 # [N, 2]
    # original -> source raster
    src_pts = src_grid.affine.apply(orig_pts)                          # [N, 2]

    # validity in the source raster (pixel j spans [j, j+1); inside when
    # 0 <= p < W).  A cell whose center is inside is resampled with border
    # (replicate) padding so its bilinear support never pulls zeros.
    sh, sw = src_grid.raster_hw
    in_src = (
        (src_pts[..., 0] >= 0.0) & (src_pts[..., 0] < sw)
        & (src_pts[..., 1] >= 0.0) & (src_pts[..., 1] < sh)
    )
    if dst_grid.valid is not None:
        in_src = in_src & dst_grid.valid.reshape(-1).to(src.device)
    valid = in_src.reshape(dst_grid.height, dst_grid.width)

    grid = _to_grid_sample_coords(src_pts, src_grid).reshape(
        src.size(0), dst_grid.height, dst_grid.width, 2
    )
    out = F.grid_sample(
        src, grid, mode=mode, align_corners=False, padding_mode="border"
    )
    out = out * in_src.reshape(src.size(0), 1, dst_grid.height, dst_grid.width).to(out.dtype)
    if was_2d:
        out = out[0]
    return out, valid


def resize_geometry_size(
    raster_hw: Tuple[int, int], pixel_budget: int, patch_size: int = 14
) -> Tuple[int, int]:
    """Compute the geometry raster size for a given source raster.

    Preserves the aspect ratio, keeps both dimensions positive multiples of
    ``patch_size``, and stays within ``pixel_budget`` pixels.  This follows
    GMOS's ``pil_image_to_tensor_pi3`` sizing (budget 255000, dims % 14).
    """
    sh, sw = raster_hw
    if sh <= 0 or sw <= 0:
        raise ValueError(f"invalid raster size {raster_hw}")
    scale = (pixel_budget / float(sh * sw)) ** 0.5
    target_w, target_h = sw * scale, sh * scale
    k = int(round(target_w / patch_size))
    m = int(round(target_h / patch_size))
    k = max(1, k)
    m = max(1, m)
    while (k * patch_size) * (m * patch_size) > pixel_budget:
        if k / m > target_w / target_h:
            k -= 1
        else:
            m -= 1
    return m * patch_size, k * patch_size


def geo_to_swin_grid(
    geo_raster_hw: Tuple[int, int],
    geo_patch: int,
    swin_raster_hw: Tuple[int, int],
    swin_stride: int,
    geo_affine: PixelAffine,
    swin_affine: PixelAffine,
) -> Tuple[FeatureGrid, FeatureGrid]:
    """Convenience: build the geometry and Swin feature grids for a pair of
    rasters that share the same spatial augmentation (the reference
    shared-view path)."""
    gh, gw = geo_raster_hw
    return (
        FeatureGrid(gh // geo_patch, gw // geo_patch, geo_patch, geo_raster_hw, geo_affine),
        FeatureGrid(swin_raster_hw[0] // swin_stride, swin_raster_hw[1] // swin_stride,
                    swin_stride, swin_raster_hw, swin_affine),
    )


def resample_validity(src_grid: FeatureGrid, dst_grid: FeatureGrid,
                      device="cpu") -> torch.Tensor:
    """For each cell of ``dst_grid``: True iff its center maps into the
    ``src_grid`` raster (and the dst cell itself is valid).  This is the
    boolean counterpart of :func:`resample_features` without interpolation.
    """
    dst_pts = dst_grid.cell_centers_raster(device)
    orig_pts = dst_grid.affine.invert().apply(dst_pts)
    src_pts = src_grid.affine.apply(orig_pts)
    sh, sw = src_grid.raster_hw
    in_src = (
        (src_pts[..., 0] >= 0.0) & (src_pts[..., 0] < sw)
        & (src_pts[..., 1] >= 0.0) & (src_pts[..., 1] < sh)
    )
    if dst_grid.valid is not None:
        in_src = in_src & dst_grid.valid.reshape(-1).to(device)
    return in_src.reshape(dst_grid.height, dst_grid.width)

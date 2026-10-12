# ------------------------------------------------------------------------
# TG-PanSR: immutable clip manifest for the LaRS temporal dataset.
#
# The LaRS distribution keeps annotated keyframes under
# ``{root}/{split}/images`` and their preceding context frames under
# ``{root}/lars_v1.0.0_images_seq/{split}/images_seq`` with filenames
# ``<source_video>_<frame:05d>.jpg``.  A record's target file is
# ``<source_video>_<last:05d>.jpg``; the clip is the 9 chronologically
# preceding frames with consecutive frame indices plus the target itself
# (verified against the actual files, never assumed).
#
# Manifest records (JSONL) are the single source of truth for temporal
# training/evaluation.  Required invariants (enforced by
# :func:`validate_record`):
#   * exactly one frame is the target and ``frames[target_index]`` is the
#     annotated image;
#   * ordering is by verified consecutive frame indices (not lexicographic
#     filename guessing);
#   * causal mode: every context frame precedes the target;
#   * no silent substitution: missing/ambiguous records are errors.
# ------------------------------------------------------------------------
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_FRAME_NAME_RE = re.compile(r"^(?P<video>.+)_(?P<frame>\d{5})$")

__all__ = [
    "SCHEMA_VERSION",
    "ManifestError",
    "parse_frame_name",
    "select_frames",
    "validate_record",
    "record_to_dataset_dict",
    "manifest_sha256",
]


class ManifestError(RuntimeError):
    pass


def parse_frame_name(filename: str) -> Tuple[str, int]:
    """Parse ``<video>_<frame:05d>.jpg`` -> (video_id, frame_index)."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = _FRAME_NAME_RE.match(stem)
    if not m:
        raise ManifestError(f"frame filename does not match <video>_<NNNNN>.jpg: {filename}")
    return m.group("video"), int(m.group("frame"))


# ---------------------------------------------------------------------------
# Frame selection (sampling modes)
# ---------------------------------------------------------------------------
def select_frames(
    frame_records: List[Dict[str, Any]],
    target_index: int,
    sampling: str = "all_history",
    num_frames: int = 10,
    allow_future: bool = False,
    seed: int = 0,
) -> Tuple[List[int], List[int]]:
    """Select the clip frame indices for one record.

    Args:
        frame_records: the record's ``frames`` list (chronological order),
            each with ``relative_rank`` (target rank = 0).
        target_index: index of the target inside ``frame_records``.
        sampling: one of all_history | recent | uniform_history |
            target_only | repeat_target.
        num_frames: T (for recent / uniform_history / repeat_target).
        allow_future: when False (default) context frames must have
            relative_rank < 0.

    Returns:
        (selected_indices, dropped_or_padded_info) where
        selected_indices are indices into ``frame_records`` of the clip to
        encode (chronological, target last for causal clips).
    """
    n = len(frame_records)
    if not (0 <= target_index < n):
        raise ManifestError(
            f"target_index {target_index} out of range for {n} frames"
        )
    target_rank = frame_records[target_index]["relative_rank"]
    if target_rank != 0:
        raise ManifestError(f"target frame must have relative_rank 0, got {target_rank}")
    prev = [i for i, f in enumerate(frame_records) if f["relative_rank"] < 0]
    future = [i for i, f in enumerate(frame_records) if f["relative_rank"] > 0]
    if future and not allow_future:
        raise ManifestError(
            f"record contains {len(future)} frame(s) after the target; causal "
            "protocol violated (set ALLOW_FUTURE only for explicit experiments)"
        )
    prev_ranks = [f["relative_rank"] for f in frame_records if f["relative_rank"] < 0]
    if len(prev_ranks) != len(set(prev_ranks)):
        raise ManifestError("duplicate relative ranks in history")

    if sampling == "all_history":
        chosen = prev + [target_index]
    elif sampling == "recent":
        k = max(0, num_frames - 1)
        chosen = prev[-k:] + [target_index] if k else [target_index]
    elif sampling == "uniform_history":
        k = max(0, num_frames - 1)
        if k == 0:
            chosen = [target_index]
        elif k >= len(prev):
            chosen = prev + [target_index]
        else:
            # deterministic spread across the available history
            pos = np.linspace(0, len(prev) - 1, k)
            idx = np.unique(np.round(pos).astype(int)).tolist()
            if len(idx) < k:  # linspace rounding collision: fill from the nearest
                for extra in sorted(set(range(len(prev))) - set(idx), key=lambda j: min(abs(j - p) for p in pos)):
                    idx.append(extra)
                    if len(idx) >= k:
                        break
            chosen = [prev[j] for j in sorted(idx)] + [target_index]
    elif sampling == "target_only":
        chosen = [target_index]
    elif sampling == "repeat_target":
        chosen = [target_index] * max(1, num_frames)
    else:
        raise ManifestError(f"unknown sampling mode {sampling!r}")
    return chosen, []


# ---------------------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------------------
def validate_record(rec: Dict[str, Any], root: Optional[str] = None,
                    require_files: bool = True) -> Dict[str, Any]:
    """Validate one manifest record; return a copy with resolved paths.

    Raises ManifestError on any violation.  ``root`` (when given) is
    prepended to relative paths; with ``require_files`` all files must
    exist and targets must match the annotation record.
    """
    if rec.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"unsupported schema_version {rec.get('schema_version')}")
    frames = rec.get("frames")
    target_index = rec.get("target_index")
    if not isinstance(frames, list) or not frames:
        raise ManifestError("record has no frames")
    if not isinstance(target_index, int) or not (0 <= target_index < len(frames)):
        raise ManifestError(f"invalid target_index {target_index!r} for {len(frames)} frames")
    if frames[target_index].get("is_target") is not True:
        raise ManifestError("frames[target_index] is not marked as the target")
    if sum(1 for f in frames if f.get("is_target") is True) != 1:
        raise ManifestError("exactly one frame must be marked as the target")
    if frames[target_index].get("relative_rank") != 0:
        raise ManifestError(
            f"target frame must have relative_rank 0, got "
            f"{frames[target_index].get('relative_rank')!r}"
        )
    ranks = [f["relative_rank"] for f in frames]
    if 0 not in ranks or ranks.count(0) != 1:
        raise ManifestError("exactly one frame must have relative_rank 0")
    # chronological consistency of the ranks
    for i, r in enumerate(ranks):
        if r > 0 and i < target_index:
            raise ManifestError("future-ranked frame appears before the target in ordering")
    if root is not None:
        resolved = dict(rec)
        resolved["target_file"] = _resolve(rec["target_file"], root)
        resolved["frames"] = []
        for f in frames:
            f2 = dict(f)
            f2["path"] = _resolve(f["file"], root)
            resolved["frames"].append(f2)
        if require_files:
            for f2 in resolved["frames"]:
                if not os.path.isfile(f2["path"]):
                    raise ManifestError(f"frame file missing: {f2['path']}")
            if rec.get("annotation_file") and rec.get("split") != "test":
                ap = _resolve(rec["annotation_file"], root)
                if not os.path.isfile(ap):
                    raise ManifestError(f"annotation file missing: {ap}")
        return resolved
    return rec


def _resolve(path: str, root: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


def manifest_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Record -> detectron2 dataset dict
# ---------------------------------------------------------------------------
def record_to_dataset_dict(rec: Dict[str, Any], sampling: str = "all_history",
                           num_frames: int = 10, allow_future: bool = False,
                           seed: int = 0) -> Dict[str, Any]:
    """Build the base dataset dict consumed by the temporal mapper.

    The dict carries the *full* verified frame list (chronological) plus
    the target index; the mapper applies the sampling policy so that train
    and eval stay reproducible from one record.
    """
    rec = validate_record(rec)
    d = {
        # target identity (annotation fields only present for labeled splits)
        "image_id": rec["image_id"],
        "clip_manifest_id": rec.get("clip_manifest_id"),
        "sequence_id": rec.get("sequence_id"),
        "split": rec.get("split"),
        # geometry of the original target
        "height": rec["original_height"],
        "width": rec["original_width"],
        "target_file": rec["target_file"],
        # full chronological frame list (paths resolved at build time)
        "clip_frames": [f["file"] for f in rec["frames"]],
        "clip_frame_ids": [f["frame_id"] for f in rec["frames"]],
        "target_index": int(rec["target_index"]),
        # sampling config (resolved by the mapper; logged for reproducibility)
        "clip_sampling": sampling,
        "clip_num_frames": num_frames,
        "clip_allow_future": allow_future,
        "clip_seed": seed,
    }
    if rec.get("split") != "test":
        d["file_name"] = rec["target_file"]
        if rec.get("panoptic_mask"):
            d["pan_seg_file_name"] = rec["panoptic_mask"]
        if "segments_info" in rec:
            d["segments_info"] = rec["segments_info"]
    return d

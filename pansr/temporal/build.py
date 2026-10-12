# ------------------------------------------------------------------------
# TG-PanSR: LaRS clip discovery and manifest construction.
#
# Layout handled here (verified against the local distribution, see
# docs/data_audit.md):
#
#   $LARS_ROOT/{train,val}/images/<video>_<frame:05d>.jpg      annotated
#   $LARS_ROOT/{train,val}/panoptic_masks/<video>_<frame:05d>.png
#   $LARS_ROOT/{train,val}/{panoptic|mmdet}_annotations.json
#   $LARS_ROOT/{test}/images/...                                (unlabeled)
#   $LARS_ROOT/lars_v1.0.0_images_seq/{train,val,test}/images_seq/<video>_<frame:05d>.jpg
#
# A target ``<video>_<last>.jpg`` owns the clip
# ``<video>_{last-9 .. last}.jpg`` (nine preceding frames + target),
# verified file by file.
# ------------------------------------------------------------------------
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .manifest import ManifestError, SCHEMA_VERSION, parse_frame_name

logger = logging.getLogger(__name__)

CLIP_CONTEXT_FRAMES = 9  # official LaRS: nine preceding frames + keyframe


@dataclass
class BuildReport:
    split: str
    n_targets: int = 0
    n_ok: int = 0
    n_missing_frames: int = 0
    n_gaps: int = 0
    n_target_copy_mismatch: int = 0
    missing: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    target_copy_mismatch: List[str] = field(default_factory=list)
    duplicates_hashed: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "n_targets": self.n_targets,
            "n_ok": self.n_ok,
            "n_missing_frames": self.n_missing_frames,
            "n_gaps": self.n_gaps,
            "n_target_copy_mismatch": self.n_target_copy_mismatch,
            "missing_examples": self.missing[:50],
            "gap_examples": self.gaps[:50],
            "target_copy_mismatch_examples": self.target_copy_mismatch[:50],
            "duplicates_hashed": self.duplicates_hashed,
            "duration_seconds": self.duration_seconds,
        }


def _seq_dir(lars_root: str, split: str) -> str:
    for cand in (
        os.path.join(lars_root, "lars_v1.0.0_images_seq", split, "images_seq"),
        os.path.join(lars_root, "images_seq", split, "images_seq"),
        os.path.join(lars_root, "images_seq", split),
    ):
        if os.path.isdir(cand):
            return cand
    raise ManifestError(f"sequence image directory not found under {lars_root} for split {split}")


def _annotation_file(lars_root: str, split: str, annotation_file: str) -> str:
    p = os.path.join(lars_root, split, annotation_file)
    if os.path.isfile(p):
        return p
    alt = os.path.join(lars_root, split, "panoptic_annotations.json")
    if annotation_file == "mmdet_annotations.json" and os.path.isfile(alt):
        logger.warning(
            "%s not found; falling back to %s for clip discovery "
            "(category IDs differ, but file names/sizes are identical)",
            p, alt,
        )
        return alt
    raise ManifestError(f"annotation file not found: {p}")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _list_seq_video_frames(seq_dir: str) -> Dict[str, List[int]]:
    """video_id -> sorted list of available frame indices (flat dir)."""
    out: Dict[str, List[int]] = defaultdict(list)
    for name in os.listdir(seq_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            video, frame = parse_frame_name(name)
        except ManifestError:
            continue
        out[video].append(frame)
    for v in out:
        out[v].sort()
    return out


def build_manifest_for_split(
    lars_root: str,
    split: str,
    annotation_file: str = "mmdet_annotations.json",
    clip_context: int = CLIP_CONTEXT_FRAMES,
    strict: bool = True,
    verify_target_copies: bool = True,
) -> Tuple[List[Dict[str, Any]], BuildReport]:
    """Discover and validate clips for one split; return (records, report).

    With ``strict`` (default) any target with missing context frames or
    non-consecutive indices is an error (the record is not written and the
    problem is reported).  With ``strict=False`` such targets are skipped
    and reported.
    """
    ann_path = _annotation_file(lars_root, split, annotation_file)
    with open(ann_path) as f:
        anns = json.load(f)
    seq_dir = _seq_dir(lars_root, split)
    seq_frames = _list_seq_video_frames(seq_dir)

    report = BuildReport(split=split)
    records: List[Dict[str, Any]] = []
    seen_pairs = set()

    for img in anns.get("images", []):
        fname = img["file_name"]
        video, last = parse_frame_name(fname)
        ann = next(
            (a for a in anns["annotations"] if a["image_id"] == img["id"]), None
        )
        if ann is None:
            raise ManifestError(f"no annotation for image {img['id']} ({fname})")
        report.n_targets += 1

        available = seq_frames.get(video, [])
        if not available or available[-1] < last:
            report.n_missing_frames += 1
            report.missing.append(f"{video}: target frame {last} not in sequence archive")
            if strict:
                raise ManifestError(f"target {fname}: target frame not in sequence archive")
            continue

        # expected window is the last (clip_context+1) consecutive frames;
        # if the sequence archive starts *later* than that (video begins at
        # frame > 0), the history is truncated to the verified available
        # frames (documented in the record; no silent substitution).
        first_avail = available[0]
        start = max(0, last - clip_context)
        history_truncated = False
        if start < first_avail:
            if first_avail > last:
                report.n_missing_frames += 1
                report.missing.append(f"{video}: no context frames before {last}")
                if strict:
                    raise ManifestError(f"target {fname}: no context frames")
                continue
            start = first_avail
            history_truncated = True
        need = list(range(start, last + 1))
        avail_set = set(available)
        missing = [f for f in need if f not in avail_set]
        if missing:
            if all(f < first_avail for f in missing):
                report.n_missing_frames += 1
                report.missing.append(f"{video}: missing frames {missing}")
            else:
                report.n_gaps += 1
                report.gaps.append(f"{video}: non-consecutive window around frame {last} (missing {missing})")
            if strict:
                raise ManifestError(f"target {fname}: missing/non-consecutive frames {missing}")
            continue

        # target copy consistency (seq dir vs images dir)
        seq_target = os.path.join(seq_dir, fname)
        img_target = os.path.join(lars_root, split, "images", fname)
        if verify_target_copies and os.path.isfile(img_target):
            report.duplicates_hashed += 1
            h1 = _sha256(seq_target)
            h2 = _sha256(img_target)
            if h1 != h2:
                # different encodings are possible; record for the audit
                report.n_target_copy_mismatch += 1
                report.target_copy_mismatch.append(f"{fname} (byte hash differs)")

        key = (video, last)
        if key in seen_pairs:
            raise ManifestError(f"duplicate target {fname} in {split}")
        seen_pairs.add(key)

        frames = []
        for i, fr in enumerate(need):
            is_target = fr == last
            frames.append({
                "file": os.path.join(split, "images", fname) if is_target
                       else os.path.join("lars_v1.0.0_images_seq", split, "images_seq",
                                         f"{video}_{fr:05d}.jpg"),
                "frame_id": f"{video}_{fr:05d}",
                "timestamp_seconds": None,
                "relative_rank": fr - last,
                "is_target": is_target,
            })
        rec: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "image_id": img["id"],
            "sequence_id": video,
            "source_video_id": video,
            "target_file": os.path.join(split, "images", fname),
            "frames": frames,
            "target_index": last - start,
            "annotation_file": os.path.join(split, os.path.basename(ann_path)),
            "panoptic_mask": os.path.join(split, "panoptic_masks",
                                          os.path.splitext(fname)[0] + ".png"),
            "original_height": img["height"],
            "original_width": img["width"],
        }
        if history_truncated or (last - start) < clip_context:
            rec["history_truncated"] = True
            rec["history_note"] = (
                "sequence archive starts at frame "
                f"{need[0]}; target has {len(need) - 1} verified preceding "
                "frames instead of 9 (first annotated keyframe of the source)"
            )
        records.append(rec)
        report.n_ok += 1

    return records, report


def split_overlap_report(lars_root: str, splits=("train", "val"),
                         annotation_file: str = "mmdet_annotations.json") -> Dict[str, Any]:
    """Check for suspicious cross-split raw-frame / source-video overlap."""
    sets = {}
    videos = {}
    for s in splits:
        ann_path = _annotation_file(lars_root, s, annotation_file)
        with open(ann_path) as f:
            anns = json.load(f)
        files = {img["file_name"] for img in anns.get("images", [])}
        vids = {parse_frame_name(f)[0] for f in files}
        sets[s] = files
        videos[s] = vids
    out: Dict[str, Any] = {"pairwise": {}}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            out["pairwise"][f"{a}∩{b}"] = {
                "target_files": len(sets[a] & sets[b]),
                "source_videos": len(videos[a] & videos[b]),
                "source_video_examples": sorted(videos[a] & videos[b])[:20],
            }
    return out

# ------------------------------------------------------------------------
# TG-PanSR: temporal dataset registration.
#
# The temporal datasets (``lars_train_temporal_panoptic`` /
# ``lars_val_temporal_panoptic``) are built from immutable JSONL manifests
# (see pansr.temporal.build / tools/temporal/build_manifest.py) on top of
# the *same* LaRS annotation files that the vanilla datasets use (the
# mmdet-converted ``mmdet_annotations.json``).  Metadata is identical to
# the vanilla registration so the original evaluator and export logic
# apply unchanged.
# ------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from detectron2.data import DatasetCatalog, MetadataCatalog

from .manifest import ManifestError, manifest_sha256, record_to_dataset_dict

logger = logging.getLogger(__name__)

__all__ = ["build_temporal_dataset_dicts", "register_temporal_lars"]


def _load_manifest(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise ManifestError(f"manifest not found: {path}")
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ManifestError(f"manifest {path}:{i+1}: invalid JSON: {e}")
    if not records:
        raise ManifestError(f"manifest {path} is empty")
    return records


def _load_annotation_index(
    lars_root: str, annotation_file: str,
    cache: Dict[str, Tuple[Dict[Any, Dict[str, Any]], Dict[int, bool]]],
) -> Tuple[Dict[Any, Dict[str, Any]], Dict[int, bool]]:
    """(image_id -> annotation record, category_id -> isthing), cached per file."""
    if annotation_file in cache:
        return cache[annotation_file]
    path = os.path.join(lars_root, annotation_file)
    with open(path) as f:
        anns = json.load(f)
    index = {a["image_id"]: a for a in anns.get("annotations", [])}
    isthing = {c["id"]: bool(c["isthing"]) for c in anns.get("categories", [])}
    cache[annotation_file] = (index, isthing)
    return index, isthing


def build_temporal_dataset_dicts(
    manifest_path: str,
    lars_root: str,
    sampling: str = "all_history",
    num_frames: int = 10,
    allow_future: bool = False,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Load a manifest and materialize the temporal dataset dicts.

    ``segments_info`` (per-instance category/id/iscrowd/cls_weight) is
    joined from the same annotation file the manifest record points at, so
    the temporal dict is annotation-identical to the vanilla dataset dict.
    """
    records = _load_manifest(manifest_path)
    root = os.path.abspath(lars_root)
    ann_cache: Dict[str, Tuple[Dict[Any, Dict[str, Any]], Dict[int, bool]]] = {}
    out = []
    for rec in records:
        from .manifest import validate_record

        rec = validate_record(rec, root=root, require_files=False)  # fast path
        d = record_to_dataset_dict(
            rec, sampling=sampling, num_frames=num_frames,
            allow_future=allow_future, seed=seed,
        )
        # resolve paths against the dataset root
        d["target_file"] = os.path.join(root, d["target_file"])
        d["clip_frames"] = [os.path.join(root, p) for p in d["clip_frames"]]
        if "pan_seg_file_name" in d:
            d["pan_seg_file_name"] = os.path.join(root, d["pan_seg_file_name"])
        if d.get("split") != "test" and rec.get("annotation_file"):
            index, isthing_map = _load_annotation_index(
                root, rec["annotation_file"], ann_cache
            )
            ann = index.get(rec["image_id"])
            if ann is None:
                raise ManifestError(
                    f"manifest record image_id {rec['image_id']} not found in "
                    f"{rec['annotation_file']}"
                )
            # mirror the vanilla get_lars_dataset: attach isthing + cls_weight
            segments_info = []
            for seg in ann["segments_info"]:
                seg = dict(seg)
                seg["isthing"] = int(isthing_map[seg["category_id"]])
                seg["cls_weight"] = 1.0
                segments_info.append(seg)
            d["segments_info"] = segments_info
        out.append(d)
    return out


def register_temporal_lars(
    cfg,
    manifest_train: Optional[str] = None,
    manifest_val: Optional[str] = None,
    lars_root: Optional[str] = None,
    name_train: str = "lars_train_temporal_panoptic",
    name_val: str = "lars_val_temporal_panoptic",
) -> None:
    """Register the temporal datasets and (re)use the vanilla LaRS metadata.

    Must be called after ``add_pansr_config`` + config merge so that the
    base LaRS registration (which provides the metadata) has run.  The
    metadata is copied from the corresponding vanilla dataset
    (``lars_{split}_panoptic``) so category mappings, colors and
    evaluator type are exactly the original ones.
    """
    t = cfg.TEMPORAL
    if manifest_train is None:
        manifest_train = t.MANIFEST_TRAIN
    if manifest_val is None:
        manifest_val = t.MANIFEST_VAL
    if lars_root is None:
        lars_root = os.environ.get("LARS_ROOT", "datasets/LaRS")

    if manifest_train and name_train not in DatasetCatalog:
        DatasetCatalog.register(
            name_train,
            lambda: build_temporal_dataset_dicts(
                manifest_train, lars_root,
                sampling=t.SAMPLING, num_frames=t.NUM_FRAMES,
                allow_future=t.ALLOW_FUTURE, seed=t.SAMPLING_SEED,
            ),
        )
        _copy_metadata(name_train, "lars_train_panoptic")
        logger.info(
            "Registered temporal train dataset %s from %s (sha256=%s)",
            name_train, manifest_train, manifest_sha256(manifest_train)[:12],
        )
    if manifest_val and name_val not in DatasetCatalog:
        DatasetCatalog.register(
            name_val,
            lambda: build_temporal_dataset_dicts(
                manifest_val, lars_root,
                sampling=t.SAMPLING, num_frames=t.NUM_FRAMES,
                allow_future=t.ALLOW_FUTURE, seed=t.SAMPLING_SEED,
            ),
        )
        _copy_metadata(name_val, "lars_val_panoptic")
        logger.info(
            "Registered temporal val dataset %s from %s (sha256=%s)",
            name_val, manifest_val, manifest_sha256(manifest_val)[:12],
        )


def _copy_metadata(dst: str, src: str) -> None:
    src_meta = MetadataCatalog.get(src)
    dst_meta = MetadataCatalog.get(dst)
    for key, value in src_meta.as_dict().items():
        if key == "name" or key in dst_meta.__dict__:
            continue
        dst_meta.set(**{key: value})

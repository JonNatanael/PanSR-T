#!/usr/bin/env python3
"""Run a trained (TG-)PanSR model on LaRS test and build a submission ZIP.

The default checkpoint is the filename stored in MODEL_DIR/last_checkpoint.
Predictions follow the MaCVi panoptic wire format exactly: R is the LaRS
class ID and the 16-bit instance ID is G*256+B.  PNGs are stored directly
at the root of the generated ZIP.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CLASS_ID_MAP = {
    1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16,
    7: 17, 8: 19, 9: 1, 10: 3, 11: 5, 12: 0,
}
THING_CONTIGUOUS_IDS = set(range(1, 9))
VALID_LARS_CLASSES = {0, 1, 3, 5, 11, 12, 13, 14, 15, 16, 17, 19}
STUFF_LARS_CLASSES = {0, 1, 3, 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Training output directory")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Checkpoint path/name; defaults to MODEL_DIR/last_checkpoint",
    )
    parser.add_argument(
        "--config-file", default=None,
        help="Config file; defaults to MODEL_DIR/config.yaml",
    )
    parser.add_argument("--lars-root", default=os.environ.get("LARS_ROOT", "datasets/LaRS"))
    parser.add_argument("--manifest", default="test.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--zip", dest="zip_path", default=None,
        help="Submission ZIP path (default: MODEL_DIR/lars_test_submission.zip)",
    )
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--geometry-checkpoint", default=None)
    parser.add_argument("--gmos-root", default=os.environ.get("GMOS_ROOT", ""))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--cache", choices=("auto", "require", "off"), default="auto",
        help="Use a complete compatible test Pi3 cache when available",
    )
    parser.add_argument("--resume", action="store_true", help="Keep valid existing PNGs")
    parser.add_argument("--limit", type=int, default=None, help="Debug only: infer first N records")
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:1, or cpu")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != "test":
                raise ValueError(f"{path}:{line_no}: expected split='test'")
            records.append(record)
    if not records:
        raise ValueError(f"empty manifest: {path}")
    return records


def resolve_checkpoint(model_dir: str, requested: str | None) -> str:
    if requested is None:
        marker = os.path.join(model_dir, "last_checkpoint")
        if not os.path.isfile(marker):
            raise FileNotFoundError(f"last-checkpoint marker not found: {marker}")
        with open(marker) as f:
            requested = f.read().strip()
        if not requested:
            raise ValueError(f"empty last-checkpoint marker: {marker}")
    path = os.path.expanduser(requested)
    if not os.path.isabs(path):
        # A bare filename normally refers to MODEL_DIR; an explicitly
        # reachable relative path is also accepted from the current cwd.
        path = path if os.path.isfile(path) else os.path.join(model_dir, path)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def register_inference_metadata(name: str = "lars_test_inference") -> str:
    from detectron2.data import MetadataCatalog

    MetadataCatalog.get(name).set(
        thing_dataset_id_to_contiguous_id={k: i for i, k in enumerate(
            (11, 12, 13, 14, 15, 16, 17, 19), 1
        )},
        stuff_dataset_id_to_contiguous_id={
            k: i for i, k in enumerate((11, 12, 13, 14, 15, 16, 17, 19, 1, 3, 5, 0), 1)
        },
        category_id_map=CLASS_ID_MAP,
        ignore_label=255,
        label_divisor=256 * 256,
    )
    return name


def compatible_cache_count(cfg, records: Iterable[Dict[str, Any]], cache_dir: str) -> Tuple[int, int]:
    from pansr.temporal.cache import GeometryFeatureCache, content_fingerprint
    from pansr.temporal.manifest import select_frames

    checkpoint = os.path.expanduser(cfg.TEMPORAL.GEOMETRY.CHECKPOINT)
    if not os.path.isfile(checkpoint):
        return 0, sum(1 for _ in records)
    checkpoint_hash = content_fingerprint(checkpoint)
    records = list(records)
    hits = 0
    for record in records:
        chosen, _ = select_frames(
            record["frames"], int(record["target_index"]),
            sampling=cfg.TEMPORAL.SAMPLING,
            num_frames=cfg.TEMPORAL.NUM_FRAMES,
            allow_future=cfg.TEMPORAL.ALLOW_FUTURE,
        )
        frame_ids = [record["frames"][i]["frame_id"] for i in chosen]
        key = GeometryFeatureCache.make_canonical_key(
            checkpoint_sha256=checkpoint_hash,
            feature_mode=cfg.TEMPORAL.GEOMETRY.FEATURE_MODE,
            requested_dtype=cfg.TEMPORAL.GEOMETRY.DTYPE,
            pixel_budget=cfg.TEMPORAL.GEOMETRY.PIXEL_BUDGET,
            frame_ids=frame_ids,
            target_index=chosen.index(int(record["target_index"])),
        )
        if os.path.isfile(os.path.join(cache_dir, "test", key + ".npz")):
            hits += 1
    return hits, len(records)


def build_cfg(args: argparse.Namespace, checkpoint: str, records: List[Dict[str, Any]]):
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from pansr import add_pansr_config
    from pansr.temporal.config import add_temporal_config, validate_temporal_config

    model_dir = os.path.abspath(args.model_dir)
    config_file = args.config_file or os.path.join(model_dir, "config.yaml")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"config not found: {config_file}")
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_pansr_config(cfg)
    add_temporal_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = model_dir
    cfg.MODEL.WEIGHTS = checkpoint
    if args.device:
        cfg.MODEL.DEVICE = args.device
    metadata_name = register_inference_metadata()
    cfg.DATASETS.TRAIN = (metadata_name,)
    cfg.DATASETS.TEST = (metadata_name,)

    if cfg.MODEL.META_ARCHITECTURE == "TemporalPanSR":
        cfg.TEMPORAL.MANIFEST_VAL = os.path.abspath(args.manifest)
        if args.geometry_checkpoint:
            cfg.TEMPORAL.GEOMETRY.CHECKPOINT = os.path.abspath(args.geometry_checkpoint)
        if args.gmos_root:
            cfg.TEMPORAL.GEOMETRY.GMOS_ROOT = os.path.abspath(args.gmos_root)
        cache_dir = args.cache_dir or cfg.TEMPORAL.GEOMETRY.CACHE_DIR or "pi3_cache"
        cache_dir = os.path.abspath(cache_dir)
        hits, total = compatible_cache_count(cfg, records, cache_dir) if args.cache != "off" else (0, len(records))
        if args.cache == "require" and hits != total:
            raise RuntimeError(
                f"compatible Pi3 test cache incomplete: {hits}/{total} entries in {cache_dir}"
            )
        use_cache = args.cache != "off" and hits == total
        if use_cache:
            cfg.TEMPORAL.PREPROCESS_MODE = "canonical_full_frame_cache"
            cfg.TEMPORAL.GEOMETRY.CACHE_MODE = "clip_features"
            cfg.TEMPORAL.GEOMETRY.CACHE_DIR = cache_dir
            print(f"Pi3 cache: using {hits}/{total} compatible entries from {cache_dir}")
        else:
            cfg.TEMPORAL.PREPROCESS_MODE = "shared_view_online"
            cfg.TEMPORAL.GEOMETRY.CACHE_MODE = "none"
            cfg.TEMPORAL.GEOMETRY.CACHE_DIR = ""
            print(f"Pi3 cache: {hits}/{total} compatible entries; using online Pi3 inference")
        validate_temporal_config(cfg, is_train=False)
    cfg.freeze()
    return cfg


def encode_lars_png(panoptic_seg: torch.Tensor, segments_info: List[Dict[str, Any]]) -> np.ndarray:
    panoptic = panoptic_seg.detach().cpu().numpy()
    output = np.zeros((*panoptic.shape, 3), dtype=np.uint8)  # uncovered pixels are VOID
    next_instance = 1
    for segment in segments_info:
        contiguous_id = int(segment["category_id"])
        if contiguous_id not in CLASS_ID_MAP:
            raise ValueError(f"unknown contiguous category ID {contiguous_id}")
        class_id = CLASS_ID_MAP[contiguous_id]
        if bool(segment["isthing"]):
            if contiguous_id not in THING_CONTIGUOUS_IDS:
                raise ValueError(f"segment marked thing has stuff category {contiguous_id}")
            if next_instance > 65535:
                raise ValueError("prediction contains more than 65535 thing instances")
            instance_id = next_instance
            next_instance += 1
        else:
            instance_id = 0
        mask = panoptic == int(segment["id"])
        output[mask, 0] = class_id
        output[mask, 1] = instance_id // 256
        output[mask, 2] = instance_id % 256
    return output


def validate_png(path: str, expected_hw: Tuple[int, int]) -> None:
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (expected_hw[1], expected_hw[0]):
            raise ValueError(
                f"invalid output {path}: mode={image.mode}, size={image.size}, "
                f"expected RGB/{(expected_hw[1], expected_hw[0])}"
            )
        array = np.asarray(image)
    classes = set(np.unique(array[..., 0]).tolist())
    if not classes <= VALID_LARS_CLASSES:
        raise ValueError(f"invalid class IDs in {path}: {sorted(classes - VALID_LARS_CLASSES)}")
    instance_ids = array[..., 1].astype(np.uint16) * 256 + array[..., 2].astype(np.uint16)
    if np.any((np.isin(array[..., 0], list(STUFF_LARS_CLASSES))) & (instance_ids != 0)):
        raise ValueError(f"stuff/VOID pixels have nonzero instance IDs in {path}")
    if np.any((~np.isin(array[..., 0], list(STUFF_LARS_CLASSES))) & (instance_ids == 0)):
        raise ValueError(f"thing pixels have zero instance IDs in {path}")


def make_zip(png_paths: List[str], zip_path: str) -> None:
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    tmp = zip_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in png_paths:
            archive.write(path, arcname=os.path.basename(path))
    os.replace(tmp, zip_path)


def main() -> None:
    args = parse_args()
    args.model_dir = os.path.abspath(os.path.expanduser(args.model_dir))
    args.lars_root = os.path.abspath(os.path.expanduser(args.lars_root))
    args.manifest = os.path.abspath(os.path.expanduser(args.manifest))
    os.environ["LARS_ROOT"] = args.lars_root
    checkpoint = resolve_checkpoint(args.model_dir, args.checkpoint)
    records = load_jsonl(args.manifest)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        if not args.no_zip:
            raise ValueError("--limit creates an incomplete submission; also pass --no-zip")
        records = records[:args.limit]

    output_dir = os.path.abspath(args.output_dir or os.path.join(args.model_dir, "lars_test_predictions"))
    os.makedirs(output_dir, exist_ok=True)
    if not args.resume and any(Path(output_dir).iterdir()):
        raise RuntimeError(f"output directory is not empty (use --resume): {output_dir}")

    cfg = build_cfg(args, checkpoint, records)
    from detectron2.modeling import build_model
    from pansr.checkpoint_utils import load_initial_weights
    from pansr.temporal.dataset import build_temporal_dataset_dicts
    from pansr.temporal.mapper import TemporalCOCOPanopticDatasetMapper

    model = build_model(cfg)
    load_initial_weights(model, checkpoint)
    model.eval()
    mapper = TemporalCOCOPanopticDatasetMapper(cfg, False)
    dataset = build_temporal_dataset_dicts(
        args.manifest, args.lars_root,
        sampling=cfg.TEMPORAL.SAMPLING,
        num_frames=cfg.TEMPORAL.NUM_FRAMES,
        allow_future=cfg.TEMPORAL.ALLOW_FUTURE,
        seed=cfg.TEMPORAL.SAMPLING_SEED,
    )[:len(records)]

    outputs: List[str] = []
    inferred = 0
    started = time.perf_counter()
    for record, item in tqdm(zip(records, dataset), total=len(records), desc="LaRS test inference"):
        stem = os.path.splitext(os.path.basename(record["target_file"]))[0]
        output_path = os.path.join(output_dir, stem + ".png")
        expected_hw = (int(record["original_height"]), int(record["original_width"]))
        if args.resume and os.path.isfile(output_path):
            try:
                validate_png(output_path, expected_hw)
                outputs.append(output_path)
                continue
            except ValueError:
                pass
        mapped = mapper(item)
        with torch.inference_mode():
            prediction = model([mapped])[0]
        panoptic, segments = prediction["panoptic_seg"]
        rgb = encode_lars_png(panoptic, segments)
        Image.fromarray(rgb, mode="RGB").save(output_path, format="PNG")
        validate_png(output_path, expected_hw)
        outputs.append(output_path)
        inferred += 1

    elapsed = time.perf_counter() - started
    if len(outputs) != len(records) or len({os.path.basename(p) for p in outputs}) != len(records):
        raise RuntimeError("output count/name collision: refusing to create submission ZIP")
    zip_path = None
    if not args.no_zip:
        zip_path = os.path.abspath(args.zip_path or os.path.join(args.model_dir, "lars_test_submission.zip"))
        make_zip(outputs, zip_path)
    print(json.dumps({
        "checkpoint": checkpoint,
        "predictions": len(outputs),
        "newly_inferred": inferred,
        "output_dir": output_dir,
        "submission_zip": zip_path,
        "elapsed_seconds": round(elapsed, 3),
        "fps_new_predictions": round(inferred / elapsed, 3) if inferred else None,
    }, indent=2))


if __name__ == "__main__":
    main()

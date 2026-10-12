# Data Audit: LaRS

Dataset root: `/home/jon/disk2/datasets/LaRS` (`LARS_ROOT`).
Build tooling: `tools/temporal/build_manifest.py` (+
`tools/temporal/convert_annotations.py`, `tools/temporal/audit_environment.py`).
Build report: `output/manifests/build_report.json` (schema 1).

## 1. On-disk layout (verified file-by-file during manifest build)

```
$LARS_ROOT/
  train/
    images/<video>_<frame:05d>.jpg            annotated keyframes (1278x958)
    panoptic_masks/<video>_<frame:05d>.png    panoptic id masks (RGB-encoded)
    semantic_masks/                            (not used by this project)
    panoptic_annotations.json                  11 LaRS categories
    mmdet_annotations.json                     mmdet-ordered instance/panoptic mix
    image_annotations.json, image_list.txt
  val/  (same layout)
  test/ (images only, unlabeled — not used)
  lars_v1.0.0_images_seq/{train,val,test}/images_seq/<video>_<frame:05d>.jpg
      full sequence frames (the temporal context source)
  lars_v1.0.0_annotations.zip, lars_v1.0.0_images_seq.zip
```

**Clip contract.** An annotated target `<video>_<last>.jpg` owns the
clip `<video>_{last-9 .. last}.jpg` — the nine preceding sequence frames
+ the target keyframe — verified file by file (all 2605+198 targets,
0 missing frames, 0 gaps, 0 target-copy mismatches; the target in the
sequence directory is byte-identical to the annotated copy). The target
is always the **last** frame of its clip (causal context).

## 2. Manifests (built, hashed, integrity-tested)

| Split | Records | sha256 |
| --- | --- | --- |
| `output/manifests/train.jsonl` | 2605 targets | `07f484acdabba25ec68abe9bbc68da21fdc6dd4fcd805bb48582a7db1f3432d4` |
| `output/manifests/val.jsonl` | 198 targets | `2d9ddef690d28e30b38fe8aebe5b8e1cafa7995a5b1803e4f3203f17fe431a54` |

Record schema (v1): `split`, `image_id`, `sequence_id`,
`source_video_id`, `target_file` (annotated image, relative to
`LARS_ROOT`), and `frames` — 10 entries
`{file, frame_id, timestamp_seconds, relative_rank (-9..0), is_target}`
pointing at the sequence-directory frames. Cross-split check:
train∩val = 0 shared targets and 0 shared source videos. `tests/temporal/test_manifest.py::test_real_manifests_counts_and_integrity`
re-verifies counts + hashes + per-record file existence (fast hash, no
image decode) and stays green.

## 3. Labels

The LaRS panoptic JSON defines **11 categories** (original ids):

| Orig id | Name | isthing | Model id (contiguous) |
| --- | --- | --- | --- |
| 11 | Boat/ship | thing | 1 |
| 12 | Row boats | thing | 2 |
| 13 | Paddle board | thing | 3 |
| 14 | Buoy | thing | 4 |
| 15 | Swimmer | thing | 5 |
| 16 | Animal | thing | 6 |
| 17 | Float | thing | 7 |
| 19 | Other | thing | 8 |
| 1 | Static Obstacle | stuff | 9 |
| 3 | Water | stuff | 10 |
| 5 | Sky | stuff | 11 |
| — | VOID (background) | — | 0 |

`MODEL.NUM_CLASSES = 12` (0..11). The model's contiguous id → original
LaRS panoptic id map (used for evaluation ids and export) is the static
`metadata["category_id_map"]` in `pansr/data/datasets/register_lars.py`:
`{1:11, 2:12, 3:13, 4:14, 5:15, 6:16, 7:17, 8:19, 9:1, 10:3, 11:5, 12:0}`.
The temporal dataset registers the **same metadata** as the vanilla LaRS
registration, so vanilla and temporal models are evaluated with
identical ids (`gt_cls_weight` follows the standard panoptic
thing/stuff weights from `mmdet_annotations.json`).

## 4. Export encoding (`tools/temporal/export_lars.py`)

`panoptic_to_lars_rgb` (shared with `predict.py`) writes LaRS-style RGB
id masks: per segment, `R = original category id`, `G = instance id % 256`,
`B = instance id // 256`; stuff segments use `(cat, 0, 0)`; unsegmented
background is `(255, 0, 0)`. One PNG per target frame plus
`export_index.json` (input image, target file, clip frames, segment
summary).

## 5. Notes

- All images are 1278×958 (3:2-ish maritime frames); no per-image size
  variance within a split (verified in the annotations).
- `mmdet_annotations.json` is the training annotation file (mmdet
  category ordering — the ordering behind the contiguous ids above);
  `panoptic_annotations.json` carries the original LaRS ids and the
  official category list. `convert_annotations.py --check` verifies the
  two orderings round-trip through `category_id_map`.
- The `test/` split has no labels and is out of scope for training and
  evaluation (the LaRS benchmark protocol uses `val`).

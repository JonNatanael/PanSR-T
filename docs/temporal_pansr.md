# TG-PanSR: Temporal-Geometric PanSR on LaRS — Design and Usage

This document describes the implemented temporal-geometric extension of
PanSR on LaRS (per `Temporal_Geometric_PanSR_Implementation_Prompt.md`),
its data and coordinate contracts, the configuration surface, and the
runbook for training, evaluation, export and probing.

See also:
- `docs/repository_audit.md` — repository/env audit, deviations, fixes
- `docs/data_audit.md` — dataset layout, manifests, label mapping
- `docs/experiment_protocol.md` — reference protocol, controls, resources
- `docs/implementation_report.md` — what was executed and measured

## 1. Model

`TemporalPanSR` (registered meta-arch, `pansr/temporal/model.py`) =
unmodified PanSR + three additions:

```
target frame ──► Swin backbone (unchanged) ──► res2..res5 appearance pyramid
                                                     │
context clip ──► frozen Pi3 (joint T-frame encode)  │  [geometry_encoder.*]
               └─► target-frame features [1024, h, w]
                                                     ▼
                                    GMOS-style bottleneck fusion
                                    (temporal_fusion.*)
                                                     │
                                    residual pyramid injection
                                    (zero-init, per level res2..res5)
                                                     ▼
                                     PanSR head + losses (unchanged)
```

1. **Frozen geometric branch** (`pansr/temporal/geometry.py`): the GMOS
   Pi3 encoder (`gmos` checkout, `core/pi3_encoder.py`) runs a *joint*
   encoding of the whole selected clip (cross-frame attention over all
   frames, 2D-spatial RoPE only — **no temporal position**, so the
   encoder is permutation-invariant in frame order; see
   `tests/temporal/test_geometry.py`). Only the *target frame's* output
   patch grid is kept: `[1024, gh/14, gw/14]`. Pi3 is loaded from the
   Pi3 checkpoint, frozen (`requires_grad_(False)`), forced to `eval()`
   in every mode, and excluded from the optimizer. The encoder always
   runs in its verified precision (fp32 on sm75; bf16 where the SDPA
   backend supports it) inside an `autocast(enabled=False)` context so
   the frozen distribution is not silently downcast by surrounding AMP.
2. **Bottleneck fusion** (`pansr/temporal/fusion.py`): a GMOS-style
   bottleneck fuses the frozen geometry features with the trainable
   appearance features from `FUSION.APPEARANCE_SOURCE` (default res4):
   per-level adapters project appearance, a shared cross-attention
   bottleneck with normalized-2D-sincos position encodings mixes the
   two, a local-refine stack cleans up the geometry tokens, and a
   learned sigmoid gate (init 0.5) blends. `FUSION.MODE="off"` or
   `TEMPORAL.ENABLED=False` disables the branch entirely (exact vanilla
   path, verified bit-exact in `test_model_parity.py`).
   `FUSION.ZERO_GEOMETRY=True` (control B2) runs the same fusion with a
   zero geometry tensor and no Pi3 — bit-exact vanilla parity as well.
3. **Residual pyramid injection**: per-level residual adapters
   (`res2..res5`) inject a `FUSION.INJECT_DIM`-wide delta into each
   native Swin level. Their final 1×1 conv is **zero-initialized**, so
   at initialization the pyramid is *exactly* the vanilla Swin output
   (bit-exact, verified), and the head/losses are unchanged. The first
   gradients flow only through the zero-init output convs
   (`adapters.*.out.{weight,bias}`); after one optimizer step the rest
   of the fusion receives gradients (`test_fusion.py`).

**Backward compatibility**: with `TEMPORAL.ENABLED=False` (or the
vanilla configs) the forward path is byte-identical to the original
PanSR (`test_model_parity.py`: bit-exact eval parity, unchanged output
schema `{panoptic_seg, ocp_aux}`, shared-weight equality). The only
vanilla-path code change is an overridable
`PanSR.extract_backbone_features()` integration point
(`pansr/pansr_model.py`), which in the vanilla model is exactly
`self.backbone(images.tensor)`.

## 2. Data: clips, manifests, sampling

- **Manifests** (immutable JSONL, built by
  `tools/temporal/build_manifest.py`): one record per annotated target
  frame with its *verified* clip (9 consecutive preceding sequence
  frames + target, file-by-file hash checks; truncated history is
  recorded, never substituted). See `docs/data_audit.md` for counts,
  hashes and layout.
- **Sampling** (`pansr/temporal/manifest.py::select_frames`, causal):
  `all_history` (default, T=10), `recent` (k most recent context +
  target), `uniform_history`, `target_only` (single-frame control),
  `repeat_target` (explicit control). `ALLOW_FUTURE=False` by default;
  the target's clip position is always known
  (`target_index`, `TARGET_POLICY="explicit"`).
- **Mapper** (`pansr/temporal/mapper.py`): the *target* branch is the
  original PanSR LSJ mapper path (bit-identical, enforced by
  `test_mapper_parity.py`); the *same* sampled transform list is
  applied to every context frame, and the clip is resized
  (aspect-preserving, `cv2.INTER_LINEAR`) to the geometry raster —
  dims divisible by the Pi3 patch size (14) within
  `GEOMETRY.PIXEL_BUDGET` — and handed to the model as float RGB
  `[0, 1]`. The mapper emits, per sample:
  `image` (uint8 target raster), `instances` (train only),
  `clip_rgb` `[T,3,gh,gw]`, `target_index`, `frame_ids`, and
  `branch_transforms` (original/crop/geo sizes + the
  original→crop→geo affine chain, see §3). Copy-paste augmentation is
  disabled for the temporal model and its matched controls.

## 3. Coordinate bookkeeping (`pansr/temporal/alignment.py`)

All transforms are recorded as `PixelAffine` (pixel-center convention:
`x' = s(x + 0.5) − 0.5 + o`... implemented as `x' = s·x + o` with
`o = s·0.5 − 0.5` for resizes, i.e. dst pixel `y` samples src at
`(y+0.5)/s − 0.5`), composed in pipeline order
(`A.compose(B)` applies A then B, i.e. `B(A(x))`).

- `FeatureGrid` places cell centers at `(k+0.5)·stride` in the raster
  (true region centers) and converts them to grid-sample coordinates
  `n = 2p/(s·Wf) − 1` (feature pixel position `u = p/s`). With this
  convention, identity resampling, stride-aligned crops and flips of
  stride-divisible rasters are **exact** (checkerboard tests in
  `test_alignment.py`).
- The geometry raster (original→geo affine) is resampled onto each
  Swin level's feature grid; cells without geometry support are
  zeroed and a validity mask is passed to the fusion
  (`resample_features` / `resample_validity`).
- `test_alignment.py` cross-checks the analytic convention against
  cv2/PIL bilinear empirics and fvcore ground truth for
  crop/pad/flip. (fvcore `ResizeTransform.apply_coords` is *not* used
  as resize ground truth: it plain-scales without the half-pixel
  correction.)

The same conventions are what make the (optional)
`canonical_full_frame_cache` protocol coherent: offline features are
mapped into the augmented view through the stored transforms instead of
being stretched onto the crop (brief §10.2).

## 4. Training (`pansr/temporal/trainer.py`)

`TemporalTrainer` = PanSR `Trainer` +
- temporal train/test loaders (temporal mapper);
- explicit parameter groups: Pi3 excluded (frozen, asserted), new
  temporal modules at `TEMPORAL.OPTIM.NEW_MODULE_LR`, backbone at
  `BASE_LR·BACKBONE_MULTIPLIER`, everything else at `BASE_LR`;
  uniqueness asserted (every trainable parameter in exactly one group);
- deliberate gradient accumulation: one d2 iteration =
  `ACCUMULATION_STEPS` data batches (losses averaged); scheduler and
  checkpoints count optimizer updates;
- checkpoint persistence: the d2 0.6 checkpointer is extended with
  `optimizer`, `scheduler` and (AMP) `grad_scaler` checkpointables so
  save/resume restores the full training state. Old checkpoints (model
  only) still load — d2 only loads checkpointable keys that are
  present.
- `TEMPORAL.OPTIM.RUN_FINAL_EVAL` (default `True`): detectron2 0.6
  unconditionally runs a full-val evaluation after the last training
  iteration; on the 11 GB workstation that final evaluation is not
  runnable (original-resolution panoptic postprocess while optimizer
  state is live), so the workstation probe configs set it `False` and
  run evaluation separately (`export_lars.py` /
  `train_net.py --eval_only`).

Effective-batch accounting is asserted at construction:
`IMS_PER_BATCH × world × ACC == TEMPORAL.OPTIM.EXPECTED_EFFECTIVE_BATCH`.

## 5. Runbook

Environment (all paths overridable via env vars, see
`tests/temporal/conftest.py`):

```bash
export PANSR_ROOT=/media/jon/disk2/code/PanSR-T/PanSR
export GMOS_ROOT=/media/jon/disk2/code/PanSR-T/gmos
export LARS_ROOT=/home/jon/disk2/datasets/LaRS
export PI3_CKPT=/home/jon/disk2/models/pi3/model.safetensors
export PANSR_CKPT=/home/jon/disk2/models/pansr/pansr_lars_swin_l.safetensors
PY=/media/jon/disk2/code/PanSR/.conda/envs/pansr/bin/python
cd "$PANSR_ROOT"
```

Environment audit / manifest build / clip inspection:

```bash
$PY tools/temporal/audit_environment.py --gmos-root "$GMOS_ROOT" \
    --output output/temporal_environment.json \
    --pansr-checkpoint "$PANSR_CKPT" --pi3-checkpoint "$PI3_CKPT" \
    --lars-root "$LARS_ROOT" \
    --manifest-train output/manifests/train.jsonl \
    --manifest-val output/manifests/val.jsonl
$PY tools/temporal/build_manifest.py --lars-root "$LARS_ROOT" \
    --output-dir output/manifests --strict
$PY tools/temporal/inspect_clip.py --manifest output/manifests/train.jsonl \
    --index 0 --lars-root "$LARS_ROOT" --output-dir output/clip_inspection \
    --slicing recent --num-frames 8 --pixel-budget 28224
```

Smoke test (config validation via `--dry-run`; real run trains a few
updates on a clip subset and runs one val forward):

```bash
$PY tools/temporal/smoke_test.py \
    --config-file configs/lars/temporal/tg_pansr_swin_l_local11g.yaml \
    --pan-sr-checkpoint "$PANSR_CKPT" --geometry-checkpoint "$PI3_CKPT" \
    --train-manifest output/manifests/train.jsonl \
    --val-manifest output/manifests/val.jsonl \
    --clips 4 --steps 20 --output-dir output/temporal_smoke
```

Training (temporal) and the matched vanilla fine-tuning control:

```bash
# temporal (TG-PanSR)
CUDA_VISIBLE_DEVICES=1 $PY train_net.py --num-gpus 1 \
    --config-file configs/lars/temporal/tg_pansr_swin_l_local11g.yaml \
    MODEL.WEIGHTS "$PANSR_CKPT" \
    TEMPORAL.GEOMETRY.CHECKPOINT "$PI3_CKPT" \
    TEMPORAL.MANIFEST_TRAIN output/manifests/train.jsonl \
    TEMPORAL.MANIFEST_VAL output/manifests/val.jsonl

# matched vanilla control (same checkpoint, same schedule, copy-paste
# disabled for matching)
CUDA_VISIBLE_DEVICES=1 $PY train_net.py --num-gpus 1 \
    --config-file configs/lars/panoptic/pansr_Swin_L.yaml \
    MODEL.WEIGHTS "$PANSR_CKPT" \
    INPUT.COPY_PASTE.ENABLED False
```

Evaluation-only (official d2 panoptic evaluator over the val manifest):

```bash
CUDA_VISIBLE_DEVICES=1 $PY train_net.py --num-gpus 1 --eval_only \
    --config-file configs/lars/temporal/tg_pansr_swin_l_local11g.yaml \
    MODEL.WEIGHTS output/tg_pansr/.../model_final.pth
```

LaRS RGB export (one prediction per target):

```bash
CUDA_VISIBLE_DEVICES=1 $PY tools/temporal/export_lars.py \
    --config-file configs/lars/temporal/tg_pansr_swin_l_local11g.yaml \
    --weights output/tg_pansr/.../model_final.pth \
    --geometry-checkpoint "$PI3_CKPT" \
    --val-manifest output/manifests/val.jsonl \
    --lars-root "$LARS_ROOT" --output-dir output/lars_export --limit 198
```

Offline canonical geometry cache (§10.2 protocol; NOT equivalent to
online shared-view geometry — a separate documented mode):

```bash
CUDA_VISIBLE_DEVICES=1 $PY tools/temporal/extract_geometry.py \
    --manifest output/manifests/val.jsonl --lars-root "$LARS_ROOT" \
    --geometry-checkpoint "$PI3_CKPT" --gmos-root "$GMOS_ROOT" \
    --output-dir output/geometry_cache --index-from 0 --index-to 100 \
    --pixel-budget 65000 --slicing recent --num-frames 8
```

Profiling (synchronized GPU timing per branch + memory) and experiment
comparison:

```bash
CUDA_VISIBLE_DEVICES=1 $PY tools/temporal/profile_model.py \
    --config-file configs/lars/temporal/tg_pansr_swin_l_local11g.yaml \
    --weights "$PANSR_CKPT" --geometry-checkpoint "$PI3_CKPT" \
    --train-manifest output/manifests/train.jsonl \
    --lars-root "$LARS_ROOT" --clips 3 --iters 4 --output-dir output/profile
$PY tools/temporal/compare_experiments.py output/runA output/runB \
    --label A B --output comparison.json
```

Overfit probe (M3 gate: fixed real-clip subset, full-pass mean loss must
drop, `target_only` control included, fresh weights per run). The subset
is cycled in a deterministic fixed order by default (per-clip
convergence is measurable); `--random-order` mirrors the real
TrainingSampler:

```bash
CUDA_VISIBLE_DEVICES=1 $PY tools/temporal/overfit_probe.py \
    --config-file configs/lars/temporal/tg_pansr_swin_l_overfit_probe.yaml \
    --pan-sr-checkpoint "$PANSR_CKPT" --geometry-checkpoint "$PI3_CKPT" \
    --train-manifest output/manifests/train.jsonl \
    --lars-root "$LARS_ROOT" --clips 8 --iters 300 \
    --output-dir output/tg_pansr/overfit_probe
```

Tests:

```bash
$PY -m pytest tests/temporal -m "not gpu"                    # fast CPU suite
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -m pytest tests/temporal -m gpu                      # real-data GPU suite
```

## 6. Known workstation constraints (11 GB, sm75)

- Pi3 runs fp32 (no bf16 SDPA backend on Turing); the config's
  `DTYPE: bf16` falls back with a warning.
- The reference resolution (1024² input, 255000-pixel geometry) needs
  ≥~24 GB and bf16 hardware; the workstation configs
  (`tg_pansr_swin_l_local11g.yaml` / `..._overfit_probe.yaml`) validate
  the same code paths at 448² (T=8 `recent`) with `MODEL.SWIN.USE_CHECKPOINT=True`
  and reduced pixel budgets. They must not be used for reported
  reference results.
- The final full-val evaluation after training is disabled on the
  workstation (`RUN_FINAL_EVAL: False`) and run separately (it is
  too memory-hungry while optimizer state is live).
- AMP warm-down: at zero-init the first ~10–11 fp16 optimizer updates
  are skipped by the GradScaler (scale 65536 → ~64); this is expected
  and documented, not a failure.

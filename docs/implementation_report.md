# TG-PanSR — Implementation Report

Workstation validation run of the Temporal-Geometric PanSR (TG-PanSR) extension
per `Temporal_Geometric_PanSR_Implementation_Prompt.md`. Full reference-protocol
training is out of scope on this 11 GB workstation and is specified, not
executed, here. All GPU work used **GPU1 only** (single-job-at-a-time policy;
GPU0 is used by other workloads).

---

## 1. What was implemented

**Architecture** (all new code under `pansr/temporal/`, ~3.5k lines; 13 configs;
10 tools; 10 test files + conftest/small-model fixtures):

* `TemporalPanSR` meta-architecture (`model.py`): the unchanged PanSR head,
  criterion, postprocessing and evaluation, with the appearance backbone output
  augmented by a **frozen Pi3 geometric encoder** (757.9 M params, imported from
  the read-only GMOS reference, never trained) and a **GMOS-style bottleneck
  fusion** (5.0 M params, trainable). Fusion reads `res4`, refines through a
  3-layer transformer with normalized-2D-sincos positions and a learned gate,
  and injects **zero-initialized residual adapters into res2/res3/res4/res5**,
  so the model at initialization is numerically identical to vanilla PanSR
  (`test_model_parity` verifies this, including the `ZERO_GEOMETRY` control).
* The only change to the vanilla model file is one overridable seam:
  `PanSR.extract_backbone_features()` (`pansr/pansr_model.py`, +18 lines).
  Vanilla behavior is untouched; `train_net.py` selects `Trainer` vs
  `TemporalTrainer` by meta-architecture.
* `TemporalTrainer`: d2 0.6 trainer with explicit parameter groups (new-module
  LR 1e-4, non-backbone 1e-5, backbone 1e-6), gradient accumulation, AMP,
  DDP-safe unused-parameter handling, and a final-eval kill-switch
  (`TEMPORAL.OPTIM.RUN_FINAL_EVAL`).
* Data: immutable JSONL clip manifests (`build_manifest.py`, 2605 train / 198
  val), causal clip contract (target = LaRS annotated frame, `relative_rank 0`,
  no future frames), temporal mapper with the original LSJ pipeline on the
  target frame and a budget-divisible shared-view geometry raster, and an
  optional content-addressed feature cache (schema 1, atomic npz).
* Checkpoints: `pansr/checkpoint_utils.py` — validated `.safetensors`/`.pth`
  loading (d2 0.6's checkpointer only reads `.pth`), new-module prefixes
  allowed missing, plus the documented legacy-structure adaptation (shared
  `_bbox_embed` expanded into 108 per-layer alias keys; `decoder_norm`
  identity-init tolerated).
* Tools (`tools/temporal/`): `audit_environment`, `convert_annotations`,
  `build_manifest`, `inspect_clip`, `overfit_probe`, `smoke_test`,
  `profile_model`, `extract_geometry`, `export_lars` (+ `predict_vis.py`
  overlay visualizer, `compare_experiments`).

**Backward compatibility**: `TEMPORAL.ENABLED=False` (or a vanilla config)
runs the original code path; the published PanSR checkpoint loads into both
models; trained temporal checkpoints round-trip through the d2 checkpointer
(`model_final.pth`) with 0 missing / 0 unexpected / 0 mismatched keys
(verified by the smoke test and `test_save_resume_roundtrip`).

## 2. Deviations from the brief, with reasons

| # | Deviation | Reason |
|---|-----------|--------|
| 1 | `tg_pansr_swin_l_local11g.yaml`: `IMAGE_SIZE 448` (ref 1024), `PIXEL_BUDGET 28224` (ref 255000), `T=8 recent` (ref `T=10 all_history`) | Measured OOM matrix on 10.57 GiB usable: T=10 all_history at 448 OOMs in the mask point-sampler at every tested budget (28224/25000/20000/50000, ~+0.2 GiB over the limit); 448/T=8 at budget 50000 OOMs; 480²–768² OOM at any tested budget. 448/T=8/28224 is the measured largest fitting train config (peak 10.02 GiB over 300 iterations). |
| 2 | `MODEL.SWIN.USE_CHECKPOINT: True` in local11g (vanilla default False) | Activation checkpointing for the Swin blocks is required for the train step to fit at 448 with AdamW state live. |
| 3 | `TEMPORAL.OPTIM.RUN_FINAL_EVAL: False` in local11g | d2 0.6 runs the final eval unconditionally at original resolution; the single postprocess allocation (2.317 GiB) plus live optimizer OOMs here. Eval is run via `export_lars.py` / B0 eval instead. |
| 4 | Pi3 runs **fp32** on this workstation (config says bf16) | No bf16 SDPA backend on sm75 (RTX 2080 Ti); the encoder logs and falls back to fp32. The reference config is unchanged and will use bf16 on sm80+ hardware. |
| 5 | Overfit probe gate = **5-pass windowed means**, ≥2 % relative drop, fixed-order cycling default | Loss is stochastic per-iteration even at fixed weights (decoder/fusion dropout + DN noise); single-pass comparisons flipped sign across equivalent runs. Windowed pass means (first 5 vs last 5 full passes) are stable. Fixed order makes the probe deterministic per clip. Both controls (with/without context) are required to pass. |
| 6 | d2 0.6 adaptations: `EventStorage.latest()` returns `(value, iter)` tuples; `.pth`-only checkpointer resume | Measured d2 0.6 behavior; loss-capture hooks and the trainer handle both shapes / both formats. |
| 7 | `train_net.py` refactored (178 → 41 lines): trainer extraction to `pansr/trainer.py`, `--eval_only`, safetensors weights, trainer selection | Required for the temporal entry points; behavior preserved (the vanilla eval path was executed as the B0 job below). |
| 8 | OpenCV-5 fix in `extract_geometry.py` (`np.frombuffer` before `imdecode`) | Installed cv2 5.0.0 rejects raw `bytes` (4.x accepted them); environment drift. |
| 9 | Copy-paste disabled in all temporal configs | Per brief §5.1 (matched fine-tuning controls). |

No scientific quantities (losses, metrics, sampling, schedule) were changed to
make anything fit; every deviation above is a workstation resource constraint
or a measured library-behavior adaptation, and the reference config
(`tg_pansr_swin_l.yaml`) keeps the brief's values (1024², 255000, T=10
all_history, bf16).

## 3. Environment, provenance, and protocol parameters

From `output/temporal_environment.json` (complete, hash-verified):

* Python 3.10.20; torch 2.4.1+cu124 (CUDA 12.4); detectron2 0.6; timm 1.0.28;
  numpy 2.2.6; cv2 5.0.0; safetensors 0.8.0.
* GPUs: 2× NVIDIA RTX 2080 Ti (11.35 GB, sm75); **bf16 flash-SDPA: unavailable**
  (Pi3 fp32 fallback locally). All GPU jobs: GPU1,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
* PanSR working copy: repo commit `148183ff319db15d6c2e14a80642b9d58c7e85f9`
  plus this project's own files (dirty, intentional — see
  `docs/repository_audit.md`). GMOS reference (read-only):
  `99d314acec331bfa3a023042d997a32df37335ec`.
* Checkpoints:
  * PanSR LaRS Swin-L: `/home/jon/disk2/models/pansr/pansr_lars_swin_l.safetensors`,
    sha256 `e9fecf6fa5a5bf6762521c6e4abfd8ba8145c587530c3596a6c81a298f566e0e`
  * Pi3: `/home/jon/disk2/models/pi3/model.safetensors`,
    sha256 `33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d`
* LaRS: `/home/jon/disk2/datasets/LaRS`; manifests
  `output/manifests/train.jsonl` (2605 records,
  sha256 `07f484acdabba25ec68abe9bbc68da21fdc6dd4fcd805bb48582a7db1f3432d4`)
  and `val.jsonl` (198,
  sha256 `2d9ddef690d28e30b38fe8aebe5b8e1cafa7995a5b1803e4f3203f17fe431a54`).
  Label mapping and encoding details: `docs/data_audit.md`.

**Target-selection policy**: explicit target (the LaRS annotated frame; always
the last frame of a causal clip, `relative_rank 0`); `ALLOW_FUTURE=False`;
`MISSING_POLICY=error` for training runs (`fallback_baseline` exists for
DDP robustness). Sampling: `all_history` T=10 (reference), `recent` T=8
(local11g). Preprocessing: `shared_view_online` (geometry raster = the target
view's LSJ crop resized aspect-preserving to the budget-divisible raster,
INTER_LINEAR, RGB in [0,1]); `extract_geometry.py` implements the separate
documented canonical full-frame protocol for the file cache. Cache:
`none` by default; file cache is content-addressed (Pi3 checkpoint sha +
feature mode + dtype + budget + frame ids + per-frame fingerprints), schema 1,
atomic writes.

**Trainable / frozen (measured, this session)**: total 1012.4 M =
Pi3 frozen **757.9 M** (requires_grad=False) + Swin backbone 195.2 M +
head/decoder 54.2 M + new fusion **5.0 M**; trainable **254.4 M**.

**Effective batch / schedule (reference config)**: micro-batch 1 ×
`ACCUMULATION_STEPS 8` = effective batch 8; `MAX_ITER 10000` optimizer updates;
LR warmup 500 iters (×0.1) then MultiStep (8000, 9000); LRs: new modules 1e-4,
other non-backbone 1e-5, backbone 1e-6; AMP fp16 enabled; full-model gradient
clipping.

## 4. Tests, commands, artifacts

Environment for every GPU command:
`LARS_ROOT=/home/jon/disk2/datasets/LaRS GMOS_ROOT=/media/jon/disk2/code/PanSR-T/gmos
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
python `/media/jon/disk2/code/PanSR/.conda/envs/pansr/bin/python`, cwd
`/media/jon/disk2/code/PanSR-T/PanSR`.

### 4.1 Unit/integration test matrix

| Suite | Result | Command |
|-------|--------|---------|
| CPU (`-m "not gpu"`) | **81 passed**, 13 deselected (GPU-marked) | `python -m pytest tests/temporal -m "not gpu" -q --basetemp=/media/jon/disk2/tmp/pytest-basemp/runNN` |
| Geometry (GPU) | **7/7 passed** | `python -m pytest tests/temporal/test_geometry.py -m gpu --basetemp=...` |
| End-to-end integration (GPU) | **6/6 passed** (~3 min) | `python -m pytest tests/temporal/test_integration_gpu.py -m gpu --basetemp=...` |
| **Full GPU matrix** | **13/13 passed** | `python -m pytest tests/temporal -m gpu --basetemp=/media/jon/disk2/tmp/pytest-basemp/runNN` |

Deselected/skipped: the 13 GPU tests are excluded from CPU runs by marker
(single-GPU serialization policy on this box; each trained test writes a
5.7 GiB `model_final.pth`, so pytest temp must live on the large disk — root
`/tmp` fills up). The 10 reference-protocol/§11.2 configurations are not
tests and were not run at scale (see §6).

Integration suite contents: clear failure on missing checkpoint, train-batch
contract (shapes/dtypes/causality), val loader with one prediction per target,
forward+backward learning (loss drops, gradients flow to fusion only),
save/resume round-trip, and vanilla-trainer checkpointable parity.

### 4.2 M3 overfit probe (official run)

```
python tools/temporal/overfit_probe.py \
  --config-file configs/lars/temporal/tg_pansr_swin_l_overfit_probe.yaml \
  --pan-sr-checkpoint /home/jon/disk2/models/pansr/pansr_lars_swin_l.safetensors \
  --geometry-checkpoint /home/jon/disk2/models/pi3/model.safetensors \
  --train-manifest output/manifests/train.jsonl \
  --clips 8 --iters 300 --output-dir output/tg_pansr/overfit_probe
```

**Status ok** (`output/tg_pansr/overfit_probe/overfit_summary.json`). 8 clips,
300 optimizer updates each, batch 1, fixed order; gate = 5-pass windowed
means, ≥2 % drop:

| Run | Pass-mean first 5 → last 5 | Ratio | Zero-init verified |
|-----|---------------------------|-------|--------------------|
| `recent` T=8 (temporal) | 25.8174 → 20.7667 | **0.8044** | yes |
| `target_only` control | 26.4241 → 21.3267 | **0.8071** | yes |

Zero-initialization of the fusion verified before training; adapters updated
after. Peak 10.02 GiB. Per the brief, a 20 % probe-loss reduction on 8 clips
is an overfit/sanity signal only — it is *not* evidence of improved PQ.

### 4.3 Smoke test (local11g config, 4 clips × 20 steps)

```
python tools/temporal/smoke_test.py \
  --config-file configs/lars/temporal/tg_pansr_swin_l_local11g.yaml \
  --pan-sr-checkpoint /home/jon/disk2/models/pansr/pansr_lars_swin_l.safetensors \
  --geometry-checkpoint /home/jon/disk2/models/pi3/model.safetensors \
  --train-manifest output/manifests/train.jsonl --val-manifest output/manifests/val.jsonl \
  --clips 4 --steps 20 --output-dir output/temporal_smoke
```

**Status ok** (`output/temporal_smoke/smoke_report.json`): losses finite over
all 20 steps; adapters updated; initial-weights load 0 mismatched / 0
unexpected; after freeing the trainer stack, a freshly built model reloaded
the checkpointer's `model_final.pth` (round-trip 0 missing / 0 unexpected /
0 mismatched) and completed a full val forward
(`panoptic_shape [720, 1280]`, 11 segments). Peak 9.70 GiB, 187.7 s.
Log: `/tmp/smoke_test4.log`.

### 4.4 Profile (local11g, 3 clips, 1 warmup + 3 eval forwards, 3 train steps)

```
python tools/temporal/profile_model.py \
  --config-file configs/lars/temporal/tg_pansr_swin_l_local11g.yaml \
  --weights /home/jon/disk2/models/pansr/pansr_lars_swin_l.safetensors \
  --geometry-checkpoint /home/jon/disk2/models/pi3/model.safetensors \
  --train-manifest output/manifests/train.jsonl \
  --lars-root /home/jon/disk2/datasets/LaRS --clips 3 --iters 4 \
  --output-dir output/profile
```

**Status ok** (`output/profile/profile_report.json`). Train steps mirror d2 0.6
`AMPTrainer.run_step` (forward under autocast, backward/optimizer outside).

Warmup forwards are excluded from the statistics (timers collect only after
`--warmup`).

| (ms) | eval (n=2) | train (n=3) |
|------|-----------|-------------|
| full eval forward | 381.1 ± 9.5 | — |
| Swin backbone | 47.8 | 51.6 (fp16) |
| Pi3 joint encoding (8 frames) | 229.5 | 227.7 |
| alignment + fusion | 6.4 | 29.9 |
| PanSR head | 83.2 | 132.0 |
| full train step (fwd+bwd+step) | — | 1128.8 ± 312.3 |

Throughput 2.62 clips/s (eval, 448²). Peak 7.17 GiB allocated / 7.49
reserved. Per brief §10.3 the Pi3 figure is the **joint** encoding of all 8
selected frames, not a per-frame cost. Swin recompute (USE_CHECKPOINT) is
inside backward, i.e. only visible in the train-step number. Log:
`/tmp/profile_run5.log`.

### 4.5 Geometry cache extraction (real 2-clip run)

```
python tools/temporal/extract_geometry.py \
  --manifest output/manifests/train.jsonl --lars-root /home/jon/disk2/datasets/LaRS \
  --geometry-checkpoint /home/jon/disk2/models/pi3/model.safetensors \
  --gmos-root /media/jon/disk2/code/PanSR-T/gmos --output-dir output/geometry_cache \
  --index-from 0 --index-to 2 --slicing recent --num-frames 8 --pixel-budget 28224
```

**2/2 clips extracted** (28.6 s), atomic schema-1 npz under
`output/geometry_cache/train/` (content-addressed names); entry:
`features` float32 `[1024, 10, 14]` (target frame at the 140×196 canonical
raster), `schema=1`, JSON metadata (clip id, geo_hw, frame ids, budget,
fingerprint). Log: `/tmp/extract_geom3.log`.

### 4.6 LaRS export (overfit-probe `recent` weights, 8 val targets)

```
python tools/temporal/export_lars.py \
  --config-file configs/lars/temporal/tg_pansr_swin_l_local11g.yaml \
  --weights output/tg_pansr/overfit_probe/recent/model_final.pth \
  --geometry-checkpoint /home/jon/disk2/models/pi3/model.safetensors \
  --val-manifest output/manifests/val.jsonl \
  --lars-root /home/jon/disk2/datasets/LaRS --output-dir output/lars_export --limit 8
```

**8/8 targets exported** (30.6 s) to `output/lars_export/panoptic_predictions/`
(+ `export_index.json`). PNGs are 720×1280×3 uint8 original-resolution masks;
spot-checked encoding matches the original `inference.py` convention: stuff
`(cat, 0, 0)`, things `(cat, id%256, id//256)` (observed `(13, 4, 0)`),
unfilled background `(255, 0, 0)`. Log: `/tmp/export_lars2.log`.

### 4.7 B0 baseline eval (vanilla PanSR, published weights)

```
python train_net.py --num-gpus 1 --eval_only \
  --config-file configs/lars/panoptic/pansr_Swin_L.yaml \
  MODEL.WEIGHTS /home/jon/disk2/models/pansr/pansr_lars_swin_l.safetensors \
  INPUT.MIN_SIZE_TEST 768 INPUT.MAX_SIZE_TEST 1024 \
  OUTPUT_DIR output/b0_eval
```

Full val set (198 targets) at a reduced test resolution (see §5/§6). Log:
`/tmp/b0_eval2.log`; results in `output/b0_eval/`.

## 5. Measured results — strictly separated

**Published reference (NOT reproduced here)**: PanSR LaRS Swin-L **PQ 57.3**
(PanSR README/paper [S2][S16], evaluated at the original 1024 short-edge
protocol). It is quoted only as the reference baseline for the planned
experiments.

**Locally measured on this workstation** (RTX 2080 Ti 11 GB, sm75, Pi3 fp32,
448²/T=8/28224/USE_CHECKPOINT unless stated):

* Train-step peak memory **10.02 GiB** (300 iters, overfit probe); smoke
  (train 20 steps + val forward on a bare model) peak **9.70 GiB**; profile
  peak **7.17 GiB** allocated. Single val forward (bare model) peak
  **8.31 GiB** (integration suite measurement).
* OOM matrix at 448²: T=10 all_history OOMs at budgets 28224/25000/20000/
  50000 (~+0.2 GiB over the limit, in the mask point-sampler); T=8 at
  budget 50000 OOMs; 480²–768² OOM at any tested budget. Note the LSJ
  transform (`ResizeScale(0.5, 2.0, IS)` + `FixedSizeCrop(IS)`) makes the
  Swin input vary per iteration in [0.5·IS, IS], so fits must be measured
  over many steps, not one.
* Timing: eval forward 381 ms/clip (2.62 clips/s); full train step
  1129 ms/update-equivalent (batch 1); Pi3 joint encoding 230 ms per 8-frame
  clip (fp32, sm75 — the reference bf16 hardware will differ).
* Overfit probe pass-mean ratios 0.8044 (recent) / 0.8071 (target_only
  control) — sanity signal only, not PQ.
* B0 local eval (vanilla PanSR, published weights), full 198-target val set
  at 768 short-edge / 1024 max: **PQ 52.759** (SQ 76.444, RQ 60.730; Things
  37.001, Stuff 94.779; 11 categories). Completed in ~75 s inference without
  OOM. This is a local measurement at reduced resolution and is **not
  comparable** to the published 57.3 (evaluated at the original 1024/2048
  short-edge protocol).

**Unrun** (require the training workstation): the reference-protocol training
(1024², budget 255000, T=10 all_history, bf16, 10000 updates), the full
§11.2 experiment matrix at scale, full-resolution full-val-set evaluation,
and any reproduction of the published PQ. None of these are claimed to have
passed; they are specified in `docs/experiment_protocol.md`.

## 6. Limitations and next experiment

**Limitations**

1. 11 GB VRAM confines local runs to the memory-reduced local11g config; all
   reported timings/memory are for 448²/T=8, not the reference 1024²/T=10.
2. Pi3 fp32 on sm75: no bf16 SDPA backend, so geometry timing/memory here
   overstate the reference bf16 hardware.
3. B0 local eval ran at 768/1024 (1024/2048 OOMs on this card); its PQ is a
   local measurement only.
4. Probe and smoke results are wiring/sanity evidence. Per the brief, no
   shape or overfit result is evidence of improved panoptic accuracy.

**Single prioritized next experiment** (training workstation, ≥24 GB, bf16):
run the matched pair **B1 vs G1** from `docs/experiment_protocol.md` at the
reference protocol (1024², PIXEL_BUDGET 255000, T=10 all_history, bf16,
10000 updates, identical seed/schedule/eval, copy-paste off in both):
**B1** = vanilla PanSR continued from the published checkpoint; **G1** =
TemporalPanSR with the frozen Pi3 + fusion from the same checkpoint. Compare
PQ on the full val set (original-resolution protocol). B1 isolates
"continued training under the matched budget"; G1 − B1 is the first evidence
for (or against) the temporal-geometric gain. GR/G10 and the T/budget
ablations follow only if G1 is not clearly worse.

---

## Appendix: artifact index

| Artifact | Path |
|----------|------|
| Environment manifest (hashes) | `output/temporal_environment.json` |
| Manifests (train/val) | `output/manifests/{train,val}.jsonl` |
| Overfit probe | `output/tg_pansr/overfit_probe/{recent,target_only,overfit_summary.json}` |
| Smoke test | `output/temporal_smoke/smoke_report.json` (+ `run/model_final.pth`) |
| Profile | `output/profile/profile_report.json` |
| Geometry cache (2 clips) | `output/geometry_cache/train/*.npz`, `extraction_report.json` |
| LaRS export (8 targets) | `output/lars_export/` |
| B0 eval | `output/b0_eval/` |
| Docs | `docs/{repository_audit,data_audit,experiment_protocol,temporal_pansr,implementation_report}.md` |
| Logs (ephemeral, /tmp) | `smoke_test4.log`, `profile_run5.log`, `extract_geom3.log`, `export_lars2.log`, `b0_eval2.log`, `integ_gpu_run13.log`, `overfit_probe5.log` |

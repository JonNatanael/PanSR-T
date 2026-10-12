# Experiment Protocol (TG-PanSR on LaRS)

Config surface: `configs/lars/temporal/`. All temporal configs derive
from `Base-TG.yaml` (itself derived from the upstream
`panoptic/pansr_Swin_L.yaml`), so matched runs share data, schedule and
evaluation and differ only in the flagged fields.

## 1. Reference fine-tuning protocol (workstation-scale: `tg_pansr_swin_l.yaml`)

| Parameter | Reference value | Config key |
| --- | --- | --- |
| Starting weights | shared PanSR Swin-L checkpoint (hash in `output/temporal_environment.json`) | `MODEL.WEIGHTS` |
| Context | T = 10 (9 verified preceding frames + target, target last), causal (`ALLOW_FUTURE=False`) | `TEMPORAL.NUM_FRAMES / SAMPLING="all_history"` |
| Geometry | online shared-view Pi3, frozen, last-decoder features (1024-w), pixel budget 255 000 (≈ 506×503 raster) | `TEMPORAL.GEOMETRY.*` |
| Fusion | GMOS-style bottleneck (256-d, 3 refine layers, 8 heads, 2048 FFN, dropout 0.1) + residual injection at res2..res5 (64-d, zero-initialized output, learned gate) | `TEMPORAL.FUSION.*` |
| Copy-paste | **disabled** for the temporal model and all matched controls | `INPUT.COPY_PASTE.ENABLED=False` |
| Effective batch | 8 annotated target clips = 1 micro × 1 world × **8 accumulation** (loss-averaged; optimizer/scheduler/checkpoints count optimizer updates) | `SOLVER.IMS_PER_BATCH=1`, `TEMPORAL.OPTIM.ACCUMULATION_STEPS=8`, `EXPECTED_EFFECTIVE_BATCH=8` (asserted at trainer build) |
| Optimizer | AdamW, full-model gradient clipping as resolved by the base config | — |
| Learning rates | BASE_LR 1e-5 (existing non-backbone PanSR modules), backbone ×0.1 (Swin 1e-6), new temporal modules 1e-4, Pi3 frozen/excluded | `SOLVER.BASE_LR/BACKBONE_MULTIPLIER`, `TEMPORAL.OPTIM.NEW_MODULE_LR` |
| Updates / schedule | 10 000 optimizer updates; WarmupMultiStepLR: 500 warmup from 0.1, milestones (8000, 9000) — explicitly re-set for the 10 k run (the upstream 81 k/85.5 k milestones are not inherited) | `SOLVER.MAX_ITER/STEPS/WARMUP_*` |
| Checkpoint / eval | every 1 000 updates; val eval every 1 000 | `SOLVER.CHECKPOINT_PERIOD`, `TEST.EVAL_PERIOD` |
| AMP | enabled (fp16) | `SOLVER.AMP.ENABLED=True` |
| Weight init | new modules random, residual output convs zero-initialized (verified in tests); reset to the common init before every reported run | `TEMPORAL.FUSION.RESIDUAL_INIT="zero_output"` |

Parameter groups are built by explicit module ownership
(`pansr/temporal/trainer.py::build_temporal_parameter_groups`): every
trainable parameter in exactly one group, frozen Pi3 asserted excluded,
norm/bias/positional weight-decay exceptions preserved from the
original trainer.

### Workstation deviations (documented, not silent)

This 11 GB box cannot run the reference resolutions; the local configs
deviate **only** where hardware requires it:

- `tg_pansr_swin_l_local11g.yaml`: 448² input, 28 224-pixel geometry
  budget (168² raster, 12×12 tokens/frame), **T=8 `recent`** (instead
  of T=10 `all_history`) — the *measured* largest train-step
  configuration that fits 10.57 GiB (peak 10.02 GiB over 300
  iterations incl. the worst-case LSJ crop; every T=10 variant at
  448² and every larger image size 480²–768² OOMs, see the config
  header). `MODEL.SWIN.USE_CHECKPOINT=True` (activation checkpointing;
  the reference keeps it `False`), `TEMPORAL.OPTIM.RUN_FINAL_EVAL=False`
  (d2 0.6's unconditional final-eval OOMs here). The T=8 reduction is
  a memory deviation from the reference context length, not a change
  of the reference protocol.
  (d2 0.6's unconditional final-eval OOMs here; see
  `docs/repository_audit.md` §4). The probe config
  (`tg_pansr_swin_l_overfit_probe.yaml`) further reduces to 448²,
  28 224-pixel budget, T=8 `recent` sampling, no accumulation,
  BASE_LR 1e-4 for fast overfitting.
- Pi3 dtype: `bf16` where the SDPA backend supports it; sm75 falls back
  to fp32 (logged; identical frozen-encoder path).

These deviations do **not** change the protocol semantics (T, schedule,
batch accounting, controls); they change only per-step resource usage.

## 2. Comparison matrix (§11.2) — executable configs

| ID | Config | Input / change |
| --- | --- | --- |
| B0 | (no training) `pansr_Swin_L.yaml` + published checkpoint, `--eval_only` | the actual local starting point |
| B1 | `baseline_continued.yaml` | same PanSR, exact matched fine-tune, `TEMPORAL.ENABLED=False` (vanilla mapper, no geometry) |
| B2 | `appearance_adapter_control.yaml` | same trainable fusion architecture + schedule, geometry = constant zero tensor (no Pi3 call) — a **capacity** control, not a Pi3 parameter/compute match |
| G1 | `geo_target_only.yaml` | true T=1 Pi3 call on the target only |
| GR | `geo_repeat_target.yaml` | target repeated to T=10 (same clip length/compute, no new views) |
| G3 | `geo_t3.yaml` | 3 genuine causal frames |
| G5 | `geo_t5.yaml` | 5 genuine causal frames |
| G10 | `tg_pansr_swin_l.yaml` (= `geo_t10.yaml`) | main proposed model, all verified frames |
| GF | `geo_concat_only.yaml` | G10 fusion without spatial-transformer refinement |
| GS | `geo_no_res2.yaml` | G10 without res2 injection |

G1 / GR / G10 must be compared under the **same** preprocessing/cache
protocol and schedule (they share `Base-TG.yaml` by construction).
A G10-vs-B0 gap alone is not evidence for the temporal hypothesis —
B1 (matched fine-tune) and GR (matched compute, no new views) are the
decisive controls.

## 3. Context-use diagnostics (§11.3)

Implemented as perturbations in `tests/temporal/test_geometry.py` and
available via `tools/temporal/inspect_clip.py` / `profile_model.py`
(`--slicing repeat_target`, `target_only`, `recent` subsets):
repeated-target context, context-content change (target features must
change with context content), frame-order permutation (Pi3's
permutation invariance is a measured property, not a failure signal),
and residual-delta inspection before/after training
(`compare_experiments.py`). A model that ignores context must be
reported as a negative result.

## 4. Evaluation protocol (§11.1)

- Official LaRS train/val split; evaluate only annotated val targets
  (198 clips).
- Export: `panoptic_to_lars_rgb` — R = original category id,
  G/B = instance id (stuff `(cat,0,0)`, background `(255,0,0)`);
  byte-order verified against the official evaluator conventions
  (`tests/temporal/test_export.py`, incl. instance ids > 255).
- Primary metric: official panoptic PQ (overall, SQ/RQ, thing/stuff,
  per-class). The published LaRS-PanSR PQ of **57.3 is a reference
  number only** (from the upstream paper/checkpoint); any locally
  computed value must be labeled as a local measurement and never
  conflated with it.
- `COCOPanopticEvaluator` runs end-to-end through the temporal mapper
  (requires `file_name` in the batch — fixed, see audit doc).

## 5. Resource measurement plan (§10.3)

`tools/temporal/profile_model.py` measures, with synchronized CUDA
timing and documented warmup, per component: backbone, Pi3 joint
encoding (image encode + cross-frame decode as one joint cost — do not
divide by T), alignment/fusion, head, train step; plus peak
allocated/reserved memory and throughput. Pi3 joint encoding is the
dominant geometric cost; only the target-frame feature grid is
retained, but the full clip is encoded. Parameter counts are reported
as total vs trainable (the frozen Pi3 must not be hidden from the
resource comparison).

## 6. What was executed on this workstation vs elsewhere

Runnable here (11 GB): B0 small-val eval, overfit probe (M3 gate),
smoke test, profile at probe resolution, geometry-cache writes at
reduced budget, export of a few targets, full test suite.
Executed (workstation validation): B0 local eval at 768 short-edge /
1024 max on the full 198-target val set → **PQ 52.759** (All; local
measurement at reduced resolution, *not* the published 57.3) — see
`docs/implementation_report.md` §5.
**Not runnable here** (to be run on the training workstation): the 10 k-
update reference protocol and the full §11.2 matrix at 1024² /
255 k-pixel budget. All such results are explicitly marked "unrun" in
`docs/implementation_report.md`.

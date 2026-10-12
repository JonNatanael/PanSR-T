# Repository & Environment Audit

Audited against the working copy at commit `148183ff` (PanSR, "Remove old
references") and the GMOS checkout at `99d314ac`. Full machine state is
captured in `output/temporal_environment.json` (written by
`tools/temporal/audit_environment.py`).

## 1. Environment

| Item | Value |
| --- | --- |
| Python | 3.10.20 (`/media/jon/disk2/code/PanSR/.conda/envs/pansr`) |
| PyTorch | 2.4.1+cu124 (CUDA 12.4) |
| detectron2 | 0.6 (source install at `/media/jon/disk2/code/PanSR/detectron2`, pinned to repo commit `148183ff`) |
| GPU | 1× NVIDIA GeForce RTX 2080 Ti, 11.35 GB, sm_75 (no bf16, no flash-SDPA for bf16) |
| Key libs | timm 1.0.28, numpy 2.2.6, cv2 5.0.0, safetensors 0.8.0, panopticapi present |
| Deformable attention | compiled op verified working (`tools/temporal/audit_environment.py`) |

Workstation constraint: **10.57 GiB usable VRAM** — the full reference
protocol (1024², 255k-pixel budget) does **not** fit here; all local GPU
runs use the probe / `local11g` configs (both 448², reduced geometry).
All GPU work in this project used **GPU 1 only**
(`CUDA_VISIBLE_DEVICES=1`) and never ran two GPU jobs concurrently.

## 2. Working-copy state

The working copy is dirty **by design**: the project adds new modules and
touches exactly two upstream files, minimally.

### Modified upstream files (2)

| File | Change | Why |
| --- | --- | --- |
| `pansr/pansr_model.py` | +18 lines: new `PanSR.extract_backbone_features(images, batched_inputs)` method; `forward` calls it instead of `self.backbone(images.tensor)` | The **single overridable integration point** for the temporal branch. `TemporalPanSR` overrides it (geometry features + residual pyramid adaptation); the vanilla model runs the original statement, so vanilla behavior is bit-exact (verified in `tests/temporal/test_model_parity.py`). |
| `train_net.py` | −155/+25: the `Trainer(DefaultTrainer)` class moved verbatim into `pansr/trainer.py`; entry point now picks `TemporalTrainer` vs `Trainer` by `MODEL.META_ARCHITECTURE` and routes `--eval_only` / `MODEL.WEIGHTS` through `pansr.checkpoint_utils.load_model_weights` so the distributed `.safetensors` PanSR checkpoint works | Temporal training needs the same trainer base plus temporal loaders/parameter groups; the extraction is the only way to keep one implementation. `train_net.py` behavior for vanilla configs is unchanged. |

### New project files (untracked)

- `pansr/trainer.py` — extracted `Trainer` (unchanged semantics) plus
  full training-state checkpointables: `optimizer`, `scheduler`,
  `grad_scaler` are added to the `DetectionCheckpointer`, so
  `model_final.pth`/resume restore the complete state (d2 0.6's
  checkpointer only persists what is registered; pre-change checkpoints
  still load — missing keys are simply skipped).
- `pansr/checkpoint_utils.py` — validated `.pth`/`.safetensors` loading
  (see §3).
- `pansr/temporal/` — the temporal package (16 modules: `alignment`,
  `build`, `cache`, `checkpoint`, `config`, `dataset`, `fusion`,
  `geometry`, `manifest`, `mapper`, `model`, `predictor`, `trainer`,
  …) — see `docs/temporal_pansr.md`.
- `configs/lars/temporal/` — 13 configs (reference, workstation, probe,
  and the §11.2 comparison matrix).
- `tools/temporal/` — 9 tools (`audit_environment`, `build_manifest`,
  `convert_annotations`, `inspect_clip`, `smoke_test`, `extract_geometry`,
  `export_lars`, `profile_model`, `compare_experiments`,
  `overfit_probe`); each supports `--dry-run` (CPU, no model build).
- `tests/temporal/` — 12 test modules + `conftest.py`; 81 CPU tests,
  13 GPU tests (7 geometry + 6 integration).
- `pytest.ini`, `docs/` — markers (`gpu`, `slow`, `checkpoint`) and
  documentation.

### Pre-existing untracked entries (NOT part of this project)

`.cache/`, `.conda/`, `.tmp_lars/`, `ZED_predictions_bled2_2/`,
`ZED_predictions_bled2_9/`, `outputs/`, `predict_vis.py` — untouched.
(`predict.py` had a temporary debug print added during debugging which
was removed; it is net-identical to upstream, `git diff predict.py` is
empty.)

## 3. Legacy checkpoint adaptation (`pansr/checkpoint_utils.py`)

The distributed LaRS checkpoint
(`/home/jon/disk2/models/pansr/pansr_lars_swin_l.safetensors`,
sha256 `e9fecf6f…566e0e`) was exported from the original training code,
which drifted structurally from this checkout in two ways:

1. **Shared box MLP → per-layer aliases.** The predictor shares one box
   MLP across all decoder layers (`pansr_decoder.py`:
   `box_embed_layerlist = [_bbox_embed for i in range(num_layers)]`,
   `decoder.bbox_embed = self.bbox_embed`), so the checkpoint stores only
   the six `predictor._bbox_embed.layers.*` tensors, while the current
   `state_dict()` contains the per-layer alias keys
   `predictor.bbox_embed.{i}.layers.*` **and**
   `predictor.decoder.bbox_embed.{i}.layers.*` — 108 alias keys expanded
   from the 6 shared sources (shape-asserted, regex-validated).
2. **`predictor.decoder_norm`** (final decoder LayerNorm) post-dates the
   checkpoint. A fresh `nn.LayerNorm` is identity (weight=1, bias=0),
   exactly reproducing the pre-norm behavior the checkpoint was trained
   with, so these two keys are allowed to be missing **only** for legacy
   (shared-`_bbox_embed`) checkpoints.

For temporal models, `geometry_encoder.*` / `temporal_fusion.*` keys are
additionally allowed to be missing when loading the original PanSR
weights (fresh init). Any other missing key or any shape mismatch is
fatal. Loading the real checkpoint reports: 0 mismatched, 0 unexpected,
2 legacy-allowed missing, 108 expanded aliases (see
`test_vanilla_checkpoint_loads` and the integration tests).

## 4. detectron2 0.6 facts measured (deviations documented here)

- `detectron2.modeling.build_model` runs
  `model.to(torch.device(cfg.MODEL.DEVICE))` — the model is GPU-placed by
  the builder; `PanSRTrainer.__init__`'s extra `.to()` is a no-op kept
  for clarity (verified in the installed d2 source).
- **The final eval is unconditional.** `DefaultTrainer.build_hooks`
  always appends an `EvalHook` with `eval_after_train=True`, regardless
  of `TEST.EVAL_PERIOD` (a period of 0 does not disable the post-training
  eval). On this 11 GB workstation that final eval OOMs (original-res
  `sem_seg_postprocess` needs a single 2.317 GiB upsample while the
  optimizer state is still live). Fix: `TEMPORAL.OPTIM.RUN_FINAL_EVAL`
  (default `True` = reference behavior; the workstation configs set it
  `False`) and `TemporalTrainer.build_hooks()` drops the `EvalHook` when
  it is `False`.
- `EventStorage.latest()` maps `key -> (value, iteration)` tuples (tools
  unpacking it must handle the tuple; a bug here was caught by
  `overfit_probe.py` on first real run and fixed).
- `Checkpoint` optimizer state is
  `{"state": {<param idx>: {"step", "exp_avg", ...}}, "param_groups": [...]}`;
  the checkpointer only loads checkpointable keys that are present in an
  old checkpoint.
- `Checkpointer` only reads `.pth`; `.safetensors` weights go through
  `load_model_weights` (see §3).
- `yaml.unsafe_load` warning from `fvcore` when loading the repo configs
  (upstream behavior, harmless for these files).

## 5. Fixes made in this project (beyond new files)

All are in the new files unless stated; none change vanilla model
outputs (bit-exact parity tests in `tests/temporal/test_model_parity.py`
and `test_mapper_parity.py`).

1. **`pansr/temporal/model.py::_geometry_for_sample`** — moved the
   context clip with `bi["clip_rgb"].to(self.device)`; the batch dict's
   clip was left on CPU by the mapper and the real `train_net.py` flow
   would have crashed at step 1 (earlier smokes manually moved batches
   and masked the bug).
2. **`pansr/temporal/mapper.py`** — the val branch now includes
   `"file_name"` (from `target_file`); `COCOPanopticEvaluator.process`
   requires it (final eval crashed without it).
3. **`pansr/temporal/geometry.py`** — the encoder constructor now
   normalizes `device` through `torch.device(...)` (accepts strings).
4. **Resize/anchor coordinate convention** (`pansr/temporal/alignment.py`
   + mapper) — the crop→geometry affine is computed with the raster
   convention the Pi3 encoder actually uses (floor division for the
   token grid, center-aligned half-pixel offsets). A previous version
   off-by-one'd the resample for some clip sizes; the corrected
   convention shifts resampled geometry features by ≤ ~0.5 raster pixel
   relative to the older behavior (validity unaffected) — see
   `tests/temporal/test_alignment.py`.
5. **`pansr/trainer.py`** — training-state persistence (see §2).
6. **`tests/temporal/conftest.py`** — resource guards are session-scoped
   fixtures (required by pytest for the shared CUDA/GPU session state).
7. **`tests/temporal/test_integration_gpu.py`** — cleanup helper:
   unbinding model/trainer references must happen in the test's own
   frame (`del` inside a helper does not unbind the caller's local),
   and `_gpu_freed()` asserts < 64 MiB remains after each phase; without
   this, a stale reference in one phase cascaded into a GPU OOM in a
   later phase (and pytest keeps a failed test's traceback frame —
   including its GPU locals — alive for the next test).
8. **`tools/temporal/overfit_probe.py`, `smoke_test.py`** —
   `EventStorage.latest()` returns `(value, iteration)` tuples in d2 0.6;
   the loss-capture hooks now unpack them.
9. **`train_net.py`** — removed an unused `DetectionCheckpointer` import
   from `detectron2.engine` (in d2 0.6 it lives in `detectron2.checkpoint`);
   the `--eval_only` path crashed at import on first real run.
10. **`tools/temporal/profile_model.py`** — (a) added the required
    `--geometry-checkpoint` argument (building the temporal model otherwise
    fails config validation with an empty `GEOMETRY.CHECKPOINT`); (b) the
    manual train step now mirrors d2 0.6 `AMPTrainer.run_step` (forward
    under `torch.cuda.amp.autocast`, backward/optimizer outside) — without
    it PanSR's criterion crashed (fp16 `gt_cls_weight` index-put into an
    fp32 tensor); (c) branch timers: CUDA events recorded before/after the
    forward (pre-hook + hook; recording both after measures ~0 ms), the Pi3
    encoder is timed by wrapping its `encode()` method (it is not invoked
    via `__call__`), and eval/train times are reported separately.
11. **`tools/temporal/export_lars.py`** — added the required
    `--geometry-checkpoint` argument; and the model reads its panoptic
    metadata from `cfg.DATASETS.TRAIN[0]`, so when no train manifest is set
    the tool registers the temporal train dataset from the val manifest
    (metadata only; the train loader is never built).
12. **`tools/temporal/extract_geometry.py`** — (a) OpenCV 5.x: `imdecode`
    requires an ndarray (`np.frombuffer`), not raw `bytes`; (b) the encoder
    now receives the target's position *inside the selected frames*
    (matching the online mapper's `ordered_idx.index(target_index)`), not
    the full-clip index.
13. **`tools/temporal/smoke_test.py`** — val-forward rework: after training
    the whole trainer stack is freed first (the loss-capture hook holds the
    trainer; optimizer state plus the ~8.3 GiB eval-forward peak OOMs on the
    11 GB card), then a freshly built model reloads the checkpointer's
    `model_final.pth` through the validated loader (round-trip check) and
    runs the val forward without optimizer state.

## 6. Pi3 / GMOS notes

- GMOS is a **read-only reference** at
  `/media/jon/disk2/code/PanSR-T/gmos` (commit `99d314ac`); nothing there
  is modified. `Pi3GeometryEncoder` imports `gmos/core/pi3_encoder.py`.
- sm75 has **no bf16 SDPA backend** → the frozen encoder runs in fp32
  (logged warning at build time; this is the verified-precision path and
  is identical on the training workstation if it lacks bf16 too).
- Measured property: Pi3's cross-frame attention uses 2D-spatial RoPE
  only (no temporal position), so its output is **permutation-invariant
  in frame order** (max measured diff 1.9e-6 vs original order; exact
  reproducibility). The genuinely temporal signal in the model is that
  *context content* changes the target features — covered by
  `test_geometry.py::test_context_changes_target_features`.

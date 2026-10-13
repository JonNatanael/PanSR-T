# ------------------------------------------------------------------------
# TG-PanSR: Temporal-Geometric PanSR configuration.
#
# Adds the ``TEMPORAL`` configuration node and validates incompatible
# settings.  This module must be imported (via ``add_temporal_config``)
# *before* a config file is merged, mirroring ``add_pansr_config``.
# ------------------------------------------------------------------------
from detectron2.config import CfgNode as CN

TEMPORAL_SAMPLING_MODES = (
    "all_history",
    "recent",
    "uniform_history",
    "target_only",
    "repeat_target",
)

TEMPORAL_PREPROCESS_MODES = (
    "shared_view_online",
    "canonical_full_frame_cache",
)

GEOMETRY_BACKENDS = ("pi3_gmos",)


def add_temporal_config(cfg):
    """Register all TG-PanSR config keys (idempotent)."""
    try:
        already = getattr(cfg, "TEMPORAL")
    except AttributeError:
        already = None
    if already is not None and already.get("ENABLED", None) is not None:
        # already registered
        return

    cfg.TEMPORAL = CN()
    # Master switch.  When False the model runs the exact vanilla PanSR
    # feature path and requires no manifest and no geometry checkpoint.
    cfg.TEMPORAL.ENABLED = True

    # Manifests (JSONL, produced by tools/temporal/build_manifest.py).
    # Relative paths are resolved against the current working directory.
    # Required (non-empty) whenever TEMPORAL.ENABLED is True.
    cfg.TEMPORAL.MANIFEST_TRAIN = ""
    cfg.TEMPORAL.MANIFEST_VAL = ""

    # Number of frames in the geometric clip (T).  For the standard LaRS
    # record this is 10 (9 preceding frames + annotated target).
    cfg.TEMPORAL.NUM_FRAMES = 10

    # Frame selection mode; see pansr.temporal.manifest for semantics.
    cfg.TEMPORAL.SAMPLING = "all_history"
    # Target policy: "explicit" reads the target index from the manifest
    # record (the only supported policy in v1).
    cfg.TEMPORAL.TARGET_POLICY = "explicit"
    # Causal protocol: selected context frames must precede the target.
    cfg.TEMPORAL.ALLOW_FUTURE = False
    # What to do when context frames are missing on disk:
    # "error" (default, strict) or "fallback_baseline" (run vanilla path for
    # that sample and log the count).
    cfg.TEMPORAL.MISSING_POLICY = "error"
    # Optional context dropout: with this probability a *valid* context
    # frame is physically removed before the Pi3 call (target is kept).
    # The frozen encoder has no temporal padding mask, so removed frames
    # are simply not passed.  0.0 disables it.
    cfg.TEMPORAL.FRAME_DROPOUT = 0.0
    # Seed used for reproducible sampling (uniform_history, dropout).
    cfg.TEMPORAL.SAMPLING_SEED = 0

    # Preprocessing protocol.
    #   "shared_view_online":            Pi3 runs on the spatially augmented
    #       clip (same crop/flip/resize as the target), online.
    #   "canonical_full_frame_cache":    Pi3 ran offline on the raw full-frame
    #       clip (deterministic preprocessing); the model loads cached
    #       features and maps them into the augmented target view.
    cfg.TEMPORAL.PREPROCESS_MODE = "shared_view_online"

    cfg.TEMPORAL.GEOMETRY = CN()
    cfg.TEMPORAL.GEOMETRY.BACKEND = "pi3_gmos"
    # Path to the Pi3 checkpoint (.safetensors or .pth).  Required when the
    # geometry branch is active; validated up front (file must exist).
    cfg.TEMPORAL.GEOMETRY.CHECKPOINT = ""
    # Root of the GMOS checkout containing core/pi3_encoder.py (falls back
    # to the GMOS_ROOT environment variable).
    cfg.TEMPORAL.GEOMETRY.GMOS_ROOT = ""
    # Pi3 is frozen by definition of the reference architecture.
    cfg.TEMPORAL.GEOMETRY.FROZEN = True
    # Which Pi3 decoder output to consume:
    #   "last_decoder"      (default) final block, [B,T,h,w,1024]
    #   "tapped:[7,15,23,31]" cross-frame block taps, combined with learned
    #       weights (see pansr.temporal.geometry.Pi3GeometryEncoder).
    cfg.TEMPORAL.GEOMETRY.FEATURE_MODE = "last_decoder"
    cfg.TEMPORAL.GEOMETRY.TAPPED_LAYERS = (7, 15, 23, 31)
    # Expected feature width for the configured extraction mode.  Mismatches
    # are configuration errors, never fixed by slicing arbitrary channels.
    cfg.TEMPORAL.GEOMETRY.FEATURE_WIDTH = 1024
    # Geometry pixel budget (H_geo * W_geo <= PIXEL_BUDGET, dims divisible
    # by the backend patch size).  255000 follows GMOS preprocessing.
    cfg.TEMPORAL.GEOMETRY.PIXEL_BUDGET = 255000
    # Numerical precision for the frozen encoder: "fp32" or "bf16"
    # ("bf16" is effectively fp32 compute — Pi3's fp32 normalization buffers
    # upcast bf16 inputs; reference-protocol behavior), or "bf16_strict"
    # (true bf16 compute: the frozen weights/buffers are cast to bf16;
    # saves ~1.5GB VRAM but deviates from the reference protocol by ~1-2%
    # relative feature error — use only when VRAM-constrained).
    cfg.TEMPORAL.GEOMETRY.DTYPE = "bf16"
    # Feature cache: "none" (always recompute) or a directory for the
    # final-clip frozen-feature cache (shared_view_online only).
    cfg.TEMPORAL.GEOMETRY.CACHE_MODE = "none"
    cfg.TEMPORAL.GEOMETRY.CACHE_DIR = ""

    cfg.TEMPORAL.FUSION = CN()
    # "gmos_bottleneck_residual": projectors + merge + spatial transformer
    #   refinement + per-level residual injection (reference architecture).
    # "concat_only": projectors + merge only, no spatial transformer.
    # "off": no fusion (used by the zero-geometry capacity control B2 via
    #   zero geometry input; see configs).
    cfg.TEMPORAL.FUSION.MODE = "gmos_bottleneck_residual"
    # Swin level used as the appearance bottleneck source.
    cfg.TEMPORAL.FUSION.APPEARANCE_SOURCE = "res4"
    cfg.TEMPORAL.FUSION.DIM = 256
    cfg.TEMPORAL.FUSION.REFINE_LAYERS = 3
    cfg.TEMPORAL.FUSION.NUM_HEADS = 8
    cfg.TEMPORAL.FUSION.FFN_DIM = 2048
    cfg.TEMPORAL.FUSION.DROPOUT = 0.1
    cfg.TEMPORAL.FUSION.POSITION_ENCODING = "normalized_2d_sincos"
    # Swin levels receiving residual injection.
    cfg.TEMPORAL.FUSION.INJECT_FEATURES = ("res2", "res3", "res4", "res5")
    # Hidden width of the per-level residual adapters.
    cfg.TEMPORAL.FUSION.INJECT_DIM = 64
    # "zero_output": final residual projection initialized to zero, gate
    #   logits zero (sigmoid 0.5) -> initial features equal the original.
    cfg.TEMPORAL.FUSION.RESIDUAL_INIT = "zero_output"
    cfg.TEMPORAL.FUSION.LEARNED_GATE = True
    # "auto": derive the geometry zero-input control (B2) from the target
    #   crop lattice; used by the appearance_adapter_control config.
    cfg.TEMPORAL.FUSION.ZERO_GEOMETRY = False

    cfg.TEMPORAL.OPTIM = CN()
    # Learning rate for newly introduced trainable modules
    # (geometry adapters + fusion).  Existing modules follow SOLVER.BASE_LR
    # (with BACKBONE_MULTIPLIER for the backbone).
    cfg.TEMPORAL.OPTIM.NEW_MODULE_LR = 1e-4
    # Gradient accumulation: optimizer updates happen every ACC steps of
    # data.  Effective batch = IMS_PER_BATCH * world_size * ACC.
    cfg.TEMPORAL.OPTIM.ACCUMULATION_STEPS = 8
    # Assert the effective target batch during training (0 disables).
    cfg.TEMPORAL.OPTIM.EXPECTED_EFFECTIVE_BATCH = 8
    # detectron2 0.6 always runs a full-val evaluation after the last
    # training iteration (EvalHook.eval_after_train, regardless of
    # TEST.EVAL_PERIOD).  On the 11 GB workstation that final evaluation
    # is not runnable while the optimizer state is still live (original-
    # resolution panoptic postprocess + val-sized backbone forward); set
    # False in the workstation probe configs.  The reference protocol
    # keeps True.
    cfg.TEMPORAL.OPTIM.RUN_FINAL_EVAL = True


def validate_temporal_config(cfg, is_train):
    """Validate temporal settings; raise on incompatible combinations.

    Called after config merge (before model/data construction).
    """
    t = cfg.TEMPORAL
    assert t.SAMPLING in TEMPORAL_SAMPLING_MODES, (
        f"TEMPORAL.SAMPLING must be one of {TEMPORAL_SAMPLING_MODES}, got {t.SAMPLING!r}"
    )
    assert t.TARGET_POLICY == "explicit", (
        f"TEMPORAL.TARGET_POLICY must be 'explicit', got {t.TARGET_POLICY!r}"
    )
    assert t.PREPROCESS_MODE in TEMPORAL_PREPROCESS_MODES, (
        f"TEMPORAL.PREPROCESS_MODE must be one of {TEMPORAL_PREPROCESS_MODES}"
    )
    assert t.MISSING_POLICY in ("error", "fallback_baseline"), (
        f"TEMPORAL.MISSING_POLICY must be 'error' or 'fallback_baseline', got {t.MISSING_POLICY!r}"
    )
    assert 0.0 <= t.FRAME_DROPOUT < 1.0
    assert t.NUM_FRAMES >= 1
    assert t.GEOMETRY.BACKEND in GEOMETRY_BACKENDS
    assert t.GEOMETRY.FEATURE_MODE == "last_decoder" or t.GEOMETRY.FEATURE_MODE.startswith("tapped:"), (
        f"TEMPORAL.GEOMETRY.FEATURE_MODE must be 'last_decoder' or 'tapped:<layers>'"
    )
    assert t.GEOMETRY.DTYPE in ("fp32", "bf16", "bf16_strict")
    assert t.GEOMETRY.CACHE_MODE in ("none", "clip_features")
    assert t.FUSION.MODE in ("gmos_bottleneck_residual", "concat_only", "off")
    assert t.FUSION.RESIDUAL_INIT == "zero_output"
    assert t.FUSION.ZERO_GEOMETRY in (False, "auto")
    assert len(t.FUSION.INJECT_FEATURES) >= 1
    assert t.FUSION.APPEARANCE_SOURCE in ("res2", "res3", "res4", "res5")

    if t.ENABLED:
        assert t.GEOMETRY.CHECKPOINT, (
            "TEMPORAL.ENABLED=True requires TEMPORAL.GEOMETRY.CHECKPOINT to be set"
        )
        if is_train:
            assert t.MANIFEST_TRAIN, "TEMPORAL.ENABLED=True requires TEMPORAL.MANIFEST_TRAIN"
        else:
            assert t.MANIFEST_VAL, "TEMPORAL.ENABLED=True requires TEMPORAL.MANIFEST_VAL"
        if t.PREPROCESS_MODE == "canonical_full_frame_cache":
            assert t.GEOMETRY.CACHE_DIR, (
                "TEMPORAL.PREPROCESS_MODE='canonical_full_frame_cache' requires "
                "TEMPORAL.GEOMETRY.CACHE_DIR with pre-extracted features"
            )
        # 'target_only' still runs the geometric branch with a single frame;
        # it must not silently become the vanilla baseline.
        if t.SAMPLING == "target_only":
            assert not t.FUSION.ZERO_GEOMETRY, (
                "target_only sampling with zero geometry is contradictory; "
                "use TEMPORAL.ENABLED=False for the vanilla baseline"
            )
        # 'repeat_target' is only legal as an explicit named control.
        if t.SAMPLING == "repeat_target":
            assert t.NUM_FRAMES >= 2, "repeat_target requires NUM_FRAMES >= 2"

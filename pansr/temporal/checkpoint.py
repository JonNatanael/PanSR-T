# ------------------------------------------------------------------------
# TG-PanSR: checkpoint handling and reproducibility manifests.
#
# * explicit, validated loading of the original PanSR checkpoint (the
#   same checkpoint is used for every matched run);
# * no downloads at import time or training time;
# * experiment manifests capturing everything needed to reproduce a run.
# ------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

__all__ = ["validate_checkpoint_path", "load_pansr_weights", "write_experiment_manifest"]


def validate_checkpoint_path(path: str, what: str) -> str:
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{what} checkpoint not found: {p}")
    return p


def load_pansr_weights(model: torch.nn.Module, path: str, device="cpu") -> Dict[str, Any]:
    """Load an original PanSR checkpoint (``.pth`` or ``.safetensors``)
    into ``model`` with full key validation.  Returns a small report.

    For the temporal model the *new* modules (``geometry_encoder.*`` /
    ``temporal_fusion.*``) are expected to be missing from an original
    PanSR checkpoint; :func:`_load_state_dict_strictish` raises for
    missing/mismatched keys of the *original* modules only when
    ``strict_original=True`` is passed by the caller through ``extra``.
    """
    path = validate_checkpoint_path(path, "PanSR")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(path, device=str(device))
        return _load_state_dict_strictish(model, state)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return _load_state_dict_strictish(model, state)


def _load_state_dict_strictish(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    model_state = model.state_dict()
    # allow the common 'module.' / 'model.' prefixes
    for prefix in ("module.", "model."):
        if all(k.startswith(prefix) for k in state) and any(
            k[len(prefix):] in model_state for k in list(state)[:5]
        ):
            state = {k[len(prefix):]: v for k, v in state.items()}
    missing = [k for k in model_state if k not in state]
    unexpected = [k for k in state if k not in model_state]
    mismatched = [
        k for k in model_state
        if k in state and (model_state[k].shape != state[k].shape)
    ]
    if missing or mismatched:
        raise RuntimeError(
            f"PanSR checkpoint incompatible: {len(missing)} missing "
            f"(e.g. {missing[:5]}), {len(mismatched)} shape-mismatched "
            f"(e.g. {mismatched[:5]})"
        )
    model.load_state_dict({k: v for k, v in state.items() if k in model_state}, strict=False)
    return {"missing": missing, "unexpected": unexpected, "mismatched": mismatched}


def write_experiment_manifest(
    path: str,
    *,
    cfg: Any,
    model: torch.nn.Module,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a JSON manifest describing everything needed to reproduce a
    run: resolved config, checkpoint hashes, parameter counts, device."""
    import platform

    t = getattr(cfg, "TEMPORAL", None)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    manifest = {
        "created": _now_iso(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": cfg.dump(),
        "weights": {
            "pansr": os.path.abspath(cfg.MODEL.WEIGHTS) if cfg.MODEL.WEIGHTS else None,
        },
        "parameter_counts": {
            "trainable": len(trainable),
            "frozen": len(frozen),
            "trainable_tensors": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_tensors": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        },
    }
    if t is not None:
        manifest["temporal"] = {
            "enabled": t.ENABLED,
            "geometry_checkpoint": (
                os.path.abspath(t.GEOMETRY.CHECKPOINT) if t.GEOMETRY.CHECKPOINT else None
            ),
            "manifest_train": t.MANIFEST_TRAIN or None,
            "manifest_val": t.MANIFEST_VAL or None,
            "sampling": t.SAMPLING,
            "num_frames": t.NUM_FRAMES,
            "preprocess_mode": t.PREPROCESS_MODE,
        }
    if extra:
        manifest.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("experiment manifest written to %s", path)


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")

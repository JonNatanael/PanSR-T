# ------------------------------------------------------------------------
# TG-PanSR: temporal-geometric extension of PanSR.
#
# This package is optional: with ``TEMPORAL.ENABLED=False`` (or when this
# package is not imported) PanSR runs exactly as before and needs neither
# Pi3 checkpoints nor the GMOS checkout.
#
# Importing this package registers:
#   * the ``TEMPORAL`` config node            (config.add_temporal_config)
#   * the ``TemporalPanSR`` meta architecture (model.TemporalPanSR)
#   * the ``lars_temporal_panoptic`` mapper   (mapper, used by the trainer)
#
# Nothing here downloads weights, loads checkpoints, or imports GMOS/Pi3
# at import time.  Heavy dependencies are loaded lazily by
# ``geometry.build_geometry_encoder`` when a model actually needs them.
# ------------------------------------------------------------------------
from .config import (
    TEMPORAL_PREPROCESS_MODES,
    TEMPORAL_SAMPLING_MODES,
    add_temporal_config,
    validate_temporal_config,
)
from . import alignment
from . import cache
from . import checkpoint
from . import fusion
from . import geometry
from . import manifest
from .model import TemporalPanSR  # registers "TemporalPanSR" meta arch
from .mapper import TemporalCOCOPanopticDatasetMapper
from .predictor import TemporalPredictor
from .trainer import TemporalTrainer
from .dataset import register_temporal_lars, build_temporal_dataset_dicts

__all__ = [
    "TEMPORAL_PREPROCESS_MODES",
    "TEMPORAL_SAMPLING_MODES",
    "add_temporal_config",
    "validate_temporal_config",
    "alignment",
    "cache",
    "checkpoint",
    "fusion",
    "geometry",
    "manifest",
    "TemporalPanSR",
    "TemporalCOCOPanopticDatasetMapper",
    "TemporalPredictor",
    "TemporalTrainer",
    "register_temporal_lars",
    "build_temporal_dataset_dicts",
]

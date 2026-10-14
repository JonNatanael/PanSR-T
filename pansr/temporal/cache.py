# ------------------------------------------------------------------------
# TG-PanSR: frozen geometry feature cache.
#
# The cache stores *outputs of the frozen Pi3 encoder*.  Online keys cover
# one specific joint clip and preprocessing operation; canonical keys cover
# deterministic raw-full-frame extraction before training augmentation.
#
#   * cache schema version
#   * model/checkpoint sha256 + feature mode + dtype
#   * geometry pixel budget
#   * ordered selected frame ids + target index
#   * per-frame content fingerprints (file sha256)
#   * the exact spatial/photometric transform (branch_transforms)
#
# Trainable modules are never part of the cached value.  Writes are
# atomic; reads validate schema/shape/dtype and reject stale entries.
# ------------------------------------------------------------------------
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1

__all__ = ["GeometryFeatureCache", "GeometryCacheMiss", "content_fingerprint"]


class GeometryCacheMiss(KeyError):
    pass


def content_fingerprint(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class GeometryFeatureCache:
    def __init__(self, cache_dir: str, split: str, read_write: bool = True):
        self.cache_dir = os.path.abspath(cache_dir)
        self.split = split
        self.read_write = read_write
        self._dir = os.path.join(self.cache_dir, split)
        if read_write:
            os.makedirs(self._dir, exist_ok=True)
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    @staticmethod
    def make_key(
        *,
        checkpoint_sha256: str,
        feature_mode: str,
        compute_dtype: str,
        pixel_budget: int,
        frame_ids: Any,
        frame_fingerprints: Any,
        target_index: int,
        branch_transforms: Dict[str, Any],
    ) -> str:
        payload = {
            "schema": CACHE_SCHEMA_VERSION,
            "checkpoint_sha256": checkpoint_sha256,
            "feature_mode": feature_mode,
            "compute_dtype": compute_dtype,
            "pixel_budget": int(pixel_budget),
            "frame_ids": list(frame_ids),
            "frame_fingerprints": list(frame_fingerprints),
            "target_index": int(target_index),
            "branch_transforms": json.loads(json.dumps(branch_transforms, sort_keys=True)),
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    @staticmethod
    def make_canonical_key(
        *,
        checkpoint_sha256: str,
        feature_mode: str,
        requested_dtype: str,
        pixel_budget: int,
        frame_ids: Any,
        target_index: int,
    ) -> str:
        """Return the stable lookup key for an offline full-frame clip.

        Unlike :meth:`make_key`, this deliberately has no augmentation in
        the key: canonical features are extracted before random training
        transforms and are mapped into each sampled view at load time.
        """
        payload = {
            "schema": CACHE_SCHEMA_VERSION,
            "protocol": "canonical_full_frame_cache",
            "checkpoint_sha256": str(checkpoint_sha256),
            "feature_mode": str(feature_mode),
            "requested_dtype": str(requested_dtype),
            "pixel_budget": int(pixel_budget),
            "frame_ids": list(frame_ids),
            "target_index": int(target_index),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    # ------------------------------------------------------------------
    def _path(self, key: str):
        return os.path.join(self._dir, key + ".npz")

    def get_with_metadata(self, key: str):
        path = self._path(key)
        if not os.path.isfile(path):
            self.misses += 1
            return None, None
        try:
            with np.load(path, allow_pickle=False) as npz:
                if int(npz["schema"]) != CACHE_SCHEMA_VERSION:
                    raise GeometryCacheMiss(f"stale schema in {path}")
                arr = npz["features"]
                raw_meta = npz["metadata"].item()
                if isinstance(raw_meta, bytes):
                    raw_meta = raw_meta.decode("utf-8")
                meta = json.loads(raw_meta)
                if tuple(arr.shape) != tuple(meta["shape"]):
                    raise GeometryCacheMiss(f"shape mismatch in {path}")
                tensor = torch.from_numpy(arr.astype(np.float32))
            self.hits += 1
            return tensor, meta
        except GeometryCacheMiss as e:
            logger.warning("geometry cache: %s", e)
            self.misses += 1
            return None, None
        except Exception as e:  # corrupt entry
            logger.warning("geometry cache: corrupt entry %s (%s); treating as miss", path, e)
            self.misses += 1
            return None, None

    def get(self, key: str) -> Optional[torch.Tensor]:
        features, _ = self.get_with_metadata(key)
        return features

    def put(self, key: str, features: torch.Tensor, metadata: Dict[str, Any]) -> None:
        if not self.read_write:
            return
        arr = features.detach().cpu().numpy().astype(np.float32)
        payload_meta = dict(metadata)
        payload_meta["shape"] = list(arr.shape)
        path = self._path(key)
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(
                    f,
                    features=arr,
                    schema=np.array(CACHE_SCHEMA_VERSION),
                    metadata=json.dumps(payload_meta, sort_keys=True).encode("utf-8"),
                )
            os.replace(tmp, path)  # atomic
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ------------------------------------------------------------------
    def report(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}

"""Model-update aggregation: FedAvg + optional OTA/AirComp distortion.

* ``fedavg`` — sample-count-weighted average of client update vectors.
* ``ota_distort`` — additive Gaussian distortion on the aggregated update; its
  std is tied to the **AirComp aggregation MSE** (sim/aircomp.py) via a config
  scale: ``std = scale * sqrt(MSE)``. FedAvg works unchanged when OTA is off.
"""
from __future__ import annotations

import numpy as np


def fedavg(updates, sample_counts) -> np.ndarray:
    """Sample-weighted average of update vectors -> (D,)."""
    weights = np.asarray(sample_counts, dtype=float)
    weights = weights / max(weights.sum(), 1e-12)
    return np.average(np.stack(updates), axis=0, weights=weights)


def ota_distort(agg_update: np.ndarray, mse: float, scale: float,
                rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise (std = scale * sqrt(MSE)) to the aggregated update."""
    std = float(scale) * np.sqrt(max(float(mse), 0.0))
    if std <= 0.0:
        return agg_update
    return agg_update + rng.normal(0.0, std, size=agg_update.shape)


def aggregate(updates, sample_counts, *, ota: bool = False, mse: float = 0.0,
              scale: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """FedAvg, optionally followed by OTA/AirComp distortion."""
    agg = fedavg(updates, sample_counts)
    if ota and rng is not None:
        agg = ota_distort(agg, mse, scale, rng)
    return agg

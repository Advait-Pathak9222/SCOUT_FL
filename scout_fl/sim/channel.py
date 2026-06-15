"""Communication channels (client -> parameter server) for AirComp.

Power gain  g_k = large_scale_k * |small_scale_k|^2, where the large-scale term
follows a reference-SNR distance-decay law and the small-scale term is Rayleigh
or Rician fading. Reference-SNR parametrization (not raw dBm) keeps P*g/sigma^2
in a realistic range (same calibration lesson as sim/sensing.py).
"""
from __future__ import annotations

import numpy as np


def comm_channel_gains(clients: np.ndarray, bs: np.ndarray, rng: np.random.Generator,
                       *, snr_ref_db: float = 20.0, ref_distance: float = 10.0,
                       pathloss_exponent: float = 3.0, model: str = "rician",
                       rician_k_db: float = 6.0) -> np.ndarray:
    """Return per-client channel power gains ``g_k = |h_k|^2`` -> (K,)."""
    clients = np.asarray(clients, dtype=float)
    bs = np.asarray(bs, dtype=float)
    dist = np.clip(np.linalg.norm(clients - bs, axis=1), ref_distance, None)   # (K,)
    large = 10.0 ** (snr_ref_db / 10.0) * (ref_distance / dist) ** pathloss_exponent
    K = dist.shape[0]

    if model == "rayleigh":
        h = (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2.0)
    elif model == "rician":
        kappa = 10.0 ** (rician_k_db / 10.0)
        los = np.sqrt(kappa / (kappa + 1.0))
        nlos = np.sqrt(1.0 / (kappa + 1.0))
        h = los + nlos * (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2.0)
    else:
        raise ValueError(f"unknown fading model {model!r}")
    return large * np.abs(h) ** 2                                              # (K,)

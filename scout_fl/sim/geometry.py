"""Spatial geometry for clients and targets.

Everything downstream (FIM direction, range-dependent angle precision, path
loss) is derived from the client->target geometry computed here. All operations
are vectorized over (clients K, targets M).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def pairwise_geometry(clients: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Compute per (client, target) range, bearing and unit vectors.

    Parameters
    ----------
    clients : array_like, shape (K, 2)
    targets : array_like, shape (M, 2)

    Returns
    -------
    dict with:
      range   : (K, M) Euclidean client->target distance
      u       : (K, M, 2) radial unit vector (client -> target); the direction
                a range measurement informs
      v       : (K, M, 2) tangential unit vector (u rotated +90 deg); the
                direction an angle/cross-range measurement informs
      bearing : (K, M) bearing angle (rad) of target as seen from client
      clients : (K, 2), targets : (M, 2)
    """
    clients = np.asarray(clients, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if clients.ndim != 2 or clients.shape[1] != 2:
        raise ValueError(f"clients must be (K,2), got {clients.shape}")
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError(f"targets must be (M,2), got {targets.shape}")

    delta = targets[None, :, :] - clients[:, None, :]      # (K, M, 2)
    rng = np.linalg.norm(delta, axis=-1)                   # (K, M)
    safe = np.clip(rng, 1e-12, None)[..., None]
    u = delta / safe                                       # (K, M, 2)
    v = np.stack([-u[..., 1], u[..., 0]], axis=-1)         # (K, M, 2) +90 deg
    bearing = np.arctan2(delta[..., 1], delta[..., 0])     # (K, M)
    return {
        "range": rng,
        "u": u,
        "v": v,
        "bearing": bearing,
        "clients": clients,
        "targets": targets,
    }


def sample_positions(rng: np.random.Generator, n: int, area: np.ndarray) -> np.ndarray:
    """Uniformly sample ``n`` 2-D positions in a rectangular ``area`` (width,height)."""
    area = np.asarray(area, dtype=float)
    return rng.uniform(0.0, 1.0, size=(n, 2)) * area

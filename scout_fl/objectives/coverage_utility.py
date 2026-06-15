"""Coverage / freshness over a spatial region map.

Two pieces:

* ``CoverageMap`` — the *dynamic* per-region uncertainty state evolved BETWEEN
  rounds:  U_r(t+1) = rho_r * U_r(t) + xi_r - sum_{k in S_t} c_{k,r}  (clipped >=0).
  Unsensed regions accumulate uncertainty (freshness/AoI); sensed regions drop.

* ``CoverageUtility`` — the *per-round* selection term given the current map:
      f_cov(S) = sum_r U_r * g( sum_{k in S} c_{k,r} ),   g concave, g(0)=0
  monotone submodular for nonnegative U_r and concave nondecreasing ``g``.

``region_centers`` / ``contribution_matrix`` build the geometry-derived
client->region contribution ``c_{k,r}``.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def region_centers(area, num_regions: int) -> np.ndarray:
    """Near-square grid of ``num_regions`` region centres in a 2-D ``area``."""
    area = np.asarray(area, dtype=float)
    nx = int(round(np.sqrt(num_regions)))
    nx = max(nx, 1)
    ny = int(np.ceil(num_regions / nx))
    xs = (np.arange(nx) + 0.5) / nx * area[0]
    ys = (np.arange(ny) + 0.5) / ny * area[1]
    grid = np.array([[x, y] for y in ys for x in xs])
    return grid[:num_regions]


def contribution_matrix(clients, centers, sensing_range: float,
                        weight=None) -> np.ndarray:
    """Client->region sensing contribution ``c_{k,r}`` in (0, 1] (Gaussian in distance)."""
    clients = np.asarray(clients, dtype=float)
    centers = np.asarray(centers, dtype=float)
    d2 = ((clients[:, None, :] - centers[None, :, :]) ** 2).sum(-1)   # (K, R)
    contrib = np.exp(-d2 / (2.0 * float(sensing_range) ** 2))
    if weight is not None:
        contrib = contrib * np.asarray(weight, dtype=float)[:, None]
    return contrib


class CoverageMap:
    """Dynamic region-uncertainty state (freshness/AoI map)."""

    def __init__(self, num_regions: int, rho: float = 0.9, innovation: float = 0.05,
                 u_init: float = 1.0, u_max: float | None = None) -> None:
        self.R = int(num_regions)
        self.rho = float(rho)
        self.xi = float(innovation)
        self.u_max = u_max
        self.U = np.full(self.R, float(u_init))

    def update(self, selected: Iterable[int], contrib: np.ndarray) -> np.ndarray:
        idx = list(selected)
        covered = contrib[idx].sum(axis=0) if idx else np.zeros(self.R)
        self.U = self.rho * self.U + self.xi - covered
        self.U = np.clip(self.U, 0.0, self.u_max)
        return self.U


class CoverageUtility:
    """Per-round coverage term over the current uncertainty map."""

    _G = {
        "exp": lambda x: 1.0 - np.exp(-x),
        "min": lambda x: np.minimum(x, 1.0),
    }

    def __init__(self, region_uncertainty: np.ndarray, contrib: np.ndarray,
                 g: str = "exp") -> None:
        self.U = np.asarray(region_uncertainty, dtype=float)     # (R,)
        self.C = np.asarray(contrib, dtype=float)                # (K, R)
        self.K, self.R = self.C.shape
        if g not in self._G:
            raise ValueError(f"unknown saturating g={g!r}; choose from {list(self._G)}")
        self.g = self._G[g]

    def value(self, subset: Iterable[int]) -> float:
        idx = list(subset)
        acc = self.C[idx].sum(axis=0) if idx else np.zeros(self.R)
        return float((self.U * self.g(acc)).sum())

    def init_state(self) -> np.ndarray:
        return np.zeros(self.R)                                   # accumulated contribution

    def add(self, state: np.ndarray, k: int) -> np.ndarray:
        return state + self.C[k]

    def marginal_gain(self, state: np.ndarray, k: int) -> float:
        new = state + self.C[k]
        return float((self.U * (self.g(new) - self.g(state))).sum())

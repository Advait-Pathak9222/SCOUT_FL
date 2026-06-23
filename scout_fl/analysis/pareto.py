"""Pareto-front analysis + hypervolume for multi-objective method comparison.

Each method contributes ONE mean objective vector (e.g. accuracy, log-det
coverage-diversity, -CRB, -MSE, fairness). We min-max normalize across methods
(flipping 'lower is better' objectives so larger is always better), then report:
- ``pareto_front``      : which methods are non-dominated (Pareto-optimal);
- ``per_method_volume`` : each method's "all-round" score = MEAN of its normalized
                          objectives in [0,1] (robust; NOT the product-of-coords, which
                          collapses to 0 whenever a method is worst on any single axis);
- ``hypervolume``       : the dominated hypervolume of the whole method SET
                          (Monte-Carlo; the front-quality summary).
"""
from __future__ import annotations

import numpy as np


def normalize_objectives(points: np.ndarray, directions) -> np.ndarray:
    """Min-max normalize to [0,1] with larger=better. ``directions``: +1 (max) / -1 (min)."""
    pts = np.asarray(points, dtype=float) * np.asarray(directions, dtype=float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (pts - lo) / span


def pareto_front(points_hib: np.ndarray) -> np.ndarray:
    """Boolean non-dominated mask for points where larger is better on every axis."""
    pts = np.asarray(points_hib, dtype=float)
    n = pts.shape[0]
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                nd[i] = False
                break
    return nd


def per_method_volume(points_hib: np.ndarray) -> np.ndarray:
    """All-round score = MEAN of the normalized objectives (robust, in [0,1]).

    The earlier product-of-coords ('dominated box volume') collapsed any method to 0 if
    it was worst on even one axis (min-max forces the worst to exactly 0), which hid
    strong all-round methods and mislabelled the decision. The arithmetic mean is monotone,
    never spuriously zero, and reads as 'average normalized standing across objectives'."""
    return np.mean(np.clip(np.asarray(points_hib, dtype=float), 0.0, 1.0), axis=1)


def hypervolume(points_hib: np.ndarray, n_mc: int = 200000,
                rng: np.random.Generator | None = None) -> float:
    """Monte-Carlo dominated hypervolume of the SET (ref = origin) in [0,1]^d."""
    pts = np.asarray(points_hib, dtype=float)
    rng = rng if rng is not None else np.random.default_rng(0)
    samples = rng.random((int(n_mc), pts.shape[1]))
    dominated = np.any(np.all(samples[:, None, :] <= pts[None, :, :], axis=2), axis=1)
    return float(dominated.mean())

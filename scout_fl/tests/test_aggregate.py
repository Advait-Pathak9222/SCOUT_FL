"""Tests for the aggregate-eval layer: Pareto analysis + spatial non-IID.

Run:  pytest scout_fl/tests -q
"""
from __future__ import annotations

import numpy as np

from scout_fl.analysis.pareto import (hypervolume, normalize_objectives,
                                      pareto_front, per_method_volume)
from scout_fl.fl.partitioning import partition_spatial


# -------------------------------------------------------------------- pareto
def test_pareto_front_dominance():
    tradeoff = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    assert pareto_front(tradeoff).tolist() == [True, True, True]   # all non-dominated
    dominated = np.array([[1.0, 1.0], [0.5, 0.5]])
    assert pareto_front(dominated).tolist() == [True, False]       # first dominates second


def test_normalize_directions_and_range():
    pts = np.array([[0.9, 0.1], [0.5, 0.9]])      # obj0 higher-better, obj1 lower-better
    n = normalize_objectives(pts, [1, -1])
    assert n.shape == (2, 2) and n.min() >= -1e-9 and n.max() <= 1 + 1e-9
    # method 0 wins both after flipping obj1 -> should dominate
    assert np.all(n[0] >= n[1] - 1e-9)


def test_per_method_volume_and_hypervolume():
    pts = np.array([[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]])
    vol = per_method_volume(pts)
    assert vol[0] > vol[1] > vol[2]
    hv = hypervolume(pts, n_mc=50000, rng=np.random.default_rng(0))
    assert abs(hv - 1.0) < 0.05                    # (1,1) dominates the whole unit box


# ------------------------------------------------------------------- spatial
def test_partition_spatial_valid_and_clustered():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 3000)
    cluster = np.array([0, 0, 1, 1, 2, 2])         # 6 clients, 3 clusters
    parts = partition_spatial(labels, cluster, alpha=0.2, rng=rng)
    flat = np.concatenate(parts)
    assert len(parts) == 6
    assert len(set(flat.tolist())) == len(flat)    # disjoint

    def dist(idx):
        h = np.bincount(labels[idx], minlength=10).astype(float)
        return h / max(h.sum(), 1.0)
    within = np.linalg.norm(dist(parts[0]) - dist(parts[1]))     # same cluster
    cross = np.linalg.norm(dist(parts[0]) - dist(parts[4]))      # different cluster
    assert within <= cross + 0.2                   # within-cluster distributions more similar

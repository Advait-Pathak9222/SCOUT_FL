"""Tests for the FL dataset pipeline: partitioning (NumPy-only) + tensorization.

The real torchvision download is exercised by the smoke run / a cached-data test
(skipped when MNIST is not present), so the suite stays fast and offline-safe.

Run:  pytest scout_fl/tests -q
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from scout_fl.fl.partitioning import (partition, partition_dirichlet,
                                       partition_iid, partition_report)


def _labels(n=2000, c=10, seed=0):
    return np.random.default_rng(seed).integers(0, c, size=n)


def test_iid_is_a_balanced_partition():
    y = _labels(); K = 10
    parts = partition_iid(y, K, np.random.default_rng(1))
    flat = np.concatenate(parts)
    assert len(parts) == K
    assert len(flat) == len(y)
    assert len(set(flat.tolist())) == len(y)           # disjoint + covers all
    sizes = [len(p) for p in parts]
    assert max(sizes) - min(sizes) <= 1                 # balanced


def test_dirichlet_is_a_valid_partition():
    y = _labels(); K = 10
    parts = partition_dirichlet(y, K, 0.5, np.random.default_rng(2), min_size=1)
    flat = np.concatenate(parts)
    assert len(parts) == K
    assert len(flat) == len(y)
    assert len(set(flat.tolist())) == len(flat)         # disjoint
    assert min(len(p) for p in parts) >= 1


def test_dirichlet_skew_increases_as_alpha_decreases():
    y = _labels(n=6000); K = 10
    skewed = partition_report(y, partition_dirichlet(y, K, 0.1, np.random.default_rng(3)))
    uniform = partition_report(y, partition_dirichlet(y, K, 100.0, np.random.default_rng(4)))
    assert skewed["mean_top_class_fraction"] > uniform["mean_top_class_fraction"]


def test_partition_reproducible_with_seed():
    y = _labels()
    p1 = partition(y, 8, "dirichlet", 0.3, np.random.default_rng(5))
    p2 = partition(y, 8, "dirichlet", 0.3, np.random.default_rng(5))
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        partition(_labels(), 4, "spatial")             # not implemented yet


def test_to_tensors_shape_and_dtype():
    import torch
    from scout_fl.fl.datasets import _to_tensors
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=(20, 28, 28)).astype("uint8")
    targets = rng.integers(0, 10, size=20)
    x, y = _to_tensors(data, targets, (0.1307,), (0.3081,))
    assert tuple(x.shape) == (20, 1, 28, 28)
    assert x.dtype == torch.float32 and y.dtype == torch.long


def test_load_mnist_when_cached():
    """Runs only if MNIST was already downloaded (e.g. by the smoke run)."""
    pytest.importorskip("torchvision")
    if not os.path.exists("data/MNIST"):
        pytest.skip("MNIST not downloaded; run the smoke script first")
    from scout_fl.fl.datasets import build_client_datasets, load_dataset
    ds = load_dataset("mnist", root="data", download=False)
    assert ds.num_classes == 10 and tuple(ds.input_shape) == (1, 28, 28)
    parts = partition(np.asarray(ds.y_train), 10, "iid", rng=np.random.default_rng(0))
    clients = build_client_datasets(ds.x_train, ds.y_train, parts)
    assert len(clients) == 10 and len(clients[0]) > 0

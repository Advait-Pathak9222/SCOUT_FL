"""Federated dataset loading (MNIST / Fashion-MNIST) via torchvision.

Automatic download is controlled by config (``fl.download``). Images are loaded
as normalized float tensors so the partitioner can index them directly, and
per-client ``TensorDataset``s are built from index partitions.

DeepSense 6G / WiMANS (the semi-real wireless-sensing datasets) are deliberately
NOT here yet — they come only after A1-Full works on synthetic + MNIST/Fashion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

# per-dataset normalization (mean, std)
_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
}
_TORCHVISION_CLS = {"mnist": "MNIST", "fashion_mnist": "FashionMNIST"}


@dataclass
class FLDataset:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    num_classes: int
    input_shape: tuple


def _to_tensors(data, targets, mean: Sequence[float], std: Sequence[float]):
    """Convert raw (N,H,W) uint8 images + labels to normalized float tensors."""
    x = torch.as_tensor(np.asarray(data), dtype=torch.float32).div(255.0)
    if x.ndim == 3:
        x = x.unsqueeze(1)                          # (N, 1, H, W)
    m = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
    x = (x - m) / s
    y = torch.as_tensor(np.asarray(targets), dtype=torch.long)
    return x, y


def load_dataset(name: str, root: str = "data", download: bool = True) -> FLDataset:
    """Load MNIST/Fashion-MNIST as normalized tensors (download controlled by config)."""
    key = name.lower()
    if key not in _TORCHVISION_CLS:
        raise ValueError(f"unknown dataset {name!r}; supported: {list(_TORCHVISION_CLS)}")
    import torchvision  # imported lazily so the rest of the repo needs no torchvision

    cls = getattr(torchvision.datasets, _TORCHVISION_CLS[key])
    train = cls(root=root, train=True, download=download)
    test = cls(root=root, train=False, download=download)
    mean, std = _STATS[key]
    x_train, y_train = _to_tensors(train.data, train.targets, mean, std)
    x_test, y_test = _to_tensors(test.data, test.targets, mean, std)
    return FLDataset(
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        num_classes=int(y_train.max()) + 1, input_shape=tuple(x_train.shape[1:]),
    )


def build_client_datasets(x: torch.Tensor, y: torch.Tensor, parts):
    """Build per-client ``TensorDataset``s from index partitions."""
    from torch.utils.data import TensorDataset
    datasets = []
    for p in parts:
        idx = torch.as_tensor(np.asarray(p), dtype=torch.long)
        datasets.append(TensorDataset(x[idx], y[idx]))
    return datasets

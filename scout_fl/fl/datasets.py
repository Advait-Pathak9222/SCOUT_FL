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
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "emnist": ((0.1751,), (0.3332,)),               # FEMNIST-style handwritten characters
}
_TORCHVISION_CLS = {"mnist": "MNIST", "fashion_mnist": "FashionMNIST",
                    "cifar10": "CIFAR10", "cifar100": "CIFAR100", "emnist": "EMNIST"}
# EMNIST = the source data behind LEAF/FEMNIST; the 'balanced' split gives 47 classes of
# digits+letters, used here as the FEMNIST federated handwritten-character task.
_EMNIST_SPLIT = "balanced"


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
    x = torch.as_tensor(np.asarray(data), dtype=torch.float32)
    if x.ndim == 3:
        x = x.unsqueeze(1)                          # (N, H, W) grayscale -> (N, 1, H, W)
    elif x.ndim == 4 and x.shape[-1] in (1, 3):
        x = x.permute(0, 3, 1, 2).contiguous()      # (N, H, W, C) RGB -> (N, C, H, W)
    x = x / 255.0
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
    if key == "emnist":                              # EMNIST needs the split arg
        train = cls(root=root, split=_EMNIST_SPLIT, train=True, download=download)
        test = cls(root=root, split=_EMNIST_SPLIT, train=False, download=download)
    else:
        train = cls(root=root, train=True, download=download)
        test = cls(root=root, train=False, download=download)
    mean, std = _STATS[key]
    x_train, y_train = _to_tensors(train.data, train.targets, mean, std)
    x_test, y_test = _to_tensors(test.data, test.targets, mean, std)
    if key == "emnist":                              # EMNIST images are stored transposed
        x_train = x_train.transpose(-1, -2).contiguous()
        x_test = x_test.transpose(-1, -2).contiguous()
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


def load_fl_dataset(name: str, root: str = "data", download: bool = True) -> FLDataset:
    """Unified loader for every campaign FL task; dispatches by dataset name.

    * image classification (mnist / fashion_mnist / cifar10 / cifar100 / emnist=FEMNIST)
      -> torchvision via ``load_dataset``;
    * uci_har (smartphone Human Activity Recognition, tabular) -> ``datasets_extra``;
    * external wireless-sensing sources (deepmimo / deepsense6g / radarscenes) ->
      ``datasets_external`` scaffolded adapters (real file if present, else synthetic fallback).
    """
    key = name.lower()
    if key in _TORCHVISION_CLS:
        return load_dataset(key, root=root, download=download)
    if key in ("uci_har", "har"):
        from scout_fl.fl.datasets_extra import load_uci_har
        return load_uci_har(root=root, download=download)
    if key in ("deepmimo", "deepsense6g", "deepsense", "radarscenes"):
        from scout_fl.fl.datasets_external import load_external_classification
        return load_external_classification(key, root=root)
    raise ValueError(f"unknown dataset {name!r}")

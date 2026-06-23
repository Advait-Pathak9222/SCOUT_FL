"""Lightweight models for fast per-round federated training + flat-param utils.

Kept deliberately small (this is not an ML-architecture paper): an MLP and an
optional small CNN, selected by config. The flat-parameter helpers support
update-vector extraction (for the learning-utility embeddings) and applying
aggregated updates.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Flatten -> Linear -> ReLU -> Linear."""

    def __init__(self, input_dim: int, num_classes: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """2 conv blocks + linear head; input-size-adaptive (MNIST 1x28x28 or CIFAR 3x32x32)."""

    def __init__(self, in_channels: int, num_classes: int, input_hw=(28, 28)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, in_channels, *input_hw)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(model_type: str, input_shape, num_classes: int) -> nn.Module:
    """Construct a model from config.

    ``input_shape`` is (channels, H, W) for image tasks or (features,) for tabular /
    feature-vector tasks (UCI HAR, DeepSense/RadarScenes feature stand-ins).
    """
    input_shape = tuple(input_shape)
    input_dim = int(np.prod(input_shape))
    if model_type == "mlp":
        return MLP(input_dim, num_classes)
    if model_type == "small_cnn":
        if len(input_shape) != 3:
            raise ValueError(f"small_cnn needs (C,H,W) input, got {input_shape}; use model='mlp'")
        c, h, w = input_shape
        return SmallCNN(c, num_classes, input_hw=(h, w))
    raise ValueError(f"unknown model type {model_type!r} (use 'mlp' or 'small_cnn')")


# ----------------------------------------------------------- flat-param utils
def get_flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    flat = torch.as_tensor(flat, dtype=torch.float32, device=next(model.parameters()).device)
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset:offset + n].view_as(p))
        offset += n


def get_flat_grad(model: nn.Module) -> torch.Tensor:
    grads = [(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1)
             for p in model.parameters()]
    return torch.cat(grads)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

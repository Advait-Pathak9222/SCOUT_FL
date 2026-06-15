"""Lightweight models for fast per-round federated training + flat-param utils.

Kept deliberately small (this is not an ML-architecture paper): an MLP and an
optional small CNN, selected by config. The flat-parameter helpers support
update-vector extraction (for the learning-utility embeddings) and applying
aggregated updates.
"""
from __future__ import annotations

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
    """2 conv blocks + linear head (small; ~28x28 inputs)."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(model_type: str, input_shape, num_classes: int) -> nn.Module:
    """Construct a model from config. ``input_shape`` = (channels, H, W)."""
    c, h, w = input_shape
    if model_type == "mlp":
        return MLP(c * h * w, num_classes)
    if model_type == "small_cnn":
        return SmallCNN(c, num_classes)
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

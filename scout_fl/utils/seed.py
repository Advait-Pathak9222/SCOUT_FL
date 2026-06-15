"""Reproducible randomness.

All stochastic components take an explicit ``numpy.random.Generator`` produced
here — no reliance on global RNG state inside algorithms, so the same seed
reproduces the same selections and results.
"""
from __future__ import annotations

import os
import random

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return a fresh, isolated NumPy Generator for a given seed."""
    return np.random.default_rng(seed)


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python/NumPy/(optional) Torch globals and return a NumPy Generator."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is optional until the FL step
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - torch not yet a hard dependency
        pass
    return make_rng(seed)

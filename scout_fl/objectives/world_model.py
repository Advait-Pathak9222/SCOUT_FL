"""Lightweight online world model for RECA-FL.

The model is deliberately small: online ridge regression predicts per-client
effects from cheap probe features and exposes residual/calibration diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationReport:
    rmse: float
    ece: float
    n: int


class WorldModel:
    """Multi-output online ridge predictor."""

    def __init__(self, feature_dim: int, output_dim: int = 5, l2: float = 1.0) -> None:
        self.feature_dim = int(feature_dim)
        self.output_dim = int(output_dim)
        self.A = float(l2) * np.eye(self.feature_dim)
        self.B = np.zeros((self.feature_dim, self.output_dim))
        self.W = np.zeros((self.feature_dim, self.output_dim))
        self.pred_norms: list[float] = []
        self.err_norms: list[float] = []

    def predict(self, features) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        return x @ self.W

    def update(self, features, targets) -> np.ndarray:
        x = np.atleast_2d(np.asarray(features, dtype=float))
        y = np.atleast_2d(np.asarray(targets, dtype=float))
        if y.shape[1] != self.output_dim:
            raise ValueError(f"expected output_dim={self.output_dim}, got {y.shape[1]}")
        pred = self.predict(x)
        err = y - pred
        self.pred_norms.extend(np.linalg.norm(pred, axis=1).astype(float).tolist())
        self.err_norms.extend(np.linalg.norm(err, axis=1).astype(float).tolist())
        self.A += x.T @ x
        self.B += x.T @ y
        self.W = np.linalg.solve(self.A, self.B)
        return err

    def residuals(self, features, targets) -> np.ndarray:
        return np.atleast_2d(np.asarray(targets, dtype=float)) - self.predict(features)

    def calibration_report(self, bins: int = 5, window: int | None = None) -> CalibrationReport:
        p = np.asarray(self.pred_norms[-window:] if window else self.pred_norms, dtype=float)
        e = np.asarray(self.err_norms[-window:] if window else self.err_norms, dtype=float)
        if p.size == 0:
            return CalibrationReport(rmse=0.0, ece=0.0, n=0)
        rmse = float(np.sqrt(np.mean(e ** 2)))
        if np.allclose(p.max(), p.min()):
            return CalibrationReport(rmse=rmse, ece=float(abs(e.mean() - p.mean())), n=int(p.size))
        edges = np.linspace(float(p.min()), float(p.max()), int(bins) + 1)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (p >= lo) & (p <= hi if hi == edges[-1] else p < hi)
            if np.any(mask):
                ece += float(mask.mean()) * abs(float(e[mask].mean()) - float(p[mask].mean()))
        return CalibrationReport(rmse=rmse, ece=float(ece), n=int(p.size))


def regime_signature_from_residuals(residuals, embedding=None):
    """Build a compact residual signature for adapter matching."""
    r = np.atleast_2d(np.asarray(residuals, dtype=float))
    if r.size == 0:
        mean = np.zeros(1)
        var = np.ones(1)
    else:
        mean = r.mean(axis=0)
        var = r.var(axis=0) + 1e-6
    emb = mean if embedding is None else np.asarray(embedding, dtype=float)
    half = max(1, mean.size // 2)
    return {
        "embedding": emb,
        "grad_residual": mean[:half],
        "sensing_residual": mean[half:],
        "residual_mean": mean,
        "residual_var": var,
    }

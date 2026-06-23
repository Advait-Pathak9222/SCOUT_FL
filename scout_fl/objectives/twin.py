"""Hybrid counterfactual twin (SCOUT-FL v2 / JEDI-FL): the *learned residual* part.

Most counterfactual marginals (CRB, AirComp MSE, energy, latency) are computed
analytically. The hard-to-model part — the realized learning loss-drop under
AirComp distortion — is predicted by this lightweight ONLINE ridge regressor
(recursive least squares), calibrated each round from observed
(selection-feature -> realized loss-drop) pairs. Cheap, reproducible, and
sharpens the analytical learning term without a heavy neural twin.
"""
from __future__ import annotations

import numpy as np


class ResidualTwin:
    """Online ridge regression (RLS): predict a scalar residual from features."""

    def __init__(self, dim: int, l2: float = 1.0) -> None:
        self.dim = int(dim)
        self.A = float(l2) * np.eye(self.dim)     # regularized Gram
        self.b = np.zeros(self.dim)
        self.w = np.zeros(self.dim)

    def predict(self, x) -> float:
        return float(np.asarray(x, dtype=float) @ self.w)

    def update(self, x, y: float) -> np.ndarray:
        """Add one observation (x -> y) and refresh the weights."""
        x = np.asarray(x, dtype=float)
        self.A += np.outer(x, x)
        self.b += float(y) * x
        self.w = np.linalg.solve(self.A, self.b)
        return self.w

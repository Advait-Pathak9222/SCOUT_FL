"""Constant-velocity (CV) Gaussian random-walk target mobility (design §0.2.1).

State per target: x = [px, py, vx, vy]. Discrete white-noise-acceleration model:

    x_{t+1} = F x_t + w_t,   w_t ~ N(0, Q(sigma_p))

with the standard CV transition/process-noise matrices (dt = 1 round):

    F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
    Q = sigma_p^2 * [[dt^4/4, 0, dt^3/2, 0], [0, dt^4/4, 0, dt^3/2],
                     [dt^3/2, 0, dt^2,   0], [0, dt^3/2, 0, dt^2  ]]

``sigma_p`` is the per-round acceleration std in arena units/round^2; the design
sweep is sigma_p in {0, 0.01, 0.02, 0.05, 0.1, 0.2} with sigma_p = 0.05 the
nominal mobility point. sigma_p = 0 reproduces the stationary campaign exactly.

Arena bounds are handled by reflection (position folded back, velocity sign
flipped). Reflection breaks strict linear-Gaussianity, so the NEES consistency
test (tests/test_infra_tracker.py) runs unbounded; in-arena runs accept the
mild model mismatch (documented, and visible in tracking RMSE vs CRB).
"""
from __future__ import annotations

import numpy as np


def cv_matrices(sigma_p: float, dt: float = 1.0):
    """Return (F, Q) for the CV white-noise-acceleration model."""
    dt = float(dt)
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    q = float(sigma_p) ** 2
    d4, d3, d2 = dt ** 4 / 4.0, dt ** 3 / 2.0, dt ** 2
    Q = q * np.array([[d4, 0, d3, 0],
                      [0, d4, 0, d3],
                      [d3, 0, d2, 0],
                      [0, d3, 0, d2]], dtype=float)
    return F, Q


class CVMobility:
    """CV random-walk motion for M targets, seeded and replayable.

    Also serves as the ground-truth trajectory logger (design §0.2.3): every
    ``step()`` appends the new positions to ``self.trajectory``.
    """

    def __init__(self, targets0: np.ndarray, sigma_p: float, rng: np.random.Generator,
                 area=None, dt: float = 1.0) -> None:
        p0 = np.asarray(targets0, dtype=float)
        if p0.ndim != 2 or p0.shape[1] != 2:
            raise ValueError(f"targets0 must be (M,2), got {p0.shape}")
        self.M = p0.shape[0]
        self.sigma_p = float(sigma_p)
        self.dt = float(dt)
        self.rng = rng
        self.area = None if area is None else np.asarray(area, dtype=float)
        self.F, self.Q = cv_matrices(sigma_p, dt)
        self.state = np.zeros((self.M, 4))
        self.state[:, :2] = p0
        self.trajectory = [p0.copy()]                     # ground-truth log, index = round

    @property
    def positions(self) -> np.ndarray:
        return self.state[:, :2].copy()

    @property
    def velocities(self) -> np.ndarray:
        return self.state[:, 2:].copy()

    def step(self) -> np.ndarray:
        """Advance one round; return the new (M, 2) positions."""
        if self.sigma_p > 0.0:
            accel = self.rng.normal(0.0, self.sigma_p, size=(self.M, 2))
        else:
            accel = np.zeros((self.M, 2))
        # exact WNA sampling: w = [0.5 dt^2 a, dt a] with a ~ N(0, sigma_p^2 I)
        self.state = self.state @ self.F.T
        self.state[:, :2] += 0.5 * self.dt ** 2 * accel
        self.state[:, 2:] += self.dt * accel
        if self.area is not None:
            self._reflect()
        self.trajectory.append(self.positions)
        return self.positions

    def _reflect(self) -> None:
        for d in range(2):
            hi = self.area[d]
            below, above = self.state[:, d] < 0.0, self.state[:, d] > hi
            self.state[below, d] = -self.state[below, d]
            self.state[above, d] = 2.0 * hi - self.state[above, d]
            flip = below | above
            self.state[flip, 2 + d] = -self.state[flip, 2 + d]
            # fold repeatedly drifted states back into range (rare, extreme sigma_p)
            self.state[:, d] = np.clip(self.state[:, d], 0.0, hi)

    def trajectory_array(self) -> np.ndarray:
        """(T+1, M, 2) ground-truth positions logged so far."""
        return np.stack(self.trajectory)

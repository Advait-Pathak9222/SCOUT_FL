"""Per-target Bayesian tracker in information-filter form (design §0.2.2).

Predict with the CV model (F, Q); update by adding the selected set's position
FIM increment J_m(S_t) as measurement information:

    Y_post = Y_pred + H^T J_m(S_t) H          (H picks the position block)

Implementation: J_m(S) is a 2x2 PSD position FIM (possibly rank-deficient when
few/no clients observe target m). We eigendecompose J = sum_i lam_i u_i u_i^T
and apply one scalar Kalman update per eigen-direction with measurement

    z_i = u_i^T p_true + n_i,   n_i ~ N(0, 1 / lam_i)

which is algebraically the information-filter update above and simultaneously
draws a *consistent* noisy measurement, so tracking RMSE (vs the ground-truth
trajectory) and NEES are both measurable. With zero information (empty S or
lam_i <= tol) no update occurs — the posterior honestly widens under Q.

The NEES consistency test (design R4: abort the batch if it fails) lives in
``nees_consistency`` below and runs in preflight (tests/test_infra_tracker.py).
"""
from __future__ import annotations

import numpy as np

from scout_fl.infra.mobility import cv_matrices


class InformationKalmanTracker:
    """M independent 4-state (pos, vel) Kalman filters driven by FIM increments."""

    def __init__(self, init_pos: np.ndarray, sigma_p: float, rng: np.random.Generator,
                 pos_var0: float = 25.0, vel_var0: float = 1.0, dt: float = 1.0) -> None:
        p0 = np.asarray(init_pos, dtype=float)
        self.M = p0.shape[0]
        self.F, self.Q = cv_matrices(sigma_p, dt)
        self.rng = rng
        self.x = np.zeros((self.M, 4))
        self.x[:, :2] = p0                                 # start at the known initial position
        self.P = np.zeros((self.M, 4, 4))
        for m in range(self.M):
            self.P[m] = np.diag([pos_var0, pos_var0, vel_var0, vel_var0])

    # ------------------------------------------------------------ predict/update
    def predict(self) -> None:
        for m in range(self.M):
            self.x[m] = self.F @ self.x[m]
            self.P[m] = self.F @ self.P[m] @ self.F.T + self.Q

    def update(self, J_pos: np.ndarray, true_pos: np.ndarray, tol: float = 1e-9) -> None:
        """Fuse the round's position FIM J_pos (M,2,2) with simulated measurements
        at ``true_pos`` (M,2)."""
        J_pos = np.asarray(J_pos, dtype=float)
        true_pos = np.asarray(true_pos, dtype=float)
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        for m in range(self.M):
            lam, U = np.linalg.eigh(0.5 * (J_pos[m] + J_pos[m].T))
            for i in range(2):
                if lam[i] <= tol:
                    continue
                r = 1.0 / float(lam[i])                    # scalar measurement variance
                h = U[:, i] @ H                            # (4,) measurement row
                z = float(U[:, i] @ true_pos[m]) + self.rng.normal(0.0, np.sqrt(r))
                Ph = self.P[m] @ h
                s = float(h @ Ph) + r
                k_gain = Ph / s
                self.x[m] = self.x[m] + k_gain * (z - float(h @ self.x[m]))
                self.P[m] = self.P[m] - np.outer(k_gain, Ph)
                self.P[m] = 0.5 * (self.P[m] + self.P[m].T)  # keep symmetric

    def predicted_trace_reduction(self, J_pos: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """tr(P) reduction per target if J_pos were fused now (no state mutation).

        The covariance update is measurement-value-independent, so this estimates
        the per-round sensing injection used by the adaptive-threshold controller.
        """
        J_pos = np.asarray(J_pos, dtype=float)
        H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1.0
        out = np.zeros(self.M)
        for m in range(self.M):
            P = self.P[m].copy()
            before = float(np.trace(P[:2, :2]))
            lam, U = np.linalg.eigh(0.5 * (J_pos[m] + J_pos[m].T))
            for i in range(2):
                if lam[i] <= tol:
                    continue
                r = 1.0 / float(lam[i])
                h = U[:, i] @ H
                Ph = P @ h
                s = float(h @ Ph) + r
                P = P - np.outer(Ph, Ph) / s
                P = 0.5 * (P + P.T)
            out[m] = before - float(np.trace(P[:2, :2]))
        return out

    # ------------------------------------------------------------ metrics
    def trace_pos(self) -> np.ndarray:
        """tr(P) over the position block, per target -> (M,)."""
        return np.trace(self.P[:, :2, :2], axis1=-2, axis2=-1)

    def rmse(self, true_pos: np.ndarray) -> np.ndarray:
        """Position error per target -> (M,)."""
        d = self.x[:, :2] - np.asarray(true_pos, dtype=float)
        return np.linalg.norm(d, axis=1)

    def nees(self, true_state: np.ndarray) -> np.ndarray:
        """Normalized estimation error squared per target (dof = 4) -> (M,)."""
        out = np.zeros(self.M)
        for m in range(self.M):
            e = self.x[m] - true_state[m]
            out[m] = float(e @ np.linalg.solve(self.P[m], e))
        return out


def nees_consistency(sigma_p: float = 0.05, steps: int = 120, n_mc: int = 24,
                     fim_scale: float = 0.5, seed: int = 7) -> dict:
    """Monte-Carlo NEES consistency check on synthetic CV trajectories (design R4).

    Runs ``n_mc`` independent single-target tracks with random anisotropic
    position-FIM increments each step, and returns the time-averaged NEES with
    the chi-square acceptance interval for dof=4. All results are computed at a
    fixed seed, so the preflight verdict is deterministic.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    F, Q = cv_matrices(sigma_p)
    nees_t = np.zeros((n_mc, steps))
    for i in range(n_mc):
        truth = np.zeros(4)
        truth[:2] = rng.uniform(20.0, 80.0, size=2)
        truth[2:] = rng.normal(0.0, 0.2, size=2)
        trk = InformationKalmanTracker(truth[None, :2], sigma_p, rng)
        for t in range(steps):
            a = rng.normal(0.0, sigma_p, size=2)
            truth = F @ truth
            truth[:2] += 0.5 * a
            truth[2:] += a
            trk.predict()
            # random-strength, random-orientation rank-2 FIM (like a client subset's)
            th = rng.uniform(0.0, np.pi)
            u = np.array([np.cos(th), np.sin(th)])
            v = np.array([-u[1], u[0]])
            J = (fim_scale * rng.uniform(0.2, 1.0) * np.outer(u, u)
                 + fim_scale * rng.uniform(0.01, 0.2) * np.outer(v, v))
            trk.update(J[None], truth[None, :2])
            nees_t[i, t] = trk.nees(truth[None])[0]

    dof = 4
    # drop the short transient while P converges from the prior
    burn = min(10, steps // 4)
    avg = float(nees_t[:, burn:].mean())
    n_eff = n_mc                                          # conservative: rounds are correlated
    lo = float(stats.chi2.ppf(0.025, dof * n_eff)) / n_eff
    hi = float(stats.chi2.ppf(0.975, dof * n_eff)) / n_eff
    return {"avg_nees": avg, "dof": dof, "lo": lo, "hi": hi,
            "n_mc": n_mc, "steps": steps, "sigma_p": sigma_p,
            "consistent": bool(lo <= avg <= hi)}

"""Time-varying mixed learning/sensing utility fed to the SCOUT-FL greedy selector.

    U_t(S) = w_learn * f_learn(S)/norm_L
             + sum_m ws_m * [logdet(J0_m + sum_{k in S} J_{k,m}) - logdet(J0_m)] / norm_m

Both terms are monotone submodular for PSD FIMs and nonnegative weights, so the
existing lazy/greedy selector and its (1-1/e) guarantee apply unchanged. Per-term
(and per-target) normalization makes the controller weights scale-comparable:
a scalar lambda_s enters as w_learn = 1-lambda_s, ws_m = lambda_s/M; the DPP queue
weights enter as ws_m = Z_m/V. Exposes the incremental API (init_state/add/
marginal_gain) that ScoutGreedy consumes.
"""
from __future__ import annotations

import numpy as np

from scout_fl.sim.crb import logdet_spd


class MixedUtility:
    def __init__(self, learning, fim: np.ndarray, prior_fim: np.ndarray,
                 w_learn: float, w_sense: np.ndarray, K: int) -> None:
        self.learning = learning
        self.J = np.asarray(fim, dtype=float)              # (K, M, d, d)
        self.J0 = np.asarray(prior_fim, dtype=float)       # (M, d, d)
        self.M = self.J.shape[1]
        self.K = int(K)
        self.w_learn = float(w_learn)
        self.ws = np.asarray(w_sense, dtype=float)         # (M,)
        self._logdet_J0 = logdet_spd(self.J0)              # (M,)
        # per-term normalizers from the full set (scale each contribution to ~1)
        full = list(range(K))
        self.norm_L = max(float(learning.value(full)), 1e-9)
        acc_full = self.J0 + self.J.sum(axis=0)
        self.norm_m = np.maximum(logdet_spd(acc_full) - self._logdet_J0, 1e-9)  # (M,)

    # --------------------------------------------------------- set function
    def value(self, subset) -> float:
        idx = list(subset)
        v = self.w_learn * self.learning.value(idx) / self.norm_L
        acc = self.J0 + (self.J[idx].sum(axis=0) if idx else 0.0)
        gain = (logdet_spd(acc) - self._logdet_J0) / self.norm_m
        return float(v + (self.ws * gain).sum())

    # --------------------------------------------------------- incremental
    def init_state(self):
        return {"learn": self.learning.init_state(),
                "fim": np.array(self.J0, dtype=float, copy=True)}

    def add(self, state, k: int):
        return {"learn": self.learning.add(state["learn"], k),
                "fim": state["fim"] + self.J[k]}

    def marginal_gain(self, state, k: int) -> float:
        lg = self.w_learn * self.learning.marginal_gain(state["learn"], k) / self.norm_L
        cur = logdet_spd(state["fim"])
        new = logdet_spd(state["fim"] + self.J[k])
        sg = (self.ws * (new - cur) / self.norm_m).sum()
        return float(lg + sg)

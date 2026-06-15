"""Sensing utility f_sense: weighted log-det Fisher-information gain (D-optimal).

    f_sense(S) = sum_m w_m [ logdet(J_0,m + sum_{k in S} J_{k,m}) - logdet(J_0,m) ]

This is monotone non-decreasing and submodular for PSD ``J_{k,m}`` (matrix
concavity of log-det; sums preserve submodularity) — the property that gives
the greedy selector its (1 - 1/e) guarantee.

The class exposes both a plain set-function API (``value``/``crb``/``rmse``)
and an incremental, stateful API (``init_state``/``add``/``marginal_gain``) so
lazy-greedy (CELF) avoids recomputing accumulated FIMs from scratch.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from scout_fl.sim.crb import crb_trace, logdet_spd


class SensingUtility:
    """log-det FIM utility over a fixed per-round client/target FIM cache."""

    def __init__(self, fim_cache: np.ndarray, prior_fim: np.ndarray,
                 target_weights: Iterable[float] | None = None) -> None:
        self.J = np.asarray(fim_cache, dtype=float)        # (K, M, d, d)
        self.J0 = np.asarray(prior_fim, dtype=float)       # (M, d, d)
        if self.J.ndim != 4:
            raise ValueError(f"fim_cache must be (K,M,d,d), got {self.J.shape}")
        self.K, self.M = self.J.shape[:2]
        self.dim = self.J.shape[-1]
        if target_weights is None:
            self.w = np.ones(self.M)
        else:
            self.w = np.asarray(target_weights, dtype=float)
            if self.w.shape != (self.M,):
                raise ValueError("target_weights must have length M")
        self._logdet_J0 = logdet_spd(self.J0)              # (M,)

    # ----------------------------------------------------------- set-function
    def accumulated(self, subset: Iterable[int]) -> np.ndarray:
        """Accumulated FIM per target for a client subset -> (M, d, d)."""
        acc = np.array(self.J0, dtype=float, copy=True)
        idx = list(subset)
        if idx:
            acc = acc + self.J[idx].sum(axis=0)
        return acc

    def value(self, subset: Iterable[int]) -> float:
        """f_sense(subset)."""
        acc = self.accumulated(subset)
        return float((self.w * (logdet_spd(acc) - self._logdet_J0)).sum())

    def crb(self, subset: Iterable[int], reg: float = 0.0) -> np.ndarray:
        """Per-target CRB (trace of inverse accumulated FIM) -> (M,)."""
        return crb_trace(self.accumulated(subset), reg=reg)

    def rmse(self, subset: Iterable[int], reg: float = 0.0) -> np.ndarray:
        """Per-target localization RMSE lower-bound proxy = sqrt(CRB) -> (M,)."""
        return np.sqrt(np.clip(self.crb(subset, reg=reg), 0.0, None))

    # --------------------------------------------------- incremental (greedy)
    def init_state(self) -> np.ndarray:
        """Initial accumulated-FIM state (= prior) for incremental selection."""
        return np.array(self.J0, dtype=float, copy=True)

    def add(self, state: np.ndarray, k: int) -> np.ndarray:
        """Return new state after adding client ``k`` to the accumulated FIM."""
        return state + self.J[k]

    def marginal_gain(self, state: np.ndarray, k: int) -> float:
        """f_sense gain of adding client ``k`` to the current accumulated state."""
        new = state + self.J[k]
        return float((self.w * (logdet_spd(new) - logdet_spd(state))).sum())

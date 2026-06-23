"""Online / bandit client selection for the P7 regret result.

When per-client sensing quality (SNR -> Fisher information) is UNKNOWN and must be learned
from noisy per-round observations, we use a Combinatorial-UCB (CUCB) selector (Chen-Wang-Yuan
ICML'13 / JMLR'16) with the greedy (1-1/e)-oracle: maintain an empirical mean + pull count per
client, form an optimistic upper-confidence SNR, build the (optimistic) Fisher information, and
run the same greedy log-det selection. Semi-bandit feedback: the true SNR of each SELECTED client
is observed (noisily) and its estimate updated. Against the offline greedy oracle on the true
utility this attains sublinear alpha-regret ~ O~(sqrt(N K T)) (alpha = 1 - 1/e).
"""
from __future__ import annotations

import numpy as np


class CUCBSensingSelector:
    """Combinatorial-UCB over per-client sensing SNR (unknown means, semi-bandit feedback)."""

    def __init__(self, K: int, ucb_c: float = 1.0) -> None:
        self.K = int(K)
        self.c = float(ucb_c)
        self.mean = np.zeros(self.K)      # empirical mean SNR estimate
        self.count = np.zeros(self.K)     # pull counts
        self.t = 0

    def ucb_snr(self) -> np.ndarray:
        """Optimistic UCB index per client; never-pulled clients are explored first."""
        self.t += 1
        bonus = self.c * np.sqrt(np.maximum(np.log(self.t + 1.0), 0.0) / np.maximum(self.count, 1e-9))
        ucb = self.mean + bonus
        return np.where(self.count < 1, 1e9, ucb)        # force initial exploration of each arm

    def update(self, selected, observed_snr) -> None:
        """Semi-bandit update: refresh the mean SNR estimate of each selected client."""
        obs = np.asarray(observed_snr, dtype=float)
        for k in selected:
            self.count[k] += 1.0
            self.mean[k] += (obs[k] - self.mean[k]) / self.count[k]

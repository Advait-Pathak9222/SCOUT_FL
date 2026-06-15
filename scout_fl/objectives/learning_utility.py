"""Learning utility f_learn: DivFL-style facility location over client gradient
(or representation) embeddings.

    f_learn(S) = sum_{j in [K]} max_{k in S} sim(j, k),     sim >= 0

This rewards a *representative, diverse* selected set (covering the gradient
landscape), and is monotone submodular for nonnegative similarities — so it
composes with the sensing/coverage/fairness terms while preserving the greedy
(1 - 1/e) guarantee.

Milestone status: the embedding matrix is supplied externally. In the no-FL
loop it is a spatial-viewpoint surrogate; at the FL step (Step 7) it becomes the
actual per-client gradient/feature embeddings.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


class LearningUtility:
    """Facility-location submodular utility over a fixed similarity matrix."""

    def __init__(self, embeddings: np.ndarray | None = None, *,
                 similarity: np.ndarray | None = None, sigma: float | None = None) -> None:
        if similarity is not None:
            self.S = np.asarray(similarity, dtype=float)
        elif embeddings is not None:
            G = np.asarray(embeddings, dtype=float)             # (K, d)
            d2 = ((G[:, None, :] - G[None, :, :]) ** 2).sum(-1)  # (K, K)
            if sigma is None:
                upper = d2[np.triu_indices(G.shape[0], k=1)]
                med = float(np.median(upper)) if upper.size else 1.0
                sigma = np.sqrt(med) if med > 0 else 1.0
            self.S = np.exp(-d2 / (2.0 * float(sigma) ** 2))    # (K, K) in (0, 1]
        else:
            raise ValueError("provide either embeddings or a similarity matrix")
        self.K = self.S.shape[0]

    # set-function ----------------------------------------------------------
    def value(self, subset: Iterable[int]) -> float:
        idx = list(subset)
        if not idx:
            return 0.0
        return float(self.S[:, idx].max(axis=1).sum())

    # incremental (lazy-greedy) --------------------------------------------
    def init_state(self) -> np.ndarray:
        return np.zeros(self.K)                                  # best sim per j so far

    def add(self, state: np.ndarray, k: int) -> np.ndarray:
        return np.maximum(state, self.S[:, k])

    def marginal_gain(self, state: np.ndarray, k: int) -> float:
        return float(np.maximum(self.S[:, k] - state, 0.0).sum())

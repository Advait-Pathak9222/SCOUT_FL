"""Loss-based client selection baseline: pick the top-budget clients by local
loss (a power-of-choice-style heuristic). Geometry/sensing-agnostic."""
from __future__ import annotations

import time

import numpy as np

from scout_fl.selection.base import Selector, SelectionResult


class LossSelector(Selector):
    name = "loss"

    def select(self, scores, budget: int, **_) -> SelectionResult:
        start = time.perf_counter()
        scores = np.asarray(scores, dtype=float)
        order = np.argsort(-scores, kind="stable")[:budget]
        return SelectionResult(
            selected=sorted(int(k) for k in order),
            select_time=time.perf_counter() - start,
        )

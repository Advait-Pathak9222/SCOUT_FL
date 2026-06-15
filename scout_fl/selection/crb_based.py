"""CRB-only baseline: greedily add the client that most reduces aggregate CRB
(A-optimal greedy).

A-optimality (trace of inverse FIM) is only *weakly* submodular, so this greedy
lacks the clean (1 - 1/e) guarantee of the log-det objective — but it is a
strong, geometry-aware baseline and a useful contrast for SCOUT-FL's D-optimal
selector.
"""
from __future__ import annotations

import time

from scout_fl.selection.base import Selector, SelectionResult


class CRBSelector(Selector):
    name = "crb_only"

    def select(self, utility, num_clients: int, budget: int, **_) -> SelectionResult:
        start = time.perf_counter()
        chosen: list[int] = []
        remaining = set(range(num_clients))

        def aggregate_crb(subset) -> float:
            return float((utility.w * utility.crb(subset)).sum())

        while remaining and len(chosen) < budget:
            best_k = min(remaining, key=lambda c: aggregate_crb(chosen + [c]))
            chosen.append(best_k)
            remaining.discard(best_k)
        return SelectionResult(
            selected=sorted(chosen),
            select_time=time.perf_counter() - start,
        )

"""Random client selection baseline (uniform without replacement)."""
from __future__ import annotations

import time

import numpy as np

from scout_fl.selection.base import Selector, SelectionResult


class RandomSelector(Selector):
    name = "random"

    def select(self, num_clients: int, budget: int,
               rng: np.random.Generator, **_) -> SelectionResult:
        start = time.perf_counter()
        size = min(budget, num_clients)
        chosen = rng.choice(num_clients, size=size, replace=False)
        return SelectionResult(
            selected=sorted(int(k) for k in chosen),
            select_time=time.perf_counter() - start,
        )

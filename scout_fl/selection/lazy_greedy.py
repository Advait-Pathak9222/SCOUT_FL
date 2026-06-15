"""CELF lazy-greedy for monotone submodular maximization under a cardinality
budget (Minoux 1978; Leskovec et al. 2007).

Exploiting submodularity, a popped element whose cached marginal gain is still
the heap maximum after a freshness check is optimal to add — so most marginal
gains are never recomputed. Complexity is close to O(budget * K) marginal
evaluations in the worst case but typically far fewer, which is what keeps
per-round selection cheap as K grows.

The ``utility`` must expose the incremental API:
``init_state()``, ``add(state, k) -> state``, ``marginal_gain(state, k) -> float``.
"""
from __future__ import annotations

import heapq
from typing import Iterable


def lazy_greedy(utility, num_clients: int, budget: int,
                candidates: Iterable[int] | None = None):
    """Return ``(selected, final_state, n_marginal_evals)``."""
    cand = list(range(num_clients)) if candidates is None else list(candidates)
    state = utility.init_state()

    heap = []
    evals = 0
    for k in cand:
        gain = utility.marginal_gain(state, k)
        evals += 1
        heap.append((-gain, k, 0))  # (neg gain, client, freshness generation)
    heapq.heapify(heap)

    selected: list[int] = []
    generation = 0
    while heap and len(selected) < budget:
        neg_gain, k, gen = heapq.heappop(heap)
        if gen == generation:
            selected.append(k)
            state = utility.add(state, k)
            generation += 1
        else:
            fresh = utility.marginal_gain(state, k)
            evals += 1
            heapq.heappush(heap, (-fresh, k, generation))
    return selected, state, evals

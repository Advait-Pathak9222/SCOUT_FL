"""SCOUT-FL (A1) selector: greedy maximization of the (composite) utility under a
cardinality budget, with an optional hard-constraint feasibility filter.

* Unconstrained -> lazy-greedy (CELF), with the (1 - 1/e) guarantee.
* With a ``feasible(selected, k)`` predicate -> constraint-integrated greedy that
  only adds feasible candidates (e.g. AirComp MSE / latency / energy / CRB).
  If no candidate is feasible at a step, it relaxes and LOGS the relaxation
  (``relaxed_steps``) rather than silently violating the constraint.

``utility`` (sensing-only ``SensingUtility`` or the composite ``TotalUtility``)
only needs the incremental API: ``init_state``/``add``/``marginal_gain``.
"""
from __future__ import annotations

import time
from typing import Callable

from scout_fl.selection.base import Selector, SelectionResult
from scout_fl.selection.lazy_greedy import lazy_greedy


def naive_greedy(utility, num_clients: int, budget: int, candidates=None):
    """Plain greedy (re-evaluates every candidate each step). Used for tests."""
    remaining = set(range(num_clients)) if candidates is None else set(candidates)
    state = utility.init_state()
    selected, evals = [], 0
    while remaining and len(selected) < budget:
        best_k, best_gain = None, float("-inf")
        for k in remaining:
            gain = utility.marginal_gain(state, k)
            evals += 1
            if gain > best_gain:
                best_k, best_gain = k, gain
        selected.append(best_k)
        state = utility.add(state, best_k)
        remaining.discard(best_k)
    return selected, state, evals


def constrained_greedy(utility, num_clients: int, budget: int,
                       feasible: Callable[[list, int], bool] | None = None,
                       relax: bool = True, candidates=None):
    """Greedy under a feasibility predicate; returns ``(selected, state, evals, relaxed)``."""
    remaining = set(range(num_clients)) if candidates is None else set(candidates)
    state = utility.init_state()
    selected, evals, relaxed = [], 0, 0
    while remaining and len(selected) < budget:
        feas = [k for k in remaining if (feasible is None or feasible(selected, k))]
        if feas:
            pool = feas
        elif relax:
            pool, relaxed = list(remaining), relaxed + 1   # infeasible step -> relax + log
        else:
            break
        best_k, best_gain = None, float("-inf")
        for k in pool:
            gain = utility.marginal_gain(state, k)
            evals += 1
            if gain > best_gain:
                best_k, best_gain = k, gain
        selected.append(best_k)
        state = utility.add(state, best_k)
        remaining.discard(best_k)
    return selected, state, evals, relaxed


class ScoutGreedy(Selector):
    name = "scout_greedy"

    def __init__(self, use_lazy: bool = True) -> None:
        self.use_lazy = use_lazy

    def select(self, utility, num_clients: int, budget: int,
               feasible: Callable[[list, int], bool] | None = None,
               candidates=None, **_) -> SelectionResult:
        start = time.perf_counter()
        relaxed = 0
        if feasible is not None:
            sel, _, evals, relaxed = constrained_greedy(
                utility, num_clients, budget, feasible=feasible, candidates=candidates)
        elif self.use_lazy:
            sel, _, evals = lazy_greedy(utility, num_clients, budget, candidates)
        else:
            sel, _, evals = naive_greedy(utility, num_clients, budget, candidates)
        return SelectionResult(
            selected=sorted(int(k) for k in sel),
            select_time=time.perf_counter() - start,
            info={"marginal_evals": evals,
                  "lazy": self.use_lazy and feasible is None,
                  "relaxed_steps": relaxed},
        )

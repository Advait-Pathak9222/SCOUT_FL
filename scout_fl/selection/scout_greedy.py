"""SCOUT-FL selector: greedy maximization of the (composite) utility under a
cardinality budget, with optional hard-constraint feasibility (v1) OR soft
primal-dual penalties (v2).

* Unconstrained        -> lazy-greedy (CELF), (1 - 1/e) guarantee.
* feasible(S, k)        -> v1 hard-gate constraint-integrated greedy (+ relax-and-log).
* penalty_fn(S, k)      -> v2 soft primal-dual penalty: score = marginal - penalty
                           (no hard exclusion; fixes the v1 CRB-inversion bug).

``utility`` (SensingUtility or composite TotalUtility) only needs the incremental
API: init_state / add / marginal_gain.
"""
from __future__ import annotations

import time
from typing import Callable

from scout_fl.selection.base import Selector, SelectionResult
from scout_fl.selection.lazy_greedy import lazy_greedy


def naive_greedy(utility, num_clients: int, budget: int, candidates=None):
    """Plain greedy (re-evaluates every candidate each step)."""
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
    """v1 hard-gate greedy; returns (selected, state, evals, relaxed)."""
    remaining = set(range(num_clients)) if candidates is None else set(candidates)
    state = utility.init_state()
    selected, evals, relaxed = [], 0, 0
    while remaining and len(selected) < budget:
        feas = [k for k in remaining if (feasible is None or feasible(selected, k))]
        if feas:
            pool = feas
        elif relax:
            pool, relaxed = list(remaining), relaxed + 1
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


def penalized_greedy(utility, num_clients: int, budget: int,
                     penalty_fn: Callable[[list, int], float], candidates=None):
    """v2 soft primal-dual greedy: maximize (marginal_gain - penalty_fn(S, k))."""
    remaining = set(range(num_clients)) if candidates is None else set(candidates)
    state = utility.init_state()
    selected, evals = [], 0
    while remaining and len(selected) < budget:
        best_k, best_score = None, float("-inf")
        for k in remaining:
            score = utility.marginal_gain(state, k) - float(penalty_fn(selected, k))
            evals += 1
            if score > best_score:
                best_k, best_score = k, score
        selected.append(best_k)
        state = utility.add(state, best_k)
        remaining.discard(best_k)
    return selected, state, evals


class ScoutGreedy(Selector):
    name = "scout_greedy"

    def __init__(self, use_lazy: bool = True) -> None:
        self.use_lazy = use_lazy

    def select(self, utility, num_clients: int, budget: int,
               feasible: Callable[[list, int], bool] | None = None,
               penalty_fn: Callable[[list, int], float] | None = None,
               candidates=None, **_) -> SelectionResult:
        start = time.perf_counter()
        relaxed = 0
        if penalty_fn is not None:                       # v2 soft primal-dual
            sel, _, evals = penalized_greedy(utility, num_clients, budget, penalty_fn, candidates)
        elif feasible is not None:                       # v1 hard gate
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
                  "lazy": self.use_lazy and feasible is None and penalty_fn is None,
                  "relaxed_steps": relaxed},
        )

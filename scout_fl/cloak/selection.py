"""M1 — leakage-capped selection (design §2.4.1) + privacy-baseline selectors.

M1 = SCOUT greedy maximizing the (composite) utility subject to |S| <= K, the
AirComp-MSE budget (soft primal-dual penalty), AND a per-client cumulative
leakage cap J_k^leak(1..t) <= J_max (knapsack-like feasibility). A client is
infeasible this round if selecting it would push its cumulative leakage-FIM trace
over J_max; if no client is feasible the gate relaxes (and logs it) so a round is
never empty. Caps force participation rotation (measured as a Jain-fairness side
effect, design §2.4.1).
"""
from __future__ import annotations

import math

import numpy as np


def _safe(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("-inf")
    return v if math.isfinite(v) else float("-inf")


def leakage_capped_greedy(utility, num_clients, budget, *, accountant, snr_up,
                          atten=1.0, r_floor=None, mse_penalty_fn=None):
    """Greedy max of ``utility`` with a per-client leakage cap + soft MSE penalty.

    ``accountant`` is a live LeakageAccountant (its cumulative J is read, not
    mutated, here). ``r_floor`` None -> uncapped (pure utility); otherwise a client
    is infeasible this round if selecting it would drop its position CRB floor below
    ``r_floor`` meters (the exact, anisotropy-correct privacy guarantee). Returns
    (selected, n_relaxed_steps).
    """
    atten = np.broadcast_to(np.asarray(atten, dtype=float), (num_clients,))
    remaining = set(range(num_clients))
    state = utility.init_state()
    selected, relaxed = [], 0

    def feasible(k):
        if r_floor is None:
            return True
        return accountant.projected_crb_floor(k, snr_up[k], atten[k]) >= r_floor

    while remaining and len(selected) < budget:
        pool = [k for k in remaining if feasible(k)]
        if not pool:                                        # relax-and-log (never empty round)
            pool, relaxed = list(remaining), relaxed + 1
        best_k, best_score = None, float("-inf")
        for k in pool:
            gain = _safe(utility.marginal_gain(state, k))
            pen = _safe(mse_penalty_fn(selected, k)) if mse_penalty_fn else 0.0
            score = float("-inf") if not math.isfinite(gain + pen) else gain - pen
            if score > best_score:
                best_k, best_score = k, score
        if best_k is None:
            best_k = min(pool)
        selected.append(best_k)
        state = utility.add(state, best_k)
        remaining.discard(best_k)
    return sorted(int(k) for k in selected), relaxed


def random_selection(num_clients, budget, rng):
    return sorted(int(k) for k in rng.choice(num_clients, size=budget, replace=False))

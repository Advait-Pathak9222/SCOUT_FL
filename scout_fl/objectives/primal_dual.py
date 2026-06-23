"""Primal-dual adaptive constraints (SCOUT-FL v2) — replaces the hard AirComp-MSE gate.

Each soft constraint c (aggregation MSE, energy, latency, ...) gets a dual
variable mu_c >= 0. During selection, a candidate's marginal score is reduced by
sum_c mu_c * cost-violation; after the round, the duals ascend on the realized
violation:

    mu_c  <-  [ mu_c + lr_c * (cost_c(S_t) - limit_c) ]_+

So constraint-violating clients are *softly* deprioritized in proportion to the
current violation pressure — no client is hard-excluded. This fixes SCOUT-FL v1's
CRB-inversion, where the hard MSE gate dropped sensing-strong but comm-weak
clients. Standard dual-ascent gives bounded long-run violation (feasibility on
average) without fixed hand-tuned weights.
"""
from __future__ import annotations

import numpy as np


class DualState:
    """Dual variables for a set of soft (constraint) costs."""

    def __init__(self, limits: dict, lr=0.5, init: float = 0.0) -> None:
        # only active constraints (non-None limit) get a dual
        self.limits = {k: float(v) for k, v in limits.items() if v is not None}
        self.lr = {k: (float(lr) if np.isscalar(lr) else float(lr.get(k, 0.5))) for k in self.limits}
        self.mu = {k: float(init) for k in self.limits}

    def penalty(self, costs: dict) -> float:
        """sum_c mu_c * cost_c  (caller passes the per-constraint cost/violation)."""
        return float(sum(self.mu[k] * float(costs.get(k, 0.0)) for k in self.mu))

    def violation_penalty(self, abs_costs: dict) -> float:
        """sum_c mu_c * max(0, cost_c - limit_c)  (penalize exceeding the limit)."""
        return float(sum(self.mu[k] * max(0.0, float(abs_costs.get(k, 0.0)) - self.limits[k])
                         for k in self.mu))

    def update(self, realized: dict) -> dict:
        """Dual ascent on realized constraint values; returns the current duals."""
        for k in self.mu:
            viol = float(realized.get(k, 0.0)) - self.limits[k]
            self.mu[k] = max(0.0, self.mu[k] + self.lr[k] * viol)
        return dict(self.mu)


class ParticipationDual:
    """Per-client participation-fairness dual (Lyapunov virtual queue) for JEDI-FL.

    Long-term fairness as a constraint, not a tuned weight: each round every client
    should be selected with probability ``target = budget / K``. A per-client deficit
    queue grows when a client is skipped and drains when it is selected:

        q_k  <-  [ q_k + lr * (target - 1[k in S_t]) ]_+

    Chronically under-served clients accumulate deficit, so a deficit-proportional
    selection bonus (auto-scaled to information units in JointInformationUtility)
    eventually forces their selection — the drift-plus-penalty guarantee. This
    replaces JEDI's fixed log-age prior, which was too weak against the large
    information terms (Jain index trailed SCOUT-FL v2). Weight-free: ``target`` is
    set by the problem (budget/K) and the bonus scale is data-driven, not tuned.
    """

    def __init__(self, K: int, budget: int, lr: float = 1.0) -> None:
        self.K = int(K)
        self.target = float(budget) / float(K)
        self.lr = float(lr)
        self.q = np.zeros(self.K)

    @property
    def deficit(self) -> np.ndarray:
        return self.q

    def update(self, selected) -> np.ndarray:
        """Drift step on the realized selection; returns the updated deficit queue."""
        sel = np.zeros(self.K)
        idx = np.asarray(list(selected), dtype=int)
        if idx.size:
            sel[idx] = 1.0
        self.q = np.maximum(0.0, self.q + self.lr * (self.target - sel))
        return self.q

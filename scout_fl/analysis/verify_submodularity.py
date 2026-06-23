"""Numerically verify monotonicity + submodularity of a set-function.

Submodularity (diminishing returns): for ``A subseteq B`` and ``x not in B``,
    f(A + x) - f(A)  >=  f(B + x) - f(B).
Monotonicity: every marginal gain ``f(A + x) - f(A) >= 0``.

This is a guardrail: if these fail for ``f_sense`` (beyond numerical tolerance),
a modeling/implementation bug exists and the greedy guarantee does not apply.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def verify_submodular(value_fn: Callable[[set], float], ground_set: Sequence[int],
                      n_samples: int, rng: np.random.Generator,
                      tol: float = 1e-9) -> dict:
    """Sample ``(A subseteq B, x)`` triples and check the two properties."""
    gs = list(ground_set)
    n = len(gs)
    submod_violations = 0
    mono_violations = 0
    max_violation = 0.0

    for _ in range(int(n_samples)):
        perm = list(rng.permutation(gs))
        a, b = sorted(int(x) for x in rng.integers(0, n + 1, size=2))
        set_a = set(perm[:a])
        set_b = set(perm[:b])                  # A subseteq B by construction
        rest = [x for x in gs if x not in set_b]
        if not rest:
            continue
        x = int(rng.choice(rest))
        gain_a = value_fn(set_a | {x}) - value_fn(set_a)
        gain_b = value_fn(set_b | {x}) - value_fn(set_b)
        if gain_a + tol < gain_b:              # diminishing returns violated
            submod_violations += 1
            max_violation = max(max_violation, gain_b - gain_a)
        if gain_a < -tol:                      # negative marginal gain
            mono_violations += 1

    return {
        "samples": int(n_samples),
        "submodularity_violations": submod_violations,
        "submodularity_violation_rate": submod_violations / max(int(n_samples), 1),
        "max_violation": float(max_violation),
        "monotonicity_violations": mono_violations,
        "is_submodular": submod_violations == 0,
        "is_monotone": mono_violations == 0,
    }


def submodularity_ratio(value_fn: Callable[[set], float], ground_set: Sequence[int],
                        n_samples: int, rng: np.random.Generator) -> dict:
    """Estimate the submodularity ratio gamma (Das & Kempe 2011) of a monotone function:

        gamma = min over (S, R) of  [ sum_{x in R} (f(S+x) - f(S)) ] / [ f(S u R) - f(S) ].

    gamma >= 1 <=> submodular (greedy keeps 1 - 1/e); 0 < gamma < 1 => weakly submodular,
    greedy keeps the weaker 1 - e^{-gamma}. This quantifies how far JEDI-FL's COUPLED
    joint-EIG objective (the kappa(MSE) shared-MAC term breaks exact submodularity) is from
    submodular — and thus what greedy guarantee survives. Run it with use_kappa=False to
    confirm the DECOUPLED objective is submodular (gamma ~ 1), and with use_kappa=True for
    the coupled ratio.
    """
    gs = list(ground_set)
    n = len(gs)
    ratios = []
    for _ in range(int(n_samples)):
        perm = list(rng.permutation(gs))
        s_size = int(rng.integers(0, max(1, n // 2)))
        set_s = set(perm[:s_size])
        rest = [x for x in gs if x not in set_s]
        if not rest:
            continue
        r_size = int(rng.integers(1, len(rest) + 1))
        set_r = set(int(x) for x in rng.choice(rest, size=r_size, replace=False))
        f_s = value_fn(set_s)
        sum_marg = sum(value_fn(set_s | {x}) - f_s for x in set_r)
        joint = value_fn(set_s | set_r) - f_s
        if joint > 1e-12:
            ratios.append(sum_marg / joint)
    if not ratios:
        return {"samples": int(n_samples), "gamma_min": float("nan"), "gamma_mean": float("nan")}
    ratios = np.asarray(ratios)
    return {
        "samples": int(n_samples),
        "gamma_min": float(ratios.min()),               # the guaranteeing ratio (greedy: 1 - e^{-gamma})
        "gamma_mean": float(ratios.mean()),
        "gamma_clipped": float(min(1.0, ratios.min())),
        "approx_submodular": bool(ratios.min() >= 1.0 - 1e-6),
    }

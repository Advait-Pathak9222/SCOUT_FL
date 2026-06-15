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

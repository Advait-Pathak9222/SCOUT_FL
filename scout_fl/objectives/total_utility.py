"""Composite SCOUT-FL utility: a nonnegative weighted sum of the term utilities.

    U(S) = sum_term  (w_term / norm_term) * f_term(S)

Because each term (learning, sensing, coverage, fairness) is monotone
submodular and the weights are nonnegative, ``U`` is monotone submodular too —
so the same lazy-greedy selector and the (1 - 1/e) guarantee apply unchanged.

Optional per-term normalizers make the weights scale-comparable (the plan's
"normalize utilities + sensitivity sweep" stance for any soft weight that
survives the constrained formulation).

Exposes the same incremental API as the individual utilities, so
``ScoutGreedy``/``lazy_greedy`` consume it without modification.
"""
from __future__ import annotations

from typing import Iterable, Mapping


class TotalUtility:
    def __init__(self, terms: Mapping[str, object],
                 weights: Mapping[str, float] | None = None,
                 normalizers: Mapping[str, float] | None = None) -> None:
        self.terms = dict(terms)
        self.w = {name: 1.0 for name in self.terms}
        if weights:
            self.w.update(weights)
        self.norm = {name: 1.0 for name in self.terms}
        if normalizers:
            self.norm.update({k: (v if abs(v) > 1e-12 else 1.0) for k, v in normalizers.items()})

    def _coeff(self, name: str) -> float:
        return float(self.w[name]) / float(self.norm[name])

    # set-function ----------------------------------------------------------
    def value(self, subset: Iterable[int]) -> float:
        subset = list(subset)
        return float(sum(self._coeff(n) * t.value(subset) for n, t in self.terms.items()))

    def components(self, subset: Iterable[int]) -> dict[str, float]:
        subset = list(subset)
        return {n: float(t.value(subset)) for n, t in self.terms.items()}

    # incremental (lazy-greedy) --------------------------------------------
    def init_state(self) -> dict:
        return {n: t.init_state() for n, t in self.terms.items()}

    def add(self, state: dict, k: int) -> dict:
        return {n: t.add(state[n], k) for n, t in self.terms.items()}

    def marginal_gain(self, state: dict, k: int) -> float:
        return float(sum(self._coeff(n) * t.marginal_gain(state[n], k)
                         for n, t in self.terms.items()))

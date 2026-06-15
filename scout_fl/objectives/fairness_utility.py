"""Fairness utility f_fair: reward clients under-served recently.

Per-round term:  f_fair(S) = sum_{k in S} phi(age_k),  phi nondecreasing, phi(0)=0,
where ``age_k`` is rounds since client ``k`` was last selected. This is modular
(hence monotone submodular) and acts as an age/staleness bonus that prevents the
sensing/learning terms from permanently starving low-value-but-fair clients.

``update`` ages the counters between rounds (reset selected -> 0, others +1).
A region-level fairness analogue (saturating coverage of under-served regions)
is the documented extension.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


class FairnessUtility:
    _PHI = {
        "log": lambda a: np.log1p(a),
        "linear": lambda a: a.astype(float),
    }

    def __init__(self, num_clients: int, phi: str = "log") -> None:
        self.K = int(num_clients)
        self.age = np.zeros(self.K)
        if phi not in self._PHI:
            raise ValueError(f"unknown phi={phi!r}; choose from {list(self._PHI)}")
        self.phi = self._PHI[phi]

    # set-function (modular) ------------------------------------------------
    def value(self, subset: Iterable[int]) -> float:
        idx = list(subset)
        if not idx:
            return 0.0
        return float(self.phi(self.age[idx]).sum())

    def init_state(self):
        return None                              # modular: gain independent of state

    def add(self, state, k: int):
        return None

    def marginal_gain(self, state, k: int) -> float:
        return float(self.phi(np.asarray([self.age[k]]))[0])

    # dynamics --------------------------------------------------------------
    def update(self, selected: Iterable[int]) -> np.ndarray:
        mask = np.ones(self.K, dtype=bool)
        idx = list(selected)
        if idx:
            mask[idx] = False
        self.age = np.where(mask, self.age + 1.0, 0.0)
        return self.age

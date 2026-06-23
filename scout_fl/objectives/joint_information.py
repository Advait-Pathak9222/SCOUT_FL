"""JEDI-FL: joint experimental-design / expected-information-gain selection.

ONE information objective (no fixed per-objective weights), maximized greedily:

    U(S) = [ sensing-info(S) + coverage/freshness-info(S) + fairness-prior(S) ]
           + rho * kappa(MSE_agg(S)) * learning-info(S)

* sensing + coverage are log-det / saturating information terms (D-optimal), with
  the dynamic coverage map acting as the sensing prior (freshness);
* learning-info is the gradient-diversity (facility-location) information, scaled
  by kappa(MSE) = 1/(1 + AirComp aggregation MSE): **AirComp MSE enters as
  OBSERVATION NOISE** — a weak-comm client raises MSE, shrinks kappa, and so
  self-deprioritizes for learning. No hard gate, no MSE weight (fixes SCOUT v1);
* fairness/freshness of clients enters through a primal-dual participation
  DEFICIT (a Lyapunov virtual queue, see ParticipationDual) auto-scaled to
  information units — under-served clients carry more "prior uncertainty" and are
  eventually forced in. Adaptive and weight-free; replaces the old fixed log-age
  prior, which was too weak against the large information terms;
* rho is a per-dimension information NORMALIZER (auto-set so the learning block
  is commensurate with sensing+coverage), NOT a preference weight;
* kappa couples ALL selected clients through the shared MAC, so the marginal
  value is externality-aware (Shapley/Oort assume additivity).

Physically: AirComp MSE corrupts the aggregated GRADIENT (learning), not the
radar echo (sensing) — hence MSE scales only the learning block.

Greedy is run with the NAIVE evaluator (not lazy/CELF) because the kappa coupling
makes the objective only approximately submodular.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from scout_fl.sim.aircomp import aggregation_mse


class JointInformationUtility:
    """The flags below exist for the ablation study (component knockouts):
    ``use_sensing``/``use_learning``/``use_coverage`` drop a block; ``use_kappa=False``
    removes the MSE-as-observation-noise coupling (kappa=1); ``externality=False`` makes
    the marginal additive (per-client kappa, Shapley-style); ``rho=1.0`` disables the
    auto-normalizer (fixed scalarization). Defaults reproduce full JEDI-FL.
    """

    def __init__(self, sensing, coverage, learning, fair_deficit, channel_gains, *,
                 power: float = 1.0, sigma2: float = 1.0, rho: float | None = None,
                 fair_scale: float = 1.0, use_sensing: bool = True, use_learning: bool = True,
                 use_coverage: bool = True, use_kappa: bool = True, externality: bool = True,
                 learn_mult=None) -> None:
        self.sensing = sensing
        self.coverage = coverage
        self.learning = learning
        self.g = np.asarray(channel_gains, dtype=float)
        self.P = float(power)
        self.s2 = float(sigma2)
        self.fair_scale = float(fair_scale)
        self.use_sensing, self.use_learning, self.use_coverage = use_sensing, use_learning, use_coverage
        self.use_kappa, self.externality = use_kappa, externality
        # optional per-client learning multiplier from the trust-gated residual twin;
        # affects only the SELECTION marginal (not value()) so a bad twin cannot corrupt
        # the clean information measure. Multiplier in [1-trust, 1+trust] => trust=0 -> 1.0.
        self.learn_mult = None if learn_mult is None else np.asarray(learn_mult, dtype=float)
        self.K = sensing.K
        # participation deficit (Lyapunov virtual queue); zeros => fairness inactive
        self.deficit = (np.zeros(self.K) if fair_deficit is None
                        else np.asarray(fair_deficit, dtype=float))
        full = list(range(self.K))
        sense_full = (sensing.value(full) if use_sensing else 0.0) + \
                     (coverage.value(full) if use_coverage else 0.0)
        learn_full = max(learning.value(full), 1e-9)
        # auto-normalizer: make learning-nats commensurate with sensing+coverage-nats
        self.rho = float(rho) if rho is not None else float(max(sense_full, 1e-9) / learn_full)
        # data-driven fairness scale: one unit of deficit ~ the average per-client
        # sensing+coverage information (keeps fairness weight-free, in information units)
        self.fair_unit = self.fair_scale * float(max(sense_full, 1e-9)) / max(self.K, 1)

    def _kappa(self, subset) -> float:
        if not self.use_kappa:                                # ablation: MSE not observation noise
            return 1.0
        mse = aggregation_mse(self.g, subset, power=self.P, sigma2=self.s2)
        return 1.0 / (1.0 + mse)

    # ----------------------------------------------------------- set-function
    def value(self, subset: Iterable[int]) -> float:
        S = list(subset)
        if not S:
            return 0.0
        v = self.fair_unit * float(self.deficit[S].sum())
        if self.use_sensing:
            v += self.sensing.value(S)
        if self.use_coverage:
            v += self.coverage.value(S)
        if self.use_learning:
            v += self.rho * self._kappa(S) * self.learning.value(S)
        return float(v)

    # --------------------------------------------------- incremental (greedy)
    def init_state(self) -> dict:
        return {"sense": self.sensing.init_state(), "cov": self.coverage.init_state(),
                "learn": self.learning.init_state(), "S": [], "Lval": 0.0}

    def add(self, state: dict, k: int) -> dict:
        # carry L(S) = learning.value(S) incrementally so the marginal can telescope value()
        L_new = state["Lval"] + self.learning.marginal_gain(state["learn"], k)
        return {"sense": self.sensing.add(state["sense"], k),
                "cov": self.coverage.add(state["cov"], k),
                "learn": self.learning.add(state["learn"], k),
                "S": state["S"] + [k], "Lval": L_new}

    def marginal_gain(self, state: dict, k: int) -> float:
        g_total = self.fair_unit * float(self.deficit[k])      # participation-deficit bonus
        if self.use_sensing:
            g_total += self.sensing.marginal_gain(state["sense"], k)
        if self.use_coverage:
            g_total += self.coverage.marginal_gain(state["cov"], k)
        if self.use_learning:
            dL = self.learning.marginal_gain(state["learn"], k)
            if self.learn_mult is not None:                    # trust-gated twin residual (selection only)
                dL *= float(self.learn_mult[k])
            if self.externality:
                # EXACT telescoping of value(): adding k rescales the whole accumulated
                # learning mass by the new kappa (shared-MAC externality), so a weak-channel
                # client that lowers kappa is penalized for degrading EVERYONE's learning info.
                L_S = state["Lval"]
                old = self.rho * self._kappa(state["S"]) * L_S if state["S"] else 0.0
                g_total += self.rho * self._kappa(state["S"] + [k]) * (L_S + dL) - old
            else:
                # additive (Shapley-style) ablation: per-client kappa on the increment only
                g_total += self.rho * self._kappa([k]) * dL
        return float(g_total)

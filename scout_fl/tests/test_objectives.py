"""Tests for the composite objectives layer (learning/coverage/fairness/total).

Run:  pytest scout_fl/tests -q
"""
from __future__ import annotations

import numpy as np

from scout_fl.analysis.verify_submodularity import verify_submodular
from scout_fl.objectives.constraints import Constraints
from scout_fl.objectives.coverage_utility import (CoverageMap, CoverageUtility,
                                                  contribution_matrix, region_centers)
from scout_fl.objectives.fairness_utility import FairnessUtility
from scout_fl.objectives.learning_utility import LearningUtility
from scout_fl.objectives.sensing_utility import SensingUtility
from scout_fl.objectives.total_utility import TotalUtility
from scout_fl.selection.scout_greedy import ScoutGreedy, naive_greedy
from scout_fl.selection.lazy_greedy import lazy_greedy
from scout_fl.sim.fim import db_to_linear, per_client_target_fim, prior_fim
from scout_fl.sim.geometry import pairwise_geometry


# ----------------------------------------------------------------- learning
def test_learning_submodular_and_monotone():
    rng = np.random.default_rng(0)
    util = LearningUtility(embeddings=rng.normal(size=(15, 6)))
    rep = verify_submodular(util.value, range(util.K), 2000, np.random.default_rng(1))
    assert rep["is_submodular"] and rep["is_monotone"]


def test_learning_incremental_matches_value():
    rng = np.random.default_rng(0)
    util = LearningUtility(embeddings=rng.normal(size=(12, 5)))
    subset = [1, 4, 9]
    state = util.init_state()
    for k in subset:
        state = util.add(state, k)
    assert abs(util.value(subset) - float(state.sum())) < 1e-9


# ----------------------------------------------------------------- coverage
def _coverage(seed=0, K=12, R=9):
    rng = np.random.default_rng(seed)
    clients = rng.uniform(0, 100, size=(K, 2))
    centers = region_centers([100, 100], R)
    C = contribution_matrix(clients, centers, sensing_range=30.0)
    return CoverageUtility(rng.uniform(0.5, 1.5, size=R), C)


def test_coverage_submodular_and_monotone():
    cov = _coverage()
    rep = verify_submodular(cov.value, range(cov.K), 2000, np.random.default_rng(2))
    assert rep["is_submodular"] and rep["is_monotone"]


def test_coverage_map_dynamics():
    R = 9
    C = np.zeros((3, R)); C[0, 0] = 1.0          # client 0 fully covers region 0
    cmap = CoverageMap(R, rho=0.9, innovation=0.05, u_init=1.0)
    cmap.update([0], C)
    assert cmap.U[0] == 0.0                       # sensed region: 0.9*1+0.05-1 = -0.05 -> clipped
    assert abs(cmap.U[1] - (0.9 * 1.0 + 0.05)) < 1e-9   # unsensed region persists/ages


# ----------------------------------------------------------------- fairness
def test_fairness_modular_and_aging():
    fair = FairnessUtility(5)
    fair.age = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert abs(fair.marginal_gain(None, 4) - np.log1p(4)) < 1e-9
    fair.update([0, 1])
    assert fair.age[0] == 0.0 and fair.age[1] == 0.0   # selected reset
    assert fair.age[2] == 3.0                          # unselected incremented (2 -> 3)


# -------------------------------------------------------------------- total
def _composite(seed=3, K=12, M=2):
    rng = np.random.default_rng(seed)
    clients = rng.uniform(0, 100, (K, 2)); targets = rng.uniform(0, 100, (M, 2))
    geom = pairwise_geometry(clients, targets)
    fim = per_client_target_fim(geom, db_to_linear(rng.uniform(0, 20, K)), 1.0, 0.05)
    sensing = SensingUtility(fim, prior_fim(M, 1e-3))
    learning = LearningUtility(embeddings=rng.normal(size=(K, 6)))
    centers = region_centers([100, 100], 9)
    coverage = CoverageUtility(rng.uniform(0.5, 1.5, 9),
                               contribution_matrix(clients, centers, 30.0))
    fair = FairnessUtility(K); fair.age = rng.uniform(0, 5, K)
    total = TotalUtility(
        {"learning": learning, "sensing": sensing, "coverage": coverage, "fairness": fair},
        weights={"learning": 1.0, "sensing": 1.0, "coverage": 1.0, "fairness": 1.0})
    return total, K


def test_total_utility_submodular():
    total, K = _composite()
    rep = verify_submodular(total.value, range(K), 2500, np.random.default_rng(4))
    assert rep["is_submodular"] and rep["is_monotone"]


def test_total_utility_greedy_budget_and_lazy_equivalence():
    total, K = _composite()
    res = ScoutGreedy().select(utility=total, num_clients=K, budget=5)
    assert len(res.selected) == 5 and len(set(res.selected)) == 5
    lazy_sel, _, _ = lazy_greedy(total, K, 5)
    naive_sel, _, _ = naive_greedy(total, K, 5)
    assert lazy_sel == naive_sel


# ----------------------------------------------------------------- constraints
def test_constraints_crb_feasibility():
    c = Constraints(crb_max=[1.0, 1.0])
    assert c.evaluate(crb=[0.5, 0.8])["feasible"]
    assert not c.evaluate(crb=[0.5, 2.0])["feasible"]
    assert c.evaluate(crb=None)["feasible"]        # inactive when no value provided

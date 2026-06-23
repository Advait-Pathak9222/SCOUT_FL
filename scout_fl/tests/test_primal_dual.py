"""Tests for SCOUT-FL v2 primitives: primal-dual duals, penalized greedy, twin.

Run:  pytest scout_fl/tests -q
"""
from __future__ import annotations

import numpy as np

from scout_fl.objectives.primal_dual import DualState
from scout_fl.objectives.twin import ResidualTwin
from scout_fl.objectives.sensing_utility import SensingUtility
from scout_fl.selection.scout_greedy import penalized_greedy, naive_greedy
from scout_fl.sim.fim import db_to_linear, per_client_target_fim, prior_fim
from scout_fl.sim.geometry import pairwise_geometry


def _sensing_util(seed=0, K=10, M=2):
    rng = np.random.default_rng(seed)
    geom = pairwise_geometry(rng.uniform(0, 100, (K, 2)), rng.uniform(0, 100, (M, 2)))
    fim = per_client_target_fim(geom, db_to_linear(rng.uniform(0, 20, K)), 1.0, 0.05)
    return SensingUtility(fim, prior_fim(M, 1e-3))


# ------------------------------------------------------------------- duals
def test_dual_only_active_constraints():
    d = DualState({"mse": 0.1, "energy": None, "latency": None})
    assert set(d.mu) == {"mse"}


def test_dual_ascends_on_violation_and_clips():
    d = DualState({"mse": 0.1}, lr=1.0)
    d.update({"mse": 0.3})                  # violation +0.2 -> mu = 0.2
    assert abs(d.mu["mse"] - 0.2) < 1e-9
    d.update({"mse": 0.0})                  # slack -0.1 -> mu = max(0, 0.2-0.1)=0.1
    assert abs(d.mu["mse"] - 0.1) < 1e-9
    for _ in range(20):
        d.update({"mse": 0.0})              # persistent slack -> clamps at 0
    assert d.mu["mse"] == 0.0


def test_violation_penalty():
    d = DualState({"mse": 0.1}); d.mu["mse"] = 2.0
    assert abs(d.violation_penalty({"mse": 0.25}) - 2.0 * 0.15) < 1e-9
    assert d.violation_penalty({"mse": 0.05}) == 0.0     # within limit -> no penalty


# --------------------------------------------------------- penalized greedy
def test_penalized_greedy_zero_penalty_matches_naive():
    util = _sensing_util()
    sel_p, _, _ = penalized_greedy(util, util.K, 4, penalty_fn=lambda S, k: 0.0)
    sel_n, _, _ = naive_greedy(util, util.K, 4)
    assert sel_p == sel_n


def test_penalized_greedy_excludes_heavily_penalized_client():
    util = _sensing_util()
    banned = 3
    pen = lambda S, k: (1e6 if k == banned else 0.0)
    sel, _, _ = penalized_greedy(util, util.K, 4, penalty_fn=pen)
    assert banned not in sel and len(sel) == 4


# -------------------------------------------------------------------- twin
def test_residual_twin_learns_linear_map():
    rng = np.random.default_rng(0)
    w_true = np.array([1.5, -2.0, 0.5])
    twin = ResidualTwin(dim=3, l2=1e-3)
    for _ in range(200):
        x = rng.normal(size=3)
        twin.update(x, float(x @ w_true))
    assert np.allclose(twin.w, w_true, atol=1e-2)

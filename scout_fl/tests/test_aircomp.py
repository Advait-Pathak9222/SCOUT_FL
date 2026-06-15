"""Tests for the AirComp / channel / energy modules and constraint-aware greedy.

Run:  pytest scout_fl/tests -q
"""
from __future__ import annotations

import numpy as np

from scout_fl.objectives.sensing_utility import SensingUtility
from scout_fl.selection.scout_greedy import constrained_greedy
from scout_fl.sim.aircomp import aggregation_mse, aircomp_eta, min_gain_for_mse
from scout_fl.sim.channel import comm_channel_gains
from scout_fl.sim.energy_latency import round_energy_latency
from scout_fl.sim.fim import db_to_linear, per_client_target_fim, prior_fim
from scout_fl.sim.geometry import pairwise_geometry


def test_more_clients_reduce_mse_equal_gains():
    g = np.ones(6)
    assert aggregation_mse(g, [0, 1, 2, 3]) < aggregation_mse(g, [0, 1])   # n^2 in denom


def test_weak_channel_inflates_mse():
    g = np.array([1.0, 1.0, 0.01])
    assert aggregation_mse(g, [0, 1, 2]) > aggregation_mse(g, [0, 1])      # min g drops


def test_more_power_reduces_mse():
    g = np.array([0.5, 0.8, 1.0])
    hi = aggregation_mse(g, [0, 1, 2], power=1.0)
    lo = aggregation_mse(g, [0, 1, 2], power=0.25)
    assert hi < lo
    assert aircomp_eta(g, [0, 1, 2], power=1.0) == 0.5


def test_min_gain_for_mse_meets_target():
    eps, budget = 0.1, 5
    g_min = min_gain_for_mse(eps, budget, power=1.0, sigma2=1.0)
    g = np.full(budget, g_min)
    assert aggregation_mse(g, list(range(budget))) <= eps + 1e-9


def test_constrained_greedy_respects_gate_and_budget():
    rng = np.random.default_rng(0)
    K, M = 12, 2
    clients = rng.uniform(0, 100, (K, 2)); targets = rng.uniform(0, 100, (M, 2))
    geom = pairwise_geometry(clients, targets)
    fim = per_client_target_fim(geom, db_to_linear(rng.uniform(0, 20, K)), 1.0, 0.05)
    util = SensingUtility(fim, prior_fim(M, 1e-3))
    g = rng.uniform(0.0, 1.0, K)
    thr = 0.5
    sel, _, _, relaxed = constrained_greedy(util, K, 4, feasible=lambda S, k: g[k] >= thr)
    assert len(sel) == 4
    if int(np.sum(g >= thr)) >= 4:
        assert relaxed == 0 and all(g[k] >= thr for k in sel)


def test_relax_logs_when_too_few_feasible():
    rng = np.random.default_rng(1)
    fim = per_client_target_fim(
        pairwise_geometry(rng.uniform(0, 100, (6, 2)), rng.uniform(0, 100, (1, 2))),
        db_to_linear(rng.uniform(0, 20, 6)), 1.0, 0.05)
    util = SensingUtility(fim, prior_fim(1, 1e-3))
    sel, _, _, relaxed = constrained_greedy(util, 6, 4, feasible=lambda S, k: k == 0)  # only 1 feasible
    assert len(sel) == 4 and relaxed > 0      # relaxes (and logs) the rest


def test_channel_gains_positive():
    rng = np.random.default_rng(2)
    g = comm_channel_gains(rng.uniform(0, 100, (10, 2)), [50, 50], rng)
    assert g.shape == (10,) and np.all(g > 0)


def test_energy_latency_nonnegative():
    rng = np.random.default_rng(3)
    el = round_energy_latency([0, 1, 2], rng.uniform(0.1, 1.0, 8))
    assert el["latency"] >= 0 and el["energy"] >= 0
    assert round_energy_latency([], np.ones(8))["latency"] == 0.0

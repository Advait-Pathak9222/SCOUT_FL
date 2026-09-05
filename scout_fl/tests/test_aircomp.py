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


def test_mse_scales_with_power_in_physical_units():
    """Regression: the MSE must stay exactly inversely proportional to P even when
    eta = P * g_min is far below 1 (physical link budget: eta ~ 1e-14..1e-10 W).

    A fixed numerical floor on eta (previously max(eta, 1e-12)) capped the aggregation
    MSE at sigma2/(|S|^2 * floor), which made the AirComp error independent of transmit
    power across the whole low-power half of the campaign sweep.
    """
    from scout_fl.sim.link_budget import dbm_to_watt, thermal_noise_power_w
    sigma2 = thermal_noise_power_w(1e6, 7.0)          # ~2.0e-14 W
    g = np.full(10, 2.5e-7)                            # physical path gain (~-66 dB)
    sel = list(range(10))
    ref = aggregation_mse(g, sel, power=dbm_to_watt(0.0), sigma2=sigma2)
    for tx_dbm in (-10.0, -20.0, -30.0, -35.0):
        got = aggregation_mse(g, sel, power=dbm_to_watt(tx_dbm), sigma2=sigma2)
        assert np.isclose(got, ref * 10.0 ** (-tx_dbm / 10.0), rtol=1e-9), \
            f"MSE not inversely proportional to P at {tx_dbm} dBm"


def test_zero_gain_gives_infinite_mse():
    """A client with no link makes the channel-inversion aggregate undefined."""
    g = np.array([1.0, 0.5, 0.0])
    assert aggregation_mse(g, [0, 1]) < float("inf")
    assert aggregation_mse(g, [0, 1, 2]) == float("inf")


def test_interference_enters_only_through_the_noise_floor():
    """I and sigma^2 are interchangeable: only their sum reaches the aggregation error."""
    g = np.full(8, 0.25)
    sel = list(range(8))
    a = aggregation_mse(g, sel, power=1.0, sigma2=2.0, interference=3.0)
    b = aggregation_mse(g, sel, power=1.0, sigma2=5.0, interference=0.0)
    assert np.isclose(a, b), "interference must be indistinguishable from thermal noise"
    # a rise of Delta dB in the floor is a tightening of the budget by the same Delta dB
    base = aggregation_mse(g, sel, power=1.0, sigma2=1.0)
    for delta_db in (3.0, 6.0, 10.0):
        lifted = 10.0 ** (delta_db / 10.0)
        got = aggregation_mse(g, sel, power=1.0, sigma2=1.0, interference=lifted - 1.0)
        assert np.isclose(got, base * lifted, rtol=1e-9)


def test_per_client_power_budget_binds_at_the_weakest_link():
    """rho* = min_k |h_k| sqrt(P_k / pi): the binding client is the one minimising g_k P_k."""
    g = np.array([1.0, 1.0, 1.0, 1.0])
    equal = aggregation_mse(g, [0, 1, 2, 3], power=1.0, sigma2=1.0)
    het = aggregation_mse(g, [0, 1, 2, 3], power=np.array([1.0, 1.0, 1.0, 0.25]), sigma2=1.0)
    assert np.isclose(het, 4.0 * equal), "the smallest budget must set the denoising factor"
    # a weak channel and a small power budget are interchangeable through their product
    swap = aggregation_mse(np.array([1.0, 1.0, 1.0, 0.25]), [0, 1, 2, 3], power=1.0, sigma2=1.0)
    assert np.isclose(swap, het)


def test_update_power_scales_the_error():
    """Higher per-entry update power costs headroom under a fixed transmit budget."""
    g = np.full(4, 1.0)
    ref = aggregation_mse(g, [0, 1, 2, 3], power=1.0, sigma2=1.0, update_power=1.0)
    for pi in (0.01, 0.5, 4.0):
        got = aggregation_mse(g, [0, 1, 2, 3], power=1.0, sigma2=1.0, update_power=pi)
        assert np.isclose(got, ref * pi, rtol=1e-9)


def test_min_gain_threshold_inverts_the_error_formula():
    """The v1 gate threshold must be the exact inverse of the error expression."""
    for eps in (1e-2, 1e-3, 1e-4):
        for I in (0.0, 1.0):
            gmin = min_gain_for_mse(eps, budget=10, power=2.0, sigma2=1.0, interference=I)
            got = aggregation_mse(np.full(10, gmin), list(range(10)), power=2.0,
                                  sigma2=1.0, interference=I)
            assert np.isclose(got, eps, rtol=1e-9)

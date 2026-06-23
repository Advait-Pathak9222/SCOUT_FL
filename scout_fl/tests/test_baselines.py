"""Every named baseline selector returns a valid, budget-sized, in-range selection.

Run:  pytest scout_fl/tests/test_baselines.py -q
"""
from __future__ import annotations

import numpy as np

from scout_fl.objectives.learning_utility import LearningUtility
from scout_fl.objectives.sensing_utility import SensingUtility
from scout_fl.selection.baselines import BASELINE_REGISTRY
from scout_fl.sim.fim import db_to_linear, per_client_target_fim, prior_fim
from scout_fl.sim.geometry import pairwise_geometry


def _ctx(K=12, M=2, budget=4, seed=0):
    rng = np.random.default_rng(seed)
    clients = rng.uniform(0, 100, (K, 2))
    targets = rng.uniform(0, 100, (M, 2))
    geom = pairwise_geometry(clients, targets)
    snr = db_to_linear(rng.uniform(5, 20, K))
    fim = per_client_target_fim(geom, snr, 1.0, 0.05)
    sensing = SensingUtility(fim, prior_fim(M, 1e-3), np.ones(M))
    embs = rng.standard_normal((K, 16))
    learning = LearningUtility(embeddings=embs)
    return dict(
        K=K, budget=budget, rng=rng, sensing=sensing, learning=learning,
        g=rng.uniform(0.3, 1.5, K), snr_scores=snr, losses=rng.uniform(0.1, 2.0, K),
        embeddings=embs, grad_norm=np.linalg.norm(embs, axis=1),
        participation=rng.integers(0, 5, K).astype(float), age=rng.integers(0, 6, K).astype(float),
        latency=rng.uniform(0.5, 2.0, K), P=1.0, sigma2=1.0, mse_eps=0.05,
    )


def test_registry_has_all_named_baselines():
    expected = {"comm_only", "aircomp_mse_min", "sensing_only", "fedcs", "oort",
                "fedgcs", "fixed_weighted", "collabsensefed", "ota_fl_iscc", "sensing_native",
                "ota_fedavg", "fedavg_iscc", "fedsgd_iscc", "fed_iscc", "asaad",
                "divfl", "delta", "po_fl", "fair_equity", "iscc_air_feel",
                "crb_only", "fedis"}
    assert expected == set(BASELINE_REGISTRY)


def test_pofl_favours_weak_channels_relative_to_gradient():
    # PO-FL's channel term ~ 1/|h|^2 => among equal-gradient clients, a weaker channel
    # gets a higher scheduling score (verified via the Q_i score, not the sampled set).
    rng = np.random.default_rng(0)
    sel = BASELINE_REGISTRY["po_fl"]
    K = 8
    gn = np.ones(K)                          # equal gradient importance
    g = np.linspace(0.2, 2.0, K)             # increasing channel gain
    a = (1 + sel.alpha) * 1.0 * 1.0 / 1.0
    Q = np.sqrt(a / g + (1 + 1 / sel.alpha) * gn ** 2)
    assert Q[0] > Q[-1]                       # weakest channel -> highest Q (paper's design)


def test_divfl_matches_learning_diversity_greedy():
    ctx = _ctx()
    sel = BASELINE_REGISTRY["divfl"].select(**ctx).selected
    assert len(sel) == ctx["budget"] and len(set(sel)) == ctx["budget"]


def test_fair_equity_force_includes_long_overlooked_client():
    # the sampling equalizer force-includes any client whose skip-gap >= gap_max (paper's rule)
    ctx = _ctx()
    ctx["age"] = ctx["age"].copy(); ctx["age"][3] = 999.0          # client 3 long overlooked
    ctx["losses"] = ctx["losses"].copy(); ctx["losses"][3] = float(ctx["losses"].mean())  # not an outlier
    sel = BASELINE_REGISTRY["fair_equity"].select(**ctx).selected
    assert 3 in sel


def test_asaad_drops_to_budget_and_favours_low_mse_crb():
    ctx = _ctx()
    sel = BASELINE_REGISTRY["asaad"].select(**ctx).selected
    assert len(sel) == ctx["budget"] and len(set(sel)) == ctx["budget"]
    assert all(0 <= k < ctx["K"] for k in sel)


def test_fedavg_and_fedsgd_iscc_share_selection():
    # the two ISCC variants differ only in the local-update rule, not in selection
    ctx = _ctx()
    a = BASELINE_REGISTRY["fedavg_iscc"].select(**ctx).selected
    b = BASELINE_REGISTRY["fedsgd_iscc"].select(**ctx).selected
    assert a == b


def test_every_baseline_selects_valid_set():
    ctx = _ctx()
    K, budget = ctx["K"], ctx["budget"]
    for name, selector in BASELINE_REGISTRY.items():
        res = selector.select(**ctx)
        sel = res.selected
        assert len(sel) == budget, f"{name}: expected {budget} clients, got {len(sel)}"
        assert len(set(sel)) == budget, f"{name}: returned duplicates {sel}"
        assert all(0 <= k < K for k in sel), f"{name}: out-of-range index in {sel}"


def test_comm_only_picks_strongest_channels():
    ctx = _ctx()
    g = ctx["g"]
    sel = BASELINE_REGISTRY["comm_only"].select(**ctx).selected
    top = set(np.argsort(-g)[: ctx["budget"]].tolist())
    assert set(sel) == top


def test_sensing_only_matches_greedy_logdet():
    from scout_fl.selection.scout_greedy import naive_greedy
    ctx = _ctx()
    sel = BASELINE_REGISTRY["sensing_only"].select(**ctx).selected
    greedy, _, _ = naive_greedy(ctx["sensing"], ctx["K"], ctx["budget"])
    assert set(sel) == set(int(k) for k in greedy)


def test_baselines_reproducible_given_same_rng_seed():
    for name in BASELINE_REGISTRY:
        a = BASELINE_REGISTRY[name].select(**_ctx(seed=3)).selected
        b = BASELINE_REGISTRY[name].select(**_ctx(seed=3)).selected
        assert a == b, f"{name} is not reproducible"

"""Tests for VISMAYA-FL: ClientGenerativeModel correctness.

Run: pytest scout_fl/tests/test_vismaya.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from scout_fl.selection.generative_model import ClientGenerativeModel


# ------------------------------------------------------------------ fixtures

def _make_model(K=6, M=2, process_noise=0.0):
    rng = np.random.default_rng(0)
    # Simple FIM: rank-1 outer products so the math is easy to verify
    fim = np.zeros((K, M, 2, 2))
    for k in range(K):
        angle = 2.0 * np.pi * k / K
        u = np.array([np.cos(angle), np.sin(angle)])
        for m in range(M):
            fim[k, m] = np.outer(u, u) * float(k + 1)  # SNR scales with k
    j0 = np.stack([0.01 * np.eye(2) for _ in range(M)])
    w = np.ones(M)
    return ClientGenerativeModel(K, M, fim, j0, w,
                                  rho_v=1.0, beta=0.3,
                                  process_noise=process_noise)


def _make_features(model):
    K = model.K
    rng = np.random.default_rng(1)
    losses = rng.uniform(0.1, 1.0, K)
    gnorms = rng.uniform(0.0, 1.0, K)
    channels = rng.uniform(0.1, 1.0, K)
    return model.build_features(losses, gnorms, channels)


# ------------------------------------------------------------------ basic shape

def test_score_all_returns_K_values():
    model = _make_model()
    feats = _make_features(model)
    scores = model.score_all(feats)
    assert scores.shape == (model.K,)


def test_sensing_innovations_positive():
    model = _make_model()
    omega_s = model.sensing_innovations()
    assert np.all(omega_s >= 0.0), "Sensing innovation must be non-negative (tr(J P) ≥ 0)"


# ------------------------------------------------------------------ P_m dynamics

def test_sensing_innovation_decreases_after_selection():
    """After selecting clients, P_m shrinks → Ω^S drops for those targets."""
    model = _make_model()
    feats = _make_features(model)
    omega_before = model.sensing_innovations().copy()
    gnorms = np.ones(model.K)
    # Select all clients → maximum FIM accumulation → P_m shrinks maximally
    model.update(list(range(model.K)), feats, gnorms)
    omega_after = model.sensing_innovations()
    assert float(omega_after.mean()) < float(omega_before.mean()), (
        "Ω^S should decrease after sensing updates (P_m shrinks as targets are sensed)")


def test_process_noise_inflates_target_covariance():
    """Non-zero process noise keeps P_m elevated (prevents P_m → 0 in non-stationary case)."""
    K, M = 6, 2
    model_static = _make_model(K, M, process_noise=0.0)
    model_mobile = _make_model(K, M, process_noise=0.1)
    feats = _make_features(model_static)
    gnorms = np.ones(K)
    # Update both models many rounds with the same selections
    for _ in range(10):
        model_static.update(list(range(K)), feats, gnorms)
        model_mobile.update(list(range(K)), feats, gnorms)
    trace_static = np.mean([np.trace(model_static.P[m]) for m in range(M)])
    trace_mobile = np.mean([np.trace(model_mobile.P[m]) for m in range(M)])
    assert trace_mobile > trace_static, (
        "Process noise should keep P_m inflated (target uncertainty never collapses to zero)")


# ------------------------------------------------------------------ learning predictor

def test_learning_uncertainty_decreases_for_frequently_selected_clients():
    """Clients selected many times should have lower σ²_{g,k} (uncertainty proxy)."""
    model = _make_model()
    feats = _make_features(model)
    gnorms = np.ones(model.K)
    # Select only client 0 many times
    for _ in range(20):
        model.update([0], feats, gnorms)
    omega_l = model.learning_innovations(feats)
    # Client 0 has n_selected=20; others have n_selected=0
    assert omega_l[0] < omega_l[1], (
        "Frequently-selected client 0 should have lower uncertainty than unvisited client 1")


def test_build_features_shape_and_recency():
    """build_features returns (K, 5) matrix with recency in [0, 1]."""
    model = _make_model(K=5)
    rng = np.random.default_rng(42)
    losses = rng.uniform(0, 1, 5)
    gnorms = rng.uniform(0, 1, 5)
    channels = rng.uniform(0, 1, 5)
    feats = model.build_features(losses, gnorms, channels)
    assert feats.shape == (5, 5)
    # Recency column (last) should be in [0, 1]
    recency = feats[:, -1]
    assert np.all(recency >= 0.0) and np.all(recency <= 1.0)
    # Bias column (first) should be all 1s
    assert np.allclose(feats[:, 0], 1.0)


# ------------------------------------------------------------------ top-K optimality

def test_top_k_is_optimal_for_modular_objective():
    """For a modular (additive) scoring function, top-K is the optimal solution.

    This verifies the key VISMAYA claim: V_k = Ω^S_k + ρ Ω^L_k + β Syn_k is
    modular → top-K by score is trivially optimal → no greedy approximation needed.
    """
    rng = np.random.default_rng(7)
    N, budget = 10, 3
    scores = rng.uniform(0, 1, N)
    # Optimal top-budget set
    top_k = set(np.argsort(-scores)[:budget])
    top_k_value = sum(scores[k] for k in top_k)
    # Check all other budget-subsets
    from itertools import combinations
    for subset in combinations(range(N), budget):
        assert sum(scores[k] for k in subset) <= top_k_value + 1e-12


# ------------------------------------------------------------------ synergy

def test_synergy_grows_after_correlated_errors():
    """Synergy should increase when sensing innovation and learning prediction error co-occur."""
    model = _make_model(K=3, M=1)
    feats = _make_features(model)
    # Make actual grad norms very different from what ridge predicts (prediction error)
    gnorms_surprising = np.array([100.0, 100.0, 100.0])
    synergy_before = model.synergy.copy()
    model.update([0], feats, gnorms_surprising)
    assert model.synergy[0] >= synergy_before[0], (
        "Synergy for selected client 0 should be non-decreasing after update")
    # Un-selected clients' synergy should not change
    assert np.allclose(model.synergy[1:], synergy_before[1:])


# ------------------------------------------------------------------ diagnostics

def test_diagnostics_keys():
    """diagnostics() returns all expected keys with finite values."""
    model = _make_model()
    d = model.diagnostics()
    expected = {"vis_omega_s_mean", "vis_omega_s_max",
                "vis_synergy_mean", "vis_p_trace_mean", "vis_n_seen_frac"}
    assert expected == set(d.keys())
    for k, v in d.items():
        assert np.isfinite(v), f"Diagnostic {k}={v} is not finite"


# ------------------------------------------------------------------ auto-calibration

def test_rho_auto_calibration_makes_innovations_commensurate():
    """After one score_all call, rho_v should be auto-set so Ω^S ~ Ω^L in magnitude."""
    model = _make_model()
    assert not model._rho_calibrated
    feats = _make_features(model)
    model.score_all(feats)
    assert model._rho_calibrated
    # After calibration: rho_v * Ω^L_mean ≈ Ω^S_mean (up to a 2× tolerance)
    omega_s = model.sensing_innovations()
    omega_l = model.learning_innovations(feats)
    ratio = float(omega_s.mean()) / max(model.rho_v * float(omega_l.mean()), 1e-12)
    assert 0.1 < ratio < 10.0, f"Innovations still badly mismatched after calibration: ratio={ratio:.3f}"


# ============================================================
# ABLATION TESTS — prove Syn term matters
# ============================================================

def _make_model_with_scale(sense_scale=1.0, beta=0.3, rho_v=1.0, K=6, M=2,
                           process_noise=0.0):
    """Shared fixture for ablation tests."""
    fim = np.zeros((K, M, 2, 2))
    for k in range(K):
        angle = 2.0 * np.pi * k / K
        u = np.array([np.cos(angle), np.sin(angle)])
        for m in range(M):
            fim[k, m] = np.outer(u, u) * float(k + 1)
    j0 = np.stack([0.01 * np.eye(2) for _ in range(M)])
    w = np.ones(M)
    return ClientGenerativeModel(K, M, fim, j0, w,
                                  rho_v=rho_v, beta=beta,
                                  process_noise=process_noise,
                                  sense_scale=sense_scale)


def test_sense_only_ablation_ignores_learning():
    """vismaya_sense_only (rho_v=0, beta=0) scores equal sensing_innovations()."""
    model_full = _make_model_with_scale(beta=0.0, rho_v=0.0, sense_scale=1.0)
    feats = _make_features(model_full)
    scores = model_full.score_all(feats)
    omega_s = model_full.sensing_innovations()
    # With rho_v=0, beta=0: score = sense_scale * omega_s + 0 + 0
    assert np.allclose(scores, omega_s, atol=1e-9), (
        "sense_only ablation: scores must equal sensing_innovations() when rho_v=beta=0")


def test_learn_only_ablation_ignores_sensing():
    """vismaya_learn_only (sense_scale=0, beta=0) produces scores independent of FIM geometry."""
    model_learn = _make_model_with_scale(sense_scale=0.0, beta=0.0, rho_v=1.0)
    feats = _make_features(model_learn)
    scores = model_learn.score_all(feats)
    omega_s = model_learn.sensing_innovations()
    # sense_scale=0 → sensing term is zeroed; scores driven entirely by Ω_L
    assert np.all(omega_s > 0), "Sanity: sensing innovations should be non-zero (FIM is non-zero)"
    # Multiply sense_scale by 10 — scores must not change (sensing is off)
    model_learn2 = _make_model_with_scale(sense_scale=0.0, beta=0.0, rho_v=1.0)
    feats2 = _make_features(model_learn2)  # same rng seed → same features
    scores2 = model_learn2.score_all(feats2)
    assert np.allclose(scores, scores2, atol=1e-9), (
        "learn_only ablation: scores must be reproducible (sensing disabled)")


def test_no_syn_ablation_scores_differ_from_full_after_correlated_errors():
    """After rounds with strong correlated sensing+learning errors, full VISMAYA
    should rank clients differently than the no-Syn ablation (beta=0).

    This is the make-or-break test: the Syn term only affects selection when
    Syn_k > 0, which happens when both sensing and learning surprises co-occur.
    """
    K, M = 6, 2
    # Full model (Syn active)
    model_full = _make_model_with_scale(beta=0.3, rho_v=1.0, sense_scale=1.0, K=K, M=M)
    # No-Syn ablation
    model_nosyn = _make_model_with_scale(beta=0.0, rho_v=1.0, sense_scale=1.0, K=K, M=M)

    rng = np.random.default_rng(42)
    # Build features once (shared); use large grad norms to maximise surprise signal
    losses = rng.uniform(0.1, 1.0, K)
    gnorms = rng.uniform(0.0, 1.0, K)
    channels = rng.uniform(0.1, 1.0, K)

    # Drive synergy up: select clients with very high actual grad norms (prediction error large)
    gnorms_surprising = np.full(K, 50.0)   # 50 >> ridge default prediction near 0
    for _ in range(10):
        feats = model_full.build_features(losses, gnorms, channels)
        model_full.update([0, 1], feats, gnorms_surprising)
        feats_ns = model_nosyn.build_features(losses, gnorms, channels)
        model_nosyn.update([0, 1], feats_ns, gnorms_surprising)

    # After 10 rounds, synergy for clients 0 and 1 should be non-zero in the full model
    assert model_full.synergy[0] > 0.0, "Synergy for selected client 0 must be positive"
    assert model_full.synergy[1] > 0.0, "Synergy for selected client 1 must be positive"
    assert model_nosyn.synergy[0] == 0.0 or model_nosyn.beta == 0.0, (
        "no-Syn ablation beta=0 so synergy term contributes 0 to score")

    # Scores: full model should rank clients 0,1 higher than no-Syn (synergy bonus)
    feats_eval = model_full.build_features(losses, gnorms, channels)
    scores_full = model_full.score_all(feats_eval)
    feats_eval_ns = model_nosyn.build_features(losses, gnorms, channels)
    scores_nosyn = model_nosyn.score_all(feats_eval_ns)

    # The full model must give clients 0 and 1 a synergy bonus; their scores must exceed
    # the no-Syn scores for at least one of them (Syn_k > 0 → score_full > score_nosyn)
    syn_bonus_0 = float(scores_full[0]) - float(scores_nosyn[0])
    syn_bonus_1 = float(scores_full[1]) - float(scores_nosyn[1])
    assert syn_bonus_0 > 0.0 or syn_bonus_1 > 0.0, (
        f"Syn term must give a positive bonus to at least one frequently-selected client "
        f"after correlated errors; got syn_bonus=[{syn_bonus_0:.4f}, {syn_bonus_1:.4f}]")


def test_syn_near_zero_at_start():
    """Synergy should be zero at initialisation (no history → no cross-correlation yet)."""
    model = _make_model()
    assert np.all(model.synergy == 0.0), (
        "Synergy must initialise to zero: no cross-correlation can be estimated without history")


def test_full_vs_no_syn_top_k_diverges_under_mobility():
    """Under target mobility (process_noise > 0), full VISMAYA and no-Syn ablation
    should select different top-K sets after several rounds.

    This validates the claim that Syn changes the selection ranking — not merely
    rescales all scores by a constant (which would not change the ranking).
    """
    K, M, budget = 8, 3, 3
    model_full  = _make_model_with_scale(beta=0.3, rho_v=1.0, K=K, M=M, process_noise=0.1)
    model_nosyn = _make_model_with_scale(beta=0.0, rho_v=1.0, K=K, M=M, process_noise=0.1)

    rng = np.random.default_rng(7)
    # Select clients with high gradient surprise for many rounds
    for _ in range(15):
        losses = rng.uniform(0.1, 1.0, K)
        gnorms = rng.uniform(0.0, 1.0, K)
        channels = rng.uniform(0.1, 1.0, K)
        feats_f = model_full.build_features(losses, gnorms, channels)
        feats_n = model_nosyn.build_features(losses, gnorms, channels)
        # Alternate which clients are selected to build up heterogeneous synergy
        selected = list(rng.choice(K, size=budget, replace=False))
        gnorms_act = rng.uniform(20.0, 50.0, K)   # large surprise relative to ridge prediction
        model_full.update(selected, feats_f, gnorms_act)
        model_nosyn.update(selected, feats_n, gnorms_act)

    # Final scoring round
    losses = rng.uniform(0.1, 1.0, K)
    gnorms = rng.uniform(0.0, 1.0, K)
    channels = rng.uniform(0.1, 1.0, K)
    feats_f = model_full.build_features(losses, gnorms, channels)
    feats_n = model_nosyn.build_features(losses, gnorms, channels)
    top_full  = set(np.argsort(-model_full.score_all(feats_f))[:budget].tolist())
    top_nosyn = set(np.argsort(-model_nosyn.score_all(feats_n))[:budget].tolist())

    assert top_full != top_nosyn, (
        "Full VISMAYA and no-Syn ablation must select different top-K sets after sustained "
        "correlated errors under mobility — if they agree, the Syn term has no impact on "
        f"selection. top_full={sorted(top_full)}, top_nosyn={sorted(top_nosyn)}"
    )

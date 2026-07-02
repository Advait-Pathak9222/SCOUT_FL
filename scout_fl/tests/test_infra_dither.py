"""Preflight gate (design Step 2): M2 zero-sum dither must show EXACT aggregate
invariance under perfect sync (the FL update / epsilon_agg is unchanged), and a
quantified, growing residual under sync error.
"""
import numpy as np

from scout_fl.infra.dither import ZeroSumDither, eavesdropper_crb_inflation


def test_masks_sum_to_zero():
    dith = ZeroSumDither(dim=64, sigma_d=1.0, base_seed=0)
    M = dith.masks(selected=[2, 5, 7, 9, 11])
    assert np.allclose(M.sum(axis=0), 0.0, atol=1e-12)         # exact pairwise cancellation


def test_equal_weight_aggregate_invariance():
    dith = ZeroSumDither(dim=128, sigma_d=2.0, base_seed=3)
    rng = np.random.default_rng(0)
    selected = list(range(10))
    updates = rng.standard_normal((10, 128))
    agg_clean = updates.mean(axis=0)
    agg_dith = (updates + dith.masks(selected)).mean(axis=0)   # equal-weight OTA-FedAvg
    assert np.allclose(agg_clean, agg_dith, atol=1e-12)        # EXACT invariance, perfect sync


def test_perfect_sync_residual_is_zero():
    dith = ZeroSumDither(dim=200, sigma_d=1.0, base_seed=1)
    res = dith.aggregate_residual(selected=list(range(8)), sigma_sync=0.0)
    assert np.linalg.norm(res) < 1e-9


def test_sync_error_residual_grows():
    dith = ZeroSumDither(dim=200, sigma_d=1.0, base_seed=1)
    rng = np.random.default_rng(0)
    r_small = np.linalg.norm(dith.aggregate_residual(list(range(8)), 0.01, rng))
    r_big = np.linalg.norm(dith.aggregate_residual(list(range(8)), 0.2, rng))
    assert r_big > r_small > 0.0                               # honest: sync offset breaks cancellation


def test_dither_variance_matches_sigma_d():
    dith = ZeroSumDither(dim=4000, sigma_d=1.5, base_seed=7)
    M = dith.masks(selected=list(range(12)))
    per_client_var = M.var(axis=1).mean()
    assert abs(per_client_var - 1.5 ** 2) < 0.3               # per-coordinate variance ~ sigma_d^2


def test_eavesdropper_inflation_monotone():
    base = eavesdropper_crb_inflation(0.0, snr_eve=10.0, n_receivers=1)
    more = eavesdropper_crb_inflation(1.0, snr_eve=10.0, n_receivers=1)
    colluding = eavesdropper_crb_inflation(1.0, snr_eve=10.0, n_receivers=3)
    assert base == 1.0 and more > base                        # dither inflates eavesdropper CRB
    assert colluding < more                                    # colluding receivers average it down

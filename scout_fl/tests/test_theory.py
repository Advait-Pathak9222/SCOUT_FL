"""Tests for the physical link budget + theory-validation modules (P6/P3-dual/P7).

Run:  pytest scout_fl/tests/test_theory.py -q
"""
from __future__ import annotations

import numpy as np

from scout_fl.analysis.convergence import _ols
from scout_fl.analysis.regret import _slope_loglog
from scout_fl.analysis.feasibility import _tail_slope
from scout_fl.selection.online import CUCBSensingSelector
from scout_fl.sim.link_budget import (dbm_to_watt, path_gain_linear, received_snr_db,
                                      thermal_noise_power_w)


# ----------------------------------------------------------- physical link budget
def test_thermal_noise_matches_minus174_dbm_per_hz():
    # kTB at 290K, NF=0 -> -174 dBm/Hz; with B=1e6 Hz the noise power ~ -174+60 = -114 dBm
    s2 = thermal_noise_power_w(1e6, noise_figure_db=0.0, temperature_k=290.0)
    dbm = 10 * np.log10(s2) + 30
    assert abs(dbm - (-114.0)) < 0.5


def test_dbm_to_watt():
    assert abs(dbm_to_watt(30.0) - 1.0) < 1e-9      # 30 dBm = 1 W
    assert abs(dbm_to_watt(0.0) - 1e-3) < 1e-12     # 0 dBm = 1 mW


def test_path_gain_decreases_with_distance_and_snr_is_db():
    g_near = path_gain_linear(10.0, carrier_ghz=3.5, exponent=3.0)
    g_far = path_gain_linear(100.0, carrier_ghz=3.5, exponent=3.0)
    assert g_near > g_far > 0
    snr = received_snr_db(dbm_to_watt(0.0), np.array([g_near, g_far]),
                          thermal_noise_power_w(1e6, 7.0))
    assert snr[0] > snr[1]                           # nearer client has higher SNR (dB)


# ----------------------------------------------------------- CUCB (P7)
def test_cucb_forces_initial_exploration_then_tracks_mean():
    b = CUCBSensingSelector(K=5, ucb_c=1.0)
    assert np.all(b.ucb_snr() >= 1e8)                # all unpulled -> forced exploration
    b.update([0, 1], np.array([2.0, 4.0, 0, 0, 0]))
    idx = b.ucb_snr()
    assert idx[2] >= 1e8 and idx[0] < 1e8            # 2 still unpulled, 0 now has a finite index
    assert abs(b.mean[1] - 4.0) < 1e-9               # empirical mean tracks observation


# ----------------------------------------------------------- P6 regression helper
def test_ols_recovers_known_signs():
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.standard_normal(n)                      # grad signal
    x2 = rng.standard_normal(n)                      # agg mse
    y = 0.5 * x1 - 0.3 * x2 + 0.05 * rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x1, x2])
    beta, se, t, p, r2 = _ols(y, X)
    assert beta[1] > 0 and beta[2] < 0               # recovers + / - signs
    assert r2 > 0.9 and p[1] < 0.01 and p[2] < 0.01


# ----------------------------------------------------------- P7 / feasibility slopes
def test_loglog_slope_sqrt_is_half_and_linear_is_one():
    T = np.arange(1, 500)
    assert abs(_slope_loglog(T, 3.0 * np.sqrt(T)) - 0.5) < 0.05
    assert abs(_slope_loglog(T, 2.0 * T) - 1.0) < 0.05


def test_tail_slope_flat_series_is_zero():
    y = np.zeros(50)                                 # no violation -> running-avg flat -> slope ~0
    assert abs(_tail_slope(y)) < 1e-9

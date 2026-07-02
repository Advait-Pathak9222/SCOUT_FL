"""Preflight gate (design R4): the Kalman tracker must be NEES-consistent, else all
tracking results are meaningless and the batch must abort.

Also checks the CV mobility model's basic invariants and the covariance-only
trace-reduction helper used by the adaptive-threshold controller.
"""
import numpy as np
import pytest

from scout_fl.infra.mobility import CVMobility, cv_matrices
from scout_fl.infra.tracker import InformationKalmanTracker, nees_consistency


def test_nees_consistency_stationary():
    res = nees_consistency(sigma_p=0.0, steps=120, n_mc=24, seed=1)
    assert res["consistent"], f"NEES {res['avg_nees']:.2f} outside [{res['lo']:.2f},{res['hi']:.2f}] (sigma_p=0)"


def test_nees_consistency_mobile():
    res = nees_consistency(sigma_p=0.05, steps=120, n_mc=24, seed=2)
    assert res["consistent"], f"NEES {res['avg_nees']:.2f} outside [{res['lo']:.2f},{res['hi']:.2f}] (sigma_p=0.05)"


def test_cv_matrices_shapes():
    F, Q = cv_matrices(0.05)
    assert F.shape == (4, 4) and Q.shape == (4, 4)
    assert np.allclose(F @ np.array([1, 2, 3, 4]), [1 + 3, 2 + 4, 3, 4])   # x += v*dt
    assert np.all(np.linalg.eigvalsh(Q) >= -1e-12)                        # PSD process noise


def test_mobility_stationary_is_frozen():
    rng = np.random.default_rng(0)
    m = CVMobility(np.array([[50.0, 50.0], [30.0, 70.0]]), sigma_p=0.0, rng=rng)
    for _ in range(20):
        m.step()
    assert np.allclose(m.positions, [[50.0, 50.0], [30.0, 70.0]])         # sigma_p=0 => no motion


def test_mobility_logs_trajectory():
    rng = np.random.default_rng(0)
    m = CVMobility(np.array([[50.0, 50.0]]), sigma_p=0.05, rng=rng, area=[100.0, 100.0])
    for _ in range(10):
        m.step()
    traj = m.trajectory_array()
    assert traj.shape == (11, 1, 2)                                       # T+1 logged positions


def test_predicted_trace_reduction_nonnegative():
    rng = np.random.default_rng(3)
    trk = InformationKalmanTracker(np.array([[50.0, 50.0]]), 0.05, rng)
    J = np.array([[[1.0, 0.0], [0.0, 0.5]]])
    red = trk.predicted_trace_reduction(J)
    assert red[0] > 0.0                                                   # adding info shrinks tr(P)
    # and it does not mutate state
    before = trk.trace_pos()[0]
    trk.predicted_trace_reduction(J)
    assert trk.trace_pos()[0] == before

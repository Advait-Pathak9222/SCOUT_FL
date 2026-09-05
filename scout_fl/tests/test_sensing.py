"""Milestone-1 unit tests: FIM/CRB correctness, submodularity, greedy, gate.

Run:  pytest scout_fl/tests -q
"""
from __future__ import annotations

import itertools

import numpy as np

from scout_fl.analysis.verify_submodularity import verify_submodular
from scout_fl.objectives.sensing_utility import SensingUtility
from scout_fl.selection.lazy_greedy import lazy_greedy
from scout_fl.selection.scout_greedy import ScoutGreedy, naive_greedy, penalized_greedy
from scout_fl.sim.crb import logdet_spd
from scout_fl.sim.fim import db_to_linear, per_client_target_fim, prior_fim
from scout_fl.sim.geometry import pairwise_geometry


def _microbench_utility():
    clients = np.array([[0.0, 0.0], [5.0, 6.0], [50.0, 40.0]])
    targets = np.array([[50.0, 0.0]])
    geom = pairwise_geometry(clients, targets)
    snr = db_to_linear([20.0, 20.0, 10.0])
    fim = per_client_target_fim(geom, snr, k_range=1.0, k_angle=0.02)
    j0 = prior_fim(1, 1e-3)
    return SensingUtility(fim, j0)


def _random_utility(seed=0, K=12, M=3):
    rng = np.random.default_rng(seed)
    clients = rng.uniform(0, 100, size=(K, 2))
    targets = rng.uniform(0, 100, size=(M, 2))
    geom = pairwise_geometry(clients, targets)
    snr = db_to_linear(rng.uniform(0, 25, size=K))
    fim = per_client_target_fim(geom, snr, k_range=1.0, k_angle=0.05)
    return SensingUtility(fim, prior_fim(M, 1e-3))


def test_fim_is_psd():
    util = _random_utility()
    eigvals = np.linalg.eigvalsh(util.J)        # (K, M, 2)
    assert np.all(eigvals > -1e-9)


def test_logdet_monotone_in_clients():
    util = _microbench_utility()
    base = util.value([0])
    more = util.value([0, 2])
    assert more >= base - 1e-12               # adding a client never reduces log-det


def test_crb_decreases_with_more_information():
    util = _microbench_utility()
    crb_one = float(util.crb([0]).sum())
    crb_two = float(util.crb([0, 2]).sum())
    assert crb_two < crb_one                  # complementary client reduces CRB


def test_microbenchmark_ordering():
    """The crux: complementary pair {1,3} beats the redundant high-SNR pair {1,2}."""
    util = _microbench_utility()
    crb_12 = float(util.crb([0, 1]).sum())    # two high-SNR, same bearing
    crb_13 = float(util.crb([0, 2]).sum())    # high-SNR + complementary medium-SNR
    crb_23 = float(util.crb([1, 2]).sum())
    assert crb_13 < crb_12
    assert crb_23 < crb_12


def test_scout_greedy_picks_complementary_pair():
    util = _microbench_utility()
    res = ScoutGreedy(use_lazy=True).select(utility=util, num_clients=util.K, budget=2)
    assert 2 in res.selected                  # must include the complementary client (idx 2)


def test_lazy_matches_naive_greedy():
    util = _random_utility(seed=1, K=15)
    lazy_sel, _, _ = lazy_greedy(util, util.K, budget=5)
    naive_sel, _, _ = naive_greedy(util, util.K, budget=5)
    assert lazy_sel == naive_sel              # CELF == plain greedy for monotone submodular


def test_greedy_respects_budget():
    util = _random_utility()
    res = ScoutGreedy().select(utility=util, num_clients=util.K, budget=4)
    assert len(res.selected) == 4
    assert len(set(res.selected)) == 4


def test_f_sense_is_submodular():
    util = _random_utility(seed=2, K=12)
    report = verify_submodular(util.value, list(range(util.K)),
                               n_samples=2000, rng=np.random.default_rng(7))
    assert report["is_submodular"]
    assert report["is_monotone"]


def test_reproducible_selection():
    u1, u2 = _random_utility(seed=3), _random_utility(seed=3)
    s1 = ScoutGreedy().select(utility=u1, num_clients=u1.K, budget=5).selected
    s2 = ScoutGreedy().select(utility=u2, num_clients=u2.K, budget=5).selected
    assert s1 == s2


class _NanUtility:
    def init_state(self):
        return []

    def marginal_gain(self, state, k):
        return float("nan")

    def add(self, state, k):
        return state + [k]


def test_greedy_handles_nonfinite_marginals_without_none_selection():
    util = _NanUtility()
    assert naive_greedy(util, 4, 3)[0] == [0, 1, 2]
    assert lazy_greedy(util, 4, 3)[0] == [0, 1, 2]
    assert penalized_greedy(util, 4, 3, penalty_fn=lambda S, k: float("nan"))[0] == [0, 1, 2]
    res = ScoutGreedy(use_lazy=False).select(
        utility=util, num_clients=4, budget=3, feasible=lambda S, k: False)
    assert res.selected == [0, 1, 2]


def test_echo_snr_tracks_transmit_power():
    """The transmission that carries the update also illuminates the target, so the echo
    SNR must move with the transmit power once the reference power is declared."""
    from scout_fl.sim.sensing import sensing_snr
    geom = {"range": np.full((4, 2), 20.0)}
    ref = sensing_snr(geom, 20.0, 2.0, ref_distance=10.0,
                      tx_power_dbm=-15.0, ref_tx_power_dbm=-15.0)
    base = sensing_snr(geom, 20.0, 2.0, ref_distance=10.0)
    assert np.allclose(ref, base), "the reference power must leave the calibration untouched"
    for tx, factor in ((-5.0, 10.0), (-25.0, 0.1), (-35.0, 0.01)):
        got = sensing_snr(geom, 20.0, 2.0, ref_distance=10.0,
                          tx_power_dbm=tx, ref_tx_power_dbm=-15.0)
        assert np.allclose(got, base * factor, rtol=1e-9)
    # without a declared reference the sensing axis stays independent of the power
    assert np.allclose(sensing_snr(geom, 20.0, 2.0, ref_distance=10.0, tx_power_dbm=-35.0), base)


def test_fading_redraw_changes_the_channel_but_not_the_geometry():
    from scout_fl.sim.channel import large_scale_gain, small_scale_fading
    clients = np.random.default_rng(0).uniform(0.0, 100.0, (30, 2))
    bs = np.array([50.0, 50.0])
    L = large_scale_gain(clients, bs, pathloss_model="physical")
    rng = np.random.default_rng(3)
    blocks = [L * small_scale_fading(30, rng) for _ in range(6)]
    assert not np.allclose(blocks[0], blocks[1]), "fading must move between coherence blocks"
    ratios = np.stack([b / L for b in blocks])
    assert 0.5 < ratios.mean() < 2.0, "fading power should average near one"

"""Fair-testbed tests: clustered viewpoints must restore the 'high-SNR != value'
gap so that diversity / joint-information selection can beat SNR-only on the
log-det coverage-diversity metric.

Run:  pytest scout_fl/tests -q
"""
from __future__ import annotations

import numpy as np

from scout_fl.experiments.run_synthetic import build_scenario
from scout_fl.objectives.sensing_utility import SensingUtility
from scout_fl.selection.scout_greedy import ScoutGreedy
from scout_fl.selection.snr_based import SNRSelector
from scout_fl.utils.config import Config


def _clustered_cfg(K=20, num_clusters=5, M=3):
    return Config({
        "network": {"num_clients": K, "num_targets": M, "num_regions": 9, "area_size": [100.0, 100.0]},
        "geometry": {"layout": "clustered", "num_clusters": num_clusters, "cluster_spread": 3.0,
                     "bs_position": [50.0, 50.0], "random_clients": True, "random_targets": True},
        "sensing": {"ref_snr_db": 20.0, "ref_distance": 10.0, "pathloss_exponent": 2.0,
                    "rcs_mean": 1.0, "rcs_std": 0.3, "k_range": 1.0, "k_angle": 0.05,
                    "prior_fim": 1.0e-3, "target_weights": [1.0, 1.0, 1.0]},
        "coverage": {"sensing_range": 18.0, "saturating": "exp"},
    })


def test_clustered_assignment_is_valid():
    scn = build_scenario(_clustered_cfg(K=20, num_clusters=5), np.random.default_rng(0))
    assert scn.cluster_assignment.shape == (20,)
    assert set(scn.cluster_assignment.tolist()) == set(range(5))   # all clusters populated


def test_clustered_snr_is_redundant_diversity_wins():
    """Core fair-testbed property: log-det-greedy (diversity-aware) selection achieves
    at least the log-det of SNR-only selection AND spans at least as many viewpoint
    clusters — i.e., SNR-only piles into the near (redundant) cluster."""
    rng = np.random.default_rng(0)
    scn = build_scenario(_clustered_cfg(K=20, num_clusters=5, M=3), rng)
    util = SensingUtility(scn.fim, scn.j0, scn.w)
    budget = 5
    snr = SNRSelector().select(scores=scn.snr.sum(axis=1), budget=budget).selected
    greedy = ScoutGreedy().select(utility=util, num_clients=scn.K, budget=budget).selected
    ca = scn.cluster_assignment
    assert util.value(greedy) >= util.value(snr) - 1e-9                    # diversity >= SNR-only on log-det
    assert len(set(ca[greedy].tolist())) >= len(set(ca[snr].tolist()))    # spans >= viewpoint clusters


def test_random_layout_still_works():
    cfg = _clustered_cfg(K=12, num_clusters=4, M=2)
    cfg.geometry["layout"] = "random"
    scn = build_scenario(cfg, np.random.default_rng(1))
    assert scn.clients.shape == (12, 2) and scn.K == 12

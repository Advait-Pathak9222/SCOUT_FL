"""UCI HAR + external wireless-sensing dataset adapters (synthetic-fallback paths).

These tests never touch the network: with no real data present each adapter must
return a valid synthetic ``FLDataset`` so the campaign runs end-to-end offline.

Run:  pytest scout_fl/tests/test_datasets_extra.py -q
"""
from __future__ import annotations

import numpy as np

from scout_fl.fl.datasets import FLDataset, load_fl_dataset
from scout_fl.fl.datasets_external import (load_channel_realizations,
                                           load_external_classification,
                                           load_sensing_geometry)
from scout_fl.fl.datasets_extra import load_uci_har


def _check(ds: FLDataset, n_features=None, n_classes=None):
    assert isinstance(ds, FLDataset)
    assert ds.x_train.ndim >= 2 and ds.x_test.ndim >= 2
    assert ds.y_train.min() >= 0 and int(ds.y_train.max()) < ds.num_classes
    assert ds.x_train.shape[0] == ds.y_train.shape[0]
    if n_features is not None:
        assert ds.input_shape == (n_features,)
    if n_classes is not None:
        assert ds.num_classes == n_classes


def test_uci_har_synthetic_fallback_shapes():
    ds = load_uci_har(root="data", download=False)        # no real data -> synthetic
    _check(ds, n_features=561, n_classes=6)


def test_external_classification_fallbacks():
    for name, feats, cls in [("deepmimo", 256, 64), ("deepsense6g", 512, 64),
                             ("radarscenes", 128, 6)]:
        _check(load_external_classification(name, root="data"), n_features=feats, n_classes=cls)


def test_dispatcher_routes_extra_and_external():
    _check(load_fl_dataset("uci_har", root="data", download=False), n_features=561, n_classes=6)
    _check(load_fl_dataset("radarscenes", root="data"), n_features=128, n_classes=6)


def test_channel_and_geometry_fallbacks():
    rng = np.random.default_rng(0)
    gains = load_channel_realizations("deepmimo", num_clients=20, rng=rng, root="data")
    assert gains.shape == (20,) and np.all(gains > 0)
    clients, targets = load_sensing_geometry("radarscenes", 15, 3, rng, root="data")
    assert clients.shape == (15, 2) and targets.shape == (3, 2)


def test_build_scenario_uses_external_geometry_source():
    # geometry.source -> the external sensing-geometry adapter feeds the wireless sim
    from scout_fl.experiments.run_synthetic import build_scenario
    from scout_fl.utils.config import load_config
    cfg = load_config("scout_fl/configs/campaign_main.yaml",
                      ["geometry.source=radarscenes", "network.num_clients=12",
                       "network.num_targets=3"])
    scn = build_scenario(cfg, np.random.default_rng(0))
    assert scn.K == 12 and scn.clients.shape == (12, 2)
    assert scn.fim.shape[0] == 12 and scn.cluster_assignment.shape == (12,)

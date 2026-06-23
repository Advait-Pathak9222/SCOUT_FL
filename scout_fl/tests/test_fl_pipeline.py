"""Tests for the FL pipeline: models, client, aggregation, server, and a tiny
synthetic end-to-end ``run_one`` (no MNIST needed).

Run:  pytest scout_fl/tests -q
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from scout_fl.fl.aggregation import aggregate, fedavg, ota_distort
from scout_fl.fl.client import local_train, probe_loss_and_embedding
from scout_fl.fl.models import build_model, get_flat_params, num_params, set_flat_params
from scout_fl.fl.server import FLServer
from scout_fl.utils.config import Config, load_config


def _fake_dataset(n=40, shape=(1, 8, 8), classes=2, seed=0):
    rng = torch.Generator().manual_seed(seed)
    x = torch.rand((n, *shape), generator=rng)
    y = torch.randint(0, classes, (n,), generator=rng)
    return TensorDataset(x, y)


# ------------------------------------------------------------------- models
def test_build_model_forward_shapes():
    for mtype in ("mlp", "small_cnn"):
        model = build_model(mtype, (1, 28, 28), 10)
        assert model(torch.randn(5, 1, 28, 28)).shape == (5, 10)
        rgb = build_model(mtype, (3, 32, 32), 100)         # CIFAR-100 shape
        assert rgb(torch.randn(4, 3, 32, 32)).shape == (4, 100)


def test_flat_param_roundtrip():
    model = build_model("mlp", (1, 8, 8), 2)
    flat = get_flat_params(model)
    assert flat.numel() == num_params(model)
    set_flat_params(model, torch.zeros_like(flat))
    assert torch.allclose(get_flat_params(model), torch.zeros_like(flat))


# ------------------------------------------------------------------- client
def test_local_train_changes_weights():
    model = build_model("mlp", (1, 8, 8), 2)
    before = get_flat_params(model).clone()
    out = local_train(model, _fake_dataset(), epochs=1, lr=0.1, batch_size=8)
    after = get_flat_params(model)
    assert not torch.allclose(before, after)               # weights moved
    assert out["update"].shape == (num_params(model),)
    assert out["num_samples"] == 40 and out["loss"] >= 0


def test_probe_returns_loss_and_embedding():
    model = build_model("mlp", (1, 8, 8), 2)
    before = get_flat_params(model).clone()
    loss, emb = probe_loss_and_embedding(model, _fake_dataset(), batch_size=8)
    assert loss >= 0 and emb.shape == (num_params(model),)
    assert torch.allclose(before, get_flat_params(model))  # probe must NOT change weights


# -------------------------------------------------------------- aggregation
def test_fedavg_equal_weight_is_mean():
    u = [np.ones(5), 3 * np.ones(5)]
    assert np.allclose(fedavg(u, [10, 10]), 2 * np.ones(5))
    assert np.allclose(fedavg(u, [30, 10]), 0.25 * (3 * np.ones(5)) + 0.75 * np.ones(5))


def test_aggregate_ota_off_equals_fedavg_and_on_is_finite():
    u = [np.ones(8), np.zeros(8)]
    base = aggregate(u, [1, 1], ota=False)
    assert np.allclose(base, fedavg(u, [1, 1]))
    noisy = aggregate(u, [1, 1], ota=True, mse=0.1, scale=1.0, rng=np.random.default_rng(0))
    assert noisy.shape == (8,) and np.all(np.isfinite(noisy))
    assert np.allclose(ota_distort(base, mse=0.0, scale=1.0, rng=np.random.default_rng(0)), base)


# ------------------------------------------------------------------- server
def test_server_evaluate_and_apply_update():
    server = FLServer(build_model("mlp", (1, 8, 8), 2))
    x = torch.rand(20, 1, 8, 8); y = torch.randint(0, 2, (20,))
    loss, acc = server.evaluate(x, y)
    assert loss >= 0 and 0.0 <= acc <= 1.0
    base = server.global_flat()
    server.apply_aggregated_update(base, np.ones_like(base))
    assert np.allclose(server.global_flat(), base + 1.0, atol=1e-5)


# -------------------------------------------------------------------- config
def test_fl_config_loads_required_fields():
    cfg = load_config("scout_fl/configs/fl_synthetic_small.yaml")
    assert cfg.fl.dataset in ("mnist", "fashion_mnist")
    assert int(cfg.fl.rounds) >= 1 and int(cfg.network.budget) >= 1
    assert isinstance(cfg.selection.methods, list) and "scout_greedy" in cfg.selection.methods
    assert cfg.constraints.mse_agg_max is not None
    assert isinstance(cfg.aircomp.bandwidth, float)        # numeric coercion of 1.0e+6


# --------------------------------------------------- end-to-end (synthetic)
def _tiny_cfg():
    return Config({
        "network": {"budget": 2},
        "fl": {"device": "cpu", "rounds": 2, "batch_size": 8, "probe_batches": 1,
               "local_epochs": 1, "lr": 0.1, "optimizer": "sgd", "model": "mlp"},
        "coverage": {"rho": 0.9, "innovation": 0.05, "saturating": "exp"},
        "objectives": {"alpha_learning": 1.0, "lambda_sense": 1.0,
                       "lambda_coverage": 0.5, "lambda_fairness": 0.3},
        "aircomp": {"enabled": True, "power": 1.0, "sigma2": 1.0, "ota_distortion": False,
                    "ota_noise_scale": 0.5, "bandwidth": 1.0e6, "model_bits": 1.0e5},
        "constraints": {"mse_agg_max": 0.1},
        "energy": {"cpu_cycles": 1.0e7, "cpu_freq": 1.0e9, "kappa": 1.0e-27,
                   "e_sense": 0.1, "t_sense": 0.01},
    })


def _tiny_scenario(seed=0):
    from scout_fl.experiments.run_synthetic import Scenario
    from scout_fl.objectives.coverage_utility import contribution_matrix, region_centers
    from scout_fl.sim.fim import db_to_linear, per_client_target_fim, prior_fim
    from scout_fl.sim.geometry import pairwise_geometry
    rng = np.random.default_rng(seed)
    K, M, R = 4, 1, 4
    clients = rng.uniform(0, 100, (K, 2)); targets = rng.uniform(0, 100, (M, 2))
    geom = pairwise_geometry(clients, targets)
    snr = db_to_linear(rng.uniform(5, 20, K))
    fim = per_client_target_fim(geom, snr, 1.0, 0.05)
    C = contribution_matrix(clients, region_centers([100, 100], R), 30.0)
    scn = Scenario(snr=snr[:, None], fim=fim, j0=prior_fim(M, 1e-3), w=np.ones(M),
                   C=C, sim=np.eye(K), K=K, M=M, R=R, clients=clients)
    return scn


def test_run_one_end_to_end_tiny():
    from scout_fl.experiments.run_fl_synthetic import run_one
    cfg, scn = _tiny_cfg(), _tiny_scenario()
    clients = [_fake_dataset(n=24, seed=i) for i in range(scn.K)]
    x_te = torch.rand(20, 1, 8, 8); y_te = torch.randint(0, 2, (20,))
    g = np.array([0.6, 0.9, 1.2, 0.7])
    rows, part = run_one("scout_greedy", cfg, scn, g, clients, x_te, y_te, (1, 8, 8), 2, base_seed=0)
    assert len(rows) == 2
    assert len(rows[0]["selected"]) == 2
    assert 0.0 <= rows[-1]["test_acc"] <= 1.0
    assert part.sum() == 2 * 2                              # budget * rounds selections


def test_run_one_reproducible_selection():
    from scout_fl.experiments.run_fl_synthetic import run_one
    cfg, scn = _tiny_cfg(), _tiny_scenario()
    clients = [_fake_dataset(n=24, seed=i) for i in range(scn.K)]
    x_te = torch.rand(20, 1, 8, 8); y_te = torch.randint(0, 2, (20,))
    g = np.array([0.6, 0.9, 1.2, 0.7])
    r1, _ = run_one("scout_greedy", cfg, scn, g, clients, x_te, y_te, (1, 8, 8), 2, base_seed=0)
    r2, _ = run_one("scout_greedy", cfg, scn, g, clients, x_te, y_te, (1, 8, 8), 2, base_seed=0)
    assert [r["selected"] for r in r1] == [r["selected"] for r in r2]

"""CloakFL training loop: M1 leakage-capped selection + leakage accounting + M2/M3.

One resumable unit = one (mode, r_floor, seed) FL training. Reuses the campaign's
FL primitives; differences vs run_fl_synthetic:
  * selection is the M1 leakage-capped greedy (cloak/selection.py) over the SCOUT
    composite utility, with a per-client cumulative-leakage cap J_max derived from
    the run's target client-CRB floor r_floor;
  * a LeakageAccountant (BS side, A1) and an eavesdropper accountant (A2) accumulate
    per-client position Fisher information each round -> client CRB floor r_k;
  * M2 zero-sum dither is applied to the (equal-weight AirComp) aggregate so the FL
    update is exactly invariant (verified in the E-C2c on/off comparison);
  * M3 obfuscation attenuates BS-usable leakage at a logged AirComp-MSE cost.

Aggregation is EQUAL-WEIGHT (OTA-FedAvg / AirComp) so M2's zero-sum masks cancel
exactly (design §2.4.2). Artifacts follow utils/runstore schema.
"""
from __future__ import annotations

import time

import numpy as np

from scout_fl.experiments.run_fl_synthetic import _physical_params, _subsample
from scout_fl.experiments.run_synthetic import build_scenario
from scout_fl.fl.client import local_train, probe_loss_and_embedding
from scout_fl.fl.datasets import build_client_datasets
from scout_fl.fl.models import build_model, num_params
from scout_fl.fl.partitioning import partition, partition_spatial
from scout_fl.fl.server import FLServer
from scout_fl.infra.dither import ZeroSumDither
from scout_fl.infra.leakage import LeakageAccountant
from scout_fl.objectives.coverage_utility import CoverageMap, CoverageUtility
from scout_fl.objectives.fairness_utility import FairnessUtility
from scout_fl.objectives.learning_utility import LearningUtility
from scout_fl.objectives.primal_dual import DualState
from scout_fl.objectives.sensing_utility import SensingUtility
from scout_fl.objectives.total_utility import TotalUtility
from scout_fl.sim.aircomp import aggregation_mse
from scout_fl.sim.channel import comm_channel_gains
from scout_fl.sim.energy_latency import round_energy_latency
from scout_fl.cloak.mechanisms import mode_params
from scout_fl.cloak.selection import leakage_capped_greedy, random_selection
from scout_fl.utils.device import resolve_device
from scout_fl.utils.runstore import load_unit, save_unit, unit_path
from scout_fl.utils.seed import seed_everything


def run_cloak_seed(cfg, ds, seed, mode, r_floor, *, runs_root=None, tag="cloak", point="base",
                   method=None):
    """Run one CloakFL unit; return (rows, objectives). Resumable via runs_root."""
    method = method or f"{mode}__r{r_floor:g}"
    path = unit_path(runs_root, tag, point, method, seed) if runs_root else None
    if path is not None:
        cached = load_unit(path)
        if cached is not None:
            print(f"  [resume] {tag}/{point}/{method} seed{seed}: {len(cached['rounds'])} rounds")
            return cached["rounds"], cached["objectives"]

    mp = mode_params(mode)
    device = resolve_device(cfg.fl.get("device", "auto"))
    rng = seed_everything(int(seed))
    scn = build_scenario(cfg, rng)
    K, budget, rounds = scn.K, int(cfg.network.budget), int(cfg.fl.rounds)
    bs = np.asarray(cfg.geometry.bs_position, dtype=float)

    phys_cfg = cfg.get("physical", {})
    g = comm_channel_gains(
        scn.clients, bs, rng, snr_ref_db=cfg.channel.snr_ref_db,
        ref_distance=cfg.channel.reference_distance, pathloss_exponent=cfg.channel.pathloss_exponent,
        model=cfg.channel.model, rician_k_db=cfg.channel.rician_k_db,
        pathloss_model=("physical" if phys_cfg and phys_cfg.get("enabled") else "reference_snr"),
        carrier_ghz=float(phys_cfg.get("carrier_ghz", 3.5)) if phys_cfg else 3.5)
    if getattr(scn, "compute_het", None) is None:
        scn.compute_het = rng.uniform(0.1, 1.0, scn.K)

    x_tr, y_tr = _subsample(ds.x_train, ds.y_train, cfg.fl.get("subsample_train"), rng)
    x_te, y_te = _subsample(ds.x_test, ds.y_test, cfg.fl.get("subsample_test"), rng)
    if cfg.fl.non_iid == "spatial":
        parts = partition_spatial(np.asarray(y_tr), scn.cluster_assignment,
                                  cfg.fl.get("dirichlet_alpha", 0.5), np.random.default_rng(seed))
    else:
        parts = partition(np.asarray(y_tr), scn.K, cfg.fl.non_iid,
                          cfg.fl.get("dirichlet_alpha", 0.5), np.random.default_rng(seed), min_size=1)
    client_datasets = build_client_datasets(x_tr, y_tr, parts)

    server = FLServer(build_model(cfg.fl.model, ds.input_shape, ds.num_classes), device=device)
    _avg_local = float(np.mean([len(d) for d in client_datasets])) if client_datasets else 1.0
    phys = _physical_params(cfg, num_params(server.model), _avg_local)
    if phys is not None:
        P, sigma2, model_bits, cpu_cycles = phys["power"], phys["sigma2"], phys["model_bits"], phys["cpu_cycles"]
    else:
        P, sigma2 = float(cfg.aircomp.power), float(cfg.aircomp.sigma2)
        model_bits, cpu_cycles = float(cfg.aircomp.model_bits), float(cfg.energy.cpu_cycles)
    snr_up = P * np.asarray(g, dtype=float) / sigma2
    mse_eps = cfg.constraints.mse_agg_max
    aircomp_on = bool(cfg.aircomp.enabled)
    duals = DualState({"mse": mse_eps}, lr=float(cfg.constraints.get("dual_lr", 0.5)))

    sensing = SensingUtility(scn.fim, scn.j0, scn.w)         # stationary targets (mobility optional)
    cmap = CoverageMap(scn.R, rho=cfg.coverage.rho, innovation=cfg.coverage.innovation, u_init=1.0)
    fair = FairnessUtility(K)

    # leakage: BS (A1) and eavesdropper (A2) accountants
    kr, ka = float(cfg.sensing.k_range), float(cfg.sensing.k_angle)
    prior_std = float(cfg.get("cloak", {}).get("prior_client_std_m", 100.0))
    bs_acct = LeakageAccountant(scn.clients, bs, k_range=kr, k_angle=ka, prior_std_m=prior_std)
    eve_acct = LeakageAccountant(scn.clients, bs, k_range=kr, k_angle=ka, prior_std_m=prior_std)
    cap_floor = r_floor if mp["J_max"] == "cap" else None    # CRB-floor cap (None = uncapped)
    dith = ZeroSumDither(dim=num_params(server.model), sigma_d=float(cfg.get("cloak", {}).get("sigma_d", 1.0)),
                         base_seed=int(seed))

    participation = np.zeros(K)
    rows = []
    for t in range(rounds):
        g_flat = server.global_flat()
        tic = time.perf_counter()
        losses = np.zeros(K)
        embs = []
        for k in range(K):
            server.set_global(g_flat)
            lk, ek = probe_loss_and_embedding(server.model, client_datasets[k],
                                              batch_size=cfg.fl.batch_size, device=device,
                                              max_batches=int(cfg.fl.get("probe_batches", 1)))
            losses[k] = lk
            embs.append(ek)
        embs = np.stack(embs)
        probe_time = time.perf_counter() - tic

        learning = LearningUtility(embeddings=embs)
        coverage = CoverageUtility(cmap.U, scn.C, g=cfg.coverage.saturating)
        full = list(range(K))
        weights = {"learning": cfg.objectives.alpha_learning, "sensing": cfg.objectives.lambda_sense,
                   "coverage": cfg.objectives.lambda_coverage, "fairness": cfg.objectives.lambda_fairness}
        norms = {"learning": max(learning.value(full), 1e-9), "sensing": max(sensing.value(full), 1e-9),
                 "coverage": max(coverage.value(full), 1e-9), "fairness": max(fair.value(full), 1e-9)}
        total = TotalUtility({"learning": learning, "sensing": sensing, "coverage": coverage,
                              "fairness": fair}, weights=weights, normalizers=norms)

        tic = time.perf_counter()
        relaxed = 0
        if mp["selector"] == "random":
            selected = random_selection(K, budget, rng)
        else:
            def penalty_fn(S, k):
                mse_k = aggregation_mse(g, S + [k], power=P, sigma2=sigma2)
                return duals.mu.get("mse", 0.0) * max(0.0, mse_k - (mse_eps or 0.0))
            selected, relaxed = leakage_capped_greedy(
                total, K, budget, accountant=bs_acct, snr_up=snr_up, atten=mp["leak_atten"],
                r_floor=cap_floor, mse_penalty_fn=penalty_fn if aircomp_on else None)
        sel_time = time.perf_counter() - tic
        participation[selected] += 1

        # local training + EQUAL-WEIGHT AirComp aggregation (so M2 masks cancel exactly)
        tic = time.perf_counter()
        updates, train_losses = [], []
        for k in selected:
            server.set_global(g_flat)
            out = local_train(server.model, client_datasets[k], epochs=int(cfg.fl.local_epochs),
                              lr=float(cfg.fl.lr), batch_size=int(cfg.fl.batch_size),
                              optimizer=cfg.fl.optimizer, device=device)
            updates.append(out["update"]); train_losses.append(out["loss"])
        U = np.stack(updates)
        if mp["m2_dither"]:
            U = U + dith.masks(selected)                     # zero-sum: column mean unchanged
        agg = U.mean(axis=0)                                 # equal-weight OTA-FedAvg
        mse = aggregation_mse(g, selected, power=P, sigma2=sigma2) * mp["mse_infl"] if aircomp_on else 0.0
        if aircomp_on:
            duals.update({"mse": float(mse)})
        server.apply_aggregated_update(g_flat, agg)
        train_time = time.perf_counter() - tic
        test_loss, test_acc = server.evaluate(x_te, y_te)

        # leakage accounting (BS uses leak_atten; eavesdropper uses eaves_atten)
        bs_acct.observe(selected, snr_up, atten=mp["leak_atten"])
        eve_acct.observe(selected, snr_up, atten=mp["eaves_atten"])
        bs_sum = bs_acct.summary()
        eve_r = eve_acct.crb_floor()

        el = round_energy_latency(selected, g, power=P, sigma2=sigma2,
                                  bandwidth=cfg.aircomp.bandwidth, model_bits=model_bits,
                                  cpu_cycles=cpu_cycles, cpu_freq=cfg.energy.cpu_freq,
                                  kappa=cfg.energy.kappa, e_sense=cfg.energy.e_sense,
                                  t_sense=cfg.energy.t_sense)
        row = {
            "method": method, "round": t,
            "train_loss": round(float(np.mean(train_losses)), 5),
            "test_loss": round(float(test_loss), 5), "test_acc": round(float(test_acc), 5),
            "selected": list(selected),
            "learning_util": round(float(learning.value(selected)), 4),
            "sensing_logdet": round(float(sensing.value(selected)), 4),
            "coverage_util": round(float(coverage.value(selected)), 4),
            "fairness_util": round(float(fair.value(selected)), 4),
            "crb": round(float((scn.w * sensing.crb(selected)).sum()), 5),
            "agg_mse": round(float(mse), 8),
            "energy": round(float(el["energy"]), 6), "latency": round(float(el["latency"]), 6),
            "probe_time": round(probe_time, 4), "select_time": round(sel_time, 5),
            "train_time": round(train_time, 4), "round_time": round(probe_time + sel_time + train_time, 4),
            # ---- CloakFL privacy metrics ----
            "leak_r_median": round(bs_sum["leak_r_median"], 5),
            "leak_r_min": round(bs_sum["leak_r_min"], 5),
            "leak_trace_max": round(bs_sum["leak_trace_max"], 6),
            "eaves_r_median": round(float(np.median(eve_r)), 5),
            "r_floor": float(r_floor), "cloak_mode": mode, "relaxed_steps": int(relaxed),
        }
        rows.append(row)
        cmap.update(selected, scn.C)
        fair.update(selected)
        if path is not None:
            save_unit(path, _meta(cfg, method, seed, point, tag, scn, mode, r_floor), rows, complete=False)

    objectives = _cloak_objectives(rows, participation, K, r_floor, mode)
    if path is not None:
        save_unit(path, _meta(cfg, method, seed, point, tag, scn, mode, r_floor),
                  rows, complete=True, objectives=objectives)
    return rows, objectives


def _meta(cfg, method, seed, point, tag, scn, mode, r_floor):
    return {"method": method, "seed": int(seed), "point": point, "tag": tag, "K": scn.K,
            "budget": int(cfg.network.budget), "rounds": int(cfg.fl.rounds),
            "dataset": cfg.fl.dataset, "model": cfg.fl.model,
            "cloak_mode": mode, "r_floor": float(r_floor), "program": "cloak"}


def _cloak_objectives(rows, participation, K, r_floor, mode):
    accs = [r["test_acc"] for r in rows]
    jain = float(participation.sum() ** 2 / (K * np.square(participation).sum() + 1e-12))
    return {
        "acc": float(accs[-1]), "best_acc": float(max(accs)),
        "logdet": float(np.mean([r["sensing_logdet"] for r in rows])),
        "logdet_final": float(rows[-1]["sensing_logdet"]),
        "crb": float(np.mean([r["crb"] for r in rows])),
        "crb_final": float(rows[-1]["crb"]),
        "agg_mse": float(np.mean([r["agg_mse"] for r in rows])),
        "energy": float(np.mean([r["energy"] for r in rows])),
        "jain": jain,
        "leak_r_median": float(rows[-1]["leak_r_median"]),
        "leak_r_min": float(rows[-1]["leak_r_min"]),
        "eaves_r_median": float(rows[-1]["eaves_r_median"]),
        "r_floor": float(r_floor), "cloak_mode": mode,
    }

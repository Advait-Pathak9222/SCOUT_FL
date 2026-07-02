"""TEMPO-FL training loop: mobility + Kalman tracker + schedule/controller + selection.

One resumable unit = one (point, method, seed) full FL training under a temporal
learning/sensing policy. Reuses the campaign's FL primitives (probe / local_train /
AirComp aggregate / evaluate) unchanged; the only differences vs run_fl_synthetic are:
  * targets move each round (CVMobility) and the per-target FIM is recomputed;
  * a per-target information-filter Kalman tracker maintains P_{t,m} and tracking RMSE;
  * selection uses the time-varying MixedUtility whose weights come from an open-loop
    Schedule or a closed-loop Controller (Threshold / DPP / MPC);
  * per-round rows carry lambda_s(t), tr(P), tracking RMSE, constraint-violation.

Artifacts match utils/runstore schema, so analysis/collect + report_common load them.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from scout_fl.experiments.run_fl_synthetic import _physical_params, _subsample
from scout_fl.experiments.run_synthetic import build_scenario
from scout_fl.fl.aggregation import aggregate
from scout_fl.fl.client import local_train, probe_loss_and_embedding
from scout_fl.fl.datasets import build_client_datasets
from scout_fl.fl.models import build_model, num_params
from scout_fl.fl.partitioning import partition, partition_spatial
from scout_fl.fl.server import FLServer
from scout_fl.infra.mobility import CVMobility, cv_matrices
from scout_fl.infra.tracker import InformationKalmanTracker
from scout_fl.objectives.learning_utility import LearningUtility
from scout_fl.objectives.sensing_utility import SensingUtility
from scout_fl.sim.aircomp import aggregation_mse
from scout_fl.sim.channel import comm_channel_gains
from scout_fl.sim.energy_latency import round_energy_latency
from scout_fl.sim.fim import per_client_target_fim
from scout_fl.sim.geometry import pairwise_geometry
from scout_fl.sim.sensing import sensing_snr
from scout_fl.selection.scout_greedy import ScoutGreedy
from scout_fl.tempo.controllers import Controller, ControlContext
from scout_fl.tempo.mixed_utility import MixedUtility
from scout_fl.tempo.schedules import Schedule
from scout_fl.utils.device import resolve_device
from scout_fl.utils.runstore import load_unit, save_unit, unit_path
from scout_fl.utils.seed import seed_everything

_TEMPO_DIAG = ["lambda_s", "trP_mean", "trP_worst", "track_rmse_mean", "track_rmse_worst",
               "constraint_violated", "sigma_p", "crb_final", "tempo_tau_eff",
               "tempo_Zmax", "tempo_mpc_kind"]


def _fim_at(clients, targets, snr, cfg):
    geom = pairwise_geometry(clients, targets)
    return per_client_target_fim(geom, snr, cfg.sensing.k_range, cfg.sensing.k_angle)


def _sensing_at(clients, targets, cfg):
    geom = pairwise_geometry(clients, targets)
    rcs = np.ones(targets.shape[0], dtype=float)
    return sensing_snr(geom, cfg.sensing.ref_snr_db, cfg.sensing.pathloss_exponent,
                       rcs=rcs, ref_distance=cfg.sensing.ref_distance)


def run_tempo_seed(cfg, ds, seed, method, schedule: Optional[Schedule] = None,
                   controller: Optional[Controller] = None, *, sigma_p=0.05,
                   p_max=None, mission="sustained", runs_root=None, tag="tempo", point="base",
                   sigma_p_ctrl=None, l_noise=0.0, inner="v2"):
    """Run one TEMPO unit; return (rows, objectives). Resumable via runs_root.

    E-T5 robustness knobs (design §1.5):
      sigma_p_ctrl — the CONTROLLER/TRACKER's believed process-noise std (mis-specified
                     Q ablation: e.g. 2x/0.5x the true sigma_p); ground-truth mobility
                     always uses the true sigma_p. None -> correctly specified.
      l_noise      — std of multiplicative log-normal noise on the L_t estimate the
                     controller sees (noisy learning-state ablation). 0 -> exact.
      inner        — 'v2' (default: greedy + soft primal-dual AirComp-MSE penalty, the
                     SCOUT-FL v2 machinery) or 'plain' (greedy on the mixed utility with
                     NO MSE penalty) — the inner-selector swap ablation showing the
                     schedule, not the inner selector, carries the gain.
    """
    path = unit_path(runs_root, tag, point, method, seed) if runs_root else None
    if path is not None:
        cached = load_unit(path)
        if cached is not None:
            print(f"  [resume] {tag}/{point}/{method} seed{seed}: {len(cached['rounds'])} rounds")
            return cached["rounds"], cached["objectives"]

    device = resolve_device(cfg.fl.get("device", "auto"))
    rng = seed_everything(int(seed))
    scn = build_scenario(cfg, rng)
    K, budget, rounds = scn.K, int(cfg.network.budget), int(cfg.fl.rounds)

    # channel gains (same construction as run_seed)
    phys_cfg = cfg.get("physical", {})
    g = comm_channel_gains(
        scn.clients, np.asarray(cfg.geometry.bs_position, dtype=float), rng,
        snr_ref_db=cfg.channel.snr_ref_db, ref_distance=cfg.channel.reference_distance,
        pathloss_exponent=cfg.channel.pathloss_exponent, model=cfg.channel.model,
        rician_k_db=cfg.channel.rician_k_db,
        pathloss_model=("physical" if phys_cfg and phys_cfg.get("enabled") else "reference_snr"),
        carrier_ghz=float(phys_cfg.get("carrier_ghz", 3.5)) if phys_cfg else 3.5)
    if getattr(scn, "compute_het", None) is None:
        scn.compute_het = rng.uniform(0.1, 1.0, scn.K)

    # data
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
    mse_eps = cfg.constraints.mse_agg_max
    aircomp_on = bool(cfg.aircomp.enabled)
    ota_on = bool(cfg.aircomp.get("ota_distortion", False))
    ota_scale = float(cfg.aircomp.get("ota_noise_scale", 1.0))
    from scout_fl.objectives.primal_dual import DualState
    duals = DualState({"mse": mse_eps}, lr=float(cfg.constraints.get("dual_lr", 0.5)))

    targets0 = np.asarray(scn.targets, dtype=float)
    area = np.asarray(cfg.network.area_size, dtype=float)
    mobility = CVMobility(targets0, sigma_p, rng, area=area)        # ground truth: TRUE sigma_p
    sp_belief = float(sigma_p if sigma_p_ctrl is None else sigma_p_ctrl)
    tracker = InformationKalmanTracker(targets0, sp_belief, rng)    # estimator: BELIEVED Q
    _, Q_belief = cv_matrices(sp_belief)
    q_growth = float(np.trace(Q_belief[:2, :2]))
    if p_max is None:
        p_max = float(cfg.get("tempo", {}).get("p_max", 20.0))
    l_noise_rng = np.random.default_rng(int(seed) + 7919) if l_noise > 0 else None

    participation = np.zeros(K)
    L_t = 0.0
    inj_hat = 1.0
    inj_seeded = False
    rows = []

    for t in range(rounds):
        if t > 0:
            mobility.step()
            tracker.predict()
        true_pos = mobility.positions                       # (M, 2) ground truth this round
        snr = _sensing_at(scn.clients, true_pos, cfg)
        fim = _fim_at(scn.clients, true_pos, snr, cfg)
        sensing = SensingUtility(fim, scn.j0, scn.w)
        trP = tracker.trace_pos()                           # (M,) predicted covariance trace

        # bootstrap injection estimate (once), for the adaptive threshold controller
        if not inj_seeded:
            top = np.argsort(-snr.sum(axis=1))[:budget]
            J_sel = fim[top].sum(axis=0)                     # (M,2,2)
            red = tracker.predicted_trace_reduction(J_sel)
            inj_hat = float(max(np.mean(red), 1e-6))
            inj_seeded = True

        # --- probe clients (loss + gradient embedding) ---
        tic = time.perf_counter()
        g_flat = server.global_flat()
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

        # --- policy: weights for this round ---
        # noisy-L_t ablation: the controller observes a corrupted learning state
        L_obs = L_t if l_noise_rng is None else float(L_t * l_noise_rng.lognormal(0.0, l_noise))
        if controller is not None:
            ctx = ControlContext(t=t, T=rounds, trP=trP, L_t=L_obs, p_max=p_max, M=scn.M,
                                 inj_hat=inj_hat, q_growth=q_growth)
            decision = controller.decide(ctx)
            w_learn, w_sense, lam = decision.w_learn, decision.w_sense, decision.lam
            pol_diag = decision.diag
        else:
            lam = schedule(t, rounds)
            w_learn = 1.0 - lam
            w_sense = np.full(scn.M, lam / max(scn.M, 1))
            pol_diag = {"tempo_mode": "schedule"}

        # --- select via greedy on the mixed utility (soft AirComp-MSE penalty, v2-style) ---
        tic = time.perf_counter()
        mixed = MixedUtility(learning, fim, scn.j0, w_learn, w_sense, K)

        def penalty_fn(S, k):
            mse_k = aggregation_mse(g, S + [k], power=P, sigma2=sigma2)
            return duals.mu.get("mse", 0.0) * max(0.0, mse_k - (mse_eps or 0.0))

        use_penalty = aircomp_on and inner != "plain"          # inner-selector swap ablation
        res = ScoutGreedy(use_lazy=False).select(utility=mixed, num_clients=K, budget=budget,
                                                 penalty_fn=penalty_fn if use_penalty else None)
        sel_time = time.perf_counter() - tic
        selected = res.selected
        participation[selected] += 1

        # --- local training + AirComp aggregation ---
        tic = time.perf_counter()
        updates, counts, train_losses = [], [], []
        for k in selected:
            server.set_global(g_flat)
            out = local_train(server.model, client_datasets[k], epochs=int(cfg.fl.local_epochs),
                              lr=float(cfg.fl.lr), batch_size=int(cfg.fl.batch_size),
                              optimizer=cfg.fl.optimizer, device=device)
            updates.append(out["update"]); counts.append(out["num_samples"]); train_losses.append(out["loss"])
        train_time = time.perf_counter() - tic

        mse = aggregation_mse(g, selected, power=P, sigma2=sigma2) if aircomp_on else 0.0
        if aircomp_on:
            duals.update({"mse": float(mse)})
        agg = aggregate(updates, counts, ota=ota_on, mse=mse, scale=ota_scale, rng=rng)
        server.apply_aggregated_update(g_flat, agg)
        test_loss, test_acc = server.evaluate(x_te, y_te)

        # --- tracker update with the selected set's FIM increment ---
        J_sel_pos = fim[selected].sum(axis=0) if selected else np.zeros((scn.M, 2, 2))
        trP_before = tracker.trace_pos()
        tracker.update(J_sel_pos, true_pos)
        trP_after = tracker.trace_pos()
        rmse = tracker.rmse(true_pos)
        realized_red = float(np.mean(trP_before - trP_after))
        if lam > 0.05 and realized_red > 0:                 # EMA the injection estimate
            inj_hat = 0.8 * inj_hat + 0.2 * realized_red

        # --- learning energy state L_t (EMA of ||agg||^2) ---
        grad_sq = float(np.dot(agg, agg))
        L_t = grad_sq if t == 0 else 0.7 * L_t + 0.3 * grad_sq

        el = round_energy_latency(selected, g, power=P, sigma2=sigma2,
                                  bandwidth=cfg.aircomp.bandwidth, model_bits=model_bits,
                                  cpu_cycles=cpu_cycles, cpu_freq=cfg.energy.cpu_freq,
                                  kappa=cfg.energy.kappa, e_sense=cfg.energy.e_sense,
                                  t_sense=cfg.energy.t_sense)
        crb_now = float((scn.w * sensing.crb(selected)).sum())
        trP_worst = float(np.max(trP_after))
        row = {
            "method": method, "round": t,
            "train_loss": round(float(np.mean(train_losses)), 5),
            "test_loss": round(float(test_loss), 5), "test_acc": round(float(test_acc), 5),
            "selected": list(selected),
            "learning_util": round(float(learning.value(selected)), 4),
            "sensing_logdet": round(float(sensing.value(selected)), 4),
            "coverage_util": 0.0, "fairness_util": 0.0,
            "crb": round(crb_now, 5),
            "agg_mse": round(float(mse), 8), "grad_sq": round(grad_sq, 8),
            "dual_mse": round(float(duals.mu.get("mse", 0.0)), 6),
            "mse_violation": round(float(max(0.0, mse - (mse_eps or 0.0))), 8),
            "energy": round(float(el["energy"]), 6), "latency": round(float(el["latency"]), 6),
            "probe_time": round(probe_time, 4), "select_time": round(sel_time, 5),
            "train_time": round(train_time, 4), "round_time": round(probe_time + sel_time + train_time, 4),
            # ---- TEMPO diagnostics ----
            "lambda_s": round(float(lam), 4),
            "trP_mean": round(float(np.mean(trP_after)), 5),
            "trP_worst": round(trP_worst, 5),
            "track_rmse_mean": round(float(np.mean(rmse)), 5),
            "track_rmse_worst": round(float(np.max(rmse)), 5),
            "constraint_violated": int(trP_worst > p_max),
            "sigma_p": float(sigma_p),
        }
        for kk in ("tempo_tau_eff", "tempo_Zmax", "tempo_mpc_kind"):
            row[kk] = pol_diag.get(kk, 0.0)
        rows.append(row)

        if controller is not None:
            controller.observe(ControlContext(t=t, T=rounds, trP=trP_after, L_t=L_obs,
                                              p_max=p_max, M=scn.M, inj_hat=inj_hat, q_growth=q_growth))
        meta = _meta(cfg, method, seed, point, tag, scn, sigma_p, mission, p_max,
                     sp_belief, l_noise, inner)
        if path is not None:
            save_unit(path, meta, rows, complete=False)

    objectives = _tempo_objectives(rows, participation, K, p_max)
    if path is not None:
        save_unit(path, _meta(cfg, method, seed, point, tag, scn, sigma_p, mission, p_max,
                              sp_belief, l_noise, inner),
                  rows, complete=True, objectives=objectives)
    return rows, objectives


def _meta(cfg, method, seed, point, tag, scn, sigma_p, mission, p_max,
          sp_belief=None, l_noise=0.0, inner="v2"):
    return {"method": method, "seed": int(seed), "point": point, "tag": tag, "K": scn.K,
            "budget": int(cfg.network.budget), "rounds": int(cfg.fl.rounds),
            "dataset": cfg.fl.dataset, "model": cfg.fl.model,
            "sigma_p": float(sigma_p), "mission": mission, "p_max": float(p_max),
            "sigma_p_ctrl": float(sp_belief if sp_belief is not None else sigma_p),
            "l_noise": float(l_noise), "inner": str(inner),
            "program": "tempo"}


def _tempo_objectives(rows, participation, K, p_max):
    accs = [r["test_acc"] for r in rows]
    trP = np.array([r["trP_worst"] for r in rows])
    win = 20
    worst_window = float(max((trP[i:i + win].mean() for i in range(max(1, len(trP) - win + 1))),
                             default=float(trP.mean())))
    jain = float(participation.sum() ** 2 / (K * np.square(participation).sum() + 1e-12))
    return {
        "acc": float(accs[-1]), "best_acc": float(max(accs)),
        "logdet": float(np.mean([r["sensing_logdet"] for r in rows])),
        "logdet_final": float(rows[-1]["sensing_logdet"]),
        "crb": float(np.mean([r["crb"] for r in rows])),
        "crb_final": float(rows[-1]["crb"]),
        "agg_mse": float(np.mean([r["agg_mse"] for r in rows])),
        "energy": float(np.mean([r["energy"] for r in rows])),
        "round_s": float(np.mean([r["round_time"] for r in rows])),
        "jain": jain,
        "time_avg_trP": float(trP.mean()),
        "worst_window_trP": worst_window,
        "final_trP": float(trP[-1]),
        "track_rmse": float(np.mean([r["track_rmse_worst"] for r in rows])),
        "track_rmse_final": float(rows[-1]["track_rmse_worst"]),
        "pct_violation": float(np.mean([r["constraint_violated"] for r in rows])),
        "p_max": float(p_max),
    }

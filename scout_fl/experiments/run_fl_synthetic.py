"""A1-Full SCOUT-FL: first end-to-end federated training run on MNIST/Fashion.

Each round: probe all clients (cheap gradient embedding + local loss) -> build
the composite SCOUT-FL utility (real learning + sensing log-det FIM + coverage/
freshness + fairness) -> select under the AirComp-MSE feasibility gate -> train
the selected clients locally -> FedAvg (optionally OTA-distorted by the AirComp
MSE) -> evaluate -> update coverage/fairness state -> log FL + ISAC metrics.

Runs each selection method (SCOUT-FL + minimal baselines) as its own federated
training for a first comparison.

Run:
  python -m scout_fl.experiments.run_fl_synthetic --config scout_fl/configs/fl_synthetic_small.yaml [--quick]
  make fl-synthetic-small
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import torch
import yaml

from scout_fl.experiments.run_synthetic import build_scenario
from scout_fl.fl.aggregation import aggregate
from scout_fl.fl.client import local_train, probe_loss_and_embedding
from scout_fl.fl.datasets import build_client_datasets, load_fl_dataset
from scout_fl.fl.models import build_model, num_params
from scout_fl.analysis.pareto import hypervolume, normalize_objectives, pareto_front, per_method_volume
from scout_fl.fl.partitioning import partition, partition_report, partition_spatial
from scout_fl.fl.server import FLServer
from scout_fl.objectives.coverage_utility import CoverageMap, CoverageUtility
from scout_fl.objectives.fairness_utility import FairnessUtility
from scout_fl.objectives.joint_information import JointInformationUtility
from scout_fl.objectives.primal_dual import DualState, ParticipationDual
from scout_fl.objectives.twin import ResidualTwin
from scout_fl.objectives.learning_utility import LearningUtility
from scout_fl.objectives.sensing_utility import SensingUtility
from scout_fl.objectives.total_utility import TotalUtility
from scout_fl.selection.baselines import BASELINE_REGISTRY
from scout_fl.selection.generative_model import ClientGenerativeModel
from scout_fl.selection.loss_based import LossSelector
from scout_fl.selection.random import RandomSelector
from scout_fl.selection.scout_greedy import ScoutGreedy
from scout_fl.selection.snr_based import SNRSelector
from scout_fl.sim.aircomp import aggregation_mse, min_gain_for_mse
from scout_fl.sim.channel import comm_channel_gains
from scout_fl.sim.energy_latency import round_energy_latency
from scout_fl.utils.config import load_config, to_plain
from scout_fl.utils.device import describe_device, resolve_device
from scout_fl.utils.logging_utils import RunLogger
from scout_fl.utils.runstore import (load_unit, participation_from_rows, save_unit,
                                      unit_path)
from scout_fl.utils.seed import seed_everything


def _apply_quick(cfg) -> None:
    """Shrink everything for a fast smoke run."""
    cfg.network.num_clients = 10
    cfg.network.budget = 3
    cfg.fl.rounds = 3
    cfg.fl.subsample_train = 2000
    cfg.fl.subsample_test = 1000
    cfg.fl.local_epochs = 1


def _subsample(x, y, n, rng):
    if not n or n >= len(y):
        return x, y
    idx = rng.choice(len(y), size=int(n), replace=False)
    return x[idx], y[idx]


def _zscore(a):
    """Standardize a vector (zero mean, unit std); all-zeros if constant."""
    a = np.asarray(a, dtype=float)
    s = a.std()
    return (a - a.mean()) / s if s > 1e-9 else np.zeros_like(a)


def _physical_params(cfg, n_params, avg_local_samples):
    """Genuine physical units from the link budget (or None if physical mode is off):
    sigma^2 = thermal noise k_B*T*F*B (W); P from dBm (W); model payload = #params*bits;
    compute = cycles_per_sample * local_samples * epochs (cycles). Energy->J, latency->s."""
    phys = cfg.get("physical", {})
    if not phys or not phys.get("enabled"):
        return None
    from scout_fl.sim.link_budget import dbm_to_watt, thermal_noise_power_w
    B = float(cfg.aircomp.bandwidth)
    sigma2 = thermal_noise_power_w(B, float(phys.get("noise_figure_db", 7.0)),
                                   float(phys.get("temperature_k", 290.0)))
    P = dbm_to_watt(float(phys.get("tx_power_dbm", 0.0)))
    model_bits = float(n_params) * float(phys.get("bits_per_param", 32))
    cpu_cycles = (float(phys.get("cycles_per_sample", 1.0e6)) * float(max(avg_local_samples, 1.0))
                  * float(cfg.fl.get("local_epochs", 1)))
    return {"power": P, "sigma2": sigma2, "model_bits": model_bits, "cpu_cycles": cpu_cycles}


# ISCC system-baseline variants: AirComp distortion is intrinsic to OTA-FL / ISCC methods,
# while Random-K FedAvg uses ideal (digital) aggregation -> isolates the AirComp effect.
_OTA_FORCE_ON = {"ota_fedavg", "fedavg_iscc", "fedsgd_iscc", "fed_iscc"}
_OTA_FORCE_OFF = {"random"}
# FedSGD-ISCC uses the single-step (FedSGD) local-update rule; the rest use FedAvg (E epochs).
_FEDSGD_METHODS = {"fedsgd_iscc"}

# VISMAYA-FL ablation variants — three component knockouts for the ablation study.
# Each maps to keyword overrides passed to ClientGenerativeModel.__init__.
# The make-or-break experiment: vismaya vs vismaya_no_syn under process_noise > 0
# (mobility). If Syn contributes, the full method should clearly outperform no-Syn
# on CRB under target movement and on test_acc under sensing-learning drift.
_VISMAYA_ABLATIONS = {
    "vismaya_no_syn":     {"beta": 0.0},                        # Ω_S + Ω_L, no synergy
    "vismaya_sense_only": {"beta": 0.0, "rho_v": 0.0},         # Ω_S only (≈ Kalman-CRB selector)
    "vismaya_learn_only": {"beta": 0.0, "sense_scale": 0.0},   # Ω_L only (≈ EMC without sensing)
}

# JEDI-FL ablation variants (component knockouts) for the ablation study. Keys prefixed
# with "_" are handled in the runner (fairness/gate); the rest pass to JointInformationUtility.
_JEDI_ABLATIONS = {
    "jedi_no_sensing":     {"use_sensing": False},     # drop the sensing block
    "jedi_no_learning":    {"use_learning": False},    # drop the learning block
    "jedi_no_coverage":    {"use_coverage": False},    # drop coverage/freshness
    "jedi_no_kappa":       {"use_kappa": False},       # MSE NOT observation noise (kappa=1)
    "jedi_no_externality": {"externality": False},     # additive marginal (Shapley-style)
    "jedi_fixed_rho":      {"rho": 1.0},               # no auto-normalizer (fixed scalarization)
    "jedi_no_fairness":    {"_no_fairness": True},     # no participation dual (deficit=0)
    "jedi_hard_gate":      {"use_kappa": False, "_hard_gate": True},  # SCOUT-v1-style MSE gate
    "jedi_twin":           {"_use_twin": True},        # + trust-gated learned-residual twin
}

# JEDI-FL per-round diagnostics for the paper figures: the information-block split
# (emergent learning->sensing schedule), the participation-deficit (fairness) state, and
# the trust-gated twin's trust + surrogate-validity correlation.
_JEDI_DIAG_KEYS = ["jedi_sense_info", "jedi_learn_info", "jedi_fair_bonus",
                   "jedi_learn_frac", "jedi_deficit_mean", "jedi_deficit_max",
                   "jedi_twin_trust", "jedi_twin_corr"]
_VISMAYA_DIAG_KEYS = ["vis_omega_s_mean", "vis_omega_s_max",
                      "vis_synergy_mean", "vis_p_trace_mean", "vis_n_seen_frac"]


def _jedi_diagnostics(joint, fair_dual, sensing, coverage, learning, selected,
                      twin_trust=0.0, twin_corr=0.0) -> dict:
    """Decompose JEDI's objective on the selected set into its information blocks.

    * ``jedi_sense_info``  = sensing + coverage log-det information of S (nats);
    * ``jedi_learn_info``  = rho * kappa(MSE) * learning information of S (nats) — the
      AirComp-noise-scaled learning block;
    * ``jedi_learn_frac``  = learn / (learn + sense): the EMERGENT SCHEDULE signal —
      high early (learning dominates), decaying as gradients shrink near convergence
      so sensing/coverage take over, with NO scheduler;
    * ``jedi_fair_bonus``  = total participation-deficit bonus carried by S;
    * ``jedi_deficit_{mean,max}`` = the virtual-queue state (fairness self-correction);
    * ``jedi_twin_trust`` = the residual twin's trust in [0,1] (gates its influence;
      0 => twin ignored), ``jedi_twin_corr`` = its predicted-vs-realized loss-drop
      correlation (surrogate-validity: positive => the twin is predictive).
    """
    sel = list(selected)
    sense_info = float(sensing.value(sel) + coverage.value(sel))
    learn_info = float(joint.rho * joint._kappa(sel) * learning.value(sel))
    fair_bonus = float(joint.fair_unit * fair_dual.deficit[sel].sum())
    denom = learn_info + sense_info
    return {
        "jedi_sense_info": round(sense_info, 4),
        "jedi_learn_info": round(learn_info, 4),
        "jedi_fair_bonus": round(fair_bonus, 4),
        "jedi_learn_frac": round(learn_info / denom, 4) if denom > 0 else 0.0,
        "jedi_deficit_mean": round(float(fair_dual.deficit.mean()), 4),
        "jedi_deficit_max": round(float(fair_dual.deficit.max()), 4),
        "jedi_twin_trust": round(float(twin_trust), 4),
        "jedi_twin_corr": round(float(twin_corr), 4),
    }


def run_one(method, cfg, scn, g, client_datasets, x_test, y_test,
            input_shape, num_classes, base_seed, out_path=None, meta=None):
    """Run a full federated training for one selection ``method``; return rows + participation.

    If ``out_path`` is given, the per-round rows are written to that JSON after EACH
    round (atomic), marked ``complete`` with final objectives at the end — this is the
    resumable per-round result store (see utils/runstore.py)."""
    device = resolve_device(cfg.fl.get("device", "auto"))
    K, budget, rounds = scn.K, int(cfg.network.budget), int(cfg.fl.rounds)
    rng = np.random.default_rng(base_seed)
    torch.manual_seed(base_seed)

    server = FLServer(build_model(cfg.fl.model, input_shape, num_classes), device=device)
    cmap = CoverageMap(scn.R, rho=cfg.coverage.rho, innovation=cfg.coverage.innovation, u_init=1.0)
    fair = FairnessUtility(K)
    sensing = SensingUtility(scn.fim, scn.j0, scn.w)              # static across rounds
    full = list(range(K))

    aircomp_on = bool(cfg.aircomp.enabled)
    # Physical units (genuine link budget) override the normalized P/sigma2/model_bits/cpu_cycles
    # so energy is in Joules, latency in seconds, and P*g/sigma2 is a true SNR (see sim/link_budget.py).
    _avg_local = float(np.mean([len(d) for d in client_datasets])) if client_datasets else 1.0
    phys = _physical_params(cfg, num_params(server.model), _avg_local)
    if phys is not None:
        P, sigma2, model_bits, cpu_cycles = phys["power"], phys["sigma2"], phys["model_bits"], phys["cpu_cycles"]
    else:
        P, sigma2 = float(cfg.aircomp.power), float(cfg.aircomp.sigma2)
        model_bits, cpu_cycles = float(cfg.aircomp.model_bits), float(cfg.energy.cpu_cycles)
    mse_eps = cfg.constraints.mse_agg_max
    duals = DualState({"mse": mse_eps}, lr=float(cfg.constraints.get("dual_lr", 0.5)))  # SCOUT-FL v2
    fair_dual = ParticipationDual(K, budget,                # JEDI participation fairness
                                  lr=float(cfg.objectives.get("fair_dual_lr", 1.0)))
    ota_on = bool(cfg.aircomp.get("ota_distortion", False))
    ota_scale = float(cfg.aircomp.get("ota_noise_scale", 1.0))
    obj = cfg.objectives
    participation = np.zeros(K)
    rows = []

    # Trust-gated learned-residual twin (only active for the jedi_twin ablation / opt-in):
    # predicts realized loss-drop from per-client features; its influence is scaled by a
    # trust in [0,1] that tracks its measured predicted-vs-realized correlation, starting
    # at 0 (a bad/cold twin is ignored, so JEDI is never worse than its analytic form).
    use_twin = bool(_JEDI_ABLATIONS.get(method, {}).get("_use_twin")
                    or (method == "jedi" and cfg.objectives.get("jedi_use_twin", False)))
    twin = ResidualTwin(dim=4, l2=1.0)
    twin_trust, twin_corr, prev_loss = 0.0, 0.0, None
    tw_pred_hist, tw_real_hist = [], []

    # VISMAYA-FL generative model (stateful across rounds; instantiated once per run).
    # Config keys: vismaya.{rho_v, beta, process_noise, ema_alpha}
    # Ablation variants (vismaya_no_syn / vismaya_sense_only / vismaya_learn_only) override
    # specific parameters via _VISMAYA_ABLATIONS but share all other config keys.
    vis_cfg = cfg.get("vismaya", {}) or {}
    if method == "vismaya" or method in _VISMAYA_ABLATIONS:
        _vis_overrides = _VISMAYA_ABLATIONS.get(method, {})
        vis_model = ClientGenerativeModel(
            K, scn.M, scn.fim, scn.j0, scn.w,
            rho_v=float(_vis_overrides.get("rho_v", vis_cfg.get("rho_v", 1.0))),
            beta=float(_vis_overrides.get("beta", vis_cfg.get("beta", 0.3))),
            process_noise=float(vis_cfg.get("process_noise", 0.0)),
            ema_alpha=float(vis_cfg.get("ema_alpha", 0.1)),
            sense_scale=float(_vis_overrides.get("sense_scale", 1.0)),
        )
    else:
        vis_model = None
    vis_feats = None   # populated in the selection block each round

    for t in range(rounds):
        print(f"[run] method={method} seed={base_seed} round={t + 1}/{rounds}", flush=True)
        g_flat = server.global_flat()
        # --- probe every client on the current global model (loss + grad embedding) ---
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

        # --- composite SCOUT-FL utility (real embeddings; no placeholders) ---
        learning = LearningUtility(embeddings=embs)
        coverage = CoverageUtility(cmap.U, scn.C, g=cfg.coverage.saturating)
        weights = {"learning": obj.alpha_learning, "sensing": obj.lambda_sense,
                   "coverage": obj.lambda_coverage, "fairness": obj.lambda_fairness}
        norms = {"learning": max(learning.value(full), 1e-9), "sensing": max(sensing.value(full), 1e-9),
                 "coverage": max(coverage.value(full), 1e-9), "fairness": max(fair.value(full), 1e-9)}
        total = TotalUtility(
            {"learning": learning, "sensing": sensing, "coverage": coverage, "fairness": fair},
            weights=weights, normalizers=norms)

        # per-client latency = comm time (channel-dependent) + compute time (per-client
        # compute-speed heterogeneity) + sensing time. The per-client compute term makes
        # latency NOT a pure function of channel gain, so FedCS/ISCC selectors differ from comm_only.
        _rate = cfg.aircomp.bandwidth * np.log2(1.0 + P * np.asarray(g) / sigma2)
        _het = scn.compute_het if getattr(scn, "compute_het", None) is not None else np.ones(K)
        per_client_latency = (model_bits / np.clip(_rate, 1e-9, None)
                              + cpu_cycles / (cfg.energy.cpu_freq * _het) + cfg.energy.t_sense)

        # trust-gated twin: per-client feature [bias, loss, |grad|, gain] (standardized) ->
        # predicted loss-drop -> bounded relative learning multiplier in [1-trust, 1+trust].
        twin_feats, learn_mult = None, None
        if use_twin:
            twin_feats = np.stack([np.ones(K), _zscore(losses),
                                   _zscore(np.linalg.norm(embs, axis=1)), _zscore(g)], axis=1)
            preds = twin_feats @ twin.w
            learn_mult = 1.0 + twin_trust * np.tanh(_zscore(preds))

        # --- select ---
        diag = dict.fromkeys(_JEDI_DIAG_KEYS, 0.0)             # JEDI-FL diagnostics (paper figures)
        vis_diag = dict.fromkeys(_VISMAYA_DIAG_KEYS, 0.0)      # VISMAYA-FL diagnostics
        tic = time.perf_counter()
        if method == "scout_greedy":
            feasible = None
            if aircomp_on and mse_eps is not None:
                g_min = min_gain_for_mse(mse_eps, budget, P, sigma2)
                feasible = lambda S, k: g[k] >= g_min          # AirComp-MSE gate
            res = ScoutGreedy().select(utility=total, num_clients=K, budget=budget, feasible=feasible)
        elif method == "scout_v2":
            def penalty_fn(S, k):                              # soft primal-dual MSE penalty (no hard gate)
                mse_k = aggregation_mse(g, S + [k], power=P, sigma2=sigma2)
                return duals.mu.get("mse", 0.0) * max(0.0, mse_k - (mse_eps or 0.0))
            res = ScoutGreedy().select(utility=total, num_clients=K, budget=budget, penalty_fn=penalty_fn)
        elif method == "jedi" or method in _JEDI_ABLATIONS:    # joint experimental-design (+ ablations)
            abl = _JEDI_ABLATIONS.get(method, {})
            deficit = np.zeros(K) if abl.get("_no_fairness") else fair_dual.deficit
            jkw = {k: v for k, v in abl.items() if not k.startswith("_")}
            joint = JointInformationUtility(sensing, coverage, learning, deficit, g,
                                            power=P, sigma2=sigma2, learn_mult=learn_mult, **jkw)
            feasible = None
            if abl.get("_hard_gate") and aircomp_on and mse_eps is not None:
                g_min = min_gain_for_mse(mse_eps, budget, P, sigma2)
                feasible = lambda S, k: g[k] >= g_min          # SCOUT-v1-style hard MSE gate
            res = ScoutGreedy(use_lazy=False).select(utility=joint, num_clients=K, budget=budget,
                                                     feasible=feasible)
            diag = _jedi_diagnostics(joint, fair_dual, sensing, coverage, learning, res.selected,
                                     twin_trust=twin_trust, twin_corr=twin_corr)
        elif method == "vismaya" or method in _VISMAYA_ABLATIONS:
            # VISMAYA-FL (full + ablations): top-K by innovation score.
            # Ablations zero out individual terms via sense_scale / rho_v / beta at init time.
            # Build features from current probe: [1, loss_z, grad_norm_z, channel_z, recency]
            vis_feats = vis_model.build_features(
                losses, np.linalg.norm(embs, axis=1), np.asarray(g))
            vis_scores = vis_model.score_all(vis_feats)
            # Primal-dual soft MSE penalty (same as SCOUT-v2; no hard gate)
            mse_k = np.array([aggregation_mse(g, [k], power=P, sigma2=sigma2) for k in range(K)])
            resource_cost = duals.mu.get("mse", 0.0) * np.maximum(0.0, mse_k - (mse_eps or 0.0))
            net_scores = vis_scores - resource_cost
            order = np.argsort(-net_scores)
            selected_arr = sorted(int(order[i]) for i in range(budget))
            res = type("_R", (), {"selected": selected_arr})()
            vis_diag = vis_model.diagnostics()
        elif method == "random":
            res = RandomSelector().select(num_clients=K, budget=budget, rng=rng)
        elif method == "loss":
            res = LossSelector().select(scores=losses, budget=budget)
        elif method == "snr_only":
            res = SNRSelector().select(scores=scn.snr.sum(axis=1), budget=budget)
        elif method in BASELINE_REGISTRY:
            res = BASELINE_REGISTRY[method].select(
                K=K, budget=budget, rng=rng, sensing=sensing, learning=learning,
                g=g, snr_scores=scn.snr.sum(axis=1), losses=losses, embeddings=embs,
                grad_norm=np.linalg.norm(embs, axis=1), participation=participation.copy(),
                age=fair.age.copy(), latency=per_client_latency, P=P, sigma2=sigma2, mse_eps=mse_eps)
        else:
            raise ValueError(f"unknown selection method {method!r}")
        sel_time = time.perf_counter() - tic
        selected = res.selected
        participation[selected] += 1

        # --- local training of selected clients (FedSGD = single step; else FedAvg) ---
        tic = time.perf_counter()
        max_steps = 1 if method in _FEDSGD_METHODS else None
        updates, counts, train_losses = [], [], []
        for k in selected:
            server.set_global(g_flat)
            out = local_train(server.model, client_datasets[k], epochs=int(cfg.fl.local_epochs),
                              lr=float(cfg.fl.lr), batch_size=int(cfg.fl.batch_size),
                              optimizer=cfg.fl.optimizer, device=device, max_steps=max_steps)
            updates.append(out["update"]); counts.append(out["num_samples"]); train_losses.append(out["loss"])
        train_time = time.perf_counter() - tic

        # --- aggregate (FedAvg; OTA-distorted for OTA-FL/ISCC methods) + apply ---
        tic = time.perf_counter()
        ota_this = True if method in _OTA_FORCE_ON else (False if method in _OTA_FORCE_OFF else ota_on)
        mse = aggregation_mse(g, selected, power=P, sigma2=sigma2) if aircomp_on else 0.0
        if method in ("scout_v2", "vismaya", *_VISMAYA_ABLATIONS) and aircomp_on:
            duals.update({"mse": float(mse)})                  # dual ascent on realized violation
        agg = aggregate(updates, counts, ota=ota_this, mse=mse, scale=ota_scale, rng=rng)
        server.apply_aggregated_update(g_flat, agg)
        agg_time = time.perf_counter() - tic

        # --- evaluate + ISAC metrics ---
        test_loss, test_acc = server.evaluate(x_test, y_test)

        # --- twin: learn feature->realized-loss-drop, refresh trust (surrogate validity) ---
        if use_twin:
            delta = max(0.0, prev_loss - test_loss) if prev_loss is not None else 0.0
            for k in selected:
                twin.update(twin_feats[k], delta)
            tw_pred_hist.append(float(np.mean((twin_feats @ twin.w)[selected])))
            tw_real_hist.append(delta)
            if len(tw_real_hist) >= 4:
                ph, rh = tw_pred_hist[-12:], tw_real_hist[-12:]
                if np.std(ph) > 1e-9 and np.std(rh) > 1e-9:
                    c = float(np.corrcoef(ph, rh)[0, 1])
                    if np.isfinite(c):
                        twin_corr = c
                        twin_trust = float(np.clip(c, 0.0, 1.0))  # bad twin (c<=0) -> ignored
            prev_loss = test_loss

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
            # P6 convergence: ||aggregated update||^2 is the per-round descent driver (proxy for
            # the eta/2 ||grad F||^2 term); regressed against agg_mse in analysis/convergence.py.
            "grad_sq": round(float(np.dot(agg, agg)), 8),
            # P2 primal-dual feasibility: the MSE dual mu and the realized constraint violation.
            "dual_mse": round(float(duals.mu.get("mse", 0.0)), 6),
            "mse_violation": round(float(max(0.0, mse - (mse_eps or 0.0))), 8),
            "energy": round(float(el["energy"]), 6), "latency": round(float(el["latency"]), 6),
            "probe_time": round(probe_time, 4), "select_time": round(sel_time, 5),
            "train_time": round(train_time, 4), "agg_time": round(agg_time, 5),
            "round_time": round(probe_time + sel_time + train_time + agg_time, 4),
        }
        row.update(diag)                                       # JEDI-FL diagnostics
        row.update(vis_diag)                                   # VISMAYA-FL diagnostics

        # VISMAYA generative model update: call AFTER agg (need actual grad norms).
        # All ablation variants share the same update logic so their P_m and ridge
        # twin evolve correctly — only the scoring weights differ, not the state dynamics.
        if vis_model is not None and vis_feats is not None:
            _actual_gnorms = np.zeros(K)
            for _i, _k in enumerate(selected):
                _actual_gnorms[_k] = float(np.linalg.norm(updates[_i]))
            vis_model.update(list(selected), vis_feats, _actual_gnorms)

        rows.append(row)
        if out_path is not None:                               # checkpoint after every round (resumable)
            save_unit(out_path, meta or {}, rows, complete=False)

        # --- evolve ISAC state for next round ---
        cmap.update(selected, scn.C)
        fair.update(selected)
        fair_dual.update(selected)                             # JEDI participation deficit

    if out_path is not None:
        save_unit(out_path, meta or {}, rows, complete=True,
                  objectives=_objectives(rows, participation, K))
    return rows, participation


def _objectives(rows, participation, K):
    """Per-run scalar objectives.

    Aggregation convention (deliberate, stated for the paper): ``acc`` is the FINAL-round
    test accuracy (the trained model's quality), while sensing/comm objectives
    (``logdet``/``crb``/``agg_mse``/``energy``) are MEANS over rounds — i.e. the *average
    sustained* sensing/comm quality maintained throughout the mission, matching the
    perception-maintenance framing (the network must keep sensing well WHILE training).
    For transparency the FINAL-round sensing values (``logdet_final``/``crb_final``) are
    also reported so the alternative end-state convention is available; the Pareto/bake-off
    use the round-mean (_OBJ_KEYS)."""
    accs = [r["test_acc"] for r in rows]
    jain = float(participation.sum() ** 2 / (K * np.square(participation).sum() + 1e-12))
    return {"acc": float(accs[-1]), "best_acc": float(max(accs)),
            "logdet": float(np.mean([r["sensing_logdet"] for r in rows])),
            "crb": float(np.mean([r["crb"] for r in rows])),
            "agg_mse": float(np.mean([r["agg_mse"] for r in rows])),
            "jain": jain, "energy": float(np.mean([r["energy"] for r in rows])),
            "round_s": float(np.mean([r["round_time"] for r in rows])),
            "logdet_final": float(rows[-1]["sensing_logdet"]),
            "crb_final": float(rows[-1]["crb"])}


def run_unit(method, cfg, scn, g, client_datasets, x_te, y_te, input_shape, num_classes,
             seed, runs_root=None, tag="run", point="base"):
    """Resumable single (point, method, seed) unit: load from runs/ if complete, else run+save."""
    path = unit_path(runs_root, tag, point, method, seed) if runs_root else None
    if path is not None:
        cached = load_unit(path)
        if cached is not None:
            rows = cached["rounds"]
            part = participation_from_rows(rows, scn.K)
            print(f"  [resume] {point}/{method} seed{seed}: loaded {len(rows)} rounds from {path}")
            return rows, part, cached["objectives"]
    print(f"  [start] {point}/{method} seed{seed}: {int(cfg.fl.rounds)} rounds", flush=True)
    meta = {"method": method, "seed": int(seed), "point": point, "tag": tag, "K": scn.K,
            "budget": int(cfg.network.budget), "rounds": int(cfg.fl.rounds),
            "dataset": cfg.fl.dataset, "model": cfg.fl.model}
    rows, part = run_one(method, cfg, scn, g, client_datasets, x_te, y_te,
                         input_shape, num_classes, base_seed=seed, out_path=path, meta=meta)
    return rows, part, _objectives(rows, part, scn.K)


def run_seed(cfg, ds, seed, runs_root=None, tag="run", point="base"):
    """Run every selection method for one seed; return per-method objectives + rows + report."""
    rng = seed_everything(seed)
    scn = build_scenario(cfg, rng)
    chan_source = cfg.channel.get("source", "synthetic")
    if chan_source != "synthetic":
        # real per-client comm channel gains from an external dataset (synthetic fallback)
        from scout_fl.fl.datasets_external import load_channel_realizations
        g = load_channel_realizations(chan_source, scn.K, rng, root=cfg.fl.get("data_root", "data"))
    else:
        _phys = cfg.get("physical", {})
        g = comm_channel_gains(scn.clients, np.asarray(cfg.geometry.bs_position, dtype=float), rng,
                               snr_ref_db=cfg.channel.snr_ref_db, ref_distance=cfg.channel.reference_distance,
                               pathloss_exponent=cfg.channel.pathloss_exponent,
                               model=cfg.channel.model, rician_k_db=cfg.channel.rician_k_db,
                               pathloss_model=("physical" if _phys and _phys.get("enabled") else "reference_snr"),
                               carrier_ghz=float(_phys.get("carrier_ghz", 3.5)) if _phys else 3.5)
    # per-client compute-speed heterogeneity (stragglers): makes per-client latency depend on
    # COMPUTE as well as channel gain, so resource-aware baselines (FedCS, FedAVG/FedSGD-ISCC)
    # are genuinely distinct from communication-only selection. Drawn AFTER g so the channel
    # realization is unchanged. Shared across methods within a seed (fair comparison).
    if getattr(scn, "compute_het", None) is None:
        scn.compute_het = rng.uniform(0.1, 1.0, scn.K)   # up to 10x straggler spread
    x_tr, y_tr = _subsample(ds.x_train, ds.y_train, cfg.fl.get("subsample_train"), rng)
    x_te, y_te = _subsample(ds.x_test, ds.y_test, cfg.fl.get("subsample_test"), rng)
    if cfg.fl.non_iid == "spatial":
        parts = partition_spatial(np.asarray(y_tr), scn.cluster_assignment,
                                  cfg.fl.get("dirichlet_alpha", 0.5), np.random.default_rng(seed))
    else:
        parts = partition(np.asarray(y_tr), scn.K, cfg.fl.non_iid,
                          cfg.fl.get("dirichlet_alpha", 0.5), np.random.default_rng(seed), min_size=1)
    client_datasets = build_client_datasets(x_tr, y_tr, parts)
    report = partition_report(np.asarray(y_tr), parts, ds.num_classes)

    objs, rows_all = {}, []
    for method in cfg.selection.get("methods", ["scout_greedy"]):
        rows, _participation, objs_m = run_unit(
            method, cfg, scn, g, client_datasets, x_te, y_te, ds.input_shape, ds.num_classes,
            seed, runs_root=runs_root, tag=tag, point=point)
        for r in rows:
            r2 = dict(r); r2["seed"] = seed; rows_all.append(r2)
        objs_m = dict(objs_m); objs_m["seed"] = int(seed)      # carry seed for paired-test alignment
        objs[method] = objs_m
    return objs, rows_all, report


_OBJ_KEYS = ["acc", "logdet", "crb", "agg_mse", "jain"]
_OBJ_DIRS = [1, 1, -1, -1, 1]      # +1 higher-better, -1 lower-better
_AGG_KEYS = ("acc", "best_acc", "logdet", "crb", "agg_mse", "jain", "energy", "round_s")


def run_bakeoff(cfg, ds, seeds, runs_root=None, tag="run", point="base"):
    """Run every method for every seed; return (per_method objectives, all rows, partition report).

    ``runs_root`` enables the resumable per-round JSON store: completed (point, method,
    seed) units are loaded and skipped; the process can be killed and re-run safely."""
    per_method = defaultdict(list)
    all_rows, report = [], None
    for seed in seeds:
        objs, rows, report = run_seed(cfg, ds, seed, runs_root=runs_root, tag=tag, point=point)
        all_rows.extend(rows)
        for m, o in objs.items():
            per_method[m].append(o)
    return per_method, all_rows, report


def aggregate_results(per_method):
    """Mean/std per method + Pareto (normalized volume, non-dominated mask) over the objectives."""
    methods = list(per_method)
    agg = {m: {k: (float(np.mean([o[k] for o in per_method[m]])),
                   float(np.std([o[k] for o in per_method[m]]))) for k in _AGG_KEYS}
           for m in methods}
    mat = np.array([[agg[m][k][0] for k in _OBJ_KEYS] for m in methods])
    norm = normalize_objectives(mat, _OBJ_DIRS)
    vol = per_method_volume(norm)
    nd = pareto_front(norm)
    pareto = {m: {"aggregate_volume": round(float(vol[i]), 4), "pareto_optimal": bool(nd[i])}
              for i, m in enumerate(methods)}
    return methods, agg, norm, vol, nd, pareto


def main() -> None:
    parser = argparse.ArgumentParser(description="SCOUT-FL multi-seed Pareto bake-off")
    parser.add_argument("--config", default="scout_fl/configs/fl_synthetic_small.yaml")
    parser.add_argument("--override", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true", help="tiny fast smoke run")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    if args.quick:
        _apply_quick(cfg)
    device = resolve_device(cfg.fl.get("device", "auto"))
    print(f"[device] requested={cfg.fl.get('device', 'auto')} -> using {device} ({describe_device(device)})")
    seeds = [int(s) for s in (cfg.get("seeds") or [int(cfg.get("seed", 0))])]
    if args.quick:
        seeds = seeds[:2]
    logger = RunLogger(cfg.get("output_dir", "outputs"), "fl_synthetic", seeds[0], to_plain(cfg))
    with logger.path("config_used.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(to_plain(cfg), fh, sort_keys=False)

    ds = load_fl_dataset(cfg.fl.dataset, root=cfg.fl.data_root, download=bool(cfg.fl.download))

    runs_root = cfg.get("runs_dir", "runs")                    # resumable per-round JSON store
    tag = str(cfg.get("experiment", "fl_synthetic"))
    print(f"[runs] resumable store: {runs_root}/{tag}/base/  (re-run to resume; delete to recompute)")
    per_method, all_rows, report = run_bakeoff(cfg, ds, seeds, runs_root=runs_root, tag=tag, point="base")
    methods, agg, norm, vol, nd, pareto = aggregate_results(per_method)

    logger.save_csv("metrics.csv", _subset(all_rows, list(all_rows[0].keys())))
    logger.save_json("summary.json", {"seeds": seeds, "aggregate": agg, "pareto": pareto,
                                       "set_hypervolume": round(float(hypervolume(norm)), 4)})
    logger.save_json("partition.json", report)
    logger.save_json("per_seed.json", {m: per_method[m] for m in methods})  # for analysis/stats.py
    _print_bakeoff(logger, cfg, report, methods, agg, pareto, seeds)
    _decide_scout(methods, agg, pareto, vol)
    _maybe_stats(logger, methods, per_method)
    _maybe_plot(cfg, logger, methods, agg, norm, nd)
    _maybe_plot_jedi(cfg, logger, all_rows)


def _subset(rows, cols):
    return [{c: (str(r[c]) if c == "selected" else r.get(c)) for c in cols} for r in rows]


def _print_bakeoff(logger, cfg, report, methods, agg, pareto, seeds):
    print("\n=== SCOUT-FL bake-off (multi-seed, Pareto) ===")
    print(f"run dir: {logger.dir}")
    print(f"dataset={cfg.fl.dataset} model={cfg.fl.model} non_iid={cfg.fl.non_iid} "
          f"layout={cfg.geometry.get('layout', 'random')} | K={cfg.network.num_clients} "
          f"budget={cfg.network.budget} rounds={cfg.fl.rounds} seeds={seeds}")
    print(f"partition top-class-fraction={report['mean_top_class_fraction']:.3f}\n")
    print(f"  {'method':>13} | {'acc':>13} | {'logdet↑':>14} | {'CRB↓':>15} | "
          f"{'Jain↑':>13} | {'aggVol↑':>7} | {'Pareto':>6}")
    for m in methods:
        a = agg[m]
        print(f"  {m:>13} | {a['acc'][0]:.3f}±{a['acc'][1]:.3f} | "
              f"{a['logdet'][0]:8.2f}±{a['logdet'][1]:4.2f} | {a['crb'][0]:8.3f}±{a['crb'][1]:5.3f} | "
              f"{a['jain'][0]:.3f}±{a['jain'][1]:.3f} | {pareto[m]['aggregate_volume']:>7} | "
              f"{('  ✓' if pareto[m]['pareto_optimal'] else '  ·'):>6}")
    print("\naggVol = MEAN of min-max-normalized (acc, logdet, -CRB, -MSE, Jain); higher=all-round "
          "better. Pareto ✓ = non-dominated across those objectives. (Read per-axis too.)")


def _decide_scout(methods, agg, pareto, vol):
    idx = {m: i for i, m in enumerate(methods)}
    scout = [m for m in ("scout_greedy", "scout_v2") if m in idx]
    base = [m for m in ("random", "snr_only", "loss", "fedavg_iscc", "fixed_weighted") if m in idx]
    if not scout or not base:
        print("\n[decision] insufficient methods present to judge SCOUT vs baselines.")
        return
    best_scout = max(scout, key=lambda m: vol[idx[m]])
    best_base = max(base, key=lambda m: vol[idx[m]])
    margin = vol[idx[best_scout]] - vol[idx[best_base]]
    nd_scout = any(pareto[m]["pareto_optimal"] for m in scout)
    print(f"\n[decision] best SCOUT={best_scout} (aggVol {vol[idx[best_scout]]:.3f}) vs "
          f"best baseline={best_base} (aggVol {vol[idx[best_base]]:.3f}); margin {margin:+.3f}; "
          f"SCOUT Pareto-optimal: {nd_scout}.")
    if nd_scout and margin >= 0.05:
        print("[decision] -> SCOUT shows a CLEAR aggregate advantage: VIABLE (keep).")
    elif nd_scout and margin >= 0.0:
        print("[decision] -> SCOUT competitive but margin small: MARGINAL (needs full baselines + "
              "more seeds, or proceed to JEDI-FL).")
    else:
        print("[decision] -> SCOUT does NOT clearly improve the aggregate: recommend SKIPPING SCOUT "
              "and implementing JEDI-FL.")
    print("NOTE: heuristic call on the current (partial) baseline set + small run; the final "
          "decision needs the full high-fidelity baselines and a longer multi-seed sweep.")


def _maybe_stats(logger, methods, per_method):
    """Run the Test-E statistical report (mean±std, CI, paired t / Wilcoxon, Friedman+Nemenyi)."""
    from scout_fl.analysis.stats import format_report, statistical_report
    n_seeds = min((len(per_method[m]) for m in methods), default=0)
    if n_seeds < 2:
        print("\n[stats] only one seed — skipping significance tests (run >=2 seeds, campaign target >=5).")
        return
    reference = "jedi" if "jedi" in methods else methods[0]
    report = statistical_report({m: per_method[m] for m in methods}, reference=reference)
    logger.save_json("stats_report.json", report)
    print("\n" + format_report(report))
    if n_seeds < 5:
        print(f"[stats] NOTE: {n_seeds} seeds — under-powered; the campaign target is >=5 seeds.")


def _maybe_plot_jedi(cfg, logger, all_rows):
    """Paper figures for JEDI-FL: the emergent learning->sensing schedule and the
    participation-deficit (fairness) self-correction, averaged over seeds per round."""
    if not cfg.get("logging", {}).get("save_plots", True):
        return
    jrows = [r for r in all_rows if r.get("method") == "jedi"]
    if not jrows:
        return
    rounds = sorted({r["round"] for r in jrows})

    def avg(key):
        return [float(np.mean([r[key] for r in jrows if r["round"] == t])) for t in rounds]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(rounds, avg("jedi_learn_frac"), color="#C44E52", marker="o", ms=3)
    ax1.set_xlabel("round"); ax1.set_ylabel("learning info fraction")
    ax1.set_ylim(0, 1); ax1.set_title("Emergent learning->sensing schedule (no scheduler)")
    ax2.plot(rounds, avg("jedi_deficit_mean"), label="mean deficit", color="#4C72B0")
    ax2.plot(rounds, avg("jedi_deficit_max"), label="max deficit", color="#55A868", ls="--")
    ax2.set_xlabel("round"); ax2.set_ylabel("participation deficit (virtual queue)")
    ax2.set_title("Fairness self-correction"); ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(logger.path("plots", "jedi_diagnostics.png"), dpi=150)
    plt.close(fig)


def _maybe_plot(cfg, logger, methods, agg, norm, nd):
    if not cfg.get("logging", {}).get("save_plots", True):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    accs = [agg[m]["acc"][0] for m in methods]
    lds = [agg[m]["logdet"][0] for m in methods]
    for i, m in enumerate(methods):
        ax1.scatter(accs[i], lds[i], s=90 if nd[i] else 40, marker="*" if nd[i] else "o")
        ax1.annotate(m, (accs[i], lds[i]), fontsize=7)
    ax1.set_xlabel("test accuracy"); ax1.set_ylabel("log-det coverage-diversity")
    ax1.set_title("accuracy vs sensing (★ = Pareto-optimal)")
    ax2.bar(methods, per_method_volume(norm), color="#4C72B0")
    ax2.set_ylabel("mean normalized score"); ax2.set_title("All-round score (higher better)")
    ax2.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(logger.path("plots", "bakeoff.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

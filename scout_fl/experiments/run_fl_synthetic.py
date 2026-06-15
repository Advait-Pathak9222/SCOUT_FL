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

import numpy as np
import torch
import yaml

from scout_fl.experiments.run_synthetic import build_scenario
from scout_fl.fl.aggregation import aggregate
from scout_fl.fl.client import local_train, probe_loss_and_embedding
from scout_fl.fl.datasets import build_client_datasets, load_dataset
from scout_fl.fl.models import build_model
from scout_fl.fl.partitioning import partition, partition_report
from scout_fl.fl.server import FLServer
from scout_fl.objectives.coverage_utility import CoverageMap, CoverageUtility
from scout_fl.objectives.fairness_utility import FairnessUtility
from scout_fl.objectives.learning_utility import LearningUtility
from scout_fl.objectives.sensing_utility import SensingUtility
from scout_fl.objectives.total_utility import TotalUtility
from scout_fl.selection.loss_based import LossSelector
from scout_fl.selection.random import RandomSelector
from scout_fl.selection.scout_greedy import ScoutGreedy
from scout_fl.selection.snr_based import SNRSelector
from scout_fl.sim.aircomp import aggregation_mse, min_gain_for_mse
from scout_fl.sim.channel import comm_channel_gains
from scout_fl.sim.energy_latency import round_energy_latency
from scout_fl.utils.config import load_config, to_plain
from scout_fl.utils.logging_utils import RunLogger
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


def run_one(method, cfg, scn, g, client_datasets, x_test, y_test,
            input_shape, num_classes, base_seed):
    """Run a full federated training for one selection ``method``; return rows + participation."""
    device = cfg.fl.device
    K, budget, rounds = scn.K, int(cfg.network.budget), int(cfg.fl.rounds)
    rng = np.random.default_rng(base_seed)
    torch.manual_seed(base_seed)

    server = FLServer(build_model(cfg.fl.model, input_shape, num_classes), device=device)
    cmap = CoverageMap(scn.R, rho=cfg.coverage.rho, innovation=cfg.coverage.innovation, u_init=1.0)
    fair = FairnessUtility(K)
    sensing = SensingUtility(scn.fim, scn.j0, scn.w)              # static across rounds
    full = list(range(K))

    aircomp_on = bool(cfg.aircomp.enabled)
    P, sigma2 = float(cfg.aircomp.power), float(cfg.aircomp.sigma2)
    mse_eps = cfg.constraints.mse_agg_max
    ota_on = bool(cfg.aircomp.get("ota_distortion", False))
    ota_scale = float(cfg.aircomp.get("ota_noise_scale", 1.0))
    obj = cfg.objectives
    participation = np.zeros(K)
    rows = []

    for t in range(rounds):
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

        # --- select ---
        tic = time.perf_counter()
        if method == "scout_greedy":
            feasible = None
            if aircomp_on and mse_eps is not None:
                g_min = min_gain_for_mse(mse_eps, budget, P, sigma2)
                feasible = lambda S, k: g[k] >= g_min          # AirComp-MSE gate
            res = ScoutGreedy().select(utility=total, num_clients=K, budget=budget, feasible=feasible)
        elif method == "random":
            res = RandomSelector().select(num_clients=K, budget=budget, rng=rng)
        elif method == "loss":
            res = LossSelector().select(scores=losses, budget=budget)
        elif method == "snr_only":
            res = SNRSelector().select(scores=scn.snr.sum(axis=1), budget=budget)
        else:
            raise ValueError(f"unknown selection method {method!r}")
        sel_time = time.perf_counter() - tic
        selected = res.selected
        participation[selected] += 1

        # --- local training of selected clients ---
        tic = time.perf_counter()
        updates, counts, train_losses = [], [], []
        for k in selected:
            server.set_global(g_flat)
            out = local_train(server.model, client_datasets[k], epochs=int(cfg.fl.local_epochs),
                              lr=float(cfg.fl.lr), batch_size=int(cfg.fl.batch_size),
                              optimizer=cfg.fl.optimizer, device=device)
            updates.append(out["update"]); counts.append(out["num_samples"]); train_losses.append(out["loss"])
        train_time = time.perf_counter() - tic

        # --- aggregate (FedAvg, optionally OTA-distorted) + apply ---
        tic = time.perf_counter()
        mse = aggregation_mse(g, selected, power=P, sigma2=sigma2) if aircomp_on else 0.0
        agg = aggregate(updates, counts, ota=ota_on, mse=mse, scale=ota_scale, rng=rng)
        server.apply_aggregated_update(g_flat, agg)
        agg_time = time.perf_counter() - tic

        # --- evaluate + ISAC metrics ---
        test_loss, test_acc = server.evaluate(x_test, y_test)
        el = round_energy_latency(selected, g, power=P, sigma2=sigma2,
                                  bandwidth=cfg.aircomp.bandwidth, model_bits=cfg.aircomp.model_bits,
                                  cpu_cycles=cfg.energy.cpu_cycles, cpu_freq=cfg.energy.cpu_freq,
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
            "agg_mse": round(float(mse), 6),
            "energy": round(float(el["energy"]), 4), "latency": round(float(el["latency"]), 4),
            "probe_time": round(probe_time, 4), "select_time": round(sel_time, 5),
            "train_time": round(train_time, 4), "agg_time": round(agg_time, 5),
            "round_time": round(probe_time + sel_time + train_time + agg_time, 4),
        }
        rows.append(row)

        # --- evolve ISAC state for next round ---
        cmap.update(selected, scn.C)
        fair.update(selected)

    return rows, participation


def main() -> None:
    parser = argparse.ArgumentParser(description="A1-Full SCOUT-FL federated run")
    parser.add_argument("--config", default="scout_fl/configs/fl_synthetic_small.yaml")
    parser.add_argument("--override", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true", help="tiny fast smoke run")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    if args.quick:
        _apply_quick(cfg)
    seed = int(cfg.get("seed", 0))
    rng = seed_everything(seed)
    logger = RunLogger(cfg.get("output_dir", "outputs"), "fl_synthetic", seed, to_plain(cfg))
    with logger.path("config_used.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(to_plain(cfg), fh, sort_keys=False)

    # ISAC scenario + comm channels
    scn = build_scenario(cfg, rng)
    g = comm_channel_gains(scn.clients, np.asarray(cfg.geometry.bs_position, dtype=float), rng,
                           snr_ref_db=cfg.channel.snr_ref_db, ref_distance=cfg.channel.reference_distance,
                           pathloss_exponent=cfg.channel.pathloss_exponent,
                           model=cfg.channel.model, rician_k_db=cfg.channel.rician_k_db)

    # dataset + partition
    ds = load_dataset(cfg.fl.dataset, root=cfg.fl.data_root, download=bool(cfg.fl.download))
    x_tr, y_tr = _subsample(ds.x_train, ds.y_train, cfg.fl.get("subsample_train"), rng)
    x_te, y_te = _subsample(ds.x_test, ds.y_test, cfg.fl.get("subsample_test"), rng)
    parts = partition(np.asarray(y_tr), scn.K, cfg.fl.non_iid, cfg.fl.get("dirichlet_alpha", 0.5),
                      np.random.default_rng(seed), min_size=1)
    client_datasets = build_client_datasets(x_tr, y_tr, parts)
    report = partition_report(np.asarray(y_tr), parts, ds.num_classes)
    logger.save_json("partition.json", report)

    methods = list(cfg.selection.get("methods", ["scout_greedy"]))
    all_rows, summary, runtimes = [], [], []
    for method in methods:
        rows, participation = run_one(method, cfg, scn, g, client_datasets, x_te, y_te,
                                      ds.input_shape, ds.num_classes, base_seed=seed)
        all_rows.extend(rows)
        accs = [r["test_acc"] for r in rows]
        jain = float(participation.sum() ** 2 / (scn.K * np.square(participation).sum() + 1e-12))
        summary.append({"method": method, "final_test_acc": accs[-1], "best_test_acc": max(accs),
                        "final_crb": rows[-1]["crb"], "avg_agg_mse": round(float(np.mean([r["agg_mse"] for r in rows])), 6),
                        "jain_fairness": round(jain, 4),
                        "avg_round_s": round(float(np.mean([r["round_time"] for r in rows])), 4)})
        runtimes.append({"method": method,
                         "avg_probe_s": round(float(np.mean([r["probe_time"] for r in rows])), 4),
                         "avg_select_s": round(float(np.mean([r["select_time"] for r in rows])), 5),
                         "avg_train_s": round(float(np.mean([r["train_time"] for r in rows])), 4),
                         "avg_agg_s": round(float(np.mean([r["agg_time"] for r in rows])), 5)})

    _save_logs(logger, all_rows, summary, runtimes)
    _print_summary(logger, cfg, ds, report, summary)
    _maybe_plot(cfg, logger, all_rows, summary)


def _subset(rows, cols):
    return [{c: (str(r[c]) if c == "selected" else r[c]) for c in cols} for r in rows]


def _save_logs(logger, all_rows, summary, runtimes):
    logger.save_csv("metrics.csv", _subset(all_rows, list(all_rows[0].keys())))
    logger.save_csv("fl_metrics.csv", _subset(all_rows, ["method", "round", "train_loss", "test_loss", "test_acc"]))
    logger.save_csv("sensing_metrics.csv", _subset(
        all_rows, ["method", "round", "sensing_logdet", "crb", "coverage_util", "fairness_util",
                   "agg_mse", "energy", "latency"]))
    logger.save_csv("selected_clients.csv", _subset(all_rows, ["method", "round", "selected"]))
    logger.save_json("runtime.json", runtimes)
    logger.save_json("summary.json", summary)


def _print_summary(logger, cfg, ds, report, summary):
    print("\n=== A1-Full SCOUT-FL federated run ===")
    print(f"run dir: {logger.dir}")
    print(f"dataset={cfg.fl.dataset} model={cfg.fl.model} non_iid={cfg.fl.non_iid}"
          f"(alpha={cfg.fl.get('dirichlet_alpha')}) | clients={cfg.network.num_clients} "
          f"budget={cfg.network.budget} rounds={cfg.fl.rounds}")
    print(f"partition top-class-fraction={report['mean_top_class_fraction']:.3f} "
          f"(0.1=IID), shard sizes {report['min_size']}-{report['max_size']}\n")
    print(f"  {'method':>13} | {'final_acc':>9} | {'best_acc':>8} | {'final_CRB':>9} | "
          f"{'avg_MSE':>8} | {'Jain':>6} | {'round_s':>7}")
    for s in summary:
        print(f"  {s['method']:>13} | {s['final_test_acc']:>9} | {s['best_test_acc']:>8} | "
              f"{s['final_crb']:>9} | {s['avg_agg_mse']:>8} | {s['jain_fairness']:>6} | {s['avg_round_s']:>7}")
    print("\nA1-Full story: SCOUT-FL should reach competitive accuracy while keeping lower CRB, "
          "better coverage, and higher Jain fairness than the geometry/coverage-blind baselines.")


def _maybe_plot(cfg, logger, all_rows, summary):
    if not cfg.get("logging", {}).get("save_plots", True):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    methods = [s["method"] for s in summary]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for m in methods:
        r = [row for row in all_rows if row["method"] == m]
        ax1.plot([x["round"] for x in r], [x["test_acc"] for x in r], marker="o", ms=3, label=m)
    ax1.set_xlabel("round"); ax1.set_ylabel("test accuracy")
    ax1.set_title("Federated convergence"); ax1.legend(fontsize=8)
    ax2.bar(methods, [s["final_crb"] for s in summary], color="#8172B3")
    ax2.set_ylabel("final aggregate CRB"); ax2.set_title("Sensing (lower better)")
    ax2.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(logger.path("plots", "fl_curves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""OFAT campaign orchestrator — the full IEEE-style evaluation matrix (Tests A-E).

One-factor-at-a-time (OFAT): hold the nominal operating point (campaign_main.yaml:
N=100, K/N=10%, 3 seeds, Dirichlet=0.3, SNR=20 dB, 3 targets, 150 rounds) and vary
one axis per sweep. Every sweep point runs the full multi-seed bake-off over all
methods and records:

  Test A (Learning)        -> non-IID severity (Dirichlet), IID vs spatial, dataset track
  Test B (Wireless/AirComp)-> uplink SNR, channel model (Rayleigh/Rician)
  Test C (Sensing/ISAC)    -> number of targets, angular resolution (k_angle)
  Test D (Trade-off)       -> Pareto volume / non-dominated mask (computed at every point)
  Test E (Statistical)     -> mean+/-std, 95% CI, paired t / Wilcoxon, Friedman+Nemenyi (per point)

Usage:
  python -m scout_fl.experiments.run_campaign --dry-run               # print the matrix, no compute
  python -m scout_fl.experiments.run_campaign --quick                 # tiny end-to-end smoke
  python -m scout_fl.experiments.run_campaign --sweeps B_wireless_snr C_sensing_targets
  python -m scout_fl.experiments.run_campaign --override fl.device=cuda   # full run on GPU

Results land in outputs/campaign/<timestamp>/: matrix.csv (one row per point x method)
and campaign_summary.json (per-point Pareto winners, hypervolume, and stats reports).
"""
from __future__ import annotations

import argparse
import json

from scout_fl.analysis.pareto import hypervolume
from scout_fl.analysis.stats import statistical_report
from scout_fl.experiments.run_fl_synthetic import (_apply_quick, aggregate_results,
                                                    run_bakeoff)
from scout_fl.fl.datasets import load_fl_dataset
from scout_fl.utils.config import load_config, to_plain
from scout_fl.utils.device import describe_device, resolve_device
from scout_fl.utils.logging_utils import RunLogger

# Each sweep: a Test label, the parameter that labels the x-axis, and the list of
# points (each point = a dict of coupled config overrides applied to the nominal).
SWEEPS = {
    # ---- Test A: Learning ------------------------------------------------
    "A_learning_noniid": {"test": "A", "param": "fl.dirichlet_alpha", "points": [
        {"fl.dirichlet_alpha": 0.1}, {"fl.dirichlet_alpha": 0.3}, {"fl.dirichlet_alpha": 0.5},
    ]},
    "A_learning_partition": {"test": "A", "param": "fl.non_iid", "points": [
        {"fl.non_iid": "iid"}, {"fl.non_iid": "spatial"},
    ]},
    "A_datasets": {"test": "A", "param": "fl.dataset", "points": [
        {"fl.dataset": "fashion_mnist", "fl.model": "small_cnn"},     # debug layer
        {"fl.dataset": "cifar10", "fl.model": "small_cnn"},           # main track
        {"fl.dataset": "cifar100", "fl.model": "small_cnn"},          # harder
        {"fl.dataset": "emnist", "fl.model": "small_cnn"},            # FEMNIST-class (real handwritten)
        {"fl.dataset": "uci_har", "fl.model": "mlp"},                 # REAL sensor HAR (wireless-sensing task)
    ]},
    # ---- Test B: Wireless / AirComp -------------------------------------
    # Physical mode: sweep the real uplink Tx power (dBm) -> spans cell-edge (low SNR, AirComp
    # distortion dominant) to cell-center (high SNR). (Vary channel.snr_ref_db instead if
    # physical.enabled=false.)
    "B_wireless_snr": {"test": "B", "param": "physical.tx_power_dbm", "points": [
        {"physical.tx_power_dbm": p} for p in (-35, -30, -25, -20, -15, -10, 0)
    ]},
    "B_wireless_channel": {"test": "B", "param": "channel.model", "points": [
        {"channel.model": "rayleigh"}, {"channel.model": "rician"},
    ]},
    # ---- Test C: Sensing / ISAC -----------------------------------------
    "C_sensing_targets": {"test": "C", "param": "network.num_targets", "points": [
        {"network.num_targets": 2, "sensing.target_weights": [1.0, 1.0]},
        {"network.num_targets": 3, "sensing.target_weights": [1.0, 1.0, 1.0]},
        {"network.num_targets": 5, "sensing.target_weights": [1.0, 1.0, 1.0, 1.0, 1.0]},
    ]},
    "C_sensing_kangle": {"test": "C", "param": "sensing.k_angle", "points": [
        {"sensing.k_angle": 0.02}, {"sensing.k_angle": 0.05}, {"sensing.k_angle": 0.1},
    ]},
}
_METRICS = ["acc", "best_acc", "logdet", "crb", "agg_mse", "jain", "energy"]


def _fmt(v):
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(str(x) for x in v) + "]"
    return str(v)


def _point_overrides(point):
    return [f"{k}={_fmt(v)}" for k, v in point.items()]


def _selected_sweeps(names):
    if not names:
        return SWEEPS
    missing = [n for n in names if n not in SWEEPS]
    if missing:
        raise SystemExit(f"unknown sweep(s) {missing}; available: {list(SWEEPS)}")
    return {n: SWEEPS[n] for n in names}


def run_campaign(base_config, sweeps, base_overrides, quick, out_root):
    logger = RunLogger(out_root, "campaign", 0, {})
    matrix, summary = [], {}
    ds_cache = {}

    for sweep_name, spec in sweeps.items():
        for point in spec["points"]:
            overrides = list(base_overrides or []) + _point_overrides(point)
            cfg = load_config(base_config, overrides)
            if quick:
                _apply_quick(cfg)
            seeds = [int(s) for s in (cfg.get("seeds") or [int(cfg.get("seed", 0))])]
            if quick:
                seeds = seeds[:2]

            key = (cfg.fl.dataset, bool(cfg.fl.download))
            if key not in ds_cache:
                ds_cache[key] = load_fl_dataset(cfg.fl.dataset, root=cfg.fl.data_root,
                                                download=bool(cfg.fl.download))
            ds = ds_cache[key]

            label = point.get(spec["param"])
            print(f"\n=== [{spec['test']}] {sweep_name}: {spec['param']}={label} "
                  f"(dataset={cfg.fl.dataset}, seeds={seeds}, rounds={cfg.fl.rounds}) ===")
            point_tag = f"{sweep_name}={label}"                # resumable per-round store key
            per_method, _, _ = run_bakeoff(cfg, ds, seeds, runs_root="runs",
                                           tag="campaign", point=point_tag)
            methods, agg, norm, vol, nd, pareto = aggregate_results(per_method)
            hv = round(float(hypervolume(norm)), 4)
            winners = [m for m in methods if pareto[m]["pareto_optimal"]]
            best_vol = max(methods, key=lambda m: pareto[m]["aggregate_volume"])

            for m in methods:
                row = {"sweep": sweep_name, "test": spec["test"], "param": spec["param"],
                       "value": label, "dataset": cfg.fl.dataset, "method": m,
                       "aggVol": pareto[m]["aggregate_volume"],
                       "pareto_optimal": pareto[m]["pareto_optimal"]}
                row.update({k: round(agg[m][k][0], 5) for k in _METRICS})
                row.update({f"{k}_std": round(agg[m][k][1], 5) for k in _METRICS})
                matrix.append(row)

            summary[f"{sweep_name}::{label}"] = {
                "test": spec["test"], "param": spec["param"], "value": label,
                "dataset": cfg.fl.dataset, "set_hypervolume": hv,
                "pareto_optimal": winners, "best_aggregate_volume": best_vol,
                "stats": statistical_report(
                    {m: per_method[m] for m in methods},
                    reference="jedi" if "jedi" in methods else methods[0]),
            }
            print(f"    -> Pareto-optimal: {winners} | best aggVol: {best_vol} | hypervolume: {hv}")

    logger.save_csv("matrix.csv", matrix)
    logger.save_json("campaign_summary.json", summary)
    logger.save_json("sweeps_run.json", {n: SWEEPS[n] for n in sweeps})
    print(f"\n[campaign] {len(matrix)} rows over {sum(len(s['points']) for s in sweeps.values())} "
          f"points -> {logger.dir}")
    return logger.dir


def _dry_run(sweeps, base_config, base_overrides, quick):
    cfg = load_config(base_config, base_overrides)
    if quick:
        _apply_quick(cfg)
    n_methods = len(cfg.selection.get("methods", []))
    n_seeds = len(cfg.get("seeds") or [0])
    if quick:
        n_seeds = min(n_seeds, 2)
    print(f"Campaign matrix (nominal: N={cfg.network.num_clients}, budget={cfg.network.budget}, "
          f"rounds={cfg.fl.rounds}, seeds={n_seeds}, methods={n_methods})")
    total_pts = total_train = 0
    for name, spec in sweeps.items():
        pts = len(spec["points"])
        total_pts += pts
        total_train += pts * n_methods * n_seeds
        vals = [p.get(spec["param"]) for p in spec["points"]]
        print(f"  [{spec['test']}] {name:<22} {spec['param']:<22} {vals}")
    print(f"\nTotal: {total_pts} sweep points, "
          f"{total_train} full {cfg.fl.rounds}-round trainings ({n_methods} methods x {n_seeds} seeds each).")


def main():
    p = argparse.ArgumentParser(description="SCOUT-FL / JEDI-FL OFAT campaign (Tests A-E)")
    p.add_argument("--config", default="scout_fl/configs/campaign_main.yaml")
    p.add_argument("--sweeps", nargs="*", default=None, help=f"subset of {list(SWEEPS)}")
    p.add_argument("--override", nargs="*", default=None)
    p.add_argument("--out", default="outputs/campaign")
    p.add_argument("--quick", action="store_true", help="tiny fast smoke run")
    p.add_argument("--dry-run", action="store_true", help="print the matrix and exit")
    args = p.parse_args()

    sweeps = _selected_sweeps(args.sweeps)
    if args.dry_run:
        _dry_run(sweeps, args.config, args.override, args.quick)
        return
    device = resolve_device(load_config(args.config, args.override).fl.get("device", "auto"))
    print(f"[device] using {device} ({describe_device(device)})")
    run_campaign(args.config, sweeps, args.override, args.quick, args.out)


if __name__ == "__main__":
    main()

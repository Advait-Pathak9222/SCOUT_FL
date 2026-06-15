"""Mandatory microbenchmark: prove that high sensing-SNR != high sensing value.

    python -m scout_fl.experiments.run_microbenchmark \
        --config scout_fl/configs/microbenchmark.yaml

It (1) enumerates every client subset and tabulates log-det FIM gain, CRB, and
the RMSE proxy; (2) runs the SCOUT greedy selector, the naive SNR-only selector,
and the CRB greedy selector at the budget; (3) numerically verifies that
``f_sense`` is monotone submodular; and (4) checks the GATE: the SCOUT/CRB
geometry-aware selection must achieve a strictly lower CRB than the SNR-only
selection. If the gate fails, the rest of the pipeline should not be trusted.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from scout_fl.analysis.verify_submodularity import verify_submodular
from scout_fl.objectives.sensing_utility import SensingUtility
from scout_fl.selection.crb_based import CRBSelector
from scout_fl.selection.scout_greedy import ScoutGreedy
from scout_fl.selection.snr_based import SNRSelector
from scout_fl.sim.fim import db_to_linear, per_client_target_fim, prior_fim
from scout_fl.sim.geometry import pairwise_geometry
from scout_fl.utils.config import load_config, to_plain
from scout_fl.utils.logging_utils import RunLogger
from scout_fl.utils.seed import seed_everything


def _fmt_subset(subset) -> str:
    return "{" + ",".join(str(i + 1) for i in sorted(subset)) + "}"


def build_utility(cfg):
    clients = np.asarray(cfg.geometry.clients, dtype=float)
    targets = np.asarray(cfg.geometry.targets, dtype=float)
    geom = pairwise_geometry(clients, targets)
    snr_linear = db_to_linear(cfg.sensing.snr_db)              # (K,)
    fim = per_client_target_fim(geom, snr_linear,
                                cfg.sensing.k_range, cfg.sensing.k_angle)
    j0 = prior_fim(len(targets), cfg.sensing.prior_fim)
    util = SensingUtility(fim, j0, cfg.sensing.get("target_weights"))
    return geom, snr_linear, util


def main() -> None:
    parser = argparse.ArgumentParser(description="SCOUT-FL microbenchmark")
    parser.add_argument("--config", default="scout_fl/configs/microbenchmark.yaml")
    parser.add_argument("--override", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    seed = int(cfg.get("seed", 0))
    rng = seed_everything(seed)
    logger = RunLogger(cfg.get("output_dir", "outputs"), "microbenchmark", seed, to_plain(cfg))

    geom, snr_linear, util = build_utility(cfg)
    K = util.K
    budget = int(cfg.selection.budget)
    ids = list(range(K))

    # (1) enumerate every subset -> sensing metrics table -------------------
    subset_rows = []
    for size in range(1, K + 1):
        for subset in itertools.combinations(ids, size):
            subset_rows.append({
                "subset": _fmt_subset(subset),
                "size": size,
                "logdet_gain": round(util.value(subset), 4),
                "crb_sum": round(float(util.crb(subset).sum()), 5),
                "rmse_proxy": round(float(util.rmse(subset).sum()), 5),
            })
    logger.save_csv("all_subsets.csv", subset_rows)

    # (2) selectors at the budget -------------------------------------------
    selectors = {
        "scout_greedy": ScoutGreedy(use_lazy=True).select(
            utility=util, num_clients=K, budget=budget),
        "snr_only": SNRSelector().select(scores=snr_linear, budget=budget),
        "crb_only": CRBSelector().select(utility=util, num_clients=K, budget=budget),
    }
    selector_rows = []
    for name, res in selectors.items():
        S = res.selected
        selector_rows.append({
            "selector": name,
            "selected": _fmt_subset(S),
            "logdet": round(util.value(S), 4),
            "crb_sum": round(float(util.crb(S).sum()), 5),
            "rmse": round(float(util.rmse(S).sum()), 5),
            "select_time_ms": round(res.select_time * 1e3, 4),
        })
    logger.save_csv("selectors.csv", selector_rows)

    # (3) submodularity / monotonicity verification -------------------------
    submod = verify_submodular(util.value, ids,
                               cfg.get("verify", {}).get("submodularity_samples", 500), rng)
    logger.save_json("submodularity.json", submod)

    # (4) the gate: geometry-aware selection beats SNR-only on CRB ----------
    scout_crb = float(util.crb(selectors["scout_greedy"].selected).sum())
    snr_crb = float(util.crb(selectors["snr_only"].selected).sum())
    gate_pass = bool(scout_crb < snr_crb - 1e-9) and submod["is_submodular"]

    # ----------------------------------------------------------- report -----
    print("\n=== SCOUT-FL microbenchmark ===")
    print(f"run dir: {logger.dir}")
    print(f"clients K={K}, budget S_max={budget}, targets M={util.M}\n")
    print("Per-pair sensing value (size-2 subsets):")
    print(f"  {'subset':>8} | {'logdet_gain':>11} | {'crb_sum':>10} | {'rmse_proxy':>10}")
    for row in subset_rows:
        if row["size"] == 2:
            print(f"  {row['subset']:>8} | {row['logdet_gain']:>11} | "
                  f"{row['crb_sum']:>10} | {row['rmse_proxy']:>10}")
    print("\nSelectors @ budget:")
    for row in selector_rows:
        print(f"  {row['selector']:>13}: {row['selected']:>8}  "
              f"logdet={row['logdet']:>8}  CRB={row['crb_sum']:>10}  "
              f"RMSE={row['rmse']:>8}  ({row['select_time_ms']} ms)")
    print(f"\nSubmodularity check: submodular={submod['is_submodular']} "
          f"monotone={submod['is_monotone']} "
          f"(violations={submod['submodularity_violations']}/{submod['samples']})")
    print(f"\nGATE  scout CRB={scout_crb:.5f}  vs  snr-only CRB={snr_crb:.5f}")
    print("GATE RESULT:", "PASS  (complementary-angle selection beats high-SNR)"
          if gate_pass else "FAIL  (investigate before proceeding)")

    logger.save_json("gate.json", {
        "scout_crb": scout_crb, "snr_only_crb": snr_crb,
        "gate_pass": gate_pass,
        "scout_selected": selectors["scout_greedy"].selected,
        "snr_selected": selectors["snr_only"].selected,
        "crb_selected": selectors["crb_only"].selected,
    })
    _maybe_plot(cfg, logger, subset_rows, selector_rows)


def _maybe_plot(cfg, logger, subset_rows, selector_rows) -> None:
    if not cfg.get("logging", {}).get("save_plots", True):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    pairs = [r for r in subset_rows if r["size"] == 2]
    labels = [r["subset"] for r in pairs]
    crbs = [r["crb_sum"] for r in pairs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, crbs, color="#4C72B0")
    ax.set_yscale("log")
    ax.set_ylabel("aggregate CRB (log scale)")
    ax.set_xlabel("selected client pair")
    ax.set_title("Localization CRB per client pair\n(lower is better)")
    fig.tight_layout()
    fig.savefig(logger.path("plots", "crb_per_pair.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

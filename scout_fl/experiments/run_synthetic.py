"""Multi-round synthetic SCOUT-FL selection loop (composite utility, no FL yet).

Each round builds the sensing / learning / coverage / fairness utilities, selects
S_max clients via the composite SCOUT-FL objective, then evolves the coverage
(freshness) map and the fairness ages. It compares SCOUT-FL against
sensing-only / SNR-only / random selectors on localization CRB, region
uncertainty, and participation fairness over time.

This is the round-loop scaffold the FL pipeline (Step 7) plugs into: replace the
placeholder gradient embeddings with real per-client gradients and add
local training + OTA aggregation + accuracy logging.

Run: python -m scout_fl.experiments.run_synthetic --config scout_fl/configs/synthetic_small.yaml
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from scout_fl.objectives.coverage_utility import (CoverageMap, CoverageUtility,
                                                  contribution_matrix, region_centers)
from scout_fl.objectives.fairness_utility import FairnessUtility
from scout_fl.objectives.learning_utility import LearningUtility
from scout_fl.objectives.sensing_utility import SensingUtility
from scout_fl.objectives.total_utility import TotalUtility
from scout_fl.selection.random import RandomSelector
from scout_fl.selection.scout_greedy import ScoutGreedy
from scout_fl.selection.snr_based import SNRSelector
from scout_fl.sim.fim import per_client_target_fim, prior_fim
from scout_fl.sim.geometry import pairwise_geometry, sample_positions
from scout_fl.sim.sensing import sensing_snr
from scout_fl.utils.config import load_config, to_plain
from scout_fl.utils.logging_utils import RunLogger
from scout_fl.utils.seed import seed_everything

SELECTORS = ["scout", "sensing_only", "snr_only", "random"]


@dataclass
class Scenario:
    snr: np.ndarray          # (K, M) linear sensing SNR
    fim: np.ndarray          # (K, M, 2, 2)
    j0: np.ndarray           # (M, 2, 2)
    w: np.ndarray            # (M,) target weights
    C: np.ndarray            # (K, R) client->region contribution
    sim: np.ndarray          # (K, K) learning similarity
    K: int
    M: int
    R: int
    clients: np.ndarray
    cluster_assignment: np.ndarray = None
    compute_het: np.ndarray = None       # per-client compute-speed multiplier (straggler heterogeneity)


def _clustered_layout(cfg, rng, K, M, area):
    """Clustered client viewpoints (the fair-testbed fix).

    Clients in a cluster sit close together, so they observe the central targets
    from nearly the SAME bearing (angular redundancy) — two high-SNR clients in
    one cluster carry near-duplicate sensing information. Near clusters are closer
    to the targets (higher sensing SNR), baiting SNR-only into redundant picks,
    while a diversity / joint-information selector spreads across clusters and
    wins on log-det coverage-diversity (and CRB). Returns (clients, targets,
    cluster_assignment).
    """
    bs = np.asarray(cfg.geometry.bs_position, dtype=float)
    n_clusters = int(cfg.geometry.get("num_clusters", 5))
    spread = float(cfg.geometry.get("cluster_spread", 3.0))
    span = float(min(area))
    angles = 2.0 * np.pi * np.arange(n_clusters) / n_clusters
    radii = np.linspace(0.12, 0.45, n_clusters) * span   # monotonic: cluster 0 nearest (highest SNR) -> SNR-only piles in
    centers = bs + np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    cluster_assignment = np.arange(K) % n_clusters
    clients = np.clip(centers[cluster_assignment] + spread * rng.standard_normal((K, 2)), 0.0, area)
    targets = np.clip(bs + 0.08 * span * rng.standard_normal((M, 2)), 0.0, area)
    return clients, targets, cluster_assignment


def build_scenario(cfg, rng) -> Scenario:
    net = cfg.network
    area = np.asarray(net.area_size, dtype=float)
    K, M, R = int(net.num_clients), int(net.num_targets), int(net.num_regions)

    geom_source = cfg.geometry.get("source", "synthetic")
    layout = cfg.geometry.get("layout", "random")
    if geom_source != "synthetic":
        # real client/target positions from an external sensing dataset (synthetic fallback)
        from scout_fl.fl.datasets_external import load_sensing_geometry
        root = cfg.fl.get("data_root", "data") if cfg.get("fl") else "data"
        clients, targets = load_sensing_geometry(geom_source, K, M, rng,
                                                 area=float(min(area)), root=root)
        cluster_assignment = np.arange(K) % int(cfg.geometry.get("num_clusters", 5))
    elif layout == "clustered":
        clients, targets, cluster_assignment = _clustered_layout(cfg, rng, K, M, area)
    else:
        clients = (sample_positions(rng, K, area) if cfg.geometry.random_clients
                   else np.asarray(cfg.geometry.clients, dtype=float))
        targets = (sample_positions(rng, M, area) if cfg.geometry.random_targets
                   else np.asarray(cfg.geometry.targets, dtype=float))
        cluster_assignment = np.arange(K)            # each client its own "cluster"
    centers = region_centers(area, R)
    geom = pairwise_geometry(clients, targets)

    rcs = np.clip(rng.normal(cfg.sensing.rcs_mean, cfg.sensing.rcs_std, size=M), 1e-3, None)
    snr = sensing_snr(geom, cfg.sensing.ref_snr_db, cfg.sensing.pathloss_exponent,
                      rcs=rcs, ref_distance=cfg.sensing.ref_distance)
    fim = per_client_target_fim(geom, snr, cfg.sensing.k_range, cfg.sensing.k_angle)
    j0 = prior_fim(M, cfg.sensing.prior_fim)
    weights = (np.asarray(cfg.sensing.target_weights, dtype=float)
               if cfg.sensing.get("target_weights") else np.ones(M))

    sensing_range = float(cfg.coverage.get("sensing_range", 0.2 * float(min(area))))
    C = contribution_matrix(clients, centers, sensing_range)
    # Placeholder gradient embeddings (FL step replaces with real per-client gradients).
    embeddings = rng.normal(size=(K, 8))
    sim = LearningUtility(embeddings=embeddings).S
    return Scenario(snr, fim, j0, weights, C, sim, K, M, R, clients, cluster_assignment)


def run_loop(cfg, scn: Scenario, kind: str, seed: int):
    rng = np.random.default_rng(seed)
    K, R = scn.K, scn.R
    budget = int(cfg.network.budget)
    rounds = int(cfg.rounds)
    obj = cfg.objectives

    cmap = CoverageMap(R, rho=cfg.coverage.rho, innovation=cfg.coverage.innovation, u_init=1.0)
    fair = FairnessUtility(K)
    sensing = SensingUtility(scn.fim, scn.j0, scn.w)        # static across rounds
    learning = LearningUtility(similarity=scn.sim)          # static across rounds
    full = list(range(K))
    participation = np.zeros(K)
    rows = []

    for t in range(rounds):
        coverage = CoverageUtility(cmap.U, scn.C, g=cfg.coverage.saturating)
        weights = {"learning": obj.alpha_learning, "sensing": obj.lambda_sense,
                   "coverage": obj.lambda_coverage, "fairness": obj.lambda_fairness}
        norms = {"learning": learning.value(full), "sensing": sensing.value(full),
                 "coverage": coverage.value(full), "fairness": max(fair.value(full), 1e-9)}
        total = TotalUtility(
            {"learning": learning, "sensing": sensing, "coverage": coverage, "fairness": fair},
            weights=weights, normalizers=norms)

        if kind == "scout":
            res = ScoutGreedy().select(utility=total, num_clients=K, budget=budget)
        elif kind == "sensing_only":
            res = ScoutGreedy().select(utility=sensing, num_clients=K, budget=budget)
        elif kind == "snr_only":
            res = SNRSelector().select(scores=scn.snr.sum(axis=1), budget=budget)
        elif kind == "random":
            res = RandomSelector().select(num_clients=K, budget=budget, rng=rng)
        else:
            raise ValueError(f"unknown selector {kind!r}")

        S = res.selected
        participation[S] += 1
        crb = float((scn.w * sensing.crb(S)).sum())
        cov_val = coverage.value(S)
        cmap.update(S, scn.C)
        fair.update(S)
        rows.append({
            "round": t, "selector": kind,
            "logdet": round(sensing.value(S), 4),
            "crb_sum": round(crb, 5),
            "coverage_val": round(cov_val, 4),
            "mean_uncertainty": round(float(cmap.U.mean()), 4),
            "max_uncertainty": round(float(cmap.U.max()), 4),
            "select_ms": round(res.select_time * 1e3, 4),
        })

    jain = float(participation.sum() ** 2 / (K * np.square(participation).sum() + 1e-12))
    return rows, jain, participation


def main() -> None:
    parser = argparse.ArgumentParser(description="SCOUT-FL synthetic selection loop")
    parser.add_argument("--config", default="scout_fl/configs/synthetic_small.yaml")
    parser.add_argument("--override", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    seed = int(cfg.get("seed", 0))
    rng = seed_everything(seed)
    logger = RunLogger(cfg.get("output_dir", "outputs"), "synthetic", seed, to_plain(cfg))
    scn = build_scenario(cfg, rng)

    all_rows, summary = [], []
    for i, kind in enumerate(SELECTORS):
        rows, jain, _ = run_loop(cfg, scn, kind, seed + i)
        all_rows.extend(rows)
        crb = np.mean([r["crb_sum"] for r in rows])
        unc = [r["mean_uncertainty"] for r in rows]
        ms = np.mean([r["select_ms"] for r in rows])
        summary.append({
            "selector": kind,
            "avg_crb": round(float(crb), 5),
            "avg_mean_uncertainty": round(float(np.mean(unc)), 4),
            "final_mean_uncertainty": round(float(unc[-1]), 4),
            "jain_fairness": round(jain, 4),
            "avg_select_ms": round(float(ms), 4),
        })
    logger.save_csv("rounds.csv", all_rows)
    logger.save_csv("summary.csv", summary)

    print("\n=== SCOUT-FL synthetic selection loop ===")
    print(f"run dir: {logger.dir}")
    print(f"K={scn.K} clients, budget={cfg.network.budget}, M={scn.M} targets, "
          f"R={scn.R} regions, T={cfg.rounds} rounds\n")
    print(f"  {'selector':>13} | {'avg_CRB':>9} | {'avg_uncert':>10} | "
          f"{'final_uncert':>12} | {'Jain':>6} | {'sel_ms':>7}")
    for r in summary:
        print(f"  {r['selector']:>13} | {r['avg_crb']:>9} | {r['avg_mean_uncertainty']:>10} | "
              f"{r['final_mean_uncertainty']:>12} | {r['jain_fairness']:>6} | {r['avg_select_ms']:>7}")
    print("\nExpected: SCOUT (composite) keeps low CRB AND low region uncertainty AND high "
          "Jain fairness; sensing_only ignores coverage/fairness; snr_only ignores geometry; "
          "random is worst overall.")
    _maybe_plot(cfg, logger, all_rows, summary)


def _maybe_plot(cfg, logger, all_rows, summary) -> None:
    if not cfg.get("logging", {}).get("save_plots", True):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for kind in SELECTORS:
        rows = [r for r in all_rows if r["selector"] == kind]
        ax1.plot([r["round"] for r in rows], [r["mean_uncertainty"] for r in rows], label=kind)
    ax1.set_xlabel("round"); ax1.set_ylabel("mean region uncertainty")
    ax1.set_title("Coverage/freshness over time (lower better)"); ax1.legend(fontsize=8)
    labels = [s["selector"] for s in summary]
    ax2.bar(labels, [s["avg_crb"] for s in summary], color="#55A868")
    ax2.set_ylabel("avg aggregate CRB"); ax2.set_title("Sensing accuracy (lower better)")
    ax2.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(logger.path("plots", "synthetic_summary.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

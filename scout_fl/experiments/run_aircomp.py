"""A2 resource-allocation add-on demo: AirComp aggregation MSE + constraint-aware
selection.

Three policies (matching the A2 plan):
  1. no_resource_alloc   comm-blind SCOUT selection + un-optimized transmit power
  2. simple_power_norm    comm-blind SCOUT selection + full-power channel inversion
  3. aircomp_aware        full power + MSE-constrained selection (gate weak channels)

Guaranteed ordering: MSE(no_RA) >= MSE(simple); 'aware' meets the MSE target by
construction (gating out weak links), trading a little sensing utility for it.

Run: python -m scout_fl.experiments.run_aircomp --config scout_fl/configs/synthetic_small.yaml
"""
from __future__ import annotations

import argparse

import numpy as np

from scout_fl.experiments.run_synthetic import build_scenario
from scout_fl.objectives.constraints import Constraints
from scout_fl.objectives.coverage_utility import CoverageUtility
from scout_fl.objectives.fairness_utility import FairnessUtility
from scout_fl.objectives.learning_utility import LearningUtility
from scout_fl.objectives.sensing_utility import SensingUtility
from scout_fl.objectives.total_utility import TotalUtility
from scout_fl.selection.scout_greedy import ScoutGreedy
from scout_fl.sim.aircomp import aggregation_mse, min_gain_for_mse
from scout_fl.sim.channel import comm_channel_gains
from scout_fl.sim.energy_latency import round_energy_latency
from scout_fl.utils.config import load_config, to_plain
from scout_fl.utils.logging_utils import RunLogger
from scout_fl.utils.seed import seed_everything


def build_round0_total(cfg, scn):
    """Composite utility at the initial round (uniform coverage map, zero ages)."""
    K, R = scn.K, scn.R
    sensing = SensingUtility(scn.fim, scn.j0, scn.w)
    learning = LearningUtility(similarity=scn.sim)
    coverage = CoverageUtility(np.ones(R), scn.C, g=cfg.coverage.saturating)
    fair = FairnessUtility(K)
    full = list(range(K))
    weights = {"learning": cfg.objectives.alpha_learning, "sensing": cfg.objectives.lambda_sense,
               "coverage": cfg.objectives.lambda_coverage, "fairness": cfg.objectives.lambda_fairness}
    norms = {"learning": learning.value(full), "sensing": sensing.value(full),
             "coverage": coverage.value(full), "fairness": max(fair.value(full), 1e-9)}
    total = TotalUtility(
        {"learning": learning, "sensing": sensing, "coverage": coverage, "fairness": fair},
        weights=weights, normalizers=norms)
    return total, sensing


def main() -> None:
    parser = argparse.ArgumentParser(description="SCOUT-FL A2 AirComp demo")
    parser.add_argument("--config", default="scout_fl/configs/synthetic_small.yaml")
    parser.add_argument("--override", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    seed = int(cfg.get("seed", 0))
    rng = seed_everything(seed)
    logger = RunLogger(cfg.get("output_dir", "outputs"), "aircomp", seed, to_plain(cfg))

    scn = build_scenario(cfg, rng)
    bs = np.asarray(cfg.geometry.bs_position, dtype=float)
    g = comm_channel_gains(scn.clients, bs, rng,
                           snr_ref_db=cfg.channel.snr_ref_db,
                           ref_distance=cfg.channel.reference_distance,
                           pathloss_exponent=cfg.channel.pathloss_exponent,
                           model=cfg.channel.model, rician_k_db=cfg.channel.rician_k_db)
    total, sensing = build_round0_total(cfg, scn)

    K = scn.K
    budget = int(cfg.network.budget)
    P = float(cfg.aircomp.power)
    P0 = float(cfg.aircomp.power_unoptimized)
    sigma2 = float(cfg.aircomp.sigma2)
    mse_eps = cfg.constraints.mse_agg_max
    cons = Constraints(mse_agg_max=mse_eps, infeasible_policy=cfg.constraints.infeasible_policy)
    en = dict(sigma2=sigma2, bandwidth=cfg.aircomp.bandwidth, model_bits=cfg.aircomp.model_bits,
              cpu_cycles=cfg.energy.cpu_cycles, cpu_freq=cfg.energy.cpu_freq,
              kappa=cfg.energy.kappa, e_sense=cfg.energy.e_sense, t_sense=cfg.energy.t_sense)

    sel_blind = ScoutGreedy().select(utility=total, num_clients=K, budget=budget).selected
    g_min = min_gain_for_mse(mse_eps, budget, P, sigma2)
    aware = ScoutGreedy().select(utility=total, num_clients=K, budget=budget,
                                 feasible=lambda S, k: g[k] >= g_min)
    sel_aware = aware.selected

    rows = []

    def record(name, sel, power):
        mse = aggregation_mse(g, sel, power=power, sigma2=sigma2)
        el = round_energy_latency(sel, g, power=power, **en)
        rows.append({
            "policy": name,
            "selected": str([s + 1 for s in sel]),
            "agg_mse": round(mse, 5),
            "meets_mse_eps": cons.evaluate(mse=mse)["feasible"],
            "sensing_logdet": round(sensing.value(sel), 4),
            "min_channel_g": round(float(min(g[sel])), 4),
            "latency_s": round(el["latency"], 4),
            "energy_J": round(el["energy"], 4),
        })

    record("no_resource_alloc", sel_blind, P0)
    record("simple_power_norm", sel_blind, P)
    record("aircomp_aware", sel_aware, P)
    logger.save_csv("aircomp_policies.csv", rows)
    logger.save_json("info.json", {
        "mse_eps": mse_eps, "g_min_required": g_min,
        "aware_relaxed_steps": aware.info["relaxed_steps"],
        "n_feasible_clients": int(np.sum(g >= g_min)),
    })

    print("\n=== SCOUT-FL A2 / AirComp demo ===")
    print(f"run dir: {logger.dir}")
    print(f"K={K}, budget={budget}, MSE target eps={mse_eps}, "
          f"g_min_required={g_min:.4f}, feasible clients={int(np.sum(g >= g_min))}/{K}\n")
    print(f"  {'policy':>18} | {'agg_MSE':>9} | {'<=eps':>6} | {'logdet':>8} | "
          f"{'min_g':>7} | {'lat_s':>7} | {'E_J':>7}")
    for r in rows:
        print(f"  {r['policy']:>18} | {r['agg_mse']:>9} | {str(r['meets_mse_eps']):>6} | "
              f"{r['sensing_logdet']:>8} | {r['min_channel_g']:>7} | "
              f"{r['latency_s']:>7} | {r['energy_J']:>7}")
    if aware.info["relaxed_steps"]:
        print(f"\n[relax_and_log] aircomp_aware relaxed {aware.info['relaxed_steps']} step(s): "
              f"fewer than {budget} clients met the channel gate (logged, not silently ignored).")
    print("\nExpected: agg_MSE(no_RA) > agg_MSE(simple); aircomp_aware meets the MSE target "
          "(<=eps) by gating weak channels, trading a little sensing logdet.")
    _maybe_plot(cfg, logger, rows)


def _maybe_plot(cfg, logger, rows) -> None:
    if not cfg.get("logging", {}).get("save_plots", True):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([r["policy"] for r in rows], [r["agg_mse"] for r in rows], color="#C44E52")
    if cfg.constraints.mse_agg_max is not None:
        ax.axhline(float(cfg.constraints.mse_agg_max), ls="--", c="k", lw=1, label="MSE target eps")
        ax.legend(fontsize=8)
    ax.set_ylabel("aggregation MSE"); ax.set_title("AirComp aggregation MSE by A2 policy")
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(logger.path("plots", "aircomp_mse.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

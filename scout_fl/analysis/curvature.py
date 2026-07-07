"""Empirical total curvature of the SCOUT-FL sensing utility (TCCN E-R3).

Instantiates Remark 3 of the paper (curvature-sharpened greedy guarantee) with a
number. For the normalised sensing utility

    f(S) = U_sense(S) - U_sense(empty) ,   U_sense(S) = sum_m w_m logdet(J0 + sum_k J_km),

the total curvature is

    gamma = 1 - min_k [ f(N) - f(N \\ {k}) ] / f({k}) ,

and greedy selection under a cardinality constraint attains the factor
(1/gamma)(1 - e^{-gamma}) >= 1 - 1/e  (Conforti and Cornuejols, 1984). The prior
J0 > 0 keeps every marginal on the full ground set strictly positive, so gamma < 1
strictly and the guarantee strictly improves on the classical constant.

The computation is exact and cheap: it needs only the scenario geometry (per-client,
per-target 2x2 FIMs), no training. It is evaluated on the same scenarios (seeds) as
the main campaign operating point.

Run:  python -m scout_fl.analysis.curvature \
          --config scout_fl/configs/campaign_main.yaml --seeds 0 1 2 3 4
Writes research/paper/figures/stats/curvature.json and prints a summary line.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from scout_fl.experiments.run_synthetic import build_scenario
from scout_fl.objectives.sensing_utility import SensingUtility
from scout_fl.utils.config import load_config
from scout_fl.utils.seed import seed_everything


def sensing_curvature(scn) -> dict:
    """Exact total curvature of the normalised sensing utility for one scenario."""
    sensing = SensingUtility(scn.fim, scn.j0, scn.w)
    K = scn.K
    full = list(range(K))
    f_empty = float(sensing.value([]))
    f_full = float(sensing.value(full))

    ratios = []
    for k in range(K):
        # marginal of k on the full ground set (numerator) and alone (denominator)
        f_wo = float(sensing.value([j for j in full if j != k]))
        top = f_full - f_wo
        bot = float(sensing.value([k])) - f_empty
        if bot <= 0:
            continue                       # degenerate client, contributes nothing alone
        ratios.append(top / bot)
    ratios = np.asarray(ratios, dtype=float)
    gamma = float(1.0 - ratios.min())
    gamma = min(max(gamma, 0.0), 1.0)
    factor = float((1.0 / gamma) * (1.0 - math.exp(-gamma))) if gamma > 0 else 1.0
    return {
        "gamma": gamma,
        "greedy_factor": factor,           # (1/gamma)(1 - e^-gamma)
        "classic_factor": 1.0 - 1.0 / math.e,
        "min_ratio": float(ratios.min()),
        "median_ratio": float(np.median(ratios)),
        "clients_evaluated": int(len(ratios)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Empirical curvature of the sensing utility")
    ap.add_argument("--config", default="scout_fl/configs/campaign_main.yaml")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="research/paper/figures/stats/curvature.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    per_seed = {}
    for seed in args.seeds:
        rng = seed_everything(seed)
        scn = build_scenario(cfg, rng)
        per_seed[seed] = sensing_curvature(scn)
        r = per_seed[seed]
        print(f"[curvature] seed {seed}: gamma={r['gamma']:.4f}  "
              f"greedy factor={r['greedy_factor']:.4f}  (classic {r['classic_factor']:.4f})")

    gammas = [r["gamma"] for r in per_seed.values()]
    factors = [r["greedy_factor"] for r in per_seed.values()]
    summary = {
        "config": args.config,
        "seeds": args.seeds,
        "gamma_mean": float(np.mean(gammas)),
        "gamma_std": float(np.std(gammas, ddof=1)) if len(gammas) > 1 else 0.0,
        "greedy_factor_mean": float(np.mean(factors)),
        "classic_factor": 1.0 - 1.0 / math.e,
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "note": "gamma is the total curvature of the normalised sensing utility "
                "U_sense(S)-U_sense(empty); greedy factor is (1/gamma)(1-e^-gamma) "
                "per Conforti-Cornuejols 1984 (paper Remark 3).",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"[curvature] gamma = {summary['gamma_mean']:.4f} +/- {summary['gamma_std']:.4f} "
          f"over {len(gammas)} seeds -> greedy factor {summary['greedy_factor_mean']:.4f} "
          f"(classic 0.6321); written to {args.out}")


if __name__ == "__main__":
    main()

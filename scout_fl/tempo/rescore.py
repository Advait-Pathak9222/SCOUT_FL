"""Analytic re-scoring of existing campaign artifacts for TEMPO (near-zero GPU).

  gradient_decay_study  — E-T2 premise: does the learning energy L_t (proxy: per-round
                          ||agg||^2 = grad_sq) actually decay under heavy non-IID? Reads
                          the logged grad_sq curves at alpha in {0.1,0.3,0.5}. If flat at
                          alpha=0.1, T1's premise fails there (design R1 -> scope claims).
  static_frontier_rescore — E-T1/E-T4 null: re-score the 32 stationary campaign methods
                          through the Kalman tracker (design §1.6: "re-scored with tracker
                          where trajectories permit"). Produces each static method's
                          (final accuracy, time-averaged tr(P), terminal CRB) point — the
                          static frontier that TEMPO must dominate for GATE 1.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from scout_fl.infra import replay
from scout_fl.infra.tracker import InformationKalmanTracker
from scout_fl.objectives.sensing_utility import SensingUtility


def gradient_decay_study(out_dir, base_config="scout_fl/configs/campaign_main.yaml",
                         alphas=(0.1, 0.3, 0.5), runs_root="runs"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows, verdict = [], {}
    for a in alphas:
        point = f"A_learning_noniid={a:g}"
        files = glob.glob(os.path.join(runs_root, "campaign", point, "*.json"))
        by_round = defaultdict(list)
        for f in files:
            art = replay.load_artifact(f)
            if not art or not art.get("complete"):
                continue
            for r in art["rounds"]:
                if r.get("grad_sq") is not None:
                    by_round[int(r["round"])].append(float(r["grad_sq"]))
        if not by_round:
            verdict[str(a)] = {"decays": None, "note": "no artifacts"}
            continue
        rr = sorted(by_round)
        mean_curve = [float(np.mean(by_round[t])) for t in rr]
        for t, v in zip(rr, mean_curve):
            rows.append({"alpha": a, "round": t, "grad_sq_mean": v})
        # decay = early-window mean > late-window mean (with a clear ratio)
        n = len(mean_curve)
        early = float(np.mean(mean_curve[: max(1, n // 5)]))
        late = float(np.mean(mean_curve[-max(1, n // 5):]))
        ratio = late / early if early > 0 else float("nan")
        verdict[str(a)] = {"early": early, "late": late, "late_over_early": ratio,
                           "decays": bool(ratio < 0.7)}
    with (out / "et2_grad_decay.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["alpha", "round", "grad_sq_mean"])
        w.writeheader(); w.writerows(rows)
    (out / "et2_summary.json").write_text(json.dumps({"gradient_decay": verdict}, indent=2))
    dec = {a: v.get("decays") for a, v in verdict.items()}
    print(f"[E-T2] gradient-norm decays by alpha: {dec}")
    return verdict


def _rescore_one(cfg, artifact, p_max, sigma_p=0.0):
    """Feed one static unit's logged selections through the tracker -> tempo metrics."""
    meta = artifact.get("meta", {})
    rows = artifact.get("rounds", [])
    if not rows:
        return None
    rec = replay.reconstruct(cfg, int(meta.get("seed", 0)))
    scn = rec["scn"]
    targets = np.asarray(scn.targets, dtype=float)
    sensing = SensingUtility(scn.fim, scn.j0, scn.w)
    tracker = InformationKalmanTracker(targets, sigma_p, np.random.default_rng(int(meta.get("seed", 0))))
    trP_hist, rmse_hist, crb_hist = [], [], []
    for i, r in enumerate(rows):
        if i > 0:
            tracker.predict()
        sel = [int(k) for k in r.get("selected", [])]
        J_sel = scn.fim[sel].sum(axis=0)[:, :2, :2] if sel else np.zeros((scn.M, 2, 2))
        tracker.update(J_sel, targets)
        trP_hist.append(float(np.max(tracker.trace_pos())))
        rmse_hist.append(float(np.max(tracker.rmse(targets))))
        crb_hist.append(float((scn.w * sensing.crb(sel)).sum()))
    trP = np.array(trP_hist)
    win = 20
    worst_window = float(max((trP[j:j + win].mean() for j in range(max(1, len(trP) - win + 1))),
                             default=float(trP.mean())))
    return {"acc": float(rows[-1]["test_acc"]),
            "time_avg_trP": float(trP.mean()), "worst_window_trP": worst_window,
            "final_trP": float(trP[-1]), "crb_final": float(crb_hist[-1]),
            "track_rmse": float(np.mean(rmse_hist))}


def static_frontier_rescore(out_dir, point="A_datasets=cifar10",
                            base_config="scout_fl/configs/campaign_main.yaml",
                            p_max=20.0, sigma_p=0.0, runs_root="runs"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = replay.config_for_point(point, base_config)
    files = sorted(glob.glob(os.path.join(runs_root, "campaign", point, "*.json")))
    per_method = defaultdict(list)
    n = 0
    for f in files:
        art = replay.load_artifact(f)
        if not art or not art.get("complete"):
            continue
        m = art["meta"]["method"]
        res = _rescore_one(cfg, art, p_max, sigma_p)
        if res is None:
            continue
        res["seed"] = int(art["meta"]["seed"])
        per_method[m].append(res)
        n += 1
    rows = []
    for m in sorted(per_method):
        recs = per_method[m]
        agg = {"method": m, "n_seeds": len(recs), "is_static": True}
        for k in ("acc", "time_avg_trP", "worst_window_trP", "final_trP", "crb_final", "track_rmse"):
            agg[f"{k}_mean"] = float(np.mean([r[k] for r in recs]))
            agg[f"{k}_std"] = float(np.std([r[k] for r in recs]))
        rows.append(agg)
    with (out / "static_frontier.csv").open("w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    # also dump per-seed for paired tests
    (out / "static_frontier_per_seed.json").write_text(
        json.dumps({m: per_method[m] for m in per_method}, indent=2))
    (out / "static_frontier_summary.json").write_text(
        json.dumps({"point": point, "sigma_p": sigma_p, "n_units": n, "methods": rows}, indent=2))
    print(f"[E-T1-static] re-scored {n} static units at {point} (sigma_p={sigma_p}) -> {len(rows)} methods")
    return rows

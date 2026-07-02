"""E-C4 — "every existing method localizes its clients" (design §2.6).

Instrument the client-position leakage accountant across the existing 28+ methods
by RE-SCORING their logged per-round selections (no FL re-run): replay reconstructs
each unit's client positions + uplink SNR from (config, seed) — verified faithful in
analysis/schema_report.md — then the accountant accumulates leakage round by round.

Expected killer figure: every published ISAC-FL selector localizes its median client
to sub-meter precision within tens of rounds; sensing-aggressive methods (Asaad,
CRB-only) are the worst offenders. Standalone empirical contribution; near-zero GPU.

Fallback (design §2.6): if an artifact set is missing, run_all.sh re-runs the top-9
methods once (ec4_rerun) — but with faithful replay, all present methods are scored.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from scout_fl.infra.leakage import LeakageAccountant
from scout_fl.infra import replay


def score_unit(cfg, artifact):
    """Return per-round (round, leak_r_median, leak_r_min) for one logged unit."""
    meta = artifact.get("meta", {})
    rows = artifact.get("rounds", [])
    if not rows:
        return None
    rec = replay.reconstruct(cfg, int(meta.get("seed", 0)))
    scn, snr_up = rec["scn"], rec["snr_up"]
    bs = np.asarray(cfg.geometry.bs_position, dtype=float)
    acct = LeakageAccountant(scn.clients, bs, k_range=float(cfg.sensing.k_range),
                             k_angle=float(cfg.sensing.k_angle),
                             prior_std_m=float(cfg.get("cloak", {}).get("prior_client_std_m", 100.0)))
    out = []
    for r in rows:
        acct.observe([int(k) for k in r.get("selected", [])], snr_up, atten=1.0)
        s = acct.summary()
        out.append((int(r["round"]), s["leak_r_median"], s["leak_r_min"]))
    return out


def run_ec4(out_dir, runs_root="runs", point="A_datasets=cifar10",
            base_config="scout_fl/configs/campaign_main.yaml", seeds=None):
    """Re-score every method at ``point`` across seeds; write per-method CRB-floor curves."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = replay.config_for_point(point, base_config)
    files = sorted(glob.glob(os.path.join(runs_root, "campaign", point, "*.json")))

    med_round = defaultdict(lambda: defaultdict(list))          # method -> round -> [r_median,...]
    min_round = defaultdict(lambda: defaultdict(list))          # method -> round -> [r_worstclient,...]
    med_final = defaultdict(list)
    min_final = defaultdict(list)
    n_scored = 0
    for f in files:
        art = replay.load_artifact(f)
        if not art or not art.get("complete"):
            continue
        m = art["meta"]["method"]
        if seeds is not None and art["meta"].get("seed") not in seeds:
            continue
        curve = score_unit(cfg, art)
        if curve is None:
            continue
        n_scored += 1
        for rnd, r_med, r_min in curve:
            med_round[m][rnd].append(r_med)
            min_round[m][rnd].append(r_min)
        med_final[m].append(curve[-1][1])
        min_final[m].append(curve[-1][2])                      # final-round worst-client r

    # per-method mean curve over seeds (median AND worst-client r vs rounds)
    curve_rows = []
    for m in sorted(med_round):
        for rnd in sorted(med_round[m]):
            curve_rows.append({"method": m, "round": rnd,
                               "leak_r_median_mean": float(np.mean(med_round[m][rnd])),
                               "leak_r_median_std": float(np.std(med_round[m][rnd])),
                               "leak_r_worst_mean": float(np.mean(min_round[m][rnd]))})
    with (out / "ec4_leakage_curves.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "round", "leak_r_median_mean",
                                           "leak_r_median_std", "leak_r_worst_mean"])
        w.writeheader(); w.writerows(curve_rows)

    def _rounds_to(thresh, per_round):
        for rnd in sorted(per_round):
            if np.mean(per_round[rnd]) <= thresh:
                return rnd
        return None

    summary = {}
    for m in med_final:
        summary[m] = {
            "final_leak_r_median_m": float(np.mean(med_final[m])),
            "final_leak_r_median_std": float(np.std(med_final[m])),
            "final_leak_r_worst_m": float(np.mean(min_final[m])),      # most-exposed client (design §2.3)
            "n_seeds": len(med_final[m]),
            "rounds_worst_to_10m": _rounds_to(10.0, min_round[m]),
            "rounds_worst_to_5m": _rounds_to(5.0, min_round[m]),
        }
    # sort by worst-client exposure (the privacy-relevant metric): worst offenders first
    summary = dict(sorted(summary.items(), key=lambda kv: kv[1]["final_leak_r_worst_m"]))
    (out / "ec4_summary.json").write_text(json.dumps(
        {"point": point, "n_units_scored": n_scored, "per_method": summary}, indent=2))
    worst = next(iter(summary.items()), (None, {}))
    print(f"[E-C4] scored {n_scored} units across {len(summary)} methods; "
          f"worst offender (most-exposed client): {worst[0]} "
          f"({worst[1].get('final_leak_r_worst_m', float('nan')):.2f} m)")
    return summary

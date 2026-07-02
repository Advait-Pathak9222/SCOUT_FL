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


def score_unit(cfg, artifact, side_snr=None):
    """Return per-round (round, leak_r_median, leak_r_min[, side_r_min]) for one unit.

    ``side_snr`` (E-C6 decomposition, design §2.6): if given, a SECOND accountant is run
    in which the BS gets NO physical returns — only the selection side-channel. Each
    selection event is modeled as one coarse position observation at a fixed effective
    SNR ``side_snr`` (a modeled bound, config-exposed) instead of the uplink SNR
    (median ~25, up to ~1600). The gap between the two curves quantifies how much
    leakage is via geometry-of-selection vs via signals.
    """
    meta = artifact.get("meta", {})
    rows = artifact.get("rounds", [])
    if not rows:
        return None
    rec = replay.reconstruct(cfg, int(meta.get("seed", 0)))
    scn, snr_up = rec["scn"], rec["snr_up"]
    bs = np.asarray(cfg.geometry.bs_position, dtype=float)
    kw = dict(k_range=float(cfg.sensing.k_range), k_angle=float(cfg.sensing.k_angle),
              prior_std_m=float(cfg.get("cloak", {}).get("prior_client_std_m", 100.0)))
    acct = LeakageAccountant(scn.clients, bs, **kw)
    side = LeakageAccountant(scn.clients, bs, **kw) if side_snr else None
    side_const = np.full(scn.K, float(side_snr)) if side_snr else None
    out = []
    for r in rows:
        sel = [int(k) for k in r.get("selected", [])]
        acct.observe(sel, snr_up, atten=1.0)
        s = acct.summary()
        rec_row = [int(r["round"]), s["leak_r_median"], s["leak_r_min"]]
        if side is not None:
            side.observe(sel, side_const, atten=1.0)
            rec_row.append(side.summary()["leak_r_min"])
        out.append(tuple(rec_row))
    return out


def run_ec4(out_dir, runs_root="runs", point="A_datasets=cifar10",
            base_config="scout_fl/configs/campaign_main.yaml", seeds=None, side_snr=1.0):
    """Re-score every method at ``point`` across seeds; write per-method CRB-floor curves.

    ``side_snr`` also runs the E-C6 selection-side-channel-only decomposition
    (set None to disable)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = replay.config_for_point(point, base_config)
    files = sorted(glob.glob(os.path.join(runs_root, "campaign", point, "*.json")))

    med_round = defaultdict(lambda: defaultdict(list))          # method -> round -> [r_median,...]
    min_round = defaultdict(lambda: defaultdict(list))          # method -> round -> [r_worstclient,...]
    side_round = defaultdict(lambda: defaultdict(list))         # method -> round -> [side-channel r_min,...]
    med_final = defaultdict(list)
    min_final = defaultdict(list)
    side_final = defaultdict(list)
    n_scored = 0
    for f in files:
        art = replay.load_artifact(f)
        if not art or not art.get("complete"):
            continue
        m = art["meta"]["method"]
        if seeds is not None and art["meta"].get("seed") not in seeds:
            continue
        curve = score_unit(cfg, art, side_snr=side_snr)
        if curve is None:
            continue
        n_scored += 1
        for row in curve:
            rnd, r_med, r_min = row[0], row[1], row[2]
            med_round[m][rnd].append(r_med)
            min_round[m][rnd].append(r_min)
            if len(row) > 3:
                side_round[m][rnd].append(row[3])
        med_final[m].append(curve[-1][1])
        min_final[m].append(curve[-1][2])                      # final-round worst-client r
        if len(curve[-1]) > 3:
            side_final[m].append(curve[-1][3])

    # per-method mean curve over seeds (median AND worst-client r vs rounds)
    curve_rows = []
    for m in sorted(med_round):
        for rnd in sorted(med_round[m]):
            row = {"method": m, "round": rnd,
                   "leak_r_median_mean": float(np.mean(med_round[m][rnd])),
                   "leak_r_median_std": float(np.std(med_round[m][rnd])),
                   "leak_r_worst_mean": float(np.mean(min_round[m][rnd]))}
            if side_round[m].get(rnd):
                row["side_channel_r_worst_mean"] = float(np.mean(side_round[m][rnd]))
            curve_rows.append(row)
    cols = ["method", "round", "leak_r_median_mean", "leak_r_median_std", "leak_r_worst_mean"]
    if side_snr:
        cols.append("side_channel_r_worst_mean")
    with (out / "ec4_leakage_curves.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
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
        if side_final.get(m):
            # E-C6 decomposition: leakage via selection side-channel alone (no returns)
            summary[m]["final_side_channel_r_worst_m"] = float(np.mean(side_final[m]))
    # sort by worst-client exposure (the privacy-relevant metric): worst offenders first
    summary = dict(sorted(summary.items(), key=lambda kv: kv[1]["final_leak_r_worst_m"]))
    (out / "ec4_summary.json").write_text(json.dumps(
        {"point": point, "n_units_scored": n_scored, "side_snr": side_snr,
         "per_method": summary}, indent=2))
    worst = next(iter(summary.items()), (None, {}))
    print(f"[E-C4] scored {n_scored} units across {len(summary)} methods; "
          f"worst offender (most-exposed client): {worst[0]} "
          f"({worst[1].get('final_leak_r_worst_m', float('nan')):.2f} m)")
    return summary

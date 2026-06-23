"""Collect the resumable per-round JSON store (runs/) into flat tables for analysis.

Walks ``runs/<tag>/<point>/<method>__seed<seed>.json`` and emits, under
``runs/<tag>/`` (or a chosen output dir):

  * ``all_rounds.csv``  — one row per (point, method, seed, round) with every logged
    metric + JEDI diagnostic: the raw source for convergence curves, schedule plots, etc.
  * ``summary.csv``     — final-round / mean metrics per (point, method) aggregated
    over seeds (mean & std): the source for the headline tables.

CLI:
  python -m scout_fl.analysis.collect                 # all tags under runs/
  python -m scout_fl.analysis.collect --tag campaign  # one tag
  python -m scout_fl.analysis.collect runs --out runs/collected
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

_SUMMARY_METRICS = ["test_acc", "sensing_logdet", "crb", "agg_mse", "energy",
                    "jedi_learn_frac", "jedi_deficit_mean"]


def _iter_units(runs_root: Path, tag: str | None, complete_only: bool = True):
    """Yield (tag, point, unit-dict). By default skip incomplete/partial units so analysis
    tables and plots are never polluted by interrupted runs (mid-training round dumps)."""
    skipped = 0
    for unit in sorted(runs_root.glob("*/*/*.json")):
        rel_tag = unit.parent.parent.name
        if tag and rel_tag != tag:
            continue
        try:
            d = json.loads(unit.read_text())
        except (ValueError, OSError):
            continue
        if complete_only and not d.get("complete"):
            skipped += 1
            continue
        yield rel_tag, unit.parent.name, d
    if skipped:
        print(f"[collect] skipped {skipped} incomplete unit(s) (still training / interrupted)")


def collect(runs_root="runs", tag=None, out=None):
    runs_root = Path(runs_root)
    rows, finals = [], {}
    for rel_tag, point, d in _iter_units(runs_root, tag):
        meta = d.get("meta", {})
        method, seed = meta.get("method"), meta.get("seed")
        rnds = d.get("rounds", [])
        for r in rnds:
            flat = {"tag": rel_tag, "point": point, "method": method, "seed": seed,
                    "complete": d.get("complete", False)}
            flat.update({k: (str(v) if k == "selected" else v) for k, v in r.items()})
            rows.append(flat)
        if rnds:                                              # final-round snapshot per unit
            finals.setdefault((rel_tag, point, method), []).append(rnds[-1])

    # per-(tag, point, method) summary over seeds
    summary = []
    for (rel_tag, point, method), last_rows in sorted(finals.items()):
        s = {"tag": rel_tag, "point": point, "method": method, "n_seeds": len(last_rows)}
        for m in _SUMMARY_METRICS:
            vals = [float(r[m]) for r in last_rows if m in r and r[m] is not None]
            if vals:
                s[f"{m}_mean"] = round(float(np.mean(vals)), 5)
                s[f"{m}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 5)
        summary.append(s)

    out_dir = Path(out) if out else (runs_root / (tag or "_all"))
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "all_rounds.csv", rows)
    _write_csv(out_dir / "summary.csv", summary)
    print(f"[collect] {len(rows)} round-rows, {len(summary)} (point,method) summaries -> {out_dir}")
    return out_dir


def _write_csv(path: Path, rows: list):
    if not rows:
        path.write_text("")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))    # union of keys, stable order
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser(description="Collect runs/ per-round JSON into CSV tables")
    p.add_argument("runs_root", nargs="?", default="runs")
    p.add_argument("--tag", default=None, help="only this tag (e.g. campaign, ablation, campaign_main)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    collect(args.runs_root, args.tag, args.out)


if __name__ == "__main__":
    main()

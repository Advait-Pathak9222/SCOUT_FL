"""How far each selector is from choosing clients on channel quality alone.

A scheduler that reduces sensing to a scalar SNR has a problem in any geometry where
the targets sit near the base station. The sensing SNR and the communication gain are
then both monotone in the same distance, so any product or weighted sum of the two
ranks the clients exactly as the channel gain alone does, and the sensing term buys
nothing. A geometry aware criterion escapes this, because the Fisher information
depends on the bearing to the target and not on the range alone.

This module measures the effect rather than asserting it. For every method it computes
the Jaccard overlap between its active set and the active set that pure channel quality
selection would have chosen in the same round, averaged over rounds and seeds, and it
counts how many genuinely distinct selection trajectories the pool contains.

    python -m scout_fl.analysis.baseline_overlap --tag tccn_main
    python -m scout_fl.analysis.baseline_overlap --tag tccn_campaign --point A_datasets=cifar10
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

import numpy as np

REFERENCE = "comm_only"          # selection on channel gain alone


def _load(tag: str, point: str | None, runs_root: str = "runs") -> dict[str, dict[int, list[set]]]:
    """{method: {seed: [set of selected clients per round]}} for the complete units."""
    out: dict[str, dict[int, list[set]]] = {}
    pattern = os.path.join(runs_root, tag, "**", "*.json")
    for f in glob.glob(pattern, recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not d.get("complete"):
            continue
        m = d.get("meta", {})
        if point is not None and m.get("point") != point:
            continue
        rows = [set(r["selected"]) for r in d.get("rounds", []) if "selected" in r]
        if rows:
            out.setdefault(m["method"], {})[int(m["seed"])] = rows
    return out


def _jaccard(a: list[set], b: list[set]) -> float:
    n = min(len(a), len(b))
    if not n:
        return float("nan")
    return float(np.mean([len(a[t] & b[t]) / max(len(a[t] | b[t]), 1) for t in range(n)]))


def overlap_report(tag: str, point: str | None = None, runs_root: str = "runs") -> dict:
    sel = _load(tag, point, runs_root)
    if not sel:
        raise SystemExit(f"no complete runs under {runs_root}/{tag}"
                         + (f" at point {point}" if point else ""))

    report: dict = {"tag": tag, "point": point, "methods": sorted(sel), "reference": REFERENCE}

    # distance from channel quality selection
    if REFERENCE in sel:
        ref = sel[REFERENCE]
        rows = {}
        for m, seeds in sel.items():
            if m == REFERENCE:
                continue
            vals = [_jaccard(seeds[s], ref[s]) for s in sorted(set(seeds) & set(ref))]
            if vals:
                rows[m] = {"mean": round(float(np.mean(vals)), 4),
                           "std": round(float(np.std(vals)), 4), "seeds": len(vals)}
        report["overlap_with_channel_only"] = dict(
            sorted(rows.items(), key=lambda kv: -kv[1]["mean"]))

    # pairs that are the same selector under two names
    dupes = []
    for a, b in itertools.combinations(sorted(sel), 2):
        common = sorted(set(sel[a]) & set(sel[b]))
        if not common:
            continue
        j = float(np.mean([_jaccard(sel[a][s], sel[b][s]) for s in common]))
        if j > 0.98:
            dupes.append({"a": a, "b": b, "jaccard": round(j, 4)})
    report["duplicate_selectors"] = sorted(dupes, key=lambda d: -d["jaccard"])

    # how many distinct trajectories the pool really contains, on the first shared seed
    seed0 = min(set().union(*[set(v) for v in sel.values()]))
    sig: dict[tuple, list[str]] = {}
    for m, seeds in sel.items():
        if seed0 in seeds:
            sig.setdefault(tuple(tuple(sorted(x)) for x in seeds[seed0]), []).append(m)
    report["n_methods"] = len(sel)
    report["n_distinct_selectors"] = len(sig)
    report["identical_groups"] = [sorted(g) for g in sig.values() if len(g) > 1]
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="tccn_main")
    ap.add_argument("--point", default=None)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default=None, help="write the report as JSON here")
    args = ap.parse_args()

    rep = overlap_report(args.tag, args.point, args.runs_root)
    print(f"\nselection overlap with {REFERENCE} (channel quality alone), "
          f"tag={rep['tag']}" + (f" point={rep['point']}" if rep["point"] else ""))
    print(f"{'method':20s} {'overlap':>8s}   {'':s}")
    for m, v in rep.get("overlap_with_channel_only", {}).items():
        bar = "#" * int(round(v["mean"] * 40))
        note = "   collapses onto channel quality" if v["mean"] > 0.75 else ""
        print(f"  {m:18s} {v['mean']:6.3f}   {bar}{note}")
    print(f"\ndistinct selection trajectories: {rep['n_distinct_selectors']} "
          f"of {rep['n_methods']} methods")
    for g in rep["identical_groups"]:
        print("  identical:", ", ".join(g))
    out = args.out or os.path.join("research", "paper", "figures", "stats", "baseline_overlap.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1)
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()

"""A compact, readable summary of whatever a run store currently holds.

The unattended driver calls this after every stage so the log shows results as they
arrive rather than only at the end. It reads the resumable per round store directly, so
it works on a partially finished stage and never fails a pipeline. Nothing here writes.

    python -m scout_fl.analysis.digest --tag tccn_main
    python -m scout_fl.analysis.digest --tag tccn_campaign --by-point
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

_HEAD = "scout_v2"
_NICE = {"scout_v2": "SCOUT-FL", "scout_greedy": "SCOUT-FL dagger", "collabsensefed": "CollabSenseFed",
         "sensing_native": "Sensing-Native", "asaad": "Asaad", "fixed_weighted": "Fixed-Weighted",
         "fed_iscc": "Fed-ISCC", "ota_fl_iscc": "OTA-FL-ISCC", "iscc_air_feel": "ISCC-Air-FEEL",
         "fedavg_iscc": "FedAvg-ISCC", "fedsgd_iscc": "FedSGD-ISCC", "crb_only": "CRB-Only",
         "sensing_only": "Sensing-Only", "comm_only": "Channel-Only"}


def _units(tag: str, runs_root: str = "runs") -> list[dict]:
    out = []
    for f in glob.glob(os.path.join(runs_root, tag, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if isinstance(d, dict) and "meta" in d:
            out.append(d)
    return out


def _tail(rows: list[dict], key: str, n: int = 25, default=np.nan) -> float:
    vals = [r[key] for r in rows[-n:] if key in r and r[key] is not None]
    return float(np.mean(vals)) if vals else default


def _fmt(v: float, spec: str) -> str:
    """Print a missing quantity as a dash rather than as nan."""
    return format(v, spec) if np.isfinite(v) else "-".rjust(len(format(0.0, spec)))


def _mean(v: list[float]) -> float:
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def digest(tag: str, runs_root: str = "runs", by_point: bool = False, top: int = 20) -> None:
    units = _units(tag, runs_root)
    if not units:
        print(f"  [digest] nothing under {runs_root}/{tag} yet")
        return
    done = [u for u in units if u.get("complete")]
    print(f"  [digest] {tag}: {len(done)} of {len(units)} units complete")
    if not done:
        return

    if by_point:
        pts: dict[str, list] = {}
        for u in done:
            pts.setdefault(u["meta"].get("point", "base"), []).append(u)
        print(f"  {'operating point':32s} {'units':>6s} {'best acc':>9s} {'method':>16s}")
        for p in sorted(pts):
            us = pts[p]
            best = max(us, key=lambda u: u["objectives"].get("acc", -1))
            print(f"  {p:32s} {len(us):6d} {best['objectives']['acc']*100:8.1f}% "
                  f"{_NICE.get(best['meta']['method'], best['meta']['method']):>16s}")
        return

    agg: dict[str, dict[str, list]] = {}
    for u in done:
        m = u["meta"]["method"]
        o = u["objectives"]
        rows = u.get("rounds", [])
        a = agg.setdefault(m, {"acc": [], "crb": [], "mse": [], "dual": [], "lat": []})
        a["acc"].append(o.get("acc", np.nan) * 100)
        a["crb"].append(o.get("crb_final", o.get("crb", np.nan)))
        a["mse"].append(_tail(rows, "agg_mse"))
        a["dual"].append(_tail(rows, "dual_mse", default=0.0))
        a["lat"].append(_tail(rows, "round_latency_s"))

    order = sorted(agg, key=lambda m: -_mean(agg[m]["acc"]))
    print(f"  {'method':18s} {'seeds':>5s} {'acc %':>13s} {'CRB':>8s} "
          f"{'agg MSE':>10s} {'dual':>8s} {'latency s':>10s}")
    for m in order[:top]:
        a = agg[m]
        mark = " <" if m == _HEAD else "  "
        print(f"  {_NICE.get(m, m):18s} {len(a['acc']):5d} "
              f"{_fmt(_mean(a['acc']), '8.1f')} +- {np.std([x for x in a['acc'] if np.isfinite(x)] or [0]):3.1f} "
              f"{_fmt(_mean(a['crb']), '8.4f')} {_fmt(_mean(a['mse']), '10.3e')} "
              f"{_fmt(_mean(a['dual']), '8.3f')} {_fmt(_mean(a['lat']), '10.3f')}{mark}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--by-point", action="store_true")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    try:
        digest(args.tag, args.runs_root, args.by_point, args.top)
    except Exception as exc:                       # a digest must never fail a pipeline
        print(f"  [digest] skipped ({exc})")


if __name__ == "__main__":
    main()

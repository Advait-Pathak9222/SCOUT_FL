"""P3-dual validation — primal-dual feasibility / bounded constraint violation.

Dual ascent on the AirComp-MSE constraint (SCOUT-v2) and the participation virtual queue
(JEDI fairness) should keep the TIME-AVERAGED constraint violation bounded and decaying
(drift-plus-penalty / online-convex feasibility, Neely 2010): constraints hold ON AVERAGE
with no fixed weights, and the dual variables stay bounded (do not diverge).

From the run store we read the logged per-round `dual_mse` (the MSE dual mu), `mse_violation`
(realized (MSE - eps)_+), and the JEDI participation deficit `jedi_deficit_mean`, and check:
  (1) running-average violation  V_bar(t) = (1/t) sum_{s<=t} viol_s  is non-increasing in the
      tail / converges to a small value (slope of the tail <= ~0);
  (2) the dual / deficit stays bounded (finite, non-exploding).

CLI:  python -m scout_fl.analysis.feasibility [runs_root] [--tag campaign_main] [--method scout_v2]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from scout_fl.analysis.collect import _iter_units


def _series_by_method(runs_root: Path, tag, key):
    """Mean per-round series of `key`, per method, averaged over seeds."""
    bucket = defaultdict(lambda: defaultdict(list))   # method -> round -> [vals]
    for _tag, _point, d in _iter_units(runs_root, tag):
        m = d.get("meta", {}).get("method", "?")
        for r in d.get("rounds", []):
            if key in r and r[key] is not None:
                bucket[m][int(r["round"])].append(float(r[key]))
    out = {}
    for m, byr in bucket.items():
        rounds = sorted(byr)
        out[m] = (np.array(rounds), np.array([np.mean(byr[t]) for t in rounds]))
    return out


def _tail_slope(y):
    """Slope of the running-average over the last half (<=0 => not growing => bounded)."""
    if len(y) < 4:
        return float("nan")
    run_avg = np.cumsum(y) / (np.arange(len(y)) + 1.0)
    half = len(run_avg) // 2
    x = np.arange(half, len(run_avg))
    if len(x) < 2:
        return float("nan")
    return float(np.polyfit(x, run_avg[half:], 1)[0])


def validate(runs_root="runs", tag=None, method="scout_v2") -> dict:
    viol = _series_by_method(Path(runs_root), tag, "mse_violation")
    dual = _series_by_method(Path(runs_root), tag, "dual_mse")
    deficit = _series_by_method(Path(runs_root), tag, "jedi_deficit_mean")
    res = {"method": method}
    if method in viol:
        _, v = viol[method]
        res["final_running_avg_violation"] = float(np.cumsum(v)[-1] / len(v))
        res["violation_tail_slope"] = _tail_slope(v)
        res["violation_bounded"] = bool(res["violation_tail_slope"] <= 1e-6)
    if method in dual:
        _, mu = dual[method]
        res["dual_max"] = float(np.max(mu)); res["dual_final"] = float(mu[-1])
        res["dual_bounded"] = bool(np.isfinite(mu).all() and mu.max() < 1e6)
    # JEDI fairness queue stability (deficit should stabilize, not explode)
    if "jedi" in deficit:
        _, q = deficit["jedi"]
        res["jedi_deficit_tail_slope"] = _tail_slope(q)
        res["jedi_queue_stable"] = bool(np.isfinite(q).all() and _tail_slope(q) <= 1e-3)
    res["feasibility_supported"] = bool(res.get("violation_bounded", True)
                                        and res.get("dual_bounded", True))
    return res


def _format(res: dict) -> str:
    out = [f"=== P3-dual primal-dual feasibility (method={res['method']}) ==="]
    if "final_running_avg_violation" in res:
        out.append(f"  time-avg MSE violation = {res['final_running_avg_violation']:.3e}  "
                   f"tail-slope = {res['violation_tail_slope']:+.2e}  "
                   f"(bounded/decaying: {'YES' if res['violation_bounded'] else 'NO'})")
    if "dual_max" in res:
        out.append(f"  dual mu: max={res['dual_max']:.3e} final={res['dual_final']:.3e}  "
                   f"(bounded: {'YES' if res['dual_bounded'] else 'NO'})")
    if "jedi_deficit_tail_slope" in res:
        out.append(f"  JEDI participation deficit tail-slope = {res['jedi_deficit_tail_slope']:+.2e}  "
                   f"(queue stable: {'YES' if res['jedi_queue_stable'] else 'NO'})")
    out.append(f"  => feasibility / bounded-violation supported: "
               f"{'YES' if res['feasibility_supported'] else 'NO'}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description="P3-dual feasibility validation from runs/")
    p.add_argument("runs_root", nargs="?", default="runs")
    p.add_argument("--tag", default=None)
    p.add_argument("--method", default="scout_v2")
    args = p.parse_args()
    print(_format(validate(args.runs_root, args.tag, args.method)))


if __name__ == "__main__":
    main()

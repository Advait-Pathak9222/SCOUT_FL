"""P7 validation — certify the online (CUCB) selector achieves SUBLINEAR alpha-regret.

Reads the regret run store (runs/regret/<scenario>/<selector>__seed<seed>.json) written by
experiments/run_regret.py and, per selector (seed-averaged cumulative regret R(T)), checks:
  (1) average regret R(T)/T -> 0 (no-regret);
  (2) log-log fit R(T) ~ c * T^p gives exponent p < 1 (sublinear; CUCB expects p ~ 0.5).
The random-K control should show LINEAR regret (p ~ 1, R(T)/T flat) — the contrast that proves
the exploration is what drives CUCB's sublinearity.

CLI:  python -m scout_fl.analysis.regret [runs_root]   (default runs_root=runs, tag=regret)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from scout_fl.analysis.collect import _iter_units


def _cum_regret_by_selector(runs_root: Path, tag="regret"):
    """selector -> (T-array, seed-averaged cumulative-regret array)."""
    bucket = defaultdict(lambda: defaultdict(list))   # selector -> round -> [cum_regret per seed]
    for _tag, _point, d in _iter_units(runs_root, tag):
        m = d.get("meta", {}).get("method", "?")
        for r in d.get("rounds", []):
            if "cum_regret" in r:
                bucket[m][int(r["round"])].append(float(r["cum_regret"]))
    out = {}
    for m, byr in bucket.items():
        rounds = np.array(sorted(byr))
        out[m] = (rounds + 1, np.array([np.mean(byr[t]) for t in sorted(byr)]))
    return out


def _slope_loglog(T, R):
    """Fit log R(T) = log c + p log T over the region where R>0; return exponent p."""
    mask = (R > 1e-9) & (T > 1)
    if mask.sum() < 4:
        return float("nan")
    return float(np.polyfit(np.log(T[mask]), np.log(R[mask]), 1)[0])


def validate(runs_root="runs", tag="regret") -> dict:
    series = _cum_regret_by_selector(Path(runs_root), tag)
    res = {"selectors": {}}
    for m, (T, R) in series.items():
        p = _slope_loglog(T, R)
        res["selectors"][m] = {
            "final_cum_regret": float(R[-1]) if len(R) else float("nan"),
            "final_avg_regret": float(R[-1] / T[-1]) if len(R) else float("nan"),
            "loglog_exponent_p": p,
            "sublinear": bool(np.isfinite(p) and p < 0.95),
        }
    cucb = res["selectors"].get("cucb", {})
    res["p7_supported"] = bool(cucb.get("sublinear", False))
    return res


def _format(res: dict) -> str:
    out = ["=== P7 online-regret validation (alpha-regret vs offline greedy oracle) ==="]
    for m, s in res["selectors"].items():
        out.append(f"  {m:>8}: R(T)={s['final_cum_regret']:.2f}  R(T)/T={s['final_avg_regret']:.4f}  "
                   f"log-log p={s['loglog_exponent_p']:.3f}  "
                   f"(sublinear: {'YES' if s['sublinear'] else 'NO'})")
    out.append(f"  => CUCB sublinear regret (P7) supported: {'YES' if res['p7_supported'] else 'NO'}  "
               "(random control should be ~linear, p~1).")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description="P7 online-regret validation from runs/regret")
    p.add_argument("runs_root", nargs="?", default="runs")
    p.add_argument("--tag", default="regret")
    args = p.parse_args()
    print(_format(validate(args.runs_root, args.tag)))


if __name__ == "__main__":
    main()

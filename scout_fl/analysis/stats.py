"""Statistical-significance module for the campaign (Test E).

Turns the per-seed bake-off results into a publication-ready significance report:

  * mean +/- std and a 95% confidence interval (Student-t) per method/metric;
  * paired t-test and Wilcoxon signed-rank test of each method vs a reference
    (default: the proposed JEDI-FL), paired across seeds;
  * Friedman omnibus test across all methods (seeds = blocks);
  * Nemenyi post-hoc critical difference (Demsar 2006) with the pairwise
    significant-difference matrix.

CLI:  python -m scout_fl.analysis.stats <run_dir>
      (reads ``<run_dir>/per_seed.json`` written by the bake-off runner)

Designed for >=5 seeds (the campaign target); it runs with fewer but the tests
are then under-powered, which the report states explicitly.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

# +1 = higher-is-better, -1 = lower-is-better. Used for ranking in Friedman/Nemenyi.
DIRECTIONS = {"acc": 1, "best_acc": 1, "logdet": 1, "jain": 1,
              "crb": -1, "agg_mse": -1, "energy": -1, "round_s": -1}
DEFAULT_METRICS = ["acc", "logdet", "crb", "agg_mse", "jain"]


def summarize(values, ci: float = 0.95) -> dict:
    """Mean, std, SEM and a Student-t confidence interval for one method/metric."""
    a = np.asarray(values, dtype=float)
    n = a.size
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        sem = std / math.sqrt(n)
        half = float(stats.t.ppf(0.5 + ci / 2.0, df=n - 1)) * sem
    else:
        sem, half = 0.0, 0.0
    return {"mean": mean, "std": std, "sem": sem, "n": int(n),
            "ci_low": mean - half, "ci_high": mean + half, "ci": ci}


def paired_ttest(a, b) -> dict:
    """Paired t-test of a vs b across matched seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or np.allclose(a, b):
        return {"t": float("nan"), "p": float("nan")}
    t, p = stats.ttest_rel(a, b)
    return {"t": float(t), "p": float(p)}


def wilcoxon_test(a, b) -> dict:
    """Wilcoxon signed-rank test of a vs b (non-parametric paired)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or np.allclose(a, b):
        return {"stat": float("nan"), "p": float("nan")}
    try:
        with warnings.catch_warnings():                       # quiet small-n approximation notes
            warnings.simplefilter("ignore")
            stat, p = stats.wilcoxon(a, b)
        return {"stat": float(stat), "p": float(p)}
    except ValueError:
        return {"stat": float("nan"), "p": float("nan")}


def friedman(method_values: dict) -> dict:
    """Friedman omnibus test across >=3 methods (seeds are the blocks)."""
    methods = list(method_values)
    if len(methods) < 3:
        return {"chi2": float("nan"), "p": float("nan"), "note": "need >=3 methods"}
    cols = [np.asarray(method_values[m], float) for m in methods]
    n = min(len(c) for c in cols)
    if n < 2:
        return {"chi2": float("nan"), "p": float("nan"), "note": "need >=2 seeds"}
    cols = [c[:n] for c in cols]
    chi2, p = stats.friedmanchisquare(*cols)
    return {"chi2": float(chi2), "p": float(p), "k": len(methods), "n_blocks": n}


def average_ranks(method_values: dict, direction: int = 1) -> dict:
    """Mean rank of each method across seeds (rank 1 = best given direction)."""
    methods = list(method_values)
    n = min(len(method_values[m]) for m in methods)
    mat = np.array([np.asarray(method_values[m], float)[:n] for m in methods])  # (k, n)
    # rank within each block; higher-better -> negate so rank 1 = best
    signed = -mat if direction > 0 else mat
    ranks = np.apply_along_axis(stats.rankdata, 0, signed)                      # (k, n)
    return {m: float(ranks[i].mean()) for i, m in enumerate(methods)}


def nemenyi(method_values: dict, direction: int = 1, alpha: float = 0.05) -> dict:
    """Nemenyi post-hoc: critical difference + pairwise significant-difference matrix."""
    methods = list(method_values)
    k = len(methods)
    n = min(len(method_values[m]) for m in methods)
    if k < 3 or n < 2:
        return {"critical_difference": float("nan"), "ranks": {}, "significant": {},
                "note": "need >=3 methods and >=2 seeds"}
    ranks = average_ranks(method_values, direction)
    q = float(stats.studentized_range.ppf(1.0 - alpha, k, np.inf)) / math.sqrt(2.0)
    cd = q * math.sqrt(k * (k + 1) / (6.0 * n))
    sig = {}
    for i, mi in enumerate(methods):
        for mj in methods[i + 1:]:
            sig[f"{mi} vs {mj}"] = bool(abs(ranks[mi] - ranks[mj]) > cd)
    return {"critical_difference": cd, "alpha": alpha, "ranks": ranks, "significant": sig}


def statistical_report(per_seed: dict, metrics=None, reference: str = "jedi",
                       directions=None) -> dict:
    """Full report: per-method summary + reference comparison + Friedman + Nemenyi per metric."""
    metrics = metrics or [m for m in DEFAULT_METRICS if _has_metric(per_seed, m)]
    directions = directions or DIRECTIONS
    methods = list(per_seed)
    ref = reference if reference in methods else methods[0]

    # Align methods by SEED for the paired/blocked tests (not by list position): if every
    # record carries a "seed", restrict to the common seed set in sorted order so method A's
    # i-th value and method B's i-th value are the SAME seed. Else fall back to positional.
    ordered = _align_by_seed(per_seed, methods)

    report = {"reference": ref, "methods": methods, "metrics": {}}
    for metric in metrics:
        mv = {m: [o[metric] for o in ordered[m]] for m in methods
              if ordered[m] and metric in ordered[m][0]}
        if not mv:
            continue
        per_method = {m: summarize(v) for m, v in mv.items()}
        comparisons = {}
        for m in mv:
            if m == ref:
                continue
            comparisons[m] = {"paired_t_vs_ref": paired_ttest(mv[m], mv[ref]),
                              "wilcoxon_vs_ref": wilcoxon_test(mv[m], mv[ref])}
        report["metrics"][metric] = {
            "direction": directions.get(metric, 1),
            "summary": per_method,
            "vs_reference": comparisons,
            "friedman": friedman(mv),
            "nemenyi": nemenyi(mv, direction=directions.get(metric, 1)),
        }
    return report


def _has_metric(per_seed: dict, metric: str) -> bool:
    return any(objs and metric in objs[0] for objs in per_seed.values())


def _align_by_seed(per_seed: dict, methods: list) -> dict:
    """Return per-method record lists aligned by the common seed set (sorted). Falls back
    to positional truncation if any method's records lack a 'seed' key."""
    seeded = all(per_seed[m] and all("seed" in o for o in per_seed[m]) for m in methods)
    if seeded:
        common = sorted(set.intersection(*[{o["seed"] for o in per_seed[m]} for m in methods]))
        return {m: [next(o for o in per_seed[m] if o["seed"] == s) for s in common] for m in methods}
    n = min((len(per_seed[m]) for m in methods), default=0)
    return {m: per_seed[m][:n] for m in methods}


def format_report(report: dict) -> str:
    """Render the report as a readable text block."""
    ref = report["reference"]
    out = [f"=== Statistical report (Test E) — reference method: {ref} ==="]
    for metric, R in report["metrics"].items():
        arrow = "higher better" if R["direction"] > 0 else "lower better"
        out.append(f"\n[{metric}]  ({arrow})")
        out.append(f"  {'method':>15} | {'mean':>9} | {'95% CI':>21} | {'p(t) vs ref':>11} | {'p(W) vs ref':>11}")
        for m, s in R["summary"].items():
            cmp = R["vs_reference"].get(m)
            pt = f"{cmp['paired_t_vs_ref']['p']:.4f}" if cmp else "  (ref)"
            pw = f"{cmp['wilcoxon_vs_ref']['p']:.4f}" if cmp else "  (ref)"
            out.append(f"  {m:>15} | {s['mean']:9.4f} | [{s['ci_low']:8.4f}, {s['ci_high']:8.4f}] | "
                       f"{pt:>11} | {pw:>11}")
        fr = R["friedman"]
        out.append(f"  Friedman: chi2={fr.get('chi2', float('nan')):.3f} p={fr.get('p', float('nan')):.4f}"
                   + (f"  ({fr['note']})" if "note" in fr else ""))
        nm = R["nemenyi"]
        if not math.isnan(nm.get("critical_difference", float("nan"))):
            ranks = ", ".join(f"{m}:{r:.2f}" for m, r in sorted(nm["ranks"].items(), key=lambda kv: kv[1]))
            out.append(f"  Nemenyi CD={nm['critical_difference']:.3f} (alpha={nm['alpha']}); avg ranks: {ranks}")
    out.append("\nNote: p<0.05 indicates a statistically significant difference vs the reference. "
               "Power depends on seed count (campaign target >=5).")
    return "\n".join(out)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m scout_fl.analysis.stats <run_dir> [reference_method]")
        raise SystemExit(2)
    run_dir = Path(sys.argv[1])
    reference = sys.argv[2] if len(sys.argv) > 2 else "jedi"
    per_seed = json.loads((run_dir / "per_seed.json").read_text())
    report = statistical_report(per_seed, reference=reference)
    (run_dir / "stats_report.json").write_text(json.dumps(report, indent=2))
    print(format_report(report))
    print(f"\nsaved: {run_dir / 'stats_report.json'}")


if __name__ == "__main__":
    main()

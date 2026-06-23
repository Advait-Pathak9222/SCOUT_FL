"""Paper plots straight from the resumable per-round store (runs/).

For every (tag, point) it draws, seed-averaged:
  * convergence — test accuracy vs round (one line per method);
  * sensing convergence — CRB vs round;
  * energy-per-accuracy — final accuracy vs cumulative energy scatter (the headline
    efficiency plot: up-and-left is better).

Saved under ``runs/<tag>/plots/<point>/``.

CLI:
  python -m scout_fl.analysis.plots                 # every tag/point under runs/
  python -m scout_fl.analysis.plots --tag campaign_main
  python -m scout_fl.analysis.plots --tag ablation --max-methods 12
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from scout_fl.analysis.collect import _iter_units

# methods to always show even when truncating a crowded plot
_HIGHLIGHT = ("jedi", "jedi_twin", "scout_v2", "scout_greedy", "asaad", "collabsensefed", "random")


def _grouped(runs_root: Path, tag):
    """{(tag, point): {method: {round: [values-per-seed dicts]}}}"""
    g = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rel_tag, point, d in _iter_units(runs_root, tag):
        method = d.get("meta", {}).get("method")
        for r in d.get("rounds", []):
            g[(rel_tag, point)][method][int(r["round"])].append(r)
    return g


def _series(method_rounds, key):
    rounds = sorted(method_rounds)
    ys = [float(np.mean([r[key] for r in method_rounds[t] if r.get(key) is not None]))
          for t in rounds]
    return rounds, ys


def _order_methods(methods, max_methods):
    ordered = [m for m in _HIGHLIGHT if m in methods] + \
              [m for m in sorted(methods) if m not in _HIGHLIGHT]
    return ordered[:max_methods] if max_methods else ordered


def make_plots(runs_root="runs", tag=None, max_methods=0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[plots] matplotlib unavailable; skipping")
        return
    grouped = _grouped(Path(runs_root), tag)
    n = 0
    for (rel_tag, point), per_method in grouped.items():
        methods = _order_methods(list(per_method), max_methods)
        out_dir = Path(runs_root) / rel_tag / "plots" / point
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.2))
        for m in methods:
            rr, acc = _series(per_method[m], "test_acc")
            ax1.plot(rr, acc, label=m, lw=1.6 if m in _HIGHLIGHT else 0.9)
            rc, crb = _series(per_method[m], "crb")
            ax2.plot(rc, crb, label=m, lw=1.6 if m in _HIGHLIGHT else 0.9)
            # cumulative energy vs final accuracy
            re, en = _series(per_method[m], "energy")
            ax3.scatter(np.sum(en), acc[-1] if acc else 0.0,
                        s=70 if m in _HIGHLIGHT else 30)
            ax3.annotate(m, (np.sum(en), acc[-1] if acc else 0.0), fontsize=6)
        ax1.set_xlabel("round"); ax1.set_ylabel("test accuracy"); ax1.set_title(f"Convergence — {point}")
        ax1.legend(fontsize=5, ncol=2)
        ax2.set_xlabel("round"); ax2.set_ylabel("CRB (lower better)"); ax2.set_title("Sensing convergence")
        ax3.set_xlabel("cumulative energy"); ax3.set_ylabel("final accuracy")
        ax3.set_title("Energy-per-accuracy (up-left better)")
        fig.tight_layout()
        fig.savefig(out_dir / "convergence_energy.png", dpi=140)
        plt.close(fig)
        n += 1
    print(f"[plots] wrote {n} figure(s) under {runs_root}/<tag>/plots/")


def main():
    p = argparse.ArgumentParser(description="Plots from the runs/ per-round store")
    p.add_argument("runs_root", nargs="?", default="runs")
    p.add_argument("--tag", default=None)
    p.add_argument("--max-methods", type=int, default=0, help="cap methods per plot (0 = all)")
    args = p.parse_args()
    make_plots(args.runs_root, args.tag, args.max_methods)


if __name__ == "__main__":
    main()

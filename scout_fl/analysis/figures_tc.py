"""Figures for the TEMPO-FL / CloakFL program, in the make_figures.py house style
(two-column, serif, log CRB axes, CI shading), each with a JSON/CSV stat dump beside
it (design §3.3). Every figure is defensive: absent data -> skip with a note, never
a crash, so the analyze stage runs on partial results.

    python -m scout_fl.analysis.figures_tc [--smoke]
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "axes.titleweight": "bold", "legend.fontsize": 7.8,
        "figure.dpi": 150, "savefig.dpi": 320, "axes.grid": True, "grid.alpha": 0.28,
        "font.family": "serif", "mathtext.fontset": "cm", "pdf.fonttype": 42,
    })
    return plt


def _save(plt, fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def _dump_csv(path, rows, cols):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- TEMPO frontier
def fig_tempo_frontier(fig_dir, runs_root):
    from scout_fl.analysis.tc_load import load_units
    rows = [u for u in load_units("tempo", runs_root)
            if (u["point"] or "").startswith(("ET4_", "ET1_")) and u.get("time_avg_trP") is not None]
    if not rows:
        print("  [tempo_frontier] no units; skip"); return
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    agg = {}
    for r in rows:
        kind = "controller" if r["method"].startswith("tempo_") else (
            "oracle" if r["method"].startswith("oracle_") else "other")
        agg.setdefault((r["method"], kind), {"acc": [], "trP": []})
        agg[(r["method"], kind)]["acc"].append(r["acc"])
        agg[(r["method"], kind)]["trP"].append(r["time_avg_trP"])
    stat_rows = []
    colors = {"controller": "#c0392b", "oracle": "#1f6fb2", "other": "#9aa0a6"}
    for (m, kind), v in agg.items():
        a, t = float(np.mean(v["acc"])), float(np.mean(v["trP"]))
        ax.scatter(t, a, s=90 if kind == "controller" else 42,
                   marker="*" if kind == "controller" else "o", color=colors[kind],
                   edgecolor="black", linewidth=0.5, zorder=5 if kind == "controller" else 3)
        stat_rows.append({"method": m, "kind": kind, "acc": a, "time_avg_trP": t})
    ax.set_xlabel(r"Time-averaged tracking error  tr(P)  ($\leftarrow$ better)")
    ax.set_ylabel(r"Test accuracy  ($\uparrow$ better)")
    ax.set_title("TEMPO trajectory frontier vs static cloud")
    _save(plt, fig, fig_dir, "tempo_frontier")
    _dump_csv(os.path.join(fig_dir, "tempo_frontier.csv"), stat_rows,
              ["method", "kind", "acc", "time_avg_trP"])


# ---------------------------------------------------------------- E-T3 regime map
def fig_mobility_regime(fig_dir, runs_root):
    from scout_fl.analysis.tc_load import load_units
    rows = [u for u in load_units("tempo", runs_root) if (u["point"] or "").startswith("ET3_")]
    if not rows:
        print("  [regime_map] no E-T3 units; skip"); return
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    by = {}
    for r in rows:
        by.setdefault(r["method"], {}).setdefault(r["sigma_p"], []).append(r["acc"])
    stat_rows = []
    for m, d in by.items():
        xs = sorted(d)
        ys = [float(np.mean(d[x])) for x in xs]
        ax.plot(xs, ys, marker="o", label=m)
        for x, y in zip(xs, ys):
            stat_rows.append({"method": m, "sigma_p": x, "acc": y})
    ax.set_xlabel(r"target process noise $\sigma_p$"); ax.set_ylabel("test accuracy")
    ax.set_title("E-T3 mobility regime map"); ax.legend(fontsize=7)
    _save(plt, fig, fig_dir, "mobility_regime")
    _dump_csv(os.path.join(fig_dir, "mobility_regime.csv"), stat_rows, ["method", "sigma_p", "acc"])


# ---------------------------------------------------------------- E-C3 privacy frontier
def fig_privacy_frontier(fig_dir, runs_root):
    from scout_fl.analysis.tc_load import load_units
    rows = [u for u in load_units("cloak", runs_root) if (u["point"] or "").startswith("EC3_")]
    if not rows:
        print("  [privacy_frontier] no E-C3 units; skip"); return
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    by = {}
    for r in rows:
        by.setdefault(r["cloak_mode"], {}).setdefault(r["r_floor"], []).append(r)
    stat_rows = []
    for mode, d in sorted(by.items()):
        rfs = sorted(d)
        rworst = [float(np.mean([x["leak_r_min"] for x in d[rf]])) for rf in rfs]
        accs = [float(np.mean([x["acc"] for x in d[rf]])) for rf in rfs]
        ax.plot(rworst, accs, marker="o", label=mode)
        for rf, rw, ac in zip(rfs, rworst, accs):
            stat_rows.append({"mode": mode, "r_floor": rf, "worst_client_r": rw, "acc": ac})
    ax.set_xscale("log")
    ax.set_xlabel(r"worst-client CRB floor $r$ (m, log; $\rightarrow$ more private)")
    ax.set_ylabel("test accuracy")
    ax.set_title("E-C3 privacy–utility frontier"); ax.legend(fontsize=7)
    _save(plt, fig, fig_dir, "privacy_frontier")
    _dump_csv(os.path.join(fig_dir, "privacy_frontier.csv"), stat_rows,
              ["mode", "r_floor", "worst_client_r", "acc"])


# ---------------------------------------------------------------- E-C4 leakage curves
def fig_ec4_leakage(fig_dir, analytic_dir):
    src = Path(analytic_dir) / "cloak/ec4/ec4_leakage_curves.csv"
    if not src.exists():
        print("  [ec4_leakage] no E-C4 csv; skip"); return
    import pandas as pd
    df = pd.read_csv(src)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    hi = df.groupby("method")["leak_r_worst_mean"].last().sort_values()
    show = list(hi.index[:4]) + list(hi.index[-2:])            # worst + least offenders
    for m in show:
        sub = df[df.method == m].sort_values("round")
        ax.plot(sub["round"], sub["leak_r_worst_mean"], label=m)
    ax.axhline(1.0, ls="--", color="k", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("round"); ax.set_ylabel("worst-client CRB floor (m, log)")
    ax.set_title("E-C4 every method localizes its clients"); ax.legend(fontsize=7)
    _save(plt, fig, fig_dir, "ec4_leakage_curves")


# ---------------------------------------------------------------- E-C2 dither curves
def fig_dither(fig_dir, analytic_dir):
    import pandas as pd
    a = Path(analytic_dir) / "cloak/ec2/ec2a_sync_invariance.csv"
    b = Path(analytic_dir) / "cloak/ec2/ec2b_eavesdropper_crb.csv"
    if not (a.exists() and b.exists()):
        print("  [dither] no E-C2 csv; skip"); return
    plt = _mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    da = pd.read_csv(a)
    ax1.plot(da["sigma_sync"], da["eps_agg_mean"], marker="o", color="#c0392b")
    ax1.set_xlabel(r"sync error $\sigma_{sync}$"); ax1.set_ylabel(r"aggregate residual $\epsilon_{agg}$")
    ax1.set_title("M2 invariance vs sync error")
    db = pd.read_csv(b)
    ax2.plot(db["sigma_d2"], db["inflation_1rx"], marker="o", label="1 receiver")
    ax2.plot(db["sigma_d2"], db["inflation_3rx"], marker="s", label="3 receivers")
    ax2.axhline(10.0, ls="--", color="k", alpha=0.5)
    ax2.set_xlabel(r"dither variance $\sigma_d^2$"); ax2.set_ylabel("eavesdropper CRB inflation")
    ax2.set_title("M2 eavesdropper protection"); ax2.legend(fontsize=7)
    _save(plt, fig, fig_dir, "dither_validation")


# ---------------------------------------------------------------- E-T6 regret curve
def fig_regret(fig_dir, runs_root):
    """E-T6 empirical regret: TEMPO-DPP vs the best schedule in hindsight (design §1.5).

    The hindsight oracle is the reference policy (static + oracle schedules run at the
    same 300-round configuration) with the highest per-round mean accuracy envelope;
    regret_t = cumulative sum of (best-reference acc_t - dpp acc_t), per ET6 point.
    """
    from scout_fl.analysis.tc_load import load_rounds, load_units
    points = sorted({u["point"] for u in load_units("tempo", runs_root)
                     if (u["point"] or "").startswith("ET6_")})
    if not points:
        print("  [regret] no E-T6 units; skip"); return
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    stat_rows = []
    for point in points:
        rows = load_rounds("tempo", point, runs_root)
        if not rows:
            continue
        by = {}
        for r in rows:
            by.setdefault(r["method"], {}).setdefault(int(r["round"]), []).append(r["test_acc"])
        if "tempo_dpp" not in by:
            continue
        T = max(max(d) for d in by.values()) + 1
        curves = {m: np.array([float(np.mean(d.get(t, [np.nan]))) for t in range(T)])
                  for m, d in by.items()}
        refs = [m for m in curves if m != "tempo_dpp"]
        if not refs:
            continue
        best_ref = np.nanmax(np.stack([curves[m] for m in refs]), axis=0)
        regret = np.nancumsum(np.clip(best_ref - curves["tempo_dpp"], a_min=None, a_max=None))
        ax.plot(range(T), regret, label=point.replace("ET6_", ""))
        for t in range(0, T, max(1, T // 60)):
            stat_rows.append({"point": point, "round": t, "cum_regret": float(regret[t])})
    ax.set_xlabel("round"); ax.set_ylabel("cumulative regret vs best-in-hindsight (acc)")
    ax.set_title("E-T6 TEMPO-DPP empirical regret"); ax.legend(fontsize=7)
    _save(plt, fig, fig_dir, "et6_regret")
    _dump_csv(os.path.join(fig_dir, "et6_regret.csv"), stat_rows, ["point", "round", "cum_regret"])


# ---------------------------------------------------------------- E-T2 gradient decay
def fig_grad_decay(fig_dir, analytic_dir):
    src = Path(analytic_dir) / "tempo/et2/et2_grad_decay.csv"
    if not src.exists():
        print("  [grad_decay] no E-T2 csv; skip"); return
    import pandas as pd
    df = pd.read_csv(src)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for a, sub in df.groupby("alpha"):
        ax.plot(sub["round"], sub["grad_sq_mean"], label=rf"$\alpha$={a}")
    ax.set_yscale("log")
    ax.set_xlabel("round"); ax.set_ylabel(r"$\|\Delta\|^2$ (learning energy $L_t$, log)")
    ax.set_title("E-T2 learning-energy decay (T1 premise)"); ax.legend(fontsize=8)
    _save(plt, fig, fig_dir, "grad_decay")


def generate_all(fig_dir=None, runs_root=None, analytic_dir=None):
    fig_dir = fig_dir or str(REPO / "outputs/tempo_cloak/figures")
    runs_root = runs_root or str(REPO / "runs")
    analytic_dir = analytic_dir or str(REPO / "outputs/tempo_cloak/analytic")
    print(f"[figures] -> {fig_dir}")
    for fn, args in [(fig_tempo_frontier, (fig_dir, runs_root)),
                     (fig_mobility_regime, (fig_dir, runs_root)),
                     (fig_privacy_frontier, (fig_dir, runs_root)),
                     (fig_regret, (fig_dir, runs_root)),
                     (fig_ec4_leakage, (fig_dir, analytic_dir)),
                     (fig_dither, (fig_dir, analytic_dir)),
                     (fig_grad_decay, (fig_dir, analytic_dir))]:
        try:
            fn(*args)
        except Exception as e:                                # noqa: BLE001 - figures never abort analyze
            print(f"  [warn] {fn.__name__} failed: {e}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-dir", default=None)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--analytic-dir", default=None)
    args = ap.parse_args()
    generate_all(args.fig_dir, args.runs_root, args.analytic_dir)


if __name__ == "__main__":
    main()

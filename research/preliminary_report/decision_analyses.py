"""Decision-critical analyses for the SCOUT-FL / JEDI(VISMAYA)-FL publish decision.

Weight-free, honest evidence computed DIRECTLY from runs/ artifacts (never hardcoded
from the report). Reuses report_common.py loaders + the report's matplotlib style.

Per the schema check:
  * accuracy = objectives.acc  (final-round global test acc)
  * CRB      = objectives.crb (round-MEAN, primary)  AND  objectives.crb_final (final, secondary)
    -> every CRB analysis is produced for BOTH conventions.
  * pairing  = shared (point, seed); C_sensing_targets=5 is partial (2/5 seeds) -> reported, never imputed.

Figures -> figures/ ;  machine-readable stats -> figures/stats/ .
"""
from __future__ import annotations
import os, json, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import report_common as rc

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")
STATS = os.path.join(FIG, "stats")
os.makedirs(STATS, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5, "axes.titleweight": "bold",
    "legend.fontsize": 7.8, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.dpi": 320, "axes.grid": True, "grid.alpha": 0.28,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "font.family": "serif",
    "mathtext.fontset": "cm", "pdf.fonttype": 42, "axes.edgecolor": "#333333",
})

# ---- palette (consistent with make_figures) ----
CPROP = {"scout_v2": "#1f6fb2", "jedi": "#c0392b", "scout_greedy": "#12a5b0"}
CBASE = {"collabsensefed": "#9467bd", "sensing_native": "#8c564b"}
PROPOSED = ["scout_v2", "jedi", "scout_greedy"]
BASELINES = ["collabsensefed", "sensing_native"]

# short labels + a stable ordering for the 22 operating points
PT_LABEL = {
    "A_datasets=emnist": "EMNIST", "A_datasets=fashion_mnist": "F-MNIST",
    "A_datasets=uci_har": "UCI-HAR", "A_datasets=cifar10": "CIFAR-10",
    "A_datasets=cifar100": "CIFAR-100", "A_learning_partition=iid": "IID",
    "A_learning_partition=spatial": "spatial", "A_learning_noniid=0.1": r"$\alpha$0.1",
    "A_learning_noniid=0.3": r"$\alpha$0.3", "A_learning_noniid=0.5": r"$\alpha$0.5",
    "B_wireless_channel=rayleigh": "Rayleigh", "B_wireless_channel=rician": "Rician",
    "B_wireless_snr=0": "SNR0", "B_wireless_snr=-10": "SNR-10", "B_wireless_snr=-15": "SNR-15",
    "B_wireless_snr=-20": "SNR-20", "B_wireless_snr=-25": "SNR-25", "B_wireless_snr=-30": "SNR-30",
    "B_wireless_snr=-35": "SNR-35", "C_sensing_targets=2": "M=2", "C_sensing_targets=3": "M=3",
    "C_sensing_targets=5": "M=5",
}
PT_ORDER = list(PT_LABEL.keys())


def disp(m):
    return rc.disp(m).replace(" (ours)", "")


def _load_per_seed():
    """campaign tag -> one row per (point, method, seed) with acc, crb, crb_final, logdet."""
    df = rc.load_tag("campaign")
    keep = ["point", "method", "seed", "acc", "crb", "crb_final", "logdet"]
    return df[keep].copy()


def boot_ci(x, fn=np.mean, n=10000, alpha=0.05, seed=0):
    """Bootstrap CI of a statistic over the 1-D sample x."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    if len(x) == 1:
        return (float(fn(x)), np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = fn(x[rng.integers(0, len(x), size=(n, len(x)))], axis=1)
    return float(fn(x)), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def paired_diffs(df, a, b, metric):
    """Return DataFrame of per-(point,seed) paired diffs (a-b) for `metric`, only where BOTH present."""
    pa = df[df.method == a][["point", "seed", metric]].rename(columns={metric: "a"})
    pb = df[df.method == b][["point", "seed", metric]].rename(columns={metric: "b"})
    m = pa.merge(pb, on=["point", "seed"], how="inner")
    m["d"] = m["a"] - m["b"]
    return m


def pooled_tests(d):
    """Wilcoxon signed-rank + paired t on a vector of paired diffs; effect sizes."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    out = {"n": int(len(d)), "mean": float(np.mean(d)) if len(d) else np.nan,
           "median": float(np.median(d)) if len(d) else np.nan}
    mu, lo, hi = boot_ci(d, np.mean)
    out["mean_ci95"] = [lo, hi]
    md, mlo, mhi = boot_ci(d, np.median)
    out["median_ci95"] = [mlo, mhi]
    if len(d) >= 2 and np.any(d != 0):
        try:
            w, pw = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            out["wilcoxon_p"] = float(pw)
            # rank-biserial effect size
            n = len(d)
            out["rank_biserial"] = float(1 - 2 * w / (n * (n + 1) / 2))
        except Exception as e:
            out["wilcoxon_p"] = None
        t, pt = stats.ttest_1samp(d, 0.0)
        out["t_p"] = float(pt)
        out["cohen_dz"] = float(np.mean(d) / np.std(d, ddof=1)) if np.std(d, ddof=1) > 0 else np.inf
    else:
        out["wilcoxon_p"] = out["t_p"] = out["cohen_dz"] = out["rank_biserial"] = None
    # verdict: significant if BOTH tests agree at 0.05 AND CI excludes 0
    sig = (out.get("wilcoxon_p") is not None and out["wilcoxon_p"] < 0.05
           and out.get("t_p") is not None and out["t_p"] < 0.05
           and not (lo <= 0 <= hi))
    out["significant_0.05"] = bool(sig)
    return out


def per_point_ci(m, alpha=0.05):
    """Per operating point: mean diff + t-CI over that point's shared seeds.
    Returns dict point-> (mean, lo, hi, n). CI crossing 0 => tie."""
    res = {}
    for pt, g in m.groupby("point"):
        d = g["d"].values
        n = len(d)
        mu = float(np.mean(d))
        if n >= 2:
            se = np.std(d, ddof=1) / np.sqrt(n)
            h = stats.t.ppf(1 - alpha / 2, n - 1) * se
            lo, hi = mu - h, mu + h
        else:
            lo = hi = np.nan
        res[pt] = (mu, lo, hi, n)
    return res


# ===================================================================== ANALYSIS 1
def a1_head_to_head():
    df = _load_per_seed()
    present_pts = [p for p in PT_ORDER if p in set(df.point)]
    dump = {"_meta": {"proposed": PROPOSED, "baselines": BASELINES,
                      "accuracy": "objectives.acc (final-round)",
                      "crb_conventions": {"round_mean": "objectives.crb",
                                          "final_round": "objectives.crb_final"},
                      "pairing": "shared (point, seed); inner-join",
                      "partial_points": {}}}
    # note partial points
    for pt in present_pts:
        ns = df[(df.point == pt) & (df.method.isin(PROPOSED + BASELINES))].groupby("method").seed.nunique()
        if (ns < 5).any():
            dump["_meta"]["partial_points"][pt] = {m: int(ns.get(m, 0)) for m in PROPOSED + BASELINES}

    metrics = {"acc": ("acc", "$\\Delta$Accuracy  (proposed $-$ baseline)", 1),
               "crb_roundmean": ("crb", "$\\Delta$CRB round-mean  ($-$ = better)", -1),
               "crb_final": ("crb_final", "$\\Delta$CRB final  ($-$ = better)", -1)}

    # ---- stats for every (proposed, baseline, metric) ----
    for a, b in itertools.product(PROPOSED, BASELINES):
        for mk, (col, _, _) in metrics.items():
            m = paired_diffs(df, a, b, col)
            pooled = pooled_tests(m["d"].values)
            # per-point-mean as the more-defensible replication unit (points as "datasets")
            ppmeans = m.groupby("point")["d"].mean().values
            pp = pooled_tests(ppmeans)
            dump[f"{a}__vs__{b}__{mk}"] = {
                "pooled_point_seed": pooled,
                "per_point_mean_unit": {"n_points": int(len(ppmeans)), **{k: pp[k] for k in
                    ("mean", "mean_ci95", "wilcoxon_p", "t_p", "cohen_dz", "significant_0.05")}},
            }
    with open(os.path.join(STATS, "a1_head_to_head.json"), "w") as f:
        json.dump(dump, f, indent=2)

    # ---- figure: one per baseline, 3 panels (Δacc, ΔCRB round-mean, ΔCRB final) ----
    for b in BASELINES:
        fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.0), sharex=True)
        x = np.arange(len(present_pts))
        for ax, (mk, (col, ylab, _)) in zip(axes, metrics.items()):
            for j, a in enumerate(PROPOSED):
                m = paired_diffs(df, a, b, col)
                pp = per_point_ci(m)
                xs, ys, los, his, tie = [], [], [], [], []
                for i, pt in enumerate(present_pts):
                    if pt not in pp:
                        continue
                    mu, lo, hi, n = pp[pt]
                    xs.append(i + (j - 1) * 0.24); ys.append(mu)
                    los.append(mu - lo if not np.isnan(lo) else 0)
                    his.append(hi - mu if not np.isnan(hi) else 0)
                    tie.append(bool(np.isnan(lo) or (lo <= 0 <= hi)))
                xs = np.array(xs); ys = np.array(ys); tie = np.array(tie)
                c = CPROP[a]
                # ties: hollow grey; wins/losses: filled colour
                ax.errorbar(xs[~tie], ys[~tie], yerr=[np.array(los)[~tie], np.array(his)[~tie]],
                            fmt="o", ms=4.5, color=c, ecolor=c, elinewidth=1.0, capsize=2,
                            label=disp(a), zorder=4)
                if tie.any():
                    ax.errorbar(xs[tie], ys[tie], yerr=[np.array(los)[tie], np.array(his)[tie]],
                                fmt="o", ms=4.5, mfc="white", mec=c, ecolor="0.7",
                                elinewidth=0.8, capsize=2, zorder=3)
            ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.7)
            ax.set_ylabel(ylab)
        axes[0].legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.95)
        axes[0].set_title(f"Paired per-point $\\Delta$ vs. {disp(b)}  "
                          f"(mean $\\pm$95% CI over shared seeds; hollow = tie, CI crosses 0)")
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([PT_LABEL[p] for p in present_pts], rotation=90)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIG, f"figA1_h2h_{b}.{ext}"), bbox_inches="tight")
        plt.close(fig)
        print(f"wrote figA1_h2h_{b}")

    # ---- compact console summary ----
    print("\n=== A1 pooled paired summary (per-point-mean unit, n=points) ===")
    for a in PROPOSED:
        for b in BASELINES:
            acc = dump[f"{a}__vs__{b}__acc"]["per_point_mean_unit"]
            crb = dump[f"{a}__vs__{b}__crb_roundmean"]["per_point_mean_unit"]
            print(f" {disp(a):16s} vs {disp(b):16s} | "
                  f"Δacc={acc['mean']*100:+.2f}pp CI[{acc['mean_ci95'][0]*100:+.2f},{acc['mean_ci95'][1]*100:+.2f}] "
                  f"sig={acc['significant_0.05']} | "
                  f"Δcrb={crb['mean']:+.3f} CI[{crb['mean_ci95'][0]:+.3f},{crb['mean_ci95'][1]:+.3f}] sig={crb['significant_0.05']}")
    return dump


# ===================================================================== ANALYSIS 2
def a2_threshold_sensitivity():
    """Constrained-accuracy wins vs. the sensing bar, swept 0.05..0.20, for BOTH CRB
    conventions. Winner at a point = eligible (crb<=tau) method with max mean accuracy."""
    df = rc.load_tag("campaign")
    df["group"] = df.method.map(rc.group)
    df = df[df.group != "Ablation"]
    s = rc.agg_over_seeds(df, metrics=["acc", "crb", "crb_final"])
    present_pts = sorted(set(s.point))
    thresholds = np.round(np.arange(0.05, 0.2001, 0.005), 3)
    show = ["jedi", "scout_v2", "scout_greedy", "collabsensefed", "sensing_native",
            "asaad", "random", "divfl", "crb_only", "sensing_only"]

    dump = {"_meta": {"thresholds": thresholds.tolist(), "n_points": len(present_pts),
                      "eligible_set": "all 28 non-ablation methods",
                      "winner": "max mean-accuracy among methods with crb<=tau",
                      "report_threshold_0.10": True}}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for ax, (crbcol, title) in zip(axes, [("crb_mean", "round-mean CRB"),
                                          ("crb_final_mean", "final-round CRB")]):
        wins = {m: np.zeros(len(thresholds), int) for m in s.method.unique()}
        n_eligible_pts = np.zeros(len(thresholds), int)   # points with >=1 eligible method
        for ti, tau in enumerate(thresholds):
            for pt in present_pts:
                sub = s[s.point == pt]
                elig = sub[sub[crbcol] <= tau]
                if len(elig) == 0:
                    continue
                n_eligible_pts[ti] += 1
                w = elig.sort_values("acc_mean", ascending=False).iloc[0].method
                wins[w][ti] += 1
        dump[crbcol] = {m: wins[m].tolist() for m in show}
        dump[crbcol]["_proposed_total"] = (wins["jedi"] + wins["scout_v2"] + wins["scout_greedy"]).tolist()
        dump[crbcol]["_points_with_eligible"] = n_eligible_pts.tolist()
        for m in show:
            if wins[m].sum() == 0:
                continue
            prop = rc.is_proposed(m)
            c = CPROP.get(m, CBASE.get(m, None))
            if c is None:
                c = {"asaad": "#e67e22", "random": "#7f7f7f", "divfl": "#2ca02c",
                     "crb_only": "#bcbd22", "sensing_only": "#17becf"}.get(m, "#aaaaaa")
            ax.plot(thresholds, wins[m], "-o" if prop else "--", color=c, lw=2.4 if prop else 1.3,
                    ms=5 if prop else 3, label=disp(m), zorder=5 if prop else 3, alpha=1 if prop else 0.85)
        ax.axvline(0.10, color="black", ls=":", lw=1.2, alpha=0.7)
        ax.text(0.10, ax.get_ylim()[1] * 0.96, " report's 0.10", fontsize=7, rotation=90,
                va="top", ha="left", color="0.3")
        ax.set_xlabel("sensing bar  CRB $\\leq \\tau$"); ax.set_title(f"({title})")
    axes[0].set_ylabel("# constrained-accuracy wins (of 22 points)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.4)
    fig.suptitle("CRB-threshold sensitivity of the constrained-accuracy win count",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"figA2_threshold_sensitivity.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote figA2_threshold_sensitivity")
    with open(os.path.join(STATS, "a2_threshold_sensitivity.json"), "w") as f:
        json.dump(dump, f, indent=2)

    # console: proposed-total win stability across the band
    print("\n=== A2 constrained-acc wins across threshold band (round-mean CRB) ===")
    rm = dump["crb_mean"]
    for m in ["jedi", "scout_v2", "scout_greedy", "_proposed_total"]:
        arr = np.array(rm[m])
        print(f"  {m:18s} min={arr.min()} max={arr.max()} @0.10={arr[list(thresholds).index(0.10)]} "
              f"range=[{thresholds[arr>0].min() if (arr>0).any() else '--'}..]")
    return dump


# ===================================================================== ANALYSIS 3
def a3_dominator_counts():
    """Weight-free: per point, # methods that STRICTLY dominate each method
    (strictly higher accuracy AND strictly lower CRB). 0 => Pareto-optimal there.
    Produced for BOTH CRB conventions."""
    df = rc.load_tag("campaign")
    df["group"] = df.method.map(rc.group)
    df = df[df.group != "Ablation"]
    s = rc.agg_over_seeds(df, metrics=["acc", "crb", "crb_final"])
    present_pts = [p for p in PT_ORDER if p in set(s.point)]
    rows = ["jedi", "scout_v2", "scout_greedy", "collabsensefed", "sensing_native",
            "asaad", "divfl", "random"]

    def dom_matrix(crbcol):
        M = np.full((len(rows), len(present_pts)), np.nan)
        for j, pt in enumerate(present_pts):
            sub = s[s.point == pt]
            acc = dict(zip(sub.method, sub.acc_mean)); crb = dict(zip(sub.method, sub[crbcol]))
            for i, m in enumerate(rows):
                if m not in acc:
                    continue
                am, cm = acc[m], crb[m]
                dom = sum(1 for o in sub.method if o != m and acc[o] > am and crb[o] < cm)
                M[i, j] = dom
        return M

    dump = {"_meta": {"rows": rows, "points": present_pts,
                      "rule": "strict: acc_other>acc_m AND crb_other<crb_m",
                      "eligible_set": "28 non-ablation methods",
                      "0_means": "Pareto-optimal at that point",
                      "partial_points": {"C_sensing_targets=5": "2 of 5 seeds (seed-mean over available)"}}}
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8))
    for axi, (ax, (crbcol, title)) in enumerate(zip(axes, [("crb_mean", "round-mean CRB"),
                                                           ("crb_final_mean", "final-round CRB")])):
        M = dom_matrix(crbcol)
        med = np.nanmedian(M, axis=1)
        order = np.argsort(med)
        Mo = M[order]; rlabels = [rows[i] for i in order]; medo = med[order]
        dump[crbcol] = {"per_point": {rows[i]: [None if np.isnan(v) else int(v) for v in M[i]] for i in range(len(rows))},
                        "median_dominators": {rows[i]: (None if np.isnan(med[i]) else float(med[i])) for i in range(len(rows))},
                        "mean_dominators": {rows[i]: (None if np.all(np.isnan(M[i])) else float(np.nanmean(M[i]))) for i in range(len(rows))},
                        "frac_pareto_optimal": {rows[i]: float(np.nanmean(M[i] == 0)) for i in range(len(rows))}}
        vmax = max(1, int(np.nanmax(M)))
        im = ax.imshow(Mo, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(present_pts)))
        if axi == len(axes) - 1:
            ax.set_xticklabels([PT_LABEL[p] for p in present_pts], rotation=90, fontsize=6.4)
        else:
            ax.set_xticklabels([])
        ax.set_yticks(range(len(rlabels)))
        ax.set_yticklabels([f"{disp(m)}  (md {medo[i]:.0f})" for i, m in enumerate(rlabels)], fontsize=7.6)
        for i in range(len(rlabels)):
            if rc.is_proposed(rlabels[i]):
                ax.get_yticklabels()[i].set_fontweight("bold")
            for j in range(len(present_pts)):
                if not np.isnan(Mo[i, j]):
                    ax.text(j, i, int(Mo[i, j]), ha="center", va="center", fontsize=5.6,
                            color="white" if Mo[i, j] >= vmax * 0.6 or Mo[i, j] == 0 else "black")
        ax.set_title(f"# strict dominators ({title}); 0 = Pareto-optimal", fontsize=9.5)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02).set_label("# dominators", fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"figA3_dominators.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote figA3_dominators")
    with open(os.path.join(STATS, "a3_dominators.json"), "w") as f:
        json.dump(dump, f, indent=2)
    print("\n=== A3 median strict-dominators per method (0 = Pareto-optimal) ===")
    for conv in ["crb_mean", "crb_final_mean"]:
        md = dump[conv]["median_dominators"]; fp = dump[conv]["frac_pareto_optimal"]
        print(f"  [{conv}]")
        for m in rows:
            print(f"     {disp(m):22s} median={md[m]}  mean={dump[conv]['mean_dominators'][m]:.2f}  "
                  f"Pareto-optimal in {fp[m]*100:.0f}% of points")
    return dump


# ===================================================================== ANALYSIS 4
def a4_mobility():
    """Generative-synergy under mobility. A sigma_p SWEEP does NOT exist -> report the gap
    and the command to generate it, and SKIP the sweep plot. Provide the single available
    mobility point (sigma_p=0.05, tag ablation_vismaya) as horizon Delta (full vs synergy-off)."""
    gap = {
        "requested": "Delta accuracy / Delta CRB (full JEDI/VISMAYA vs synergy-off) vs. mobility sigma_p",
        "status": "SWEEP MISSING - only a single mobility value exists",
        "available": {"tag": "ablation_vismaya", "sigma_p": 0.05, "rounds": 50, "seeds": 5,
                      "full": "vismaya", "synergy_off": "vismaya_no_syn"},
        "missing": "no runs at sigma_p in {0.0, 0.02, 0.10, 0.20}; referenced "
                   "configs/sweep_mobility_ablation.yaml is NOT in the repo",
        "generate_command": [
            "# create a sigma_p sweep by overriding vismaya.process_noise:",
            "for pn in 0.0 0.02 0.05 0.10 0.20; do",
            "  python -m scout_fl.experiments.run_fl_synthetic \\",
            "    --config scout_fl/configs/ablation_vismaya.yaml \\",
            "    --override vismaya.process_noise=$pn experiment=mobility_pn$pn \\",
            "    'selection.methods=[vismaya,vismaya_no_syn]' 'seeds=[0,1,2,3,4]'",
            "done",
            "python -m scout_fl.analysis.collect  # then re-run a4_mobility once the sweep exists"],
    }
    print("\n=== A4 MOBILITY GAP (no sigma_p sweep; sweep plot SKIPPED) ===")
    for k in ("status", "missing"):
        print(f"  {k}: {gap[k]}")
    print("  generate with:\n    " + "\n    ".join(gap["generate_command"]))

    # ---- available single-point evidence: horizon Delta at sigma_p=0.05 ----
    rr = rc.load_rounds("ablation_vismaya", "base")
    full, off = "vismaya", "vismaya_no_syn"
    have = set(rr.method.unique())
    if not ({full, off} <= have):
        gap["single_point_plot"] = "skipped (methods missing)"
        with open(os.path.join(STATS, "a4_mobility.json"), "w") as f:
            json.dump(gap, f, indent=2)
        return gap

    def paired_round(metric):
        a = rr[rr.method == full][["seed", "round", metric]].rename(columns={metric: "f"})
        b = rr[rr.method == off][["seed", "round", metric]].rename(columns={metric: "o"})
        m = a.merge(b, on=["seed", "round"]); m["d"] = m["f"] - m["o"]
        g = m.groupby("round")["d"]
        mu = g.mean(); n = g.count()
        sd = g.std(ddof=1)
        h = stats.t.ppf(0.975, np.clip(n - 1, 1, None)) * sd / np.sqrt(n)
        return mu.index.values, mu.values, (mu - h).values, (mu + h).values

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.7))
    for ax, (metric, ylab, better) in zip(
            [a1, a2], [("test_acc", "$\\Delta$Accuracy (full $-$ synergy-off)", "up"),
                       ("crb", "$\\Delta$CRB (full $-$ synergy-off)", "down")]):
        x, mu, lo, hi = paired_round(metric)
        ax.plot(x, mu, color="#c0392b", lw=2.0)
        ax.fill_between(x, lo, hi, color="#c0392b", alpha=0.18, label="95% CI (5 seeds)")
        ax.axhline(0, color="black", ls="--", lw=1.0, alpha=0.7)
        ax.set_xlabel("round"); ax.set_ylabel(ylab)
        good = "below 0 = synergy helps" if better == "down" else "above 0 = synergy helps"
        ax.set_title(f"({good})", fontsize=9)
        if metric == "crb":
            ax.set_yscale("symlog", linthresh=0.1)
        ax.legend(fontsize=7.5)
    fig.suptitle("Generative-synergy effect at the ONLY available mobility point "
                 "($\\sigma_p{=}0.05$; a $\\sigma_p$ sweep is MISSING)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"figA4_mobility_singlepoint.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote figA4_mobility_singlepoint (single sigma_p; NOT a sweep)")

    # horizon summary (final round) with paired test over seeds
    for metric in ("test_acc", "crb"):
        a = rr[(rr.method == full)].groupby("seed")[metric].last()
        b = rr[(rr.method == off)].groupby("seed")[metric].last()
        common = sorted(set(a.index) & set(b.index))
        d = (a.loc[common] - b.loc[common]).values
        pooled = pooled_tests(d)
        gap.setdefault("single_point_final", {})[metric] = {
            "full_mean": float(a.loc[common].mean()), "synergy_off_mean": float(b.loc[common].mean()),
            "delta": pooled["mean"], "delta_ci95": pooled["mean_ci95"],
            "wilcoxon_p": pooled["wilcoxon_p"], "significant_0.05": pooled["significant_0.05"]}
    with open(os.path.join(STATS, "a4_mobility.json"), "w") as f:
        json.dump(gap, f, indent=2)
    sp = gap["single_point_final"]
    print(f"  final@sigma_p=0.05  Δacc={sp['test_acc']['delta']:+.3f} (sig={sp['test_acc']['significant_0.05']}) "
          f"| Δcrb={sp['crb']['delta']:+.3f} (sig={sp['crb']['significant_0.05']})")
    return gap


# ===================================================================== ANALYSIS 5
def _participation(tag, method):
    """Per-client selection counts across all seeds x rounds, from the raw `selected` lists.
    (Per-client ACCURACY is not logged; participation is the available tail-fairness signal.)"""
    import glob as _g
    counts = None
    K = 100
    per_seed = []
    for f in sorted(_g.glob(os.path.join(rc.RUNS, tag, "base", f"{method}__seed*.json"))):
        d = json.load(open(f))
        if not d.get("complete"):
            continue
        K = int(d["meta"].get("K", 100))
        c = np.zeros(K)
        for r in d["rounds"]:
            for k in (r.get("selected") or []):
                if 0 <= int(k) < K:
                    c[int(k)] += 1
        per_seed.append(c)
        counts = c if counts is None else counts + c
    return counts, per_seed, K


def _fairness_metrics(counts):
    x = np.asarray(counts, float)
    s = x.sum()
    jain = float((x.sum() ** 2) / (len(x) * (x ** 2).sum())) if (x ** 2).sum() > 0 else np.nan
    # Gini
    xs = np.sort(x); n = len(x); cum = np.cumsum(xs)
    gini = float((n + 1 - 2 * (cum.sum() / cum[-1])) / n) if cum[-1] > 0 else np.nan
    never = float(np.mean(x == 0))
    return {"jain": jain, "gini": gini, "frac_never_selected": never,
            "min": float(x.min()), "max": float(x.max()), "cv": float(x.std() / (x.mean() + 1e-9))}


def a5_fairness():
    """Full objective vs w/o-fairness over the full ablation horizon (30 rounds), on
    accuracy + CRB, plus a per-client PARTICIPATION CDF (per-client accuracy is unavailable)."""
    tag, full, off = "ablation", "jedi", "jedi_no_fairness"
    rr = rc.load_rounds(tag, "base")
    have = set(rr.method.unique())
    dump = {"_meta": {"tag": tag, "horizon_rounds": int(rr["round"].max()),
                      "full": full, "no_fairness": off,
                      "note": "per-client ACCURACY not logged; participation CDF is the tail-fairness proxy"}}
    if not ({full, off} <= have):
        dump["status"] = f"missing methods; have={sorted(have)}"
        json.dump(dump, open(os.path.join(STATS, "a5_fairness.json"), "w"), indent=2)
        print("A5 skipped:", dump["status"]); return dump

    def band(method, metric):
        g = rr[rr.method == method].groupby("round")[metric]
        mu, sd, n = g.mean(), g.std(ddof=1), g.count()
        h = stats.t.ppf(0.975, np.clip(n - 1, 1, None)) * sd / np.sqrt(n)
        return mu.index.values, mu.values, (mu - h).values, (mu + h).values

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    # (a) accuracy horizon
    for method, c in [(full, "#c0392b"), (off, "#7f7f7f")]:
        x, mu, lo, hi = band(method, "test_acc")
        lab = "full (with fairness)" if method == full else "w/o fairness"
        axes[0].plot(x, mu, color=c, lw=2.0, label=lab)
        axes[0].fill_between(x, lo, hi, color=c, alpha=0.16)
    axes[0].set_xlabel("round"); axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("(a) Accuracy (mean $\\pm$95% CI)"); axes[0].legend(fontsize=7.6)
    # (b) CRB horizon
    for method, c in [(full, "#c0392b"), (off, "#7f7f7f")]:
        x, mu, lo, hi = band(method, "crb")
        axes[1].plot(x, mu, color=c, lw=2.0)
        axes[1].fill_between(x, lo, hi, color=c, alpha=0.16)
    axes[1].set_xlabel("round"); axes[1].set_ylabel("CRB ($\\downarrow$ better)")
    axes[1].set_title("(b) Sensing CRB")
    axes[1].set_yscale("log")
    # (c) per-client participation CDF
    cf, _, K = _participation(tag, full)
    co, _, _ = _participation(tag, off)
    for counts, c, method in [(cf, "#c0392b", full), (co, "#7f7f7f", off)]:
        xs = np.sort(counts); ys = np.arange(1, len(xs) + 1) / len(xs)
        fm = _fairness_metrics(counts)
        lab = ("full" if method == full else "w/o fairness") + f"  (Jain {fm['jain']:.2f}, never {fm['frac_never_selected']*100:.0f}%)"
        axes[2].step(xs, ys, where="post", color=c, lw=2.0, label=lab)
        dump[method] = {"participation_fairness": fm}
    axes[2].set_xlabel(f"per-client selections over {int(rr['round'].max())} rd $\\times$5 seeds")
    axes[2].set_ylabel("cumulative fraction of clients")
    axes[2].set_title(f"(c) Participation CDF (K={K}); flatter-left = more starvation")
    axes[2].legend(fontsize=7.2, loc="lower right")
    fig.suptitle("Fairness-term diagnosis: does the fairness term trade mean accuracy for participation equity?",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"figA5_fairness.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote figA5_fairness")

    # paired final-round deltas (full - no_fairness) over seeds
    for metric in ("test_acc", "crb"):
        a = rr[rr.method == full].groupby("seed")[metric].last()
        b = rr[rr.method == off].groupby("seed")[metric].last()
        common = sorted(set(a.index) & set(b.index))
        pooled = pooled_tests((a.loc[common] - b.loc[common]).values)
        dump.setdefault("final_delta_full_minus_nofair", {})[metric] = {
            "full": float(a.loc[common].mean()), "no_fairness": float(b.loc[common].mean()),
            "delta": pooled["mean"], "delta_ci95": pooled["mean_ci95"],
            "wilcoxon_p": pooled["wilcoxon_p"], "significant_0.05": pooled["significant_0.05"]}
    json.dump(dump, open(os.path.join(STATS, "a5_fairness.json"), "w"), indent=2)
    fd = dump["final_delta_full_minus_nofair"]
    print(f"  final Δacc(full-nofair)={fd['test_acc']['delta']:+.3f} sig={fd['test_acc']['significant_0.05']} | "
          f"Δcrb={fd['crb']['delta']:+.3f} sig={fd['crb']['significant_0.05']}")
    print(f"  participation Jain: full={dump[full]['participation_fairness']['jain']:.3f} "
          f"vs no-fair={dump[off]['participation_fairness']['jain']:.3f} ; "
          f"never-selected: full={dump[full]['participation_fairness']['frac_never_selected']*100:.0f}% "
          f"vs no-fair={dump[off]['participation_fairness']['frac_never_selected']*100:.0f}%")
    return dump


# ===================================================================== ANALYSIS 6
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
        9: 3.102, 10: 3.164}


def _avg_ranks(points):
    """Avg rank (1=best) by joint ISAC score over the given points, for a fixed 9-method set."""
    df = rc.load_tag("campaign"); df["group"] = df.method.map(rc.group)
    s = rc.agg_over_seeds(df[df.group != "Ablation"], metrics=["acc", "logdet"])
    methods = ["jedi", "scout_v2", "scout_greedy", "collabsensefed", "fedgcs",
               "divfl", "oort", "asaad", "random"]
    ranks = {m: [] for m in methods}
    used = 0
    for pt in points:
        sub = s[(s.point == pt) & (s.method.isin(methods))].copy()
        if len(sub) < len(methods):
            continue
        sub["score"] = 0.5 * rc.minmax_norm(sub.acc_mean) + 0.5 * rc.minmax_norm(sub.logdet_mean)
        sub = sub.sort_values("score", ascending=False).reset_index(drop=True)
        for i, r in sub.iterrows():
            ranks[r.method].append(i + 1)
        used += 1
    return {m: float(np.mean(v)) for m, v in ranks.items()}, used, len(methods)


def a6_corrected_cd():
    """CD diagram re-done over the 21 FULLY-SEEDED points (M=5, 2 seeds, excluded from primary),
    with the correlation caveat annotated + the rank shift vs the 22-point version."""
    all_pts = [p for p in PT_ORDER if p in set(rc.load_tag("campaign").point)]
    seeded21 = [p for p in all_pts if p != "C_sensing_targets=5"]  # exclude the 2-seed point
    avg21, N, k = _avg_ranks(seeded21)
    avg22, N22, _ = _avg_ranks(all_pts)
    CD = _Q05[k] * np.sqrt(k * (k + 1) / (6.0 * N))

    dump = {"_meta": {"ranking": "joint ISAC score (0.5 acc + 0.5 logdet)",
                      "primary_points": N, "excluded": "C_sensing_targets=5 (2 of 5 seeds)",
                      "CD": CD,
                      "caveats": [
                          "The N operating points share the SAME 5 seeds and one system model; "
                          "they are NOT independent datasets. Nemenyi assumes independence, so the "
                          "CD / p-values are ANTI-CONSERVATIVE (optimistic). Treat as illustrative.",
                          "C_sensing_targets=5 has only 2/5 seeds (large variance) and is excluded "
                          "from the primary ranking; its inclusion barely shifts ranks (see rank_shift)."]},
             "avg_rank_21pts": avg21, "avg_rank_22pts": avg22,
             "rank_shift_incl_M5": {m: round(avg22[m] - avg21[m], 3) for m in avg21}}
    json.dump(dump, open(os.path.join(STATS, "a6_corrected_cd.json"), "w"), indent=2)

    items = sorted(avg21.items(), key=lambda kv: kv[1])
    vals = [v for _, v in items]
    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    ax.set_xlim(lo - 0.5, hi + 0.5); ax.set_ylim(0.02, 1.12); ax.axis("off")
    yax = 0.60
    ax.plot([lo, hi], [yax, yax], "k-", lw=1.2)
    for x in range(lo, hi + 1):
        ax.plot([x, x], [yax, yax + 0.028], "k-", lw=1.0)
        ax.text(x, yax + 0.055, str(x), ha="center", va="bottom", fontsize=8)
    ax.text((lo + hi) / 2, yax + 0.135, "average rank  (1 = best joint ISAC score)",
            ha="center", fontsize=8.5, fontweight="bold")
    ax.text((lo + hi) / 2, 1.06, f"Corrected CD diagram  (Nemenyi, $\\alpha$=0.05, "
            f"N={N} fully-seeded points; M=5 excluded)", ha="center", fontsize=10.5, fontweight="bold")
    # CD bar
    ybar = yax + 0.24
    ax.plot([lo, lo + CD], [ybar, ybar], "k-", lw=2.2)
    for xx in (lo, lo + CD):
        ax.plot([xx, xx], [ybar - 0.02, ybar + 0.02], "k-", lw=1.1)
    ax.text(lo + CD / 2, ybar + 0.03, f"CD = {CD:.2f}", ha="center", fontsize=8.5)
    half = int(np.ceil(k / 2)); step = 0.085
    for idx, (m, v) in enumerate(items):
        nm = disp(m); prop = rc.is_proposed(m); c = CPROP.get(m, CBASE.get(m, "#8a8a8a"))
        if m == "asaad":
            c = "#e67e22"
        if idx < half:
            yy = yax - 0.09 - idx * step
            ax.plot([v, v], [yax, yy], color=c, lw=1.5); ax.plot([v, lo - 0.45], [yy, yy], color=c, lw=1.5)
            ax.text(lo - 0.5, yy, nm, ha="right", va="center", fontsize=8.2,
                    fontweight="bold" if prop else "normal", color=c if prop else "black")
        else:
            yy = yax - 0.09 - (k - 1 - idx) * step
            ax.plot([v, v], [yax, yy], color=c, lw=1.5); ax.plot([v, hi + 0.45], [yy, yy], color=c, lw=1.5)
            ax.text(hi + 0.5, yy, nm, ha="left", va="center", fontsize=8.2,
                    fontweight="bold" if prop else "normal", color=c if prop else "black")
    # cliques
    vs = sorted(vals); ybars = yax - 0.035; used = []
    i = 0
    while i < len(vs):
        j = i
        while j + 1 < len(vs) and vs[j + 1] - vs[i] < CD:
            j += 1
        if j > i and not any(vs[i] >= a and vs[j] <= b for a, b in used):
            used.append((vs[i], vs[j]))
            ax.plot([vs[i] - 0.04, vs[j] + 0.04], [ybars, ybars], color="0.2", lw=3.2, solid_capstyle="round")
            ybars -= 0.026
        i += 1
    # caveat box
    ax.text(0.5, -0.02,
            "CAVEAT: the N points share the same 5 seeds + one system model — they are NOT independent "
            "datasets.\nNemenyi assumes independence, so this CD is ANTI-CONSERVATIVE (optimistic); read as "
            "illustrative, not confirmatory.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.4,
            bbox=dict(boxstyle="round", fc="#fff3cd", ec="#e0a800", lw=0.8))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"figA6_corrected_cd.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote figA6_corrected_cd")
    print(f"\n=== A6 avg ranks (N={N} fully-seeded; shift if M=5 included) ===")
    for m, v in items:
        print(f"  {disp(m):22s} rank={v:.2f}  (Δ if incl M=5: {dump['rank_shift_incl_M5'][m]:+.2f})")
    return dump


if __name__ == "__main__":
    a1_head_to_head()
    a2_threshold_sensitivity()
    a3_dominator_counts()
    a4_mobility()
    a5_fairness()
    a6_corrected_cd()

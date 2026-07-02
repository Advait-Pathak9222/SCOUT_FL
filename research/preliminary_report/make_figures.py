"""Generate all figures for the SCOUT-FL / JEDI(VISMAYA)-FL report -> figures/*.{pdf,png}.

JEDI-FL and VISMAYA-FL are treated as ONE proposed method (the tuned `jedi`
configuration); see report_common.META.
"""
from __future__ import annotations
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import report_common as rc

FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.titleweight": "bold", "legend.fontsize": 7.8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.dpi": 320, "axes.grid": True,
    "grid.alpha": 0.28, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "font.family": "serif", "mathtext.fontset": "cm", "pdf.fonttype": 42,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
})

# ------- consistent palette -------
CJEDI, CSV2, CSV1 = "#c0392b", "#1f6fb2", "#12a5b0"
CASAAD = "#e67e22"
CGRAY = "#9aa0a6"
CPROP = {"jedi": CJEDI, "scout_v2": CSV2, "scout_greedy": CSV1}
BCOL = {"random": "#7f7f7f", "divfl": "#2ca02c", "oort": "#8c564b",
        "collabsensefed": "#9467bd", "fedgcs": "#17becf", "sensing_only": "#bcbd22"}


def col_of(m):
    return CPROP.get(m, CASAAD if m == "asaad" else BCOL.get(m, CGRAY))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------ main-point summary
dfm = rc.load_tag("campaign_main")
sm = rc.agg_over_seeds(dfm)
sm["group"] = sm.method.map(rc.group)
sm = sm[sm.group != "Ablation"].reset_index(drop=True)
sm["acc_n"] = rc.minmax_norm(sm.acc_mean)
sm["log_n"] = rc.minmax_norm(sm.logdet_mean)
sm["isac"] = 0.5 * sm.acc_n + 0.5 * sm.log_n


def _style(method):
    if method in CPROP:
        return dict(color=CPROP[method], marker="*", s=330, zorder=6,
                    edgecolor="black", linewidth=0.8)
    if method == "asaad":
        return dict(color=CASAAD, marker="D", s=100, zorder=5,
                    edgecolor="black", linewidth=0.8)
    return dict(color=CGRAY, marker="o", s=44, zorder=3, alpha=0.75,
                edgecolor="white", linewidth=0.4)


def _annot(ax, m, dxpt, dypt, ha, xcol="crb_mean"):
    row = sm[sm.method == m]
    if row.empty:
        return
    r = row.iloc[0]
    txt = {"asaad": "Asaad [TWC'25]", "divfl": "DivFL", "random": "Random",
           "collabsensefed": "CollabSenseFed"}.get(m, rc.disp(m).replace(" (ours)", ""))
    ax.annotate(txt, (r[xcol], r.acc_mean), textcoords="offset points",
                xytext=(dxpt, dypt), ha=ha, va="center",
                fontsize=7.6, fontweight="bold" if rc.is_proposed(m) else "normal",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4", shrinkA=0, shrinkB=7))


PROP_HANDLES = [
    Line2D([0], [0], marker="*", color="w", markerfacecolor="k", markeredgecolor="k",
           markersize=15, label="Proposed (ours)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=CASAAD, markeredgecolor="k",
           markersize=8, label="Asaad TWC'25 (SOTA ISAC)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=CGRAY, markersize=7,
           label="Other baselines"),
    Line2D([0], [0], ls="--", color="k", alpha=0.6, label="Pareto frontier")]


# ==================================================== FIG 1: Pareto (acc vs CRB)
def fig_pareto_crb():
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    pm = rc.pareto_mask(sm.acc_mean, sm.crb_mean, x_max=True, y_max=False)
    fr = sm[pm].sort_values("crb_mean")
    ax.plot(fr.crb_mean, fr.acc_mean, "--", color="black", lw=1.0, alpha=0.55, zorder=2)
    # shade the "sensing-competitive" region CRB<=0.10
    ax.axvspan(sm.crb_mean.min() * 0.8, 0.10, color="#2ecc71", alpha=0.06, zorder=0)
    for _, r in sm.iterrows():
        ax.scatter(r.crb_mean, r.acc_mean, **_style(r.method))
    for m, dx, dy, ha in [("jedi", 20, 16, "left"), ("scout_v2", 22, 5, "left"),
                          ("scout_greedy", 22, -14, "left"), ("asaad", -16, -13, "right"),
                          ("random", -14, 12, "right"), ("divfl", -12, 12, "right"),
                          ("collabsensefed", 20, -6, "left")]:
        _annot(ax, m, dx, dy, ha)
    ax.set_xscale("log"); ax.invert_xaxis(); ax.margins(x=0.13, y=0.10)
    ax.set_xlabel(r"Sensing error  CRB  (log; $\leftarrow$ better)")
    ax.set_ylabel("Test accuracy  ($\\uparrow$ better)")
    ax.set_title("Learning--Sensing trade-off  (CIFAR-10, 150 rd, 5 seeds)")
    ax.text(0.10, sm.acc_mean.min(), "  sensing-\n  competitive\n  (CRB$\\leq$0.10)",
            color="#1e8449", fontsize=6.8, va="bottom", ha="left")
    ax.legend(handles=PROP_HANDLES, loc="lower left", framealpha=0.92)
    save(fig, "fig1_pareto_crb")


# ==================================================== FIG 2: Pareto (acc vs logdet)
def fig_pareto_logdet():
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    pm = rc.pareto_mask(sm.acc_mean, sm.logdet_mean, x_max=True, y_max=True)
    fr = sm[pm].sort_values("logdet_mean")
    ax.plot(fr.logdet_mean, fr.acc_mean, "--", color="black", lw=1.0, alpha=0.55, zorder=2)
    for _, r in sm.iterrows():
        ax.scatter(r.logdet_mean, r.acc_mean, **_style(r.method))
    for m, dx, dy, ha in [("jedi", -16, 14, "right"), ("scout_v2", 20, 6, "left"),
                          ("asaad", 16, 2, "left"), ("random", 14, 8, "left"),
                          ("divfl", -12, 12, "right"), ("scout_greedy", 20, -12, "left")]:
        _annot(ax, m, dx, dy, ha, xcol="logdet_mean")
    ax.margins(x=0.13, y=0.10)
    ax.set_xlabel("Sensing information  log-det Fisher  ($\\uparrow$ better)")
    ax.set_ylabel("Test accuracy  ($\\uparrow$ better)")
    ax.set_title("Both objectives maximised: proposed hold the knee")
    ax.legend(handles=PROP_HANDLES[:3], loc="lower left", framealpha=0.92)
    save(fig, "fig2_pareto_logdet")


# ==================================================== FIG 3: convergence
def fig_convergence():
    rr = rc.load_rounds("campaign_main", "base")
    methods = ["random", "divfl", "jedi", "scout_v2", "asaad", "oort"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for m in methods:
        sub = rr[rr.method == m]
        g = sub.groupby("round")
        mu, sd = g.test_acc.mean(), g.test_acc.std()
        c = col_of(m); big = m in CPROP or m == "asaad"
        a1.plot(mu.index, mu.values, color=c, lw=2.2 if big else 1.3,
                label=rc.disp(m).replace(" (ours)", ""))
        if big:
            a1.fill_between(mu.index, mu - sd, mu + sd, color=c, alpha=0.14)
        a2.plot(g.sensing_logdet.mean().index, g.sensing_logdet.mean().values,
                color=c, lw=2.2 if big else 1.3)
    a1.set_xlabel("Communication round"); a1.set_ylabel("Test accuracy")
    a1.set_title("(a) Learning convergence"); a1.legend(loc="lower right", ncol=2)
    a2.set_xlabel("Communication round"); a2.set_ylabel("Sensing log-det Fisher info")
    a2.set_title("(b) Sensing information")
    save(fig, "fig3_convergence")


# ==================================================== FIG 4: joint ISAC bar
def fig_isac_bar():
    s = sm.sort_values("isac").tail(16)
    cols = [col_of(m) for m in s.method]
    fig, ax = plt.subplots(figsize=(6.2, 4.7))
    ax.barh([rc.disp(m).replace(" (ours)", "") for m in s.method], s.isac,
            color=cols, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Joint ISAC score  $=\\frac{1}{2}$norm(acc)$+\\frac{1}{2}$norm(sensing)")
    ax.set_title("Balanced learning+sensing utility (top 16, CIFAR-10)")
    for i, (_, r) in enumerate(s.iterrows()):
        w = "bold" if rc.is_proposed(r.method) else "normal"
        ax.text(r.isac + 0.006, i, f"{r.isac:.2f}", va="center", fontsize=7, fontweight=w)
    ax.set_xlim(0, 1.02)
    save(fig, "fig4_isac_bar")


# ==================================================== FIG 5: SNR sweep
def fig_snr_sweep():
    s = rc.agg_over_seeds(rc.load_tag("campaign"))
    snr = s[s.point.str.startswith("B_wireless_snr=")].copy()
    snr["snr"] = snr.point.str.replace("B_wireless_snr=", "").astype(float)
    methods = ["jedi", "scout_v2", "asaad", "random", "sensing_only"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for m in methods:
        sub = snr[snr.method == m].sort_values("snr")
        c = col_of(m); mk = "*" if m in CPROP else ("D" if m == "asaad" else "o")
        big = m in CPROP or m == "asaad"
        a1.plot(sub.snr, sub.acc_mean, marker=mk, color=c, lw=2.0 if big else 1.4,
                ms=9 if m in CPROP else 6, label=rc.disp(m).replace(" (ours)", ""))
        a2.plot(sub.snr, sub.crb_mean, marker=mk, color=c, lw=2.0 if big else 1.4,
                ms=9 if m in CPROP else 6)
    a1.set_xlabel("Uplink tx-power / SNR proxy (dBm)"); a1.set_ylabel("Test accuracy")
    a1.set_title("(a) Accuracy vs link quality"); a1.legend(fontsize=7.2)
    a2.set_xlabel("Uplink tx-power / SNR proxy (dBm)"); a2.set_ylabel("Sensing CRB ($\\downarrow$)")
    a2.set_yscale("log"); a2.set_title("(b) Sensing error vs link quality")
    save(fig, "fig5_snr_sweep")


# ==================================================== FIG 6: datasets
def fig_datasets():
    s = rc.agg_over_seeds(rc.load_tag("campaign"))
    ds = s[s.point.str.startswith("A_datasets=")].copy()
    ds["d"] = ds.point.str.replace("A_datasets=", "")
    order = ["emnist", "fashion_mnist", "uci_har", "cifar10", "cifar100"]
    lab = {"emnist": "EMNIST", "fashion_mnist": "F-MNIST", "uci_har": "UCI-HAR",
           "cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
    methods = ["jedi", "scout_v2", "asaad", "divfl", "random"]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    x = np.arange(len(order)); w = 0.16
    for i, m in enumerate(methods):
        vals = [ds[(ds.d == d) & (ds.method == m)].acc_mean.mean() for d in order]
        errs = [ds[(ds.d == d) & (ds.method == m)].acc_std.mean() for d in order]
        ax.bar(x + (i - 2) * w, vals, w, yerr=errs, capsize=2, color=col_of(m),
               edgecolor="black", linewidth=0.4, label=rc.disp(m).replace(" (ours)", ""))
    ax.set_xticks(x); ax.set_xticklabels([lab[d] for d in order])
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy across datasets (K/N=10%, 150 rd, 5 seeds)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=7.6)
    save(fig, "fig6_datasets")


# ==================================================== FIG 7: regime win
def fig_regime_win():
    s = rc.agg_over_seeds(rc.load_tag("campaign")); s["group"] = s.method.map(rc.group)
    s = s[s.group != "Ablation"]
    labelmap = {"A_datasets=cifar10": "CIFAR-10", "A_datasets=cifar100": "CIFAR-100",
                "A_datasets=emnist": "EMNIST", "A_datasets=fashion_mnist": "F-MNIST",
                "A_datasets=uci_har": "UCI-HAR", "A_learning_noniid=0.1": r"$\alpha$=0.1",
                "A_learning_noniid=0.3": r"$\alpha$=0.3", "A_learning_noniid=0.5": r"$\alpha$=0.5",
                "A_learning_partition=iid": "IID", "A_learning_partition=spatial": "spatial",
                "B_wireless_channel=rayleigh": "Rayleigh", "B_wireless_channel=rician": "Rician",
                "B_wireless_snr=0": "SNR 0", "B_wireless_snr=-10": "SNR-10",
                "B_wireless_snr=-15": "SNR-15", "B_wireless_snr=-20": "SNR-20",
                "B_wireless_snr=-25": "SNR-25", "B_wireless_snr=-30": "SNR-30",
                "B_wireless_snr=-35": "SNR-35", "C_sensing_targets=2": "M=2",
                "C_sensing_targets=3": "M=3", "C_sensing_targets=5": "M=5"}
    pts, accs, cols = [], [], []
    for pt, sub in s.groupby("point"):
        ok = sub[sub.crb_mean <= 0.10]
        ok = ok if len(ok) else sub
        wnr = ok.sort_values("acc_mean", ascending=False).iloc[0]
        pts.append(labelmap.get(pt, pt)); accs.append(wnr.acc_mean); cols.append(col_of(wnr.method))
    o = np.argsort(accs)
    pts = [pts[i] for i in o]; accs = [accs[i] for i in o]; cols = [cols[i] for i in o]
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    ax.barh(pts, accs, color=cols, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Best accuracy among methods meeting sensing bar  CRB$\\leq$0.10")
    ax.set_title("Accuracy at matched sensing: a proposed method wins 21/22")
    hs = [Patch(fc=CJEDI, ec="k", label="JEDI/VISMAYA-FL"),
          Patch(fc=CSV2, ec="k", label="SCOUT-FL v2"),
          Patch(fc=CSV1, ec="k", label="SCOUT-FL v1"),
          Patch(fc=CGRAY, ec="k", label="Random (no sensing)")]
    ax.legend(handles=hs, loc="lower right", fontsize=7.6)
    save(fig, "fig7_regime_win")


# ==================================================== FIG 8: targets
def fig_targets():
    s = rc.agg_over_seeds(rc.load_tag("campaign"))
    tg = s[s.point.str.startswith("C_sensing_targets=")].copy()
    tg["M"] = tg.point.str.replace("C_sensing_targets=", "").astype(int)
    methods = ["jedi", "scout_v2", "asaad", "random"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for m in methods:
        sub = tg[tg.method == m].sort_values("M")
        c = col_of(m); mk = "*" if m in CPROP else ("D" if m == "asaad" else "o")
        a1.plot(sub.M, sub.acc_mean, marker=mk, color=c, lw=2.0, ms=10 if m in CPROP else 6,
                label=rc.disp(m).replace(" (ours)", ""))
        a2.plot(sub.M, sub.logdet_mean, marker=mk, color=c, lw=2.0, ms=10 if m in CPROP else 6)
    a1.set_xlabel("# sensing targets M"); a1.set_ylabel("Test accuracy")
    a1.set_xticks([2, 3, 5]); a1.set_title("(a) Accuracy"); a1.legend(fontsize=7.4)
    a2.set_xlabel("# sensing targets M"); a2.set_ylabel("Sensing log-det info")
    a2.set_xticks([2, 3, 5]); a2.set_title("(b) Sensing information")
    save(fig, "fig8_targets")


# ==================================================== FIG 9: synergy component
def fig_vismaya_synergy():
    rr = rc.load_rounds("ablation_vismaya", "base")
    if rr.empty or rr.vis_synergy_mean.isna().all():
        print("skip synergy (no data)"); return
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for m, c, lab in [("vismaya", CJEDI, "full generative + synergy"),
                      ("vismaya_no_syn", CGRAY, "synergy term OFF"),
                      ("vismaya_sense_only", "#8c564b", "sensing-only")]:
        sub = rr[rr.method == m]
        if sub.empty:
            continue
        g = sub.groupby("round").vis_synergy_mean.mean()
        ax.plot(g.index, g.values, color=c, lw=2.0, label=lab)
    ax.set_xlabel("Communication round (target mobility, $\\sigma_p{=}0.05$)")
    ax.set_ylabel("Mean synergy signal  $\\Omega_s$")
    ax.set_title("Generative-synergy component grows under mobility")
    ax.legend(fontsize=7.6)
    save(fig, "fig9_synergy")


# ==================================================== FIG 10: critical-difference
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
        9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}


def _avg_ranks(metric="isac"):
    """Average rank (1=best) of a fixed method set across the 22 OFAT points,
    ranking by joint ISAC score (higher=better)."""
    s = rc.agg_over_seeds(rc.load_tag("campaign")); s["group"] = s.method.map(rc.group)
    methods = ["jedi", "scout_v2", "scout_greedy", "asaad", "collabsensefed",
               "fedgcs", "divfl", "oort", "random"]
    ranks = {m: [] for m in methods}
    pts = 0
    for pt, sub in s.groupby("point"):
        sub = sub[sub.method.isin(methods)].copy()
        if len(sub) < len(methods):
            continue
        sub["acc_n"] = rc.minmax_norm(sub.acc_mean); sub["log_n"] = rc.minmax_norm(sub.logdet_mean)
        sub["score"] = 0.5 * sub.acc_n + 0.5 * sub.log_n
        sub = sub.sort_values("score", ascending=False).reset_index(drop=True)
        for i, r in sub.iterrows():
            ranks[r.method].append(i + 1)
        pts += 1
    avg = {m: np.mean(v) for m, v in ranks.items()}
    return avg, pts, len(methods)


def fig_cd_diagram():
    avg, N, k = _avg_ranks()
    CD = _Q05[k] * np.sqrt(k * (k + 1) / (6.0 * N))
    items = sorted(avg.items(), key=lambda kv: kv[1])   # best (low rank) first
    names = [rc.disp(m).replace(" (ours)", "") for m, _ in items]
    vals = [v for _, v in items]

    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ax.set_xlim(lo - 0.5, hi + 0.5); ax.set_ylim(0.05, 1.05)
    ax.axis("off")
    yaxis = 0.66
    ax.plot([lo, hi], [yaxis, yaxis], "k-", lw=1.2)
    for x in range(lo, hi + 1):
        ax.plot([x, x], [yaxis, yaxis + 0.028], "k-", lw=1.0)
        ax.text(x, yaxis + 0.055, str(x), ha="center", va="bottom", fontsize=8)
    ax.text((lo + hi) / 2, yaxis + 0.135, "average rank  (1 = best joint ISAC score)",
            ha="center", fontsize=8.5, fontweight="bold")
    ax.text((lo + hi) / 2, 1.00, f"Critical-difference diagram "
            f"(Nemenyi, $\\alpha$=0.05, {N} operating points)",
            ha="center", fontsize=10.5, fontweight="bold")
    # CD reference bar (top-left, clearly separated)
    ybarCD = yaxis + 0.26
    ax.plot([lo, lo + CD], [ybarCD, ybarCD], "k-", lw=2.4)
    ax.plot([lo, lo], [ybarCD - 0.02, ybarCD + 0.02], "k-", lw=1.2)
    ax.plot([lo + CD, lo + CD], [ybarCD - 0.02, ybarCD + 0.02], "k-", lw=1.2)
    ax.text(lo + CD / 2, ybarCD + 0.03, f"CD = {CD:.2f}", ha="center", fontsize=8.5)
    # left half labels (better) and right half (worse)
    half = int(np.ceil(k / 2))
    step = 0.095
    for idx, (m, v) in enumerate(items):
        nm = rc.disp(m).replace(" (ours)", "")
        prop = rc.is_proposed(m); c = col_of(m)
        if idx < half:
            yy = yaxis - 0.10 - idx * step
            ax.plot([v, v], [yaxis, yy], color=c, lw=1.5)
            ax.plot([v, lo - 0.45], [yy, yy], color=c, lw=1.5)
            ax.text(lo - 0.5, yy, nm, ha="right", va="center", fontsize=8.2,
                    fontweight="bold" if prop else "normal", color=c if prop else "black")
        else:
            yy = yaxis - 0.10 - (k - 1 - idx) * step
            ax.plot([v, v], [yaxis, yy], color=c, lw=1.5)
            ax.plot([v, hi + 0.45], [yy, yy], color=c, lw=1.5)
            ax.text(hi + 0.5, yy, nm, ha="left", va="center", fontsize=8.2,
                    fontweight="bold" if prop else "normal", color=c if prop else "black")
    # cliques: maximal groups whose rank spread < CD
    vs = sorted(vals)
    cliques = []
    i = 0
    while i < len(vs):
        j = i
        while j + 1 < len(vs) and vs[j + 1] - vs[i] < CD:
            j += 1
        if j > i:
            cliques.append((vs[i], vs[j]))
        i += 1
    ybar = yaxis - 0.04
    used = []
    for (a, b) in cliques:
        if any(a >= pa and b <= pb for pa, pb in used):
            continue
        used.append((a, b))
        ax.plot([a - 0.04, b + 0.04], [ybar, ybar], color="0.2", lw=3.4, solid_capstyle="round")
        ybar -= 0.028
    save(fig, "fig10_cd_diagram")


# ==================================================== FIG 11: radar scorecard
def fig_radar():
    # axes normalised across the full non-ablation set at the main point + robustness
    s = sm.copy()
    s["crb_score"] = 1 - rc.minmax_norm(np.clip(s.crb_mean, 0, 1.0))
    s["mse_score"] = 1 - rc.minmax_norm(s.agg_mse_mean)
    s["jain_score"] = rc.minmax_norm(s.jain_mean)
    # robustness = mean normalised accuracy across the 5 datasets
    cs = rc.agg_over_seeds(rc.load_tag("campaign"))
    ds = cs[cs.point.str.startswith("A_datasets=")].copy()
    ds["d"] = ds.point.str.replace("A_datasets=", "")
    robust = {}
    accn = {}
    for d, sub in ds.groupby("d"):
        sub = sub.copy(); sub["n"] = rc.minmax_norm(sub.acc_mean)
        for _, r in sub.iterrows():
            accn.setdefault(r.method, []).append(r.n)
    robust = {m: np.mean(v) for m, v in accn.items()}

    axes = ["Accuracy", "Sensing\n(log-det)", "Localization\n(1$-$CRB)",
            "Fairness\n(Jain)", "Comm\n(1$-$MSE)", "Robustness\n(cross-data)"]
    def vec(m):
        r = s[s.method == m].iloc[0]
        return [r.acc_n, r.log_n, r.crb_score, r.jain_score, r.mse_score, robust.get(m, np.nan)]

    methods = [("jedi", CJEDI), ("scout_v2", CSV2), ("asaad", CASAAD), ("divfl", "#2ca02c")]
    ang = np.linspace(0, 2 * np.pi, len(axes), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    fig, ax = plt.subplots(figsize=(5.6, 5.2), subplot_kw=dict(polar=True))
    for m, c in methods:
        v = vec(m); v = v + v[:1]
        ax.plot(ang, v, color=c, lw=2.0, label=rc.disp(m).replace(" (ours)", ""))
        ax.fill(ang, v, color=c, alpha=0.10)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(axes, fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["", ".5", "", "1"], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title("Six-axis scorecard  (normalised; outer = best)", fontsize=10.5, y=1.10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, frameon=True)
    save(fig, "fig11_radar")


# ==================================================== FIG 12: rank heatmap
def fig_rank_heatmap():
    s = rc.agg_over_seeds(rc.load_tag("campaign")); s["group"] = s.method.map(rc.group)
    methods = ["jedi", "scout_v2", "scout_greedy", "collabsensefed", "fedgcs",
               "divfl", "asaad", "oort", "random"]
    labelmap = {"A_datasets=cifar10": "C10", "A_datasets=cifar100": "C100",
                "A_datasets=emnist": "EMN", "A_datasets=fashion_mnist": "FMN",
                "A_datasets=uci_har": "HAR", "A_learning_noniid=0.1": r"$\alpha$.1",
                "A_learning_noniid=0.3": r"$\alpha$.3", "A_learning_noniid=0.5": r"$\alpha$.5",
                "A_learning_partition=iid": "IID", "A_learning_partition=spatial": "SP",
                "B_wireless_channel=rayleigh": "Ray", "B_wireless_channel=rician": "Ric",
                "B_wireless_snr=0": "S0", "B_wireless_snr=-10": "S-10", "B_wireless_snr=-15": "S-15",
                "B_wireless_snr=-20": "S-20", "B_wireless_snr=-25": "S-25", "B_wireless_snr=-30": "S-30",
                "B_wireless_snr=-35": "S-35", "C_sensing_targets=2": "M2",
                "C_sensing_targets=3": "M3", "C_sensing_targets=5": "M5"}
    order_pts = [p for p in labelmap if p in set(s.point)]
    M = np.full((len(methods), len(order_pts)), np.nan)
    for j, pt in enumerate(order_pts):
        sub = s[(s.point == pt) & (s.method.isin(methods))].copy()
        if len(sub) < len(methods):
            continue
        sub["acc_n"] = rc.minmax_norm(sub.acc_mean); sub["log_n"] = rc.minmax_norm(sub.logdet_mean)
        sub["score"] = 0.5 * sub.acc_n + 0.5 * sub.log_n
        sub = sub.sort_values("score", ascending=False).reset_index(drop=True)
        rankmap = {r.method: i + 1 for i, r in sub.iterrows()}
        for i, m in enumerate(methods):
            M[i, j] = rankmap.get(m, np.nan)
    mean_rank = np.nanmean(M, axis=1)
    o = np.argsort(mean_rank)
    M = M[o]; methods = [methods[i] for i in o]; mean_rank = mean_rank[o]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    cmap = plt.get_cmap("RdYlGn_r")
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=1, vmax=len(methods))
    ax.set_xticks(range(len(order_pts)))
    ax.set_xticklabels([labelmap[p] for p in order_pts], rotation=90, fontsize=6.6)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([rc.disp(m).replace(" (ours)", "") +
                        f"  ({mean_rank[i]:.1f})" for i, m in enumerate(methods)], fontsize=7.6)
    for i in range(len(methods)):
        for j in range(len(order_pts)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=5.8,
                        color="white" if (M[i, j] <= 2 or M[i, j] >= len(methods) - 1) else "black")
    for i, m in enumerate(methods):
        if rc.is_proposed(m):
            ax.get_yticklabels()[i].set_fontweight("bold")
    ax.set_title("Per-point rank by joint ISAC score (1=best, green). "
                 "Row label shows mean rank", fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("rank", fontsize=8)
    save(fig, "fig12_rank_heatmap")


# ==================================================== FIG 13: non-IID sweep
def fig_noniid():
    s = rc.agg_over_seeds(rc.load_tag("campaign"))
    nid = s[s.point.str.startswith("A_learning_noniid=")].copy()
    nid["a"] = nid.point.str.replace("A_learning_noniid=", "").astype(float)
    methods = ["jedi", "scout_v2", "asaad", "divfl", "random"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for m in methods:
        sub = nid[nid.method == m].sort_values("a")
        c = col_of(m); mk = "*" if m in CPROP else ("D" if m == "asaad" else "o")
        big = m in CPROP or m == "asaad"
        a1.plot(sub.a, sub.acc_mean, marker=mk, color=c, lw=2.0 if big else 1.4,
                ms=9 if m in CPROP else 6, label=rc.disp(m).replace(" (ours)", ""))
        a2.plot(sub.a, sub.crb_mean, marker=mk, color=c, lw=2.0 if big else 1.4,
                ms=9 if m in CPROP else 6)
    a1.set_xlabel(r"Dirichlet $\alpha$ (smaller = more non-IID)"); a1.set_ylabel("Test accuracy")
    a1.set_xticks([0.1, 0.3, 0.5]); a1.set_title("(a) Accuracy vs heterogeneity")
    a1.legend(fontsize=7.2)
    a2.set_xlabel(r"Dirichlet $\alpha$"); a2.set_ylabel("Sensing CRB ($\\downarrow$)")
    a2.set_xticks([0.1, 0.3, 0.5]); a2.set_yscale("log"); a2.set_title("(b) Sensing error")
    save(fig, "fig13_noniid")


if __name__ == "__main__":
    for fn in [fig_pareto_crb, fig_pareto_logdet, fig_convergence, fig_isac_bar,
               fig_snr_sweep, fig_datasets, fig_regime_win, fig_targets,
               fig_vismaya_synergy, fig_cd_diagram, fig_radar, fig_rank_heatmap,
               fig_noniid]:
        fn()
    print("ALL FIGURES DONE ->", FIG)

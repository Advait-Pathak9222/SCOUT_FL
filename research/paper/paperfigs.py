"""Publication figures for the SCOUT-FL paper (IEEE TWC).

Rules baked in:
  * SCOUT-FL is the ONLY proposed method shown. JEDI-FL is removed everywhere.
  * Comparisons are against SENSING-AWARE baselines only (fair, real-world).
  * final-round CRB is the primary sensing convention.
  * tight layout, legends never over the data, Times New Roman.

Every number is computed from runs/ via report_common; the geometry/RMSE figures use
the actual FIM code in scout_fl/sim.

Run:  python paperfigs.py    ->  figures/*.{pdf,png}
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy import stats as sstats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Ellipse, FancyArrowPatch, FancyBboxPatch, Circle, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "preliminary_report"))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import report_common as rc                      # noqa: E402
import beautify as bu                           # noqa: E402
from scout_fl.sim.fim import per_client_target_fim   # noqa: E402
from scout_fl.sim.geometry import pairwise_geometry  # noqa: E402

FIG = os.path.join(HERE, "figures")
bu.set_style()

HEAD = "scout_v2"                       # THE proposed method (labelled SCOUT-FL)
ABL = "scout_greedy"                    # hard-gate ablation of SCOUT-FL
STRONG = ["collabsensefed", "sensing_native"]        # the real competition
SOTA = "asaad"
# sensing-aware pool (ISAC + sensing primitives); pure learning/comm/naive excluded
SENSING_BASE = ["collabsensefed", "sensing_native", "asaad", "fixed_weighted", "fed_iscc",
                "ota_fl_iscc", "iscc_air_feel", "fedavg_iscc", "fedsgd_iscc",
                "crb_only", "sensing_only"]
POOL = [HEAD] + SENSING_BASE            # dominator pool = "among sensing-aware methods"
MAIN = "campaign_main"

DISP = {"scout_v2": "SCOUT-FL", "scout_greedy": "SCOUT-FL$^{\\dagger}$",
        "collabsensefed": "CollabSenseFed", "sensing_native": "Sensing-Native",
        "asaad": "Asaad", "fixed_weighted": "Fixed-Weighted", "fed_iscc": "Fed-ISCC",
        "ota_fl_iscc": "OTA-FL-ISCC", "iscc_air_feel": "ISCC-Air-FEEL",
        "fedavg_iscc": "FedAvg-ISCC", "fedsgd_iscc": "FedSGD-ISCC",
        "crb_only": "CRB-Only", "sensing_only": "Sensing-Only"}
def d(m): return DISP.get(m, m)

C = {"scout_v2": "#0f5e8c",        # deep blue  (HERO)
     "scout_greedy": "#3a8f88",    # teal (ablation)
     "collabsensefed": "#7d6ba7",  # purple
     "sensing_native": "#9c6b4e",  # brown
     "asaad": "#e0912f",           # orange
     "fixed_weighted": "#2a9d8f",  # green-teal
     "crb_only": "#c0392b",        # red
     "sensing_only": "#d46a9f",    # pink
     "ota_fl_iscc": "#4c8c3f",     # green
     "fed_iscc": "#9a8a2e",        # olive
     "fedavg_iscc": "#5b8fbf",     # steel blue
     "iscc_air_feel": "#a05195",   # magenta
     "fedsgd_iscc": "#7a7d82"}     # gray
def c(m): return C.get(m, "#aab0b6")

PT_LABEL = {
    "A_datasets=emnist": "EMNIST", "A_datasets=fashion_mnist": "F-MNIST",
    "A_datasets=uci_har": "UCI-HAR", "A_datasets=cifar10": "CIFAR-10",
    "A_datasets=cifar100": "CIFAR-100", "A_learning_partition=iid": "IID",
    "A_learning_partition=spatial": "spatial", "A_learning_noniid=0.1": r"$\alpha$0.1",
    "A_learning_noniid=0.3": r"$\alpha$0.3", "A_learning_noniid=0.5": r"$\alpha$0.5",
    "B_wireless_channel=rayleigh": "Rayleigh", "B_wireless_channel=rician": "Rician",
    "B_wireless_snr=0": "SNR 0", "B_wireless_snr=-10": "$-$10", "B_wireless_snr=-15": "$-$15",
    "B_wireless_snr=-20": "$-$20", "B_wireless_snr=-25": "$-$25", "B_wireless_snr=-30": "$-$30",
    "B_wireless_snr=-35": "$-$35", "C_sensing_targets=2": "M=2", "C_sensing_targets=3": "M=3",
    "C_sensing_targets=5": "M=5", "C_sensing_kangle=0.02": "$\\kappa$.02",
    "C_sensing_kangle=0.05": "$\\kappa$.05", "C_sensing_kangle=0.1": "$\\kappa$.1"}
PT_ORDER = list(PT_LABEL.keys())


def _agg(tag, metrics):
    df = rc.load_tag(tag); df["group"] = df.method.map(rc.group)
    return rc.agg_over_seeds(df, metrics=metrics)


def _paired(df, a, b, metric):
    pa = df[df.method == a][["point", "seed", metric]].rename(columns={metric: "a"})
    pb = df[df.method == b][["point", "seed", metric]].rename(columns={metric: "b"})
    m = pa.merge(pb, on=["point", "seed"], how="inner"); m["dd"] = m["a"] - m["b"]
    return m


def _boot_ci(x, n=8000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return float(np.mean(x)) if len(x) else np.nan, np.nan, np.nan
    r = np.random.default_rng(seed)
    bs = np.mean(x[r.integers(0, len(x), (n, len(x)))], 1)
    return float(np.mean(x)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


# ══════════════════════════════════════════════ FIG 3 — learning–sensing frontier
def fig_frontier():
    s = _agg(MAIN, ["acc", "crb_final"])
    s = s[s.method.isin(POOL + [ABL])].dropna(subset=["acc_mean", "crb_final_mean"])
    # zoom to the sensing-competitive cluster; 4 ISAC baselines with CRB>0.4 are off-scale
    # (they are strictly worse on both axes) and noted in the caption, not drawn.
    vis = s[s.crb_final_mean < 0.16].copy()
    fig, ax = plt.subplots(figsize=(3.5, 3.1))    # \columnwidth print size, 8pt text
    bu.floating_axes(ax); bu.soft_grid(ax, "both")

    best_crb = vis.crb_final_mean.min(); band_hi = best_crb * 1.35
    ax.axvspan(best_crb * 0.9, band_hi, color="#2a9d5c", alpha=0.08, zorder=0)

    keep = rc.pareto_mask(vis.acc_mean.values, vis.crb_final_mean.values, True, False)
    fr = vis[keep].sort_values("crb_final_mean")
    ax.plot(fr.crb_final_mean, fr.acc_mean, color=bu.MUTE, lw=1.3, ls="--", alpha=0.65, zorder=2,
            label="Pareto frontier")

    # distinct colour per method (no leader lines) — a compact legend carries the identities,
    # so the tight sensing cluster stays clean.  SCOUT-FL is the haloed hero diamond.
    order = [m for m in vis.method if m not in (HEAD, ABL)] + [ABL, HEAD]
    for m in order:
        r = vis[vis.method == m].iloc[0]
        if m == HEAD:
            bu.halo(ax, r.crb_final_mean, r.acc_mean, c(m), s=170)
            ax.scatter(r.crb_final_mean, r.acc_mean, s=150, marker="D", color=c(m),
                       edgecolor="white", lw=1.2, zorder=10)
        elif m == ABL:
            ax.scatter(r.crb_final_mean, r.acc_mean, s=70, marker="D", color=c(m),
                       edgecolor="white", lw=1.0, zorder=8)
        else:
            ax.scatter(r.crb_final_mean, r.acc_mean, s=55, marker="o", color=c(m),
                       edgecolor="white", lw=0.8, zorder=6)
    # only the hero gets a direct label; everything else is in the legend
    hero = vis[vis.method == HEAD].iloc[0]
    ax.annotate("SCOUT-FL", (hero.crb_final_mean, hero.acc_mean), xytext=(-10, 14),
                textcoords="offset points", ha="right", fontsize=8, fontweight="bold",
                color=c(HEAD), zorder=11)
    ax.text(band_hi, vis.acc_mean.min() - 0.004, "sensing-competitive band",
            fontsize=8, color="#1e7d46", ha="center", va="top", style="italic")

    ax.set_xscale("log"); ax.set_xlim(0.135, 0.047); ax.set_ylim(0.325, 0.495)
    tv = [0.05, 0.06, 0.07, 0.08, 0.10, 0.13]
    ax.set_xticks(tv); ax.set_xticklabels([f"{v:g}" for v in tv]); ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("final-round CRB  (log scale, $\\leftarrow$ better)")
    ax.set_ylabel("test accuracy  ($\\uparrow$ better)")

    # compact identity legend (2 columns) in the empty lower-left region
    def _h(m, hero=False, abl=False):
        mk = "D" if (hero or abl) else "o"
        return Line2D([0], [0], marker=mk, ls="", mfc=c(m), mec="w",
                      ms=9 if hero else (7 if abl else 6.5),
                      label=d(m) + (" (proposed)" if hero else (" (abl.)" if abl else "")))
    handles = [_h(HEAD, hero=True), _h(ABL, abl=True)] + \
              [_h(m) for m in ["collabsensefed", "sensing_native", "asaad", "fixed_weighted",
                               "ota_fl_iscc", "crb_only", "sensing_only"]]
    ax.legend(handles=handles, loc="lower left", fontsize=8, ncol=1, borderpad=0.3,
              handletextpad=0.25, labelspacing=0.22, framealpha=0.92)
    fig.tight_layout(pad=0.3)
    bu.save(fig, os.path.join(FIG, "fig3_frontier"))


# ══════════════════════════════════════════════ FIG 4 — paired Δ vs strong baselines
def fig_paired_delta():
    df = rc.load_tag("campaign"); df["group"] = df.method.map(rc.group)
    df = df[["point", "method", "seed", "acc", "crb_final"]]
    present = [p for p in PT_ORDER if p in set(df.point)]
    from matplotlib.colors import to_rgba
    VIV = {"collabsensefed": "#6a3fb5", "sensing_native": "#d95f02"}  # vivid, high-contrast
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True)    # \columnwidth, 8pt
    for ax, (metric, ylab, scale) in zip(
            axes, [("acc", "$\\Delta$ accuracy (pp)", 100),
                   ("crb_final", "$\\Delta$ final-CRB", 1)]):
        bu.floating_axes(ax); bu.soft_grid(ax, "y")
        ax.axhline(0, color=bu.INK, lw=1.0, ls="--", alpha=0.6)
        for j, b in enumerate(STRONG):
            m = _paired(df, HEAD, b, metric)
            xs, ys, lo, hi, tie = [], [], [], [], []
            for i, pt in enumerate(present):
                g = m[m.point == pt]["dd"].values * scale
                if len(g) < 2: continue
                mu = g.mean(); se = g.std(ddof=1) / np.sqrt(len(g))
                h = sstats.t.ppf(.975, len(g) - 1) * se
                xs.append(i + (j - 0.5) * 0.26); ys.append(mu)
                lo.append(h); hi.append(h); tie.append(abs(mu) < h)
            xs = np.array(xs); ys = np.array(ys); tie = np.array(tie)
            col = VIV.get(b, c(b))
            ax.errorbar(xs[~tie], ys[~tie], yerr=np.array(lo)[~tie], fmt="o", ms=3.8, color=col,
                        mec="white", mew=0.5, ecolor=col, elinewidth=1.2, capsize=1.5,
                        zorder=5, label=f"vs {d(b)}")
            if tie.any():
                ax.errorbar(xs[tie], ys[tie], yerr=np.array(lo)[tie], fmt="o", ms=3.5,
                            mfc="white", mec=col, mew=1.0, ecolor=to_rgba(col, 0.40),
                            elinewidth=1.0, capsize=1.5, zorder=4)
        ax.set_ylabel(ylab)
    # legend above the top panel (no in-figure title — the caption carries it)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=8,
                   frameon=False, handletextpad=0.3, columnspacing=0.9)
    axes[-1].set_xticks(range(len(present)))
    # 25 rotated labels cannot fit a 3.15in-wide axes at 10pt (needs 3.5in) — 9pt is the
    # largest size with no letter collisions
    axes[-1].set_xticklabels([PT_LABEL[p] for p in present], rotation=90, fontsize=8)
    fig.tight_layout(pad=0.4)
    bu.save(fig, os.path.join(FIG, "fig4_paired_delta"))


# ══════════════════════════════════════════════ FIG 5 — dominance both conventions
def _dominance(crbcol):
    s = _agg("campaign", ["acc", "crb", "crb_final"])
    s = s[s.method.isin(POOL)]
    present = [p for p in PT_ORDER if p in set(s.point)]
    rows = [HEAD, "collabsensefed", "sensing_native", "asaad", "crb_only", "sensing_only",
            "fixed_weighted", "ota_fl_iscc"]
    M = np.full((len(rows), len(present)), np.nan)
    for jj, pt in enumerate(present):
        sub = s[s.point == pt]
        acc = dict(zip(sub.method, sub.acc_mean)); crb = dict(zip(sub.method, sub[crbcol]))
        for ii, m in enumerate(rows):
            if m not in acc: continue
            dom = sum(1 for o in sub.method if o != m and acc[o] > acc[m] and crb[o] < crb[m])
            M[ii, jj] = dom
    return M, rows, present


def fig_dominance():
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.6))
    frac = {}
    for ax, (crbcol, title) in zip(axes, [("crb_mean", "round-mean CRB (secondary)"),
                                          ("crb_final_mean", "final-round CRB (PRIMARY)")]):
        M, rows, present = _dominance(crbcol)
        med = np.nanmedian(M, 1); order = np.argsort(med)
        Mo = M[order]; rl = [rows[i] for i in order]
        frac[crbcol] = {rows[i]: float(np.nanmean(M[i] == 0)) for i in range(len(rows))}
        vmax = max(1, int(np.nanmax(M)))
        im = ax.imshow(Mo, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=vmax)
        ax.set_yticks(range(len(rl)))
        ax.set_yticklabels([f"{d(m)}" for m in rl], fontsize=8.2)
        for lb, m in zip(ax.get_yticklabels(), rl):
            if m == HEAD: lb.set_fontweight("bold"); lb.set_color(c(HEAD))
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([PT_LABEL[p] for p in present] if crbcol == "crb_final_mean" else [],
                           rotation=90, fontsize=6.6)
        for i in range(len(rl)):
            for j in range(len(present)):
                if not np.isnan(Mo[i, j]):
                    ax.text(j, i, int(Mo[i, j]), ha="center", va="center", fontsize=5.6,
                            color="white" if (Mo[i, j] >= vmax*.6 or Mo[i, j] == 0) else "black")
        po = frac[crbcol][HEAD] * 100
        ax.set_title(f"# strict dominators — {title}   ·   SCOUT-FL Pareto-optimal in {po:.0f}% of points",
                     fontsize=8.4)
    fig.colorbar(im, ax=axes, fraction=0.018, pad=0.015).set_label("# dominators", fontsize=8)
    bu.save(fig, os.path.join(FIG, "fig5_dominance"))
    return frac


# ══════════════════════════════════════════════ FIG 6 — CRB-threshold sensitivity
def fig_threshold():
    s = _agg("campaign", ["acc", "crb", "crb_final"])
    s = s[s.method.isin(POOL)]
    present = sorted(set(s.point))
    taus = np.round(np.arange(0.045, 0.1301, 0.004), 3)
    # single-column: the two CRB conventions stacked vertically, legend below
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.0), sharex=True, sharey=True)
    for ax in axes: bu.floating_axes(ax); bu.soft_grid(ax, "y")
    # count, per threshold, how often each method is the most accurate one that still
    # meets the sensing bar CRB <= tau.  All sensing-aware competitors are included.
    wins_all = {}
    for pi, (crbcol, ttl) in enumerate([("crb_mean", "round-mean CRB"),
                                        ("crb_final_mean", "final-round CRB")]):
        ax = axes[pi]
        wins = {m: np.zeros(len(taus), int) for m in POOL}
        for ti, tau in enumerate(taus):
            for pt in present:
                sub = s[s.point == pt]; elig = sub[sub[crbcol] <= tau]
                if len(elig) == 0: continue
                wins[elig.sort_values("acc_mean", ascending=False).iloc[0].method][ti] += 1
        wins_all[crbcol] = wins
        ax.axvspan(0.07, 0.13, color="#2a9d5c", alpha=0.06, zorder=0)
        # draw non-hero methods first (thin), hero last (thick, on top)
        for m in [x for x in POOL if x != HEAD] + [HEAD]:
            if wins[m].sum() == 0:
                continue
            hl = (m == HEAD)
            ax.plot(taus, wins[m], "-o" if hl else "-", color=c(m), lw=3.4 if hl else 2.0,
                    ms=6 if hl else 0, zorder=9 if hl else 3, alpha=1 if hl else .85,
                    solid_capstyle="round")
        ax.set_title(f"({'ab'[pi]}) {ttl}", fontsize=8)
        ax.margins(x=0.02)
        ax.set_ylabel("# constrained-acc. wins")
    axes[-1].set_xlabel("sensing bar   CRB $\\leq \\tau$")
    # single shared legend below both panels — lists EVERY sensing-aware competitor.
    # methods with no constrained-accuracy win are shown muted with a "(0)" tag.
    def _wins(m): return sum(int(wins_all[cc][m].sum()) for cc in wins_all)
    handles = []
    for m in POOL:
        won = _wins(m) > 0
        lbl = d(m) + (" (prop.)" if m == HEAD else ("" if won else " (0)"))
        handles.append(Line2D([0], [0], color=c(m), lw=3.0 if m == HEAD else 2.0,
                              marker="o" if m == HEAD else None, ms=5,
                              alpha=1.0 if won else 0.4, label=lbl))
    # legend in the gap BETWEEN the two panels, 3 methods per row
    axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05),
                   ncol=3, fontsize=8, frameon=False, columnspacing=0.5,
                   handletextpad=0.25, handlelength=0.9, labelspacing=0.3)
    fig.tight_layout(pad=0.4, h_pad=1.2)
    bu.save(fig, os.path.join(FIG, "fig6_threshold"))


# ══════════════════════════════════════════════ FIG 7 — convergence dual-panel
def _roll(a, w=7):
    a = np.asarray(a, float); k = w // 2; o = np.empty_like(a)
    for i in range(len(a)): o[i] = np.nanmean(a[max(0, i-k):min(len(a), i+k+1)])
    return o


def fig_convergence():
    show = POOL                                 # every sensing-aware competitor
    rows = rc.load_rounds(MAIN, "base")
    # single-column: panels stacked vertically, shared round axis, legend below
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 5.0), sharex=True)
    for ax in (a1, a2): bu.floating_axes(ax); bu.soft_grid(ax, "both")
    drawn = []
    # draw hero last so it sits on top of the baseline bundle
    for m in [x for x in show if x != HEAD] + [HEAD]:
        sub = rows[rows.method == m]
        if sub.empty: continue
        drawn.append(m); hl = (m == HEAD)
        for metric, ax in [("test_acc", a1), ("sensing_logdet", a2)]:
            piv = sub.pivot(index="round", columns="seed", values=metric)
            sm = piv.copy()
            for col in piv.columns: sm[col] = _roll(piv[col].values)
            mu = sm.mean(1); sd = sm.std(1, ddof=1); x = mu.index.values
            ax.plot(x, mu.values, color=c(m), lw=3.6 if hl else 1.9, zorder=9 if hl else 3,
                    alpha=1 if hl else .78, solid_capstyle="round")
            if hl:  # only shade the hero band — 12 overlapping bands would be mud
                ax.fill_between(x, (mu-sd).values, (mu+sd).values, color=c(m),
                                alpha=.18, lw=0, zorder=6)
    a1.set_ylabel("test accuracy")
    a2.set_xlabel("communication round")
    a2.set_ylabel("sensing info.  $\\log\\det \\mathbf{J}(\\mathcal{S})$")
    bu.panel_tag(a1, "(a) accuracy", y=1.02)
    bu.panel_tag(a2, "(b) sensing information ($\\uparrow$ better)", y=1.02)
    order = [HEAD] + [m for m in drawn if m != HEAD]
    handles = [Line2D([0], [0], color=c(m), lw=3.0 if m == HEAD else 2.0,
                      label=d(m) + (" (prop.)" if m == HEAD else "")) for m in order]
    # legend in the gap BETWEEN the two panels, 3 methods per row
    a1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3,
              fontsize=8, frameon=False, columnspacing=0.5, handletextpad=0.25,
              handlelength=0.9, labelspacing=0.3)
    fig.tight_layout(pad=0.4, h_pad=1.2)
    bu.save(fig, os.path.join(FIG, "fig7_convergence"))


# ══════════════════════════════════════════════ FIG 8 — rank-distribution stacked bar
def fig_rank_dist():
    """Combined honesty figure: per-operating-point ACCURACY rank distribution as a stacked
    horizontal bar (rank 1 / 2 / 3 / 4+), sorted by mean rank, with the weight-free Pareto
    result (SCOUT-FL non-dominated in X% of configs, both CRB conventions) folded in as an
    annotation. Replaces the old rank heatmap and the Pareto-dominance matrix."""
    s = _agg("campaign", ["acc", "logdet"])
    s = s[s.method.isin(POOL)]
    present = [p for p in PT_ORDER if p in set(s.point)]
    methods = POOL
    cnt = {m: np.zeros(4) for m in methods}          # [rank1, rank2, rank3, rank4+]
    mrank = {m: [] for m in methods}
    npts = 0
    for pt in present:
        sub = s[(s.point == pt) & (s.method.isin(methods))].copy()
        if len(sub) < 2: continue
        sub = sub.sort_values("acc_mean", ascending=False).reset_index(drop=True)
        npts += 1
        for i, m in enumerate(sub.method):
            r = i + 1
            cnt[m][min(r, 4) - 1] += 1
            mrank[m].append(r)
    order = sorted([m for m in methods if mrank[m]], key=lambda m: np.mean(mrank[m]))

    # Pareto fold-in: SCOUT-FL non-dominated fraction under BOTH CRB conventions
    po = []
    for cc in ("crb_mean", "crb_final_mean"):
        M, rows, _ = _dominance(cc)
        po.append(float(np.nanmean(M[rows.index(HEAD)] == 0)) * 100)
    po_txt = f"{min(po):.0f}%" if abs(po[0] - po[1]) < 1 else f"{po[0]:.0f}%/{po[1]:.0f}%"

    shades = ["#12492e", "#3f8c62", "#84bd9c", "#cfe4d7"]   # dark = rank 1 (best)
    rlabels = ["rank 1", "rank 2", "rank 3", "rank 4+"]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    bu.floating_axes(ax); bu.soft_grid(ax, "x")
    ys = np.arange(len(order))[::-1]                  # best method at top
    for y, m in zip(ys, order):
        frac = cnt[m] / cnt[m].sum() * 100.0
        left = 0.0
        for b in range(4):
            ax.barh(y, frac[b], left=left, height=0.68, color=shades[b],
                    edgecolor="white", lw=0.8, zorder=4)
            if frac[b] >= 9:                          # in-segment percent labels where they fit
                ax.text(left + frac[b] / 2, y, f"{frac[b]:.0f}", ha="center", va="center",
                        fontsize=7.2, color="white" if b < 2 else bu.INK, zorder=6)
            left += frac[b]
    # highlight the hero row with a subtle outline
    yh = ys[order.index(HEAD)]
    ax.barh(yh, 100, height=0.82, facecolor="none", edgecolor=c(HEAD), lw=1.8, zorder=7)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{d(m)}  ({np.mean(mrank[m]):.1f})" for m in order], fontsize=8.6)
    for lb, m in zip(ax.get_yticklabels(), order):
        if m == HEAD: lb.set_fontweight("bold"); lb.set_color(c(HEAD))
    ax.set_xlim(0, 100); ax.set_xlabel(f"share of the {npts} operating points (%)")

    # title + folded-in Pareto line (top, no crossing arrows), legend below the title
    ax.text(0.0, 1.135, "Per-point ISAC-rank distribution (mean rank in parentheses)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2, fontweight="bold")
    ax.text(0.0, 1.075, f"SCOUT-FL leads without a clean sweep, and is Pareto-optimal in "
            f"{po_txt} of configs (both CRB conventions).",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.4,
            color=c(HEAD), style="italic")
    handles = [Patch(facecolor=shades[b], edgecolor="white", label=rlabels[b]) for b in range(4)]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4,
              fontsize=8.4, frameon=False, handletextpad=0.5, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.95), pad=0.5)
    bu.save(fig, os.path.join(FIG, "fig8_rank_dist"))


# ══════════════════════════════════════════════ FIG 9 — corrected CD diagram
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102}


def fig_cd():
    s = _agg("campaign", ["acc", "logdet"])
    methods = [HEAD, "collabsensefed", "sensing_native", "asaad", "crb_only", "sensing_only",
               "fixed_weighted", "ota_fl_iscc"]
    present = [p for p in PT_ORDER if p in set(s.point) and p != "C_sensing_targets=5"]
    ranks = {m: [] for m in methods}; used = 0
    for pt in present:
        sub = s[(s.point == pt) & (s.method.isin(methods))].copy()
        if len(sub) < len(methods): continue
        sub["score"] = 0.5*rc.minmax_norm(sub.acc_mean) + 0.5*rc.minmax_norm(sub.logdet_mean)
        sub = sub.sort_values("score", ascending=False).reset_index(drop=True)
        for i, m in enumerate(sub.method): ranks[m].append(i+1)
        used += 1
    avg = {m: float(np.mean(v)) for m, v in ranks.items()}
    k = len(methods); N = used
    CD = _Q05[k] * np.sqrt(k*(k+1)/(6.0*N))
    items = sorted(avg.items(), key=lambda kv: kv[1])
    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(8.4, 3.4)); ax.set_xlim(lo-0.5, hi+0.6); ax.set_ylim(0.0, 1.18); ax.axis("off")
    yax = 0.62
    ax.plot([lo, hi], [yax, yax], "k-", lw=1.1)
    for x in range(lo, hi+1):
        ax.plot([x, x], [yax, yax+0.03], "k-", lw=0.9); ax.text(x, yax+0.06, str(x), ha="center", fontsize=8)
    ax.text((lo+hi)/2, yax+0.15, "average rank (1 = best)", ha="center", fontsize=8.6, fontweight="bold")
    ybar = yax+0.22
    ax.plot([lo, lo+CD], [ybar, ybar], "k-", lw=2.0)
    for xx in (lo, lo+CD): ax.plot([xx, xx], [ybar-.02, ybar+.02], "k-", lw=1.0)
    ax.text(lo+CD/2, ybar+0.03, f"CD = {CD:.2f}", ha="center", fontsize=8.2)
    half = int(np.ceil(k/2)); step = 0.088
    for idx, (m, v) in enumerate(items):
        col = c(m); prop = (m == HEAD)
        if idx < half:
            yy = yax-0.09-idx*step
            ax.plot([v, v], [yax, yy], color=col, lw=1.4); ax.plot([v, lo-0.45], [yy, yy], color=col, lw=1.4)
            ax.text(lo-0.5, yy, d(m), ha="right", va="center", fontsize=8.2,
                    fontweight="bold" if prop else "normal", color=col if prop else "black")
        else:
            yy = yax-0.09-(k-1-idx)*step
            ax.plot([v, v], [yax, yy], color=col, lw=1.4); ax.plot([v, hi+0.45], [yy, yy], color=col, lw=1.4)
            ax.text(hi+0.5, yy, d(m), ha="left", va="center", fontsize=8.2,
                    fontweight="bold" if prop else "normal", color=col if prop else "black")
    vs = sorted(avg.values()); ybars = yax-0.04; usedq = []
    i = 0
    while i < len(vs):
        j = i
        while j+1 < len(vs) and vs[j+1]-vs[i] < CD: j += 1
        if j > i and not any(vs[i] >= a and vs[j] <= b for a, b in usedq):
            usedq.append((vs[i], vs[j]))
            ax.plot([vs[i]-0.04, vs[j]+0.04], [ybars, ybars], color="0.2", lw=3.0, solid_capstyle="round")
            ybars -= 0.028
        i += 1
    ax.text(0.5, 1.06, f"Corrected critical-difference diagram (N={N} points, illustrative)",
            transform=ax.transAxes, ha="center", fontsize=8.2, fontweight="bold")
    ax.text(0.5, 0.0, "Caveat: operating points share seeds and one system model (not independent) "
            "$\\Rightarrow$ CD is anti-conservative; read as illustrative, not confirmatory.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.8,
            bbox=dict(boxstyle="round", fc="#fff3cd", ec="#e0a800", lw=0.7))
    fig.tight_layout(pad=0.3)
    bu.save(fig, os.path.join(FIG, "fig9_cd"))


# ══════════════════════════════════════════════ FIG 10 — selection runtime
def fig_runtime():
    rows = rc.load_rounds(MAIN, "base")
    methods = [HEAD, "collabsensefed", "sensing_native", "asaad", "crb_only", "sensing_only",
               "fixed_weighted", "ota_fl_iscc"]
    # per-round selection wall-clock (select_time), averaged over rounds & seeds
    import glob
    st = {}
    for m in methods:
        vals = []
        for f in glob.glob(os.path.join(ROOT, "runs", MAIN, "base", f"{m}__seed*.json")):
            dd = json.load(open(f))
            if not dd.get("complete"): continue
            vals += [r.get("select_time", np.nan) for r in dd["rounds"]]
        vals = np.array([v for v in vals if v is not None and np.isfinite(v)])
        if len(vals): st[m] = (np.mean(vals)*1000, np.std(vals)*1000)
    order = sorted(st, key=lambda m: st[m][0])          # cheapest at bottom
    xmax = max(v[0] + v[1] for v in st.values())
    fig, ax = plt.subplots(figsize=(3.5, 2.5)); bu.floating_axes(ax, x=False); bu.soft_grid(ax, "x")
    ys = np.arange(len(order))[::-1]
    for y, m in zip(ys, order):
        mu, sd = st[m]; hl = (m == HEAD)
        ax.barh(y, mu, height=0.62, xerr=sd, color=c(m), edgecolor="white", lw=0.8,
                alpha=1.0 if hl else .8, error_kw=dict(ecolor="0.55", lw=0.9, capsize=2),
                zorder=6 if hl else 4)
        ax.annotate(f"{mu:.0f} ms", (mu + sd, y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, fontweight="bold" if hl else "normal", color=c(m))
    ax.set_yticks(ys); ax.set_yticklabels([d(m) for m in order], fontsize=8)
    for lb, m in zip(ax.get_yticklabels(), order):
        if m == HEAD: lb.set_fontweight("bold"); lb.set_color(c(HEAD))
    # speedup callout: SCOUT-FL vs the Asaad SOTA sensing reference (placed in open space)
    if HEAD in st and SOTA in st:
        ya = ys[order.index(SOTA)]; sp = st[SOTA][0] / st[HEAD][0]
        ax.annotate(f"SCOUT-FL:\n$\\approx{sp:.0f}\\times$ cheaper\nthan Asaad",
                    xy=(st[SOTA][0], ya + 0.32), xytext=(0.60, 0.55), textcoords="axes fraction",
                    ha="left", va="center", fontsize=8, color="#c0392b", style="italic",
                    linespacing=1.15, zorder=9,
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1,
                                    connectionstyle="arc3,rad=0.25", shrinkB=4))
    ax.set_xlabel("selection wall-clock per round (ms)")
    ax.set_xlim(0, xmax * 1.22)
    fig.tight_layout(pad=0.3)
    bu.save(fig, os.path.join(FIG, "fig10_runtime"))


# ══════════════════════════════════════════════ FIG 2 — angle-diversity via FIM ellipses
def _crb_ellipse(J, nsig=1.0):
    """Return (width, height, angle_deg) of the CRB (=inverse-FIM) 1-sigma ellipse."""
    cov = np.linalg.inv(J)
    w, V = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(V[1, np.argmax(w)], V[0, np.argmax(w)]))
    return 2 * nsig * np.sqrt(max(w)), 2 * nsig * np.sqrt(min(w)), ang


def fig_geometry_ellipse():
    """The thesis figure. Two clients are already selected from the SAME (western) bearing,
    so the target's east--west range is well pinned but its north--south position is not
    (a tall uncertainty ellipse). A redundant high-SNR client from the same bearing barely
    helps; a complementary medium-SNR client from an orthogonal bearing collapses the
    ellipse. Built from the real FIM code (scout_fl.sim.fim)."""
    target = np.array([[0.0, 0.0]])
    prior = 0.12 * np.eye(2)
    kr, ka = 1.0, 0.05
    sel = np.array([[-1.35, 0.42], [-1.35, -0.42]])       # two selected, WEST sector
    cand_A = np.array([[-1.62, 0.0]])                      # redundant: same sector, HIGH snr
    cand_B = np.array([[0.0, 1.5]])                        # complementary: NORTH, MED snr
    snr_sel, snrA, snrB = 10.0, 16.0, 8.0

    def fim_sum(clients, snr):
        geom = pairwise_geometry(clients, target)
        F = per_client_target_fim(geom, np.full((clients.shape[0], 1), snr), kr, ka)[:, 0]
        return F.sum(axis=0)

    J0 = prior + fim_sum(sel, snr_sel)
    JA = J0 + fim_sum(cand_A, snrA)
    JB = J0 + fim_sum(cand_B, snrB)

    w0, h0, a0 = _crb_ellipse(J0, nsig=1.5)
    ext = max(w0, h0) / 2 * 1.15
    ylim = max(ext, 1.75); xlim = max(ext, 2.05)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 5.0), sharex=True, sharey=True)
    RED = "#b04b3a"
    for ax, (Jnew, cand, snrc, tag, colr, keep_lbl, sub) in zip(
        (ax1, ax2),
        [(JA, cand_A, snrA, "(a) redundant client, same bearing", RED,
          "candidate:\nhigh SNR, redundant bearing", "high SNR cannot buy a new direction"),
         (JB, cand_B, snrB, "(b) diverse client, new bearing", c(HEAD),
          "candidate:\nmedium SNR, new bearing", "a new bearing collapses the ellipse")]):
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-xlim, xlim); ax.set_ylim(-ylim, ylim)

        # bearing lines to target (drawn first, under everything)
        for cl in sel:
            ax.plot([cl[0], 0], [cl[1], 0], color=bu.MUTE, lw=0.9, ls=(0, (1, 2)), alpha=.55, zorder=2)
        ax.annotate("", xy=(0, 0), xytext=tuple(cand[0]),
                    arrowprops=dict(arrowstyle="-", color=colr, lw=1.3, alpha=.7,
                                    ls=(0, (4, 2))), zorder=2)

        # "before" ellipse: filled light grey + dashed outline (identical in both panels)
        ax.add_patch(Ellipse((0, 0), w0, h0, angle=a0, facecolor=bu.MUTE, alpha=.14,
                             edgecolor="none", zorder=3))
        ax.add_patch(Ellipse((0, 0), w0, h0, angle=a0, facecolor="none",
                             edgecolor=bu.MUTE, lw=1.3, ls=(0, (5, 3)), alpha=.8, zorder=4))
        # "after" ellipse: filled colour, solid outline, on top
        wn, hn, an = _crb_ellipse(Jnew, nsig=1.5)
        ax.add_patch(Ellipse((0, 0), wn, hn, angle=an, facecolor=colr, alpha=.30,
                             edgecolor=colr, lw=2.2, zorder=5))

        shrink = (1 - np.sqrt(np.linalg.det(np.linalg.inv(Jnew)) /
                              np.linalg.det(np.linalg.inv(J0)))) * 100

        # markers
        ax.scatter(sel[:, 0], sel[:, 1], s=88, marker="D", color=bu.MUTE,
                   edgecolor="white", lw=1.0, zorder=7)
        ax.scatter(*cand[0], s=175, marker="D", color=colr, edgecolor="white", lw=1.3, zorder=8)
        ax.scatter(0, 0, s=250, marker="*", color="#c0392b", edgecolor="white", lw=1.3, zorder=9)

        # labels — placed clear of markers and ellipse
        ax.annotate("target", (0, 0), xytext=(12, 10), textcoords="offset points",
                    ha="left", fontsize=8.8, color=bu.INK, fontweight="semibold")
        ax.annotate("2 selected\n(same bearing)", sel[-1], xytext=(-4, -26),
                    textcoords="offset points", ha="center", va="top", fontsize=7.8,
                    color=bu.MUTE, linespacing=1.1)
        top = cand[0][1] > 0.4
        clx = 14 if cand[0][0] >= 0 else -12
        cla = "left" if cand[0][0] >= 0 else "right"
        cdy, cva = (-20, "top") if top else (-6, "center")   # below the diamond if it is high up
        ax.annotate(keep_lbl, cand[0], xytext=(clx, cdy), textcoords="offset points",
                    ha=cla, va=cva, fontsize=8.0, color=colr, fontweight="semibold",
                    linespacing=1.15)

        ax.set_title(tag, fontsize=8.6, pad=4)
        ax.text(0.5, -0.015, f"uncertainty area  $-${shrink:.0f}%",
                transform=ax.transAxes, ha="center", va="top", fontsize=11.5,
                color=colr, fontweight="bold")
        ax.text(0.5, -0.085, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=8.2, color=bu.MUTE, style="italic")

    # shared reference legend (top, no overlap with panels)
    fig.legend(handles=[
        Ellipse((0, 0), 1, 1, facecolor="none", edgecolor=bu.MUTE, ls=(0, (5, 3)), lw=1.3),
        Line2D([0], [0], marker="D", ls="", mfc=bu.MUTE, mec="w", ms=8),
        Line2D([0], [0], marker="*", ls="", mfc="#c0392b", mec="w", ms=12)],
        labels=["uncertainty before (both panels)", "already-selected client", "target"],
        loc="upper center", ncol=3, fontsize=8.2, frameon=False, bbox_to_anchor=(0.5, 1.005),
        handletextpad=0.5, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94), pad=0.6)
    bu.save(fig, os.path.join(FIG, "fig2_geometry_ellipse"), box=True)


# ══════════════════════════════════════════════ FIG 11 — localization RMSE vs CRB vs SNR
def fig_rmse_crb():
    """Empirical localization RMSE tracks the CRB floor across SNR: the CRB gains are real
    localization gains. Monte-Carlo: for a fixed 3-client geometry, simulate range/angle
    measurements at each SNR, solve the linearized WLS estimator, compare RMSE to sqrt(CRB)."""
    rng = np.random.default_rng(0)
    target = np.array([[0.0, 0.0]])
    clients = np.array([[-1.0, 0.2], [0.3, 1.1], [0.9, -0.7]])
    geom = pairwise_geometry(clients, target)
    snr_db = np.arange(-10, 21, 2.5)
    kr, ka = 1.0, 0.05
    crb_curve, rmse_curve = [], []
    for sdb in snr_db:
        snr = 10 ** (sdb / 10.0)
        J = per_client_target_fim(geom, np.full((clients.shape[0], 1), snr), kr, ka)[:, 0].sum(0)
        J = J + 1e-6 * np.eye(2)
        crb = np.sqrt(np.trace(np.linalg.inv(J)))
        crb_curve.append(crb)
        # Monte-Carlo estimator: each client gives a noisy position measurement with
        # covariance = inverse of its own FIM; fuse by information averaging (BLUE).
        # 20k trials keep the sampling error of the empirical RMSE below ~0.5%, so the
        # curve sits on the bound instead of fluctuating visibly around it.
        trials = 20000
        Ji = per_client_target_fim(geom, np.full((clients.shape[0], 1), snr), kr, ka)[:, 0]
        info = np.zeros((2, 2)); info_x = np.zeros((trials, 2))
        for k in range(clients.shape[0]):
            Jk = Ji[k] + 1e-6 * np.eye(2)
            z = target[0] + rng.multivariate_normal([0, 0], np.linalg.inv(Jk), size=trials)
            info += Jk; info_x += z @ Jk.T
        xhat = np.linalg.solve(info, info_x.T).T
        rmse_curve.append(np.sqrt(np.mean(np.sum((xhat - target[0]) ** 2, axis=1))))
    fig, ax = plt.subplots(figsize=(3.5, 2.4)); bu.floating_axes(ax); bu.soft_grid(ax, "both")
    ax.plot(snr_db, crb_curve, color=bu.INK, lw=1.4, ls="--", label="$\\sqrt{\\mathrm{CRB}}$ floor", zorder=5)
    ax.plot(snr_db, rmse_curve, "o", color=c(HEAD), ms=4.5, mec="white", mew=0.6,
            label="empirical RMSE", zorder=6)
    ax.set_yscale("log"); ax.set_xlabel("sensing SNR (dB)")
    ax.set_ylabel("localization error (log)")
    ax.legend(loc="upper right", fontsize=8, handletextpad=0.4, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    bu.save(fig, os.path.join(FIG, "fig11_rmse_crb"))


# ══════════════════════════════════════════════ FIG 9 — lambda trade-off sweep
def fig_lambda():
    """How the trade-off weight lambda positions SCOUT-FL on the accuracy--CRB frontier.
    Reads runs_lambda/lam_<val>/base/scout_v2 (5 seeds each), plots final accuracy (left)
    and final-round CRB (right) vs lambda; marks the operating-point lambda=1."""
    import glob as _glob
    # literal strings so the tag matches the run dirs (lam_0p0, lam_1p0, ... not lam_0/lam_1)
    lam_strs = ["0.0", "0.25", "0.5", "1.0", "2.0", "4.0", "8.0"]
    lams = [float(s) for s in lam_strs]
    xs, accs, accs_sd, crbs = [], [], [], []
    for lam, ls in zip(lams, lam_strs):
        tag = "lam_" + ls.replace(".", "p")
        fs = _glob.glob(os.path.join(ROOT, "runs_lambda", tag, "base", "scout_v2__seed*.json"))
        fa, fc = [], []
        for f in fs:
            dd = json.load(open(f))
            if not dd.get("complete"):
                continue
            fa.append(dd["objectives"]["acc"]); fc.append(dd["objectives"]["crb_final"])
        if fa:
            xs.append(max(lam, 0.06)); accs.append(np.mean(fa) * 100)
            accs_sd.append(np.std(fa) * 100); crbs.append(np.mean(fc))
    if len(xs) < 2:
        print("  (fig_lambda: not enough lambda points yet — skipped)"); return
    xs = np.array(xs); accs = np.array(accs); accs_sd = np.array(accs_sd); crbs = np.array(crbs)

    fig, axL = plt.subplots(figsize=(3.5, 2.5))   # \columnwidth print size, 8pt
    bu.floating_axes(axL); bu.soft_grid(axL, "y")
    cA, cC = c(HEAD), "#c0392b"
    axL.plot(xs, accs, "-o", color=cA, lw=2.0, ms=4.5, zorder=6, label="accuracy")
    axL.fill_between(xs, accs - accs_sd, accs + accs_sd, color=cA, alpha=.15, lw=0, zorder=3)
    axL.set_xscale("log")
    axL.set_xlabel("sensing weight  $\\lambda$  (log;  $\\lambda{=}0$ at left)")
    axL.set_ylabel("final test accuracy (%)", color=cA)
    axL.tick_params(axis="y", labelcolor=cA)

    axR = axL.twinx()
    for s in ("top", "right", "left", "bottom"):
        axR.spines[s].set_visible(True); axR.spines[s].set_color(bu.AXIS); axR.spines[s].set_linewidth(1.1)
    axR.plot(xs, crbs, "--s", color=cC, lw=1.8, ms=4, zorder=6, label="final-round CRB")
    axR.set_ylabel("final-round CRB  ($\\downarrow$ better)", color=cC)
    axR.tick_params(axis="y", labelcolor=cC, colors=bu.AXIS)

    # mark the operating point lambda = 1
    axL.axvline(1.0, color=bu.MUTE, lw=1.0, ls=(0, (2, 2)), zorder=2)
    axL.annotate("operating point $\\lambda{=}1$", (1.0, axL.get_ylim()[0]),
                 xytext=(-8, 4), textcoords="offset points", ha="right", va="bottom",
                 fontsize=8, color=bu.INK, fontweight="semibold", rotation=90)
    # xticks as lambda labels (0 at the clamped-left position)
    axL.set_xticks(xs)
    axL.set_xticklabels(["0" if l < 0.1 else f"{l:g}" for l in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0][:len(xs)]])
    axL.xaxis.set_minor_locator(plt.NullLocator())
    lines = axL.get_lines()[:1] + axR.get_lines()[:1]
    axL.legend(lines, [ln.get_label() for ln in lines], loc="lower center",
               bbox_to_anchor=(0.5, 1.01), fontsize=8, ncol=2, frameon=False,
               columnspacing=1.0, handletextpad=0.4, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    bu.save(fig, os.path.join(FIG, "fig9_lambda"))


# ══════════════════════════════════════════════ FIG — accuracy-gap distribution + Pareto %
def _gap_pareto_data():
    """Per (method, point): accuracy gap (pp) to the best sensing-aware method at that
    point, plus per-method non-dominated fraction under both CRB conventions."""
    s = _agg("campaign", ["acc", "crb", "crb_final"])
    s = s[s.method.isin(POOL)]
    present = [p for p in PT_ORDER if p in set(s.point)]
    gaps = {m: [] for m in POOL}
    for pt in present:
        sub = s[s.point == pt]
        acc = dict(zip(sub.method, sub.acc_mean)); best = max(acc.values())
        for m in POOL:
            gaps[m].append((best - acc[m]) * 100.0 if m in acc else np.nan)
    nd = {}          # nd[crbcol][method] = list of (bool non-dominated, [dominators])
    for crbcol in ("crb_mean", "crb_final_mean"):
        nd[crbcol] = {m: [] for m in POOL}
        for pt in present:
            sub = s[s.point == pt]
            acc = dict(zip(sub.method, sub.acc_mean)); crb = dict(zip(sub.method, sub[crbcol]))
            for m in POOL:
                doms = [o for o in acc if o != m and acc[o] > acc[m] and crb[o] < crb[m]]
                nd[crbcol][m].append((len(doms) == 0, doms))
    return present, gaps, nd


def fig_gap_dist():
    present, gaps, nd = _gap_pareto_data()
    med = {m: float(np.nanmedian(gaps[m])) for m in POOL}
    order = sorted(POOL, key=lambda m: med[m])
    pf = {m: 100.0 * np.mean([ok for ok, _ in nd["crb_final_mean"][m]]) for m in POOL}
    pm = {m: 100.0 * np.mean([ok for ok, _ in nd["crb_mean"][m]]) for m in POOL}
    conv_delta = max(abs(pf[m] - pm[m]) for m in POOL)
    assert round(pf[HEAD]) == 96 and round(pm[HEAD]) == 96, \
        f"SCOUT-FL Pareto % mismatch: final {pf[HEAD]:.1f}, mean {pm[HEAD]:.1f}"

    CAP = 12.0
    def mc(m): return c(m)                 # every method wears its own campaign colour
    fig, ax = plt.subplots(figsize=(3.5, 3.5))    # \columnwidth, 8pt
    bu.floating_axes(ax); bu.soft_grid(ax, "y")
    ax.axhline(0, color=bu.MUTE, lw=1.0, ls="--", zorder=2)
    rng = np.random.default_rng(3)
    for i, m in enumerate(order):
        v = np.asarray([g for g in gaps[m] if np.isfinite(g)])
        hl = (m == HEAD)
        top = 0.0
        if med[m] <= CAP:
            bp = ax.boxplot([v], positions=[i], widths=0.5, whis=1.5, showfliers=False,
                            patch_artist=True, zorder=7 if hl else 5)
            from matplotlib.colors import to_rgba as _rgba
            for box in bp["boxes"]:
                box.set(facecolor=_rgba(mc(m), 0.30 if hl else 0.18),
                        edgecolor=mc(m), lw=2.0 if hl else 1.5)
            for el in bp["whiskers"] + bp["caps"]:
                el.set(color=mc(m), lw=1.7 if hl else 1.3)
            for mln in bp["medians"]:
                mln.set(color=mc(m), lw=2.6 if hl else 2.0)
            top = max(w.get_ydata().max() for w in bp["whiskers"])
        else:                 # off-scale method: dots only, median printed above the cap
            xtra = f" $\\cdot$ {pf[m]:.0f}%" if round(pf[m]) > 0 else ""
            ax.annotate(f"median {med[m]:.1f} $\\uparrow$" + xtra, (i, CAP - 0.35),
                        ha="center", va="top", fontsize=8, color=mc(m), rotation=90,
                        fontweight="semibold")
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=7, color=mc(m),
                   alpha=0.6 if hl else 0.45, lw=0, zorder=8 if hl else 6, clip_on=True)
        # Pareto-optimal share above the top whisker (final-round CRB); 0% left unlabelled
        if round(pf[m]) > 0 and med[m] <= CAP:
            # rotated like the median tags, so adjacent labels can never collide;
            # for boxes near the cap, hang the label beside the whisker instead
            if top + 2.0 < CAP:
                ax.annotate(f"{pf[m]:.0f}%", (i, top + 0.35), ha="center", va="bottom",
                            fontsize=8, color=mc(m), rotation=90,
                            fontweight="bold" if hl else "semibold")
            else:
                ax.annotate(f"{pf[m]:.0f}%", (i - 0.40, CAP - 0.25), ha="center", va="top",
                            fontsize=8, color=mc(m), rotation=90,
                            fontweight="bold" if hl else "semibold")
        # (off-scale methods carry their Pareto share inside the median tag above)
    ax.annotate("labels: Pareto-optimal\nshare (final-round CRB)", (0.98, 0.50),
                xycoords="axes fraction", ha="right", va="top", fontsize=8,
                color=bu.MUTE, style="italic", linespacing=1.2)
    ax.set_ylim(-0.4, CAP); ax.set_xlim(-0.65, len(order) - 0.35)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([d(m) for m in order], rotation=90, fontsize=8)
    for lb, m in zip(ax.get_xticklabels(), order):
        if m == HEAD: lb.set_fontweight("bold"); lb.set_color(c(HEAD))
    ax.set_ylabel("accuracy gap to best sensing-aware method (pp)")
    fig.tight_layout(pad=0.5)
    bu.save(fig, os.path.join(FIG, "fig_gap_dist"))

    # audit-trail stats + appendix table body
    os.makedirs(os.path.join(FIG, "stats"), exist_ok=True)
    json.dump({"points": present,
               "median_gap_pp": med,
               "pareto_pct_final": pf, "pareto_pct_mean": pm,
               "convention_max_delta_pp": conv_delta,
               "scout_dominators_final": {p: dd for p, (ok, dd) in
                                          zip(present, nd["crb_final_mean"][HEAD]) if not ok},
               "scout_dominators_mean": {p: dd for p, (ok, dd) in
                                         zip(present, nd["crb_mean"][HEAD]) if not ok}},
              open(os.path.join(FIG, "stats", "gap_pareto.json"), "w"), indent=1)
    grp = {"A_datasets": "Dataset", "A_learning": "Partition",
           "B_wireless": "Channel / SNR", "C_sensing": "Target geometry"}
    lines, last = [], None
    nf = nm = 0
    for j, pt in enumerate(present):
        g = next(v for k, v in grp.items() if pt.startswith(k))
        if g != last:
            lines.append(f"\\midrule\n\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{g}}}}}\\\\")
            last = g
        okf, df_ = nd["crb_final_mean"][HEAD][j]; okm, dm_ = nd["crb_mean"][HEAD][j]
        nf += okf; nm += okm
        def cell(ok, doms):
            return "no" if ok else "yes (" + ", ".join(d(x) for x in doms) + ")"
        lab = PT_LABEL[pt].replace("$-$", "$-$")
        lines.append(f"{lab} & {cell(okf, df_)} & {cell(okm, dm_)} \\\\")
    lines.append("\\midrule")
    lines.append(f"\\textbf{{Non-dominated}} & \\textbf{{{nf}/{len(present)} "
                 f"({100*nf//len(present)}\\%)}} & \\textbf{{{nm}/{len(present)} "
                 f"({100*nm//len(present)}\\%)}} \\\\")
    with open(os.path.join(FIG, "stats", "gap_pareto_table_body.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  fig_gap_dist: SCOUT-FL Pareto {pf[HEAD]:.0f}% (final) / {pm[HEAD]:.0f}% (mean); "
          f"field-wide convention delta ±{conv_delta:.0f} pp; table body written")


def main():
    print("Rendering SCOUT-FL paper figures (sensing-aware only, no JEDI):")
    fig_geometry_ellipse()
    fig_frontier(); fig_paired_delta(); fig_threshold()
    fig_convergence(); fig_runtime(); fig_rmse_crb(); fig_lambda()
    fig_gap_dist()
    fig_power_sweep()          # Fig. 14 — transmit-power / SINR sweep (needs the campaign re-run)
    print("done ->", FIG)


if __name__ == "__main__":
    main()


# ══════════════════════════════════ TCCN FIG 12 — cognitive adaptation (E-R1)
def _load_tccn_rows(tag, method):
    """Per-seed round DataFrames for a runs_tccn experiment; [] if absent."""
    import glob as _glob
    out = []
    for f in sorted(_glob.glob(os.path.join(ROOT, "runs_tccn", tag, "base", f"{method}__seed*.json"))):
        dd = json.load(open(f))
        if dd.get("complete"):
            out.append(dd["rounds"])
    return out


def fig_adapt_sweep():
    """Paper Fig. 12 (TCCN E-R1 + E-R2 combined): 2x2 panels sharing one legend.
    (a) dual price and (b) realised MSE under the mid-run budget cut.
    (c) realised MSE and (d) accuracy across the static budget sweep."""
    import glob as _glob
    runs_v2 = _load_tccn_rows("adapt_eps", HEAD)
    runs_hg = _load_tccn_rows("adapt_eps", ABL)
    if not runs_v2:
        print("  (fig_adapt_sweep: runs_tccn/adapt_eps not found - skipped)"); return

    def stack(runs, key):
        n = min(len(r) for r in runs)
        return np.array([[r[t].get(key, np.nan) for t in range(n)] for r in runs], float)

    mu = stack(runs_v2, "dual_mse"); mse_v2 = stack(runs_v2, "agg_mse")
    eps = stack(runs_v2, "mse_eps")[0]
    x = np.arange(mu.shape[1])
    t_cut = int(np.argmax(np.diff(eps) != 0) + 1) if np.any(np.diff(eps) != 0) else None

    rows = []
    for d_ in sorted(_glob.glob(os.path.join(ROOT, "runs_tccn", "eps_*"))):
        try:
            e = float(os.path.basename(d_)[4:])
        except ValueError:
            continue
        for m in (HEAD, ABL):
            runs = _load_tccn_rows(os.path.basename(d_), m)
            if not runs: continue
            accs = [r[-1]["test_acc"] for r in runs]
            mses = [np.mean([rr["agg_mse"] for rr in r]) for r in runs]
            rows.append((e, m, np.mean(accs) * 100, np.std(accs, ddof=1) * 100,
                         np.mean(mses)))

    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.9))
    (a1, a2), (a3, a4) = axes
    for ax in axes.ravel(): bu.floating_axes(ax); bu.soft_grid(ax, "both")

    # (a)/(b): adaptation over rounds
    for ax in (a1, a2):
        ax.set_xticks([0, 75, 150]); ax.set_xlabel("round")
        if t_cut:
            ax.axvline(t_cut, color="#c0392b", lw=0.9, ls=(0, (2, 2)), zorder=2)
    a1.plot(x, mu.mean(0), color=c(HEAD), lw=1.8, zorder=6)
    a1.fill_between(x, mu.mean(0) - mu.std(0, ddof=1), mu.mean(0) + mu.std(0, ddof=1),
                    color=c(HEAD), alpha=.18, lw=0, zorder=4)
    a1.set_ylabel("dual price  $\\mu_t$")
    a2.plot(x, mse_v2.mean(0), color=c(HEAD), lw=1.8, zorder=6, label="SCOUT-FL (soft)")
    if runs_hg:
        mse_hg = stack(runs_hg, "agg_mse")
        a2.plot(x[:mse_hg.shape[1]], mse_hg.mean(0), color=c(ABL), lw=1.2, zorder=5,
                label="SCOUT-FL$^{\\dagger}$ (hard)")
    a2.plot(x, eps, color=bu.INK, lw=1.1, ls="--", zorder=7, label="budget $\\varepsilon$")
    a2.set_yscale("log")
    a2.set_yticks([1.5e-4, 2e-4, 5e-4, 1e-3])
    a2.set_yticklabels(["1.5", "2", "5", "10"])
    a2.yaxis.set_minor_locator(plt.NullLocator())
    a2.set_ylabel("agg. MSE ($\\times 10^{-4}$)")

    # (c)/(d): budget sweep
    xt = [5e-5, 1e-4, 2e-4, 1e-3]; xl = ["0.5", "1", "2", "10"]
    for ax in (a3, a4):
        ax.set_xscale("log"); ax.set_xticks(xt); ax.set_xticklabels(xl)
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel("budget $\\varepsilon$ ($\\times 10^{-4}$)")
    if rows:
        e_all = sorted({r[0] for r in rows})
        a3.plot(e_all, e_all, color=bu.INK, lw=1.0, ls="--", zorder=3)
        for m, mk in ((HEAD, "o"), (ABL, "s")):
            pts = sorted(r for r in rows if r[1] == m)
            if not pts: continue
            e = [q[0] for q in pts]; hl = m == HEAD
            a3.plot(e, [q[4] for q in pts], f"-{mk}", color=c(m), lw=1.8 if hl else 1.2,
                    ms=4 if hl else 3.5, mec="white", mew=0.5, zorder=6 if hl else 5)
            a4.errorbar(e, [q[2] for q in pts], yerr=[q[3] for q in pts], fmt=f"-{mk}",
                        color=c(m), lw=1.8 if hl else 1.2, ms=4 if hl else 3.5,
                        mec="white", mew=0.5, capsize=1.5, elinewidth=0.9,
                        zorder=6 if hl else 5)
        a3.set_yscale("log")
        a3.set_yticks(xt); a3.set_yticklabels(xl)
        a3.yaxis.set_minor_locator(plt.NullLocator())
        a3.set_ylabel("realised MSE ($\\times 10^{-4}$)")
        a4.set_ylabel("final accuracy (%)")

    for ax, tag in zip(axes.ravel(), ["(a)", "(b)", "(c)", "(d)"]):
        bu.panel_tag(ax, tag, y=1.02)
    fig.legend(*a2.get_legend_handles_labels(), loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.008),
               handlelength=1.1, handletextpad=0.3, columnspacing=0.7)
    fig.tight_layout(rect=(0, 0.055, 1, 1), pad=0.35, w_pad=1.2, h_pad=1.3)
    bu.save(fig, os.path.join(FIG, "fig12_adapt_sweep"))

    os.makedirs(os.path.join(FIG, "stats"), exist_ok=True)
    post = slice(t_cut, None) if t_cut else slice(None)
    json.dump({"t_cut": t_cut, "seeds": len(runs_v2),
               "mu_peak": float(mu.mean(0).max()),
               "mse_post_mean": float(mse_v2[:, post].mean()),
               "eps_post": float(eps[-1]),
               "sweep": [{"eps": r[0], "method": r[1], "acc_mean": r[2],
                          "acc_std": r[3], "mse_mean": r[4]} for r in rows]},
              open(os.path.join(FIG, "stats", "adapt_sweep.json"), "w"), indent=1)
    print("  fig_adapt_sweep: stats/adapt_sweep.json written")


def fig_adaptation():
    """Kept for the run script: E-R1 and E-R2 now render as one combined figure."""
    fig_adapt_sweep()


def fig_eps_sweep():
    """Merged into fig_adapt_sweep (see fig_adaptation)."""
    pass


# ═══════════════════════════ FIG 14 — behaviour across the transmit-power / SINR range
_POWER_PTS = [0, -10, -15, -20, -25, -30, -35]          # physical.tx_power_dbm, high -> low
_POWER_METHODS = [HEAD, "collabsensefed", "sensing_native", SOTA, "crb_only", "sensing_only"]


def _power_point_stats(tx_dbm, method, tail=25):
    """(acc, crb_final, agg_mse, settled dual price) per seed at one transmit power.

    agg_mse and the dual price are read from the per-round rows so the *settled*
    behaviour is reported (mean over the last `tail` rounds), not the round-1 transient.
    """
    import glob as _glob
    pat = os.path.join(ROOT, "runs", "campaign", f"B_wireless_snr={tx_dbm}",
                       f"{method}__seed*.json")
    acc, crb, mse, dual = [], [], [], []
    for f in sorted(_glob.glob(pat)):
        d = json.load(open(f))
        if not d.get("complete"):
            continue
        o = d["objectives"]
        acc.append(o["acc"] * 100.0)
        crb.append(o.get("crb_final", o["crb"]))
        rows = d.get("rounds", [])[-tail:]
        if rows:
            mse.append(float(np.mean([r["agg_mse"] for r in rows])))
            dual.append(float(np.mean([r.get("dual_mse", 0.0) or 0.0 for r in rows])))
    return map(np.asarray, (acc, crb, mse, dual))


def fig_power_sweep(eps=1e-3):
    """Paper Fig. 14: accuracy, CRB, realised aggregation MSE and the settled dual
    price across the transmit-power sweep. Interference enters only through the
    effective noise floor, so the abscissa doubles as a receive-SINR axis."""
    series = {}
    for m in _POWER_METHODS:
        xs, A, C, M, D = [], [], [], [], []
        for p in _POWER_PTS:
            acc_v, crb_v, mse, du = _power_point_stats(p, m)
            if not len(acc_v):
                continue
            xs.append(p); A.append((acc_v.mean(), acc_v.std()))
            C.append(crb_v.mean()); M.append(mse.mean() if len(mse) else np.nan)
            D.append(du.mean() if len(du) else np.nan)
        if len(xs) >= 2:
            series[m] = (np.array(xs), np.array(A), np.array(C), np.array(M), np.array(D))
    if HEAD not in series:
        print("  (fig_power_sweep: no transmit-power runs yet — skipped)"); return

    fig, axes = plt.subplots(2, 2, figsize=(3.5, 4.15))   # \columnwidth print size, 8pt
    (a1, a2), (a3, a4) = axes
    for ax in axes.ravel():
        bu.floating_axes(ax); bu.soft_grid(ax, "y")

    for m, (xs, A, C, M, D) in series.items():
        hero = (m == HEAD)
        kw = dict(color=c(m), lw=2.2 if hero else 1.3, ms=4.2 if hero else 3.0,
                  zorder=6 if hero else 3, marker=bu.mark(m) if hasattr(bu, "mark") else "o")
        a1.plot(xs, A[:, 0], "-", **kw)
        if hero:
            a1.fill_between(xs, A[:, 0] - A[:, 1], A[:, 0] + A[:, 1],
                            color=c(m), alpha=.15, lw=0, zorder=2)
        a2.plot(xs, C, "-", **kw)
        a3.plot(xs, M, "-", **kw)
        a4.plot(xs, D, "-", **kw)

    a3.axhline(eps, ls="--", lw=1.0, color="#c0392b", zorder=2)
    a3.annotate(r"budget $\varepsilon$", xy=(min(_POWER_PTS) + 1.0, eps), fontsize=7,
                color="#c0392b", va="top", ha="left", xytext=(0, -2),
                textcoords="offset points")
    # shade the powers at which SCOUT-FL's realised error reaches the budget
    xs, _, _, M, _ = series[HEAD]
    binding = xs[M >= eps * 0.95]
    if len(binding):
        for ax in (a1, a2, a3, a4):
            ax.axvspan(xs.min() - 1, binding.max() + 1, color="#c0392b", alpha=.05, lw=0, zorder=1)

    a3.set_yscale("log")
    a1.set_ylabel("final accuracy (\\%)"); a2.set_ylabel("final-round CRB")
    a3.set_ylabel("realised aggregation MSE"); a4.set_ylabel("settled dual price $\\mu$")
    for ax in (a3, a4):
        ax.set_xlabel("transmit power (dBm)")
    for ax, t in zip(axes.ravel(), ["(a)", "(b)", "(c)", "(d)"]):
        bu.panel_tag(ax, t, y=1.02)

    handles = [Line2D([], [], color=c(m), lw=2.0, label=d(m)) for m in series]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.2, frameon=False,
               bbox_to_anchor=(0.5, -0.004), handlelength=1.1, handletextpad=0.3,
               columnspacing=0.7)
    fig.tight_layout(rect=(0, 0.115, 1, 1), pad=0.35, w_pad=1.2, h_pad=1.3)
    bu.save(fig, os.path.join(FIG, "fig14_power"))
    print("  fig14_power ->", FIG)

    stats = {m: {"tx_dbm": list(map(int, s[0])), "acc": [round(float(v), 2) for v in s[1][:, 0]],
                 "crb_final": [round(float(v), 5) for v in s[2]],
                 "agg_mse": [float(f"{v:.4g}") for v in s[3]],
                 "dual": [float(f"{v:.4g}") for v in s[4]]} for m, s in series.items()}
    os.makedirs(os.path.join(FIG, "stats"), exist_ok=True)
    json.dump(stats, open(os.path.join(FIG, "stats", "power_sweep.json"), "w"), indent=1)

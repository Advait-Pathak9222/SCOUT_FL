"""Generate all LaTeX tables for the preliminary report -> tables/*.tex"""
from __future__ import annotations
import os, numpy as np, pandas as pd
from scipy import stats
import report_common as rc

TAB = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(TAB, exist_ok=True)


def w(name, s):
    open(os.path.join(TAB, name), "w").write(s)
    print("wrote", name)


def esc(x):
    return str(x).replace("&", "\\&").replace("_", "\\_")


# =============================================================== main bake-off
def main_bakeoff():
    df = rc.load_tag("campaign_main")
    s = rc.agg_over_seeds(df)
    s["group"] = s.method.map(rc.group)
    s = s[s.group != "Ablation"].copy()
    s["acc_n"] = rc.minmax_norm(s.acc_mean)
    s["log_n"] = rc.minmax_norm(s.logdet_mean)
    s["isac"] = 0.5 * s.acc_n + 0.5 * s.log_n

    best_acc = s.acc_mean.max()
    best_log = s.logdet_mean.max()
    best_crb = s.crb_mean.min()
    best_isac = s.isac.max()

    lines = [
        r"\begin{tabular}{@{}l l c c c c c@{}}", r"\toprule",
        r"Method & Family & Acc.\ (\%) & log-det $\uparrow$ & CRB $\downarrow$ "
        r"& MSE ($\times10^{-4}$) $\downarrow$ & ISAC $\uparrow$ \\", r"\midrule",
    ]
    for grp in rc.GROUP_ORDER:
        gs = s[s.group == grp]
        if gs.empty:
            continue
        gs = gs.sort_values("isac", ascending=False)
        for _, r in gs.iterrows():
            prop = rc.is_proposed(r.method)
            name = esc(rc.disp(r.method))
            if prop:
                name = r"\textbf{" + name + "}"
            acc = f"{r.acc_mean*100:.1f}\\,\\footnotesize$\\pm${r.acc_std*100:.1f}"
            if abs(r.acc_mean - best_acc) < 1e-9:
                acc = r"\underline{" + acc + "}"
            logd = f"{r.logdet_mean:.2f}"
            if abs(r.logdet_mean - best_log) < 1e-9:
                logd = r"\underline{" + logd + "}"
            crb = f"{r.crb_mean:.3f}"
            if abs(r.crb_mean - best_crb) < 1e-9:
                crb = r"\underline{" + crb + "}"
            mse = f"{r.agg_mse_mean*1e4:.2f}"
            isac = f"{r.isac:.3f}"
            if abs(r.isac - best_isac) < 1e-9:
                isac = r"\textbf{" + isac + "}"
            row = f"{name} & {grp} & {acc} & {logd} & {crb} & {mse} & {isac} \\\\"
            lines.append(row)
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    w("tab_main_bakeoff.tex", "\n".join(lines))


# =============================================================== vs Asaad head-to-head (with sig)
def head_to_head():
    df = rc.load_tag("campaign_main")
    piv = df.pivot_table(index="seed", columns="method", values="acc")
    asaad = piv["asaad"]
    rows = []
    for m in ["jedi", "scout_v2", "scout_greedy"]:
        s = rc.agg_over_seeds(df)
        rr = s[s.method == m].iloc[0]
        ar = s[s.method == "asaad"].iloc[0]
        dacc = (rr.acc_mean - ar.acc_mean) * 100
        # paired test over shared seeds
        try:
            t, p = stats.ttest_rel(piv[m], asaad)
        except Exception:
            p = np.nan
        sig = "$^{***}$" if p < 0.01 else ("$^{**}$" if p < 0.05 else ("$^{*}$" if p < 0.1 else ""))
        rows.append((rc.disp(m), dacc, sig, rr.logdet_mean - ar.logdet_mean,
                     rr.crb_mean - ar.crb_mean))
    lines = [r"\begin{tabular}{@{}l c c c@{}}", r"\toprule",
             r"vs.\ Asaad~\cite{asaad2025} & $\Delta$Acc (pp) & $\Delta$log-det & $\Delta$CRB \\",
             r"\midrule"]
    for name, dacc, sig, dlog, dcrb in rows:
        lines.append(f"\\textbf{{{esc(name)}}} & ${dacc:+.1f}${sig} & ${dlog:+.2f}$ & ${dcrb:+.3f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("tab_head_to_head.tex", "\n".join(lines))


# =============================================================== robustness across datasets
def robustness_datasets():
    df = rc.load_tag("campaign")
    s = rc.agg_over_seeds(df)
    ds = s[s.point.str.startswith("A_datasets=")].copy()
    ds["d"] = ds.point.str.replace("A_datasets=", "")
    order = ["emnist", "fashion_mnist", "uci_har", "cifar10", "cifar100"]
    lab = {"emnist": "EMNIST", "fashion_mnist": "F-MNIST", "uci_har": "UCI-HAR",
           "cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
    methods = ["jedi", "scout_v2", "scout_greedy", "asaad", "divfl", "random", "oort"]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(order) + r"@{}}", r"\toprule",
             "Method & " + " & ".join(lab[d] for d in order) + r" \\", r"\midrule"]
    # find best per dataset among these methods
    best = {d: max(ds[(ds.d == d) & (ds.method.isin(methods))].acc_mean) for d in order}
    for m in methods:
        cells = []
        for d in order:
            row = ds[(ds.d == d) & (ds.method == m)]
            if row.empty:
                cells.append("--"); continue
            v = row.acc_mean.iloc[0] * 100
            c = f"{v:.1f}"
            if abs(row.acc_mean.iloc[0] - best[d]) < 1e-9:
                c = r"\textbf{" + c + "}"
            cells.append(c)
        name = rc.disp(m)
        if rc.is_proposed(m):
            name = r"\textbf{" + esc(name) + "}"
        else:
            name = esc(name)
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("tab_robustness_datasets.tex", "\n".join(lines))


# =============================================================== JEDI ablation
def jedi_ablation():
    df = rc.load_tag("ablation")
    s = rc.agg_over_seeds(df)
    order = ["jedi", "jedi_no_sensing", "jedi_no_learning", "jedi_no_coverage",
             "jedi_no_fairness", "jedi_no_externality", "jedi_no_kappa",
             "jedi_fixed_rho", "jedi_hard_gate"]
    lines = [r"\begin{tabular}{@{}l c c c@{}}", r"\toprule",
             r"Variant & Acc.\ (\%) & log-det $\uparrow$ & CRB $\downarrow$ \\", r"\midrule"]
    for m in order:
        row = s[s.method == m]
        if row.empty:
            continue
        r = row.iloc[0]
        name = esc(rc.disp(m))
        if m == "jedi":
            name = r"\textbf{" + name + " (full)}"
        lines.append(f"{name} & {r.acc_mean*100:.1f} & {r.logdet_mean:.2f} & {r.crb_mean:.3f} \\\\")
        if m == "jedi":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("tab_jedi_ablation.tex", "\n".join(lines))


# =============================================================== VISMAYA ablation
def vismaya_ablation():
    df = rc.load_tag("ablation_vismaya")
    s = rc.agg_over_seeds(df)
    order = ["vismaya", "vismaya_no_syn", "vismaya_sense_only", "vismaya_learn_only"]
    lines = [r"\begin{tabular}{@{}l c c c@{}}", r"\toprule",
             r"Variant (mobility, $\sigma_p{=}0.05$) & Acc.\ (\%) & log-det $\uparrow$ & CRB $\downarrow$ \\",
             r"\midrule"]
    for m in order:
        row = s[s.method == m]
        if row.empty:
            continue
        r = row.iloc[0]
        name = esc(rc.disp(m))
        if m == "vismaya":
            name = r"\textbf{" + name + " (full)}"
        lines.append(f"{name} & {r.acc_mean*100:.1f} & {r.logdet_mean:.2f} & {r.crb_mean:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("tab_vismaya_ablation.tex", "\n".join(lines))


# =============================================================== experiment status
def status_table():
    tags = {"ablation": ("JEDI component ablation", 30, 12),
            "ablation_vismaya": ("VISMAYA synergy ablation", 50, 6),
            "campaign_main": ("Main bake-off (CIFAR-10)", 150, 32),
            "campaign": ("OFAT robustness campaign", 150, 32)}
    lines = [r"\begin{tabular}{@{}l l c c c@{}}", r"\toprule",
             r"Stage & Description & Rounds & Units done & Status \\", r"\midrule"]
    # campaign expected: 22 points x 32 methods x 5 seeds
    for tag, (desc, rounds, nm) in tags.items():
        df = rc.load_tag(tag)
        done = len(df)
        pts = df.point.nunique()
        expected = (22 if tag == "campaign" else pts) * nm * 5
        status = f"{done}/{expected}"
        pct = 100 * done / expected
        st = r"\textbf{100\%}" if done >= expected else f"{pct:.1f}\\%"
        lines.append(f"{esc(tag)} & {esc(desc)} & {rounds} & {status} & {st} \\\\")
    lines += [r"\midrule",
              r"P7 regret & CUCB vs.\ offline oracle & 300 & 0/2 & \textit{pending} \\",
              r"\bottomrule", r"\end{tabular}"]
    w("tab_status.tex", "\n".join(lines))


# =============================================================== experimental setup
def setup_table():
    rows = [
        ("Clients / budget", r"$N=100$, $K=10$ ($K/N=10\%$)"),
        ("Rounds / seeds", r"$150$ (ablations 30--50) / $5$"),
        ("Datasets", "CIFAR-10/100, EMNIST, F-MNIST, UCI-HAR"),
        ("Models", "small-CNN (images), MLP (HAR)"),
        ("Non-IID", r"spatial Dirichlet, $\alpha\in\{0.1,0.3,0.5\}$"),
        ("Sensing targets", r"$M\in\{2,3,5\}$ (log-det FIM, CRB)"),
        ("Channel", r"Rician/Rayleigh, 3.5\,GHz, phys.\ link budget"),
        ("Uplink power", r"$-15$\,dBm nominal; SNR sweep $-35..0$\,dBm"),
        ("Aggregation", r"AirComp OTA-FedAvg, MSE budget $10^{-3}$"),
        ("Methods", r"4 proposed variants + 25 baselines"),
        ("Operating points", r"22 OFAT sweeps (learning/wireless/sensing)"),
    ]
    lines = [r"\begin{tabular}{@{}l l@{}}", r"\toprule",
             r"Setting & Value \\", r"\midrule"]
    for k, v in rows:
        lines.append(f"{esc(k)} & {v} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("tab_setup.tex", "\n".join(lines))


# =============================================================== publishability verdict
def verdict_table():
    import numpy as np
    dfm = rc.load_tag("campaign_main")
    sm = rc.agg_over_seeds(dfm)
    piv = dfm.pivot_table(index="seed", columns="method", values="acc")
    cs = rc.agg_over_seeds(rc.load_tag("campaign")); cs["group"] = cs.method.map(rc.group)
    cs = cs[cs.group != "Ablation"]
    jrank, cwin = {}, {}
    npts = cs.point.nunique()
    for pt, sub in cs.groupby("point"):
        sub = sub.copy()
        sub["acc_n"] = rc.minmax_norm(sub.acc_mean); sub["log_n"] = rc.minmax_norm(sub.logdet_mean)
        sub["score"] = 0.5 * sub.acc_n + 0.5 * sub.log_n
        order = sub.sort_values("score", ascending=False).reset_index(drop=True)
        for i, r in order.iterrows():
            jrank.setdefault(r.method, []).append(i + 1)
        ok = sub[sub.crb_mean <= 0.10]; ok = ok if len(ok) else sub
        cwin_m = ok.sort_values("acc_mean", ascending=False).iloc[0].method
        cwin[cwin_m] = cwin.get(cwin_m, 0) + 1
    asaad_acc = sm[sm.method == "asaad"].iloc[0].acc_mean
    verd = {
        "scout_v2": r"\cellcolor{green!14}\textbf{PUBLISH (headline)} --- balanced, Pareto-robust (both CRB)",
        "jedi": r"\cellcolor{yellow!18}\textbf{REFRAME} --- accuracy-max; CRB-convention fragile",
        "scout_greedy": r"\cellcolor{orange!16}DEMOTE to ablation (acc-tie vs strong baselines)",
        "asaad": r"SOTA ISAC baseline (reference)"}
    lines = [r"\begin{tabular}{@{}l c c c c >{\raggedright\arraybackslash}p{5.2cm}@{}}",
             r"\toprule",
             r"Method & Acc.\ (\%) & Mean joint & Constr.-acc & $\Delta$Acc vs & Verdict \\",
             r" & (C-10) & rank $\downarrow$ (/28) & wins /22 & Asaad (pp) & \\", r"\midrule"]
    for m in ["scout_v2", "jedi", "scout_greedy", "asaad"]:
        r = sm[sm.method == m].iloc[0]
        name = esc(rc.disp(m).replace(" (ours)", ""))
        if rc.is_proposed(m):
            name = r"\textbf{" + name + "}"
        mr = f"{np.mean(jrank[m]):.1f}"
        cw = str(cwin.get(m, 0))
        da = "" if m == "asaad" else f"${(r.acc_mean - asaad_acc) * 100:+.1f}$"
        lines.append(f"{name} & {r.acc_mean*100:.1f} & {mr} & {cw} & {da} & {verd[m]} \\\\")
        if m == "scout_greedy":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    w("tab_verdict.tex", "\n".join(lines))


if __name__ == "__main__":
    main_bakeoff()
    head_to_head()
    robustness_datasets()
    jedi_ablation()
    vismaya_ablation()
    status_table()
    setup_table()
    verdict_table()
    print("ALL TABLES DONE")

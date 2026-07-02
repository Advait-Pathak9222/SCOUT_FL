"""Auto-write analysis/decision_summary.md (design §3.3, §0.3) — computed from artifacts.

For each gate: the pre-registered criterion, the measured value with CI, PASS/FAIL.
Plus TEMPO E-T4 and CloakFL E-C3/E-C4 evidence tables and the §3.2 decision-rule
outcome. Missing/failed units are LISTED as missing — never imputed or hardcoded.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from scout_fl.analysis.gates import evaluate_all
from scout_fl.analysis.tc_load import load_units
from scout_fl.experiments import units as U

REPO = Path(__file__).resolve().parents[2]


def _fmt_gate(name, d):
    lines = [f"### {name}: {d.get('gate', name)}"]
    verdict = d.get("pass")
    tag = "PASS ✅" if verdict is True else ("FAIL ❌" if verdict is False else "PENDING ⏳")
    lines.append(f"- **Verdict:** {tag}")
    lines.append(f"- **Pre-registered criterion:** {d.get('criterion', '(see design doc)')}")
    if "measured" in d:
        lines.append(f"- **Measured:** {d['measured']:.4f} (threshold {d.get('threshold')})")
    if "measured_pp" in d and d["measured_pp"] is not None:
        ci = d.get("ci_pp")
        ci_s = f", 95% CI [{ci[0]:.2f}, {ci[1]:.2f}] pp" if ci else ""
        extra = (f" — schedule **{d.get('winning_schedule')}** vs static **{d.get('vs_static')}**"
                 f" (Wilcoxon p={d.get('wilcoxon_p'):.4g})") if d.get("winning_schedule") else ""
        lines.append(f"- **Measured:** {d['measured_pp']:.2f} pp accuracy advantage{ci_s}{extra}")
    for k in ("eavesdropper_inflation_1rx", "sensing_logdet_cost_frac", "framing",
              "retained_median", "note"):
        if k in d:
            lines.append(f"- {k}: {d[k]}")
    return "\n".join(lines)


def _dominance_table(controller_rows, static_rows):
    """Per-controller strict-dominance count over the static cloud (acc↑ & time_avg_trP↓)."""
    stat = {}
    for r in static_rows:
        stat.setdefault(r["method"], {"acc": [], "trP": []})
        stat[r["method"]]["acc"].append(r.get("acc"))
        stat[r["method"]]["trP"].append(r.get("time_avg_trP"))
    stat_means = {m: (np.nanmean(v["acc"]), np.nanmean(v["trP"])) for m, v in stat.items()
                  if any(x is not None for x in v["trP"])}
    ctrl = {}
    for r in controller_rows:
        ctrl.setdefault(r["method"], {"acc": [], "trP": []})
        ctrl[r["method"]]["acc"].append(r.get("acc"))
        ctrl[r["method"]]["trP"].append(r.get("time_avg_trP"))
    out = []
    for m, v in ctrl.items():
        ca, ct = np.nanmean(v["acc"]), np.nanmean(v["trP"])
        dom = sum(1 for (sa, st_) in stat_means.values() if ca >= sa and ct <= st_)
        out.append({"controller": m, "acc": float(ca), "time_avg_trP": float(ct),
                    "dominates_static": dom, "n_static": len(stat_means)})
    return sorted(out, key=lambda r: -r["dominates_static"])


def _tempo_section(runs_root, analytic_dir):
    rows = [u for u in load_units("tempo", runs_root) if (u["point"] or "").startswith("ET4_")]
    if not rows:
        return "## TEMPO-FL (E-T4)\n_No E-T4 bake-off units present (gate not passed, or not yet run)._\n"
    sf = Path(analytic_dir) / "tempo/static_frontier/static_frontier_per_seed.json"
    static_rows = []
    if sf.exists():
        for m, recs in json.loads(sf.read_text()).items():
            for r in recs:
                static_rows.append({"method": m, "acc": r["acc"], "time_avg_trP": r["time_avg_trP"]})
    ctrl = [r for r in rows if r["method"].startswith("tempo_")]
    tbl = _dominance_table(ctrl, static_rows) if static_rows else []
    out = ["## TEMPO-FL (E-T4 main bake-off)",
           f"_{len(rows)} E-T4 units; {len({r['method'] for r in rows})} policies; "
           f"final-round CRB primary; strict-dominance over the re-scored static cloud._\n",
           "| controller | mean acc | mean time-avg tr(P) | strict-dominates static |",
           "|---|---|---|---|"]
    for r in tbl:
        out.append(f"| {r['controller']} | {r['acc']:.3f} | {r['time_avg_trP']:.3f} | "
                   f"{r['dominates_static']}/{r['n_static']} |")
    if not tbl:
        out.append("| _static frontier not re-scored — dominance counts unavailable_ | | | |")
    return "\n".join(out) + "\n"


def _cloak_section(runs_root, analytic_dir):
    rows = [u for u in load_units("cloak", runs_root) if (u["point"] or "").startswith("EC3_")]
    lines = ["## CloakFL (E-C3 privacy–utility frontier, E-C4 measurement)"]
    if rows:
        # frontier: per (mode) the worst-client r and accuracy at each r_floor (mean over seeds)
        bykey = {}
        for r in rows:
            bykey.setdefault((r["cloak_mode"], r["r_floor"]), []).append(r)
        lines += ["", "| mode | r_floor (m) | worst-client r (m) | median r (m) | acc | eaves r (m) |",
                  "|---|---|---|---|---|---|"]
        for (mode, rf), rs in sorted(bykey.items()):
            lines.append(f"| {mode} | {rf} | {np.mean([x['leak_r_min'] for x in rs]):.2f} | "
                         f"{np.mean([x['leak_r_median'] for x in rs]):.2f} | "
                         f"{np.mean([x['acc'] for x in rs]):.3f} | "
                         f"{np.mean([x['eaves_r_median'] for x in rs]):.2f} |")
    else:
        lines.append("_No E-C3 frontier units present (gates not passed, or not yet run)._")
    ec4 = Path(analytic_dir) / "cloak/ec4/ec4_summary.json"
    if ec4.exists():
        d = json.loads(ec4.read_text())["per_method"]
        worst = list(d.items())[:5]
        lines += ["", "**E-C4 — every existing method localizes its clients** "
                  "(worst-exposed client, most-leaky first):", "",
                  "| method | worst-client r (m) | median r (m) |", "|---|---|---|"]
        for m, v in worst:
            lines.append(f"| {m} | {v['final_leak_r_worst_m']:.2f} | {v['final_leak_r_median_m']:.2f} |")
    return "\n".join(lines) + "\n"


def _decision_rule(gates):
    g1, g2, g3 = gates["GATE1"].get("pass"), gates["GATE2"].get("pass"), gates["GATE3"].get("pass")
    # tri-state: True / False / None(pending)
    tempo = g1
    cloak = True if (g2 is True and g3 is True) else (None if (g2 is None or g3 is None) else False)
    if tempo is None or cloak is None:
        def s(v, name):
            return f"{name} {'PASSED' if v is True else ('FAILED' if v is False else 'PENDING')}"
        return ("_Provisional (some gates not yet resolved): "
                + s(tempo, "TEMPO GATE 1") + "; "
                + s(cloak, "CloakFL GATES 2+3")
                + ". The §3.2 decision resolves once the pending Phase-1 units run._")
    tempo_ok, cloak_ok = tempo is True, cloak is True
    if tempo_ok and cloak_ok:
        return ("**Both gates pass → two papers** (design §3.2): TEMPO→TWC; CloakFL→TWC/TIFS "
                "per where T-C1 lands (constructive vs impossibility).")
    if tempo_ok or cloak_ok:
        winner = "TEMPO-FL" if tempo_ok else "CloakFL"
        loser = "CloakFL" if tempo_ok else "TEMPO-FL"
        return (f"**Only {winner} passes → full resources there** (design §3.2); {loser} gets a "
                f"1-page negative-result appendix (not silently discarded).")
    if g1 is None or g2 is None or g3 is None:
        return "_Gates pending — run the analytic + Phase-1 units to resolve the decision rule._"
    return ("**Neither gate passes** (design §3.2): both frontiers were real and immovable here → "
            "strengthens the GradEcho motivation; pivot documented.")


def _missing(cfg):
    all_u = U.enumerate_units(cfg)
    missing = [u["uid"] for u in all_u if not U.is_complete(u)]
    return all_u, missing


def build_summary(cfg, analytic_dir=None, runs_root=None, out_path=None):
    analytic_dir = analytic_dir or str(REPO / cfg["outputs_root"] / "analytic")
    runs_root = runs_root or cfg["runs_root"]
    gates = evaluate_all(analytic_dir, runs_root, cfg.get("gates", {}))
    all_u, missing = _missing(cfg)

    parts = [f"# TEMPO-FL / CloakFL — Decision Summary",
             f"_Auto-generated {datetime.now().isoformat(timespec='seconds')} from run artifacts. "
             f"Every value is computed; missing units are listed, never imputed._\n",
             "## Pre-registered gate verdicts (design §1.5, §2.6)",
             _fmt_gate("GATE 1", gates["GATE1"]), "",
             _fmt_gate("GATE 2", gates["GATE2"]), "",
             _fmt_gate("GATE 3", gates["GATE3"]), "",
             "## Decision rule (design §3.2)", _decision_rule(gates), "",
             _tempo_section(runs_root, analytic_dir),
             _cloak_section(runs_root, analytic_dir),
             "## Completeness",
             f"- Units enumerated: **{len(all_u)}**; complete: **{len(all_u) - len(missing)}**; "
             f"missing: **{len(missing)}**."]
    if missing:
        parts.append(f"- First missing (up to 20): " + ", ".join(f"`{m}`" for m in missing[:20]))
        parts.append("- Missing/failed units are reported as missing (design §0.3) — not imputed.")
    text = "\n".join(parts) + "\n"

    out_path = out_path or str(REPO / "analysis/decision_summary.md")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text)
    Path(str(REPO / "analysis/gate_verdicts.json")).write_text(json.dumps(gates, indent=2))
    print(f"[decision] wrote {out_path} ({len(all_u)-len(missing)}/{len(all_u)} units complete)")
    return gates


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = U.load_campaign_config(args.config)
    if args.smoke:
        cfg = U.apply_smoke(cfg)
    build_summary(cfg, out_path=args.out)


if __name__ == "__main__":
    main()

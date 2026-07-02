"""Pre-registered gate evaluation (design §1.5, §2.6, §3.2) — computed from artifacts.

GATE 1 (E-T1): at least one oracle schedule dominates the static frontier by
  >= gate1_dominance_pp accuracy at matched-or-better time-averaged tracking, paired
  95% CI excluding zero. Computed from the re-scored static frontier + E-T1 units.
GATE 2 (E-C1): read from the entanglement study verdict (median retained >= 0.50).
GATE 3 (E-C2): read from the dither study verdict (>=10x eavesdropper CRB inflation).

All verdicts are structured dicts (criterion, measured, threshold, pass) so the
decision-summary writer never hardcodes a number.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from scout_fl.analysis.tc_load import load_units

REPO = Path(__file__).resolve().parents[2]


def _paired_ci(diff, ci=0.95):
    d = np.asarray(diff, dtype=float)
    n = d.size
    mean = float(d.mean()) if n else float("nan")
    if n < 2:
        return mean, mean, mean, float("nan")
    sem = float(d.std(ddof=1)) / math.sqrt(n)
    half = float(stats.t.ppf(0.5 + ci / 2, df=n - 1)) * sem
    # Wilcoxon p (non-parametric primary, design §3.4)
    try:
        p = float(stats.wilcoxon(d)[1]) if np.any(d != 0) else float("nan")
    except ValueError:
        p = float("nan")
    return mean, mean - half, mean + half, p


def gate1(analytic_dir, runs_root=None, dominance_pp=1.0, sigma_p=0.0):
    """GATE 1 dominance test from the static frontier + E-T1 oracle schedules."""
    adir = Path(analytic_dir)
    sf_path = adir / "tempo/static_frontier/static_frontier_per_seed.json"
    verdict = {"gate": "GATE1_dominance",
               "criterion": f">=1 oracle schedule beats the static frontier by "
                            f">={dominance_pp:g} pp accuracy at matched-or-better time-avg tracking, "
                            f"paired 95% CI excludes 0",
               "threshold_pp": float(dominance_pp), "sigma_p": sigma_p}
    if not sf_path.exists():
        verdict.update(pass_=None, measured_pp=None, note="static frontier not re-scored yet")
        return verdict
    static = json.loads(sf_path.read_text())              # method -> [{acc,time_avg_trP,seed}]

    oracle = {}
    for u in load_units("tempo", runs_root):
        if not (u["point"] or "").startswith("ET1_"):
            continue
        if float(u.get("sigma_p") or 0.0) != sigma_p:
            continue
        oracle.setdefault(u["method"], []).append(u)

    if not oracle:                                         # E-T1 units not run yet -> PENDING, not FAIL
        verdict.update({"pass": None, "measured_pp": None,
                        "note": "no E-T1 oracle-schedule units present (run the train stage)"})
        return verdict

    best = None                                            # (mean_pp, schedule, static_method, ci)
    dom_counts = {}
    for sname, srows in oracle.items():
        sacc = {r["seed"]: r["acc"] for r in srows}
        strp = {r["seed"]: r.get("time_avg_trP") for r in srows}
        s_trp_mean = np.mean([v for v in strp.values() if v is not None]) if strp else np.inf
        dom = 0
        for mname, mrecs in static.items():
            macc = {r["seed"]: r["acc"] for r in mrecs}
            mtrp = {r["seed"]: r["time_avg_trP"] for r in mrecs}
            m_trp_mean = np.mean(list(mtrp.values()))
            common = sorted(set(sacc) & set(macc))
            if len(common) < 2:
                continue
            diff = np.array([sacc[s] - macc[s] for s in common])      # accuracy advantage (fraction)
            if s_trp_mean <= m_trp_mean and diff.mean() > 0:          # tracking-dominates & better acc
                dom += 1
            mean, lo, hi, p = _paired_ci(diff)
            mean_pp = mean * 100.0
            # candidate: matched-or-better tracking, >= threshold pp, CI excludes 0
            if s_trp_mean <= m_trp_mean and mean_pp >= dominance_pp and lo > 0:
                if best is None or mean_pp > best[0]:
                    best = (mean_pp, sname, mname, (lo * 100, hi * 100, p))
        dom_counts[sname] = dom

    if best is None:
        verdict.update(pass_=False, measured_pp=0.0, dominance_counts=dom_counts,
                       note="no schedule cleared the dominance bar")
    else:
        mp, sname, mname, (lo, hi, p) = best
        verdict.update(pass_=True, measured_pp=float(mp), winning_schedule=sname,
                       vs_static=mname, ci_pp=[float(lo), float(hi)], wilcoxon_p=float(p),
                       dominance_counts=dom_counts)
    verdict["pass"] = verdict.pop("pass_")
    return verdict


def gate2(analytic_dir):
    p = Path(analytic_dir) / "cloak/ec1/ec1_summary.json"
    if not p.exists():
        return {"gate": "GATE2_entanglement", "pass": None, "note": "E-C1 not run yet"}
    return json.loads(p.read_text())["gate2"]


def gate3(analytic_dir):
    p = Path(analytic_dir) / "cloak/ec2/ec2_summary.json"
    if not p.exists():
        return {"gate": "GATE3_dither", "pass": None, "note": "E-C2 not run yet"}
    return json.loads(p.read_text())["gate3"]


def evaluate_all(analytic_dir, runs_root=None, cfg_gates=None):
    cfg_gates = cfg_gates or {}
    v = {"GATE1": gate1(analytic_dir, runs_root,
                        dominance_pp=cfg_gates.get("gate1_dominance_pp", 1.0)),
         "GATE2": gate2(analytic_dir),
         "GATE3": gate3(analytic_dir)}
    return v


def write_gate_verdicts(analytic_dir, out_path, runs_root=None, cfg_gates=None):
    v = evaluate_all(analytic_dir, runs_root, cfg_gates)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(v, indent=2))
    for g, d in v.items():
        print(f"[gate] {g}: pass={d.get('pass')} "
              + (f"(measured {d.get('measured', d.get('measured_pp'))})" if d.get('pass') is not None else "(pending)"))
    return v


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--analytic-dir", default=str(REPO / "outputs/tempo_cloak/analytic"))
    ap.add_argument("--runs-root", default=str(REPO / "runs"))
    ap.add_argument("--out", default=str(REPO / "analysis/gate_verdicts.json"))
    args = ap.parse_args()
    write_gate_verdicts(args.analytic_dir, args.out, args.runs_root)


if __name__ == "__main__":
    main()

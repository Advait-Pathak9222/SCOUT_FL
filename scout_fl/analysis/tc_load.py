"""Load TEMPO / CloakFL run artifacts into tidy per-unit and per-round tables.

Same final-round-snapshot convention as scout_fl.analysis.collect and
report_common, but keeps ALL objectives keys (tempo: time_avg_trP, pct_violation,
track_rmse, ...; cloak: leak_r_median, leak_r_worst, eaves_r_median, r_floor, ...).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"


def load_units(tag: str, runs_root=None):
    """One dict per complete (point, method, seed) unit: meta + all objectives."""
    root = Path(runs_root) if runs_root else RUNS
    rows = []
    for f in glob.glob(os.path.join(root, tag, "*", "*.json")):
        try:
            d = json.load(open(f))
        except (ValueError, OSError):
            continue
        if not d.get("complete"):
            continue
        m = d.get("meta", {})
        o = d.get("objectives", {}) or {}
        row = {"tag": tag, "point": m.get("point"), "method": m.get("method"),
               "seed": m.get("seed"), "dataset": m.get("dataset"),
               "sigma_p": m.get("sigma_p"), "mission": m.get("mission"),
               "cloak_mode": m.get("cloak_mode"), "r_floor": m.get("r_floor"),
               "program": m.get("program")}
        row.update(o)
        rows.append(row)
    return rows


def load_rounds(tag: str, point: str, runs_root=None):
    """Per-round rows for a (tag, point) — used for trajectory/leakage-curve figures."""
    root = Path(runs_root) if runs_root else RUNS
    rows = []
    for f in glob.glob(os.path.join(root, tag, point, "*.json")):
        try:
            d = json.load(open(f))
        except (ValueError, OSError):
            continue
        if not d.get("complete"):
            continue
        m = d.get("meta", {})
        for r in d.get("rounds", []):
            rr = dict(r)
            rr.update({"method": m.get("method"), "seed": m.get("seed"), "point": point})
            rows.append(rr)
    return rows


def group_by_method(rows, keys):
    """{method -> {key -> [per-seed values]}} for the given objective keys (paired by seed)."""
    out = {}
    for r in rows:
        m = r["method"]
        out.setdefault(m, {"seed": []})
        out[m]["seed"].append(r.get("seed"))
        for k in keys:
            out[m].setdefault(k, []).append(r.get(k))
    return out


def paired_on_seeds(rows_a, rows_b, key):
    """Return (a_vals, b_vals) aligned on shared seeds for one objective key."""
    da = {r["seed"]: r.get(key) for r in rows_a}
    db = {r["seed"]: r.get(key) for r in rows_b}
    common = sorted(set(da) & set(db))
    a = np.array([da[s] for s in common], dtype=float)
    b = np.array([db[s] for s in common], dtype=float)
    return a, b, common

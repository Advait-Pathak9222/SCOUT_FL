"""Resumable per-unit run store: every (point, method, seed) federated training is
one JSON file under ``runs/`` that is rewritten after EACH round.

Why: the full campaign is ~1400+ 150-round trainings. If the process dies, a
re-run must continue from where it stopped — not recompute finished work. A unit
is the resumable atom: its JSON holds the per-round rows (for analysis: plots,
tables) plus a ``complete`` flag and final objectives. On resume, a unit whose
file exists and is ``complete`` is loaded and skipped; an incomplete unit is
recomputed from scratch (at most one partial 150-round training is ever redone).

Layout:  runs/<tag>/<point>/<method>__seed<seed>.json
Each file:  {"meta": {...}, "complete": bool, "rounds": [row, ...], "objectives": {...}|null}

Writes are atomic (tmp file + os.replace) so a crash mid-write never corrupts a unit.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np


def _safe(s) -> str:
    """Filesystem-safe token for tags/points/values."""
    return re.sub(r"[^A-Za-z0-9._=+-]", "_", str(s))


def unit_path(runs_root, tag: str, point: str, method: str, seed: int) -> Path:
    return Path(runs_root) / _safe(tag) / _safe(point) / f"{_safe(method)}__seed{int(seed)}.json"


def load_unit(path: Path):
    """Return the unit dict if present and complete, else None (missing/partial/corrupt)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except (ValueError, OSError):
        return None
    return d if d.get("complete") else None


def save_unit(path: Path, meta: dict, rows: list, complete: bool, objectives=None) -> None:
    """Atomically write the unit JSON (called after every round and at completion)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "complete": bool(complete), "rounds": rows,
               "objectives": objectives}
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=_json_default))
    os.replace(tmp, p)                                        # atomic on POSIX


def participation_from_rows(rows: list, K: int) -> np.ndarray:
    """Rebuild the per-client participation vector from saved per-round selections."""
    part = np.zeros(int(K))
    for r in rows:
        for k in r.get("selected", []):
            part[int(k)] += 1
    return part


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

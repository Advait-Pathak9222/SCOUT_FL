"""Regenerate the machine-readable schema inventory + replay-faithfulness check.

Emits analysis/schema_report.json (inventory of runs/ tags, method strings, the
per-round key set, and the replay verification result). The human-authored prose
lives in analysis/schema_report.md. Run in preflight so a schema drift (new keys,
renamed methods, broken determinism) fails loudly.

    python -m scout_fl.analysis.schema_report [--runs runs] [--out analysis/schema_report.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The 9 sensing-aggressive methods for E-C4's re-run fallback (design §2.6).
TOP9_LEAKY = ["asaad", "crb_only", "sensing_only", "sensing_native", "collabsensefed",
              "fed_iscc", "ota_fl_iscc", "iscc_air_feel", "fixed_weighted"]


def inventory(runs_root: str) -> dict:
    tags = {}
    for tag_dir in sorted(glob.glob(os.path.join(runs_root, "*"))):
        if not os.path.isdir(tag_dir):
            continue
        tag = os.path.basename(tag_dir)
        methods, points, seeds, complete, keyset = set(), set(), set(), 0, set()
        for f in glob.glob(os.path.join(tag_dir, "*", "*.json")):
            b = os.path.basename(f)
            if "__seed" not in b:
                continue
            m, s = b.rsplit("__seed", 1)
            methods.add(m)
            seeds.add(int(s.split(".")[0]))
            points.add(os.path.basename(os.path.dirname(f)))
            try:
                d = json.loads(Path(f).read_text())
            except (ValueError, OSError):
                continue
            if d.get("complete"):
                complete += 1
                if d.get("rounds"):
                    keyset.update(d["rounds"][0].keys())
        tags[tag] = {"n_units": complete, "points": sorted(points),
                     "methods": sorted(methods), "seeds": sorted(seeds),
                     "round_keys": sorted(keyset)}
    return tags


def replay_check(runs_root: str, base_config: str, point="A_datasets=cifar10",
                 n_methods=4, tol=1e-3) -> dict:
    """Verify replay reconstruction matches logged round-0 sensing log-det."""
    from scout_fl.infra import replay
    results, ok_all = [], True
    cfg = None
    try:
        cfg = replay.config_for_point(point, base_config)
    except Exception as e:                                    # noqa: BLE001
        return {"ok": False, "error": f"config_for_point failed: {e}", "checks": []}
    files = sorted(glob.glob(os.path.join(runs_root, "campaign", point, "*__seed0.json")))
    for f in files[:n_methods]:
        art = replay.load_artifact(f)
        if not art or not art.get("complete"):
            continue
        try:
            ok, rp, lg, diff = replay.verify_against_artifact(cfg, art, tol=tol)
        except Exception as e:                                # noqa: BLE001
            ok, rp, lg, diff = False, float("nan"), float("nan"), float("nan")
            results.append({"method": art["meta"]["method"], "error": str(e)})
            ok_all = False
            continue
        ok_all = ok_all and ok
        results.append({"method": art["meta"]["method"], "replay_logdet": rp,
                        "logged_logdet": lg, "abs_diff": diff, "ok": ok})
    return {"ok": bool(ok_all and results), "point": point, "tol": tol, "checks": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "runs"))
    ap.add_argument("--config", default=str(REPO / "scout_fl/configs/campaign_main.yaml"))
    ap.add_argument("--out", default=str(REPO / "analysis/schema_report.json"))
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if replay verification fails (preflight gate)")
    args = ap.parse_args()

    inv = inventory(args.runs)
    rep = replay_check(args.runs, args.config)
    payload = {"inventory": inv, "replay_verification": rep, "top9_leaky": TOP9_LEAKY}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[schema] wrote {args.out}")
    for tag, d in inv.items():
        print(f"  {tag:18s} units={d['n_units']:5d} points={len(d['points']):3d} "
              f"methods={len(d['methods']):3d} seeds={d['seeds']}")
    print(f"[schema] replay verification ok={rep['ok']} "
          f"({len(rep.get('checks', []))} methods checked, tol={rep.get('tol')})")
    for c in rep.get("checks", []):
        if "error" in c:
            print(f"    {c['method']:>16}: ERROR {c['error']}")
        else:
            print(f"    {c['method']:>16}: |diff|={c['abs_diff']:.2e} ok={c['ok']}")
    if args.strict and not rep["ok"]:
        raise SystemExit("[schema] STRICT: replay verification FAILED — E-C4/E-T2 re-scoring invalid")


if __name__ == "__main__":
    main()

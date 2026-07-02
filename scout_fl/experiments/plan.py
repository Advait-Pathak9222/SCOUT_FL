"""Planning CLI for run_all.sh: list/count/budget the unit grid, with filters.

    # uids for a stage (only the ones not yet complete -> resume):
    python -m scout_fl.experiments.plan --list --stage train --incomplete-only [--smoke]
    # counts:
    python -m scout_fl.experiments.plan --count [--smoke]
    # wall-clock budget extrapolated from the smoke stage:
    python -m scout_fl.experiments.plan --budget --per-unit-seconds 42 --num-gpus 4 --stage train
    # gate-based cancellation (strict mode): drop a failed gate's downstream experiments:
    python -m scout_fl.experiments.plan --list --stage train --exclude-experiments E-T3,E-T4,E-T5,E-T6

Every FL unit counts as 1 * rounds toward the budget; analytic units are ~free.
"""
from __future__ import annotations

import argparse
from collections import Counter

from scout_fl.experiments import units as U

# Which experiments belong to which gate's downstream (design §3.2 decision rules).
# If a gate FAILS in strict mode, run_all.sh excludes these.
GATE_DOWNSTREAM = {
    "GATE1": ["E-T3", "E-T4", "E-T5", "E-T6"],                # TEMPO Phase 2/3
    "GATE2": ["E-C3", "E-C5"],                                # CloakFL constructive frontier
    "GATE3": ["E-C3", "E-C5"],                                # (E-C4 runs regardless — design §2.6)
}


def _filter(units, stage=None, experiments=None, exclude=None, incomplete_only=False):
    out = []
    for u in units:
        if stage and u["stage"] != stage:
            continue
        if experiments and u["experiment"] not in experiments:
            continue
        if exclude and u["experiment"] in exclude:
            continue
        if incomplete_only and U.is_complete(u):
            continue
        out.append(u)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stage", default=None, choices=[None, "analytic", "train"])
    ap.add_argument("--experiments", default=None, help="comma list to INCLUDE")
    ap.add_argument("--exclude-experiments", default=None, help="comma list to EXCLUDE")
    ap.add_argument("--incomplete-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--budget", action="store_true")
    ap.add_argument("--per-unit-seconds", type=float, default=None,
                    help="measured seconds for a full-rounds unit (from smoke extrapolation)")
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    args = ap.parse_args()

    cfg = U.load_campaign_config(args.config)
    if args.smoke:
        cfg = U.apply_smoke(cfg)
    rounds = cfg["fl"]["rounds"]
    include = args.experiments.split(",") if args.experiments else None
    exclude = args.exclude_experiments.split(",") if args.exclude_experiments else None
    units = _filter(U.enumerate_units(cfg), args.stage, include, exclude, args.incomplete_only)

    if args.list:
        for u in units:
            print(u["uid"])
        return
    if args.count:
        print(f"units: {len(units)}")
        for e, c in sorted(Counter(u['experiment'] for u in units).items()):
            print(f"  {e:14s} {c}")
        return
    if args.budget:
        train = [u for u in units if u["stage"] == "train"]
        if args.per_unit_seconds is None:
            print("provide --per-unit-seconds (from smoke) for a wall-clock estimate")
            return
        # extrapolate: smoke ran ~5 rounds; per_unit_seconds is already scaled to full rounds by caller
        n_slots = max(1, args.num_gpus * args.workers_per_gpu)
        total_unit_seconds = len(train) * args.per_unit_seconds
        wall = total_unit_seconds / n_slots
        print(f"train units: {len(train)} | rounds/unit: {rounds}")
        print(f"per-unit (full rounds): {args.per_unit_seconds:.1f}s | slots: {n_slots}")
        print(f"estimated wall-clock: {wall/3600:.2f} h  ({wall/60:.0f} min)  [+ analytic ~free]")
        return
    # default: summary
    print(f"{len(units)} units (stage={args.stage or 'all'}, smoke={args.smoke})")


if __name__ == "__main__":
    main()

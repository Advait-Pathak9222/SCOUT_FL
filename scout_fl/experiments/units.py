"""Expand experiments/config.yaml into the flat list of experimental units.

A unit is the resumable atom: one (experiment x setting x method x seed) for FL
training, or one analytic study. Every unit has a STABLE uid -> a deterministic
artifact path, so `is_complete(unit)` lets run_all.sh skip finished work
(design: resumability). The grid structure IS the design doc (§1.5, §2.6);
config.yaml only sets scale (seed counts, rounds, dataset subsets, toggles).

Stages (cost order): analytic (near-zero GPU) < train (FL).
Each unit dict: {uid, stage, program, experiment, gate?, tag, point, method,
seed, params, artifact}.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from scout_fl.utils.runstore import unit_path, load_unit

REPO = Path(__file__).resolve().parents[2]


def load_campaign_config(path=None) -> dict:
    path = Path(path) if path else REPO / "experiments/config.yaml"
    return yaml.safe_load(Path(path).read_text())


def _seeds(cfg, key):
    return list(cfg["seeds"]["headline" if key == "headline" else "sweep"])


def _abs(root) -> Path:
    """Resolve a (possibly relative) root against the repo so artifact paths are
    CWD-independent."""
    p = Path(root)
    return p if p.is_absolute() else (REPO / p)


# --------------------------------------------------------------------------- #
# FL-unit helpers. runs_root / outputs_root come from cfg so the smoke stage
# writes to ISOLATED roots (runs_smoke/, outputs_smoke/) and never collides with
# a real 150-round artifact of the same (point, method, seed) — otherwise resume
# would skip real units as "complete".
def _fl_unit(runs_root, program, experiment, tag, point, method, seed, params, gate=None):
    uid = f"{program}:{point}:{method}:s{seed}"
    art = unit_path(_abs(runs_root), tag, point, method, seed)
    return {"uid": uid, "stage": "train", "program": program, "experiment": experiment,
            "gate": gate, "tag": tag, "point": point, "method": method, "seed": int(seed),
            "params": params, "artifact": str(art)}


def _analytic_unit(outputs_root, program, experiment, out_rel, params, gate=None, seed=0):
    uid = f"{program}:{experiment}"
    art = _abs(outputs_root) / "analytic" / program / "done.json"
    return {"uid": uid, "stage": "analytic", "program": program, "experiment": experiment,
            "gate": gate, "tag": program, "point": experiment, "method": program, "seed": seed,
            "params": params, "artifact": str(art), "out_rel": out_rel}


# --------------------------------------------------------------------------- #
# TEMPO enumeration
def _et1_schedules(t):
    scheds = []
    for tau in t["tau_grid"]:
        scheds.append((f"oracle_lts_tau{tau}", {"kind": "learn_then_sense", "tau": tau}))
    scheds.append((f"oracle_stl_tau{t['stl_tau']}",
                   {"kind": "sense_then_learn", "tau": t["stl_tau"]}))
    for b in t["burst_lens"]:
        for p in t["burst_periods"]:
            scheds.append((f"oracle_burst_b{b}_p{p}", {"kind": "bursting", "burst_len": b, "period": p}))
    return scheds


def tempo_units(cfg):
    units = []
    rr = cfg.get("runs_root", "runs")
    tp = cfg["tempo"]
    if not tp.get("enabled", True):
        return units
    p_max = tp["p_max"]
    ds_map = cfg["datasets"]

    # E-T1 oracle schedules (GATE 1)
    t1 = tp["et1"]
    for dsk in t1["datasets"]:
        for sp in t1["sigma_p"]:
            point = f"ET1_{dsk}_sp{sp:g}"
            for name, spec in _et1_schedules(t1):
                for seed in _seeds(cfg, t1["seeds"]):
                    units.append(_fl_unit(rr, "tempo", "E-T1", "tempo", point, name, seed,
                                          {"policy": {"type": "schedule", "spec": spec},
                                           "dataset": ds_map[dsk], "sigma_p": sp, "mission": "terminal",
                                           "p_max": p_max, "alpha": t1.get("alpha", 0.3)}, gate="GATE1"))

    # E-T4 controllers + naive + tuned-static bake-off
    t4 = tp["et4"]
    policies = [("controller", c) for c in t4["controllers"]] + \
               [("schedule", n) for n in t4["naive"]] + \
               [("schedule", {"name": f"static_lam{t4['tuned_static_lambda']:g}",
                              "kind": "static", "lam": t4["tuned_static_lambda"]})]
    for dsk in t4["datasets"]:
        for sp in t4["sigma_p"]:
            for mission in t4["missions"]:
                point = f"ET4_{dsk}_sp{sp:g}_{mission}"
                for ptype, pspec in policies:
                    name = pspec["name"]
                    for seed in _seeds(cfg, t4["seeds"]):
                        units.append(_fl_unit(rr, "tempo", "E-T4", "tempo", point, name, seed,
                                              {"policy": {"type": ptype, "spec": pspec},
                                               "dataset": ds_map[dsk], "sigma_p": sp, "mission": mission,
                                               "p_max": p_max}))

    # E-T3 mobility sweep
    t3 = tp.get("et3", {})
    if t3.get("enabled"):
        for dsk in t3["datasets"]:
            for sp in t3["sigma_p"]:
                point = f"ET3_{dsk}_sp{sp:g}"
                pol_specs = [("controller", {"name": "tempo_threshold", "kind": "threshold", "adaptive": True}),
                             ("controller", {"name": "tempo_dpp", "kind": "dpp", "V": 1.0}),
                             ("schedule", {"name": "static_lam0.5", "kind": "static", "lam": 0.5})]
                for ptype, pspec in pol_specs:
                    for seed in _seeds(cfg, t3["seeds"]):
                        units.append(_fl_unit(rr, "tempo", "E-T3", "tempo", point, pspec["name"], seed,
                                              {"policy": {"type": ptype, "spec": pspec},
                                               "dataset": ds_map[dsk], "sigma_p": sp,
                                               "mission": "sustained", "p_max": p_max}))

    # E-T5 ablations (MPC horizon, DPP V)
    t5 = tp.get("et5", {})
    if t5.get("enabled"):
        for dsk in t5["datasets"]:
            for sp in t5["sigma_p"]:
                for mission in t5["missions"]:
                    point = f"ET5_{dsk}_sp{sp:g}_{mission}"
                    specs = [("controller", {"name": f"mpc_H{h}", "kind": "mpc", "horizon": h})
                             for h in t5["mpc_horizons"]]
                    specs += [("controller", {"name": f"dpp_V{v:g}", "kind": "dpp", "V": v})
                              for v in t5["dpp_V"]]
                    for ptype, pspec in specs:
                        for seed in _seeds(cfg, t5["seeds"]):
                            units.append(_fl_unit(rr, "tempo", "E-T5", "tempo", point, pspec["name"], seed,
                                                  {"policy": {"type": ptype, "spec": pspec},
                                                   "dataset": ds_map[dsk], "sigma_p": sp,
                                                   "mission": mission, "p_max": p_max}))

    # E-T6 online regret (long horizon)
    t6 = tp.get("et6", {})
    if t6.get("enabled"):
        for dsk in t6["datasets"]:
            for sp in t6["sigma_p"]:
                point = f"ET6_{dsk}_sp{sp:g}"
                for ptype, pspec in [("controller", {"name": "tempo_dpp", "kind": "dpp", "V": 1.0}),
                                     ("schedule", {"name": "static_lam0.5", "kind": "static", "lam": 0.5})]:
                    for seed in _seeds(cfg, t6["seeds"]):
                        units.append(_fl_unit(rr, "tempo", "E-T6", "tempo", point, pspec["name"], seed,
                                              {"policy": {"type": ptype, "spec": pspec},
                                               "dataset": ds_map[dsk], "sigma_p": sp, "mission": "sustained",
                                               "p_max": p_max, "rounds": t6.get("rounds", 300)}))
    return units


# --------------------------------------------------------------------------- #
# CloakFL enumeration
def cloak_units(cfg):
    units = []
    rr = cfg.get("runs_root", "runs")
    ck = cfg["cloak"]
    if not ck.get("enabled", True):
        return units
    ds_map = cfg["datasets"]

    # E-C3 privacy-utility frontier (headline)
    c3 = ck["ec3"]
    for dsk in c3["datasets"]:
        point = f"EC3_{dsk}"
        for mode in c3["modes"]:
            for rf in c3["r_floors"]:
                method = f"{mode}__r{rf:g}"
                for seed in _seeds(cfg, c3["seeds"]):
                    units.append(_fl_unit(rr, "cloak", "E-C3", "cloak", point, method, seed,
                                          {"mode": mode, "r_floor": rf, "dataset": ds_map[dsk]}))

    # E-C2c short FL run (M2 on/off)
    c2c = ck.get("ec2c", {})
    if c2c.get("enabled"):
        for dsk in c2c["datasets"]:
            point = f"EC2c_{dsk}"
            for mode in c2c["modes"]:
                method = f"{mode}__r10"
                for seed in _seeds(cfg, c2c["seeds"]):
                    units.append(_fl_unit(rr, "cloak", "E-C2c", "cloak", point, method, seed,
                                          {"mode": mode, "r_floor": 10.0, "dataset": ds_map[dsk],
                                           "rounds": c2c.get("rounds", 30)}))

    # E-C5 composition & rotation (long horizon)
    c5 = ck.get("ec5", {})
    if c5.get("enabled"):
        for dsk in c5["datasets"]:
            point = f"EC5_{dsk}"
            for mode in c5["modes"]:
                for rf in c5["r_floors"]:
                    method = f"{mode}__r{rf:g}"
                    for seed in _seeds(cfg, c5["seeds"]):
                        units.append(_fl_unit(rr, "cloak", "E-C5", "cloak", point, method, seed,
                                              {"mode": mode, "r_floor": rf, "dataset": ds_map[dsk],
                                               "rounds": c5.get("rounds", 300)}))
    return units


# --------------------------------------------------------------------------- #
# Analytic enumeration (near-zero GPU; kill tests + measurement)
def analytic_units(cfg):
    units = []
    orr = cfg.get("outputs_root", "outputs/tempo_cloak")
    ck = cfg["cloak"]
    if ck.get("enabled", True):
        units.append(_analytic_unit(orr, "ec1", "E-C1", "cloak/ec1",
                                    dict(ck["entanglement"]), gate="GATE2"))
        units.append(_analytic_unit(orr, "ec2ab", "E-C2ab", "cloak/ec2",
                                    dict(ck["dither"]), gate="GATE3"))
        units.append(_analytic_unit(orr, "ec4", "E-C4", "cloak/ec4",
                                    {"point": ck["ec4"]["point"], "seeds": _seeds(cfg, ck["ec4"]["seeds"]),
                                     "base_config": cfg["base_config"],
                                     "rerun_top9_if_missing": ck["ec4"].get("rerun_top9_if_missing", True)}))
    if cfg["tempo"].get("enabled", True):
        units.append(_analytic_unit(orr, "et2", "E-T2", "tempo/et2",
                                    {"point": cfg["existing_campaign_point"],
                                     "base_config": cfg["base_config"],
                                     "alphas": [0.1, 0.3, 0.5]}))
        units.append(_analytic_unit(orr, "tempo_rescore", "E-T1-static", "tempo/static_frontier",
                                    {"point": cfg["existing_campaign_point"],
                                     "base_config": cfg["base_config"],
                                     "p_max": cfg["tempo"]["p_max"]}, gate="GATE1"))
    return units


# --------------------------------------------------------------------------- #
def enumerate_units(cfg, stages=None):
    """All units, optionally filtered to a set of stages ('analytic','train')."""
    units = analytic_units(cfg) + tempo_units(cfg) + cloak_units(cfg)
    if stages:
        units = [u for u in units if u["stage"] in stages]
    return units


def is_complete(unit) -> bool:
    """True if the unit's artifact exists and is complete (resumability)."""
    if unit["stage"] == "train":
        return load_unit(Path(unit["artifact"])) is not None
    return Path(unit["artifact"]).exists()


def apply_smoke(cfg) -> dict:
    """Shrink every grid to one tiny unit per experiment type (smoke stage)."""
    import copy
    c = copy.deepcopy(cfg)
    sm = c["smoke"]
    # ISOLATE smoke artifacts so a 5-round smoke unit never masquerades as a real
    # completed 150-round unit under resume.
    c["runs_root"] = c.get("runs_root", "runs") + "_smoke"
    c["outputs_root"] = c.get("outputs_root", "outputs/tempo_cloak") + "_smoke"
    c["fl"]["rounds"] = sm["rounds"]
    c["fl"]["subsample_train"] = sm["subsample_train"]
    c["fl"]["subsample_test"] = sm["subsample_test"]
    c["seeds"] = {"headline": list(sm["seeds"]), "sweep": list(sm["seeds"])}
    # one dataset, one setting each
    tp, ck = c["tempo"], c["cloak"]
    tp["et1"].update(datasets=["cifar10"], sigma_p=[0.0], tau_grid=[50], burst_lens=[5], burst_periods=[25])
    tp["et4"].update(datasets=["cifar10"], sigma_p=[0.05], missions=["sustained"])
    tp["et4"]["controllers"] = tp["et4"]["controllers"][:1]
    tp["et4"]["naive"] = tp["et4"]["naive"][:1]
    for k in ("et3", "et5", "et6"):
        tp.get(k, {})["enabled"] = False
    ck["ec3"].update(datasets=["cifar10"], r_floors=[1.0], modes=["uncapped", "m1"])
    ck["ec2c"].update(datasets=["cifar10"], rounds=sm["rounds"], modes=["m1", "m1_m2"])
    ck.get("ec5", {})["enabled"] = False
    ck["entanglement"]["n_mc"] = 20
    return c

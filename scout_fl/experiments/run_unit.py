"""Run ONE experimental unit by uid (the resumable atom dispatched by run_all.sh).

    python -m scout_fl.experiments.run_unit --uid "tempo:ET1_cifar10_sp0:oracle_lts_tau50:s0"
    python -m scout_fl.experiments.run_unit --uid "cloak:EC3_cifar10:m1__r1:s3"
    python -m scout_fl.experiments.run_unit --uid "ec1:E-C1"            # analytic study

Loads experiments/config.yaml (or the smoke-shrunk config with --smoke), finds the
unit, skips it if its artifact is already complete (unless --force), else runs it.
Deterministic: the seed is threaded through and logged in every artifact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scout_fl.experiments import units as U
from scout_fl.utils.config import load_config

REPO = Path(__file__).resolve().parents[2]


def _abs(root):
    p = Path(root)
    return str(p if p.is_absolute() else (REPO / p))


def _cfg_for_fl(cfg, params):
    """Build the per-unit scout_fl Config (campaign base + dataset/rounds overrides)."""
    ds = params["dataset"]
    ov = [f"fl.dataset={ds['dataset']}", f"fl.model={ds['model']}",
          f"fl.rounds={params.get('rounds', cfg['fl']['rounds'])}",
          f"fl.dirichlet_alpha={params.get('alpha', cfg['fl']['dirichlet_alpha'])}",
          f"fl.subsample_train={cfg['fl']['subsample_train']}",
          f"fl.subsample_test={cfg['fl']['subsample_test']}",
          f"fl.device={cfg['fl']['device']}"]
    scfg = load_config(str(REPO / cfg["base_config"]), ov)
    # thread CloakFL knobs from the campaign config into the scout_fl Config
    scfg["cloak"] = {"prior_client_std_m": cfg["cloak"].get("prior_client_std_m", 100.0),
                     "sigma_d": cfg["cloak"].get("sigma_d", 1.0)}
    scfg["tempo"] = {"p_max": cfg["tempo"].get("p_max", 20.0)}
    return scfg


def _load_ds(scfg):
    from scout_fl.fl.datasets import load_fl_dataset
    return load_fl_dataset(scfg.fl.dataset, root=scfg.fl.data_root, download=bool(scfg.fl.download))


def _run_tempo(cfg, unit):
    from scout_fl.tempo.runner import run_tempo_seed
    from scout_fl.tempo.schedules import from_spec
    from scout_fl.tempo.controllers import build_controller
    p = unit["params"]
    scfg = _cfg_for_fl(cfg, p)
    ds = _load_ds(scfg)
    pol = p["policy"]
    schedule = controller = None
    if pol["type"] == "schedule":
        schedule = from_spec(pol["spec"])
    else:
        controller = build_controller(pol["spec"], T=int(scfg.fl.rounds),
                                      M=int(scfg.network.num_targets), p_max=p["p_max"])
    run_tempo_seed(scfg, ds, unit["seed"], unit["method"], schedule=schedule, controller=controller,
                   sigma_p=p["sigma_p"], p_max=p["p_max"], mission=p["mission"],
                   runs_root=_abs(cfg["runs_root"]), tag=unit["tag"], point=unit["point"],
                   sigma_p_ctrl=p.get("sigma_p_ctrl"), l_noise=p.get("l_noise", 0.0),
                   inner=p.get("inner", "v2"))


def _run_cloak(cfg, unit):
    from scout_fl.cloak.runner import run_cloak_seed
    p = unit["params"]
    scfg = _cfg_for_fl(cfg, p)
    ds = _load_ds(scfg)
    run_cloak_seed(scfg, ds, unit["seed"], p["mode"], p["r_floor"],
                   runs_root=_abs(cfg["runs_root"]), tag=unit["tag"], point=unit["point"],
                   method=unit["method"], csi_error=p.get("csi_error", 0.0))


def _run_analytic(cfg, unit):
    prog = unit["program"]
    out_root = Path(_abs(cfg["outputs_root"])) / "analytic"
    p = unit["params"]
    # E-C4 / E-T2 / static-frontier RE-SCORE the pre-existing campaign in the REAL
    # runs/ (independent of smoke isolation), so always source from REPO/runs.
    campaign_runs = str(REPO / "runs")
    if prog == "ec1":
        from scout_fl.cloak.entanglement import run_ec1
        run_ec1(out_root / "cloak/ec1", r_floor=p["r_floor"], n_mc=p["n_mc"])
    elif prog == "ec2ab":
        from scout_fl.cloak.dither_study import run_ec2ab
        run_ec2ab(out_root / "cloak/ec2", sigma_d=p["sigma_d"], snr_eve=p["snr_eve"])
    elif prog == "ec4":
        from scout_fl.cloak.ec4_measurement import run_ec4
        run_ec4(out_root / "cloak/ec4", runs_root=campaign_runs, point=p["point"],
                base_config=str(REPO / p["base_config"]), seeds=set(p["seeds"]),
                side_snr=p.get("side_snr", 1.0))
    elif prog == "et2":
        from scout_fl.tempo.rescore import gradient_decay_study
        gradient_decay_study(out_root / "tempo/et2", base_config=str(REPO / p["base_config"]),
                             alphas=tuple(p["alphas"]), runs_root=campaign_runs)
    elif prog == "tempo_rescore":
        from scout_fl.tempo.rescore import static_frontier_rescore
        static_frontier_rescore(out_root / "tempo/static_frontier", point=p["point"],
                                base_config=str(REPO / p["base_config"]), p_max=p["p_max"],
                                runs_root=campaign_runs)
    else:
        raise ValueError(f"unknown analytic program {prog!r}")
    Path(unit["artifact"]).parent.mkdir(parents=True, exist_ok=True)
    Path(unit["artifact"]).write_text('{"complete": true}')


def run_unit(cfg, unit, force=False):
    if not force and U.is_complete(unit):
        print(f"[skip] {unit['uid']} (complete)")
        return "skipped"
    print(f"[run ] {unit['uid']}  ({unit['experiment']}, stage={unit['stage']})", flush=True)
    if unit["program"] == "tempo":
        _run_tempo(cfg, unit)
    elif unit["program"] == "cloak":
        _run_cloak(cfg, unit)
    else:
        _run_analytic(cfg, unit)
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = U.load_campaign_config(args.config)
    if args.smoke:
        cfg = U.apply_smoke(cfg)
    by_uid = {u["uid"]: u for u in U.enumerate_units(cfg)}
    if args.uid not in by_uid:
        raise SystemExit(f"unknown uid {args.uid!r} (have {len(by_uid)} units; check --smoke match)")
    run_unit(cfg, by_uid[args.uid], force=args.force)


if __name__ == "__main__":
    main()

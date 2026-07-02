"""Deterministic scenario replay for re-scoring existing runs/ artifacts.

The runs/ per-round JSON logs the selected client set every round but NOT the
client positions, target positions, or per-client uplink SNR (see
analysis/schema_report.md). Those are needed by the leakage accountant (E-C4)
and the tracker re-scoring (E-T2). Because the pipeline is deterministically
seeded (utils/seed.py), the geometry + channel are fully reconstructible from
(config, seed): this module replays the EXACT RNG draw order of
run_fl_synthetic.run_seed so scn.clients / g / uplink-SNR reproduce bit-for-bit.

Faithfulness is verifiable without re-running FL: the round-0 sensing log-det of
the logged selection, recomputed on the reconstructed FIM, must match the logged
``sensing_logdet`` (stationary campaign) — see ``verify_against_artifact`` and
tests/test_infra_replay.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scout_fl.experiments.run_synthetic import build_scenario
from scout_fl.sim.channel import comm_channel_gains
from scout_fl.utils.config import load_config
from scout_fl.utils.seed import seed_everything


def _point_override_map():
    """point_tag -> list of 'key=value' overrides, mirroring run_campaign.SWEEPS."""
    from scout_fl.experiments.run_campaign import SWEEPS, _point_overrides
    out = {}
    for name, spec in SWEEPS.items():
        for point in spec["points"]:
            label = point.get(spec["param"])
            out[f"{name}={label}"] = _point_overrides(point)
    return out


def config_for_point(point: str, base_config: str, extra_overrides=None):
    """Reconstruct the Config used for a campaign point (base + point overrides)."""
    overrides = list(_point_override_map().get(point, []))
    if extra_overrides:
        overrides += list(extra_overrides)
    return load_config(base_config, overrides)


def reconstruct(cfg, seed: int):
    """Replay geometry + channel for one seed. Returns a dict with clients, targets,
    scenario, channel gains g, and per-client uplink SNR (P g / sigma2).

    RNG order MUST match run_fl_synthetic.run_seed:
        rng = seed_everything(seed); scn = build_scenario(cfg, rng);
        g = comm_channel_gains(..., rng, ...); scn.compute_het = rng.uniform(...)
    """
    rng = seed_everything(int(seed))
    scn = build_scenario(cfg, rng)
    chan_source = cfg.channel.get("source", "synthetic")
    if chan_source != "synthetic":
        from scout_fl.fl.datasets_external import load_channel_realizations
        g = load_channel_realizations(chan_source, scn.K, rng, root=cfg.fl.get("data_root", "data"))
    else:
        phys = cfg.get("physical", {})
        g = comm_channel_gains(
            scn.clients, np.asarray(cfg.geometry.bs_position, dtype=float), rng,
            snr_ref_db=cfg.channel.snr_ref_db, ref_distance=cfg.channel.reference_distance,
            pathloss_exponent=cfg.channel.pathloss_exponent, model=cfg.channel.model,
            rician_k_db=cfg.channel.rician_k_db,
            pathloss_model=("physical" if phys and phys.get("enabled") else "reference_snr"),
            carrier_ghz=float(phys.get("carrier_ghz", 3.5)) if phys else 3.5)
    # draw compute_het to keep the RNG stream consistent (unused here, but matches run_seed)
    if getattr(scn, "compute_het", None) is None:
        scn.compute_het = rng.uniform(0.1, 1.0, scn.K)

    P, sigma2 = _power_sigma2(cfg, scn)
    snr_up = P * np.asarray(g, dtype=float) / sigma2
    return {"scn": scn, "clients": scn.clients, "targets": getattr(scn, "targets", None),
            "g": np.asarray(g, dtype=float), "snr_up": snr_up, "P": P, "sigma2": sigma2}


def _power_sigma2(cfg, scn):
    """Reproduce the P and sigma2 used by the runner (physical link budget or config)."""
    phys = cfg.get("physical", {})
    if phys and phys.get("enabled"):
        from scout_fl.sim.link_budget import dbm_to_watt, thermal_noise_power_w
        B = float(cfg.aircomp.bandwidth)
        sigma2 = thermal_noise_power_w(B, float(phys.get("noise_figure_db", 7.0)),
                                       float(phys.get("temperature_k", 290.0)))
        P = dbm_to_watt(float(phys.get("tx_power_dbm", 0.0)))
        return P, sigma2
    return float(cfg.aircomp.power), float(cfg.aircomp.sigma2)


def load_artifact(path):
    """Load a runs/ unit JSON (returns None if missing/corrupt)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def verify_against_artifact(cfg, artifact, tol: float = 1e-3):
    """Check the reconstructed FIM reproduces the artifact's round-0 sensing log-det.

    Valid only for stationary runs (nonstationarity disabled), which is the case
    for the campaign. Returns (ok, replay_logdet, logged_logdet, |diff|).
    """
    from scout_fl.objectives.sensing_utility import SensingUtility
    meta = artifact.get("meta", {})
    rows = artifact.get("rounds", [])
    if not rows:
        return False, float("nan"), float("nan"), float("nan")
    rec = reconstruct(cfg, int(meta.get("seed", 0)))
    scn = rec["scn"]
    sensing = SensingUtility(scn.fim, scn.j0, scn.w)
    sel0 = [int(k) for k in rows[0].get("selected", [])]
    replay_ld = float(sensing.value(sel0))
    logged_ld = float(rows[0].get("sensing_logdet", float("nan")))
    diff = abs(replay_ld - logged_ld)
    denom = max(1.0, abs(logged_ld))
    return bool(diff / denom <= tol), replay_ld, logged_ld, diff

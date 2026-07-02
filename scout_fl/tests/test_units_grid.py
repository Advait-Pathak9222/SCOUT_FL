"""Campaign unit-grid integrity: every design-doc experiment enumerates, uids are
unique, smoke is isolated from real roots, and the new E-T5/E-C6 knobs are wired.
"""
from __future__ import annotations

from scout_fl.experiments import units as U

_EXPECTED_EXPERIMENTS = {
    "E-C1", "E-C2ab", "E-C4", "E-T2", "E-T1-static",          # analytic
    "E-T1", "E-T3", "E-T4", "E-T5", "E-T6",                   # TEMPO train
    "E-C2c", "E-C3", "E-C5", "E-C6",                          # CloakFL train
}


def test_full_grid_covers_every_design_experiment():
    cfg = U.load_campaign_config()
    units = U.enumerate_units(cfg)
    assert {u["experiment"] for u in units} == _EXPECTED_EXPERIMENTS


def test_uids_unique_and_artifacts_unique():
    cfg = U.load_campaign_config()
    units = U.enumerate_units(cfg)
    uids = [u["uid"] for u in units]
    arts = [u["artifact"] for u in units]
    assert len(uids) == len(set(uids)), "duplicate unit uids"
    assert len(arts) == len(set(arts)), "two units share an artifact path"


def test_smoke_isolated_from_real_roots():
    cfg = U.load_campaign_config()
    smoke = U.apply_smoke(cfg)
    real_arts = {u["artifact"] for u in U.enumerate_units(cfg)}
    smoke_arts = {u["artifact"] for u in U.enumerate_units(smoke)}
    assert not (real_arts & smoke_arts), "smoke and real units share artifact paths"
    assert all("runs_smoke" in a or "tempo_cloak_smoke" in a for a in smoke_arts)


def test_et5_ablation_axes_present():
    cfg = U.load_campaign_config()
    methods = {u["method"] for u in U.enumerate_units(cfg) if u["experiment"] == "E-T5"}
    # design §1.5 E-T5: horizon, V, P_max sweep, mis-specified Q, noisy L_t, inner swap
    assert any(m.startswith("mpc_H") for m in methods)
    assert any(m.startswith("dpp_V") for m in methods)
    assert any(m.startswith("dpp_pmax") for m in methods)
    assert any("qmis" in m for m in methods)
    assert any("noisyL" in m for m in methods)
    assert "dpp_innerplain" in methods
    # the Q-misspec unit must carry a controller belief different from the truth
    qmis = [u for u in U.enumerate_units(cfg)
            if u["experiment"] == "E-T5" and "qmis" in u["method"]]
    for u in qmis:
        assert u["params"]["sigma_p_ctrl"] != u["params"]["sigma_p"]


def test_ec6_units_carry_csi_error():
    cfg = U.load_campaign_config()
    ec6 = [u for u in U.enumerate_units(cfg) if u["experiment"] == "E-C6"]
    assert ec6, "E-C6 robustness units missing"
    assert all(u["params"]["csi_error"] > 0 for u in ec6)


def test_et6_has_two_configs_and_hindsight_references():
    cfg = U.load_campaign_config()
    et6 = [u for u in U.enumerate_units(cfg) if u["experiment"] == "E-T6"]
    points = {u["point"] for u in et6}
    assert len(points) >= 2, "E-T6 needs 2 configurations (design §1.5)"
    methods = {u["method"] for u in et6}
    assert "tempo_dpp" in methods
    assert len(methods - {"tempo_dpp"}) >= 2, "need reference schedules for the hindsight oracle"

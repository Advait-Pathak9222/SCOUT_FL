"""Tests for RECA-FL appraisal, world model, and adapter matching."""
from __future__ import annotations

import numpy as np

from scout_fl.fl.adapters import AdapterMatcher, ContextAdapterBank, RegimeSignature
from scout_fl.objectives.reca_appraisal import RECAAppraisal, cvar
from scout_fl.objectives.world_model import WorldModel


def _sig(x):
    x = np.asarray(x, dtype=float)
    half = max(1, x.size // 2)
    return RegimeSignature(
        embedding=x,
        grad_residual=x[:half],
        sensing_residual=x[half:],
        residual_mean=x,
        residual_var=np.ones_like(x) * 0.1,
    )


def test_cvar_increases_with_worse_tail():
    assert cvar([0.1, 0.2, 1.0], 0.8) > cvar([0.1, 0.2, 0.3], 0.8)


def test_reca_appraisal_finite_under_nonfinite_inputs():
    app = RECAAppraisal()
    res = app.evaluate([0.1, np.nan, np.inf], [1.0, np.nan], [0.2, np.inf])
    assert np.isfinite(res.risk)
    assert np.isfinite(res.mismatch)
    assert np.isfinite(res.progress)
    assert np.isfinite(res.trigger_score)
    scores = app.per_client_scores([0.2, np.nan], [2.0, np.inf], [0.5, np.nan])
    assert np.all(np.isfinite(scores))


def test_reca_trigger_requires_bounded_risk_mismatch_and_progress():
    app = RECAAppraisal(tau_trigger=0.1)
    good = app.evaluate([0.4, 0.42], [2.0, 2.2], [1.0, 0.8])
    extreme = app.evaluate([4.0, 5.0], [2.0, 2.2], [1.0, 0.8])
    noise = app.evaluate([0.4, 0.42], [8.0, 9.0], [-2.0, -1.0])
    assert good.should_accommodate
    assert not extreme.should_accommodate
    assert not noise.should_accommodate


def test_adapter_match_confidence_high_for_similar_low_for_mismatch():
    bank = ContextAdapterBank(tau_reuse=0.7)
    adapter = bank.spawn(_sig([1.0, 0.9, 0.2, 0.3]))
    adapter.state = "consolidated"
    similar = bank.match(_sig([0.95, 0.85, 0.25, 0.35]))
    different = AdapterMatcher(tau_reuse=0.0).match(
        _sig([-1.0, -0.8, 1.2, 1.1]), bank.adapters)
    assert similar.adapter_id == adapter.adapter_id
    assert similar.confidence > 0.7
    assert different.confidence < similar.confidence


def test_adapter_bank_lifecycle_consolidates_and_quarantines():
    bank = ContextAdapterBank(tau_reuse=0.7, consolidate_after=1, quarantine_below=-0.1)
    adapter = bank.spawn(_sig([1.0, 1.0, 0.0, 0.0]))
    bank.update_evidence(adapter.adapter_id, 0.2)
    assert bank.get(adapter.adapter_id).state == "consolidated"
    bad = bank.spawn(_sig([-1.0, 0.0, 1.0, 0.0]))
    bank.update_evidence(bad.adapter_id, -0.2)
    assert bank.get(bad.adapter_id).state == "quarantined"


def test_world_model_updates_and_reports_calibration():
    rng = np.random.default_rng(0)
    wm = WorldModel(feature_dim=3, output_dim=2, l2=1e-2)
    x = rng.normal(size=(20, 3))
    y = x[:, :2] + 0.1
    before = wm.calibration_report()
    wm.update(x, y)
    after = wm.calibration_report()
    assert before.n == 0
    assert after.n == 20
    assert after.rmse >= 0.0

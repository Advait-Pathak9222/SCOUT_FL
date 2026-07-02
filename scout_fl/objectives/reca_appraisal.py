"""RECA-FL technical appraisals.

RECA = Risk-bounded Epistemic Context Accommodation.  The appraisal combines
tail-risk, epistemic mismatch, and verified progress into a bounded trigger
score used for client selection and adapter lifecycle decisions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_array(x, fill: float = 0.0) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return np.nan_to_num(a, nan=fill, posinf=fill, neginf=fill)


def cvar(values, alpha: float = 0.9) -> float:
    """Upper-tail CVaR for finite values."""
    v = np.sort(_finite_array(values))
    if v.size == 0:
        return 0.0
    alpha = float(np.clip(alpha, 0.0, 0.999999))
    start = int(np.floor(alpha * (v.size - 1)))
    return float(v[start:].mean())


@dataclass
class RECAParams:
    cvar_alpha: float = 0.9
    risk_star: float = 0.4
    risk_sigma: float = 0.25
    risk_max: float = 1.0
    mismatch_max: float = 3.0
    eta_trigger: float = 1.0
    eta_risk_relief: float = 0.25
    eta_overwhelm: float = 1.0


@dataclass
class RECAResult:
    risk: float
    mismatch: float
    progress: float
    trigger_score: float
    overwhelm: float
    selection_score: float
    should_accommodate: bool


class RECAAppraisal:
    """Compute risk, mismatch, progress, and accommodation trigger scores."""

    def __init__(self, params: RECAParams | None = None, tau_trigger: float = 0.2) -> None:
        self.params = params or RECAParams()
        self.tau_trigger = float(tau_trigger)

    def evaluate(self, risk_values, mismatch_values, progress_values,
                 risk_before: float | None = None) -> RECAResult:
        p = self.params
        risk_arr = np.maximum(_finite_array(risk_values), 0.0)
        mismatch_arr = np.maximum(_finite_array(mismatch_values), 0.0)
        progress_arr = _finite_array(progress_values)

        risk = cvar(risk_arr, p.cvar_alpha)
        mismatch = float(mismatch_arr.mean()) if mismatch_arr.size else 0.0
        progress = float(progress_arr.mean()) if progress_arr.size else 0.0

        risk_band = np.exp(-((risk - p.risk_star) ** 2) / (2.0 * max(p.risk_sigma, 1e-9) ** 2))
        trigger = _sigmoid(progress) * np.log1p(mismatch) * float(risk_band)
        overwhelm = max(0.0, risk - p.risk_max) ** 2 + max(0.0, mismatch - p.mismatch_max) ** 2
        relief = 0.0 if risk_before is None else max(0.0, float(risk_before) - risk)
        score = progress + p.eta_trigger * trigger + p.eta_risk_relief * relief - p.eta_overwhelm * overwhelm
        return RECAResult(
            risk=float(risk),
            mismatch=float(mismatch),
            progress=float(progress),
            trigger_score=float(trigger),
            overwhelm=float(overwhelm),
            selection_score=float(score),
            should_accommodate=bool(trigger >= self.tau_trigger and overwhelm <= 1e-9),
        )

    def per_client_scores(self, risk, mismatch, progress) -> np.ndarray:
        """Client-wise selection scores used by the RECA selector."""
        p = self.params
        r = np.maximum(_finite_array(risk), 0.0)
        m = np.maximum(_finite_array(mismatch), 0.0)
        q = _finite_array(progress)
        band = np.exp(-((r - p.risk_star) ** 2) / (2.0 * max(p.risk_sigma, 1e-9) ** 2))
        trigger = _sigmoid(q) * np.log1p(m) * band
        overwhelm = np.maximum(0.0, r - p.risk_max) ** 2 + np.maximum(0.0, m - p.mismatch_max) ** 2
        return q + p.eta_trigger * trigger - p.eta_overwhelm * overwhelm


def _sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))

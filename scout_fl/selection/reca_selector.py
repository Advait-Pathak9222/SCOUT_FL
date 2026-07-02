"""RECA-FL client selector."""
from __future__ import annotations

import time

import numpy as np

from scout_fl.objectives.reca_appraisal import RECAAppraisal
from scout_fl.selection.base import SelectionResult, Selector


class RECASelector(Selector):
    """Top-K selector over RECA risk/mismatch/progress appraisals."""

    name = "reca"

    def __init__(self, appraisal: RECAAppraisal | None = None) -> None:
        self.appraisal = appraisal or RECAAppraisal()

    def select(self, risk, mismatch, progress, budget, **_) -> SelectionResult:
        start = time.perf_counter()
        score = self.appraisal.per_client_scores(risk, mismatch, progress)
        order = np.argsort(-score)[:int(budget)]
        aggregate = self.appraisal.evaluate(np.asarray(risk)[order],
                                            np.asarray(mismatch)[order],
                                            np.asarray(progress)[order])
        return SelectionResult(
            selected=sorted(int(k) for k in order),
            select_time=time.perf_counter() - start,
            info={
                "reca_risk": aggregate.risk,
                "reca_mismatch": aggregate.mismatch,
                "reca_progress": aggregate.progress,
                "reca_trigger_score": aggregate.trigger_score,
                "reca_overwhelm": aggregate.overwhelm,
                "reca_should_accommodate": aggregate.should_accommodate,
            },
        )

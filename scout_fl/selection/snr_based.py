"""Sensing-SNR-only baseline: pick the top-budget clients by total sensing SNR.

This is the "naive high-SNR" policy SCOUT-FL is meant to BEAT: it ignores
viewing geometry, so when the highest-SNR clients share a bearing it picks
redundant information and leaves the cross-range axis poorly observed.
"""
from __future__ import annotations

import time

import numpy as np

from scout_fl.selection.base import Selector, SelectionResult


class SNRSelector(Selector):
    name = "snr_only"

    def select(self, scores, budget: int, **_) -> SelectionResult:
        start = time.perf_counter()
        scores = np.asarray(scores, dtype=float)
        order = np.argsort(-scores, kind="stable")[:budget]
        return SelectionResult(
            selected=sorted(int(k) for k in order),
            select_time=time.perf_counter() - start,
            info={"scores": scores.tolist()},
        )

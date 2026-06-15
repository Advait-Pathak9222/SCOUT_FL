"""Hard-constraint feasibility for client selection.

Per the plan's PRIMARY (constrained) formulation, sensing/latency/energy/MSE are
hard constraints rather than weighted penalties. This module evaluates a
selected set against the active limits and reports per-constraint satisfaction +
slack. It NEVER silently ignores a constraint: an infeasible round is reported,
and the caller applies ``infeasible_policy`` (default: relax-and-log).

Milestone status: CRB is computable now (from SensingUtility); MSE/latency/
energy/power become active when the AirComp + energy/latency modules land
(Step 6). The constraint-integrated greedy (feasibility filter inside the loop)
is wired in at that step; here we provide the evaluator + policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Constraints:
    crb_max: Any = None            # per-target Gamma_m (scalar or length-M) on CRB
    mse_agg_max: float | None = None
    latency_max: float | None = None
    energy_max: float | None = None
    power_budget: float | None = None
    infeasible_policy: str = "relax_and_log"

    def evaluate(self, *, crb=None, mse=None, latency=None, energy=None,
                 power=None) -> dict[str, Any]:
        report: dict[str, Any] = {}
        self._scalar_or_vector(report, "crb", self.crb_max, crb, kind="upper")
        self._scalar(report, "mse_agg", self.mse_agg_max, mse, kind="upper")
        self._scalar(report, "latency", self.latency_max, latency, kind="upper")
        self._scalar(report, "energy", self.energy_max, energy, kind="upper")
        self._scalar(report, "power", self.power_budget, power, kind="upper")
        active = [v for v in report.values() if v is not None]
        report["feasible"] = all(v["satisfied"] for v in active) if active else True
        return report

    @staticmethod
    def _scalar(report, name, limit, value, kind="upper"):
        if limit is None or value is None:
            report[name] = None
            return
        value = float(value)
        satisfied = value <= float(limit) + 1e-9
        report[name] = {"satisfied": bool(satisfied), "value": value,
                        "limit": float(limit), "slack": float(limit) - value}

    @staticmethod
    def _scalar_or_vector(report, name, limit, value, kind="upper"):
        if limit is None or value is None:
            report[name] = None
            return
        value = np.asarray(value, dtype=float)
        limit_arr = np.asarray(limit, dtype=float)
        satisfied = bool(np.all(value <= limit_arr + 1e-9))
        report[name] = {"satisfied": satisfied, "value": value.tolist(),
                        "limit": limit_arr.tolist(),
                        "max_violation": float(np.max(value - limit_arr))}

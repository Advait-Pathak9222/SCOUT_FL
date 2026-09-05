"""AirComp / over-the-air aggregation distortion (channel-inversion model).

Computing the average of selected clients' model updates over the multiple-access
channel via channel inversion: receive scaling ``eta = P * min_{k in S} g_k``
(limited by the weakest selected link), giving the noise-limited aggregation MSE

    MSE(S) = sigma2 / (|S|^2 * P * min_{k in S} g_k).

Consequences used by the A2 resource layer:
- larger transmit power P  -> lower MSE  (power control matters);
- a weak-channel client     -> larger MSE (the min term drops);
- gating out weak channels  -> larger min g -> lower MSE (AirComp-aware selection).

``min_gain_for_mse`` inverts the formula to the per-client channel-gain threshold
needed to meet an MSE target at a full budget — the feasibility gate used by the
constraint-integrated selector.
"""
from __future__ import annotations

import numpy as np


def aggregation_mse(channel_gains, selected, *, power: float = 1.0,
                    sigma2: float = 1.0, interference: float = 0.0,
                    update_power: float = 1.0) -> float:
    """Channel-inversion AirComp aggregation MSE for a selected set.

    ``power`` is the per-client transmit budget P_k^max (scalar, or an array of
    per-client budgets for a heterogeneous fleet). ``update_power`` is the
    per-entry power pi of the model update, which sets how far the denoising
    factor can be pushed before the power budget binds:

        rho*(S) = min_k |h_k| sqrt(P_k / pi),
        MSE(S)  = (sigma2 + I) / (|S|^2 rho*(S)^2).

    ``interference`` is the aggregate co-channel power I, which enters only
    through the effective noise floor sigma2 + I.
    """
    idx = list(selected)
    if not idx:
        return float("inf")
    g = np.asarray(channel_gains, dtype=float)[idx]
    P = np.asarray(power, dtype=float)
    P = P[idx] if P.ndim else P
    pi = max(float(update_power), 1e-300)
    # rho*^2 = min_k g_k P_k / pi   (g = |h|^2), so |S|^2 rho*^2 = |S|^2 min_k g_k P_k / pi
    eta = float(np.min(g * P)) / pi
    sigma2 = float(sigma2) + float(interference)
    # Guard the *degenerate* case only (no link => infinite aggregation error). A fixed
    # numerical floor on eta must NOT be used here: under the physical link budget
    # (sim/link_budget.py) eta = P * g_min is of order 1e-14..1e-10 W, so any floor at
    # 1e-12 silently caps MSE at sigma2/(|S|^2 * floor) and makes the aggregation error
    # independent of transmit power over the whole low-power half of the sweep.
    if not np.isfinite(eta) or eta <= 0.0:
        return float("inf")
    return float(sigma2 / (len(idx) ** 2 * eta))


def aircomp_eta(channel_gains, selected, power: float = 1.0,
                update_power: float = 1.0) -> float:
    """Receive scaling ``rho*^2 = min_k g_k P_k / pi`` (the channel-inversion factor)."""
    idx = list(selected)
    if not idx:
        return 0.0
    g = np.asarray(channel_gains, dtype=float)[idx]
    P = np.asarray(power, dtype=float)
    P = P[idx] if P.ndim else P
    return float(np.min(g * P)) / max(float(update_power), 1e-300)


def min_gain_for_mse(mse_eps: float, budget: int, power: float = 1.0,
                     sigma2: float = 1.0, interference: float = 0.0,
                     update_power: float = 1.0) -> float:
    """Min channel gain a client needs so a full-budget set meets ``MSE <= eps``.

    From ``(sigma2 + I) * pi / (budget^2 * P * g_min) <= eps``.
    """
    P = float(np.min(power)) if np.ndim(power) else float(power)
    num = (float(sigma2) + float(interference)) * max(float(update_power), 1e-300)
    return float(num / (max(int(budget), 1) ** 2 * P * max(mse_eps, 1e-300)))

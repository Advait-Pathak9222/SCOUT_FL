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
                    sigma2: float = 1.0) -> float:
    """Channel-inversion AirComp aggregation MSE for a selected set."""
    idx = list(selected)
    if not idx:
        return float("inf")
    g = np.asarray(channel_gains, dtype=float)[idx]
    eta = power * float(g.min())
    return float(sigma2 / (len(idx) ** 2 * max(eta, 1e-12)))


def aircomp_eta(channel_gains, selected, power: float = 1.0) -> float:
    """Receive scaling ``eta = P * min_{k in S} g_k`` (the channel-inversion factor)."""
    idx = list(selected)
    if not idx:
        return 0.0
    return float(power * np.asarray(channel_gains, dtype=float)[idx].min())


def min_gain_for_mse(mse_eps: float, budget: int, power: float = 1.0,
                     sigma2: float = 1.0) -> float:
    """Min channel gain a client needs so a full-budget set meets ``MSE <= eps``.

    From ``sigma2 / (budget^2 * P * g_min) <= eps``.
    """
    return float(sigma2 / (max(int(budget), 1) ** 2 * power * max(mse_eps, 1e-12)))

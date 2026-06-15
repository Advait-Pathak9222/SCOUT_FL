"""Per-client, per-target Fisher Information Matrix (FIM) for 2-D localization.

Model rationale (this is the crux of SCOUT-FL's "high-SNR != high-value" claim)
------------------------------------------------------------------------------
Each client, observing a target, contributes information about the target's 2-D
position (x, y). A range (time-of-flight) measurement informs the *radial*
direction ``u`` (client->target); an angle (DoA) measurement informs the
*tangential* direction ``v``, with cross-range error growing like range^2.
Modeling the per-client position FIM in a common (x, y) frame:

    J_{k,m} = a_r * u u^T  +  a_a * v v^T,        (2x2, PSD)
    a_r = gamma_{k,m} * k_range                   (radial / range information)
    a_a = gamma_{k,m} * k_angle / range^2         (tangential / cross-range info)

with gamma the linear sensing SNR. Because J is anisotropic and oriented by the
viewing geometry, two clients at the *same* bearing pile information onto the
same axis (diminishing returns), while a client at a *complementary* bearing
fills the weak axis — so angular diversity, not raw SNR, drives the accumulated
information ``J_m(S) = J_0 + sum_{k in S} J_{k,m}`` and the resulting CRB.

The construction (sum of positively-weighted outer products) is PSD by design.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def db_to_linear(db: Any) -> np.ndarray:
    """Convert dB (e.g. SNR) to linear scale."""
    return 10.0 ** (np.asarray(db, dtype=float) / 10.0)


def per_client_target_fim(geom: dict[str, Any], snr_linear: np.ndarray,
                          k_range: float, k_angle: float) -> np.ndarray:
    """Build the (K, M, 2, 2) stack of per-client per-target position FIMs.

    Parameters
    ----------
    geom : output of :func:`scout_fl.sim.geometry.pairwise_geometry`
    snr_linear : (K,) or (K, M) linear sensing SNR (gamma)
    k_range : radial (range) information coefficient (proportional to bandwidth^2)
    k_angle : tangential (angle/cross-range) information coefficient
    """
    u = geom["u"]                       # (K, M, 2)
    v = geom["v"]                       # (K, M, 2)
    rng = geom["range"]                 # (K, M)

    snr = np.asarray(snr_linear, dtype=float)
    if snr.ndim == 1:
        snr = snr[:, None]              # (K, 1) -> broadcast over M
    a_r = snr * float(k_range)                                  # (K, M)
    a_a = snr * float(k_angle) / np.clip(rng, 1e-9, None) ** 2  # (K, M)

    uuT = u[..., :, None] * u[..., None, :]    # (K, M, 2, 2)
    vvT = v[..., :, None] * v[..., None, :]    # (K, M, 2, 2)
    fim = a_r[..., None, None] * uuT + a_a[..., None, None] * vvT
    return fim                                  # (K, M, 2, 2), PSD


def prior_fim(num_targets: int, prior: float, dim: int = 2) -> np.ndarray:
    """Prior/regularization FIM ``J_0 = prior * I`` per target -> (M, dim, dim).

    A small positive prior keeps every accumulated FIM strictly positive
    definite, so log-det and matrix inverses are numerically stable even before
    any client is selected.
    """
    eye = np.eye(dim)
    return float(prior) * np.broadcast_to(eye, (num_targets, dim, dim)).copy()

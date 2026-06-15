"""Sensing SNR from a reference-SNR distance-decay model.

Per (client, target) linear sensing SNR:

    SNR_lin = 10^(ref_snr_db/10) * RCS * (ref_distance / range)^pathloss_exponent

``ref_snr_db`` is the sensing SNR (dB) a unit-RCS target would yield at
``ref_distance``; it decays with distance. This avoids the miscalibration of a
raw dBm link budget (which omits wavelength/antenna/two-way-RCS constants and
explodes the SNR) and keeps received SNR in a realistic range for development.
A two-way radar-equation + fading-aware version arrives with the channel module
(Step 6).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def sensing_snr(geom: dict[str, Any], ref_snr_db: float, pathloss_exponent: float,
                rcs: Any = 1.0, ref_distance: float = 1.0) -> np.ndarray:
    """Return the (K, M) linear sensing SNR stack."""
    rng = np.clip(geom["range"], ref_distance, None)            # (K, M)
    snr0 = 10.0 ** (float(ref_snr_db) / 10.0)
    rcs = np.asarray(rcs, dtype=float)                          # scalar or (M,)
    decay = (float(ref_distance) / rng) ** float(pathloss_exponent)
    return snr0 * rcs * decay                                   # (K, M) linear

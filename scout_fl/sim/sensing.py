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
                rcs: Any = 1.0, ref_distance: float = 1.0,
                tx_power_dbm: float | None = None,
                ref_tx_power_dbm: float | None = None) -> np.ndarray:
    """Return the (K, M) linear sensing SNR stack.

    The echo that carries the sensing information is the same transmission that
    carries the model update, so the echo power is proportional to the transmit
    power. Passing ``tx_power_dbm`` together with the ``ref_tx_power_dbm`` at
    which ``ref_snr_db`` was calibrated shifts the whole sensing SNR by their
    difference in dB, which keeps the nominal operating point unchanged while
    making a transmit-power sweep act on the sensing axis as well as the
    communication axis. Leaving either argument as None keeps the sensing SNR
    independent of the transmit power.
    """
    rng = np.clip(geom["range"], ref_distance, None)            # (K, M)
    ref_snr_db = float(ref_snr_db)
    if tx_power_dbm is not None and ref_tx_power_dbm is not None:
        ref_snr_db += float(tx_power_dbm) - float(ref_tx_power_dbm)
    snr0 = 10.0 ** (ref_snr_db / 10.0)
    rcs = np.asarray(rcs, dtype=float)                          # scalar or (M,)
    decay = (float(ref_distance) / rng) ** float(pathloss_exponent)
    return snr0 * rcs * decay                                   # (K, M) linear

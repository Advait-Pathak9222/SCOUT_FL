"""Physical link budget — turns the normalized sim into REAL units (W, s, J, dB).

The energy/latency formulas (energy_latency.py) and the AirComp MSE (aircomp.py) were
always dimensionally correct; only their INPUTS were normalized (P=1, sigma2=1, an
SNR-referenced channel gain). This module supplies the genuinely physical inputs so the
reported numbers are Joules / seconds / real SNR by CONSTRUCTION — not by any post-hoc
linear scaling:

* noise power sigma^2 = k_B * T * F * B   (Johnson-Nyquist thermal noise, W),
* transmit power P from dBm,
* channel power gain g = 10^(-PL(d)/10) * |fading|^2 from a log-distance path-loss model
  (3GPP-style), so the received SNR = P*g/sigma^2 is a true, unitless ratio (-> dB),
* model payload in real bits = (#parameters) * bits_per_param.

Reference values (document in the paper): N0 = k_B*T = -174 dBm/Hz at T=290 K; noise
figure F ~ 7 dB; carrier 3.5 GHz; path-loss exponent 3.0-3.8 (urban); CPU 1 GHz with
~1e-28 effective switched-capacitance kappa.
"""
from __future__ import annotations

import numpy as np

K_BOLTZMANN = 1.380649e-23      # J/K
SPEED_OF_LIGHT = 2.998e8        # m/s
T_REF_K = 290.0                 # standard noise reference temperature


def dbm_to_watt(dbm: float) -> float:
    return 10.0 ** ((float(dbm) - 30.0) / 10.0)


def watt_to_dbm(w: float) -> float:
    return 10.0 * np.log10(float(w)) + 30.0


def thermal_noise_power_w(bandwidth_hz: float, noise_figure_db: float = 7.0,
                          temperature_k: float = T_REF_K) -> float:
    """Johnson-Nyquist receiver noise power sigma^2 = k_B * T * F * B  (Watts)."""
    f_lin = 10.0 ** (float(noise_figure_db) / 10.0)
    return K_BOLTZMANN * float(temperature_k) * f_lin * float(bandwidth_hz)


def free_space_pl0_db(d0_m: float, carrier_ghz: float) -> float:
    """Free-space path loss (dB) at the reference distance d0 (Friis)."""
    fc = float(carrier_ghz) * 1e9
    return 20.0 * np.log10(4.0 * np.pi * float(d0_m) * fc / SPEED_OF_LIGHT)


def path_loss_db(dist_m, carrier_ghz: float, exponent: float, d0_m: float = 1.0):
    """Log-distance path loss (dB): PL(d) = PL0(d0) + 10 n log10(d/d0)."""
    d = np.maximum(np.asarray(dist_m, dtype=float), float(d0_m))
    return free_space_pl0_db(d0_m, carrier_ghz) + 10.0 * float(exponent) * np.log10(d / float(d0_m))


def path_gain_linear(dist_m, carrier_ghz: float, exponent: float, d0_m: float = 1.0):
    """Large-scale channel power gain (dimensionless) = 10^(-PL/10)."""
    return 10.0 ** (-path_loss_db(dist_m, carrier_ghz, exponent, d0_m) / 10.0)


def received_snr_db(power_w: float, gain_linear, sigma2_w: float):
    """True received SNR in dB = 10 log10(P * g / sigma^2)."""
    g = np.asarray(gain_linear, dtype=float)
    return 10.0 * np.log10(np.clip(float(power_w) * g / float(sigma2_w), 1e-30, None))

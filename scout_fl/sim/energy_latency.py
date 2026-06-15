"""Per-round energy (J) and latency (s) for the selected clients (simple models).

* sensing:  fixed e_sense, t_sense per client.
* compute:  t = cpu_cycles / cpu_freq;  E = kappa * cpu_cycles * cpu_freq^2.
* uplink:   rate = bandwidth * log2(1 + P*g/sigma2);  t = model_bits / rate;  E = P*t.

Round latency = max over selected of (t_sense + t_comp + t_comm)  (parallel clients);
round energy = sum over selected. Values are normalized dev-scale (calibrate with
the channel module at scale-up); used for the latency/energy constraints.
"""
from __future__ import annotations

import numpy as np


def round_energy_latency(selected, channel_gains, *, power: float = 1.0, sigma2: float = 1.0,
                         bandwidth: float = 1.0e6, model_bits: float = 1.0e5,
                         cpu_cycles: float = 1.0e7, cpu_freq: float = 1.0e9,
                         kappa: float = 1.0e-27, e_sense: float = 0.1,
                         t_sense: float = 0.01) -> dict:
    idx = list(selected)
    if not idx:
        return {"latency": 0.0, "energy": 0.0, "t_comm_max": 0.0, "min_rate": 0.0}
    g = np.asarray(channel_gains, dtype=float)[idx]
    rate = bandwidth * np.log2(1.0 + power * g / sigma2)            # bits/s
    t_comm = model_bits / np.clip(rate, 1e-9, None)
    e_comm = power * t_comm
    t_comp = cpu_cycles / cpu_freq
    e_comp = kappa * cpu_cycles * cpu_freq ** 2
    latency = float(np.max(t_sense + t_comp + t_comm))
    energy = float(np.sum(e_sense + e_comp + e_comm))
    return {"latency": latency, "energy": energy,
            "t_comm_max": float(np.max(t_comm)), "min_rate": float(np.min(rate))}

"""Client-position leakage accountant (design §0.2.4, §2.3).

Every round a client transmits, the BS's return/CSI processing yields Fisher
information about *that client's own position* — the same rank-structured FIM
as target sensing (sim/fim.py), pointed at the client:

    J_k^leak(t) = atten * snr_up_k * (k_range u u^T + (k_angle / r_k^2) v v^T)

with u the BS->client radial unit vector, v tangential, r_k the BS-client
range, and snr_up_k = P g_k / sigma2 the uplink SNR. ``atten`` in [0, 1] models
mitigations (M2 dither / M3 obfuscation reduce the BS- or eavesdropper-usable
information; 1.0 = unmitigated).

Leakage composes ADDITIVELY in Fisher units across rounds (theory T-C3):
J_k^leak(1..T) = sum_t contributions. The operational privacy level is the
client-position CRB floor

    r_k = sqrt( tr( (J0 + J_k^leak(1..T))^{-1} ) )   [meters]

where J0 = I / prior_var encodes the threat model's coarse registration-level
prior (A1 already knows ~cell-level location, default std 100 m per axis —
design §2.2), so r_k starts near ~141 m (sqrt of 2 axes) and shrinks as rounds
leak information. Report median and worst-client (min) r_k.

The 2-client hand-computable case in tests/test_infra_leakage.py must match to
numerical tolerance (Phase-0 gate, design §3.1).
"""
from __future__ import annotations

import numpy as np


def client_leak_fim(clients: np.ndarray, bs: np.ndarray,
                    k_range: float, k_angle: float) -> np.ndarray:
    """Unit-SNR per-client leakage FIM stack (K, 2, 2); multiply by SNR per use."""
    clients = np.asarray(clients, dtype=float)
    bs = np.asarray(bs, dtype=float)
    delta = clients - bs[None, :]                       # BS -> client
    rng = np.clip(np.linalg.norm(delta, axis=1), 1e-9, None)
    u = delta / rng[:, None]
    v = np.stack([-u[:, 1], u[:, 0]], axis=1)
    a_r = float(k_range)
    a_a = float(k_angle) / rng ** 2
    uuT = u[:, :, None] * u[:, None, :]
    vvT = v[:, :, None] * v[:, None, :]
    return a_r * uuT + a_a[:, None, None] * vvT         # (K, 2, 2), unit SNR


class LeakageAccountant:
    """Accumulates per-client position-FIM leakage over a campaign."""

    def __init__(self, clients: np.ndarray, bs: np.ndarray, *,
                 k_range: float = 1.0, k_angle: float = 0.05,
                 prior_std_m: float = 100.0) -> None:
        self.K = int(np.asarray(clients).shape[0])
        self.unit_fim = client_leak_fim(clients, bs, k_range, k_angle)  # (K,2,2)
        self.J0 = np.eye(2) / float(prior_std_m) ** 2
        self.J = np.zeros((self.K, 2, 2))
        self.rounds_observed = np.zeros(self.K, dtype=int)

    # ---------------------------------------------------------------- account
    def round_contribution(self, k: int, snr_up: float, atten: float = 1.0) -> np.ndarray:
        """The 2x2 FIM this client would leak this round (not yet accumulated)."""
        return float(atten) * float(snr_up) * self.unit_fim[k]

    def observe(self, selected, snr_up: np.ndarray, atten=1.0) -> None:
        """Accumulate one round of leakage for the selected clients.

        ``atten`` may be a scalar or per-client (K,) attenuation in [0, 1].
        """
        snr_up = np.asarray(snr_up, dtype=float)
        att = np.broadcast_to(np.asarray(atten, dtype=float), (self.K,))
        for k in selected:
            k = int(k)
            self.J[k] += self.round_contribution(k, snr_up[k], att[k])
            self.rounds_observed[k] += 1

    # ---------------------------------------------------------------- metrics
    def crb_floor(self) -> np.ndarray:
        """Per-client position CRB floor r_k in meters -> (K,)."""
        acc = self.J0[None] + self.J
        inv = np.linalg.inv(acc)
        tr = np.trace(inv, axis1=-2, axis2=-1)
        return np.sqrt(np.clip(tr, 0.0, None))

    def trace_leak(self) -> np.ndarray:
        """tr(J_k^leak) per client (Fisher units; the cap variable for M1) -> (K,)."""
        return np.trace(self.J, axis1=-2, axis2=-1)

    def projected_trace(self, k: int, snr_up: float, atten: float = 1.0) -> float:
        """tr of client k's cumulative leakage IF selected this round."""
        return float(np.trace(self.J[k] + self.round_contribution(k, snr_up, atten)))

    def projected_crb_floor(self, k: int, snr_up: float, atten: float = 1.0) -> float:
        """Client k's position CRB floor r_k (m) IF selected this round.

        This is the CORRECT cap variable for M1: the leakage FIM is strongly
        anisotropic (radial >> cross-range), so a trace cap is dominated by the
        uninformative-for-privacy radial axis. Capping on the actual CRB floor
        (cross-range-limited) enforces the real privacy guarantee.
        """
        acc = self.J0 + self.J[k] + self.round_contribution(k, snr_up, atten)
        return float(np.sqrt(np.trace(np.linalg.inv(acc))))

    def summary(self) -> dict:
        r = self.crb_floor()
        return {"leak_r_median": float(np.median(r)),
                "leak_r_min": float(r.min()),
                "leak_trace_max": float(self.trace_leak().max())}


def cap_from_crb_floor(r_floor_m: float) -> float:
    """Trace-cap J_max (Fisher units) that guarantees CRB floor >= r_floor_m.

    For a 2x2 PSD J, tr(J^{-1}) >= 4 / tr(J) (equality iff isotropic), so
    tr(J) <= 4 / r^2  =>  r_k = sqrt(tr(J^{-1})) >= r. The cap is therefore
    conservative-safe: enforcing tr(J_k) <= 4/r^2 guarantees the floor; the
    exact r_k is always computed from the matrix inverse for reporting.
    """
    return 4.0 / float(r_floor_m) ** 2

"""M2 — aggregate-transparent (zero-sum) correlated dithering (design §2.4.2, §2.5 T-C2).

Selected clients apply correlated perturbations delta_k generated from pairwise
shared seeds with sum_k delta_k = 0 (SecAgg-style pairwise cancellation, but
analog / at the physical layer). The AirComp aggregate — hence the FL update and
epsilon_agg — is *exactly* unchanged under perfect sync, while each individual
waveform is scrambled, inflating an external eavesdropper's (A2) per-client CRB.

Construction (pairwise antisymmetric masks; exact zero-sum by construction):

    delta_k = sum_{j != k} sign(k, j) * PRF(seed_{k,j})            (dim d)

with sign(k, j) = +1 if k < j else -1 and seed_{k,j} = seed_{j,k}, so every
pair contributes +m and -m and the sum over the selected set telescopes to 0.
sigma_d^2 is the per-coordinate dither variance knob.

Under sync offset (imperfect timing), a fraction of the mask survives: the
residual aggregate error scales with sigma_sync (quantified in E-C2a; the exact
bound is T-C2a). ``aggregate_residual`` models this so the honesty experiment is
computable without FL training.
"""
from __future__ import annotations

import numpy as np


def _prf(seed_pair: int, dim: int) -> np.ndarray:
    """Deterministic pseudo-random unit-variance vector for a client pair."""
    return np.random.default_rng(int(seed_pair) & 0x7FFFFFFFFFFFFFFF).standard_normal(dim)


class ZeroSumDither:
    """Pairwise-cancelling analog dither over a selected client set.

    ``base_seed`` fixes the shared-seed matrix so the scheme is reproducible and
    every (i, j) uses the SAME seed on both sides (that symmetry is what makes
    the masks antisymmetric and the sum exactly zero).
    """

    def __init__(self, dim: int, sigma_d: float, base_seed: int = 0) -> None:
        self.dim = int(dim)
        self.sigma_d = float(sigma_d)
        self.base_seed = int(base_seed)

    def _pair_seed(self, i: int, j: int) -> int:
        a, b = (i, j) if i < j else (j, i)
        return (self.base_seed * 1_000_003 + a * 100_003 + b) & 0x7FFFFFFFFFFFFFFF

    def masks(self, selected) -> np.ndarray:
        """Return the (|S|, dim) per-client dither masks; rows sum to exactly 0."""
        S = list(selected)
        n = len(S)
        out = np.zeros((n, self.dim))
        # per-coordinate variance sigma_d^2 requires each pairwise draw scaled by
        # sigma_d / sqrt(n-1): var(delta_k) = (n-1) * (sigma_d^2/(n-1)) = sigma_d^2.
        scale = self.sigma_d / np.sqrt(max(n - 1, 1))
        for a in range(n):
            for b in range(a + 1, n):
                m = scale * _prf(self._pair_seed(S[a], S[b]), self.dim)
                out[a] += m
                out[b] -= m
        return out

    def aggregate_residual(self, selected, sigma_sync: float = 0.0,
                           rng: np.random.Generator | None = None) -> np.ndarray:
        """Aggregate of the dithers under a sync offset (design §2.6 E-C2a).

        Perfect sync (sigma_sync = 0) -> exactly zero. With a per-client random
        gain error (1 + eps_k), eps_k ~ N(0, sigma_sync^2), the surviving
        aggregate is sum_k eps_k * delta_k (first-order); returned as a length-d
        vector so ||.|| gives the residual AirComp error.
        """
        M = self.masks(selected)
        if sigma_sync <= 0.0:
            return M.sum(axis=0)                          # exactly ~0 (float round-off only)
        rng = rng or np.random.default_rng(self.base_seed + 12345)
        eps = rng.normal(0.0, float(sigma_sync), size=M.shape[0])
        return (eps[:, None] * M).sum(axis=0)


def eavesdropper_crb_inflation(sigma_d: float, snr_eve: float, n_receivers: int = 1,
                               base_fisher: float = 1.0) -> float:
    """A2 per-client localization CRB inflation factor vs the no-dither case (T-C2b).

    The eavesdropper matched-filters a client whose waveform now carries an
    unknown dither of variance sigma_d^2. Treating the dither as extra
    observation noise, the client-parameter Fisher information is scaled by
    1 / (1 + sigma_d^2 * snr_eve), so CRB inflates by (1 + sigma_d^2 * snr_eve).
    ``n_receivers`` colluding receivers average down the dither nuisance, so the
    effective inflation is 1 + sigma_d^2 * snr_eve / n_receivers.
    """
    n = max(int(n_receivers), 1)
    return float(1.0 + base_fisher * sigma_d ** 2 * snr_eve / n)

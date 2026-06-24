"""VISMAYA-FL: server-side generative model for innovation-driven client selection.

Motivation (bias-variance decomposition of ISAC-FL error)
---------------------------------------------------------
JEDI-FL minimizes the posterior VARIANCE (entropy / EIG): it reduces uncertainty
about the unknown (θ, φ). In stationary environments this is optimal. In
non-stationary ISAC (moving targets, concept drift), the server's prior becomes
BIASED — it is confident but wrong. JEDI then keeps selecting clients that confirm
the stale prior. VISMAYA corrects this by selecting clients whose transmissions
deviate most from the server's prediction (the Kalman innovation signal).

Total model error = Bias² + Variance.
JEDI handles Variance. VISMAYA handles Bias.

The server maintains two online predictors calibrated from past rounds:

1. Sensing predictor (Kalman-style target covariance):
   P_m = current target-state covariance. Clients whose FIM J_{k,m} is aligned
   with the uncertain directions of P_m contribute the most innovation:
       Ω_k^S = Σ_m w_m · tr(J_{k,m} @ P_m)
   This equals the expected Mahalanobis distance of client k's measurement from
   the server's prediction. Grows when targets move (process noise inflates P_m).

2. Learning predictor (online ridge regression on gradient norm):
   A lightweight RLS model predicts each client's gradient norm from observed
   features (loss, grad-norm, channel, recency). Clients with large predicted
   gradient energy OR high prediction uncertainty score highly:
       Ω_k^L = ĝ_k² + σ²_{g,k}    (σ²_{g,k} = 1/(1 + n_selected_k))

3. Joint synergy (EMA, the ISAC-specific term):
   In ISAC, a target that has moved drives BOTH sensing error (echo is in the wrong
   place) and learning error (client's data distribution shifted). Clients showing
   correlated sensing and learning surprises score a synergy bonus:
       Syn_k ≈ EMA_t(|Ω_k^S_err| · |ĝ_k - actual_gnorm_k|)

VISMAYA score: V_k = Ω_k^S + ρ_v · Ω_k^L + β · Syn_k
Selection: Top-K by V_k (modular objective → top-K IS the optimal solution).
Complexity: O(N log K) vs JEDI's O(NK).

Inspired by: Kalman innovation (Kalman 1960), Expected Model Change (Settles 2010),
Predictive Coding (Rao & Ballard 1999), Bayesian Surprise (Itti & Koch 2009).
ISAC contribution: joint sensing-learning synergy from shared environment dynamics.
"""
from __future__ import annotations

import numpy as np

from scout_fl.objectives.twin import ResidualTwin


def _zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    s = float(a.std())
    return (a - float(a.mean())) / s if s > 1e-9 else np.zeros_like(a)


class ClientGenerativeModel:
    """Online predictive model used by VISMAYA-FL to score all clients each round.

    Parameters
    ----------
    K : int
        Number of clients.
    M : int
        Number of sensing targets.
    fim_stack : (K, M, 2, 2) ndarray
        Per-client per-target FIM matrices (from sim/fim.py). Cached once per seed;
        assumed stationary (the target-state uncertainty P_m handles dynamics).
    j0 : (M, 2, 2) ndarray
        Prior FIM per target (J_0 = prior_fim * I from sim/fim.py).
    weights : (M,) ndarray
        Per-target weights matching SensingUtility.w.
    rho_v : float
        Sensing-to-learning normalizer. Auto-calibrated from the first round's
        innovation magnitudes so both terms are commensurate (not a preference
        weight). Pass a fixed value to disable auto-calibration.
    beta : float
        Weight of the joint synergy term [0, 1].
    process_noise : float
        Per-round process-noise variance added to P_m to model target motion.
        0.0 = stationary environment; >0 inflates P_m between rounds (VISMAYA's
        headline advantage over JEDI emerges most clearly when this is >0).
    ema_alpha : float
        Exponential moving average factor for synergy updates.
    """

    def __init__(self, K: int, M: int, fim_stack: np.ndarray, j0: np.ndarray,
                 weights: np.ndarray, *, rho_v: float = 1.0, beta: float = 0.3,
                 process_noise: float = 0.0, ema_alpha: float = 0.1,
                 sense_scale: float = 1.0) -> None:
        self.K, self.M = int(K), int(M)
        self.J_km = np.asarray(fim_stack, dtype=float)       # (K, M, 2, 2)
        self.J0 = np.asarray(j0, dtype=float)                # (M, 2, 2)
        self.w = np.asarray(weights, dtype=float)            # (M,)
        self.rho_v = float(rho_v)
        self.beta = float(beta)
        self.process_noise = float(process_noise)
        self.ema_alpha = float(ema_alpha)
        # Ablation knob: 0.0 disables the sensing innovation term (vismaya_learn_only).
        self.sense_scale = float(sense_scale)

        # --- Sensing: target state covariance P_m (Kalman) ---
        # Init: P_m = J_0^{-1} (full prior uncertainty; decreases as targets are sensed)
        _reg = 1e-9 * np.eye(2)
        self.P = np.stack([np.linalg.inv(self.J0[m] + _reg) for m in range(M)])  # (M,2,2)
        # Accumulated FIM per target; grows as selected clients observe targets
        self.J_acc = self.J0.copy()                           # (M, 2, 2)

        # --- Learning: online ridge regression (gradient norm predictor) ---
        # Feature dim = 5: [bias, loss_z, grad_norm_z, channel_z, recency]
        self._feat_dim = 5
        self.twin = ResidualTwin(dim=self._feat_dim, l2=1.0)
        self.n_selected = np.zeros(K, dtype=float)           # selection counts (recency)
        self.total_rounds = 0

        # --- Synergy: EMA of sensing_surprise × learning_surprise ---
        self.synergy = np.zeros(K, dtype=float)

        # auto-calibrate flag
        self._rho_calibrated = (rho_v != 1.0)                # skip if user fixed rho_v

    # ---------------------------------------------------------------- scoring

    def sensing_innovations(self) -> np.ndarray:
        """Ω_k^S = Σ_m w_m · tr(J_{k,m} @ P_m)  →  (K,) array.

        Measures how much of each target's current uncertainty (P_m) each client
        can resolve given its geometry (J_{k,m}). Clients viewing uncertain targets
        along their most uncertain directions score highest.
        """
        omega_s = np.zeros(self.K, dtype=float)
        for m in range(self.M):
            JP = self.J_km[:, m] @ self.P[m]                # (K, 2, 2)
            traces = np.trace(JP, axis1=-2, axis2=-1)        # (K,)
            omega_s += float(self.w[m]) * np.maximum(traces, 0.0)
        return omega_s

    def learning_innovations(self, features: np.ndarray) -> np.ndarray:
        """Ω_k^L = ĝ_k² + σ²_{g,k}  →  (K,) array.

        ĝ_k = ridge-predicted gradient norm (how large a model update to expect).
        σ²_{g,k} = 1/(1 + n_selected_k): clients rarely selected have high
        uncertainty, boosting exploration naturally.
        """
        preds = np.array([self.twin.predict(features[k]) for k in range(self.K)])
        preds = np.maximum(preds, 0.0)
        mean_pred = float(preds.mean())
        deviation_sq = (preds - mean_pred) ** 2
        uncertainty = 1.0 / (1.0 + self.n_selected)
        return deviation_sq + uncertainty

    def score_all(self, features: np.ndarray) -> np.ndarray:
        """V_k = Ω_k^S + ρ_v · Ω_k^L + β · Syn_k  →  (K,) VISMAYA scores.

        Parameters
        ----------
        features : (K, 5) ndarray
            Built by :meth:`build_features` from probe losses, grad norms, channels.
        """
        omega_s = self.sensing_innovations() * self.sense_scale
        omega_l = self.learning_innovations(features)

        # One-shot auto-calibration: equalize sensing and learning innovation scales.
        # Skipped when sense_scale=0 (learn-only ablation) to avoid div-by-zero.
        if not self._rho_calibrated and self.sense_scale > 0.0:
            s_mean = float(omega_s.mean()) + 1e-12
            l_mean = float(omega_l.mean()) + 1e-12
            self.rho_v = s_mean / l_mean
            self._rho_calibrated = True
        elif not self._rho_calibrated:
            self._rho_calibrated = True

        return omega_s + self.rho_v * omega_l + self.beta * self.synergy

    # ---------------------------------------------------------------- updates

    def update_sensing(self, selected: list[int]) -> None:
        """Kalman-update P_m after selected clients report sensing measurements.

        Posterior precision = J_acc + Σ_{k in S} J_{k,m}; covariance = inverse.
        Adds process noise Q = process_noise * I to model target motion.
        """
        for m in range(self.M):
            for k in selected:
                self.J_acc[m] += self.J_km[k, m]
            _reg = 1e-9 * np.eye(2)
            self.P[m] = np.linalg.inv(self.J_acc[m] + _reg)
            if self.process_noise > 0.0:
                self.P[m] += self.process_noise * np.eye(2)
                # Keep J_acc consistent with inflated P_m
                self.J_acc[m] = np.linalg.inv(self.P[m] + _reg)

    def update_learning(self, selected: list[int], features: np.ndarray,
                        grad_norms: np.ndarray) -> None:
        """Update ridge regression from observed (feature, grad_norm) pairs."""
        for k in selected:
            self.twin.update(features[k], float(grad_norms[k]))
            self.n_selected[k] += 1.0

    def update_synergy(self, selected: list[int], features: np.ndarray,
                       grad_norms: np.ndarray, omega_s_pre: np.ndarray) -> None:
        """Update joint synergy EMA for selected clients.

        Synergy_k ← (1-α) · Synergy_k  +  α · sensing_innov_k · learning_err_k
        """
        preds = np.array([self.twin.predict(features[k]) for k in range(self.K)])
        preds = np.maximum(preds, 0.0)
        for k in selected:
            learning_err = abs(float(grad_norms[k]) - float(preds[k]))
            cross = float(omega_s_pre[k]) * learning_err
            self.synergy[k] = (1.0 - self.ema_alpha) * self.synergy[k] + self.ema_alpha * cross

    def update(self, selected: list[int], features: np.ndarray,
               grad_norms: np.ndarray) -> None:
        """Full per-round update (call AFTER training + collecting grad norms).

        Order: synergy first (uses pre-update omega_s) → sensing → learning.
        """
        self.total_rounds += 1
        omega_s_pre = self.sensing_innovations()             # before sensing update
        self.update_synergy(selected, features, grad_norms, omega_s_pre)
        self.update_sensing(selected)
        self.update_learning(selected, features, grad_norms)

    # ---------------------------------------------------------------- helpers

    def build_features(self, losses: np.ndarray, grad_norms: np.ndarray,
                       channel_gains: np.ndarray) -> np.ndarray:
        """Build (K, 5) feature matrix for the learning innovation predictor.

        Columns: [1, zscore(loss), zscore(grad_norm), zscore(channel), recency].
        recency = (total_rounds - n_selected_k) / (total_rounds + 1), in [0, 1]:
        1.0 for clients never selected (maximum recency/staleness = high uncertainty).
        """
        recency = np.clip(
            (self.total_rounds - self.n_selected) / (self.total_rounds + 1.0),
            0.0, 1.0)
        return np.column_stack([
            np.ones(self.K),
            _zscore(losses),
            _zscore(grad_norms),
            _zscore(channel_gains),
            recency,
        ])

    def diagnostics(self) -> dict:
        """Per-round VISMAYA diagnostics for paper figures."""
        omega_s = self.sensing_innovations()
        return {
            "vis_omega_s_mean": round(float(omega_s.mean()), 5),
            "vis_omega_s_max": round(float(omega_s.max()), 5),
            "vis_synergy_mean": round(float(self.synergy.mean()), 6),
            "vis_p_trace_mean": round(float(np.mean([np.trace(self.P[m]) for m in range(self.M)])), 5),
            "vis_n_seen_frac": round(float((self.n_selected > 0).mean()), 4),
        }

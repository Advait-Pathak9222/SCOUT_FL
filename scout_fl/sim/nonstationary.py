"""Opt-in non-stationary wireless/ISAC regime engine for RECA-FL experiments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scout_fl.sim.fim import per_client_target_fim
from scout_fl.sim.geometry import pairwise_geometry
from scout_fl.sim.sensing import sensing_snr


@dataclass
class RegimeState:
    name: str
    channel_gains: np.ndarray
    sensing_snr: np.ndarray
    fim: np.ndarray
    affected_clients: np.ndarray
    diagnostics: dict


class WirelessISACNonstationarity:
    """Create round-wise channel, sensing, target-motion, and rare-data shifts.

    This class is deliberately opt-in. If ``nonstationarity.enabled`` is absent
    or false, the FL runner keeps its original stationary behavior.
    """

    def __init__(self, cfg, scn, base_g, seed: int) -> None:
        ns = cfg.get("nonstationarity", {}) or {}
        self.enabled = bool(ns.get("enabled", False))
        self.cfg = cfg
        self.ns = ns
        self.scn = scn
        self.base_g = np.asarray(base_g, dtype=float).copy()
        self.base_snr = np.asarray(scn.snr, dtype=float).copy()
        self.base_fim = np.asarray(scn.fim, dtype=float).copy()
        self.base_targets = None if getattr(scn, "targets", None) is None else np.asarray(scn.targets, dtype=float).copy()
        self.rng = np.random.default_rng(int(seed) + 991)
        K = int(scn.K)
        frac = float(ns.get("affected_fraction", 0.35))
        n_aff = max(1, min(K, int(round(frac * K))))
        mode = str(ns.get("affected_mode", "cluster0"))
        if mode == "random":
            self.affected = np.sort(self.rng.choice(K, size=n_aff, replace=False))
        elif getattr(scn, "cluster_assignment", None) is not None:
            clusters = np.asarray(scn.cluster_assignment)
            primary = int(clusters[0])
            idx = np.where(clusters == primary)[0]
            self.affected = idx[:n_aff] if idx.size >= n_aff else np.arange(n_aff)
        else:
            self.affected = np.arange(n_aff)
        self.shift_types = set(ns.get("shift_types", []))

    def round_state(self, t: int) -> RegimeState:
        if not self.enabled:
            return RegimeState("normal", self.base_g.copy(), self.base_snr.copy(),
                               self.base_fim.copy(), np.array([], dtype=int), {})

        name = self._regime_name(t)
        active = name != "normal"
        severity = self._severity(name)
        g = self.base_g.copy()
        snr = self.base_snr.copy()
        targets = None if self.base_targets is None else self.base_targets.copy()

        if active and "target_motion" in self.shift_types and targets is not None:
            targets = self._move_targets(targets, t, name, severity)
            snr = self._sensing_from_targets(targets)

        if active and "fading_distribution" in self.shift_types:
            g = self._apply_fading_shift(g, severity)

        if active and "blockage" in self.shift_types:
            att = float(self.ns.get("blockage_attenuation", 0.12))
            sense_att = float(self.ns.get("blockage_sensing_attenuation", 0.35))
            g[self.affected] *= att
            snr[self.affected] *= sense_att

        if active and "sensing_clutter" in self.shift_types:
            clutter = float(self.ns.get("clutter_snr_factor", 0.55))
            snr *= clutter
            snr[self.affected] *= float(self.ns.get("affected_clutter_factor", 0.65))

        fim = self._fim_from_snr(targets, snr)
        diag = {
            "ns_enabled": True,
            "ns_regime": name,
            "ns_active": bool(active),
            "ns_affected_clients": int(self.affected.size if active else 0),
            "ns_channel_gain_mean": round(float(np.mean(g)), 8),
            "ns_channel_gain_min": round(float(np.min(g)), 8),
            "ns_sensing_snr_mean": round(float(np.mean(snr)), 6),
            "ns_target_motion": bool(active and "target_motion" in self.shift_types),
            "ns_blockage": bool(active and "blockage" in self.shift_types),
            "ns_fading_shift": bool(active and "fading_distribution" in self.shift_types),
            "ns_sensing_clutter": bool(active and "sensing_clutter" in self.shift_types),
            "ns_rare_class_shift": bool(active and "rare_class_region" in self.shift_types),
            "ns_rare_loss_boost": 0.0,
            "ns_rare_embedding_shift": 0.0,
        }
        return RegimeState(name, g, snr, fim, self.affected.copy() if active else np.array([], dtype=int), diag)

    def adjust_probes(self, t: int, losses, embeddings):
        name = self._regime_name(t)
        if name == "normal" or "rare_class_region" not in self.shift_types:
            return losses, embeddings, {}
        losses = np.asarray(losses, dtype=float).copy()
        embeddings = np.asarray(embeddings, dtype=float).copy()
        severity = self._severity(name)
        loss_boost = float(self.ns.get("rare_loss_boost", 0.35)) * severity
        emb_shift = float(self.ns.get("rare_embedding_shift", 0.25)) * severity
        losses[self.affected] += loss_boost
        if embeddings.ndim == 2 and embeddings.shape[0] > 0:
            direction = np.zeros(embeddings.shape[1], dtype=float)
            direction[: max(1, embeddings.shape[1] // 3)] = emb_shift
            embeddings[self.affected] += direction
        return losses, embeddings, {
            "ns_rare_loss_boost": round(float(loss_boost), 6),
            "ns_rare_embedding_shift": round(float(emb_shift), 6),
        }

    def _regime_name(self, t: int) -> str:
        first = int(self.ns.get("first_shift_round", self.cfg.get("first_shift_round", 10**9)))
        first_end = int(self.ns.get("first_shift_end", self.cfg.get("first_shift_end", first)))
        second = int(self.ns.get("second_shift_round", self.cfg.get("second_shift_round", 10**9)))
        if first <= t < first_end:
            return "A"
        if t >= second:
            return "A_prime"
        return "normal"

    def _severity(self, name: str) -> float:
        base = float(self.ns.get("severity", 1.0))
        if name == "A_prime":
            return base * float(self.ns.get("aprime_similarity", 0.85))
        return base

    def _move_targets(self, targets, t: int, name: str, severity: float) -> np.ndarray:
        area = np.asarray(self.cfg.network.get("area_size", [100.0, 100.0]), dtype=float)
        first = int(self.ns.get("first_shift_round", self.cfg.get("first_shift_round", 0)))
        dt = max(0, t - first + 1)
        speed = float(self.ns.get("target_speed", 1.5)) * severity
        direction = np.array([1.0, 0.35 if name == "A" else 0.45], dtype=float)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        moved = targets + dt * speed * direction
        return np.clip(moved, 0.0, area)

    def _sensing_from_targets(self, targets) -> np.ndarray:
        geom = pairwise_geometry(self.scn.clients, targets)
        rcs = np.ones(int(self.scn.M), dtype=float)
        return sensing_snr(geom, self.cfg.sensing.ref_snr_db,
                           self.cfg.sensing.pathloss_exponent,
                           rcs=rcs, ref_distance=self.cfg.sensing.ref_distance)

    def _fim_from_snr(self, targets, snr) -> np.ndarray:
        if targets is not None:
            geom = pairwise_geometry(self.scn.clients, targets)
            return per_client_target_fim(geom, snr, self.cfg.sensing.k_range,
                                         self.cfg.sensing.k_angle)
        ratio = snr / np.clip(self.base_snr, 1e-12, None)
        return self.base_fim * ratio[..., None, None]

    def _apply_fading_shift(self, g, severity: float) -> np.ndarray:
        sigma = float(self.ns.get("fading_lognormal_sigma", 0.6)) * severity
        factors = self.rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=g.shape)
        g = g * factors
        g[self.affected] *= float(self.ns.get("affected_fading_factor", 0.55))
        return np.clip(g, 1e-18, None)

"""Context adapter bank and matching logic for RECA-FL."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RegimeSignature:
    embedding: np.ndarray
    grad_residual: np.ndarray
    sensing_residual: np.ndarray
    residual_mean: np.ndarray
    residual_var: np.ndarray
    n: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "RegimeSignature":
        return cls(
            embedding=np.asarray(d["embedding"], dtype=float),
            grad_residual=np.asarray(d["grad_residual"], dtype=float),
            sensing_residual=np.asarray(d["sensing_residual"], dtype=float),
            residual_mean=np.asarray(d["residual_mean"], dtype=float),
            residual_var=np.asarray(d["residual_var"], dtype=float),
            n=int(d.get("n", 1)),
        )


@dataclass
class ContextAdapter:
    adapter_id: str
    signature: RegimeSignature
    state: str = "active"
    use_count: int = 0
    evidence: float = 0.0


@dataclass
class MatchResult:
    adapter_id: str | None
    confidence: float
    embedding_distance: float
    residual_cosine: float
    residual_kl: float


class AdapterMatcher:
    """Compute adapter-regime similarity and a confidence in [0, 1]."""

    def __init__(self, tau_reuse: float = 0.7, distance_scale: float = 1.0) -> None:
        self.tau_reuse = float(tau_reuse)
        self.distance_scale = float(distance_scale)

    def match(self, current: RegimeSignature, adapters: list[ContextAdapter]) -> MatchResult:
        candidates = [a for a in adapters if a.state == "consolidated"]
        if not candidates:
            return MatchResult(None, 0.0, float("inf"), 0.0, float("inf"))
        best = max((self._score(current, a) for a in candidates), key=lambda x: x.confidence)
        if best.confidence < self.tau_reuse:
            return MatchResult(None, best.confidence, best.embedding_distance,
                               best.residual_cosine, best.residual_kl)
        return best

    def _score(self, cur: RegimeSignature, adapter: ContextAdapter) -> MatchResult:
        mem = adapter.signature
        emb_dist = _norm_distance(cur.embedding, mem.embedding)
        emb_conf = float(np.exp(-emb_dist / max(self.distance_scale, 1e-9)))
        cur_res = np.concatenate([cur.grad_residual, cur.sensing_residual])
        mem_res = np.concatenate([mem.grad_residual, mem.sensing_residual])
        cos = _cosine(cur_res, mem_res)
        cos_conf = 0.5 * (cos + 1.0)
        kl = _diag_kl(cur.residual_mean, cur.residual_var, mem.residual_mean, mem.residual_var)
        kl_conf = float(np.exp(-min(kl, 50.0)))
        conf = float(np.clip(0.4 * emb_conf + 0.35 * cos_conf + 0.25 * kl_conf, 0.0, 1.0))
        return MatchResult(adapter.adapter_id, conf, emb_dist, cos, kl)


class ContextAdapterBank:
    """Lifecycle manager for RECA context adapters."""

    def __init__(self, tau_reuse: float = 0.7, consolidate_after: int = 2,
                 quarantine_below: float = -0.1) -> None:
        self.adapters: list[ContextAdapter] = []
        self.matcher = AdapterMatcher(tau_reuse=tau_reuse)
        self.consolidate_after = int(consolidate_after)
        self.quarantine_below = float(quarantine_below)
        self._next_id = 0

    def spawn(self, signature: RegimeSignature) -> ContextAdapter:
        adapter = ContextAdapter(adapter_id=f"A{self._next_id}", signature=signature)
        self._next_id += 1
        self.adapters.append(adapter)
        return adapter

    def match(self, signature: RegimeSignature) -> MatchResult:
        return self.matcher.match(signature, self.adapters)

    def route_or_spawn(self, signature: RegimeSignature) -> tuple[ContextAdapter, MatchResult, bool]:
        match = self.match(signature)
        if match.adapter_id is not None:
            adapter = self.get(match.adapter_id)
            adapter.use_count += 1
            return adapter, match, True
        return self.spawn(signature), match, False

    def update_evidence(self, adapter_id: str, progress_delta: float) -> None:
        adapter = self.get(adapter_id)
        adapter.evidence += float(progress_delta)
        if adapter.evidence < self.quarantine_below:
            adapter.state = "quarantined"
        elif adapter.use_count + adapter.signature.n >= self.consolidate_after and adapter.evidence >= 0.0:
            adapter.state = "consolidated"

    def get(self, adapter_id: str) -> ContextAdapter:
        for adapter in self.adapters:
            if adapter.adapter_id == adapter_id:
                return adapter
        raise KeyError(adapter_id)


def _norm_distance(a, b) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(x.size, y.size)
    if n == 0:
        return 0.0
    return float(np.linalg.norm(x[:n] - y[:n]) / np.sqrt(n))


def _cosine(a, b) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(x.size, y.size)
    if n == 0:
        return 0.0
    x, y = x[:n], y[:n]
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 0.0 if denom <= 1e-12 else float(np.clip((x @ y) / denom, -1.0, 1.0))


def _diag_kl(mu0, var0, mu1, var1) -> float:
    m0, v0 = np.asarray(mu0, dtype=float), np.asarray(var0, dtype=float)
    m1, v1 = np.asarray(mu1, dtype=float), np.asarray(var1, dtype=float)
    n = min(m0.size, v0.size, m1.size, v1.size)
    if n == 0:
        return 0.0
    m0, v0, m1, v1 = m0[:n], np.clip(v0[:n], 1e-9, None), m1[:n], np.clip(v1[:n], 1e-9, None)
    return float(0.5 * np.sum(np.log(v1 / v0) + (v0 + (m0 - m1) ** 2) / v1 - 1.0))

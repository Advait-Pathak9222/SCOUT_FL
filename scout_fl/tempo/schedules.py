"""Open-loop lambda_s(t) schedules: E-T1 oracle hand-schedules + naive controls.

A schedule maps round t (0-indexed) to lambda_s in [0, 1], the sensing weight fed
to the inner selector; (1 - lambda_s) is the learning weight. These are OPEN-LOOP
(no feedback) — the controllers in controllers.py are the closed-loop siblings.

E-T1 grid (design §1.5):
  learn_then_sense  : lambda_s = 0 for t < tau, else lam_high      (tau in {30,50,75,100,120})
  sense_then_learn  : the reverse (control condition; expected bad — validates T1 direction)
  bursting          : sensing bursts of length b every period p     (b in {3,5,10}, p in {15,25,50})
Naive controls (design §1.6, anti-triviality):
  roundrobin        : 1:1 alternation (lambda_s = t mod 2)
  random            : lambda_s(t) ~ U[0,1] (seeded)
  linear_anneal     : lambda_s = t / (T-1)  (0 -> 1)
  two_phase         : fixed 50/50 split (lambda_s = 0.5 constant)
Tuned-static / any constant:
  static            : lambda_s = lam (grid-searched best constant is the fair static competitor)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Schedule:
    """A named lambda_s(t) with its parameters recorded for the artifact."""
    name: str
    fn: Callable[[int, int], float]
    params: dict = field(default_factory=dict)

    def __call__(self, t: int, T: int) -> float:
        return float(np.clip(self.fn(t, T), 0.0, 1.0))


def learn_then_sense(tau: int, lam_high: float = 1.0) -> Schedule:
    return Schedule("learn_then_sense",
                    lambda t, T: 0.0 if t < tau else lam_high,
                    {"tau": int(tau), "lam_high": float(lam_high)})


def sense_then_learn(tau: int, lam_high: float = 1.0) -> Schedule:
    return Schedule("sense_then_learn",
                    lambda t, T: lam_high if t < tau else 0.0,
                    {"tau": int(tau), "lam_high": float(lam_high)})


def bursting(burst_len: int, period: int, lam_high: float = 1.0) -> Schedule:
    return Schedule("bursting",
                    lambda t, T: lam_high if (t % period) < burst_len else 0.0,
                    {"burst_len": int(burst_len), "period": int(period),
                     "lam_high": float(lam_high)})


def roundrobin(lam_high: float = 1.0) -> Schedule:
    return Schedule("roundrobin", lambda t, T: lam_high * (t % 2), {"lam_high": float(lam_high)})


def random_schedule(seed: int = 0) -> Schedule:
    # Precompute a deterministic sequence so repeated calls at the same t agree.
    rng = np.random.default_rng(int(seed))
    cache: dict[int, float] = {}

    def fn(t, T):
        while len(cache) <= t:
            cache[len(cache)] = float(rng.uniform(0.0, 1.0))
        return cache[t]

    return Schedule("random", fn, {"seed": int(seed)})


def linear_anneal() -> Schedule:
    return Schedule("linear_anneal", lambda t, T: t / max(T - 1, 1), {})


def two_phase(lam: float = 0.5) -> Schedule:
    return Schedule("two_phase", lambda t, T: lam, {"lam": float(lam)})


def static(lam: float) -> Schedule:
    return Schedule("static", lambda t, T: lam, {"lam": float(lam)})


def from_spec(spec: dict) -> Schedule:
    """Build a Schedule from a config dict {kind: ..., <params>}. 'name' is ignored."""
    kind = spec["kind"]
    p = {k: v for k, v in spec.items() if k not in ("kind", "name")}
    builders = {
        "learn_then_sense": learn_then_sense, "sense_then_learn": sense_then_learn,
        "bursting": bursting, "roundrobin": roundrobin, "random": random_schedule,
        "linear_anneal": linear_anneal, "two_phase": two_phase, "static": static,
    }
    if kind not in builders:
        raise ValueError(f"unknown schedule kind {kind!r}; have {list(builders)}")
    return builders[kind](**p)

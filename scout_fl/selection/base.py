"""Selector interface and result container.

A Selector maps a round context (utility object, budget, RNG, optional scores
and constraints) to a chosen client subset plus timing/diagnostic info. The
``**ctx`` signature keeps a single call site usable across very different
policies; the composite ``RoundContext`` is formalized at the FL-integration
step.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SelectionResult:
    """Output of a selection policy for one round."""

    selected: list[int]
    select_time: float = 0.0
    info: dict = field(default_factory=dict)


class Selector:
    """Abstract base class for all selection policies."""

    name: str = "base"

    def select(self, **ctx) -> SelectionResult:  # pragma: no cover - interface
        raise NotImplementedError

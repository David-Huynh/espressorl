from __future__ import annotations

from typing import Protocol

from espresso_rl.domain.models import Recommendation
from espresso_rl.domain.optimization import OptimizationContext, PriorPoint


class Optimizer(Protocol):
    def recommend(self, context: OptimizationContext) -> Recommendation:
        ...


class StatefulOptimizerHandoff(RuntimeError):
    """Signals that a stateless optimizer must hand control to a stateful mode."""

    def __init__(self, target_mode: str, reason: str) -> None:
        super().__init__(reason)
        self.target_mode = target_mode
        self.reason = reason


class PriorProvider(Protocol):
    def get_prior_points(self, context: OptimizationContext) -> list[PriorPoint]:
        ...

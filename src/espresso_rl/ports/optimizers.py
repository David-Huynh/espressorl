from __future__ import annotations

from typing import Protocol

from espresso_rl.domain.models import Recommendation
from espresso_rl.domain.optimization import OptimizationContext


class Optimizer(Protocol):
    def recommend(self, context: OptimizationContext) -> Recommendation:
        ...

"""Minimal example for developing a stateless EspressoRL optimizer.

This module is intentionally outside ``src`` and is not a selectable runtime
mode. Production optimizers belong behind an EspressoRL port and must be wired
explicitly after their algorithm, persistence, and safety tests are complete.
"""

from __future__ import annotations

from espresso_rl.domain.models import Recommendation
from espresso_rl.domain.optimization import OptimizationContext
from espresso_rl.domain.safety import validate_recommendation


class ExampleOptimizer:
    """Implement the optimizer port without depending on any machine adapter."""

    def recommend(self, context: OptimizationContext) -> Recommendation:
        recommendation = self._recommend_candidate(context)
        validate_recommendation(
            context.current_recipe,
            recommendation,
            context.safety_bounds,
        )
        return recommendation

    def _recommend_candidate(self, context: OptimizationContext) -> Recommendation:
        raise NotImplementedError("supply a tested optimizer algorithm")

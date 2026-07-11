from __future__ import annotations

from espresso_rl.domain.models import (
    Recommendation,
    RecommendationMode,
    RecommendationStatus,
)
from espresso_rl.domain.optimization import OptimizationContext


class DeterministicOptimizer:
    """Stateless test double for application tests that do not exercise CPBO."""

    def __init__(self) -> None:
        self.contexts: list[OptimizationContext] = []

    def recommend(self, context: OptimizationContext) -> Recommendation:
        self.contexts.append(context)
        current = context.current_recipe
        return Recommendation(
            recommendation_id=f"rec_test_{len(self.contexts)}",
            created_at=context.now,
            updated_at=context.now,
            expires_at=context.now + 3600,
            install_id=context.install_id,
            machine_id=context.machine_id,
            bean_context_id=context.bean_context_id,
            grinder_context_id=context.grinder_context_id,
            grind_delta_steps_from_current=0.0,
            grind_delta_um_from_current=0.0,
            projected_relative_step_from_reference=(
                current.relative_grind_steps_from_reference
            ),
            projected_relative_grind_um_from_reference=(
                current.relative_grind_um_from_reference
            ),
            next_dose_g=current.dose_g,
            target_yield_g=current.target_yield_g,
            target_ratio=current.target_yield_g / current.dose_g,
            mode=RecommendationMode.ZERO_IMMEDIATE_BO,
            confidence=0.5,
            reason="Deterministic application test recommendation.",
            status=RecommendationStatus.PENDING,
            source_shot_id=context.shots[-1].shot_id if context.shots else None,
            grinder_step_direction=current.grinder_step_direction,
            grinder_adjustment_mode=current.grinder_adjustment_mode,
        )

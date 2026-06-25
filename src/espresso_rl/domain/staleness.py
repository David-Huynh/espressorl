from __future__ import annotations

from dataclasses import dataclass

from .models import Recipe, Recommendation, RecommendationStatus, StaleCheck


@dataclass(frozen=True)
class StaleRecommendationPolicy:
    manual_grind_change_steps: float = 2.0
    manual_dose_change_g: float = 0.5
    manual_yield_change_g: float = 3.0


def check_recommendation_staleness(
    recommendation: Recommendation,
    now: int,
    bean_context_id: str | None,
    grinder_context_id: str | None = None,
    current_recipe: Recipe | None = None,
    policy: StaleRecommendationPolicy = StaleRecommendationPolicy(),
) -> StaleCheck:
    if recommendation.status in {
        RecommendationStatus.EXPIRED,
        RecommendationStatus.IGNORED,
        RecommendationStatus.USED,
        RecommendationStatus.SUPERSEDED,
    }:
        return StaleCheck(True, f"status:{recommendation.status.value}")
    if recommendation.expires_at is not None and recommendation.expires_at <= now:
        return StaleCheck(True, "expired")
    if recommendation.bean_context_id != bean_context_id:
        return StaleCheck(True, "bean_context_changed")
    if recommendation.grinder_context_id != grinder_context_id:
        return StaleCheck(True, "grinder_context_changed")
    if current_recipe is not None and _large_manual_change(recommendation, current_recipe, policy):
        return StaleCheck(True, "manual_recipe_changed")
    return StaleCheck(False)


def _large_manual_change(
    recommendation: Recommendation,
    current_recipe: Recipe,
    policy: StaleRecommendationPolicy,
) -> bool:
    grind_changed = (
        abs(current_recipe.relative_grind_steps_from_reference - recommendation.projected_relative_step_from_reference)
        > policy.manual_grind_change_steps
    )
    dose_changed = abs(current_recipe.dose_g - recommendation.next_dose_g) > policy.manual_dose_change_g
    yield_changed = (
        abs(current_recipe.target_yield_g - recommendation.target_yield_g)
        > policy.manual_yield_change_g
    )
    return grind_changed or dose_changed or yield_changed

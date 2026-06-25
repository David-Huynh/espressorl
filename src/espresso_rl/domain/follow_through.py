from __future__ import annotations

from dataclasses import dataclass

from .models import (
    FollowThroughResult,
    FollowThroughState,
    Recommendation,
    RecommendationDecision,
    ShotRecord,
)


@dataclass(frozen=True)
class FollowThroughTolerances:
    relative_grind_steps_from_reference: float = 0.5
    dose_g: float = 0.2
    yield_g: float = 1.5


def infer_follow_through(
    shot: ShotRecord,
    recommendation: Recommendation | None,
    decision: RecommendationDecision = RecommendationDecision.UNKNOWN,
    tolerances: FollowThroughTolerances = FollowThroughTolerances(),
) -> FollowThroughResult:
    if recommendation is None:
        return FollowThroughResult(FollowThroughState.UNKNOWN, 0.2)

    if decision in {RecommendationDecision.IGNORED, RecommendationDecision.DISMISSED}:
        return FollowThroughResult(FollowThroughState.NOT_FOLLOWED, 0.0)

    if shot.relative_grind_steps_from_reference is None or shot.beverage_out_g is None:
        return FollowThroughResult(FollowThroughState.UNKNOWN, 0.2)

    grind_match = abs(shot.relative_grind_steps_from_reference - recommendation.projected_relative_step_from_reference) <= tolerances.relative_grind_steps_from_reference
    dose_match = abs(shot.dose_in_g - recommendation.next_dose_g) <= tolerances.dose_g
    yield_match = abs(shot.beverage_out_g - recommendation.target_yield_g) <= tolerances.yield_g
    matches = sum((grind_match, dose_match, yield_match))

    if matches == 3:
        return FollowThroughResult(FollowThroughState.FOLLOWED, 1.0)
    if matches >= 1:
        return FollowThroughResult(FollowThroughState.PARTIALLY_FOLLOWED, matches / 3.0)
    return FollowThroughResult(FollowThroughState.NOT_FOLLOWED, 0.0)


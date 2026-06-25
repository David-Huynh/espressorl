from __future__ import annotations

import math
from typing import Any, Iterable

from espresso_rl.application.community_priors import (
    COMMUNITY_PRIOR_OBSERVATION_NOISE,
    MAX_COMMUNITY_PRIOR_CONFIDENCE,
    community_prior_context_key,
)
from espresso_rl.domain.models import FollowThroughState, RecommendationDecision, ShotRecord, ShotType
from espresso_rl.domain.optimization import OptimizationContext, PriorPoint
from espresso_rl.ports.community import CommunityWarehouseRepository
from espresso_rl.ports.optimizers import PriorProvider


class CompositePriorProvider:
    def __init__(self, providers: Iterable[PriorProvider]) -> None:
        self._providers = list(providers)

    def get_prior_points(self, context: OptimizationContext) -> list[PriorPoint]:
        points: list[PriorPoint] = []
        for provider in self._providers:
            points.extend(provider.get_prior_points(context))
        return points


class LocalHistoryPriorProvider:
    def __init__(self, limit: int = 5) -> None:
        self._limit = limit

    def get_prior_points(self, context: OptimizationContext) -> list[PriorPoint]:
        shots = [
            shot
            for shot in context.shots
            if _shot_is_usable_local_prior(shot) and shot.relative_grind_steps_from_reference is not None
        ]
        shots = sorted(shots, key=_shot_prior_score, reverse=True)[: self._limit]
        points: list[PriorPoint] = []
        for shot in shots:
            target_ratio = shot.target_ratio or shot.target_yield_g / shot.dose_in_g
            points.append(
                PriorPoint(
                    grind_delta_um_from_current=(shot.relative_grind_steps_from_reference - context.current_recipe.relative_grind_steps_from_reference)
                    * context.current_recipe.microns_per_step,
                    dose_g=shot.dose_in_g,
                    target_yield_g=shot.target_yield_g,
                    target_ratio=target_ratio,
                    predicted_reward=max(0.0, min(1.0, shot.reward or 0.0)),
                    confidence=max(0.0, min(0.8, shot.reward_confidence * 0.8)),
                    observation_noise=0.05,
                    source="local_history",
                    reason="High-confidence local history point.",
                )
            )
        return points


class CommunityPriorProvider:
    def __init__(
        self,
        repository: CommunityWarehouseRepository,
        *,
        max_points: int = 5,
    ) -> None:
        self._repository = repository
        self._max_points = max_points

    def get_prior_points(self, context: OptimizationContext) -> list[PriorPoint]:
        context_key = community_prior_context_key(
            {
                "machine_adapter": context.machine_adapter or "unknown",
                "dose_in_g": context.current_recipe.dose_g,
                "target_yield_g": context.current_recipe.target_yield_g,
                "target_ratio": context.current_recipe.target_ratio,
            }
        )
        priors = self._repository.list_community_priors(context_key, limit=self._max_points)
        points: list[PriorPoint] = []
        for prior in priors:
            points.extend(_prior_points_from_community_prior(prior.context_key, prior.prior_json, prior.confidence, context_key))
        return points[: self._max_points]


def _prior_points_from_community_prior(
    prior_context_key: str,
    prior_json: dict[str, Any],
    prior_confidence: float,
    expected_context_key: str,
) -> list[PriorPoint]:
    if prior_context_key != expected_context_key:
        return []
    if not isinstance(prior_json, dict):
        return []
    if prior_json.get("context_key") != expected_context_key:
        return []
    zero_trust = prior_json.get("zero_trust")
    if not isinstance(zero_trust, dict):
        return []
    if zero_trust.get("validated_training_rows_only") is not True:
        return []
    if zero_trust.get("revalidated_before_aggregation") is not True:
        return []

    points_json = prior_json.get("points")
    if not isinstance(points_json, list):
        return []

    points: list[PriorPoint] = []
    for point in points_json[:5]:
        if not isinstance(point, dict):
            continue
        prior_point = _community_point_from_json(point, prior_confidence)
        if prior_point is not None:
            points.append(prior_point)
    return points


def _community_point_from_json(point: dict[str, Any], prior_confidence: float) -> PriorPoint | None:
    grind_delta_um_from_current = _number(point.get("grind_delta_um_from_current"))
    dose_g = _number(point.get("dose_g"))
    target_yield_g = _number(point.get("target_yield_g"))
    target_ratio = _number(point.get("target_ratio"))
    predicted_reward = _number(point.get("predicted_reward"))
    confidence = _number(point.get("confidence"))
    observation_noise = _number(point.get("observation_noise"))
    if None in {grind_delta_um_from_current, dose_g, target_yield_g, target_ratio, predicted_reward, confidence, observation_noise}:
        return None
    if not 5.0 <= dose_g <= 30.0:
        return None
    if not 5.0 <= target_yield_g <= 100.0:
        return None
    if not 1.2 <= target_ratio <= 3.5:
        return None
    if not 0.0 <= predicted_reward <= 1.0:
        return None

    capped_confidence = min(float(prior_confidence), confidence, MAX_COMMUNITY_PRIOR_CONFIDENCE)
    capped_noise = max(observation_noise, COMMUNITY_PRIOR_OBSERVATION_NOISE)
    if capped_confidence <= 0:
        return None

    return PriorPoint(
        grind_delta_um_from_current=grind_delta_um_from_current,
        dose_g=dose_g,
        target_yield_g=target_yield_g,
        target_ratio=target_ratio,
        predicted_reward=predicted_reward,
        confidence=capped_confidence,
        observation_noise=capped_noise,
        source="community",
        reason="Weak zero-trust community prior.",
    )


def _shot_is_usable_local_prior(shot: ShotRecord) -> bool:
    if shot.shot_type != ShotType.ESPRESSO:
        return False
    if shot.exclude_from_local_optimization or shot.optimization_weight <= 0:
        return False
    if shot.reward is None:
        return False
    if shot.recommendation_decision in {RecommendationDecision.IGNORED, RecommendationDecision.DISMISSED}:
        return False
    if shot.recommendation_followed == FollowThroughState.NOT_FOLLOWED:
        return False
    return True


def _shot_prior_score(shot: ShotRecord) -> float:
    return (shot.reward or 0.0) * max(shot.reward_confidence, 0.05) * max(shot.optimization_weight, 0.0)


def _number(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number

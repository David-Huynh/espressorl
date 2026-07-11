from __future__ import annotations

from typing import Iterable

from espresso_rl.domain.models import ShotRecord, ShotType
from espresso_rl.domain.optimization import OptimizationContext, PriorPoint
from espresso_rl.ports.optimizers import PriorProvider


DEFAULT_LOCAL_HISTORY_PRIOR_POINTS = 64
MIN_LOCAL_HISTORY_RANK_SCALE = 0.35
MAX_LOCAL_HISTORY_OBSERVATION_NOISE = 0.75


class CompositePriorProvider:
    def __init__(self, providers: Iterable[PriorProvider]) -> None:
        self._providers = list(providers)

    def get_prior_points(self, context: OptimizationContext) -> list[PriorPoint]:
        points: list[PriorPoint] = []
        for provider in self._providers:
            points.extend(provider.get_prior_points(context))
        return points


class LocalHistoryPriorProvider:
    def __init__(self, limit: int = DEFAULT_LOCAL_HISTORY_PRIOR_POINTS) -> None:
        self._limit = limit

    def get_prior_points(self, context: OptimizationContext) -> list[PriorPoint]:
        shots = [
            shot
            for shot in context.shots
            if _shot_is_usable_local_prior(shot)
            and shot.grind_observed
            and shot.relative_grind_steps_from_reference is not None
            and shot.dose_observed
            and shot.realized_yield_observed
        ]
        shots = sorted(shots, key=_shot_prior_score, reverse=True)[: self._limit]
        points: list[PriorPoint] = []
        for rank, shot in enumerate(shots, start=1):
            rank_scale = max(MIN_LOCAL_HISTORY_RANK_SCALE, 1.0 / (rank ** 0.5))
            target_yield_g = shot.realized_yield_g
            target_ratio = target_yield_g / shot.dose_in_g
            points.append(
                PriorPoint(
                    grind_delta_um_from_current=(shot.relative_grind_steps_from_reference - context.current_recipe.relative_grind_steps_from_reference)
                    * context.current_recipe.microns_per_step
                    * context.current_recipe.grinder_direction_sign,
                    dose_g=shot.dose_in_g,
                    target_yield_g=target_yield_g,
                    target_ratio=target_ratio,
                    predicted_reward=max(0.0, min(1.0, shot.reward or 0.0)),
                    confidence=max(0.0, min(0.8, shot.reward_confidence * 0.8)) * rank_scale,
                    observation_noise=min(MAX_LOCAL_HISTORY_OBSERVATION_NOISE, 0.05 / rank_scale),
                    source="local_history",
                    reason="High-confidence local history point.",
                )
            )
        return points


def _shot_is_usable_local_prior(shot: ShotRecord) -> bool:
    if shot.shot_type != ShotType.ESPRESSO:
        return False
    if shot.exclude_from_local_optimization or shot.optimization_weight <= 0:
        return False
    if shot.reward is None:
        return False
    return True


def _shot_prior_score(shot: ShotRecord) -> float:
    return (shot.reward or 0.0) * max(shot.reward_confidence, 0.05) * max(shot.optimization_weight, 0.0)

from __future__ import annotations

import torch
from torch import Tensor

from espresso_rl.domain.cpbo import (
    PreferenceLabel,
    TrustRegionAction,
    TrustRegionState,
    TrustRegionTransition,
    normalized_recipe,
)
from espresso_rl.optimizers.cpbo_config import TrustRegionConfig


CPBO_BATCH_SIZE = 1
CPBO_DIMENSION = 3


def update_trust_region(
    state: TrustRegionState,
    label: PreferenceLabel,
    *,
    candidate_center: tuple[float, float, float],
    config: TrustRegionConfig,
) -> TrustRegionState:
    label = PreferenceLabel(label)
    if state.locally_converged:
        raise ValueError("a converged local trust region must be resumed before updating")

    if label == PreferenceLabel.NEW_BETTER:
        center = normalized_recipe(candidate_center)
        success_count = state.success_count + 1
        failure_count = 0
        action = TrustRegionAction.IMPROVED
    else:
        center = state.center
        success_count = 0
        failure_count = state.failure_count + 1
        action = TrustRegionAction.NON_IMPROVEMENT

    length = state.length
    locally_converged = False
    if success_count >= config.success_tolerance:
        length = min(2.0 * length, config.maximum_length)
        success_count = 0
        failure_count = 0
        action = TrustRegionAction.EXPANDED
    elif failure_count >= config.failure_tolerance:
        length = max(config.minimum_length, length / 2.0)
        success_count = 0
        failure_count = 0
        locally_converged = length <= config.minimum_length
        action = (
            TrustRegionAction.CONVERGED
            if locally_converged
            else TrustRegionAction.CONTRACTED
        )
    transition = TrustRegionTransition(
        action=action,
        label=label,
        center_before=state.center,
        center_after=center,
        length_before=state.length,
        length_after=length,
        success_count=success_count,
        failure_count=failure_count,
        success_tolerance=config.success_tolerance,
        failure_tolerance=config.failure_tolerance,
        minimum_length=config.minimum_length,
        maximum_length=config.maximum_length,
    )
    return TrustRegionState(
        center=center,
        length=length,
        success_count=success_count,
        failure_count=failure_count,
        restart_pending=False,
        locally_converged=locally_converged,
        transitions=(*state.transitions, transition),
    )


def resume_trust_region(
    state: TrustRegionState,
    *,
    center: tuple[float, float, float],
    config: TrustRegionConfig,
    after_comparison_id: str | None,
    incumbent_shot_id: str,
    created_at: int,
    control_event_id: str | None = None,
) -> TrustRegionState:
    if not state.locally_converged:
        raise ValueError("only a converged local trust region can be resumed")
    resumed_center = normalized_recipe(center)
    transition = TrustRegionTransition(
        action=TrustRegionAction.RESUMED,
        center_before=state.center,
        center_after=resumed_center,
        length_before=state.length,
        length_after=config.initial_length,
        success_count=0,
        failure_count=0,
        success_tolerance=config.success_tolerance,
        failure_tolerance=config.failure_tolerance,
        minimum_length=config.minimum_length,
        maximum_length=config.maximum_length,
        incumbent_shot_id=incumbent_shot_id,
        after_comparison_id=after_comparison_id,
        control_event_id=control_event_id,
        created_at=created_at,
    )
    return TrustRegionState(
        center=resumed_center,
        length=config.initial_length,
        success_count=0,
        failure_count=0,
        restart_pending=False,
        locally_converged=False,
        transitions=(*state.transitions, transition),
    )


def trust_region_bounds(
    state: TrustRegionState,
    raw_lengthscales: Tensor | tuple[float, float, float],
    config: TrustRegionConfig,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    lengthscales = torch.as_tensor(raw_lengthscales, dtype=torch.float64).reshape(-1)
    if lengthscales.shape != (CPBO_DIMENSION,) or torch.any(~torch.isfinite(lengthscales)):
        raise ValueError("trust-region lengthscales must contain three finite values")
    if torch.any(lengthscales <= 0):
        raise ValueError("trust-region lengthscales must be positive")
    geometric_mean = torch.exp(torch.mean(torch.log(lengthscales)))
    shape = lengthscales / geometric_mean
    shape = shape.clamp(config.shape_factor_min, config.shape_factor_max)
    shape = shape / torch.exp(torch.mean(torch.log(shape)))
    shape = shape.clamp(config.shape_factor_min, config.shape_factor_max)
    center = torch.tensor(state.center, dtype=torch.float64)
    half_width = 0.5 * state.length * shape
    lower = torch.clamp(center - half_width, 0.0, 1.0)
    upper = torch.clamp(center + half_width, 0.0, 1.0)
    if torch.any(lower > upper) or torch.any(~torch.isfinite(lower)) or torch.any(~torch.isfinite(upper)):
        raise FloatingPointError("trust-region bounds are invalid")
    return (
        tuple(float(value) for value in lower),
        tuple(float(value) for value in upper),
    )


def validate_q_one() -> None:
    if CPBO_BATCH_SIZE != 1 or CPBO_DIMENSION != 3:
        raise AssertionError("espresso CPBO requires q=1 in a three-dimensional recipe space")

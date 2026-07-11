from __future__ import annotations

import math

import torch
from torch import Tensor

from espresso_rl.domain.cpbo import PreferenceLabel, TrustRegionState, normalized_recipe
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
    restart_was_pending = state.restart_pending
    if label == PreferenceLabel.NEW_BETTER:
        center = normalized_recipe(candidate_center)
        success_count = state.success_count + 1
        failure_count = 0
    else:
        center = state.center
        success_count = 0
        failure_count = state.failure_count + 1

    length = state.length
    if success_count >= config.success_tolerance:
        length = min(2.0 * length, config.maximum_length)
        success_count = 0
        failure_count = 0
    elif failure_count >= config.failure_tolerance:
        length = length / 2.0
        success_count = 0
        failure_count = 0

    restart_pending = False if restart_was_pending else state.restart_pending
    if length < config.minimum_length:
        length = config.initial_length
        success_count = 0
        failure_count = 0
        restart_pending = True
    return TrustRegionState(
        center=center,
        length=length,
        success_count=success_count,
        failure_count=failure_count,
        restart_pending=restart_pending,
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
    expected_failure = math.ceil(max(4 / CPBO_BATCH_SIZE, CPBO_DIMENSION / CPBO_BATCH_SIZE))
    if expected_failure != 4:
        raise AssertionError("unexpected TuRPBO failure tolerance")

from __future__ import annotations

from .models import Recipe, Recommendation, SafetyBounds


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_candidate_recipe(
    current: Recipe,
    candidate_grind_steps: float,
    candidate_dose_g: float,
    candidate_target_yield_g: float,
    bounds: SafetyBounds,
) -> Recipe:
    max_step = bounds.max_grind_delta_steps
    grind_steps = clamp(
        candidate_grind_steps,
        current.grind_steps - max_step,
        current.grind_steps + max_step,
    )
    dose_g = clamp(
        candidate_dose_g,
        max(bounds.dose_min_g, current.dose_g - bounds.max_dose_delta_g),
        min(bounds.dose_max_g, current.dose_g + bounds.max_dose_delta_g),
    )
    target_yield_g = clamp(
        candidate_target_yield_g,
        max(bounds.target_yield_min_g, current.target_yield_g - bounds.max_yield_delta_g),
        min(bounds.target_yield_max_g, current.target_yield_g + bounds.max_yield_delta_g),
    )
    ratio = clamp(
        target_yield_g / dose_g,
        bounds.target_ratio_min,
        bounds.target_ratio_max,
    )
    target_yield_g = ratio * dose_g
    return Recipe(
        grind_steps=round(grind_steps),
        grinder_step_size_um=current.grinder_step_size_um,
        dose_g=round(dose_g * 2.0) / 2.0,
        target_yield_g=round(target_yield_g, 1),
        target_ratio=ratio,
    )


def validate_recommendation(
    current: Recipe,
    recommendation: Recommendation,
    bounds: SafetyBounds,
) -> None:
    if abs(recommendation.next_grind_steps - current.grind_steps) > bounds.max_grind_delta_steps:
        raise ValueError("recommendation exceeds grind delta safety bound")
    if abs(recommendation.next_dose_g - current.dose_g) > bounds.max_dose_delta_g + 1e-9:
        raise ValueError("recommendation exceeds dose delta safety bound")
    if abs(recommendation.target_yield_g - current.target_yield_g) > bounds.max_yield_delta_g + 1e-9:
        raise ValueError("recommendation exceeds yield delta safety bound")
    if not bounds.dose_min_g <= recommendation.next_dose_g <= bounds.dose_max_g:
        raise ValueError("recommendation dose outside global bounds")
    if not bounds.target_yield_min_g <= recommendation.target_yield_g <= bounds.target_yield_max_g:
        raise ValueError("recommendation yield outside global bounds")
    if not bounds.target_ratio_min <= recommendation.target_ratio <= bounds.target_ratio_max:
        raise ValueError("recommendation ratio outside global bounds")


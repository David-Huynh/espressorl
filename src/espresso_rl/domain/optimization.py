from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import Recipe, Recommendation, SafetyBounds, ShotRecord


@dataclass(frozen=True)
class PriorPoint:
    grind_delta_um: float
    dose_g: float
    target_yield_g: float
    target_ratio: float
    predicted_reward: float
    confidence: float
    observation_noise: float
    source: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.dose_g <= 0:
            raise ValueError("prior dose_g must be positive")
        if self.target_yield_g <= 0:
            raise ValueError("prior target_yield_g must be positive")
        if self.target_ratio <= 0:
            raise ValueError("prior target_ratio must be positive")
        if not 0.0 <= self.predicted_reward <= 1.0:
            raise ValueError("prior predicted_reward must be between 0 and 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("prior confidence must be between 0 and 1")
        if self.observation_noise <= 0:
            raise ValueError("prior observation_noise must be positive")
        if not self.source:
            raise ValueError("prior source is required")


@dataclass(frozen=True)
class OptimizationContext:
    install_id: str
    machine_id: str
    bean_context_id: str | None
    machine_adapter: str | None
    current_recipe: Recipe
    shots: Sequence[ShotRecord]
    safety_bounds: SafetyBounds
    now: int
    last_recommendation: Recommendation | None = None
    grinder_context_id: str | None = None
    prior_points: Sequence[PriorPoint] = ()

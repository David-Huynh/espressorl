from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import Recipe, Recommendation, SafetyBounds, ShotRecord


@dataclass(frozen=True)
class OptimizationContext:
    install_id: str
    machine_id: str
    bean_context_id: str | None
    current_recipe: Recipe
    shots: Sequence[ShotRecord]
    safety_bounds: SafetyBounds
    now: int
    last_recommendation: Recommendation | None = None


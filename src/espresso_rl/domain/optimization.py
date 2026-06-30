from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import Recipe, Recommendation, SafetyBounds, ShotRecord

DEFAULT_OPTIMIZER_MODE = "bayesian_optimization"
OPTIMIZER_MODE_DREAMER_V3_ACTIVE = "dreamer_v3"
OPTIMIZER_MODE_DREAMER_V3_SHADOW = "dreamer_v3_shadow"
OPTIMIZER_FAMILY_BAYESIAN_OPTIMIZATION = "bayesian_optimization"
OPTIMIZER_FAMILY_TRUST_REGION_BO = "trust_region_bo"
OPTIMIZER_FAMILY_TRUST_REGION_PPO = "trust_region_ppo"
OPTIMIZER_FAMILY_DREAMER_V3 = "dreamer_v3"
VALID_OPTIMIZER_MODES = {
    DEFAULT_OPTIMIZER_MODE,
    OPTIMIZER_MODE_DREAMER_V3_ACTIVE,
    OPTIMIZER_MODE_DREAMER_V3_SHADOW,
}
SHOT_LEVEL_OPTIMIZER_FAMILIES = frozenset(
    {
        OPTIMIZER_FAMILY_BAYESIAN_OPTIMIZATION,
        OPTIMIZER_FAMILY_TRUST_REGION_BO,
        OPTIMIZER_FAMILY_TRUST_REGION_PPO,
    }
)
ADAPTIVE_PROFILE_CONTROL_OPTIMIZER_FAMILIES = frozenset({OPTIMIZER_FAMILY_DREAMER_V3})
OPTIMIZER_MODE_ALIASES = {
    "bo": DEFAULT_OPTIMIZER_MODE,
    "conservative_bo": DEFAULT_OPTIMIZER_MODE,
    "dreamer": OPTIMIZER_MODE_DREAMER_V3_ACTIVE,
    "dreamerv3": OPTIMIZER_MODE_DREAMER_V3_ACTIVE,
    "dreamer_v3_active": OPTIMIZER_MODE_DREAMER_V3_ACTIVE,
    "dreamer_shadow": OPTIMIZER_MODE_DREAMER_V3_SHADOW,
    "dreamerv3_shadow": OPTIMIZER_MODE_DREAMER_V3_SHADOW,
}


def normalize_optimizer_mode(value: object) -> str:
    mode = str(value or DEFAULT_OPTIMIZER_MODE).strip().lower()
    mode = OPTIMIZER_MODE_ALIASES.get(mode, mode)
    if mode not in VALID_OPTIMIZER_MODES:
        raise ValueError("optimizer_mode is invalid")
    return mode


def optimizer_family_allows_adaptive_profile_control(value: object) -> bool:
    family = str(value or "").strip().lower()
    return family in ADAPTIVE_PROFILE_CONTROL_OPTIMIZER_FAMILIES


def require_adaptive_profile_control_optimizer(value: object) -> None:
    if not optimizer_family_allows_adaptive_profile_control(value):
        raise ValueError("adaptive profile control is only available for DreamerV3")


@dataclass(frozen=True)
class PriorPoint:
    grind_delta_um_from_current: float
    dose_g: float
    target_yield_g: float
    target_ratio: float
    predicted_reward: float
    confidence: float
    observation_noise: float
    source: str
    reason: str | None = None
    grind_observed: bool = True
    dose_observed: bool = True
    target_yield_observed: bool = True

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
        for field_name in ("grind_observed", "dose_observed", "target_yield_observed"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"prior {field_name} must be boolean")
        if not any((self.grind_observed, self.dose_observed, self.target_yield_observed)):
            raise ValueError("prior must observe at least one action field")


@dataclass(frozen=True)
class PriorSignal:
    grind_direction: int
    ratio_direction: int
    dose_direction: int
    confidence: float
    observation_noise: float
    source: str
    reason: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("grind_direction", "ratio_direction", "dose_direction"):
            if getattr(self, field_name) not in {-1, 0, 1}:
                raise ValueError(f"{field_name} must be -1, 0, or 1")
        if not any((self.grind_direction, self.ratio_direction, self.dose_direction)):
            raise ValueError("prior signal must define at least one direction")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("prior signal confidence must be between 0 and 1")
        if self.observation_noise <= 0:
            raise ValueError("prior signal observation_noise must be positive")
        if not self.source:
            raise ValueError("prior signal source is required")


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
    prior_signals: Sequence[PriorSignal] = ()

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from espresso_rl.domain.models import Recipe, SafetyBounds

DREAMER_ACTION_SCHEMA_VERSION = 1
DREAMER_ACTION_FORMAT = "espresso_rl_dreamer_action_v1"

_ALLOWED_ACTION_FIELDS = {
    "format",
    "schema_version",
    "grind_delta_steps_from_current",
    "next_dose_g",
    "target_yield_g",
    "target_ratio",
    "confidence",
    "reason",
}


@dataclass(frozen=True)
class DreamerActionCandidate:
    grind_delta_steps_from_current: int
    next_dose_g: float
    target_yield_g: float
    confidence: float
    reason: str = "DreamerV3 candidate."
    schema_version: int = DREAMER_ACTION_SCHEMA_VERSION
    target_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.schema_version != DREAMER_ACTION_SCHEMA_VERSION:
            raise ValueError("Dreamer action schema_version is unsupported")
        object.__setattr__(
            self,
            "grind_delta_steps_from_current",
            _integer_steps(self.grind_delta_steps_from_current, "grind_delta_steps_from_current"),
        )
        object.__setattr__(self, "next_dose_g", _finite_float(self.next_dose_g, "next_dose_g"))
        object.__setattr__(self, "target_yield_g", _finite_float(self.target_yield_g, "target_yield_g"))
        if self.next_dose_g <= 0:
            raise ValueError("Dreamer next_dose_g must be positive")
        if self.target_yield_g <= 0:
            raise ValueError("Dreamer target_yield_g must be positive")
        ratio = self.target_ratio
        if ratio is None:
            ratio = self.target_yield_g / self.next_dose_g
        else:
            ratio = _finite_float(ratio, "target_ratio")
            if ratio <= 0:
                raise ValueError("Dreamer target_ratio must be positive")
            if abs(ratio - (self.target_yield_g / self.next_dose_g)) > 0.05:
                raise ValueError("Dreamer target_ratio must match target_yield_g / next_dose_g")
        object.__setattr__(self, "target_ratio", ratio)
        object.__setattr__(self, "confidence", _finite_float(self.confidence, "confidence"))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Dreamer confidence must be between 0 and 1")
        if not isinstance(self.reason, str) or len(self.reason) > 240 or any(ord(ch) < 32 for ch in self.reason):
            raise ValueError("Dreamer reason must be a safe short string")


def dreamer_action_from_payload(payload: dict[str, Any]) -> DreamerActionCandidate:
    if not isinstance(payload, dict):
        raise ValueError("Dreamer action payload must be an object")
    unknown = sorted(str(key) for key in payload if key not in _ALLOWED_ACTION_FIELDS)
    if unknown:
        raise ValueError(f"Dreamer action contains unsupported fields: {', '.join(unknown[:5])}")
    if payload.get("format", DREAMER_ACTION_FORMAT) != DREAMER_ACTION_FORMAT:
        raise ValueError("Dreamer action format is unsupported")
    grind_delta = _integer_steps(
        payload.get("grind_delta_steps_from_current"),
        "grind_delta_steps_from_current",
    )
    return DreamerActionCandidate(
        schema_version=int(payload.get("schema_version", DREAMER_ACTION_SCHEMA_VERSION)),
        grind_delta_steps_from_current=grind_delta,
        next_dose_g=payload.get("next_dose_g"),
        target_yield_g=payload.get("target_yield_g"),
        target_ratio=payload.get("target_ratio"),
        confidence=payload.get("confidence", 0.0),
        reason=str(payload.get("reason") or "DreamerV3 candidate."),
    )


def dreamer_action_to_recipe(
    action: DreamerActionCandidate,
    *,
    current: Recipe,
    bounds: SafetyBounds,
) -> Recipe:
    validate_dreamer_action(action, current=current, bounds=bounds)
    return Recipe(
        relative_grind_steps_from_reference=(
            current.relative_grind_steps_from_reference + action.grind_delta_steps_from_current
        ),
        microns_per_step=current.microns_per_step,
        dose_g=action.next_dose_g,
        target_yield_g=action.target_yield_g,
        target_ratio=action.target_ratio,
        grinder_step_direction=current.grinder_step_direction,
    )


def validate_dreamer_action(
    action: DreamerActionCandidate,
    *,
    current: Recipe,
    bounds: SafetyBounds,
) -> None:
    if abs(action.grind_delta_steps_from_current) > bounds.max_grind_delta_steps_from_current:
        raise ValueError("Dreamer action exceeds grind delta safety bound")
    if abs(action.next_dose_g - current.dose_g) > bounds.max_dose_delta_g + 1e-9:
        raise ValueError("Dreamer action exceeds dose delta safety bound")
    if abs(action.target_yield_g - current.target_yield_g) > bounds.max_yield_delta_g + 1e-9:
        raise ValueError("Dreamer action exceeds yield delta safety bound")
    if not bounds.dose_min_g <= action.next_dose_g <= bounds.dose_max_g:
        raise ValueError("Dreamer action dose outside global bounds")
    if not bounds.target_yield_min_g <= action.target_yield_g <= bounds.target_yield_max_g:
        raise ValueError("Dreamer action yield outside global bounds")
    if not bounds.target_ratio_min <= (action.target_ratio or 0.0) <= bounds.target_ratio_max:
        raise ValueError("Dreamer action ratio outside global bounds")


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Dreamer {field_name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Dreamer {field_name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Dreamer {field_name} must be finite")
    return parsed


def _integer_steps(value: object, field_name: str) -> int:
    parsed = _finite_float(value, field_name)
    if not parsed.is_integer():
        raise ValueError("Dreamer grind delta must be an integer number of relative steps")
    return int(parsed)

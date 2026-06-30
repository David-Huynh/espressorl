from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from espresso_rl.domain.models import FIXED_CADENCE_SAMPLE_INTERVAL_MS
from espresso_rl.domain.optimization import (
    OPTIMIZER_FAMILY_DREAMER_V3,
    optimizer_family_allows_adaptive_profile_control,
)

DREAMER_CONTROL_SPEC_FORMAT = "espresso_rl_dreamer_control_spec_v1"
DREAMER_CONTROL_SPEC_SCHEMA_VERSION = 1
DREAMER_MIN_OBSERVATION_INTERVAL_MS = FIXED_CADENCE_SAMPLE_INTERVAL_MS
DREAMER_MIN_DECISION_INTERVAL_MS = 500
DREAMER_DEFAULT_DECISION_INTERVAL_MS = 1000
DREAMER_MAX_DECISION_INTERVAL_MS = 10_000
DREAMER_MAX_PRESSURE_TARGET_BAR = 12.0
DREAMER_MIN_TEMPERATURE_TARGET_C = 20.0
DREAMER_MAX_TEMPERATURE_TARGET_C = 100.0
DREAMER_MAX_YIELD_STOP_TARGET_G = 90.0
DREAMER_MAX_SHOT_DURATION_S = 90.0

DREAMER_DYNAMIC_ACTION_FIELDS = (
    "pressure_target_bar",
    "flow_target_ml_s",
    "pump_duty",
    "valve_position",
    "temperature_target_c",
    "yield_stop_target_g",
    "stop",
)
DREAMER_CONTROL_CONSTRAINT_FIELDS = (
    "dynamic_control_enabled",
    "pressure_control_allowed",
    "flow_control_allowed",
    "pump_control_allowed",
    "valve_control_allowed",
    "temperature_control_allowed",
    "stop_control_allowed",
)

_CONTROL_SPEC_FIELDS = frozenset(
    {
        "format",
        "schema_version",
        "optimizer_family",
        "observation_interval_ms",
        "decision_interval_ms",
        "dynamic_control_enabled",
        "pressure_control_allowed",
        "flow_control_allowed",
        "pump_control_allowed",
        "valve_control_allowed",
        "temperature_control_allowed",
        "stop_control_allowed",
        "safety_limits",
    }
)
_SAFETY_LIMIT_FIELDS = frozenset(
    {
        "min_pressure_bar",
        "max_pressure_bar",
        "min_flow_ml_s",
        "max_flow_ml_s",
        "min_temperature_c",
        "max_temperature_c",
        "min_yield_stop_target_g",
        "max_yield_stop_target_g",
        "max_shot_duration_s",
        "min_pump_duty",
        "max_pump_duty",
        "min_valve_position",
        "max_valve_position",
    }
)


@dataclass(frozen=True)
class DreamerControlSafetyLimits:
    min_pressure_bar: float = 0.0
    max_pressure_bar: float = DREAMER_MAX_PRESSURE_TARGET_BAR
    min_flow_ml_s: float = 0.0
    max_flow_ml_s: float = 20.0
    min_temperature_c: float = DREAMER_MIN_TEMPERATURE_TARGET_C
    max_temperature_c: float = DREAMER_MAX_TEMPERATURE_TARGET_C
    min_yield_stop_target_g: float = 5.0
    max_yield_stop_target_g: float = DREAMER_MAX_YIELD_STOP_TARGET_G
    max_shot_duration_s: float = DREAMER_MAX_SHOT_DURATION_S
    min_pump_duty: float = 0.0
    max_pump_duty: float = 1.0
    min_valve_position: float = 0.0
    max_valve_position: float = 1.0

    def __post_init__(self) -> None:
        _validate_range(self.min_pressure_bar, self.max_pressure_bar, "pressure")
        _validate_range(self.min_flow_ml_s, self.max_flow_ml_s, "flow")
        _validate_range(self.min_temperature_c, self.max_temperature_c, "temperature")
        _validate_range(self.min_yield_stop_target_g, self.max_yield_stop_target_g, "yield stop target")
        _validate_range(self.min_pump_duty, self.max_pump_duty, "pump duty")
        _validate_range(self.min_valve_position, self.max_valve_position, "valve position")
        if not _is_finite_number(self.max_shot_duration_s) or self.max_shot_duration_s <= 0:
            raise ValueError("Dreamer shot duration safety limit is invalid")
        if self.max_yield_stop_target_g > DREAMER_MAX_YIELD_STOP_TARGET_G:
            raise ValueError("Dreamer yield stop target safety limit must not exceed 90g")
        if self.max_shot_duration_s > DREAMER_MAX_SHOT_DURATION_S:
            raise ValueError("Dreamer shot duration safety limit must not exceed 90s")
        if self.min_temperature_c < DREAMER_MIN_TEMPERATURE_TARGET_C:
            raise ValueError("Dreamer temperature safety limit must not go below 20C")
        if self.max_temperature_c > DREAMER_MAX_TEMPERATURE_TARGET_C:
            raise ValueError("Dreamer temperature safety limit must not exceed 100C")
        if self.max_pressure_bar > DREAMER_MAX_PRESSURE_TARGET_BAR:
            raise ValueError("Dreamer pressure safety limit must not exceed 12 bar")
        if self.max_flow_ml_s > 20.0:
            raise ValueError("Dreamer flow safety limit must not exceed 20 ml/s")

    def to_dict(self) -> dict[str, float]:
        return {
            "min_pressure_bar": float(self.min_pressure_bar),
            "max_pressure_bar": float(self.max_pressure_bar),
            "min_flow_ml_s": float(self.min_flow_ml_s),
            "max_flow_ml_s": float(self.max_flow_ml_s),
            "min_temperature_c": float(self.min_temperature_c),
            "max_temperature_c": float(self.max_temperature_c),
            "min_yield_stop_target_g": float(self.min_yield_stop_target_g),
            "max_yield_stop_target_g": float(self.max_yield_stop_target_g),
            "max_shot_duration_s": float(self.max_shot_duration_s),
            "min_pump_duty": float(self.min_pump_duty),
            "max_pump_duty": float(self.max_pump_duty),
            "min_valve_position": float(self.min_valve_position),
            "max_valve_position": float(self.max_valve_position),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "DreamerControlSafetyLimits":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("Dreamer safety_limits must be an object")
        unknown = sorted(str(key) for key in value if key not in _SAFETY_LIMIT_FIELDS)
        if unknown:
            raise ValueError(f"Dreamer safety_limits contains unsupported fields: {', '.join(unknown[:5])}")
        defaults = cls()
        return cls(
            min_pressure_bar=_optional_float(value, "min_pressure_bar", defaults.min_pressure_bar),
            max_pressure_bar=_optional_float(value, "max_pressure_bar", defaults.max_pressure_bar),
            min_flow_ml_s=_optional_float(value, "min_flow_ml_s", defaults.min_flow_ml_s),
            max_flow_ml_s=_optional_float(value, "max_flow_ml_s", defaults.max_flow_ml_s),
            min_temperature_c=_optional_float(value, "min_temperature_c", defaults.min_temperature_c),
            max_temperature_c=_optional_float(value, "max_temperature_c", defaults.max_temperature_c),
            min_yield_stop_target_g=_optional_float(
                value,
                "min_yield_stop_target_g",
                defaults.min_yield_stop_target_g,
            ),
            max_yield_stop_target_g=_optional_float(
                value,
                "max_yield_stop_target_g",
                defaults.max_yield_stop_target_g,
            ),
            max_shot_duration_s=_optional_float(value, "max_shot_duration_s", defaults.max_shot_duration_s),
            min_pump_duty=_optional_float(value, "min_pump_duty", defaults.min_pump_duty),
            max_pump_duty=_optional_float(value, "max_pump_duty", defaults.max_pump_duty),
            min_valve_position=_optional_float(value, "min_valve_position", defaults.min_valve_position),
            max_valve_position=_optional_float(value, "max_valve_position", defaults.max_valve_position),
        )


@dataclass(frozen=True)
class DreamerDynamicActionSanitization:
    sanitized_action: dict[str, Any] | None
    clamped_fields: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class DreamerControlSpec:
    observation_interval_ms: int = FIXED_CADENCE_SAMPLE_INTERVAL_MS
    decision_interval_ms: int = DREAMER_DEFAULT_DECISION_INTERVAL_MS
    dynamic_control_enabled: bool = False
    pressure_control_allowed: bool = False
    flow_control_allowed: bool = False
    pump_control_allowed: bool = False
    valve_control_allowed: bool = False
    temperature_control_allowed: bool = False
    stop_control_allowed: bool = False
    optimizer_family: str = OPTIMIZER_FAMILY_DREAMER_V3
    safety_limits: DreamerControlSafetyLimits = field(default_factory=DreamerControlSafetyLimits)
    format: str = DREAMER_CONTROL_SPEC_FORMAT
    schema_version: int = DREAMER_CONTROL_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.format != DREAMER_CONTROL_SPEC_FORMAT:
            raise ValueError("Dreamer control spec format is unsupported")
        if self.schema_version != DREAMER_CONTROL_SPEC_SCHEMA_VERSION:
            raise ValueError("Dreamer control spec schema_version is unsupported")
        observation_interval_ms = _integer_ms(self.observation_interval_ms, "observation_interval_ms")
        decision_interval_ms = _integer_ms(self.decision_interval_ms, "decision_interval_ms")
        object.__setattr__(self, "observation_interval_ms", observation_interval_ms)
        object.__setattr__(self, "decision_interval_ms", decision_interval_ms)
        if observation_interval_ms < DREAMER_MIN_OBSERVATION_INTERVAL_MS:
            raise ValueError("Dreamer observation interval is faster than the supported sensor cadence")
        if decision_interval_ms < DREAMER_MIN_DECISION_INTERVAL_MS:
            raise ValueError("Dreamer decision interval is faster than the supported control cadence")
        if decision_interval_ms > DREAMER_MAX_DECISION_INTERVAL_MS:
            raise ValueError("Dreamer decision interval is too slow for adaptive control training")
        if decision_interval_ms < observation_interval_ms:
            raise ValueError("Dreamer decision interval must not be faster than observation interval")
        if decision_interval_ms % observation_interval_ms != 0:
            raise ValueError("Dreamer decision interval must be an integer multiple of observation interval")
        for field_name in DREAMER_CONTROL_CONSTRAINT_FIELDS:
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise ValueError(f"Dreamer {field_name} must be boolean")
        if self.dynamic_control_enabled:
            if not optimizer_family_allows_adaptive_profile_control(self.optimizer_family):
                raise ValueError("Adaptive profile control is only available for Dreamer optimizers")
            if not any(self.control_allowed_for_field(field) for field in DREAMER_DYNAMIC_ACTION_FIELDS):
                raise ValueError("Dreamer dynamic control requires at least one allowed control field")
        if not isinstance(self.safety_limits, DreamerControlSafetyLimits):
            object.__setattr__(self, "safety_limits", DreamerControlSafetyLimits.from_dict(self.safety_limits))

    @property
    def decision_step_count(self) -> int:
        return self.decision_interval_ms // self.observation_interval_ms

    def is_decision_step(self, step_index: int) -> bool:
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        return step_index % self.decision_step_count == 0

    def control_allowed_for_field(self, field_name: str) -> bool:
        if field_name == "pressure_target_bar":
            return self.pressure_control_allowed
        if field_name == "flow_target_ml_s":
            return self.flow_control_allowed
        if field_name == "pump_duty":
            return self.pump_control_allowed
        if field_name == "valve_position":
            return self.valve_control_allowed
        if field_name == "temperature_target_c":
            return self.temperature_control_allowed
        if field_name in {"yield_stop_target_g", "stop"}:
            return self.stop_control_allowed
        return False

    def constraints(self) -> dict[str, bool]:
        return {field: bool(getattr(self, field)) for field in DREAMER_CONTROL_CONSTRAINT_FIELDS}

    def action_capability_mask(self) -> tuple[float, ...]:
        if not self.dynamic_control_enabled:
            return tuple(0.0 for _ in DREAMER_DYNAMIC_ACTION_FIELDS)
        return tuple(1.0 if self.control_allowed_for_field(field) else 0.0 for field in DREAMER_DYNAMIC_ACTION_FIELDS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "optimizer_family": self.optimizer_family,
            "observation_interval_ms": self.observation_interval_ms,
            "decision_interval_ms": self.decision_interval_ms,
            "dynamic_control_enabled": self.dynamic_control_enabled,
            "pressure_control_allowed": self.pressure_control_allowed,
            "flow_control_allowed": self.flow_control_allowed,
            "pump_control_allowed": self.pump_control_allowed,
            "valve_control_allowed": self.valve_control_allowed,
            "temperature_control_allowed": self.temperature_control_allowed,
            "stop_control_allowed": self.stop_control_allowed,
            "safety_limits": self.safety_limits.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DreamerControlSpec":
        if not isinstance(value, dict):
            raise ValueError("Dreamer control spec must be an object")
        unknown = sorted(str(key) for key in value if key not in _CONTROL_SPEC_FIELDS)
        if unknown:
            raise ValueError(f"Dreamer control spec contains unsupported fields: {', '.join(unknown[:5])}")
        return cls(
            format=value.get("format", DREAMER_CONTROL_SPEC_FORMAT),
            schema_version=value.get("schema_version", DREAMER_CONTROL_SPEC_SCHEMA_VERSION),
            optimizer_family=str(value.get("optimizer_family") or OPTIMIZER_FAMILY_DREAMER_V3),
            observation_interval_ms=value.get("observation_interval_ms", FIXED_CADENCE_SAMPLE_INTERVAL_MS),
            decision_interval_ms=value.get("decision_interval_ms", DREAMER_DEFAULT_DECISION_INTERVAL_MS),
            dynamic_control_enabled=value.get("dynamic_control_enabled", False),
            pressure_control_allowed=value.get("pressure_control_allowed", False),
            flow_control_allowed=value.get("flow_control_allowed", False),
            pump_control_allowed=value.get("pump_control_allowed", False),
            valve_control_allowed=value.get("valve_control_allowed", False),
            temperature_control_allowed=value.get("temperature_control_allowed", False),
            stop_control_allowed=value.get("stop_control_allowed", False),
            safety_limits=DreamerControlSafetyLimits.from_dict(value.get("safety_limits")),
        )


def validate_dynamic_action_for_control_spec(
    action: dict[str, Any] | None,
    *,
    control_spec: DreamerControlSpec,
    step_index: int,
) -> list[str]:
    if action is None:
        return []
    errors: list[str] = []
    if not isinstance(action, dict):
        return ["Dreamer dynamic action must be an object or null"]
    unknown = sorted(str(key) for key in action if key not in DREAMER_DYNAMIC_ACTION_FIELDS)
    if unknown:
        errors.append(f"Dreamer dynamic action contains unsupported fields: {', '.join(unknown[:5])}")
    if not control_spec.is_decision_step(step_index):
        errors.append("Dreamer dynamic action may only be emitted on decision steps")
    if not control_spec.dynamic_control_enabled:
        errors.append("Dreamer dynamic action provided while dynamic control is disabled")
    if "pressure_target_bar" in action and "flow_target_ml_s" in action:
        errors.append("Dreamer dynamic action cannot request pressure and flow targets at the same decision step")

    for field_name in DREAMER_DYNAMIC_ACTION_FIELDS:
        if field_name not in action:
            continue
        if not control_spec.control_allowed_for_field(field_name):
            errors.append(f"Dreamer dynamic action field {field_name} is not allowed by the control spec")
            continue
        errors.extend(_validate_action_field(field_name, action[field_name], control_spec.safety_limits))
    return errors


def sanitize_dynamic_action_for_control_spec(
    action: dict[str, Any] | None,
    *,
    control_spec: DreamerControlSpec,
    step_index: int,
) -> DreamerDynamicActionSanitization:
    if action is None:
        return DreamerDynamicActionSanitization(None)
    errors: list[str] = []
    if not isinstance(action, dict):
        return DreamerDynamicActionSanitization(None, errors=("Dreamer dynamic action must be an object or null",))
    unknown = sorted(str(key) for key in action if key not in DREAMER_DYNAMIC_ACTION_FIELDS)
    if unknown:
        errors.append(f"Dreamer dynamic action contains unsupported fields: {', '.join(unknown[:5])}")
    if not control_spec.is_decision_step(step_index):
        errors.append("Dreamer dynamic action may only be emitted on decision steps")
    if not control_spec.dynamic_control_enabled:
        errors.append("Dreamer dynamic action provided while dynamic control is disabled")
    if "pressure_target_bar" in action and "flow_target_ml_s" in action:
        errors.append("Dreamer dynamic action cannot request pressure and flow targets at the same decision step")

    sanitized: dict[str, Any] = {}
    clamped_fields: list[str] = []
    for field_name in DREAMER_DYNAMIC_ACTION_FIELDS:
        if field_name not in action:
            continue
        if not control_spec.control_allowed_for_field(field_name):
            errors.append(f"Dreamer dynamic action field {field_name} is not allowed by the control spec")
            continue
        if field_name == "stop":
            if not isinstance(action[field_name], bool):
                errors.append("Dreamer dynamic action stop must be boolean")
                continue
            sanitized[field_name] = bool(action[field_name])
            continue
        value = action[field_name]
        action_range = _action_field_range(field_name, control_spec.safety_limits)
        if action_range is None or not _is_finite_number(value):
            errors.append(f"Dreamer dynamic action field {field_name} is invalid")
            continue
        minimum, maximum = action_range
        clamped = min(max(float(value), minimum), maximum)
        sanitized[field_name] = clamped
        if clamped != float(value):
            clamped_fields.append(field_name)
    if errors:
        return DreamerDynamicActionSanitization(None, errors=tuple(errors))
    return DreamerDynamicActionSanitization(sanitized, clamped_fields=tuple(clamped_fields))


def expand_decision_actions_to_observation_steps(
    decision_actions: Sequence[dict[str, Any] | None],
    *,
    step_count: int,
    control_spec: DreamerControlSpec,
) -> list[dict[str, Any] | None]:
    if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count <= 0:
        raise ValueError("step_count must be a positive integer")
    decision_step_count = control_spec.decision_step_count
    expected_decision_count = math.ceil(step_count / decision_step_count)
    if len(decision_actions) != expected_decision_count:
        raise ValueError("decision action count does not match step_count and decision cadence")

    expanded: list[dict[str, Any] | None] = []
    for decision_index, action in enumerate(decision_actions):
        step_index = decision_index * decision_step_count
        errors = validate_dynamic_action_for_control_spec(
            action,
            control_spec=control_spec,
            step_index=step_index,
        )
        if errors:
            raise ValueError("; ".join(errors))
        held_action = dict(action) if action is not None else None
        remaining = step_count - len(expanded)
        expanded.extend(
            dict(held_action) if held_action is not None else None
            for _ in range(min(decision_step_count, remaining))
        )
    return expanded


def _validate_action_field(
    field_name: str,
    value: object,
    limits: DreamerControlSafetyLimits,
) -> list[str]:
    if field_name == "stop":
        return [] if isinstance(value, bool) else ["Dreamer dynamic action stop must be boolean"]
    action_range = _action_field_range(field_name, limits)
    if action_range is None:
        return [f"Dreamer dynamic action field {field_name} is invalid"]
    minimum, maximum = action_range
    if not _is_finite_number(value) or not minimum <= float(value) <= maximum:
        return [f"Dreamer dynamic action field {field_name} is outside safety limits"]
    return []


def _action_field_range(field_name: str, limits: DreamerControlSafetyLimits) -> tuple[float, float] | None:
    ranges = {
        "pressure_target_bar": (limits.min_pressure_bar, limits.max_pressure_bar),
        "flow_target_ml_s": (limits.min_flow_ml_s, limits.max_flow_ml_s),
        "pump_duty": (limits.min_pump_duty, limits.max_pump_duty),
        "valve_position": (limits.min_valve_position, limits.max_valve_position),
        "temperature_target_c": (limits.min_temperature_c, limits.max_temperature_c),
        "yield_stop_target_g": (limits.min_yield_stop_target_g, limits.max_yield_stop_target_g),
    }
    return ranges.get(field_name)


def _validate_range(minimum: float, maximum: float, label: str) -> None:
    if not _is_finite_number(minimum) or not _is_finite_number(maximum) or float(minimum) >= float(maximum):
        raise ValueError(f"Dreamer {label} safety limits are invalid")


def _integer_ms(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Dreamer {field_name} must be an integer millisecond value")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Dreamer {field_name} must be an integer millisecond value") from exc
    if parsed != value and not (isinstance(value, str) and str(parsed) == value.strip()):
        raise ValueError(f"Dreamer {field_name} must be an integer millisecond value")
    return parsed


def _optional_float(value: dict[str, Any], key: str, default: float) -> float:
    if key not in value:
        return default
    parsed = value[key]
    if not _is_finite_number(parsed):
        raise ValueError(f"Dreamer safety_limits.{key} must be finite")
    return float(parsed)


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


DEFAULT_DREAMER_CONTROL_SPEC = DreamerControlSpec()

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from espresso_rl.domain.dreamer_control import (
    DREAMER_MAX_PRESSURE_TARGET_BAR,
    DREAMER_MAX_SHOT_DURATION_S,
    DREAMER_MAX_TEMPERATURE_TARGET_C,
    DREAMER_MAX_YIELD_STOP_TARGET_G,
    DREAMER_MIN_TEMPERATURE_TARGET_C,
)

DREAMER_PRE_SHOT_ACTION_FORMAT = "espresso_rl_dreamer_pre_shot_action_v1"
DREAMER_PRE_SHOT_ACTION_SCHEMA_VERSION = 1
DREAMER_PRE_SHOT_ACTION_SPEC_FORMAT = "espresso_rl_dreamer_pre_shot_action_spec_v1"
DREAMER_PRE_SHOT_ACTION_SPEC_SCHEMA_VERSION = 1

DREAMER_PRE_SHOT_ACTION_FIELDS = (
    "grind_delta_steps_from_current",
    "dose_target_g",
    "yield_target_g",
    "temperature_target_c",
    "pump_target_mode",
    "pressure_target_bar",
    "flow_target_ml_s",
    "valve_open",
    "initial_stage_duration_s",
)

DREAMER_PUMP_TARGET_MODE_OFF = 0
DREAMER_PUMP_TARGET_MODE_PRESSURE = 1
DREAMER_PUMP_TARGET_MODE_FLOW = 2

_ACTION_FIELDS = frozenset({"format", "schema_version", "values", "observed", "capabilities"})
_SPEC_FIELDS = frozenset(
    {
        "format",
        "schema_version",
        "action_format",
        "action_schema_version",
        "action_fields",
        "capability_fields",
        "units",
        "bins",
    }
)
_UNITS = {
    "grind_delta_steps_from_current": "relative_steps",
    "dose_target_g": "g",
    "yield_target_g": "g",
    "temperature_target_c": "celsius",
    "pump_target_mode": "enum",
    "pressure_target_bar": "bar",
    "flow_target_ml_s": "ml_per_s",
    "valve_open": "boolean",
    "initial_stage_duration_s": "seconds",
}
_HARD_RANGES = {
    "grind_delta_steps_from_current": (-32.0, 32.0),
    "dose_target_g": (5.0, 30.0),
    "yield_target_g": (5.0, DREAMER_MAX_YIELD_STOP_TARGET_G),
    "temperature_target_c": (DREAMER_MIN_TEMPERATURE_TARGET_C, DREAMER_MAX_TEMPERATURE_TARGET_C),
    "pump_target_mode": (0.0, 2.0),
    "pressure_target_bar": (0.0, DREAMER_MAX_PRESSURE_TARGET_BAR),
    "flow_target_ml_s": (0.0, 20.0),
    "valve_open": (0.0, 1.0),
    "initial_stage_duration_s": (0.25, DREAMER_MAX_SHOT_DURATION_S),
}


def _uniform_bins(start: float, stop: float, step: float) -> tuple[float, ...]:
    count = int(round((stop - start) / step))
    return tuple(round(start + index * step, 8) for index in range(count + 1))


DEFAULT_DREAMER_PRE_SHOT_ACTION_BINS: dict[str, tuple[float, ...]] = {
    "grind_delta_steps_from_current": (
        -32.0,
        -24.0,
        -16.0,
        -12.0,
        -8.0,
        -6.0,
        -4.0,
        -3.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
        24.0,
        32.0,
    ),
    "dose_target_g": _uniform_bins(5.0, 30.0, 0.25),
    "yield_target_g": _uniform_bins(5.0, DREAMER_MAX_YIELD_STOP_TARGET_G, 1.0),
    "temperature_target_c": _uniform_bins(
        DREAMER_MIN_TEMPERATURE_TARGET_C,
        DREAMER_MAX_TEMPERATURE_TARGET_C,
        0.5,
    ),
    "pump_target_mode": (0.0, 1.0, 2.0),
    "pressure_target_bar": _uniform_bins(0.0, DREAMER_MAX_PRESSURE_TARGET_BAR, 0.25),
    "flow_target_ml_s": _uniform_bins(0.0, 20.0, 0.25),
    "valve_open": (0.0, 1.0),
    "initial_stage_duration_s": (
        0.25,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
        15.0,
        20.0,
        30.0,
        45.0,
        60.0,
        90.0,
    ),
}


@dataclass(frozen=True)
class DreamerPreShotActionSpec:
    bins: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: dict(DEFAULT_DREAMER_PRE_SHOT_ACTION_BINS)
    )
    format: str = DREAMER_PRE_SHOT_ACTION_SPEC_FORMAT
    schema_version: int = DREAMER_PRE_SHOT_ACTION_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.format != DREAMER_PRE_SHOT_ACTION_SPEC_FORMAT:
            raise ValueError("Dreamer pre-shot action spec format is unsupported")
        if self.schema_version != DREAMER_PRE_SHOT_ACTION_SPEC_SCHEMA_VERSION:
            raise ValueError("Dreamer pre-shot action spec schema_version is unsupported")
        if not isinstance(self.bins, dict) or set(self.bins) != set(DREAMER_PRE_SHOT_ACTION_FIELDS):
            raise ValueError("Dreamer pre-shot action spec bins must match canonical action fields")
        normalized: dict[str, tuple[float, ...]] = {}
        for field_name in DREAMER_PRE_SHOT_ACTION_FIELDS:
            normalized[field_name] = _validated_bins(field_name, self.bins[field_name])
        if normalized["pump_target_mode"] != (0.0, 1.0, 2.0):
            raise ValueError("Dreamer pre-shot pump target mode bins are fixed")
        if normalized["valve_open"] != (0.0, 1.0):
            raise ValueError("Dreamer pre-shot valve bins are fixed")
        if 0.0 not in normalized["grind_delta_steps_from_current"]:
            raise ValueError("Dreamer pre-shot grind delta bins must include zero")
        object.__setattr__(self, "bins", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "action_format": DREAMER_PRE_SHOT_ACTION_FORMAT,
            "action_schema_version": DREAMER_PRE_SHOT_ACTION_SCHEMA_VERSION,
            "action_fields": list(DREAMER_PRE_SHOT_ACTION_FIELDS),
            "capability_fields": list(DREAMER_PRE_SHOT_ACTION_FIELDS),
            "units": dict(_UNITS),
            "bins": {name: list(self.bins[name]) for name in DREAMER_PRE_SHOT_ACTION_FIELDS},
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "DreamerPreShotActionSpec":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("Dreamer pre-shot action spec must be an object")
        unknown = sorted(str(key) for key in value if key not in _SPEC_FIELDS)
        missing = sorted(key for key in _SPEC_FIELDS if key not in value)
        if unknown or missing:
            details = []
            if unknown:
                details.append(f"unsupported fields: {', '.join(unknown[:5])}")
            if missing:
                details.append(f"missing fields: {', '.join(missing[:5])}")
            raise ValueError(f"Dreamer pre-shot action spec is invalid ({'; '.join(details)})")
        if value.get("action_format") != DREAMER_PRE_SHOT_ACTION_FORMAT:
            raise ValueError("Dreamer pre-shot action format is unsupported")
        if value.get("action_schema_version") != DREAMER_PRE_SHOT_ACTION_SCHEMA_VERSION:
            raise ValueError("Dreamer pre-shot action schema_version is unsupported")
        expected_fields = list(DREAMER_PRE_SHOT_ACTION_FIELDS)
        if value.get("action_fields") != expected_fields or value.get("capability_fields") != expected_fields:
            raise ValueError("Dreamer pre-shot action field ordering is incompatible")
        if value.get("units") != _UNITS:
            raise ValueError("Dreamer pre-shot action units are incompatible")
        raw_bins = value.get("bins")
        if not isinstance(raw_bins, dict):
            raise ValueError("Dreamer pre-shot action bins must be an object")
        if set(raw_bins) != set(DREAMER_PRE_SHOT_ACTION_FIELDS) or any(
            not isinstance(raw_bins.get(name), (list, tuple))
            for name in DREAMER_PRE_SHOT_ACTION_FIELDS
        ):
            raise ValueError("Dreamer pre-shot action bins must match canonical action fields")
        return cls(
            format=value.get("format"),
            schema_version=value.get("schema_version"),
            bins={name: tuple(raw_bins.get(name, ())) for name in DREAMER_PRE_SHOT_ACTION_FIELDS},
        )

    def bin_value(self, field_name: str, value: object) -> tuple[int, float]:
        if field_name not in self.bins:
            raise ValueError(f"Dreamer pre-shot action field is unsupported: {field_name}")
        parsed = _finite_float(value, field_name)
        bins = self.bins[field_name]
        if parsed < bins[0] or parsed > bins[-1]:
            raise ValueError(f"Dreamer pre-shot {field_name} is outside configured bins")
        index = min(range(len(bins)), key=lambda item: (abs(bins[item] - parsed), item))
        return index, bins[index]


def build_dreamer_pre_shot_action(
    *,
    values: dict[str, Any],
    observed_fields: Iterable[str],
    capability_fields: Iterable[str],
) -> dict[str, Any]:
    observed_names = frozenset(observed_fields)
    capability_names = frozenset(capability_fields)
    unknown_observed = sorted(observed_names - set(DREAMER_PRE_SHOT_ACTION_FIELDS))
    unknown_capabilities = sorted(capability_names - set(DREAMER_PRE_SHOT_ACTION_FIELDS))
    if unknown_observed or unknown_capabilities:
        raise ValueError("Dreamer pre-shot masks contain unsupported fields")
    action = {
        "format": DREAMER_PRE_SHOT_ACTION_FORMAT,
        "schema_version": DREAMER_PRE_SHOT_ACTION_SCHEMA_VERSION,
        "values": dict(values),
        "observed": {name: name in observed_names for name in DREAMER_PRE_SHOT_ACTION_FIELDS},
        "capabilities": {name: name in capability_names for name in DREAMER_PRE_SHOT_ACTION_FIELDS},
    }
    errors = validate_dreamer_pre_shot_action(action)
    if errors:
        raise ValueError("; ".join(errors))
    return action


def validate_dreamer_pre_shot_action(
    action: object,
    *,
    spec: DreamerPreShotActionSpec | None = None,
) -> list[str]:
    spec = spec or DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC
    errors: list[str] = []
    if not isinstance(action, dict):
        return ["Dreamer pre-shot action must be an object"]
    unknown = sorted(str(key) for key in action if key not in _ACTION_FIELDS)
    missing = sorted(key for key in _ACTION_FIELDS if key not in action)
    if unknown:
        errors.append(f"Dreamer pre-shot action contains unsupported fields: {', '.join(unknown[:5])}")
    if missing:
        errors.append(f"Dreamer pre-shot action is missing fields: {', '.join(missing[:5])}")
    if action.get("format") != DREAMER_PRE_SHOT_ACTION_FORMAT:
        errors.append("Dreamer pre-shot action format is unsupported")
    if action.get("schema_version") != DREAMER_PRE_SHOT_ACTION_SCHEMA_VERSION:
        errors.append("Dreamer pre-shot action schema_version is unsupported")
    values = action.get("values")
    observed = action.get("observed")
    capabilities = action.get("capabilities")
    if not isinstance(values, dict):
        errors.append("Dreamer pre-shot action values must be an object")
        values = {}
    if not isinstance(observed, dict):
        errors.append("Dreamer pre-shot action observed mask must be an object")
        observed = {}
    if not isinstance(capabilities, dict):
        errors.append("Dreamer pre-shot action capability mask must be an object")
        capabilities = {}
    _validate_field_map(values, allow_subset=True, label="values", errors=errors)
    _validate_field_map(observed, allow_subset=False, label="observed", errors=errors)
    _validate_field_map(capabilities, allow_subset=False, label="capabilities", errors=errors)

    for field_name in DREAMER_PRE_SHOT_ACTION_FIELDS:
        is_observed = observed.get(field_name)
        is_capable = capabilities.get(field_name)
        if not isinstance(is_observed, bool):
            errors.append(f"Dreamer pre-shot observed.{field_name} must be boolean")
            continue
        if not isinstance(is_capable, bool):
            errors.append(f"Dreamer pre-shot capabilities.{field_name} must be boolean")
            continue
        has_value = field_name in values
        if is_observed != has_value:
            errors.append(f"Dreamer pre-shot {field_name} value and observed mask disagree")
            continue
        if is_observed and not is_capable:
            errors.append(f"Dreamer pre-shot {field_name} is observed without capability")
            continue
        if not has_value:
            continue
        try:
            parsed = _finite_float(values[field_name], field_name)
            _validate_hard_range(field_name, parsed)
            spec.bin_value(field_name, parsed)
        except ValueError as exc:
            errors.append(str(exc))

    mode = values.get("pump_target_mode") if observed.get("pump_target_mode") is True else None
    normalized_mode = float(mode) if isinstance(mode, (int, float)) and not isinstance(mode, bool) else None
    if mode is not None and normalized_mode not in {
        float(DREAMER_PUMP_TARGET_MODE_OFF),
        float(DREAMER_PUMP_TARGET_MODE_PRESSURE),
        float(DREAMER_PUMP_TARGET_MODE_FLOW),
    }:
        errors.append("Dreamer pre-shot pump_target_mode is invalid")
    if normalized_mode == DREAMER_PUMP_TARGET_MODE_PRESSURE and observed.get("flow_target_ml_s") is True:
        errors.append("Dreamer pre-shot pressure mode must not observe a flow target action")
    if normalized_mode == DREAMER_PUMP_TARGET_MODE_FLOW and observed.get("pressure_target_bar") is True:
        errors.append("Dreamer pre-shot flow mode must not observe a pressure target action")
    if normalized_mode == DREAMER_PUMP_TARGET_MODE_OFF and (
        observed.get("pressure_target_bar") is True or observed.get("flow_target_ml_s") is True
    ):
        errors.append("Dreamer pre-shot off mode must not observe pressure or flow target actions")
    return errors


def encode_dreamer_pre_shot_action(
    action: object,
    *,
    spec: DreamerPreShotActionSpec | None = None,
) -> tuple[tuple[float, ...], tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    spec = spec or DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC
    errors = validate_dreamer_pre_shot_action(action, spec=spec)
    if errors:
        raise ValueError("; ".join(errors))
    assert isinstance(action, dict)
    values = action["values"]
    observed = action["observed"]
    capabilities = action["capabilities"]
    encoded_values: list[float] = []
    encoded_indexes: list[int] = []
    observed_mask: list[float] = []
    capability_mask: list[float] = []
    for field_name in DREAMER_PRE_SHOT_ACTION_FIELDS:
        if observed[field_name]:
            index, bin_value = spec.bin_value(field_name, values[field_name])
            encoded_values.append(bin_value)
            encoded_indexes.append(index)
            observed_mask.append(1.0)
        else:
            encoded_values.append(0.0)
            encoded_indexes.append(0)
            observed_mask.append(0.0)
        capability_mask.append(1.0 if capabilities[field_name] else 0.0)
    return tuple(encoded_values), tuple(encoded_indexes), tuple(observed_mask), tuple(capability_mask)


def _validated_bins(field_name: str, values: object) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or not 2 <= len(values) <= 256:
        raise ValueError(f"Dreamer pre-shot {field_name} bins must contain 2..256 values")
    parsed = tuple(_finite_float(value, f"{field_name} bin") for value in values)
    if any(current <= previous for previous, current in zip(parsed, parsed[1:])):
        raise ValueError(f"Dreamer pre-shot {field_name} bins must be strictly increasing")
    minimum, maximum = _HARD_RANGES[field_name]
    if parsed[0] < minimum or parsed[-1] > maximum:
        raise ValueError(f"Dreamer pre-shot {field_name} bins exceed hard bounds")
    return parsed


def _validate_field_map(value: dict[str, Any], *, allow_subset: bool, label: str, errors: list[str]) -> None:
    unknown = sorted(str(key) for key in value if key not in DREAMER_PRE_SHOT_ACTION_FIELDS)
    if unknown:
        errors.append(f"Dreamer pre-shot {label} contains unsupported fields: {', '.join(unknown[:5])}")
    if not allow_subset:
        missing = sorted(key for key in DREAMER_PRE_SHOT_ACTION_FIELDS if key not in value)
        if missing:
            errors.append(f"Dreamer pre-shot {label} is missing fields: {', '.join(missing[:5])}")


def _validate_hard_range(field_name: str, value: float) -> None:
    minimum, maximum = _HARD_RANGES[field_name]
    if not minimum <= value <= maximum:
        raise ValueError(f"Dreamer pre-shot {field_name} is outside hard bounds")
    if field_name in {"pump_target_mode", "valve_open"} and not value.is_integer():
        raise ValueError(f"Dreamer pre-shot {field_name} must be discrete")


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        parsed = float(value) if field_name == "valve_open" else math.nan
    elif not isinstance(value, (int, float)):
        parsed = math.nan
    else:
        parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Dreamer pre-shot {field_name} must be finite")
    return parsed


DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC = DreamerPreShotActionSpec()


def dreamer_pre_shot_action_spec_sha256(spec: DreamerPreShotActionSpec) -> str:
    payload = json.dumps(
        spec.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC_SHA256 = dreamer_pre_shot_action_spec_sha256(
    DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC
)

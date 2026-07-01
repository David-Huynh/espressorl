from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

from espresso_rl.domain.dreamer_control import (
    DREAMER_DYNAMIC_ACTION_FIELDS,
    DREAMER_MAX_PRESSURE_TARGET_BAR,
    DREAMER_MAX_TEMPERATURE_TARGET_C,
    DREAMER_MAX_YIELD_STOP_TARGET_G,
    DREAMER_MIN_TEMPERATURE_TARGET_C,
)

DREAMER_LIVE_ACTION_SPEC_FORMAT = "espresso_rl_dreamer_live_action_spec_v2"
DREAMER_LIVE_ACTION_SPEC_SCHEMA_VERSION = 2

DREAMER_LIVE_ACTION_FIELDS = (
    "pump_target_mode",
    "pressure_delta_bar",
    "flow_delta_ml_s",
    "valve_position_delta",
    "temperature_delta_c",
    "yield_stop_delta_g",
    "stop",
)

_SPEC_FIELDS = frozenset(
    {
        "format",
        "schema_version",
        "action_fields",
        "capability_fields",
        "control_fields",
        "units",
        "bins",
    }
)
_UNITS = {
    "pump_target_mode": "enum_pressure_or_flow",
    "pressure_delta_bar": "bar_delta",
    "flow_delta_ml_s": "ml_per_s_delta",
    "valve_position_delta": "unit_interval_delta",
    "temperature_delta_c": "celsius_delta",
    "yield_stop_delta_g": "g_delta",
    "stop": "boolean",
}
_HARD_RANGES = {
    "pump_target_mode": (1.0, 2.0),
    "pressure_delta_bar": (-DREAMER_MAX_PRESSURE_TARGET_BAR, DREAMER_MAX_PRESSURE_TARGET_BAR),
    "flow_delta_ml_s": (-20.0, 20.0),
    "valve_position_delta": (-1.0, 1.0),
    "temperature_delta_c": (
        DREAMER_MIN_TEMPERATURE_TARGET_C - DREAMER_MAX_TEMPERATURE_TARGET_C,
        DREAMER_MAX_TEMPERATURE_TARGET_C - DREAMER_MIN_TEMPERATURE_TARGET_C,
    ),
    "yield_stop_delta_g": (-DREAMER_MAX_YIELD_STOP_TARGET_G, DREAMER_MAX_YIELD_STOP_TARGET_G),
    "stop": (0.0, 1.0),
}


def _uniform_bins(start: float, stop: float, step: float) -> tuple[float, ...]:
    count = int(round((stop - start) / step))
    return tuple(round(start + index * step, 8) for index in range(count + 1))


DEFAULT_DREAMER_LIVE_ACTION_BINS: dict[str, tuple[float, ...]] = {
    "pump_target_mode": (1.0, 2.0),
    "pressure_delta_bar": (-4.0, -3.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
    "flow_delta_ml_s": (-4.0, -3.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
    "valve_position_delta": (-1.0, 0.0, 1.0),
    "temperature_delta_c": (
        -8.0,
        -4.0,
        -2.0,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
    ),
    "yield_stop_delta_g": (-20.0, -10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
    "stop": (0.0, 1.0),
}


@dataclass(frozen=True)
class DreamerLiveActionSpec:
    bins: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: dict(DEFAULT_DREAMER_LIVE_ACTION_BINS)
    )
    format: str = DREAMER_LIVE_ACTION_SPEC_FORMAT
    schema_version: int = DREAMER_LIVE_ACTION_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.format != DREAMER_LIVE_ACTION_SPEC_FORMAT:
            raise ValueError("Dreamer live action spec format is unsupported")
        if self.schema_version != DREAMER_LIVE_ACTION_SPEC_SCHEMA_VERSION:
            raise ValueError("Dreamer live action spec schema_version is unsupported")
        if not isinstance(self.bins, dict) or set(self.bins) != set(DREAMER_LIVE_ACTION_FIELDS):
            raise ValueError("Dreamer live action spec bins must match canonical action fields")
        normalized: dict[str, tuple[float, ...]] = {}
        for field_name in DREAMER_LIVE_ACTION_FIELDS:
            normalized[field_name] = _validated_bins(field_name, self.bins[field_name])
        if 0.0 not in normalized["pressure_delta_bar"]:
            raise ValueError("Dreamer live pressure delta bins must include zero")
        if 0.0 not in normalized["flow_delta_ml_s"]:
            raise ValueError("Dreamer live flow delta bins must include zero")
        if normalized["pump_target_mode"] != (1.0, 2.0):
            raise ValueError("Dreamer live pump target mode bins are fixed to pressure and flow")
        if normalized["valve_position_delta"] != (-1.0, 0.0, 1.0):
            raise ValueError("Dreamer live valve delta bins are fixed")
        if 0.0 not in normalized["temperature_delta_c"]:
            raise ValueError("Dreamer live temperature delta bins must include zero")
        if 0.0 not in normalized["yield_stop_delta_g"]:
            raise ValueError("Dreamer live yield stop delta bins must include zero")
        if normalized["stop"] != (0.0, 1.0):
            raise ValueError("Dreamer live stop bins are fixed")
        object.__setattr__(self, "bins", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "action_fields": list(DREAMER_LIVE_ACTION_FIELDS),
            "capability_fields": list(DREAMER_LIVE_ACTION_FIELDS),
            "control_fields": list(DREAMER_DYNAMIC_ACTION_FIELDS),
            "units": dict(_UNITS),
            "bins": {name: list(self.bins[name]) for name in DREAMER_LIVE_ACTION_FIELDS},
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "DreamerLiveActionSpec":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("Dreamer live action spec must be an object")
        unknown = sorted(str(key) for key in value if key not in _SPEC_FIELDS)
        missing = sorted(key for key in _SPEC_FIELDS if key not in value)
        if unknown or missing:
            details = []
            if unknown:
                details.append(f"unsupported fields: {', '.join(unknown[:5])}")
            if missing:
                details.append(f"missing fields: {', '.join(missing[:5])}")
            raise ValueError(f"Dreamer live action spec is invalid ({'; '.join(details)})")
        expected_fields = list(DREAMER_LIVE_ACTION_FIELDS)
        if value.get("action_fields") != expected_fields or value.get("capability_fields") != expected_fields:
            raise ValueError("Dreamer live action field ordering is incompatible")
        if value.get("control_fields") != list(DREAMER_DYNAMIC_ACTION_FIELDS):
            raise ValueError("Dreamer live action control-field mapping is incompatible")
        if value.get("units") != _UNITS:
            raise ValueError("Dreamer live action units are incompatible")
        raw_bins = value.get("bins")
        if not isinstance(raw_bins, dict):
            raise ValueError("Dreamer live action bins must be an object")
        if set(raw_bins) != set(DREAMER_LIVE_ACTION_FIELDS) or any(
            not isinstance(raw_bins.get(name), (list, tuple))
            for name in DREAMER_LIVE_ACTION_FIELDS
        ):
            raise ValueError("Dreamer live action bins must match canonical action fields")
        return cls(
            format=value.get("format"),
            schema_version=value.get("schema_version"),
            bins={name: tuple(raw_bins.get(name, ())) for name in DREAMER_LIVE_ACTION_FIELDS},
        )

    def bin_value(self, field_name: str, value: object) -> tuple[int, float]:
        if field_name not in self.bins:
            raise ValueError(f"Dreamer live action field is unsupported: {field_name}")
        parsed = _finite_float(value, field_name)
        if field_name == "stop":
            parsed = 1.0 if parsed >= 0.5 else 0.0
        bins = self.bins[field_name]
        if parsed < bins[0] or parsed > bins[-1]:
            raise ValueError(f"Dreamer live {field_name} is outside configured bins")
        index = min(range(len(bins)), key=lambda item: (abs(bins[item] - parsed), item))
        return index, bins[index]


def _validated_bins(field_name: str, values: object) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or not 2 <= len(values) <= 256:
        raise ValueError(f"Dreamer live {field_name} bins must contain 2..256 values")
    parsed = tuple(_finite_float(value, f"{field_name} bin") for value in values)
    if any(current <= previous for previous, current in zip(parsed, parsed[1:])):
        raise ValueError(f"Dreamer live {field_name} bins must be strictly increasing")
    minimum, maximum = _HARD_RANGES[field_name]
    if parsed[0] < minimum or parsed[-1] > maximum:
        raise ValueError(f"Dreamer live {field_name} bins exceed hard bounds")
    return parsed


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        parsed = float(value) if field_name in {"valve_position", "stop"} else math.nan
    elif not isinstance(value, (int, float)):
        parsed = math.nan
    else:
        parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Dreamer live {field_name} must be finite")
    return parsed


DEFAULT_DREAMER_LIVE_ACTION_SPEC = DreamerLiveActionSpec()


def dreamer_live_action_spec_sha256(spec: DreamerLiveActionSpec) -> str:
    payload = json.dumps(
        spec.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


DEFAULT_DREAMER_LIVE_ACTION_SPEC_SHA256 = dreamer_live_action_spec_sha256(
    DEFAULT_DREAMER_LIVE_ACTION_SPEC
)

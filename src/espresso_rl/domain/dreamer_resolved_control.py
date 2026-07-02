from __future__ import annotations

from dataclasses import dataclass
import math

from espresso_rl.domain.dreamer_control import (
    DREAMER_MAX_PRESSURE_TARGET_BAR,
    DREAMER_MAX_TEMPERATURE_TARGET_C,
    DREAMER_MAX_YIELD_STOP_TARGET_G,
    DREAMER_MIN_TEMPERATURE_TARGET_C,
)


DREAMER_PUMP_TARGET_MODE_SIMPLE = 0
DREAMER_PUMP_TARGET_MODE_PRESSURE = 1
DREAMER_PUMP_TARGET_MODE_FLOW = 2
DREAMER_PUMP_TARGET_MODES = frozenset(
    {
        DREAMER_PUMP_TARGET_MODE_SIMPLE,
        DREAMER_PUMP_TARGET_MODE_PRESSURE,
        DREAMER_PUMP_TARGET_MODE_FLOW,
    }
)

DREAMER_RESOLVED_CONTROL_FIELDS = (
    "pump_target_mode",
    "pressure_target_bar",
    "flow_target_ml_s",
    "valve_position",
    "temperature_target_c",
    "yield_stop_target_g",
    "stop",
)

_DREAMER_MAX_FLOW_TARGET_ML_S = 20.0
_DREAMER_RESOLVED_CONTROL_RANGES = {
    "pressure_target_bar": (0.0, DREAMER_MAX_PRESSURE_TARGET_BAR),
    "flow_target_ml_s": (0.0, _DREAMER_MAX_FLOW_TARGET_ML_S),
    "valve_position": (0.0, 1.0),
    "temperature_target_c": (
        DREAMER_MIN_TEMPERATURE_TARGET_C,
        DREAMER_MAX_TEMPERATURE_TARGET_C,
    ),
    "yield_stop_target_g": (5.0, DREAMER_MAX_YIELD_STOP_TARGET_G),
    "stop": (0.0, 1.0),
}


@dataclass(frozen=True)
class DreamerResolvedControl:
    """Applied machine-control state used consistently by the RSSM."""

    values: tuple[float, ...]
    observed_mask: tuple[float, ...]

    def __post_init__(self) -> None:
        expected = len(DREAMER_RESOLVED_CONTROL_FIELDS)
        if len(self.values) != expected or len(self.observed_mask) != expected:
            raise ValueError("Dreamer resolved control shape is invalid")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in self.values
        ):
            raise ValueError("Dreamer resolved control values must be finite")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) not in {0.0, 1.0}
            for value in self.observed_mask
        ):
            raise ValueError("Dreamer resolved control mask must be binary")
        mode = self.value("pump_target_mode")
        if not self.masked("pump_target_mode"):
            raise ValueError("Dreamer resolved pump target mode must be observed")
        if mode not in DREAMER_PUMP_TARGET_MODES:
            raise ValueError("Dreamer resolved pump target mode is invalid")
        expected_pressure = mode == DREAMER_PUMP_TARGET_MODE_PRESSURE
        expected_flow = mode == DREAMER_PUMP_TARGET_MODE_FLOW
        if self.masked("pressure_target_bar") != expected_pressure:
            raise ValueError("Dreamer resolved pressure target must match pressure mode")
        if self.masked("flow_target_ml_s") != expected_flow:
            raise ValueError("Dreamer resolved flow target must match flow mode")
        for field_name, (minimum, maximum) in _DREAMER_RESOLVED_CONTROL_RANGES.items():
            value = self.value(field_name)
            if self.masked(field_name):
                if not minimum <= value <= maximum:
                    raise ValueError(f"Dreamer resolved {field_name} is outside hard bounds")
            elif value != 0.0:
                raise ValueError(f"Dreamer resolved masked {field_name} must be zero")
        stop_value = self.value("stop")
        if self.masked("stop") and stop_value not in {0.0, 1.0}:
            raise ValueError("Dreamer resolved stop must be binary")

    def value(self, field_name: str) -> float:
        return self.values[DREAMER_RESOLVED_CONTROL_FIELDS.index(field_name)]

    def masked(self, field_name: str) -> bool:
        return self.observed_mask[DREAMER_RESOLVED_CONTROL_FIELDS.index(field_name)] > 0.5

    def to_dict(self) -> dict[str, float | bool | int | None]:
        result: dict[str, float | bool | int | None] = {}
        for index, field_name in enumerate(DREAMER_RESOLVED_CONTROL_FIELDS):
            if self.observed_mask[index] <= 0.5:
                result[field_name] = None
            elif field_name == "pump_target_mode":
                result[field_name] = int(round(self.values[index]))
            elif field_name == "stop":
                result[field_name] = self.values[index] > 0.5
            else:
                result[field_name] = self.values[index]
        return result


def resolve_applied_dreamer_control(
    *,
    pump_target_mode: int,
    pressure_target_bar: object | None = None,
    flow_target_ml_s: object | None = None,
    valve_position: object | None = None,
    temperature_target_c: object | None = None,
    yield_stop_target_g: object | None = None,
    stop: object | None = None,
) -> DreamerResolvedControl:
    if isinstance(pump_target_mode, bool) or pump_target_mode not in DREAMER_PUMP_TARGET_MODES:
        raise ValueError("Dreamer resolved pump target mode is invalid")
    if pump_target_mode != DREAMER_PUMP_TARGET_MODE_PRESSURE and pressure_target_bar is not None:
        raise ValueError("Dreamer resolved pressure target conflicts with pump target mode")
    if pump_target_mode != DREAMER_PUMP_TARGET_MODE_FLOW and flow_target_ml_s is not None:
        raise ValueError("Dreamer resolved flow target conflicts with pump target mode")
    values = {field_name: 0.0 for field_name in DREAMER_RESOLVED_CONTROL_FIELDS}
    mask = {field_name: 0.0 for field_name in DREAMER_RESOLVED_CONTROL_FIELDS}
    values["pump_target_mode"] = float(pump_target_mode)
    mask["pump_target_mode"] = 1.0

    if pump_target_mode == DREAMER_PUMP_TARGET_MODE_PRESSURE:
        values["pressure_target_bar"] = _bounded_float(
            pressure_target_bar,
            "pressure_target_bar",
            0.0,
            DREAMER_MAX_PRESSURE_TARGET_BAR,
        )
        mask["pressure_target_bar"] = 1.0
    elif pump_target_mode == DREAMER_PUMP_TARGET_MODE_FLOW:
        values["flow_target_ml_s"] = _bounded_float(
            flow_target_ml_s,
            "flow_target_ml_s",
            0.0,
            _DREAMER_MAX_FLOW_TARGET_ML_S,
        )
        mask["flow_target_ml_s"] = 1.0

    optional_ranges = {
        "valve_position": (valve_position, 0.0, 1.0),
        "temperature_target_c": (
            temperature_target_c,
            DREAMER_MIN_TEMPERATURE_TARGET_C,
            DREAMER_MAX_TEMPERATURE_TARGET_C,
        ),
        "yield_stop_target_g": (yield_stop_target_g, 5.0, DREAMER_MAX_YIELD_STOP_TARGET_G),
    }
    for field_name, (raw_value, minimum, maximum) in optional_ranges.items():
        if raw_value is None:
            continue
        values[field_name] = _bounded_float(raw_value, field_name, minimum, maximum)
        mask[field_name] = 1.0
    if stop is not None:
        if not isinstance(stop, bool):
            raise ValueError("Dreamer resolved stop must be boolean when observed")
        values["stop"] = 1.0 if stop else 0.0
        mask["stop"] = 1.0

    return DreamerResolvedControl(
        values=tuple(values[field_name] for field_name in DREAMER_RESOLVED_CONTROL_FIELDS),
        observed_mask=tuple(mask[field_name] for field_name in DREAMER_RESOLVED_CONTROL_FIELDS),
    )


def _bounded_float(value: object, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Dreamer resolved {label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"Dreamer resolved {label} is outside hard bounds")
    return parsed

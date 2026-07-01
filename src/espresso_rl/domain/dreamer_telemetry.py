from __future__ import annotations

import math
from dataclasses import dataclass

from espresso_rl.domain.models import FIXED_CADENCE_SAMPLE_INTERVAL_MS

DREAMER_LIVE_TELEMETRY_SCHEMA_VERSION = 1
DREAMER_AUTO_PROFILE_ID = "dreamer_auto"
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


@dataclass(frozen=True)
class DreamerLiveTelemetryCapabilities:
    pressure_control_allowed: bool
    flow_control_allowed: bool
    pump_control_allowed: bool
    valve_control_allowed: bool
    temperature_control_allowed: bool
    stop_control_allowed: bool

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"Dreamer telemetry capability {field_name} must be boolean")
        if not any(getattr(self, field_name) for field_name in self.__dataclass_fields__):
            raise ValueError("Dreamer telemetry must declare at least one control capability")

    def to_dict(self) -> dict[str, bool]:
        return {
            field_name: bool(getattr(self, field_name))
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class DreamerLiveTelemetry:
    """Canonical fixed-cadence observation for adaptive profile inference."""

    machine_id: str
    shot_id: str
    profile_id: str
    step_index: int
    elapsed_ms: int
    pressure_bar: float
    pressure_target_bar: float
    pump_flow_ml_s: float
    pump_flow_target_ml_s: float
    beverage_flow_g_s: float
    weight_g: float
    temperature_c: float
    temperature_target_c: float
    pump_target_mode: int
    valve_open: bool
    target_yield_g: float
    capabilities: DreamerLiveTelemetryCapabilities
    sample_interval_ms: int = FIXED_CADENCE_SAMPLE_INTERVAL_MS
    schema_version: int = DREAMER_LIVE_TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DREAMER_LIVE_TELEMETRY_SCHEMA_VERSION:
            raise ValueError("Dreamer telemetry schema_version is unsupported")
        if self.sample_interval_ms != FIXED_CADENCE_SAMPLE_INTERVAL_MS:
            raise ValueError(
                f"Dreamer telemetry sample_interval_ms must be {FIXED_CADENCE_SAMPLE_INTERVAL_MS}"
            )
        _bounded_string(self.machine_id, "machine_id", maximum=160)
        _bounded_string(self.shot_id, "shot_id", maximum=200)
        _bounded_string(self.profile_id, "profile_id", maximum=120)
        _non_negative_int(self.step_index, "step_index")
        _non_negative_int(self.elapsed_ms, "elapsed_ms")
        expected_elapsed_ms = self.step_index * self.sample_interval_ms
        if self.elapsed_ms != expected_elapsed_ms:
            raise ValueError("Dreamer telemetry elapsed_ms does not match its fixed-cadence step")

        ranges = {
            "pressure_bar": (0.0, 15.0),
            "pressure_target_bar": (0.0, 15.0),
            "pump_flow_ml_s": (0.0, 20.0),
            "pump_flow_target_ml_s": (0.0, 20.0),
            "beverage_flow_g_s": (0.0, 20.0),
            "weight_g": (-1.0, 120.0),
            "temperature_c": (0.0, 160.0),
            "temperature_target_c": (0.0, 160.0),
            "target_yield_g": (5.0, 90.0),
        }
        for field_name, (minimum, maximum) in ranges.items():
            value = _bounded_float(getattr(self, field_name), field_name, minimum, maximum)
            object.__setattr__(self, field_name, value)

        if isinstance(self.pump_target_mode, bool) or self.pump_target_mode not in DREAMER_PUMP_TARGET_MODES:
            raise ValueError("Dreamer telemetry pump_target_mode is invalid")
        if not isinstance(self.valve_open, bool):
            raise ValueError("Dreamer telemetry valve_open must be boolean")
        if not isinstance(self.capabilities, DreamerLiveTelemetryCapabilities):
            raise ValueError("Dreamer telemetry capabilities are invalid")

    @property
    def episode_key(self) -> str:
        return f"{self.machine_id}|{self.shot_id}|{self.profile_id}"

    def observation(self) -> tuple[float, ...]:
        return (
            self.pressure_bar,
            self.pump_flow_ml_s,
            self.beverage_flow_g_s,
            self.weight_g,
            self.temperature_c,
        )

    def observed_profile_targets(self) -> tuple[float, ...]:
        return (
            self.pressure_target_bar,
            self.pump_flow_target_ml_s,
            self.temperature_target_c,
            float(self.pump_target_mode),
            1.0 if self.valve_open else 0.0,
        )


def _bounded_string(value: object, field_name: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"Dreamer telemetry {field_name} is invalid")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Dreamer telemetry {field_name} must be a non-negative integer")
    return value


def _bounded_float(value: object, field_name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Dreamer telemetry {field_name} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Dreamer telemetry {field_name} must be finite")
    if not minimum <= parsed <= maximum:
        raise ValueError(f"Dreamer telemetry {field_name} is out of range")
    return parsed

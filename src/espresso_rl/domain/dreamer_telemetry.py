from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from espresso_rl.domain.dreamer_episodes import validate_dreamer_episode
from espresso_rl.domain.dreamer_taste import normalize_dreamer_taste_objective
from espresso_rl.domain.models import FIXED_CADENCE_SAMPLE_INTERVAL_MS

DREAMER_LIVE_TELEMETRY_SCHEMA_VERSION = 2
DREAMER_LIVE_CONTEXT_WINDOW_SIZE = 16
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
    pump_mode_control_allowed: bool
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


@dataclass(frozen=True)
class DreamerLiveEpisodeContext:
    install_id: str
    machine_id: str
    timestamp: int
    bean_context_id: str | None
    bean_context_name: str | None
    grinder_context_id: str | None
    relative_grind_steps_from_reference: float | None
    relative_grind_um_from_reference: float | None
    dose_g: float
    initial_target_yield_g: float
    microns_per_step: float
    step_direction: str
    profile_id: str
    profile_type: str
    profile_phase_count: int
    taste_objective: dict[str, str]
    historical_episodes: tuple[dict[str, Any], ...] = ()
    grind_observed: bool = False
    dose_observed: bool = False
    initial_target_yield_observed: bool = True

    def __post_init__(self) -> None:
        _bounded_string(self.install_id, "context install_id", maximum=160)
        _bounded_string(self.machine_id, "context machine_id", maximum=160)
        _non_negative_int(self.timestamp, "context timestamp")
        for field_name, maximum in (
            ("bean_context_id", 160),
            ("bean_context_name", 160),
            ("grinder_context_id", 160),
        ):
            value = getattr(self, field_name)
            if value is not None:
                _bounded_string(value, f"context {field_name}", maximum=maximum)
        for field_name in (
            "grind_observed",
            "dose_observed",
            "initial_target_yield_observed",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"Dreamer telemetry context {field_name} must be boolean")
        for field_name in (
            "relative_grind_steps_from_reference",
            "relative_grind_um_from_reference",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _finite_float(value, f"context {field_name}"))
        object.__setattr__(self, "dose_g", _bounded_float(self.dose_g, "context dose_g", 5.0, 30.0))
        object.__setattr__(
            self,
            "initial_target_yield_g",
            _bounded_float(self.initial_target_yield_g, "context initial_target_yield_g", 5.0, 90.0),
        )
        object.__setattr__(
            self,
            "microns_per_step",
            _bounded_float(self.microns_per_step, "context microns_per_step", 0.1, 100.0),
        )
        if self.step_direction not in {"higher_is_finer", "higher_is_coarser"}:
            raise ValueError("Dreamer telemetry context step_direction is invalid")
        _bounded_string(self.profile_id, "context profile_id", maximum=120)
        _bounded_string(self.profile_type, "context profile_type", maximum=80)
        if (
            isinstance(self.profile_phase_count, bool)
            or not isinstance(self.profile_phase_count, int)
            or not 1 <= self.profile_phase_count <= 128
        ):
            raise ValueError("Dreamer telemetry context profile_phase_count is invalid")
        object.__setattr__(self, "taste_objective", normalize_dreamer_taste_objective(self.taste_objective))
        if (
            not isinstance(self.historical_episodes, tuple)
            or len(self.historical_episodes) > DREAMER_LIVE_CONTEXT_WINDOW_SIZE
        ):
            raise ValueError("Dreamer telemetry context history exceeds the 16-episode window")
        for episode in self.historical_episodes:
            errors = validate_dreamer_episode(episode)
            if errors:
                raise ValueError(f"Dreamer telemetry context episode is invalid: {errors[0]}")
            group_key = episode["group_key"]
            if group_key["machine_id"] != self.machine_id:
                raise ValueError("Dreamer telemetry context episode machine does not match")
            if group_key.get("grinder_context_id") != self.grinder_context_id:
                raise ValueError("Dreamer telemetry context episode grinder does not match")

    @property
    def target_ratio(self) -> float:
        return self.initial_target_yield_g / self.dose_g

    def static_context(self) -> dict[str, Any]:
        return {
            "relative_grind_steps_from_reference": self.relative_grind_steps_from_reference or 0.0,
            "relative_grind_um_from_reference": self.relative_grind_um_from_reference or 0.0,
            "dose_g": self.dose_g,
            "initial_target_yield_g": self.initial_target_yield_g,
            "target_ratio": self.target_ratio,
            "grind_observed": self.grind_observed,
            "dose_observed": self.dose_observed,
            "initial_target_yield_observed": self.initial_target_yield_observed,
            "microns_per_step": self.microns_per_step,
            "step_direction": self.step_direction,
            "profile_id": self.profile_id,
            "profile_type": self.profile_type,
            "profile_phase_count": self.profile_phase_count,
            "taste_objective": dict(self.taste_objective),
        }


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


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Dreamer telemetry {field_name} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Dreamer telemetry {field_name} must be finite")
    return parsed

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from espresso_rl.domain.models import FIXED_CADENCE_SAMPLE_INTERVAL_MS
from espresso_rl.domain.recipe_limits import RECIPE_DOMAIN_OUTPUT_MAX_G

MIN_VALID_EPOCH_MS = 1_600_000_000_000
MAX_INT64 = (1 << 63) - 1


def _identifier(value: str, field_name: str, max_length: int = 160) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > max_length
    ):
        raise ValueError(f"{field_name} must be a non-empty short string")
    return value


def _integer(value: int, field_name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{field_name} out of range")
    return value


def _number(value: float, field_name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{field_name} out of range")
    return parsed


@dataclass(frozen=True)
class LiveShotStartedEvent:
    shot_id: str
    install_id: str
    machine_id: str
    timestamp_ms: int
    sample_interval_ms: int
    weight_source: str
    flow_source: str
    schema_version: int = 1

    event_type: ClassVar[str] = "live_shot_started"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported live-shot schema_version")
        object.__setattr__(self, "shot_id", _identifier(self.shot_id, "shot_id"))
        object.__setattr__(self, "install_id", _identifier(self.install_id, "install_id"))
        object.__setattr__(self, "machine_id", _identifier(self.machine_id, "machine_id"))
        object.__setattr__(
            self,
            "timestamp_ms",
            _integer(self.timestamp_ms, "timestamp_ms", MIN_VALID_EPOCH_MS, MAX_INT64),
        )
        if self.sample_interval_ms != FIXED_CADENCE_SAMPLE_INTERVAL_MS:
            raise ValueError(
                f"sample_interval_ms must be {FIXED_CADENCE_SAMPLE_INTERVAL_MS}"
            )
        object.__setattr__(self, "weight_source", _identifier(self.weight_source, "weight_source", 80))
        object.__setattr__(self, "flow_source", _identifier(self.flow_source, "flow_source", 80))


@dataclass(frozen=True)
class LiveShotSampleEvent:
    shot_id: str
    install_id: str
    machine_id: str
    timestamp_ms: int
    sequence: int
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
    schema_version: int = 1

    event_type: ClassVar[str] = "live_shot_sample"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported live-shot schema_version")
        object.__setattr__(self, "shot_id", _identifier(self.shot_id, "shot_id"))
        object.__setattr__(self, "install_id", _identifier(self.install_id, "install_id"))
        object.__setattr__(self, "machine_id", _identifier(self.machine_id, "machine_id"))
        object.__setattr__(
            self,
            "timestamp_ms",
            _integer(self.timestamp_ms, "timestamp_ms", MIN_VALID_EPOCH_MS, MAX_INT64),
        )
        object.__setattr__(self, "sequence", _integer(self.sequence, "sequence", 0, 65_535))
        object.__setattr__(self, "elapsed_ms", _integer(self.elapsed_ms, "elapsed_ms", 0, 65_535))
        ranges = {
            "pressure_bar": (0.0, 15.0),
            "pressure_target_bar": (0.0, 15.0),
            "pump_flow_ml_s": (0.0, 20.0),
            "pump_flow_target_ml_s": (0.0, 20.0),
            "beverage_flow_g_s": (0.0, 20.0),
            "weight_g": (-1.0, RECIPE_DOMAIN_OUTPUT_MAX_G),
            "temperature_c": (0.0, 160.0),
            "temperature_target_c": (0.0, 160.0),
        }
        for field_name, (minimum, maximum) in ranges.items():
            object.__setattr__(
                self,
                field_name,
                _number(getattr(self, field_name), field_name, minimum, maximum),
            )
        object.__setattr__(
            self,
            "pump_target_mode",
            _integer(self.pump_target_mode, "pump_target_mode", 0, 2),
        )
        if not isinstance(self.valve_open, bool):
            raise ValueError("valve_open must be boolean")


@dataclass(frozen=True)
class LiveShotEndedEvent:
    shot_id: str
    install_id: str
    machine_id: str
    timestamp_ms: int
    final_sequence: int
    elapsed_ms: int
    end_state: str
    schema_version: int = 1

    event_type: ClassVar[str] = "live_shot_ended"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported live-shot schema_version")
        object.__setattr__(self, "shot_id", _identifier(self.shot_id, "shot_id"))
        object.__setattr__(self, "install_id", _identifier(self.install_id, "install_id"))
        object.__setattr__(self, "machine_id", _identifier(self.machine_id, "machine_id"))
        object.__setattr__(
            self,
            "timestamp_ms",
            _integer(self.timestamp_ms, "timestamp_ms", MIN_VALID_EPOCH_MS, MAX_INT64),
        )
        object.__setattr__(
            self,
            "final_sequence",
            _integer(self.final_sequence, "final_sequence", 0, 65_535),
        )
        object.__setattr__(self, "elapsed_ms", _integer(self.elapsed_ms, "elapsed_ms", 0, 65_535))
        if self.end_state not in {"finished", "manual_or_interrupted"}:
            raise ValueError("end_state is invalid")


LiveShotEvent = LiveShotStartedEvent | LiveShotSampleEvent | LiveShotEndedEvent


class LiveShotSessionStatus(StrEnum):
    ACTIVE = "active"
    ENDED = "ended"
    RECONCILED = "reconciled"
    EXPIRED = "expired"


@dataclass(frozen=True)
class LiveShotSession:
    shot_id: str
    install_id: str
    machine_id: str
    started_at_ms: int
    sample_interval_ms: int
    weight_source: str
    flow_source: str
    status: LiveShotSessionStatus = LiveShotSessionStatus.ACTIVE
    last_sequence: int | None = None
    sample_count: int = 0
    gap_count: int = 0
    ended_at_ms: int | None = None
    end_state: str | None = None
    reconciled_at_ms: int | None = None
    updated_at_ms: int = 0

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .models import (
    GrinderCalibrationMode,
    GrinderStepDirection,
    MachineState,
    RecommendationApplyStatus,
    RecommendationDecision,
    Recipe,
    ShotType,
    VALID_TASTE_TAGS,
)

VALID_CORRECTION_TAGS = {
    "changed_manually",
    "bad_puck_prep",
    "channeling_suspected",
    "utility_brew",
    "did_not_follow_grind",
    "did_not_follow_dose",
    "did_not_follow_yield",
}

VALID_FINAL_PHASE_TYPES = {"preinfusion", "brew"}
VALID_FINAL_PUMP_TARGETS = {"simple", "pressure", "flow"}
VALID_SHOT_END_STATES = {"finished", "manual_or_interrupted", "unknown"}


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must contain only finite numbers")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only finite numbers") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must contain only finite numbers")
    return parsed


def _numbers(values: list[Any], field_name: str) -> list[float]:
    try:
        return [_number(v, field_name) for v in values]
    except TypeError as exc:
        raise ValueError(f"{field_name} must contain only finite numbers") from exc


def _optional_string(value: Any, field_name: str, max_len: int = 120) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a short string")
    parsed = value.strip()
    if not parsed:
        return None
    if len(parsed) > max_len:
        raise ValueError(f"{field_name} must be a short string")
    return parsed


def _optional_enum(value: Any, field_name: str, allowed: set[str]) -> str | None:
    parsed = _optional_string(value, field_name, max_len=80)
    if parsed is None:
        return None
    if parsed not in allowed:
        raise ValueError(f"{field_name} is invalid")
    return parsed


def _optional_int_range(value: Any, field_name: str, minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} out of range")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field_name} out of range")
    return value


def _optional_number_range(value: Any, field_name: str, minimum: float, maximum: float) -> float | None:
    if value is None:
        return None
    parsed = _number(value, field_name)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{field_name} out of range")
    return parsed


@dataclass(frozen=True)
class ShotProfileEvent:
    shot_id: str
    install_id: str
    machine_id: str
    machine_adapter: str
    timestamp: int
    time_ms: list[float]
    pressure: list[float]
    target_pressure: list[float]
    flow: list[float]
    target_flow: list[float]
    weight: list[float]
    microns_per_step: float
    dose_in_g: float
    target_yield_g: float
    schema_version: int = 1
    relative_grind_steps_from_reference: float | None = None
    beverage_out_g: float | None = None
    shot_time_s: float | None = None
    bean_context_id: str | None = None
    bean_context_name: str | None = None
    grinder_context_id: str | None = None
    grinder_calibration_mode: GrinderCalibrationMode = GrinderCalibrationMode.RELATIVE_CALIBRATED
    grinder_step_direction: GrinderStepDirection = GrinderStepDirection.HIGHER_IS_FINER
    grinder_reference_label: str = "reference"
    current_absolute_step: float | None = None
    absolute_reference_step: float | None = None
    recommendation_id: str | None = None
    shot_type: ShotType = ShotType.ESPRESSO
    utility: bool = False
    exclude_from_local_optimization: bool = False
    local_optimization_enabled: bool = True
    optimization_weight: float | None = None
    rating_prompt_allowed: bool = True
    weight_source: str | None = None
    flow_source: str | None = None
    flow_units: str | None = None
    pump_flow_source: str | None = None
    pump_flow_units: str | None = None
    pump_flow_calibration_required: bool = False
    profile_id: str | None = None
    profile_label: str | None = None
    profile_type: str | None = None
    profile_phase_count: int | None = None
    final_phase_index: int | None = None
    final_phase_name: str | None = None
    final_phase_type: str | None = None
    final_phase_elapsed_s: float | None = None
    final_pump_target: str | None = None
    final_target_pressure: float | None = None
    final_target_flow: float | None = None
    final_valve_open: bool | None = None
    profile_temperature_c: float | None = None
    final_phase_temperature_c: float | None = None
    shot_end_state: str | None = None

    event_type: str = field(default="shot_profile", init=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported shot profile schema_version")
        object.__setattr__(self, "shot_type", ShotType(self.shot_type))
        object.__setattr__(self, "time_ms", _numbers(self.time_ms, "time_ms"))
        for name in ("pressure", "target_pressure", "flow", "target_flow", "weight"):
            object.__setattr__(self, name, _numbers(getattr(self, name), name))
        object.__setattr__(self, "microns_per_step", _number(self.microns_per_step, "microns_per_step"))
        object.__setattr__(self, "dose_in_g", _number(self.dose_in_g, "dose_in_g"))
        object.__setattr__(self, "target_yield_g", _number(self.target_yield_g, "target_yield_g"))
        if self.relative_grind_steps_from_reference is not None:
            object.__setattr__(self, "relative_grind_steps_from_reference", _number(self.relative_grind_steps_from_reference, "relative_grind_steps_from_reference"))
        object.__setattr__(
            self,
            "grinder_calibration_mode",
            GrinderCalibrationMode(self.grinder_calibration_mode),
        )
        object.__setattr__(
            self,
            "grinder_step_direction",
            GrinderStepDirection(self.grinder_step_direction),
        )
        object.__setattr__(
            self,
            "grinder_reference_label",
            _optional_string(self.grinder_reference_label, "grinder_reference_label", 80) or "reference",
        )
        if self.current_absolute_step is not None:
            object.__setattr__(
                self,
                "current_absolute_step",
                _number(self.current_absolute_step, "current_absolute_step"),
            )
        if self.absolute_reference_step is not None:
            object.__setattr__(
                self,
                "absolute_reference_step",
                _number(self.absolute_reference_step, "absolute_reference_step"),
            )
        if self.beverage_out_g is not None:
            object.__setattr__(self, "beverage_out_g", _number(self.beverage_out_g, "beverage_out_g"))
        if self.shot_time_s is not None:
            object.__setattr__(self, "shot_time_s", _number(self.shot_time_s, "shot_time_s"))
        object.__setattr__(self, "bean_context_id", _optional_string(self.bean_context_id, "bean_context_id", 160))
        object.__setattr__(self, "bean_context_name", _optional_string(self.bean_context_name, "bean_context_name"))
        object.__setattr__(self, "grinder_context_id", _optional_string(self.grinder_context_id, "grinder_context_id"))
        if self.optimization_weight is not None:
            object.__setattr__(self, "optimization_weight", _number(self.optimization_weight, "optimization_weight"))
        object.__setattr__(self, "weight_source", _optional_string(self.weight_source, "weight_source", 80))
        object.__setattr__(self, "flow_source", _optional_string(self.flow_source, "flow_source", 80))
        object.__setattr__(self, "flow_units", _optional_string(self.flow_units, "flow_units", 40))
        object.__setattr__(self, "pump_flow_source", _optional_string(self.pump_flow_source, "pump_flow_source", 80))
        object.__setattr__(self, "pump_flow_units", _optional_string(self.pump_flow_units, "pump_flow_units", 40))
        object.__setattr__(self, "profile_id", _optional_string(self.profile_id, "profile_id", 120))
        object.__setattr__(self, "profile_label", _optional_string(self.profile_label, "profile_label", 120))
        object.__setattr__(self, "profile_type", _optional_string(self.profile_type, "profile_type", 80))
        object.__setattr__(
            self,
            "profile_phase_count",
            _optional_int_range(self.profile_phase_count, "profile_phase_count", 0, 100),
        )
        object.__setattr__(
            self,
            "final_phase_index",
            _optional_int_range(self.final_phase_index, "final_phase_index", 0, 100),
        )
        object.__setattr__(self, "final_phase_name", _optional_string(self.final_phase_name, "final_phase_name", 120))
        object.__setattr__(
            self,
            "final_phase_type",
            _optional_enum(self.final_phase_type, "final_phase_type", VALID_FINAL_PHASE_TYPES),
        )
        object.__setattr__(
            self,
            "final_phase_elapsed_s",
            _optional_number_range(self.final_phase_elapsed_s, "final_phase_elapsed_s", 0, 600),
        )
        object.__setattr__(
            self,
            "final_pump_target",
            _optional_enum(self.final_pump_target, "final_pump_target", VALID_FINAL_PUMP_TARGETS),
        )
        object.__setattr__(
            self,
            "final_target_pressure",
            _optional_number_range(self.final_target_pressure, "final_target_pressure", 0, 15),
        )
        object.__setattr__(
            self,
            "final_target_flow",
            _optional_number_range(self.final_target_flow, "final_target_flow", 0, 25),
        )
        if self.final_valve_open is not None and not isinstance(self.final_valve_open, bool):
            raise ValueError("final_valve_open must be boolean")
        object.__setattr__(
            self,
            "profile_temperature_c",
            _optional_number_range(self.profile_temperature_c, "profile_temperature_c", 0, 160),
        )
        object.__setattr__(
            self,
            "final_phase_temperature_c",
            _optional_number_range(self.final_phase_temperature_c, "final_phase_temperature_c", 0, 160),
        )
        object.__setattr__(
            self,
            "shot_end_state",
            _optional_enum(self.shot_end_state, "shot_end_state", VALID_SHOT_END_STATES),
        )
        lengths = {
            len(self.time_ms),
            len(self.pressure),
            len(self.target_pressure),
            len(self.flow),
            len(self.target_flow),
            len(self.weight),
        }
        if len(lengths) != 1:
            raise ValueError("shot profile arrays must have matching lengths")
        if self.microns_per_step <= 0:
            raise ValueError("microns_per_step must be positive")
        if self.dose_in_g <= 0:
            raise ValueError("dose_in_g must be positive")
        if self.target_yield_g <= 0:
            raise ValueError("target_yield_g must be positive")
        if self.beverage_out_g is not None and self.beverage_out_g <= 0:
            raise ValueError("beverage_out_g must be positive when present")
        if self.shot_time_s is not None and self.shot_time_s <= 0:
            raise ValueError("shot_time_s must be positive when present")
        if self.optimization_weight is not None and not 0.0 <= self.optimization_weight <= 1.0:
            raise ValueError("optimization_weight must be between 0 and 1")


@dataclass(frozen=True)
class ShotFeedbackEvent:
    shot_id: str
    install_id: str
    machine_id: str
    timestamp: int
    recommendation_id: str | None = None
    rating: int | None = None
    taste_tags: list[str] = field(default_factory=list)
    user_note: str | None = None
    skipped: bool = False
    source: str = "unknown"
    schema_version: int = 1

    event_type: str = field(default="shot_feedback", init=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported feedback schema_version")
        if self.rating is not None and not 1 <= self.rating <= 5:
            raise ValueError("rating must be 1..5 or null")
        if self.skipped and (self.rating is not None or self.taste_tags):
            raise ValueError("skipped feedback cannot include a rating or taste tags")
        if not self.skipped and self.rating is None:
            raise ValueError("rating is required unless feedback is skipped")
        invalid_tags = set(self.taste_tags) - VALID_TASTE_TAGS
        if invalid_tags:
            raise ValueError(f"invalid taste tags: {sorted(invalid_tags)}")


@dataclass(frozen=True)
class ShotCorrectionEvent:
    shot_id: str
    install_id: str
    machine_id: str
    timestamp: int
    exclude_from_local_optimization: bool | None = None
    shot_type: ShotType | None = None
    grind_followed: bool | None = None
    dose_followed: bool | None = None
    yield_followed: bool | None = None
    correction_tags: list[str] = field(default_factory=list)
    source: str = "unknown"
    schema_version: int = 1

    event_type: str = field(default="shot_correction", init=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported correction schema_version")
        if not self.shot_id:
            raise ValueError("shot_id is required")
        if self.shot_type is not None:
            object.__setattr__(self, "shot_type", ShotType(self.shot_type))
        invalid_tags = set(self.correction_tags) - VALID_CORRECTION_TAGS
        if invalid_tags:
            raise ValueError(f"invalid correction tags: {sorted(invalid_tags)}")


@dataclass(frozen=True)
class UploadQueueMaintenanceEvent:
    install_id: str
    machine_id: str
    timestamp: int
    action: str = "requeue_valid_rejected"
    limit: int = 25
    bean_context_id: str | None = None
    grinder_context_id: str | None = None
    local_record_id: str | None = None
    source: str = "unknown"
    schema_version: int = 1

    event_type: str = field(default="upload_queue_maintenance", init=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported upload maintenance schema_version")
        if self.action not in {"requeue_valid_rejected", "purge_rejected"}:
            raise ValueError("unsupported upload maintenance action")
        if not 1 <= int(self.limit) <= 500:
            raise ValueError("upload maintenance limit must be between 1 and 500")
        object.__setattr__(self, "limit", int(self.limit))
        object.__setattr__(self, "grinder_context_id", _optional_string(self.grinder_context_id, "grinder_context_id"))
        object.__setattr__(self, "local_record_id", _optional_string(self.local_record_id, "local_record_id", 160))


@dataclass(frozen=True)
class RecommendationDecisionEvent:
    recommendation_id: str
    decision: RecommendationDecision
    timestamp: int
    install_id: str | None = None
    machine_id: str | None = None
    edited_fields: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    schema_version: int = 1

    event_type: str = field(default="recommendation_decision", init=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported decision schema_version")
        object.__setattr__(self, "decision", RecommendationDecision(self.decision))
        allowed_edits = {
            "projected_relative_step_from_reference",
            "next_dose_g",
            "target_yield_g",
            "target_ratio",
        }
        unknown = set(self.edited_fields) - allowed_edits
        if unknown:
            raise ValueError(f"unsupported edited fields: {sorted(unknown)}")


@dataclass(frozen=True)
class RecommendationApplyEvent:
    recommendation_id: str
    status: RecommendationApplyStatus
    timestamp: int
    install_id: str | None = None
    machine_id: str | None = None
    applied_fields: dict[str, Any] = field(default_factory=dict)
    manual_fields: list[str] = field(default_factory=list)
    failed_fields: dict[str, Any] = field(default_factory=dict)
    message: str | None = None
    source: str = "unknown"
    schema_version: int = 1

    event_type: str = field(default="recommendation_apply", init=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported apply schema_version")
        object.__setattr__(self, "status", RecommendationApplyStatus(self.status))
        allowed_fields = {
            "projected_relative_step_from_reference",
            "projected_relative_grind_um_from_reference",
            "next_dose_g",
            "target_yield_g",
            "target_ratio",
        }
        unknown_applied = set(self.applied_fields) - allowed_fields
        if unknown_applied:
            raise ValueError(f"unsupported applied fields: {sorted(unknown_applied)}")
        unknown_failed = set(self.failed_fields) - allowed_fields
        if unknown_failed:
            raise ValueError(f"unsupported failed fields: {sorted(unknown_failed)}")
        unknown_manual = set(self.manual_fields) - allowed_fields
        if unknown_manual:
            raise ValueError(f"unsupported manual fields: {sorted(unknown_manual)}")


@dataclass(frozen=True)
class MachineStateEvent:
    install_id: str
    machine_id: str
    machine_adapter: str
    timestamp: int
    state: MachineState
    schema_version: int = 1
    bean_context_id: str | None = None
    bean_context_name: str | None = None
    grinder_context_id: str | None = None
    grinder_calibration_mode: GrinderCalibrationMode = GrinderCalibrationMode.RELATIVE_CALIBRATED
    grinder_step_direction: GrinderStepDirection = GrinderStepDirection.HIGHER_IS_FINER
    grinder_reference_label: str = "reference"
    relative_grind_steps_from_reference: float | None = None
    microns_per_step: float | None = None
    current_absolute_step: float | None = None
    absolute_reference_step: float | None = None
    dose_in_g: float | None = None
    target_yield_g: float | None = None
    source: str = "unknown"

    event_type: str = field(default="machine_state", init=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported machine state schema_version")
        object.__setattr__(self, "state", MachineState(self.state))
        object.__setattr__(self, "bean_context_id", _optional_string(self.bean_context_id, "bean_context_id", 160))
        object.__setattr__(self, "bean_context_name", _optional_string(self.bean_context_name, "bean_context_name"))
        object.__setattr__(self, "grinder_context_id", _optional_string(self.grinder_context_id, "grinder_context_id"))
        object.__setattr__(
            self,
            "grinder_calibration_mode",
            GrinderCalibrationMode(self.grinder_calibration_mode),
        )
        object.__setattr__(
            self,
            "grinder_step_direction",
            GrinderStepDirection(self.grinder_step_direction),
        )
        object.__setattr__(
            self,
            "grinder_reference_label",
            _optional_string(self.grinder_reference_label, "grinder_reference_label", 80) or "reference",
        )
        if self.current_absolute_step is not None:
            object.__setattr__(
                self,
                "current_absolute_step",
                _number(self.current_absolute_step, "current_absolute_step"),
            )
        if self.absolute_reference_step is not None:
            object.__setattr__(
                self,
                "absolute_reference_step",
                _number(self.absolute_reference_step, "absolute_reference_step"),
            )
        if self.microns_per_step is not None and self.microns_per_step <= 0:
            raise ValueError("microns_per_step must be positive when present")
        if self.dose_in_g is not None and self.dose_in_g <= 0:
            raise ValueError("dose_in_g must be positive when present")
        if self.target_yield_g is not None and self.target_yield_g <= 0:
            raise ValueError("target_yield_g must be positive when present")

    def current_recipe(self) -> Recipe | None:
        if (
            self.relative_grind_steps_from_reference is None
            or self.microns_per_step is None
            or self.dose_in_g is None
            or self.target_yield_g is None
        ):
            return None
        return Recipe(
            relative_grind_steps_from_reference=self.relative_grind_steps_from_reference,
            microns_per_step=self.microns_per_step,
            dose_g=self.dose_in_g,
            target_yield_g=self.target_yield_g,
            grinder_step_direction=self.grinder_step_direction,
        )

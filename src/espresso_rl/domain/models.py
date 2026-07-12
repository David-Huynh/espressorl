from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from espresso_rl.domain.taste_goal import TasteGoal


PROFILE_SHAPE = (5, 100)
PROFILE_DTYPE = np.float32
AUX_PROFILE_SHAPE = (PROFILE_SHAPE[1],)
PUMP_TARGET_MODE_DTYPE = np.uint8
PUMP_TARGET_MODE_VALUES = {0, 1, 2}
FIXED_CADENCE_SAMPLE_INTERVAL_MS = 250
FIXED_CADENCE_MAX_STEPS = 500


def _optional_profile_vector(value: np.ndarray | list[float] | None, field_name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=PROFILE_DTYPE)
    if array.shape != AUX_PROFILE_SHAPE or not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must have shape {AUX_PROFILE_SHAPE}")
    return array


def _optional_pump_target_mode_vector(value: np.ndarray | list[int] | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=PUMP_TARGET_MODE_DTYPE)
    if array.shape != AUX_PROFILE_SHAPE or any(int(mode) not in PUMP_TARGET_MODE_VALUES for mode in array):
        raise ValueError(f"pump_target_mode_profile must have shape {AUX_PROFILE_SHAPE}")
    return array


class RecommendationMode(str, Enum):
    CPBO_GLOBAL_PREVIOUS = "cpbo_global_previous"
    CPBO_BEST_INCUMBENT = "cpbo_best_incumbent"


class RecommendationStatus(str, Enum):
    PENDING = "pending"
    SHOWN = "shown"
    ACCEPTED = "accepted"
    EDITED = "edited"
    IGNORED = "ignored"
    EXPIRED = "expired"
    USED = "used"
    SUPERSEDED = "superseded"


class RecommendationDecision(str, Enum):
    ACCEPTED = "accepted"
    EDITED = "edited"
    IGNORED = "ignored"
    DISMISSED = "dismissed"
    UNKNOWN = "unknown"


class RecommendationApplyStatus(str, Enum):
    UNKNOWN = "unknown"
    APPLIED = "applied"
    PARTIALLY_APPLIED = "partially_applied"
    MANUAL_REQUIRED = "manual_required"
    FAILED = "failed"


class FollowThroughState(str, Enum):
    FOLLOWED = "followed"
    PARTIALLY_FOLLOWED = "partially_followed"
    NOT_FOLLOWED = "not_followed"
    UNKNOWN = "unknown"


class MachineState(str, Enum):
    WAKE = "wake"
    IDLE = "idle"
    BREWING = "brewing"
    SLEEP = "sleep"
    STANDBY = "standby"
    UNKNOWN = "unknown"


class ShotType(str, Enum):
    ESPRESSO = "espresso"
    UTILITY_FLUSH = "utility_flush"
    CLEANING = "cleaning"
    CALIBRATION = "calibration"
    UNKNOWN = "unknown"


class UploadQueueStatus(str, Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    FAILED = "failed"
    REJECTED = "rejected"
    DISABLED = "disabled"


class GrinderCalibrationMode(str, Enum):
    UNCALIBRATED = "uncalibrated"
    RELATIVE_CALIBRATED = "relative_calibrated"
    ABSOLUTE_DISPLAY_CALIBRATED = "absolute_display_calibrated"


class GrinderStepDirection(str, Enum):
    HIGHER_IS_FINER = "higher_is_finer"
    HIGHER_IS_COARSER = "higher_is_coarser"


class GrinderAdjustmentMode(str, Enum):
    STEPPED = "stepped"
    STEPLESS = "stepless"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def now_ts() -> int:
    return int(time.time())


@dataclass(frozen=True)
class Recipe:
    relative_grind_steps_from_reference: float
    microns_per_step: float
    dose_g: float
    target_yield_g: float
    target_ratio: float | None = None
    grinder_step_direction: GrinderStepDirection = GrinderStepDirection.HIGHER_IS_FINER
    grinder_adjustment_mode: GrinderAdjustmentMode = GrinderAdjustmentMode.STEPPED

    def __post_init__(self) -> None:
        object.__setattr__(self, "grinder_step_direction", GrinderStepDirection(self.grinder_step_direction))
        object.__setattr__(self, "grinder_adjustment_mode", GrinderAdjustmentMode(self.grinder_adjustment_mode))
        if self.microns_per_step <= 0:
            raise ValueError("microns_per_step must be positive")
        if self.dose_g <= 0:
            raise ValueError("dose_g must be positive")
        if self.target_yield_g <= 0:
            raise ValueError("target_yield_g must be positive")
        if self.target_ratio is None:
            object.__setattr__(self, "target_ratio", self.target_yield_g / self.dose_g)

    @property
    def relative_grind_um_from_reference(self) -> float:
        return self.relative_grind_steps_from_reference * self.microns_per_step * self.grinder_direction_sign

    @property
    def grinder_direction_sign(self) -> int:
        return 1 if self.grinder_step_direction == GrinderStepDirection.HIGHER_IS_FINER else -1


@dataclass(frozen=True)
class SafetyBounds:
    dose_min_g: float = 14.0
    dose_max_g: float = 22.0
    target_yield_min_g: float = 20.0
    target_yield_max_g: float = 60.0
    target_ratio_min: float = 1.2
    target_ratio_max: float = 3.5
    max_grind_delta_steps_from_current: int = 5
    max_dose_delta_g: float = 1.0
    max_yield_delta_g: float = 8.0


@dataclass
class Recommendation:
    recommendation_id: str
    created_at: int
    updated_at: int
    expires_at: int | None
    install_id: str
    machine_id: str
    bean_context_id: str | None
    grind_delta_steps_from_current: float
    grind_delta_um_from_current: float
    projected_relative_step_from_reference: float
    projected_relative_grind_um_from_reference: float
    next_dose_g: float
    target_yield_g: float
    target_ratio: float
    mode: RecommendationMode
    confidence: float
    reason: str
    status: RecommendationStatus = RecommendationStatus.PENDING
    grinder_context_id: str | None = None
    profile_id: str | None = None
    raw_profile_hash: str | None = None
    shown_count: int = 0
    accepted_at: int | None = None
    ignored_at: int | None = None
    edited_at: int | None = None
    used_at: int | None = None
    superseded_at: int | None = None
    source_shot_id: str | None = None
    optimization_run_id: str | None = None
    comparison_anchor_shot_id: str | None = None
    comparison_mode: str | None = None
    preference_feedback_required: bool = False
    apply_status: RecommendationApplyStatus = RecommendationApplyStatus.UNKNOWN
    apply_acknowledged_at: int | None = None
    applied_fields: dict[str, Any] = field(default_factory=dict)
    manual_fields: list[str] = field(default_factory=list)
    apply_error: str | None = None
    grinder_calibration_mode: GrinderCalibrationMode = GrinderCalibrationMode.RELATIVE_CALIBRATED
    grinder_step_direction: GrinderStepDirection = GrinderStepDirection.HIGHER_IS_FINER
    grinder_adjustment_mode: GrinderAdjustmentMode = GrinderAdjustmentMode.STEPPED
    grinder_reference_label: str = "reference"
    current_absolute_step: float | None = None
    absolute_reference_step: float | None = None
    projected_absolute_step: float | None = None
    taste_goal: TasteGoal = field(default_factory=TasteGoal.balanced)

    def __post_init__(self) -> None:
        self.mode = RecommendationMode(self.mode)
        self.status = RecommendationStatus(self.status)
        self.apply_status = RecommendationApplyStatus(self.apply_status)
        self.grinder_calibration_mode = GrinderCalibrationMode(self.grinder_calibration_mode)
        self.grinder_step_direction = GrinderStepDirection(self.grinder_step_direction)
        self.grinder_adjustment_mode = GrinderAdjustmentMode(self.grinder_adjustment_mode)
        if not isinstance(self.taste_goal, TasteGoal):
            self.taste_goal = TasteGoal.from_dict(self.taste_goal)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        if self.next_dose_g <= 0:
            raise ValueError("next_dose_g must be positive")
        if self.target_yield_g <= 0:
            raise ValueError("target_yield_g must be positive")
        if self.target_ratio <= 0:
            raise ValueError("target_ratio must be positive")
        if self.comparison_mode not in {None, "global_previous", "best_incumbent"}:
            raise ValueError("comparison_mode is invalid")
        if not isinstance(self.preference_feedback_required, bool):
            raise ValueError("preference_feedback_required must be boolean")
        if self.preference_feedback_required:
            if not self.optimization_run_id:
                raise ValueError("preference feedback requires optimization_run_id")
            if not self.comparison_anchor_shot_id:
                raise ValueError("preference feedback requires comparison_anchor_shot_id")
            if self.comparison_mode is None:
                raise ValueError("preference feedback requires comparison_mode")

    def active_at(self, timestamp: int) -> bool:
        if self.status in {
            RecommendationStatus.IGNORED,
            RecommendationStatus.EXPIRED,
            RecommendationStatus.USED,
            RecommendationStatus.SUPERSEDED,
        }:
            return False
        return self.expires_at is None or self.expires_at > timestamp


@dataclass
class FollowThroughResult:
    state: FollowThroughState
    attribution_weight: float


@dataclass(frozen=True)
class StaleCheck:
    stale: bool
    reason: str | None = None


@dataclass
class UploadQueueItem:
    upload_id: str
    local_record_type: str
    local_record_id: str
    payload_hash: str
    payload_json: str
    status: UploadQueueStatus
    attempt_count: int = 0
    last_attempt_at: int | None = None
    next_retry_at: int | None = None
    error_message: str | None = None
    created_at: int = field(default_factory=now_ts)
    updated_at: int = field(default_factory=now_ts)

    def __post_init__(self) -> None:
        self.status = UploadQueueStatus(self.status)
        if not self.upload_id:
            raise ValueError("upload_id is required")
        if not self.local_record_type:
            raise ValueError("local_record_type is required")
        if not self.local_record_id:
            raise ValueError("local_record_id is required")
        if not self.payload_hash:
            raise ValueError("payload_hash is required")
        if not self.payload_json:
            raise ValueError("payload_json is required")


@dataclass(frozen=True)
class FixedCadenceShotSequence:
    """Canonical per-shot telemetry on an exact, model-safe time grid."""

    sample_interval_ms: int
    pressure_bar: np.ndarray
    pressure_target_bar: np.ndarray
    pump_flow_ml_s: np.ndarray
    pump_flow_target_ml_s: np.ndarray
    beverage_flow_g_s: np.ndarray
    weight_g: np.ndarray
    temperature_c: np.ndarray
    temperature_target_c: np.ndarray
    pump_target_mode: np.ndarray
    valve_open: np.ndarray

    def __post_init__(self) -> None:
        if self.sample_interval_ms != FIXED_CADENCE_SAMPLE_INTERVAL_MS:
            raise ValueError(
                f"sample_interval_ms must be {FIXED_CADENCE_SAMPLE_INTERVAL_MS}"
            )

        lengths: set[int] = set()
        ranges = {
            "pressure_bar": (0.0, 15.0),
            "pressure_target_bar": (0.0, 15.0),
            "pump_flow_ml_s": (0.0, 20.0),
            "pump_flow_target_ml_s": (0.0, 20.0),
            "beverage_flow_g_s": (0.0, 20.0),
            "weight_g": (-1.0, 120.0),
            "temperature_c": (0.0, 160.0),
            "temperature_target_c": (0.0, 160.0),
        }
        for field_name, (minimum, maximum) in ranges.items():
            array = np.asarray(getattr(self, field_name), dtype=PROFILE_DTYPE)
            if array.ndim != 1 or not np.all(np.isfinite(array)):
                raise ValueError(f"{field_name} must be a finite one-dimensional array")
            if np.any(array < minimum) or np.any(array > maximum):
                raise ValueError(f"{field_name} out of range")
            object.__setattr__(self, field_name, array)
            lengths.add(len(array))

        pump_modes = np.asarray(self.pump_target_mode, dtype=PUMP_TARGET_MODE_DTYPE)
        if pump_modes.ndim != 1 or any(int(mode) not in PUMP_TARGET_MODE_VALUES for mode in pump_modes):
            raise ValueError("pump_target_mode contains invalid values")
        object.__setattr__(self, "pump_target_mode", pump_modes)
        lengths.add(len(pump_modes))

        valve = np.asarray(self.valve_open, dtype=np.uint8)
        if valve.ndim != 1 or any(int(value) not in {0, 1} for value in valve):
            raise ValueError("valve_open contains invalid values")
        object.__setattr__(self, "valve_open", valve)
        lengths.add(len(valve))

        if len(lengths) != 1:
            raise ValueError("fixed-cadence sequence channels must have matching lengths")
        step_count = next(iter(lengths), 0)
        if not 2 <= step_count <= FIXED_CADENCE_MAX_STEPS:
            raise ValueError(
                f"fixed-cadence sequence must contain 2..{FIXED_CADENCE_MAX_STEPS} steps"
            )

    @property
    def step_count(self) -> int:
        return len(self.pressure_bar)

    def to_dict(self, *, ndigits: int = 4) -> dict[str, Any]:
        return {
            "sample_interval_ms": self.sample_interval_ms,
            "pressure_bar": _rounded_vector(self.pressure_bar, ndigits),
            "pressure_target_bar": _rounded_vector(self.pressure_target_bar, ndigits),
            "pump_flow_ml_s": _rounded_vector(self.pump_flow_ml_s, ndigits),
            "pump_flow_target_ml_s": _rounded_vector(self.pump_flow_target_ml_s, ndigits),
            "beverage_flow_g_s": _rounded_vector(self.beverage_flow_g_s, ndigits),
            "weight_g": _rounded_vector(self.weight_g, ndigits),
            "temperature_c": _rounded_vector(self.temperature_c, ndigits),
            "temperature_target_c": _rounded_vector(self.temperature_target_c, ndigits),
            "pump_target_mode": [int(value) for value in self.pump_target_mode],
            "valve_open": [bool(value) for value in self.valve_open],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FixedCadenceShotSequence":
        if not isinstance(value, dict):
            raise ValueError("fixed-cadence sequence must be an object")
        return cls(
            sample_interval_ms=value.get("sample_interval_ms"),
            pressure_bar=value.get("pressure_bar"),
            pressure_target_bar=value.get("pressure_target_bar"),
            pump_flow_ml_s=value.get("pump_flow_ml_s"),
            pump_flow_target_ml_s=value.get("pump_flow_target_ml_s"),
            beverage_flow_g_s=value.get("beverage_flow_g_s"),
            weight_g=value.get("weight_g"),
            temperature_c=value.get("temperature_c"),
            temperature_target_c=value.get("temperature_target_c"),
            pump_target_mode=value.get("pump_target_mode"),
            valve_open=value.get("valve_open"),
        )


def _rounded_vector(value: np.ndarray, ndigits: int) -> list[float]:
    return [round(float(item), ndigits) for item in value]


@dataclass
class ShotRecord:
    shot_id: str
    timestamp: int
    install_id: str
    machine_id: str
    machine_adapter: str
    profile: np.ndarray
    microns_per_step: float
    dose_in_g: float
    target_yield_g: float
    grind_observed: bool = True
    dose_observed: bool = True
    target_yield_observed: bool = True
    relative_grind_steps_from_reference: float | None = None
    relative_grind_um_from_reference: float | None = None
    beverage_out_g: float | None = None
    brew_ratio: float | None = None
    target_ratio: float | None = None
    shot_time_s: float | None = None
    bean_context_id: str | None = None
    bean_context_name: str | None = None
    grinder_context_id: str | None = None
    taste_goal: TasteGoal = field(default_factory=TasteGoal.balanced)
    recommendation_id: str | None = None
    raw_profile_available: bool = True
    raw_profile_hash: str | None = None
    recommended_grind_delta_steps_from_current: float | None = None
    recommended_grind_delta_um_from_current: float | None = None
    recommended_projected_relative_step_from_reference: float | None = None
    recommended_dose_g: float | None = None
    recommended_target_yield_g: float | None = None
    recommended_target_ratio: float | None = None
    recommendation_decision: RecommendationDecision = RecommendationDecision.UNKNOWN
    recommendation_followed: FollowThroughState = FollowThroughState.UNKNOWN
    recommendation_attribution_weight: float = 0.0
    shot_type: ShotType = ShotType.ESPRESSO
    exclude_from_local_optimization: bool = False
    optimization_weight: float = 1.0
    grind_followed: bool | None = None
    dose_followed: bool | None = None
    yield_followed: bool | None = None
    grind_recommendation_trust: float = 0.0
    dose_recommendation_trust: float = 0.0
    yield_recommendation_trust: float = 0.0
    weight_source: str | None = None
    flow_source: str | None = None
    flow_units: str | None = None
    pump_flow_source: str | None = None
    pump_flow_units: str | None = None
    pump_flow_calibration_required: bool = False
    profile_flow_valid: bool = True
    profile_flow_masked: bool = False
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
    beverage_flow_profile: np.ndarray | None = None
    temperature_profile: np.ndarray | None = None
    target_temperature_profile: np.ndarray | None = None
    pump_target_mode_profile: np.ndarray | None = None
    fixed_cadence_sequence: FixedCadenceShotSequence | None = None
    shot_end_state: str | None = None
    grinder_calibration_mode: GrinderCalibrationMode = GrinderCalibrationMode.RELATIVE_CALIBRATED
    grinder_step_direction: GrinderStepDirection = GrinderStepDirection.HIGHER_IS_FINER
    grinder_adjustment_mode: GrinderAdjustmentMode = GrinderAdjustmentMode.STEPPED
    grinder_reference_label: str = "reference"
    current_absolute_step: float | None = None
    absolute_reference_step: float | None = None
    created_at: int = field(default_factory=now_ts)
    updated_at: int = field(default_factory=now_ts)

    basket_size_ml: float = 18.0
    user_id: str = ""

    def __post_init__(self) -> None:
        self.profile = np.asarray(self.profile, dtype=PROFILE_DTYPE)
        if self.profile.shape != PROFILE_SHAPE:
            raise ValueError(f"profile must have shape {PROFILE_SHAPE}")
        self.beverage_flow_profile = _optional_profile_vector(
            self.beverage_flow_profile,
            "beverage_flow_profile",
        )
        self.temperature_profile = _optional_profile_vector(self.temperature_profile, "temperature_profile")
        self.target_temperature_profile = _optional_profile_vector(
            self.target_temperature_profile,
            "target_temperature_profile",
        )
        self.pump_target_mode_profile = _optional_pump_target_mode_vector(self.pump_target_mode_profile)
        if self.fixed_cadence_sequence is not None and not isinstance(
            self.fixed_cadence_sequence,
            FixedCadenceShotSequence,
        ):
            self.fixed_cadence_sequence = FixedCadenceShotSequence.from_dict(self.fixed_cadence_sequence)
        self.shot_type = ShotType(self.shot_type)
        self.recommendation_decision = RecommendationDecision(self.recommendation_decision)
        self.recommendation_followed = FollowThroughState(self.recommendation_followed)
        self.grinder_calibration_mode = GrinderCalibrationMode(self.grinder_calibration_mode)
        self.grinder_step_direction = GrinderStepDirection(self.grinder_step_direction)
        self.grinder_adjustment_mode = GrinderAdjustmentMode(self.grinder_adjustment_mode)
        if not isinstance(self.taste_goal, TasteGoal):
            self.taste_goal = TasteGoal.from_dict(self.taste_goal)
        self.optimization_weight = float(self.optimization_weight)
        if not 0.0 <= self.optimization_weight <= 1.0:
            raise ValueError("optimization_weight must be between 0 and 1")
        for field_name in ("grind_observed", "dose_observed", "target_yield_observed"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        if self.microns_per_step <= 0:
            raise ValueError("microns_per_step must be positive")
        if self.dose_in_g <= 0:
            raise ValueError("dose_in_g must be positive")
        if self.target_yield_g <= 0:
            raise ValueError("target_yield_g must be positive")
        if self.relative_grind_um_from_reference is None and self.relative_grind_steps_from_reference is not None:
            self.relative_grind_um_from_reference = (
                self.relative_grind_steps_from_reference * self.microns_per_step * self.grinder_direction_sign
            )
        if self.relative_grind_steps_from_reference is None and self.relative_grind_um_from_reference is not None:
            self.relative_grind_steps_from_reference = (
                self.relative_grind_um_from_reference / self.microns_per_step / self.grinder_direction_sign
            )
        if self.relative_grind_steps_from_reference is None:
            self.grind_observed = False
        if self.beverage_out_g is not None and self.brew_ratio is None and self.dose_observed:
            self.brew_ratio = self.beverage_out_g / self.dose_in_g
        if self.target_ratio is None:
            self.target_ratio = self.target_yield_g / self.dose_in_g
        if self.shot_type != ShotType.ESPRESSO or self.exclude_from_local_optimization:
            self.optimization_weight = 0.0
            self.recommendation_attribution_weight = 0.0
        for field_name in (
            "grind_recommendation_trust",
            "dose_recommendation_trust",
            "yield_recommendation_trust",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
            setattr(self, field_name, value)

    @staticmethod
    def new_id() -> str:
        return new_id("shot")

    @property
    def shot_profile(self) -> np.ndarray:
        return self.profile

    @shot_profile.setter
    def shot_profile(self, value: np.ndarray) -> None:
        self.profile = np.asarray(value, dtype=PROFILE_DTYPE)

    @property
    def dose_g(self) -> float:
        return self.dose_in_g

    @dose_g.setter
    def dose_g(self, value: float) -> None:
        self.dose_in_g = value

    @property
    def action_grind_delta_um_from_current(self) -> float:
        return self.recommended_grind_delta_um_from_current or 0.0

    @action_grind_delta_um_from_current.setter
    def action_grind_delta_um_from_current(self, value: float) -> None:
        self.recommended_grind_delta_um_from_current = value

    @property
    def action_dose_g(self) -> float:
        return self.recommended_dose_g or self.dose_in_g

    @action_dose_g.setter
    def action_dose_g(self, value: float) -> None:
        self.recommended_dose_g = value

    def to_recipe(self) -> Recipe:
        if self.relative_grind_steps_from_reference is None:
            raise ValueError("shot has no relative_grind_steps_from_reference")
        return Recipe(
            relative_grind_steps_from_reference=self.relative_grind_steps_from_reference,
            microns_per_step=self.microns_per_step,
            dose_g=self.dose_in_g,
            target_yield_g=self.target_yield_g,
            target_ratio=self.target_ratio,
            grinder_step_direction=self.grinder_step_direction,
            grinder_adjustment_mode=self.grinder_adjustment_mode,
        )

    @property
    def realized_yield_g(self) -> float:
        if self.beverage_out_g is not None and self.beverage_out_g > 0:
            return self.beverage_out_g
        return self.target_yield_g

    @property
    def realized_yield_observed(self) -> bool:
        return (
            self.beverage_out_g is not None and self.beverage_out_g > 0
        ) or self.target_yield_observed

    @property
    def action_observed(self) -> dict[str, bool]:
        return {
            "grind": self.grind_observed,
            "dose": self.dose_observed,
            "target_yield": self.target_yield_observed,
        }

    @property
    def grinder_direction_sign(self) -> int:
        return 1 if self.grinder_step_direction == GrinderStepDirection.HIGHER_IS_FINER else -1

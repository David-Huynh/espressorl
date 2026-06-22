from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

PROFILE_SHAPE = (5, 100)
PROFILE_DTYPE = np.float32


class RecommendationMode(str, Enum):
    ZERO_OBSERVE = "zero_observe"
    ZERO_IMMEDIATE_BO = "zero_immediate_bo"
    WARM_STARTED_BO = "warm_started_bo"
    LOCAL_BO = "local_bo"
    DREAMER_CANDIDATE = "dreamer_candidate"
    DREAMER_ACTIVE = "dreamer_active"
    BO_FALLBACK = "bo_fallback"


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


VALID_TASTE_TAGS = {
    "sour",
    "bitter",
    "weak",
    "harsh",
    "thin",
    "channeling_suspected",
    "balanced",
    "astringent",
    "too_fast",
    "too_slow",
    "muddy",
    "dry",
    "sweet",
    "good_body",
}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def now_ts() -> int:
    return int(time.time())


@dataclass(frozen=True)
class Recipe:
    grind_steps: float
    grinder_step_size_um: float
    dose_g: float
    target_yield_g: float
    target_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.grinder_step_size_um <= 0:
            raise ValueError("grinder_step_size_um must be positive")
        if self.dose_g <= 0:
            raise ValueError("dose_g must be positive")
        if self.target_yield_g <= 0:
            raise ValueError("target_yield_g must be positive")
        if self.target_ratio is None:
            object.__setattr__(self, "target_ratio", self.target_yield_g / self.dose_g)

    @property
    def grind_um(self) -> float:
        return self.grind_steps * self.grinder_step_size_um


@dataclass(frozen=True)
class SafetyBounds:
    dose_min_g: float = 14.0
    dose_max_g: float = 22.0
    target_yield_min_g: float = 20.0
    target_yield_max_g: float = 60.0
    target_ratio_min: float = 1.2
    target_ratio_max: float = 3.5
    max_grind_delta_steps: int = 5
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
    grind_delta_steps: int
    grind_delta_um: float
    next_grind_steps: float
    next_grind_um: float
    next_dose_g: float
    target_yield_g: float
    target_ratio: float
    mode: RecommendationMode
    confidence: float
    reason: str
    status: RecommendationStatus = RecommendationStatus.PENDING
    shown_count: int = 0
    accepted_at: int | None = None
    ignored_at: int | None = None
    edited_at: int | None = None
    used_at: int | None = None
    superseded_at: int | None = None
    source_shot_id: str | None = None
    apply_status: RecommendationApplyStatus = RecommendationApplyStatus.UNKNOWN
    apply_acknowledged_at: int | None = None
    applied_fields: dict[str, Any] = field(default_factory=dict)
    manual_fields: list[str] = field(default_factory=list)
    apply_error: str | None = None

    def __post_init__(self) -> None:
        self.mode = RecommendationMode(self.mode)
        self.status = RecommendationStatus(self.status)
        self.apply_status = RecommendationApplyStatus(self.apply_status)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        if self.next_dose_g <= 0:
            raise ValueError("next_dose_g must be positive")
        if self.target_yield_g <= 0:
            raise ValueError("target_yield_g must be positive")
        if self.target_ratio <= 0:
            raise ValueError("target_ratio must be positive")

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
class RewardResult:
    reward: float
    confidence: float


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


@dataclass
class ShotRecord:
    shot_id: str
    timestamp: int
    install_id: str
    machine_id: str
    machine_adapter: str
    profile: np.ndarray
    grinder_step_size_um: float
    dose_in_g: float
    target_yield_g: float
    grind_steps: float | None = None
    grind_um: float | None = None
    beverage_out_g: float | None = None
    brew_ratio: float | None = None
    target_ratio: float | None = None
    shot_time_s: float | None = None
    bean_context_id: str | None = None
    recommendation_id: str | None = None
    raw_profile_available: bool = True
    raw_profile_hash: str | None = None
    recommended_grind_delta_steps: int | None = None
    recommended_grind_delta_um: float | None = None
    recommended_next_grind_steps: float | None = None
    recommended_dose_g: float | None = None
    recommended_target_yield_g: float | None = None
    recommended_target_ratio: float | None = None
    recommendation_decision: RecommendationDecision = RecommendationDecision.UNKNOWN
    recommendation_followed: FollowThroughState = FollowThroughState.UNKNOWN
    recommendation_attribution_weight: float = 0.0
    human_rating: int | None = None
    taste_tags: list[str] = field(default_factory=list)
    feedback_recorded: bool = False
    profile_score: float | None = None
    profile_mse: float | None = None
    reward: float | None = None
    reward_confidence: float = 0.0
    shot_type: ShotType = ShotType.ESPRESSO
    exclude_from_local_optimization: bool = False
    optimization_weight: float = 1.0
    rating_prompt_allowed: bool = True
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
    shot_end_state: str | None = None
    created_at: int = field(default_factory=now_ts)
    updated_at: int = field(default_factory=now_ts)

    # Compatibility fields for the existing Dreamer modules. They are not used
    # by the active application service, but keeping them here avoids making the
    # future Dreamer cleanup part of this boundary refactor.
    machine_pressure_bar: float = 9.0
    basket_size_ml: float = 18.0
    bean_roast_level: str | None = None
    bean_days_off_roast: int | None = None
    bean_origin: str | None = None
    bean_process: str | None = None
    user_id: str = ""
    grinder_model: str = ""

    def __post_init__(self) -> None:
        self.profile = np.asarray(self.profile, dtype=PROFILE_DTYPE)
        if self.profile.shape != PROFILE_SHAPE:
            raise ValueError(f"profile must have shape {PROFILE_SHAPE}")
        self.shot_type = ShotType(self.shot_type)
        self.recommendation_decision = RecommendationDecision(self.recommendation_decision)
        self.recommendation_followed = FollowThroughState(self.recommendation_followed)
        self.optimization_weight = float(self.optimization_weight)
        if not 0.0 <= self.optimization_weight <= 1.0:
            raise ValueError("optimization_weight must be between 0 and 1")
        if self.grinder_step_size_um <= 0:
            raise ValueError("grinder_step_size_um must be positive")
        if self.dose_in_g <= 0:
            raise ValueError("dose_in_g must be positive")
        if self.target_yield_g <= 0:
            raise ValueError("target_yield_g must be positive")
        if self.grind_um is None and self.grind_steps is not None:
            self.grind_um = self.grind_steps * self.grinder_step_size_um
        if self.grind_steps is None and self.grind_um is not None:
            self.grind_steps = self.grind_um / self.grinder_step_size_um
        if self.beverage_out_g is not None and self.brew_ratio is None:
            self.brew_ratio = self.beverage_out_g / self.dose_in_g
        if self.target_ratio is None:
            self.target_ratio = self.target_yield_g / self.dose_in_g
        invalid_tags = set(self.taste_tags) - VALID_TASTE_TAGS
        if invalid_tags:
            raise ValueError(f"invalid taste tags: {sorted(invalid_tags)}")
        if self.human_rating is not None and not 1 <= self.human_rating <= 5:
            raise ValueError("human_rating must be 1..5")
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
    def step_size_um(self) -> float:
        return self.grinder_step_size_um

    @step_size_um.setter
    def step_size_um(self, value: float) -> None:
        self.grinder_step_size_um = value

    @property
    def action_grind_delta_um(self) -> float:
        return self.recommended_grind_delta_um or 0.0

    @action_grind_delta_um.setter
    def action_grind_delta_um(self, value: float) -> None:
        self.recommended_grind_delta_um = value

    @property
    def action_dose_g(self) -> float:
        return self.recommended_dose_g or self.dose_in_g

    @action_dose_g.setter
    def action_dose_g(self, value: float) -> None:
        self.recommended_dose_g = value

    def to_recipe(self) -> Recipe:
        if self.grind_steps is None:
            raise ValueError("shot has no grind_steps")
        return Recipe(
            grind_steps=self.grind_steps,
            grinder_step_size_um=self.grinder_step_size_um,
            dose_g=self.dose_in_g,
            target_yield_g=self.target_yield_g,
            target_ratio=self.target_ratio,
        )

    def as_optimizer_payload(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "reward": self.reward,
            "reward_confidence": self.reward_confidence,
            "human_rating": self.human_rating,
            "taste_tags": list(self.taste_tags),
            "follow_through": self.recommendation_followed.value,
        }

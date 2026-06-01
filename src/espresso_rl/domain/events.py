from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import (
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


def _numbers(values: list[Any], field_name: str) -> list[float]:
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only numbers") from exc


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
    grinder_step_size_um: float
    dose_in_g: float
    target_yield_g: float
    schema_version: int = 1
    grind_steps: float | None = None
    beverage_out_g: float | None = None
    shot_time_s: float | None = None
    bean_context_id: str | None = None
    recommendation_id: str | None = None
    shot_type: ShotType = ShotType.ESPRESSO
    utility: bool = False
    exclude_from_local_optimization: bool = False
    local_optimization_enabled: bool = True
    optimization_weight: float | None = None
    rating_prompt_allowed: bool = True

    event_type: str = field(default="shot_profile", init=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported shot profile schema_version")
        object.__setattr__(self, "shot_type", ShotType(self.shot_type))
        object.__setattr__(self, "time_ms", _numbers(self.time_ms, "time_ms"))
        for name in ("pressure", "target_pressure", "flow", "target_flow", "weight"):
            object.__setattr__(self, name, _numbers(getattr(self, name), name))
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
        if self.grinder_step_size_um <= 0:
            raise ValueError("grinder_step_size_um must be positive")
        if self.dose_in_g <= 0:
            raise ValueError("dose_in_g must be positive")
        if self.target_yield_g <= 0:
            raise ValueError("target_yield_g must be positive")
        if self.beverage_out_g is not None and self.beverage_out_g <= 0:
            raise ValueError("beverage_out_g must be positive when present")
        if self.shot_time_s is not None and self.shot_time_s <= 0:
            raise ValueError("shot_time_s must be positive when present")
        if self.optimization_weight is not None and not 0.0 <= float(self.optimization_weight) <= 1.0:
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
    source: str = "unknown"
    schema_version: int = 1

    event_type: str = field(default="upload_queue_maintenance", init=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported upload maintenance schema_version")
        if self.action != "requeue_valid_rejected":
            raise ValueError("unsupported upload maintenance action")
        if not 1 <= int(self.limit) <= 500:
            raise ValueError("upload maintenance limit must be between 1 and 500")
        object.__setattr__(self, "limit", int(self.limit))


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
            "next_grind_steps",
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
            "next_grind_steps",
            "next_grind_um",
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
    grind_steps: float | None = None
    grinder_step_size_um: float | None = None
    dose_in_g: float | None = None
    target_yield_g: float | None = None
    source: str = "unknown"

    event_type: str = field(default="machine_state", init=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported machine state schema_version")
        object.__setattr__(self, "state", MachineState(self.state))
        if self.grinder_step_size_um is not None and self.grinder_step_size_um <= 0:
            raise ValueError("grinder_step_size_um must be positive when present")
        if self.dose_in_g is not None and self.dose_in_g <= 0:
            raise ValueError("dose_in_g must be positive when present")
        if self.target_yield_g is not None and self.target_yield_g <= 0:
            raise ValueError("target_yield_g must be positive when present")

    def current_recipe(self) -> Recipe | None:
        if (
            self.grind_steps is None
            or self.grinder_step_size_um is None
            or self.dose_in_g is None
            or self.target_yield_g is None
        ):
            return None
        return Recipe(
            grind_steps=self.grind_steps,
            grinder_step_size_um=self.grinder_step_size_um,
            dose_g=self.dose_in_g,
            target_yield_g=self.target_yield_g,
        )

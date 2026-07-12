from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping

from espresso_rl.domain.community import (
    COMMUNITY_COMPARISON_LABELS,
    COMMUNITY_COMPARISON_MODES,
)


OFFLINE_DATASET_FORMAT = "espresso_rl_offline_preference_dataset_v1"
OFFLINE_EXAMPLE_FORMAT = "espresso_rl_offline_preference_example_v1"

_CONTEXT_FIELDS = (
    "install_id",
    "machine_id",
    "machine_adapter",
    "bean_context_id",
    "grinder_context_id",
    "profile_id",
    "profile_label",
    "profile_type",
)
_RECIPE_FIELDS = (
    "action_observed",
    "grinder_calibration_mode",
    "grinder_adjustment_mode",
    "microns_per_step",
    "step_direction",
    "reference_label",
    "relative_grind_steps_from_reference",
    "relative_grind_um_from_reference",
    "current_absolute_step",
    "absolute_reference_step",
    "dose_in_g",
    "target_yield_g",
    "target_ratio",
)
_REALIZED_FIELDS = (
    "beverage_out_g",
    "brew_ratio",
    "shot_time_s",
    "shot_end_state",
    "final_phase_index",
    "final_phase_name",
    "final_phase_type",
    "final_phase_elapsed_s",
    "final_pump_target",
    "final_target_pressure",
    "final_target_flow",
    "final_valve_open",
    "profile_temperature_c",
    "final_phase_temperature_c",
)
_TRAJECTORY_FIELDS = (
    "profile_resampled",
    "beverage_flow_profile",
    "temperature_profile",
    "target_temperature_profile",
    "pump_target_mode_profile",
    "fixed_cadence_sequence",
)
_QUALITY_FIELDS = (
    "raw_profile_available",
    "raw_profile_hash",
    "weight_source",
    "flow_source",
    "flow_units",
    "pump_flow_source",
    "pump_flow_units",
    "pump_flow_calibration_required",
    "profile_flow_valid",
    "profile_flow_masked",
    "exclude_from_local_optimization",
)
_FORBIDDEN_SCALAR_FEEDBACK_FIELDS = frozenset(
    {
        "human_rating",
        "rating",
        "taste_tags",
        "reward",
        "reward_confidence",
        "profile_score",
        "profile_mse",
    }
)


@dataclass(frozen=True)
class OfflinePreferenceExample:
    """One immutable pairwise label joined to two physical shot trajectories."""

    comparison_id: str
    optimization_run_id: str
    label: str
    comparison_mode: str
    created_at: int
    new_shot: dict[str, Any]
    anchor_shot: dict[str, Any]
    comparison_trust_weight: float
    new_shot_trust_weight: float
    anchor_shot_trust_weight: float
    recommendation_id: str | None = None

    def __post_init__(self) -> None:
        _required_id(self.comparison_id, "comparison_id")
        _required_id(self.optimization_run_id, "optimization_run_id")
        if self.label not in COMMUNITY_COMPARISON_LABELS:
            raise ValueError("offline comparison label is invalid")
        if self.comparison_mode not in COMMUNITY_COMPARISON_MODES:
            raise ValueError("offline comparison mode is invalid")
        if isinstance(self.created_at, bool) or not isinstance(self.created_at, int) or self.created_at < 0:
            raise ValueError("offline comparison created_at must be a nonnegative integer")
        for field_name in (
            "comparison_trust_weight",
            "new_shot_trust_weight",
            "anchor_shot_trust_weight",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and between 0 and 1")

        new_shot = _validated_shot(copy.deepcopy(self.new_shot), "new_shot")
        anchor_shot = _validated_shot(copy.deepcopy(self.anchor_shot), "anchor_shot")
        if new_shot["shot_id"] == anchor_shot["shot_id"]:
            raise ValueError("offline comparison requires two distinct physical shots")
        if new_shot["timestamp"] > self.created_at or anchor_shot["timestamp"] > self.created_at:
            raise ValueError("offline comparison cannot precede either physical shot")
        _require_matching_context(new_shot, anchor_shot)
        object.__setattr__(self, "new_shot", new_shot)
        object.__setattr__(self, "anchor_shot", anchor_shot)

    @property
    def example_weight(self) -> float:
        return min(
            self.comparison_trust_weight,
            self.new_shot_trust_weight,
            self.anchor_shot_trust_weight,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": OFFLINE_EXAMPLE_FORMAT,
            "comparison": {
                "comparison_id": self.comparison_id,
                "optimization_run_id": self.optimization_run_id,
                "new_shot_id": self.new_shot["shot_id"],
                "anchor_shot_id": self.anchor_shot["shot_id"],
                "label": self.label,
                "comparison_mode": self.comparison_mode,
                "created_at": self.created_at,
                "recommendation_id": self.recommendation_id,
            },
            "new_shot": _export_shot(self.new_shot),
            "anchor_shot": _export_shot(self.anchor_shot),
            "trust": {
                "comparison": self.comparison_trust_weight,
                "new_shot": self.new_shot_trust_weight,
                "anchor_shot": self.anchor_shot_trust_weight,
                "example": self.example_weight,
            },
        }

    @classmethod
    def from_joined_payloads(
        cls,
        *,
        comparison_payload: Mapping[str, Any],
        new_shot_payload: Mapping[str, Any],
        anchor_shot_payload: Mapping[str, Any],
        comparison_trust_weight: float,
        new_shot_trust_weight: float,
        anchor_shot_trust_weight: float,
    ) -> "OfflinePreferenceExample":
        comparison = dict(comparison_payload)
        if comparison.get("event_type") != "comparison_record":
            raise ValueError("offline comparison payload must be a comparison_record")
        new_shot = dict(new_shot_payload)
        anchor_shot = dict(anchor_shot_payload)
        if comparison.get("new_shot_id") != new_shot.get("shot_id"):
            raise ValueError("offline new_shot_id does not match the joined physical shot")
        if comparison.get("anchor_shot_id") != anchor_shot.get("shot_id"):
            raise ValueError("offline anchor_shot_id does not match the joined physical shot")
        for field_name in ("install_id", "machine_id", "bean_context_id", "grinder_context_id"):
            expected = comparison.get(field_name)
            if expected != new_shot.get(field_name) or expected != anchor_shot.get(field_name):
                raise ValueError(f"offline comparison {field_name} does not match both physical shots")
        _require_profile_scope(comparison, new_shot, anchor_shot)
        return cls(
            comparison_id=str(comparison.get("comparison_id", "")),
            optimization_run_id=str(comparison.get("optimization_run_id", "")),
            label=str(comparison.get("label", "")),
            comparison_mode=str(comparison.get("comparison_mode", "")),
            created_at=comparison.get("created_at"),
            new_shot=new_shot,
            anchor_shot=anchor_shot,
            comparison_trust_weight=comparison_trust_weight,
            new_shot_trust_weight=new_shot_trust_weight,
            anchor_shot_trust_weight=anchor_shot_trust_weight,
            recommendation_id=_optional_id(comparison.get("recommendation_id")),
        )


def _validated_shot(shot: dict[str, Any], field_name: str) -> dict[str, Any]:
    if shot.get("event_type") != "shot_record":
        raise ValueError(f"offline {field_name} must be a shot_record")
    _required_id(shot.get("shot_id"), f"{field_name}.shot_id")
    _required_id(shot.get("install_id"), f"{field_name}.install_id")
    _required_id(shot.get("machine_id"), f"{field_name}.machine_id")
    timestamp = shot.get("timestamp")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise ValueError(f"{field_name}.timestamp must be a nonnegative integer")
    forbidden = _FORBIDDEN_SCALAR_FEEDBACK_FIELDS.intersection(shot)
    if forbidden:
        raise ValueError(
            f"offline {field_name} contains scalar feedback fields: {', '.join(sorted(forbidden))}"
        )
    _assert_plain_json(shot, field_name)
    return shot


def _require_matching_context(new_shot: Mapping[str, Any], anchor_shot: Mapping[str, Any]) -> None:
    for field_name in ("install_id", "machine_id", "bean_context_id", "grinder_context_id"):
        if new_shot.get(field_name) != anchor_shot.get(field_name):
            raise ValueError(f"offline physical shots have mixed {field_name} contexts")
    _require_profile_scope({}, new_shot, anchor_shot)


def _require_profile_scope(
    comparison: Mapping[str, Any],
    new_shot: Mapping[str, Any],
    anchor_shot: Mapping[str, Any],
) -> None:
    comparison_profile = comparison.get("profile_id")
    new_profile = new_shot.get("profile_id")
    anchor_profile = anchor_shot.get("profile_id")
    if comparison_profile is not None:
        if comparison_profile != new_profile or comparison_profile != anchor_profile:
            raise ValueError("offline comparison profile_id does not match both physical shots")
        return
    if new_profile is not None or anchor_profile is not None:
        if new_profile != anchor_profile:
            raise ValueError("offline physical shots have mixed profile contexts")
        return
    comparison_hash = comparison.get("raw_profile_hash")
    new_hash = new_shot.get("raw_profile_hash")
    anchor_hash = anchor_shot.get("raw_profile_hash")
    if comparison_hash is not None and (comparison_hash != new_hash or comparison_hash != anchor_hash):
        raise ValueError("offline comparison profile hash does not match both physical shots")
    if new_hash != anchor_hash:
        raise ValueError("offline physical shots have mixed profile-hash contexts")


def _export_shot(shot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "shot_id": shot["shot_id"],
        "timestamp": shot["timestamp"],
        "context": _selected(shot, _CONTEXT_FIELDS),
        "recipe": _selected(shot, _RECIPE_FIELDS),
        "realized_outcome": _selected(shot, _REALIZED_FIELDS),
        "trajectory": _selected(shot, _TRAJECTORY_FIELDS),
        "quality": _selected(shot, _QUALITY_FIELDS),
    }


def _selected(source: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field_name: copy.deepcopy(source.get(field_name)) for field_name in fields}


def _assert_plain_json(value: Any, field_name: str, depth: int = 0) -> None:
    if depth > 12:
        raise ValueError(f"{field_name} exceeds the maximum JSON nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _assert_plain_json(item, field_name, depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 120:
                raise ValueError(f"{field_name} contains an invalid JSON key")
            _assert_plain_json(item, field_name, depth + 1)
        return
    raise ValueError(f"{field_name} contains a non-JSON value")


def _required_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field_name} is invalid")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} is invalid")
    return value


def _optional_id(value: Any) -> str | None:
    if value is None:
        return None
    return _required_id(value, "recommendation_id")

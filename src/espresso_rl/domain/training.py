from __future__ import annotations

import math
from typing import Any

from espresso_rl.domain.models import VALID_TASTE_TAGS

TRAINING_DATASET_FORMAT = "espresso_rl_training_dataset_v1"
TRAINING_TRANSITION_FORMAT = "espresso_rl_training_transition_v1"
TRAINING_SCHEMA_VERSION = 1
TRAINING_SOURCE_KINDS = frozenset({"community_validated_shot", "local_validated_shot"})

FORBIDDEN_TRAINING_FIELD_NAMES = frozenset(
    {
        "current_absolute_step",
        "absolute_reference_step",
        "projected_absolute_step",
        "absolute_grind_size_um",
        "total_step_count",
        "min_step",
        "max_step",
    }
)

_ROOT_FIELDS = frozenset(
    {
        "format",
        "schema_version",
        "training_row_id",
        "source",
        "context",
        "action",
        "recommendation",
        "observation",
        "reward",
    }
)
_SOURCE_FIELDS = frozenset({"source_kind", "source_validation_id", "install_id", "payload_hash", "trust_weight"})
_CONTEXT_FIELDS = frozenset(
    {
        "machine_id",
        "machine_adapter",
        "bean_context_id",
        "grinder_context_id",
        "microns_per_step",
        "step_direction",
    }
)
_ACTION_FIELDS = frozenset(
    {
        "relative_grind_steps_from_reference",
        "relative_grind_um_from_reference",
        "dose_g",
        "target_yield_g",
        "target_ratio",
    }
)
_RECOMMENDATION_FIELDS = frozenset(
    {
        "recommendation_id",
        "grind_delta_steps_from_current",
        "grind_delta_um_from_current",
        "projected_relative_step_from_reference",
        "projected_relative_grind_um_from_reference",
        "next_dose_g",
        "target_yield_g",
        "target_ratio",
        "decision",
        "follow_through",
        "attribution_weight",
        "field_trust",
    }
)
_FIELD_TRUST_FIELDS = frozenset({"grind", "dose", "yield"})
_OBSERVATION_FIELDS = frozenset(
    {
        "shot_id",
        "timestamp",
        "beverage_out_g",
        "brew_ratio",
        "shot_time_s",
        "profile_resampled",
        "raw_profile_available",
        "raw_profile_hash",
        "profile_score",
        "profile_mse",
        "profile_flow_valid",
        "profile_flow_masked",
        "profile_id",
        "profile_type",
        "profile_phase_count",
        "final_pump_target",
        "final_target_pressure",
        "final_target_flow",
        "profile_temperature_c",
        "final_phase_temperature_c",
        "beverage_flow_profile",
        "temperature_profile",
        "target_temperature_profile",
        "pump_target_mode_profile",
        "fixed_cadence_sequence",
        "shot_end_state",
    }
)
_REWARD_FIELDS = frozenset(
    {
        "human_rating",
        "taste_tags",
        "reward",
        "confidence",
        "feedback_recorded",
        "optimization_weight",
    }
)


def validate_training_transition(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(row, dict):
        return ["training transition must be an object"]

    _reject_unknown_fields(row, _ROOT_FIELDS, errors, path="transition")
    _reject_forbidden_training_fields(row, errors)
    if row.get("format") != TRAINING_TRANSITION_FORMAT:
        errors.append("training transition format is unsupported")
    if row.get("schema_version") != TRAINING_SCHEMA_VERSION:
        errors.append("training transition schema_version is unsupported")
    _require_positive_int(row.get("training_row_id"), "training_row_id", errors)

    source = _require_object(row, "source", errors)
    if source is not None:
        _reject_unknown_fields(source, _SOURCE_FIELDS, errors, path="source")
        if source.get("source_kind") not in TRAINING_SOURCE_KINDS:
            errors.append("source.source_kind is invalid")
        _require_positive_int(source.get("source_validation_id"), "source.source_validation_id", errors)
        _require_nonempty_string(source.get("install_id"), "source.install_id", errors)
        _require_number_range(source.get("trust_weight"), "source.trust_weight", 0.0, 1.0, errors)
        payload_hash = source.get("payload_hash")
        if payload_hash is not None and not _is_sha256(payload_hash):
            errors.append("source.payload_hash must be a sha256 hex digest")

    context = _require_object(row, "context", errors)
    if context is not None:
        _reject_unknown_fields(context, _CONTEXT_FIELDS, errors, path="context")
        _require_nonempty_string(context.get("machine_id"), "context.machine_id", errors)
        adapter = context.get("machine_adapter")
        if adapter is not None:
            _require_nonempty_string(adapter, "context.machine_adapter", errors)
        _optional_string(context.get("bean_context_id"), "context.bean_context_id", errors)
        _optional_string(context.get("grinder_context_id"), "context.grinder_context_id", errors)
        _require_number_range(context.get("microns_per_step"), "context.microns_per_step", 0.1, 100.0, errors)
        if context.get("step_direction") not in {"higher_is_finer", "higher_is_coarser"}:
            errors.append("context.step_direction is invalid")

    action = _require_object(row, "action", errors)
    if action is not None:
        _reject_unknown_fields(action, _ACTION_FIELDS, errors, path="action")
        _require_finite_number(action.get("relative_grind_steps_from_reference"), "action.relative_grind_steps_from_reference", errors)
        _require_finite_number(action.get("relative_grind_um_from_reference"), "action.relative_grind_um_from_reference", errors)
        _require_number_range(action.get("dose_g"), "action.dose_g", 5.0, 30.0, errors)
        _require_number_range(action.get("target_yield_g"), "action.target_yield_g", 5.0, 100.0, errors)
        _require_number_range(action.get("target_ratio"), "action.target_ratio", 1.2, 3.5, errors)

    recommendation = row.get("recommendation")
    if recommendation is not None:
        if not isinstance(recommendation, dict):
            errors.append("recommendation must be an object or null")
        else:
            _reject_unknown_fields(recommendation, _RECOMMENDATION_FIELDS, errors, path="recommendation")
            _optional_string(recommendation.get("recommendation_id"), "recommendation.recommendation_id", errors)
            _optional_int(recommendation.get("grind_delta_steps_from_current"), "recommendation.grind_delta_steps_from_current", errors)
            _optional_finite_number(recommendation.get("grind_delta_um_from_current"), "recommendation.grind_delta_um_from_current", errors)
            _optional_finite_number(
                recommendation.get("projected_relative_step_from_reference"),
                "recommendation.projected_relative_step_from_reference",
                errors,
            )
            _optional_finite_number(
                recommendation.get("projected_relative_grind_um_from_reference"),
                "recommendation.projected_relative_grind_um_from_reference",
                errors,
            )
            _optional_number_range(recommendation.get("next_dose_g"), "recommendation.next_dose_g", 5.0, 30.0, errors)
            _optional_number_range(
                recommendation.get("target_yield_g"),
                "recommendation.target_yield_g",
                5.0,
                100.0,
                errors,
            )
            _optional_number_range(recommendation.get("target_ratio"), "recommendation.target_ratio", 1.2, 3.5, errors)
            _optional_number_range(
                recommendation.get("attribution_weight"),
                "recommendation.attribution_weight",
                0.0,
                1.0,
                errors,
            )
            _optional_enum(
                recommendation.get("decision"),
                "recommendation.decision",
                {"accepted", "edited", "ignored", "dismissed", "unknown"},
                errors,
            )
            _optional_enum(
                recommendation.get("follow_through"),
                "recommendation.follow_through",
                {"followed", "partially_followed", "not_followed", "unknown"},
                errors,
            )
            field_trust = recommendation.get("field_trust")
            if field_trust is not None:
                if not isinstance(field_trust, dict):
                    errors.append("recommendation.field_trust must be an object")
                else:
                    _reject_unknown_fields(field_trust, _FIELD_TRUST_FIELDS, errors, path="recommendation.field_trust")
                    _optional_number_range(field_trust.get("grind"), "recommendation.field_trust.grind", 0.0, 1.0, errors)
                    _optional_number_range(field_trust.get("dose"), "recommendation.field_trust.dose", 0.0, 1.0, errors)
                    _optional_number_range(field_trust.get("yield"), "recommendation.field_trust.yield", 0.0, 1.0, errors)

    observation = _require_object(row, "observation", errors)
    if observation is not None:
        _reject_unknown_fields(observation, _OBSERVATION_FIELDS, errors, path="observation")
        _require_nonempty_string(observation.get("shot_id"), "observation.shot_id", errors)
        _require_number_range(observation.get("timestamp"), "observation.timestamp", 0.0, 9_007_199_254_740_991.0, errors)
        _optional_number_range(observation.get("beverage_out_g"), "observation.beverage_out_g", 0.0, 120.0, errors)
        _optional_number_range(observation.get("brew_ratio"), "observation.brew_ratio", 0.1, 10.0, errors)
        _optional_number_range(observation.get("shot_time_s"), "observation.shot_time_s", 0.0, 180.0, errors)
        _optional_bool(observation.get("raw_profile_available"), "observation.raw_profile_available", errors)
        raw_profile_hash = observation.get("raw_profile_hash")
        if raw_profile_hash is not None and not _is_sha256(raw_profile_hash):
            errors.append("observation.raw_profile_hash must be a sha256 hex digest")
        _optional_number_range(observation.get("profile_score"), "observation.profile_score", 0.0, 1.0, errors)
        _optional_number_range(observation.get("profile_mse"), "observation.profile_mse", 0.0, 1_000_000.0, errors)
        _optional_bool(observation.get("profile_flow_valid"), "observation.profile_flow_valid", errors)
        _optional_bool(observation.get("profile_flow_masked"), "observation.profile_flow_masked", errors)
        _optional_string(observation.get("profile_id"), "observation.profile_id", errors)
        _optional_string(observation.get("profile_type"), "observation.profile_type", errors)
        _optional_int(observation.get("profile_phase_count"), "observation.profile_phase_count", errors)
        _optional_enum(observation.get("final_pump_target"), "observation.final_pump_target", {"simple", "pressure", "flow"}, errors)
        _optional_number_range(observation.get("final_target_pressure"), "observation.final_target_pressure", 0.0, 15.0, errors)
        _optional_number_range(observation.get("final_target_flow"), "observation.final_target_flow", 0.0, 25.0, errors)
        _require_number_range(observation.get("profile_temperature_c"), "observation.profile_temperature_c", 0.0, 160.0, errors)
        _require_number_range(observation.get("final_phase_temperature_c"), "observation.final_phase_temperature_c", 0.0, 160.0, errors)
        _optional_profile_vector(
            observation.get("beverage_flow_profile"),
            "observation.beverage_flow_profile",
            0.0,
            20.0,
            errors,
        )
        _optional_profile_vector(observation.get("temperature_profile"), "observation.temperature_profile", 0.0, 160.0, errors)
        _optional_profile_vector(
            observation.get("target_temperature_profile"),
            "observation.target_temperature_profile",
            0.0,
            160.0,
            errors,
        )
        _optional_pump_target_mode_profile(
            observation.get("pump_target_mode_profile"),
            "observation.pump_target_mode_profile",
            errors,
        )
        _optional_fixed_cadence_sequence(
            observation.get("fixed_cadence_sequence"),
            "observation.fixed_cadence_sequence",
            errors,
        )
        _optional_string(observation.get("shot_end_state"), "observation.shot_end_state", errors)
        profile = observation.get("profile_resampled")
        if profile is not None:
            _validate_profile(profile, errors)

    reward = _require_object(row, "reward", errors)
    if reward is not None:
        _reject_unknown_fields(reward, _REWARD_FIELDS, errors, path="reward")
        _optional_number_range(reward.get("human_rating"), "reward.human_rating", 1.0, 5.0, errors)
        _optional_number_range(reward.get("reward"), "reward.reward", 0.0, 1.0, errors)
        _require_number_range(reward.get("confidence"), "reward.confidence", 0.0, 1.0, errors)
        _require_number_range(reward.get("optimization_weight"), "reward.optimization_weight", 0.0, 1.0, errors)
        _optional_bool(reward.get("feedback_recorded"), "reward.feedback_recorded", errors)
        taste_tags = reward.get("taste_tags")
        if taste_tags is not None:
            if not isinstance(taste_tags, list):
                errors.append("reward.taste_tags must be a list")
            else:
                invalid = [tag for tag in taste_tags if not isinstance(tag, str) or tag not in VALID_TASTE_TAGS]
                if invalid:
                    errors.append("reward.taste_tags contains invalid values")

    return errors


def _reject_unknown_fields(value: dict[str, Any], allowed: frozenset[str], errors: list[str], *, path: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        errors.append(f"{path} contains unknown fields: {', '.join(unknown[:10])}")


def _reject_forbidden_training_fields(value: Any, errors: list[str], *, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}" if path else key_text
            if key_text in FORBIDDEN_TRAINING_FIELD_NAMES:
                errors.append(f"{item_path} is not allowed in training data")
            _reject_forbidden_training_fields(item, errors, path=item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_training_fields(item, errors, path=f"{path}[{index}]")


def _require_object(row: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any] | None:
    value = row.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be an object")
        return None
    return value


def _require_positive_int(value: object, label: str, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{label} must be a positive integer")


def _optional_int(value: object, label: str, errors: list[str]) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{label} must be an integer")


def _require_nonempty_string(value: object, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        errors.append(f"{label} must be a short non-empty string")


def _optional_string(value: object, label: str, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, str) or len(value) > 160:
        errors.append(f"{label} must be a short string")


def _optional_enum(value: object, label: str, allowed: set[str], errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, str) or value not in allowed:
        errors.append(f"{label} is invalid")


def _optional_bool(value: object, label: str, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, bool):
        errors.append(f"{label} must be boolean")


def _require_number_range(value: object, label: str, minimum: float, maximum: float, errors: list[str]) -> None:
    if not _is_finite_number(value) or not minimum <= float(value) <= maximum:
        errors.append(f"{label} out of range")


def _optional_number_range(value: object, label: str, minimum: float, maximum: float, errors: list[str]) -> None:
    if value is None:
        return
    _require_number_range(value, label, minimum, maximum, errors)


def _require_finite_number(value: object, label: str, errors: list[str]) -> None:
    if not _is_finite_number(value):
        errors.append(f"{label} must be finite")


def _optional_finite_number(value: object, label: str, errors: list[str]) -> None:
    if value is None:
        return
    _require_finite_number(value, label, errors)


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value)


def _validate_profile(profile: object, errors: list[str]) -> None:
    if not isinstance(profile, list) or len(profile) != 5:
        errors.append("observation.profile_resampled must have 5 channels")
        return
    for channel in profile:
        if not isinstance(channel, list) or len(channel) != 100:
            errors.append("observation.profile_resampled channels must have 100 samples")
            return
        if not all(_is_finite_number(value) for value in channel):
            errors.append("observation.profile_resampled contains non-finite values")
            return


def _optional_profile_vector(
    values: object,
    label: str,
    minimum: float,
    maximum: float,
    errors: list[str],
) -> None:
    if values is None:
        return
    if not isinstance(values, list) or len(values) != 100:
        errors.append(f"{label} must have 100 samples")
        return
    if not all(_is_finite_number(value) for value in values):
        errors.append(f"{label} contains non-finite values")
        return
    if not all(minimum <= float(value) <= maximum for value in values):
        errors.append(f"{label} out of range")


def _optional_pump_target_mode_profile(values: object, label: str, errors: list[str]) -> None:
    if values is None:
        return
    if not isinstance(values, list) or len(values) != 100:
        errors.append(f"{label} must have 100 samples")
        return
    if not all(isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 2 for value in values):
        errors.append(f"{label} contains invalid pump target mode values")


def _optional_fixed_cadence_sequence(value: object, label: str, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    numeric_ranges = {
        "pressure_bar": (0.0, 15.0),
        "pressure_target_bar": (0.0, 15.0),
        "pump_flow_ml_s": (0.0, 20.0),
        "pump_flow_target_ml_s": (0.0, 20.0),
        "beverage_flow_g_s": (0.0, 20.0),
        "weight_g": (-1.0, 120.0),
        "temperature_c": (0.0, 160.0),
        "temperature_target_c": (0.0, 160.0),
    }
    allowed = {"sample_interval_ms", "pump_target_mode", "valve_open", *numeric_ranges}
    _reject_unknown_fields(value, frozenset(allowed), errors, path=label)
    if value.get("sample_interval_ms") != 250:
        errors.append(f"{label}.sample_interval_ms must be 250")

    lengths: set[int] = set()
    for key, (minimum, maximum) in numeric_ranges.items():
        values = value.get(key)
        if not isinstance(values, list):
            errors.append(f"{label}.{key} must be a list")
            continue
        lengths.add(len(values))
        if not all(_is_finite_number(item) for item in values):
            errors.append(f"{label}.{key} contains non-finite values")
        elif not all(minimum <= float(item) <= maximum for item in values):
            errors.append(f"{label}.{key} out of range")

    pump_modes = value.get("pump_target_mode")
    if not isinstance(pump_modes, list):
        errors.append(f"{label}.pump_target_mode must be a list")
    else:
        lengths.add(len(pump_modes))
        if not all(isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 2 for item in pump_modes):
            errors.append(f"{label}.pump_target_mode contains invalid values")

    valve_open = value.get("valve_open")
    if not isinstance(valve_open, list):
        errors.append(f"{label}.valve_open must be a list")
    else:
        lengths.add(len(valve_open))
        if not all(isinstance(item, bool) for item in valve_open):
            errors.append(f"{label}.valve_open contains invalid values")

    if len(lengths) != 1:
        errors.append(f"{label} channels must have matching lengths")
    elif not 2 <= next(iter(lengths)) <= 500:
        errors.append(f"{label} must contain 2..500 steps")

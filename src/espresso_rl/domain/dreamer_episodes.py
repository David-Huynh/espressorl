from __future__ import annotations

import math
from typing import Any

from espresso_rl.domain.dreamer_control import (
    DREAMER_CONTROL_CONSTRAINT_FIELDS,
    DREAMER_DYNAMIC_ACTION_FIELDS,
    DREAMER_MAX_PRESSURE_TARGET_BAR,
    DREAMER_MIN_TEMPERATURE_TARGET_C,
    DREAMER_MAX_TEMPERATURE_TARGET_C,
    DREAMER_MAX_YIELD_STOP_TARGET_G,
)
from espresso_rl.domain.dreamer_pre_shot import validate_dreamer_pre_shot_action
from espresso_rl.domain.dreamer_taste import validate_dreamer_taste_objective
from espresso_rl.domain.models import (
    FIXED_CADENCE_MAX_STEPS,
    FIXED_CADENCE_SAMPLE_INTERVAL_MS,
    VALID_TASTE_TAGS,
)
from espresso_rl.domain.training import FORBIDDEN_TRAINING_FIELD_NAMES, TRAINING_SOURCE_KINDS

DREAMER_EPISODE_FORMAT = "espresso_rl_dreamer_episode_v4"
DREAMER_EPISODE_SCHEMA_VERSION = 4

DREAMER_PROFILE_CHANNELS = (
    "pressure_bar",
    "pump_flow_ml_s",
    "beverage_flow_g_s",
    "weight_g",
    "temperature_c",
)

_ROOT_FIELDS = frozenset(
    {
        "format",
        "schema_version",
        "episode_id",
        "source_training_row_id",
        "sample_interval_ms",
        "group_key",
        "source",
        "context",
        "static_context",
        "pre_shot_action",
        "steps",
        "terminal",
        "recommendation",
    }
)
_GROUP_KEY_FIELDS = frozenset({"install_id", "machine_id", "bean_context_id", "grinder_context_id"})
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
_STATIC_CONTEXT_FIELDS = frozenset(
    {
        "relative_grind_steps_from_reference",
        "relative_grind_um_from_reference",
        "dose_g",
        "initial_target_yield_g",
        "target_ratio",
        "grind_observed",
        "dose_observed",
        "initial_target_yield_observed",
        "microns_per_step",
        "step_direction",
        "profile_id",
        "profile_type",
        "profile_phase_count",
        "taste_objective",
    }
)
_STEP_FIELDS = frozenset({"step_index", "elapsed_ms", "observation", "observed_profile_target", "dynamic_action", "constraints"})
_STEP_OBSERVATION_FIELDS = frozenset(DREAMER_PROFILE_CHANNELS)
_OBSERVED_PROFILE_TARGET_FIELDS = frozenset(
    {
        "pressure_target_bar",
        "pressure_target_active",
        "flow_target_ml_s",
        "flow_target_active",
        "temperature_target_c",
        "temperature_target_active",
        "pump_target_mode",
        "valve_open",
    }
)
_DYNAMIC_ACTION_FIELDS = frozenset(DREAMER_DYNAMIC_ACTION_FIELDS)
_DYNAMIC_ACTION_FORBIDDEN_FIELDS = frozenset(
    {
        "relative_grind_steps_from_reference",
        "relative_grind_um_from_reference",
        "dose_g",
        "initial_target_yield_g",
        "target_yield_g",
        "target_ratio",
        "grind_delta_steps_from_current",
        "next_dose_g",
    }
)
_CONSTRAINT_FIELDS = frozenset(
    {
        *DREAMER_CONTROL_CONSTRAINT_FIELDS,
    }
)
_TERMINAL_FIELDS = frozenset(
    {
        "shot_id",
        "timestamp",
        "beverage_out_g",
        "brew_ratio",
        "shot_time_s",
        "human_rating",
        "taste_tags",
        "reward",
        "confidence",
        "feedback_recorded",
        "optimization_weight",
        "profile_score",
        "profile_mse",
        "profile_flow_valid",
        "profile_flow_masked",
        "final_pump_target",
        "final_target_pressure",
        "final_target_flow",
        "profile_temperature_c",
        "final_phase_temperature_c",
        "shot_end_state",
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


def validate_dreamer_episode(episode: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(episode, dict):
        return ["Dreamer episode must be an object"]

    _reject_unknown_fields(episode, _ROOT_FIELDS, errors, path="episode")
    _reject_forbidden_fields(episode, errors)
    if episode.get("format") != DREAMER_EPISODE_FORMAT:
        errors.append("Dreamer episode format is unsupported")
    if episode.get("schema_version") != DREAMER_EPISODE_SCHEMA_VERSION:
        errors.append("Dreamer episode schema_version is unsupported")
    _require_nonempty_string(episode.get("episode_id"), "episode.episode_id", errors)
    _require_positive_int(episode.get("source_training_row_id"), "episode.source_training_row_id", errors)
    if episode.get("sample_interval_ms") != FIXED_CADENCE_SAMPLE_INTERVAL_MS:
        errors.append(
            f"episode.sample_interval_ms must be {FIXED_CADENCE_SAMPLE_INTERVAL_MS}"
        )

    group_key = _require_object(episode, "group_key", errors)
    if group_key is not None:
        _validate_group_key(group_key, errors)

    source = _require_object(episode, "source", errors)
    if source is not None:
        _validate_source(source, errors)

    context = _require_object(episode, "context", errors)
    if context is not None:
        _validate_context(context, errors)

    static_context = _require_object(episode, "static_context", errors)
    if static_context is not None:
        _validate_static_context(static_context, errors)

    pre_shot_action = _require_object(episode, "pre_shot_action", errors)
    if pre_shot_action is not None:
        errors.extend(validate_dreamer_pre_shot_action(pre_shot_action))

    steps = episode.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append("episode.steps must be a non-empty list")
    elif len(steps) > FIXED_CADENCE_MAX_STEPS:
        errors.append("episode.steps contains too many steps")
    else:
        for index, step in enumerate(steps):
            _validate_step(step, index, errors)

    terminal = _require_object(episode, "terminal", errors)
    if terminal is not None:
        _validate_terminal(terminal, errors)

    recommendation = episode.get("recommendation")
    if recommendation is not None:
        if not isinstance(recommendation, dict):
            errors.append("episode.recommendation must be an object or null")
        else:
            _validate_recommendation(recommendation, errors)

    return errors


def _validate_group_key(group_key: dict[str, Any], errors: list[str]) -> None:
    _reject_unknown_fields(group_key, _GROUP_KEY_FIELDS, errors, path="episode.group_key")
    _require_nonempty_string(group_key.get("install_id"), "episode.group_key.install_id", errors)
    _require_nonempty_string(group_key.get("machine_id"), "episode.group_key.machine_id", errors)
    _optional_string(group_key.get("bean_context_id"), "episode.group_key.bean_context_id", errors)
    _optional_string(group_key.get("grinder_context_id"), "episode.group_key.grinder_context_id", errors)


def _validate_source(source: dict[str, Any], errors: list[str]) -> None:
    _reject_unknown_fields(source, _SOURCE_FIELDS, errors, path="episode.source")
    if source.get("source_kind") not in TRAINING_SOURCE_KINDS:
        errors.append("episode.source.source_kind is invalid")
    _require_positive_int(source.get("source_validation_id"), "episode.source.source_validation_id", errors)
    _require_nonempty_string(source.get("install_id"), "episode.source.install_id", errors)
    _require_number_range(source.get("trust_weight"), "episode.source.trust_weight", 0.0, 1.0, errors)
    payload_hash = source.get("payload_hash")
    if payload_hash is not None and not _is_sha256(payload_hash):
        errors.append("episode.source.payload_hash must be a sha256 hex digest")


def _validate_context(context: dict[str, Any], errors: list[str]) -> None:
    _reject_unknown_fields(context, _CONTEXT_FIELDS, errors, path="episode.context")
    _require_nonempty_string(context.get("machine_id"), "episode.context.machine_id", errors)
    _optional_string(context.get("machine_adapter"), "episode.context.machine_adapter", errors)
    _optional_string(context.get("bean_context_id"), "episode.context.bean_context_id", errors)
    _optional_string(context.get("grinder_context_id"), "episode.context.grinder_context_id", errors)
    _require_number_range(context.get("microns_per_step"), "episode.context.microns_per_step", 0.1, 100.0, errors)
    if context.get("step_direction") not in {"higher_is_finer", "higher_is_coarser"}:
        errors.append("episode.context.step_direction is invalid")


def _validate_static_context(static_context: dict[str, Any], errors: list[str]) -> None:
    _reject_unknown_fields(static_context, _STATIC_CONTEXT_FIELDS, errors, path="episode.static_context")
    _require_finite_number(
        static_context.get("relative_grind_steps_from_reference"),
        "episode.static_context.relative_grind_steps_from_reference",
        errors,
    )
    _require_finite_number(
        static_context.get("relative_grind_um_from_reference"),
        "episode.static_context.relative_grind_um_from_reference",
        errors,
    )
    _require_number_range(static_context.get("dose_g"), "episode.static_context.dose_g", 5.0, 30.0, errors)
    _require_number_range(
        static_context.get("initial_target_yield_g"),
        "episode.static_context.initial_target_yield_g",
        5.0,
        100.0,
        errors,
    )
    _require_number_range(static_context.get("target_ratio"), "episode.static_context.target_ratio", 1.2, 3.5, errors)
    _require_bool(static_context.get("grind_observed"), "episode.static_context.grind_observed", errors)
    _require_bool(static_context.get("dose_observed"), "episode.static_context.dose_observed", errors)
    _require_bool(
        static_context.get("initial_target_yield_observed"),
        "episode.static_context.initial_target_yield_observed",
        errors,
    )
    _require_number_range(static_context.get("microns_per_step"), "episode.static_context.microns_per_step", 0.1, 100.0, errors)
    if static_context.get("step_direction") not in {"higher_is_finer", "higher_is_coarser"}:
        errors.append("episode.static_context.step_direction is invalid")
    _optional_string(static_context.get("profile_id"), "episode.static_context.profile_id", errors)
    _optional_string(static_context.get("profile_type"), "episode.static_context.profile_type", errors)
    _optional_int(static_context.get("profile_phase_count"), "episode.static_context.profile_phase_count", errors)

    errors.extend(
        validate_dreamer_taste_objective(
            static_context.get("taste_objective"),
            path="episode.static_context.taste_objective",
        )
    )


def _validate_step(step: object, expected_index: int, errors: list[str]) -> None:
    if not isinstance(step, dict):
        errors.append(f"episode.steps[{expected_index}] must be an object")
        return
    _reject_unknown_fields(step, _STEP_FIELDS, errors, path=f"episode.steps[{expected_index}]")
    step_index = step.get("step_index")
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index != expected_index:
        errors.append(f"episode.steps[{expected_index}].step_index must match its position")
    elapsed_ms = step.get("elapsed_ms")
    expected_elapsed_ms = expected_index * FIXED_CADENCE_SAMPLE_INTERVAL_MS
    if elapsed_ms != expected_elapsed_ms:
        errors.append(
            f"episode.steps[{expected_index}].elapsed_ms must be {expected_elapsed_ms}"
        )

    observation = _require_object(step, "observation", errors, path=f"episode.steps[{expected_index}]")
    if observation is not None:
        _validate_step_observation(observation, expected_index, errors)

    observed_profile_target = _require_object(step, "observed_profile_target", errors, path=f"episode.steps[{expected_index}]")
    if observed_profile_target is not None:
        _reject_unknown_fields(
            observed_profile_target,
            _OBSERVED_PROFILE_TARGET_FIELDS,
            errors,
            path=f"episode.steps[{expected_index}].observed_profile_target",
        )
        _require_number_range(
            observed_profile_target.get("pressure_target_bar"),
            f"episode.steps[{expected_index}].observed_profile_target.pressure_target_bar",
            0.0,
            15.0,
            errors,
        )
        _require_bool(
            observed_profile_target.get("pressure_target_active"),
            f"episode.steps[{expected_index}].observed_profile_target.pressure_target_active",
            errors,
        )
        _require_number_range(
            observed_profile_target.get("flow_target_ml_s"),
            f"episode.steps[{expected_index}].observed_profile_target.flow_target_ml_s",
            0.0,
            20.0,
            errors,
        )
        _require_bool(
            observed_profile_target.get("flow_target_active"),
            f"episode.steps[{expected_index}].observed_profile_target.flow_target_active",
            errors,
        )
        _require_number_range(
            observed_profile_target.get("temperature_target_c"),
            f"episode.steps[{expected_index}].observed_profile_target.temperature_target_c",
            0.0,
            160.0,
            errors,
        )
        _require_bool(
            observed_profile_target.get("temperature_target_active"),
            f"episode.steps[{expected_index}].observed_profile_target.temperature_target_active",
            errors,
        )
        pump_target_mode = observed_profile_target.get("pump_target_mode")
        if (
            isinstance(pump_target_mode, bool)
            or not isinstance(pump_target_mode, int)
            or pump_target_mode not in {0, 1, 2}
        ):
            errors.append(
                f"episode.steps[{expected_index}].observed_profile_target.pump_target_mode is invalid"
            )
        _require_bool(
            observed_profile_target.get("valve_open"),
            f"episode.steps[{expected_index}].observed_profile_target.valve_open",
            errors,
        )

    dynamic_action = step.get("dynamic_action")
    if dynamic_action is not None:
        if not isinstance(dynamic_action, dict):
            errors.append(f"episode.steps[{expected_index}].dynamic_action must be an object or null")
        else:
            _validate_dynamic_action(dynamic_action, expected_index, errors)

    constraints = _require_object(step, "constraints", errors, path=f"episode.steps[{expected_index}]")
    if constraints is not None:
        _validate_constraints(constraints, expected_index, errors)


def _validate_step_observation(observation: dict[str, Any], step_index: int, errors: list[str]) -> None:
    _reject_unknown_fields(observation, _STEP_OBSERVATION_FIELDS, errors, path=f"episode.steps[{step_index}].observation")
    _require_number_range(observation.get("pressure_bar"), f"episode.steps[{step_index}].observation.pressure_bar", 0.0, 15.0, errors)
    _require_number_range(
        observation.get("pump_flow_ml_s"),
        f"episode.steps[{step_index}].observation.pump_flow_ml_s",
        0.0,
        20.0,
        errors,
    )
    _require_number_range(
        observation.get("beverage_flow_g_s"),
        f"episode.steps[{step_index}].observation.beverage_flow_g_s",
        0.0,
        20.0,
        errors,
    )
    _require_number_range(observation.get("weight_g"), f"episode.steps[{step_index}].observation.weight_g", -1.0, 120.0, errors)
    _require_number_range(observation.get("temperature_c"), f"episode.steps[{step_index}].observation.temperature_c", 0.0, 160.0, errors)


def _validate_dynamic_action(dynamic_action: dict[str, Any], step_index: int, errors: list[str]) -> None:
    forbidden = sorted(str(key) for key in dynamic_action if key in _DYNAMIC_ACTION_FORBIDDEN_FIELDS)
    if forbidden:
        errors.append(f"episode.steps[{step_index}].dynamic_action contains static recipe fields: {', '.join(forbidden[:10])}")
    _reject_unknown_fields(dynamic_action, _DYNAMIC_ACTION_FIELDS, errors, path=f"episode.steps[{step_index}].dynamic_action")
    pump_mode = dynamic_action.get("pump_target_mode")
    _optional_number_range(
        pump_mode,
        f"episode.steps[{step_index}].dynamic_action.pump_target_mode",
        1.0,
        2.0,
        errors,
    )
    if pump_mode is not None and (isinstance(pump_mode, bool) or not isinstance(pump_mode, int)):
        errors.append(f"episode.steps[{step_index}].dynamic_action.pump_target_mode must be an integer")
    _optional_number_range(
        dynamic_action.get("pressure_target_bar"),
        f"episode.steps[{step_index}].dynamic_action.pressure_target_bar",
        0.0,
        DREAMER_MAX_PRESSURE_TARGET_BAR,
        errors,
    )
    _optional_number_range(dynamic_action.get("flow_target_ml_s"), f"episode.steps[{step_index}].dynamic_action.flow_target_ml_s", 0.0, 20.0, errors)
    _optional_number_range(dynamic_action.get("valve_position"), f"episode.steps[{step_index}].dynamic_action.valve_position", 0.0, 1.0, errors)
    _optional_number_range(
        dynamic_action.get("temperature_target_c"),
        f"episode.steps[{step_index}].dynamic_action.temperature_target_c",
        DREAMER_MIN_TEMPERATURE_TARGET_C,
        DREAMER_MAX_TEMPERATURE_TARGET_C,
        errors,
    )
    _optional_number_range(
        dynamic_action.get("yield_stop_target_g"),
        f"episode.steps[{step_index}].dynamic_action.yield_stop_target_g",
        5.0,
        DREAMER_MAX_YIELD_STOP_TARGET_G,
        errors,
    )
    _optional_bool(dynamic_action.get("stop"), f"episode.steps[{step_index}].dynamic_action.stop", errors)
    has_pressure = dynamic_action.get("pressure_target_bar") is not None
    has_flow = dynamic_action.get("flow_target_ml_s") is not None
    if (has_pressure or has_flow) and pump_mode is None:
        errors.append(f"episode.steps[{step_index}].dynamic_action.pump_target_mode is required")
    if pump_mode == 1 and has_flow:
        errors.append(f"episode.steps[{step_index}].dynamic_action pressure mode cannot include flow")
    if pump_mode == 2 and has_pressure:
        errors.append(f"episode.steps[{step_index}].dynamic_action flow mode cannot include pressure")


def _validate_constraints(constraints: dict[str, Any], step_index: int, errors: list[str]) -> None:
    _reject_unknown_fields(constraints, _CONSTRAINT_FIELDS, errors, path=f"episode.steps[{step_index}].constraints")
    for key in sorted(_CONSTRAINT_FIELDS):
        _require_bool(constraints.get(key), f"episode.steps[{step_index}].constraints.{key}", errors)


def _validate_terminal(terminal: dict[str, Any], errors: list[str]) -> None:
    _reject_unknown_fields(terminal, _TERMINAL_FIELDS, errors, path="episode.terminal")
    _require_nonempty_string(terminal.get("shot_id"), "episode.terminal.shot_id", errors)
    _require_number_range(terminal.get("timestamp"), "episode.terminal.timestamp", 0.0, 9_007_199_254_740_991.0, errors)
    _optional_number_range(terminal.get("beverage_out_g"), "episode.terminal.beverage_out_g", 0.0, 120.0, errors)
    _optional_number_range(terminal.get("brew_ratio"), "episode.terminal.brew_ratio", 0.1, 10.0, errors)
    _optional_number_range(terminal.get("shot_time_s"), "episode.terminal.shot_time_s", 0.0, 180.0, errors)
    _optional_number_range(terminal.get("human_rating"), "episode.terminal.human_rating", 1.0, 5.0, errors)
    _optional_number_range(terminal.get("reward"), "episode.terminal.reward", 0.0, 1.0, errors)
    _require_number_range(terminal.get("confidence"), "episode.terminal.confidence", 0.0, 1.0, errors)
    _optional_bool(terminal.get("feedback_recorded"), "episode.terminal.feedback_recorded", errors)
    _require_number_range(terminal.get("optimization_weight"), "episode.terminal.optimization_weight", 0.0, 1.0, errors)
    _optional_number_range(terminal.get("profile_score"), "episode.terminal.profile_score", 0.0, 1.0, errors)
    _optional_number_range(terminal.get("profile_mse"), "episode.terminal.profile_mse", 0.0, 1_000_000.0, errors)
    _optional_bool(terminal.get("profile_flow_valid"), "episode.terminal.profile_flow_valid", errors)
    _optional_bool(terminal.get("profile_flow_masked"), "episode.terminal.profile_flow_masked", errors)
    _optional_enum(terminal.get("final_pump_target"), "episode.terminal.final_pump_target", {"simple", "pressure", "flow"}, errors)
    _optional_number_range(terminal.get("final_target_pressure"), "episode.terminal.final_target_pressure", 0.0, 15.0, errors)
    _optional_number_range(terminal.get("final_target_flow"), "episode.terminal.final_target_flow", 0.0, 25.0, errors)
    _require_number_range(terminal.get("profile_temperature_c"), "episode.terminal.profile_temperature_c", 0.0, 160.0, errors)
    _require_number_range(terminal.get("final_phase_temperature_c"), "episode.terminal.final_phase_temperature_c", 0.0, 160.0, errors)
    _optional_string(terminal.get("shot_end_state"), "episode.terminal.shot_end_state", errors)
    taste_tags = terminal.get("taste_tags")
    if taste_tags is not None:
        if not isinstance(taste_tags, list):
            errors.append("episode.terminal.taste_tags must be a list")
        else:
            invalid = [tag for tag in taste_tags if not isinstance(tag, str) or tag not in VALID_TASTE_TAGS]
            if invalid:
                errors.append("episode.terminal.taste_tags contains invalid values")


def _validate_recommendation(recommendation: dict[str, Any], errors: list[str]) -> None:
    _reject_unknown_fields(recommendation, _RECOMMENDATION_FIELDS, errors, path="episode.recommendation")
    _optional_string(recommendation.get("recommendation_id"), "episode.recommendation.recommendation_id", errors)
    _optional_finite_number(
        recommendation.get("grind_delta_steps_from_current"),
        "episode.recommendation.grind_delta_steps_from_current",
        errors,
    )
    _optional_finite_number(recommendation.get("grind_delta_um_from_current"), "episode.recommendation.grind_delta_um_from_current", errors)
    _optional_finite_number(
        recommendation.get("projected_relative_step_from_reference"),
        "episode.recommendation.projected_relative_step_from_reference",
        errors,
    )
    _optional_finite_number(
        recommendation.get("projected_relative_grind_um_from_reference"),
        "episode.recommendation.projected_relative_grind_um_from_reference",
        errors,
    )
    _optional_number_range(recommendation.get("next_dose_g"), "episode.recommendation.next_dose_g", 5.0, 30.0, errors)
    _optional_number_range(recommendation.get("target_yield_g"), "episode.recommendation.target_yield_g", 5.0, 100.0, errors)
    _optional_number_range(recommendation.get("target_ratio"), "episode.recommendation.target_ratio", 1.2, 3.5, errors)
    _optional_number_range(recommendation.get("attribution_weight"), "episode.recommendation.attribution_weight", 0.0, 1.0, errors)
    _optional_enum(recommendation.get("decision"), "episode.recommendation.decision", {"accepted", "edited", "ignored", "dismissed", "unknown"}, errors)
    _optional_enum(
        recommendation.get("follow_through"),
        "episode.recommendation.follow_through",
        {"followed", "partially_followed", "not_followed", "unknown"},
        errors,
    )
    field_trust = recommendation.get("field_trust")
    if field_trust is not None:
        if not isinstance(field_trust, dict):
            errors.append("episode.recommendation.field_trust must be an object")
        else:
            _reject_unknown_fields(field_trust, _FIELD_TRUST_FIELDS, errors, path="episode.recommendation.field_trust")
            _optional_number_range(field_trust.get("grind"), "episode.recommendation.field_trust.grind", 0.0, 1.0, errors)
            _optional_number_range(field_trust.get("dose"), "episode.recommendation.field_trust.dose", 0.0, 1.0, errors)
            _optional_number_range(field_trust.get("yield"), "episode.recommendation.field_trust.yield", 0.0, 1.0, errors)


def _reject_unknown_fields(value: dict[str, Any], allowed: frozenset[str], errors: list[str], *, path: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        errors.append(f"{path} contains unknown fields: {', '.join(unknown[:10])}")


def _reject_forbidden_fields(value: Any, errors: list[str], *, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}" if path else key_text
            if key_text in FORBIDDEN_TRAINING_FIELD_NAMES:
                errors.append(f"{item_path} is not allowed in Dreamer episode data")
            _reject_forbidden_fields(item, errors, path=item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, errors, path=f"{path}[{index}]")


def _require_object(row: dict[str, Any], key: str, errors: list[str], *, path: str = "episode") -> dict[str, Any] | None:
    value = row.get(key)
    if not isinstance(value, dict):
        errors.append(f"{path}.{key} must be an object")
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


def _optional_enum(value: object, label: str, allowed: set[str], errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, str) or value not in allowed:
        errors.append(f"{label} is invalid")


def _require_nonempty_string(value: object, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        errors.append(f"{label} must be a short non-empty string")


def _optional_string(value: object, label: str, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, str) or len(value) > 160:
        errors.append(f"{label} must be a short string")


def _require_bool(value: object, label: str, errors: list[str]) -> None:
    if not isinstance(value, bool):
        errors.append(f"{label} must be boolean")


def _optional_bool(value: object, label: str, errors: list[str]) -> None:
    if value is None:
        return
    _require_bool(value, label, errors)


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

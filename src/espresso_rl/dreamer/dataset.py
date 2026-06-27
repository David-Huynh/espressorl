from __future__ import annotations

"""
Dreamer training dataset helpers.

The legacy `sample_batch` path keeps existing shot-sequence training code
working. The canonical episode loader below converts validated
`espresso_rl_training_transition_v1` rows into variable-length shot episodes
whose recurrent time steps are always 250 milliseconds apart.
"""

import json
import random
from typing import Any, Iterable

import numpy as np
import torch

from espresso_rl.domain.dreamer_control import (
    DEFAULT_DREAMER_CONTROL_SPEC,
    DREAMER_CONTROL_CONSTRAINT_FIELDS,
    DREAMER_DYNAMIC_ACTION_FIELDS,
    DreamerControlSpec,
    validate_dynamic_action_for_control_spec,
)
from espresso_rl.domain.dreamer_episodes import (
    DREAMER_EPISODE_FORMAT,
    DREAMER_PROFILE_CHANNELS,
    DREAMER_EPISODE_SCHEMA_VERSION,
    validate_dreamer_episode,
)
from espresso_rl.domain.training import validate_training_transition

from ..models import ShotRecord
from .actor import FactoredCategoricalActor

SEQ_LEN = 8
MIN_SHOTS = 4

_PUMP_TARGET_MODE_PRESSURE = 1
_PUMP_TARGET_MODE_FLOW = 2
_DEFAULT_CONSTRAINTS = DEFAULT_DREAMER_CONTROL_SPEC.constraints()
DREAMER_OBSERVATION_FEATURES = DREAMER_PROFILE_CHANNELS
DREAMER_OBSERVED_TARGET_FEATURES = (
    "pressure_target_bar",
    "flow_target_ml_s",
    "temperature_target_c",
    "pump_target_mode",
    "valve_open",
)
DREAMER_DYNAMIC_ACTION_FEATURES = DREAMER_DYNAMIC_ACTION_FIELDS
DREAMER_CONSTRAINT_FEATURES = DREAMER_CONTROL_CONSTRAINT_FIELDS
DREAMER_STATIC_CONTEXT_FEATURES = (
    "relative_grind_steps_from_reference",
    "relative_grind_um_from_reference",
    "dose_g",
    "initial_target_yield_g",
    "target_ratio",
    "microns_per_step",
    "step_direction_sign",
    "profile_phase_count",
    "taste_objective_auto",
    "taste_objective_acidity",
    "taste_objective_sweetness",
    "taste_objective_clarity",
    "taste_objective_body",
    "taste_objective_bitterness",
    "taste_objective_chocolatiness",
    "taste_objective_fruitiness",
    "taste_objective_roastiness",
)
DREAMER_TERMINAL_FEATURES = (
    "beverage_out_g",
    "brew_ratio",
    "shot_time_s",
    "human_rating",
    "reward",
    "confidence",
    "feedback_recorded",
    "optimization_weight",
    "profile_score",
    "profile_mse",
    "profile_flow_valid",
    "profile_flow_masked",
    "final_pump_target_pressure",
    "final_pump_target_flow",
    "final_target_pressure",
    "final_target_flow",
    "profile_temperature_c",
    "final_phase_temperature_c",
)
_TASTE_LEVEL_VALUES = {
    None: 0.0,
    "none": 0.0,
    "low": 1.0 / 3.0,
    "medium": 2.0 / 3.0,
    "high": 1.0,
}


class DreamerEpisodeDatasetError(ValueError):
    pass


def _encode_action(shot: ShotRecord) -> tuple[int, int]:
    """Convert a ShotRecord's stored action back to discrete indices."""
    delta_steps = shot.action_grind_delta_um_from_current / max(shot.microns_per_step, 1e-6)
    grind_idx = FactoredCategoricalActor.encode_grind(round(delta_steps))
    dose_idx = FactoredCategoricalActor.encode_dose(shot.action_dose_g)
    return grind_idx, dose_idx


def _pad_window(window: list[ShotRecord], target_len: int) -> list[ShotRecord]:
    """Left-pad a window shorter than target_len by repeating the first element."""
    if len(window) >= target_len:
        return window
    pad = [window[0]] * (target_len - len(window))
    return pad + window


def sample_batch(
    shots: list[ShotRecord],
    batch_size: int = 16,
    seq_len: int = SEQ_LEN,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor] | None:
    """
    Sample a random batch of contiguous historical shot sequences.

    This is the existing batch format used by the current Dreamer prototype.
    It treats each completed shot as one sequence element. New offline RSSM
    training should consume the canonical episode loader instead.
    """
    if len(shots) < MIN_SHOTS:
        return None

    obs_list, action_list, reward_list, cont_list = [], [], [], []

    for _ in range(batch_size):
        n = len(shots)
        if n >= seq_len:
            start = random.randint(0, n - seq_len)
            window = shots[start : start + seq_len]
        else:
            window = _pad_window(shots, seq_len)

        obs = np.stack([s.shot_profile for s in window])
        rewards = np.array([s.reward or 0.0 for s in window], dtype=np.float32)
        conts = np.ones(seq_len, dtype=np.float32)
        actions = [_encode_action(s) for s in window]

        obs_list.append(obs)
        action_list.append(actions)
        reward_list.append(rewards)
        cont_list.append(conts)

    return {
        "obs": torch.tensor(np.stack(obs_list), dtype=torch.float32, device=device),
        "actions": torch.tensor(action_list, dtype=torch.long, device=device),
        "rewards": torch.tensor(np.stack(reward_list), dtype=torch.float32, device=device),
        "conts": torch.tensor(np.stack(cont_list), dtype=torch.float32, device=device),
    }


def load_dreamer_episodes_from_jsonl(
    training_rows_jsonl: str,
    *,
    require_profile: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(training_rows_jsonl.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DreamerEpisodeDatasetError(f"training row {line_number} is not valid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise DreamerEpisodeDatasetError(f"training row {line_number} must be an object")
        rows.append(row)
    return build_dreamer_episodes_from_training_rows(rows, require_profile=require_profile)


def build_dreamer_episodes_from_training_rows(
    rows: Iterable[dict[str, Any]],
    *,
    require_profile: bool = True,
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for row in rows:
        row_id = row.get("training_row_id") if isinstance(row, dict) else "unknown"
        if not isinstance(row, dict):
            raise DreamerEpisodeDatasetError("training row must be an object")
        errors = validate_training_transition(row)
        if errors:
            raise DreamerEpisodeDatasetError(f"training row {row_id} failed validation: {'; '.join(errors[:10])}")

        profile = row["observation"].get("profile_resampled")
        if profile is None:
            if require_profile:
                raise DreamerEpisodeDatasetError(f"training row {row_id} is missing observation.profile_resampled")
            continue
        _require_dreamer_profile_inputs(row["observation"], row_id)

        episode = _episode_from_training_row(row)
        episode_errors = validate_dreamer_episode(episode)
        if episode_errors:
            raise DreamerEpisodeDatasetError(f"training row {row_id} produced invalid Dreamer episode: {'; '.join(episode_errors[:10])}")
        episodes.append(episode)

    return sorted(episodes, key=_episode_sort_key)


def build_dreamer_episode_batch(
    episodes: Iterable[dict[str, Any]],
    *,
    pad_to_step_count: int | None = None,
    control_spec: DreamerControlSpec | dict[str, Any] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    sorted_episodes = sorted(_validated_episodes(episodes), key=_episode_sort_key)
    if not sorted_episodes:
        raise DreamerEpisodeDatasetError("Dreamer episode batch must contain at least one episode")
    resolved_control_spec = _resolve_control_spec(control_spec)

    max_steps = max(len(episode["steps"]) for episode in sorted_episodes)
    if pad_to_step_count is not None:
        if pad_to_step_count < max_steps:
            raise DreamerEpisodeDatasetError("pad_to_step_count cannot be shorter than the longest episode")
        max_steps = pad_to_step_count

    batch_size = len(sorted_episodes)
    observations = np.zeros((batch_size, max_steps, len(DREAMER_OBSERVATION_FEATURES)), dtype=np.float32)
    observed_targets = np.zeros((batch_size, max_steps, len(DREAMER_OBSERVED_TARGET_FEATURES)), dtype=np.float32)
    observed_target_mask = np.zeros_like(observed_targets)
    dynamic_actions = np.zeros((batch_size, max_steps, len(DREAMER_DYNAMIC_ACTION_FEATURES)), dtype=np.float32)
    dynamic_action_mask = np.zeros_like(dynamic_actions)
    control_action_mask = np.zeros_like(dynamic_actions)
    constraints = np.zeros((batch_size, max_steps, len(DREAMER_CONSTRAINT_FEATURES)), dtype=np.float32)
    decision_step_mask = np.zeros((batch_size, max_steps), dtype=np.float32)
    elapsed_seconds = np.zeros((batch_size, max_steps, 1), dtype=np.float32)
    step_duration_seconds = np.zeros((batch_size, max_steps, 1), dtype=np.float32)
    step_mask = np.zeros((batch_size, max_steps), dtype=np.float32)
    continuations = np.zeros((batch_size, max_steps), dtype=np.float32)
    rewards = np.zeros((batch_size, max_steps), dtype=np.float32)
    static_context = np.zeros((batch_size, len(DREAMER_STATIC_CONTEXT_FEATURES)), dtype=np.float32)
    terminal = np.zeros((batch_size, len(DREAMER_TERMINAL_FEATURES)), dtype=np.float32)
    episode_weights = np.zeros(batch_size, dtype=np.float32)
    source_training_row_ids = np.zeros(batch_size, dtype=np.int64)
    episode_ids: list[str] = []

    for batch_index, episode in enumerate(sorted_episodes):
        steps = episode["steps"]
        valid_step_count = len(steps)
        held_dynamic_action: dict[str, Any] | None = None
        static_context[batch_index] = _encode_static_context(episode["static_context"])
        terminal[batch_index] = _encode_terminal(episode["terminal"])
        terminal_reward = _finite_or_zero(episode["terminal"].get("reward"))
        episode_weights[batch_index] = _episode_weight(episode)
        source_training_row_ids[batch_index] = int(episode["source_training_row_id"])
        episode_ids.append(str(episode["episode_id"]))

        for step_index, step in enumerate(steps):
            observations[batch_index, step_index] = _encode_object(
                step["observation"],
                DREAMER_OBSERVATION_FEATURES,
            )
            observed_targets[batch_index, step_index] = _encode_object(
                step["observed_profile_target"],
                DREAMER_OBSERVED_TARGET_FEATURES,
            )
            observed_target_mask[batch_index, step_index] = _encode_observed_target_mask(step["observed_profile_target"])
            decision_action = step.get("dynamic_action")
            if decision_action is not None:
                action_errors = validate_dynamic_action_for_control_spec(
                    decision_action,
                    control_spec=resolved_control_spec,
                    step_index=step_index,
                )
                if action_errors:
                    label = episode.get("episode_id", batch_index)
                    raise DreamerEpisodeDatasetError(
                        f"Dreamer episode {label} dynamic action is invalid: {'; '.join(action_errors[:10])}"
                    )
                held_dynamic_action = dict(decision_action)
            action_values, action_mask = _encode_dynamic_action(held_dynamic_action)
            dynamic_actions[batch_index, step_index] = action_values
            dynamic_action_mask[batch_index, step_index] = action_mask
            constraints[batch_index, step_index] = _encode_bool_object(
                step["constraints"],
                DREAMER_CONSTRAINT_FEATURES,
            )
            if resolved_control_spec.is_decision_step(step_index):
                decision_step_mask[batch_index, step_index] = 1.0
                control_action_mask[batch_index, step_index] = np.asarray(
                    resolved_control_spec.action_capability_mask(),
                    dtype=np.float32,
                )
            elapsed_seconds[batch_index, step_index, 0] = float(step["elapsed_ms"]) / 1000.0
            step_duration_seconds[batch_index, step_index, 0] = float(episode["sample_interval_ms"]) / 1000.0
            step_mask[batch_index, step_index] = 1.0
            continuations[batch_index, step_index] = 1.0 if step_index < valid_step_count - 1 else 0.0
        rewards[batch_index, valid_step_count - 1] = terminal_reward

    target_device = torch.device(device) if device is not None else None
    return {
        "observations": torch.tensor(observations, dtype=torch.float32, device=target_device),
        "observed_profile_targets": torch.tensor(observed_targets, dtype=torch.float32, device=target_device),
        "observed_profile_target_mask": torch.tensor(observed_target_mask, dtype=torch.float32, device=target_device),
        "dynamic_actions": torch.tensor(dynamic_actions, dtype=torch.float32, device=target_device),
        "dynamic_action_mask": torch.tensor(dynamic_action_mask, dtype=torch.float32, device=target_device),
        "control_action_mask": torch.tensor(control_action_mask, dtype=torch.float32, device=target_device),
        "constraints": torch.tensor(constraints, dtype=torch.float32, device=target_device),
        "decision_step_mask": torch.tensor(decision_step_mask, dtype=torch.float32, device=target_device),
        "elapsed_seconds": torch.tensor(elapsed_seconds, dtype=torch.float32, device=target_device),
        "step_duration_seconds": torch.tensor(step_duration_seconds, dtype=torch.float32, device=target_device),
        "step_mask": torch.tensor(step_mask, dtype=torch.float32, device=target_device),
        "continuations": torch.tensor(continuations, dtype=torch.float32, device=target_device),
        "rewards": torch.tensor(rewards, dtype=torch.float32, device=target_device),
        "static_context": torch.tensor(static_context, dtype=torch.float32, device=target_device),
        "terminal": torch.tensor(terminal, dtype=torch.float32, device=target_device),
        "episode_weights": torch.tensor(episode_weights, dtype=torch.float32, device=target_device),
        "source_training_row_ids": torch.tensor(source_training_row_ids, dtype=torch.long, device=target_device),
        "episode_ids": tuple(episode_ids),
        "control_spec": resolved_control_spec.to_dict(),
        "feature_names": {
            "observations": DREAMER_OBSERVATION_FEATURES,
            "observed_profile_targets": DREAMER_OBSERVED_TARGET_FEATURES,
            "observed_profile_target_mask": DREAMER_OBSERVED_TARGET_FEATURES,
            "dynamic_actions": DREAMER_DYNAMIC_ACTION_FEATURES,
            "control_action_mask": DREAMER_DYNAMIC_ACTION_FEATURES,
            "constraints": DREAMER_CONSTRAINT_FEATURES,
            "static_context": DREAMER_STATIC_CONTEXT_FEATURES,
            "terminal": DREAMER_TERMINAL_FEATURES,
        },
    }


def _validated_episodes(episodes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise DreamerEpisodeDatasetError(f"Dreamer episode {index} must be an object")
        errors = validate_dreamer_episode(episode)
        if errors:
            label = episode.get("episode_id", index)
            raise DreamerEpisodeDatasetError(f"Dreamer episode {label} failed validation: {'; '.join(errors[:10])}")
        validated.append(episode)
    return validated


def _resolve_control_spec(control_spec: DreamerControlSpec | dict[str, Any] | None) -> DreamerControlSpec:
    if control_spec is None:
        return DEFAULT_DREAMER_CONTROL_SPEC
    if isinstance(control_spec, DreamerControlSpec):
        return control_spec
    return DreamerControlSpec.from_dict(control_spec)


def _encode_static_context(static_context: dict[str, Any]) -> np.ndarray:
    encoded = np.zeros(len(DREAMER_STATIC_CONTEXT_FEATURES), dtype=np.float32)
    encoded[_feature_index(DREAMER_STATIC_CONTEXT_FEATURES, "relative_grind_steps_from_reference")] = _finite_or_zero(
        static_context.get("relative_grind_steps_from_reference")
    )
    encoded[_feature_index(DREAMER_STATIC_CONTEXT_FEATURES, "relative_grind_um_from_reference")] = _finite_or_zero(
        static_context.get("relative_grind_um_from_reference")
    )
    encoded[_feature_index(DREAMER_STATIC_CONTEXT_FEATURES, "dose_g")] = _finite_or_zero(static_context.get("dose_g"))
    encoded[_feature_index(DREAMER_STATIC_CONTEXT_FEATURES, "initial_target_yield_g")] = _finite_or_zero(
        static_context.get("initial_target_yield_g")
    )
    encoded[_feature_index(DREAMER_STATIC_CONTEXT_FEATURES, "target_ratio")] = _finite_or_zero(static_context.get("target_ratio"))
    encoded[_feature_index(DREAMER_STATIC_CONTEXT_FEATURES, "microns_per_step")] = _finite_or_zero(static_context.get("microns_per_step"))
    encoded[_feature_index(DREAMER_STATIC_CONTEXT_FEATURES, "step_direction_sign")] = (
        1.0 if static_context.get("step_direction") == "higher_is_finer" else -1.0
    )
    encoded[_feature_index(DREAMER_STATIC_CONTEXT_FEATURES, "profile_phase_count")] = _finite_or_zero(
        static_context.get("profile_phase_count")
    )

    taste_objective = static_context.get("taste_objective") or {}
    encoded[_feature_index(DREAMER_STATIC_CONTEXT_FEATURES, "taste_objective_auto")] = (
        1.0 if taste_objective.get("mode") == "auto" else 0.0
    )
    for attribute in (
        "acidity",
        "sweetness",
        "clarity",
        "body",
        "bitterness",
        "chocolatiness",
        "fruitiness",
        "roastiness",
    ):
        encoded[_feature_index(DREAMER_STATIC_CONTEXT_FEATURES, f"taste_objective_{attribute}")] = _TASTE_LEVEL_VALUES[
            taste_objective.get(attribute)
        ]
    return encoded


def _encode_terminal(terminal: dict[str, Any]) -> np.ndarray:
    encoded = np.zeros(len(DREAMER_TERMINAL_FEATURES), dtype=np.float32)
    for feature in DREAMER_TERMINAL_FEATURES:
        if feature == "final_pump_target_pressure":
            encoded[_feature_index(DREAMER_TERMINAL_FEATURES, feature)] = 1.0 if terminal.get("final_pump_target") == "pressure" else 0.0
            continue
        if feature == "final_pump_target_flow":
            encoded[_feature_index(DREAMER_TERMINAL_FEATURES, feature)] = 1.0 if terminal.get("final_pump_target") == "flow" else 0.0
            continue
        value = terminal.get(feature)
        encoded[_feature_index(DREAMER_TERMINAL_FEATURES, feature)] = _bool_or_number(value)
    return encoded


def _encode_object(value: dict[str, Any], features: tuple[str, ...]) -> np.ndarray:
    encoded = np.zeros(len(features), dtype=np.float32)
    for feature_index, feature in enumerate(features):
        encoded[feature_index] = _bool_or_number(value.get(feature))
    return encoded


def _encode_bool_object(value: dict[str, Any], features: tuple[str, ...]) -> np.ndarray:
    encoded = np.zeros(len(features), dtype=np.float32)
    for feature_index, feature in enumerate(features):
        encoded[feature_index] = 1.0 if value.get(feature) is True else 0.0
    return encoded


def _encode_observed_target_mask(observed_profile_target: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            1.0 if observed_profile_target.get("pressure_target_active") is True else 0.0,
            1.0 if observed_profile_target.get("flow_target_active") is True else 0.0,
            1.0 if observed_profile_target.get("temperature_target_active") is True else 0.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )


def _encode_dynamic_action(action: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(len(DREAMER_DYNAMIC_ACTION_FEATURES), dtype=np.float32)
    mask = np.zeros(len(DREAMER_DYNAMIC_ACTION_FEATURES), dtype=np.float32)
    if action is None:
        return values, mask
    for feature_index, feature in enumerate(DREAMER_DYNAMIC_ACTION_FEATURES):
        if feature not in action:
            continue
        mask[feature_index] = 1.0
        values[feature_index] = _bool_or_number(action.get(feature))
    return values, mask


def _episode_weight(episode: dict[str, Any]) -> float:
    source_weight = _clamp01(_finite_or_zero(episode["source"].get("trust_weight")))
    optimization_weight = _clamp01(_finite_or_zero(episode["terminal"].get("optimization_weight")))
    confidence = _clamp01(_finite_or_zero(episode["terminal"].get("confidence")))
    return source_weight * optimization_weight * confidence


def _feature_index(features: tuple[str, ...], feature: str) -> int:
    return features.index(feature)


def _bool_or_number(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _finite_or_zero(value)


def _finite_or_zero(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)):
        return float(value)
    return 0.0


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _episode_from_training_row(row: dict[str, Any]) -> dict[str, Any]:
    source = row["source"]
    context = row["context"]
    action = row["action"]
    observation = row["observation"]
    reward = row["reward"]
    sequence = observation["fixed_cadence_sequence"]

    group_key = {
        "install_id": source["install_id"],
        "machine_id": context["machine_id"],
        "bean_context_id": context.get("bean_context_id"),
        "grinder_context_id": context.get("grinder_context_id"),
    }
    episode = {
        "format": DREAMER_EPISODE_FORMAT,
        "schema_version": DREAMER_EPISODE_SCHEMA_VERSION,
        "episode_id": f"training_row_{row['training_row_id']}",
        "source_training_row_id": row["training_row_id"],
        "sample_interval_ms": sequence["sample_interval_ms"],
        "group_key": group_key,
        "source": dict(source),
        "context": dict(context),
        "static_context": _static_context(action, context, observation),
        "steps": _fixed_cadence_steps(sequence, observation),
        "terminal": _terminal(observation, reward),
    }
    recommendation = row.get("recommendation")
    if recommendation is not None:
        episode["recommendation"] = dict(recommendation)
    return _drop_none_values(episode)


def _static_context(
    action: dict[str, Any],
    context: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    return _drop_none_values(
        {
            "relative_grind_steps_from_reference": action["relative_grind_steps_from_reference"],
            "relative_grind_um_from_reference": action["relative_grind_um_from_reference"],
            "dose_g": action["dose_g"],
            "initial_target_yield_g": action["target_yield_g"],
            "target_ratio": action["target_ratio"],
            "microns_per_step": context["microns_per_step"],
            "step_direction": context["step_direction"],
            "profile_id": observation.get("profile_id"),
            "profile_type": observation.get("profile_type"),
            "profile_phase_count": observation.get("profile_phase_count"),
            "taste_objective": {"mode": "auto"},
        }
    )


def _fixed_cadence_steps(sequence: dict[str, Any], observation: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    sample_interval_ms = int(sequence["sample_interval_ms"])
    step_count = len(sequence["pressure_bar"])
    flow_targets_masked = observation.get("profile_flow_masked") is True
    for index in range(step_count):
        pressure_target = sequence["pressure_target_bar"][index]
        flow_target = sequence["pump_flow_target_ml_s"][index]
        pump_target_mode = sequence["pump_target_mode"][index]
        pressure_target_active = pump_target_mode == _PUMP_TARGET_MODE_PRESSURE
        flow_target_active = pump_target_mode == _PUMP_TARGET_MODE_FLOW
        if flow_targets_masked:
            flow_target_active = False
        step_observation = {
            "pressure_bar": sequence["pressure_bar"][index],
            "pump_flow_ml_s": sequence["pump_flow_ml_s"][index],
            "beverage_flow_g_s": sequence["beverage_flow_g_s"][index],
            "weight_g": sequence["weight_g"][index],
            "temperature_c": sequence["temperature_c"][index],
        }
        observed_profile_target = {
            "pressure_target_bar": pressure_target,
            "pressure_target_active": pressure_target_active,
            "flow_target_ml_s": flow_target,
            "flow_target_active": flow_target_active,
            "temperature_target_c": sequence["temperature_target_c"][index],
            "temperature_target_active": sequence["temperature_target_c"][index] > 0.0,
            "pump_target_mode": pump_target_mode,
            "valve_open": sequence["valve_open"][index],
        }
        steps.append(
            {
                "step_index": index,
                "elapsed_ms": index * sample_interval_ms,
                "observation": step_observation,
                "observed_profile_target": observed_profile_target,
                "dynamic_action": None,
                "constraints": dict(_DEFAULT_CONSTRAINTS),
            }
        )
    return steps


def _require_dreamer_profile_inputs(observation: dict[str, Any], row_id: object) -> None:
    value = observation.get("fixed_cadence_sequence")
    if not isinstance(value, dict):
        raise DreamerEpisodeDatasetError(
            f"training row {row_id} is missing required observation.fixed_cadence_sequence"
        )


def _terminal(observation: dict[str, Any], reward: dict[str, Any]) -> dict[str, Any]:
    return _drop_none_values(
        {
            "shot_id": observation["shot_id"],
            "timestamp": observation["timestamp"],
            "beverage_out_g": observation.get("beverage_out_g"),
            "brew_ratio": observation.get("brew_ratio"),
            "shot_time_s": observation.get("shot_time_s"),
            "human_rating": reward.get("human_rating"),
            "taste_tags": list(reward.get("taste_tags") or []),
            "reward": reward.get("reward"),
            "confidence": reward["confidence"],
            "feedback_recorded": reward.get("feedback_recorded"),
            "optimization_weight": reward["optimization_weight"],
            "profile_score": observation.get("profile_score"),
            "profile_mse": observation.get("profile_mse"),
            "profile_flow_valid": observation.get("profile_flow_valid"),
            "profile_flow_masked": observation.get("profile_flow_masked"),
            "final_pump_target": observation.get("final_pump_target"),
            "final_target_pressure": observation.get("final_target_pressure"),
            "final_target_flow": observation.get("final_target_flow"),
            "profile_temperature_c": observation.get("profile_temperature_c"),
            "final_phase_temperature_c": observation.get("final_phase_temperature_c"),
            "shot_end_state": observation.get("shot_end_state"),
        }
    )


def _episode_sort_key(episode: dict[str, Any]) -> tuple[str, str, str, str, float, int]:
    group_key = episode["group_key"]
    terminal = episode["terminal"]
    return (
        str(group_key.get("install_id") or ""),
        str(group_key.get("machine_id") or ""),
        str(group_key.get("bean_context_id") or ""),
        str(group_key.get("grinder_context_id") or ""),
        float(terminal["timestamp"]),
        int(episode["source_training_row_id"]),
    )


def _drop_none_values(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}

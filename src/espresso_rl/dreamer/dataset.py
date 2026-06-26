from __future__ import annotations

"""
Dreamer training dataset helpers.

The legacy `sample_batch` path keeps existing shot-sequence training code
working. The canonical episode loader below converts validated
`espresso_rl_training_transition_v1` rows into shot episodes where each
resampled profile sample is one recurrent time step.
"""

import json
import random
from typing import Any, Iterable

import numpy as np
import torch

from espresso_rl.domain.dreamer_episodes import (
    DREAMER_EPISODE_FORMAT,
    DREAMER_EPISODE_SCHEMA_VERSION,
    validate_dreamer_episode,
)
from espresso_rl.domain.training import validate_training_transition

from ..models import ShotRecord
from .actor import FactoredCategoricalActor

SEQ_LEN = 8
MIN_SHOTS = 4

_PROFILE_SAMPLE_COUNT = 100
_DEFAULT_CONSTRAINTS = {
    "dynamic_control_enabled": False,
    "pressure_control_allowed": False,
    "flow_control_allowed": False,
    "pump_control_allowed": False,
    "valve_control_allowed": False,
    "temperature_control_allowed": False,
    "stop_control_allowed": False,
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

        episode = _episode_from_training_row(row)
        episode_errors = validate_dreamer_episode(episode)
        if episode_errors:
            raise DreamerEpisodeDatasetError(f"training row {row_id} produced invalid Dreamer episode: {'; '.join(episode_errors[:10])}")
        episodes.append(episode)

    return sorted(episodes, key=_episode_sort_key)


def _episode_from_training_row(row: dict[str, Any]) -> dict[str, Any]:
    source = row["source"]
    context = row["context"]
    action = row["action"]
    observation = row["observation"]
    reward = row["reward"]
    profile = observation["profile_resampled"]

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
        "group_key": group_key,
        "source": dict(source),
        "context": dict(context),
        "static_context": _static_context(action, context, observation),
        "steps": _profile_steps(profile),
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


def _profile_steps(profile: list[list[float]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    last_index = _PROFILE_SAMPLE_COUNT - 1
    for index in range(_PROFILE_SAMPLE_COUNT):
        steps.append(
            {
                "step_index": index,
                "elapsed_fraction": round(index / last_index, 6),
                "observation": {
                    "pressure_bar": profile[0][index],
                    "target_pressure_bar": profile[1][index],
                    "flow_ml_s": profile[2][index],
                    "target_flow_ml_s": profile[3][index],
                    "weight_g": profile[4][index],
                },
                "observed_profile_target": {
                    "pressure_target_bar": profile[1][index],
                    "flow_target_ml_s": profile[3][index],
                },
                "dynamic_action": None,
                "constraints": dict(_DEFAULT_CONSTRAINTS),
            }
        )
    return steps


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

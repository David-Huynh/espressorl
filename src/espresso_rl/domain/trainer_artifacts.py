from __future__ import annotations

import math
from typing import Any

from espresso_rl.domain.dreamer_control import DEFAULT_DREAMER_CONTROL_SPEC, DreamerControlSpec
from espresso_rl.domain.dreamer_live_action import (
    DEFAULT_DREAMER_LIVE_ACTION_SPEC,
    DreamerLiveActionSpec,
)
from espresso_rl.domain.dreamer_pre_shot import (
    DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC,
    DreamerPreShotActionSpec,
)
from espresso_rl.domain.dreamer_taste import (
    DEFAULT_DREAMER_TASTE_OBJECTIVE_SPEC,
    DreamerTasteObjectiveSpec,
)
from espresso_rl.domain.model_manifest import MODEL_FAMILY_DREAMER_V3

TRAINING_CONFIG_FORMAT = "espresso_rl_training_config_v1"
TRAINING_CONFIG_SCHEMA_VERSION = 1
TRAINER_AUDIT_REPORT_FORMAT = "espresso_rl_trainer_audit_report_v1"
TRAINER_ARTIFACT_STAGE_CONTRACT_ONLY = "artifact_contract_only"
TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE = "world_model_smoke"
TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW = "world_model_train_preview"
TRAINER_ARTIFACT_STAGE_WORLD_MODEL_RELEASE_CANDIDATE = "world_model_release_candidate"
DREAMER_V3_WORLD_MODEL_PRESETS = frozenset({"espresso_debug", "espresso_small", "espresso_medium"})
TRAINER_ARTIFACT_STAGES = frozenset(
    {
        TRAINER_ARTIFACT_STAGE_CONTRACT_ONLY,
        TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE,
        TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW,
        TRAINER_ARTIFACT_STAGE_WORLD_MODEL_RELEASE_CANDIDATE,
    }
)

_WORLD_MODEL_RUN_SUFFIXES = (
    "epochs",
    "batch_size",
    "learning_rate",
    "model_preset",
    "deter_dim",
    "hidden_dim",
    "stoch_size",
    "class_size",
    "action_embed_dim",
    "reward_bins",
    "unimix",
    "free_nats",
    "gradient_steps_per_epoch",
    "validation_split",
    "early_stop_patience",
    "imagination_horizon",
    "imagination_actor_hidden_dim",
    "imagination_critic_hidden_dim",
    "imagination_actor_entropy_scale",
    "pre_shot_behavior_loss_scale",
    "imagination_lambda_return",
    "imagination_discount",
    "actor_critic_train_steps",
    "actor_learning_rate",
    "critic_learning_rate",
    "imagination_batch_size",
    "actor_critic_gradient_clip_norm",
)
_WORLD_MODEL_PREVIEW_KEYS = frozenset(f"world_model_preview_{suffix}" for suffix in _WORLD_MODEL_RUN_SUFFIXES)
_WORLD_MODEL_RELEASE_KEYS = frozenset(f"world_model_release_{suffix}" for suffix in _WORLD_MODEL_RUN_SUFFIXES) | frozenset(
    {
        "world_model_release_min_train_episodes",
        "world_model_release_min_validation_episodes",
    }
)
_TRAINING_CONFIG_FIELDS = frozenset(
    {
        "format",
        "schema_version",
        "model_family",
        "artifact_stage",
        "dreamer_control_spec",
        "dreamer_pre_shot_action_spec",
        "dreamer_live_action_spec",
        "dreamer_taste_objective_spec",
        "world_model_smoke_steps",
        "seed",
        "notes",
    }
    | _WORLD_MODEL_PREVIEW_KEYS
    | _WORLD_MODEL_RELEASE_KEYS
)


def validate_training_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(config, dict):
        return ["training config must be an object"]
    _reject_unknown_fields(config, _TRAINING_CONFIG_FIELDS, errors)
    if config.get("format") != TRAINING_CONFIG_FORMAT:
        errors.append("training config format is unsupported")
    if config.get("schema_version") != TRAINING_CONFIG_SCHEMA_VERSION:
        errors.append("training config schema_version is unsupported")
    if config.get("model_family") != MODEL_FAMILY_DREAMER_V3:
        errors.append("training config model_family is unsupported")
    artifact_stage = config.get("artifact_stage")
    if artifact_stage not in TRAINER_ARTIFACT_STAGES:
        errors.append("training config artifact_stage is unsupported")
    smoke_steps = config.get("world_model_smoke_steps")
    if smoke_steps is not None and (
        artifact_stage != TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE
        or isinstance(smoke_steps, bool)
        or not isinstance(smoke_steps, int)
        or not 1 <= smoke_steps <= 20
    ):
        errors.append("training config world_model_smoke_steps is invalid")
    _validate_world_model_run_config(
        config,
        artifact_stage,
        errors,
        prefix="world_model_preview",
        keys=_WORLD_MODEL_PREVIEW_KEYS,
        required_stage=TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW,
        stage_label="world_model_train_preview",
        max_epochs=50,
        max_batch_size=128,
        max_gradient_steps_per_epoch=32,
        max_early_stop_patience=20,
        max_imagination_horizon=32,
        max_actor_critic_train_steps=128,
        max_imagination_batch_size=128,
    )
    _validate_world_model_run_config(
        config,
        artifact_stage,
        errors,
        prefix="world_model_release",
        keys=_WORLD_MODEL_RELEASE_KEYS,
        required_stage=TRAINER_ARTIFACT_STAGE_WORLD_MODEL_RELEASE_CANDIDATE,
        stage_label="world_model_release_candidate",
        max_epochs=100_000,
        max_batch_size=4096,
        max_gradient_steps_per_epoch=100_000,
        max_early_stop_patience=100_000,
        max_imagination_horizon=256,
        max_actor_critic_train_steps=1_000_000,
        max_imagination_batch_size=4096,
    )
    try:
        DreamerControlSpec.from_dict(config.get("dreamer_control_spec"))
    except ValueError as exc:
        errors.append(str(exc))
    if "dreamer_pre_shot_action_spec" not in config:
        errors.append("training config dreamer_pre_shot_action_spec is required")
    else:
        try:
            DreamerPreShotActionSpec.from_dict(config.get("dreamer_pre_shot_action_spec"))
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    if "dreamer_live_action_spec" not in config:
        errors.append("training config dreamer_live_action_spec is required")
    else:
        try:
            DreamerLiveActionSpec.from_dict(config.get("dreamer_live_action_spec"))
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    try:
        DreamerTasteObjectiveSpec.from_dict(config.get("dreamer_taste_objective_spec"))
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        errors.append("training config seed must be a uint32 integer")
    notes = config.get("notes")
    if notes is not None and (not isinstance(notes, str) or len(notes) > 500 or _has_control_chars(notes)):
        errors.append("training config notes must be a safe short string")
    return errors


def default_training_config(
    *,
    seed: int = 0,
    artifact_stage: str = TRAINER_ARTIFACT_STAGE_CONTRACT_ONLY,
) -> dict[str, Any]:
    config = {
        "format": TRAINING_CONFIG_FORMAT,
        "schema_version": TRAINING_CONFIG_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY_DREAMER_V3,
        "artifact_stage": artifact_stage,
        "dreamer_control_spec": DEFAULT_DREAMER_CONTROL_SPEC.to_dict(),
        "dreamer_pre_shot_action_spec": DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC.to_dict(),
        "dreamer_live_action_spec": DEFAULT_DREAMER_LIVE_ACTION_SPEC.to_dict(),
        "dreamer_taste_objective_spec": DEFAULT_DREAMER_TASTE_OBJECTIVE_SPEC.to_dict(),
        "seed": seed,
    }
    if artifact_stage == TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE:
        config["world_model_smoke_steps"] = 2
    if artifact_stage == TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW:
        config.update(
            {
                "world_model_preview_epochs": 3,
                "world_model_preview_batch_size": 4,
                "world_model_preview_learning_rate": 0.001,
                "world_model_preview_model_preset": "espresso_debug",
                "world_model_preview_deter_dim": 32,
                "world_model_preview_hidden_dim": 32,
                "world_model_preview_stoch_size": 4,
                "world_model_preview_class_size": 4,
                "world_model_preview_action_embed_dim": 16,
                "world_model_preview_reward_bins": 41,
                "world_model_preview_unimix": 0.01,
                "world_model_preview_free_nats": 1.0,
                "world_model_preview_gradient_steps_per_epoch": 1,
                "world_model_preview_validation_split": 0.25,
                "world_model_preview_early_stop_patience": 2,
                "world_model_preview_imagination_horizon": 3,
                "world_model_preview_imagination_actor_hidden_dim": 32,
                "world_model_preview_imagination_critic_hidden_dim": 32,
                "world_model_preview_imagination_actor_entropy_scale": 0.0003,
                "world_model_preview_pre_shot_behavior_loss_scale": 1.0,
                "world_model_preview_imagination_lambda_return": 0.95,
                "world_model_preview_imagination_discount": 0.997,
                "world_model_preview_actor_critic_train_steps": 3,
                "world_model_preview_actor_learning_rate": 0.0003,
                "world_model_preview_critic_learning_rate": 0.0003,
                "world_model_preview_imagination_batch_size": 4,
                "world_model_preview_actor_critic_gradient_clip_norm": 10.0,
            }
        )
    if artifact_stage == TRAINER_ARTIFACT_STAGE_WORLD_MODEL_RELEASE_CANDIDATE:
        config.update(
            {
                "world_model_release_epochs": 100,
                "world_model_release_batch_size": 64,
                "world_model_release_learning_rate": 0.0003,
                "world_model_release_model_preset": "espresso_small",
                "world_model_release_deter_dim": 128,
                "world_model_release_hidden_dim": 128,
                "world_model_release_stoch_size": 16,
                "world_model_release_class_size": 16,
                "world_model_release_action_embed_dim": 32,
                "world_model_release_reward_bins": 101,
                "world_model_release_unimix": 0.01,
                "world_model_release_free_nats": 1.0,
                "world_model_release_gradient_steps_per_epoch": 128,
                "world_model_release_validation_split": 0.2,
                "world_model_release_early_stop_patience": 20,
                "world_model_release_imagination_horizon": 16,
                "world_model_release_imagination_actor_hidden_dim": 128,
                "world_model_release_imagination_critic_hidden_dim": 128,
                "world_model_release_imagination_actor_entropy_scale": 0.0003,
                "world_model_release_pre_shot_behavior_loss_scale": 1.0,
                "world_model_release_imagination_lambda_return": 0.95,
                "world_model_release_imagination_discount": 0.997,
                "world_model_release_actor_critic_train_steps": 10_000,
                "world_model_release_actor_learning_rate": 0.0001,
                "world_model_release_critic_learning_rate": 0.0001,
                "world_model_release_imagination_batch_size": 64,
                "world_model_release_actor_critic_gradient_clip_norm": 10.0,
                "world_model_release_min_train_episodes": 128,
                "world_model_release_min_validation_episodes": 32,
            }
        )
    errors = validate_training_config(config)
    if errors:
        raise ValueError("; ".join(errors[:10]))
    return config


def _validate_world_model_run_config(
    config: dict[str, Any],
    artifact_stage: object,
    errors: list[str],
    *,
    prefix: str,
    keys: frozenset[str],
    required_stage: str,
    stage_label: str,
    max_epochs: int,
    max_batch_size: int,
    max_gradient_steps_per_epoch: int,
    max_early_stop_patience: int,
    max_imagination_horizon: int,
    max_actor_critic_train_steps: int,
    max_imagination_batch_size: int,
) -> None:
    has_fields = any(key in config for key in keys)
    if artifact_stage == required_stage:
        missing = sorted(key for key in keys if key not in config)
        if missing:
            errors.append(f"training config missing {stage_label} fields: {', '.join(missing[:10])}")
            return
    if not has_fields:
        return
    if artifact_stage != required_stage:
        errors.append(f"training config {prefix} fields require {stage_label}")
        return
    _require_int_range(config.get(f"{prefix}_epochs"), f"{prefix}_epochs", 1, max_epochs, errors)
    _require_int_range(config.get(f"{prefix}_batch_size"), f"{prefix}_batch_size", 1, max_batch_size, errors)
    _require_float_range(config.get(f"{prefix}_learning_rate"), f"{prefix}_learning_rate", 1e-6, 0.1, errors)
    if config.get(f"{prefix}_model_preset") not in DREAMER_V3_WORLD_MODEL_PRESETS:
        errors.append(f"training config {prefix}_model_preset is invalid")
    _require_int_range(config.get(f"{prefix}_deter_dim"), f"{prefix}_deter_dim", 8, 2048, errors)
    _require_int_range(config.get(f"{prefix}_hidden_dim"), f"{prefix}_hidden_dim", 8, 2048, errors)
    _require_int_range(config.get(f"{prefix}_stoch_size"), f"{prefix}_stoch_size", 2, 64, errors)
    _require_int_range(config.get(f"{prefix}_class_size"), f"{prefix}_class_size", 2, 128, errors)
    _require_int_range(config.get(f"{prefix}_action_embed_dim"), f"{prefix}_action_embed_dim", 4, 256, errors)
    _require_int_range(config.get(f"{prefix}_reward_bins"), f"{prefix}_reward_bins", 3, 255, errors)
    if isinstance(config.get(f"{prefix}_reward_bins"), int) and config[f"{prefix}_reward_bins"] % 2 == 0:
        errors.append(f"training config {prefix}_reward_bins must be odd")
    _require_float_range(config.get(f"{prefix}_unimix"), f"{prefix}_unimix", 0.0, 0.2, errors)
    _require_float_range(config.get(f"{prefix}_free_nats"), f"{prefix}_free_nats", 0.0, 10.0, errors)
    _require_int_range(
        config.get(f"{prefix}_gradient_steps_per_epoch"),
        f"{prefix}_gradient_steps_per_epoch",
        1,
        max_gradient_steps_per_epoch,
        errors,
    )
    _require_float_range(config.get(f"{prefix}_validation_split"), f"{prefix}_validation_split", 0.05, 0.5, errors)
    _require_int_range(
        config.get(f"{prefix}_early_stop_patience"),
        f"{prefix}_early_stop_patience",
        1,
        max_early_stop_patience,
        errors,
    )
    _require_int_range(
        config.get(f"{prefix}_imagination_horizon"),
        f"{prefix}_imagination_horizon",
        1,
        max_imagination_horizon,
        errors,
    )
    _require_int_range(
        config.get(f"{prefix}_imagination_actor_hidden_dim"),
        f"{prefix}_imagination_actor_hidden_dim",
        8,
        2048,
        errors,
    )
    _require_int_range(
        config.get(f"{prefix}_imagination_critic_hidden_dim"),
        f"{prefix}_imagination_critic_hidden_dim",
        8,
        2048,
        errors,
    )
    _require_float_range(
        config.get(f"{prefix}_imagination_actor_entropy_scale"),
        f"{prefix}_imagination_actor_entropy_scale",
        0.0,
        1.0,
        errors,
    )
    _require_float_range(
        config.get(f"{prefix}_pre_shot_behavior_loss_scale"),
        f"{prefix}_pre_shot_behavior_loss_scale",
        0.0,
        100.0,
        errors,
    )
    _require_float_range(
        config.get(f"{prefix}_imagination_lambda_return"),
        f"{prefix}_imagination_lambda_return",
        0.0,
        1.0,
        errors,
    )
    _require_float_range(
        config.get(f"{prefix}_imagination_discount"),
        f"{prefix}_imagination_discount",
        0.0,
        1.0,
        errors,
    )
    _require_int_range(
        config.get(f"{prefix}_actor_critic_train_steps"),
        f"{prefix}_actor_critic_train_steps",
        1,
        max_actor_critic_train_steps,
        errors,
    )
    _require_float_range(
        config.get(f"{prefix}_actor_learning_rate"),
        f"{prefix}_actor_learning_rate",
        1e-6,
        0.1,
        errors,
    )
    _require_float_range(
        config.get(f"{prefix}_critic_learning_rate"),
        f"{prefix}_critic_learning_rate",
        1e-6,
        0.1,
        errors,
    )
    _require_int_range(
        config.get(f"{prefix}_imagination_batch_size"),
        f"{prefix}_imagination_batch_size",
        1,
        max_imagination_batch_size,
        errors,
    )
    _require_float_range(
        config.get(f"{prefix}_actor_critic_gradient_clip_norm"),
        f"{prefix}_actor_critic_gradient_clip_norm",
        0.1,
        100.0,
        errors,
    )
    if prefix == "world_model_release":
        _require_int_range(
            config.get("world_model_release_min_train_episodes"),
            "world_model_release_min_train_episodes",
            1,
            1_000_000_000,
            errors,
        )
        _require_int_range(
            config.get("world_model_release_min_validation_episodes"),
            "world_model_release_min_validation_episodes",
            1,
            1_000_000_000,
            errors,
        )


def _require_int_range(value: object, label: str, minimum: int, maximum: int, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        errors.append(f"training config {label} is invalid")


def _require_float_range(value: object, label: str, minimum: float, maximum: float, errors: list[str]) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        errors.append(f"training config {label} is invalid")


def _reject_unknown_fields(value: dict[str, Any], allowed: frozenset[str], errors: list[str]) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        errors.append(f"training config contains unknown fields: {', '.join(unknown[:10])}")


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 32 and ch not in "\n\r\t" for ch in value) or any(ord(ch) == 127 for ch in value)

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
DREAMER_V3_WORLD_MODEL_PRESETS = frozenset({"espresso_debug", "espresso_small", "espresso_medium"})
TRAINER_ARTIFACT_STAGES = frozenset(
    {
        TRAINER_ARTIFACT_STAGE_CONTRACT_ONLY,
        TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE,
        TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW,
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
        "world_model_preview_epochs",
        "world_model_preview_batch_size",
        "world_model_preview_learning_rate",
        "world_model_preview_model_preset",
        "world_model_preview_deter_dim",
        "world_model_preview_hidden_dim",
        "world_model_preview_stoch_size",
        "world_model_preview_class_size",
        "world_model_preview_action_embed_dim",
        "world_model_preview_reward_bins",
        "world_model_preview_unimix",
        "world_model_preview_free_nats",
        "world_model_preview_gradient_steps_per_epoch",
        "world_model_preview_validation_split",
        "world_model_preview_early_stop_patience",
        "world_model_preview_imagination_horizon",
        "world_model_preview_imagination_actor_hidden_dim",
        "world_model_preview_imagination_critic_hidden_dim",
        "world_model_preview_imagination_actor_entropy_scale",
        "world_model_preview_pre_shot_behavior_loss_scale",
        "world_model_preview_imagination_lambda_return",
        "world_model_preview_imagination_discount",
        "world_model_preview_actor_critic_train_steps",
        "world_model_preview_actor_learning_rate",
        "world_model_preview_critic_learning_rate",
        "world_model_preview_imagination_batch_size",
        "world_model_preview_actor_critic_gradient_clip_norm",
        "seed",
        "notes",
    }
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
    _validate_preview_config(config, artifact_stage, errors)
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
    errors = validate_training_config(config)
    if errors:
        raise ValueError("; ".join(errors[:10]))
    return config


def _validate_preview_config(config: dict[str, Any], artifact_stage: object, errors: list[str]) -> None:
    preview_keys = {
        "world_model_preview_epochs",
        "world_model_preview_batch_size",
        "world_model_preview_learning_rate",
        "world_model_preview_model_preset",
        "world_model_preview_deter_dim",
        "world_model_preview_hidden_dim",
        "world_model_preview_stoch_size",
        "world_model_preview_class_size",
        "world_model_preview_action_embed_dim",
        "world_model_preview_reward_bins",
        "world_model_preview_unimix",
        "world_model_preview_free_nats",
        "world_model_preview_gradient_steps_per_epoch",
        "world_model_preview_validation_split",
        "world_model_preview_early_stop_patience",
        "world_model_preview_imagination_horizon",
        "world_model_preview_imagination_actor_hidden_dim",
        "world_model_preview_imagination_critic_hidden_dim",
        "world_model_preview_imagination_actor_entropy_scale",
        "world_model_preview_pre_shot_behavior_loss_scale",
        "world_model_preview_imagination_lambda_return",
        "world_model_preview_imagination_discount",
        "world_model_preview_actor_critic_train_steps",
        "world_model_preview_actor_learning_rate",
        "world_model_preview_critic_learning_rate",
        "world_model_preview_imagination_batch_size",
        "world_model_preview_actor_critic_gradient_clip_norm",
    }
    has_preview_fields = any(key in config for key in preview_keys)
    if artifact_stage == TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW:
        missing = sorted(key for key in preview_keys if key not in config)
        if missing:
            errors.append(f"training config missing preview fields: {', '.join(missing[:10])}")
            return
    if not has_preview_fields:
        return
    if artifact_stage != TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW:
        errors.append("training config world_model_preview fields require world_model_train_preview")
        return
    _require_int_range(config.get("world_model_preview_epochs"), "world_model_preview_epochs", 1, 50, errors)
    _require_int_range(config.get("world_model_preview_batch_size"), "world_model_preview_batch_size", 1, 128, errors)
    _require_float_range(config.get("world_model_preview_learning_rate"), "world_model_preview_learning_rate", 1e-6, 0.1, errors)
    if config.get("world_model_preview_model_preset") not in DREAMER_V3_WORLD_MODEL_PRESETS:
        errors.append("training config world_model_preview_model_preset is invalid")
    _require_int_range(config.get("world_model_preview_deter_dim"), "world_model_preview_deter_dim", 8, 2048, errors)
    _require_int_range(config.get("world_model_preview_hidden_dim"), "world_model_preview_hidden_dim", 8, 2048, errors)
    _require_int_range(config.get("world_model_preview_stoch_size"), "world_model_preview_stoch_size", 2, 64, errors)
    _require_int_range(config.get("world_model_preview_class_size"), "world_model_preview_class_size", 2, 128, errors)
    _require_int_range(config.get("world_model_preview_action_embed_dim"), "world_model_preview_action_embed_dim", 4, 256, errors)
    _require_int_range(config.get("world_model_preview_reward_bins"), "world_model_preview_reward_bins", 3, 255, errors)
    if isinstance(config.get("world_model_preview_reward_bins"), int) and config["world_model_preview_reward_bins"] % 2 == 0:
        errors.append("training config world_model_preview_reward_bins must be odd")
    _require_float_range(config.get("world_model_preview_unimix"), "world_model_preview_unimix", 0.0, 0.2, errors)
    _require_float_range(config.get("world_model_preview_free_nats"), "world_model_preview_free_nats", 0.0, 10.0, errors)
    _require_int_range(
        config.get("world_model_preview_gradient_steps_per_epoch"),
        "world_model_preview_gradient_steps_per_epoch",
        1,
        32,
        errors,
    )
    _require_float_range(config.get("world_model_preview_validation_split"), "world_model_preview_validation_split", 0.05, 0.5, errors)
    _require_int_range(
        config.get("world_model_preview_early_stop_patience"),
        "world_model_preview_early_stop_patience",
        1,
        20,
        errors,
    )
    _require_int_range(
        config.get("world_model_preview_imagination_horizon"),
        "world_model_preview_imagination_horizon",
        1,
        32,
        errors,
    )
    _require_int_range(
        config.get("world_model_preview_imagination_actor_hidden_dim"),
        "world_model_preview_imagination_actor_hidden_dim",
        8,
        2048,
        errors,
    )
    _require_int_range(
        config.get("world_model_preview_imagination_critic_hidden_dim"),
        "world_model_preview_imagination_critic_hidden_dim",
        8,
        2048,
        errors,
    )
    _require_float_range(
        config.get("world_model_preview_imagination_actor_entropy_scale"),
        "world_model_preview_imagination_actor_entropy_scale",
        0.0,
        1.0,
        errors,
    )
    _require_float_range(
        config.get("world_model_preview_pre_shot_behavior_loss_scale"),
        "world_model_preview_pre_shot_behavior_loss_scale",
        0.0,
        100.0,
        errors,
    )
    _require_float_range(
        config.get("world_model_preview_imagination_lambda_return"),
        "world_model_preview_imagination_lambda_return",
        0.0,
        1.0,
        errors,
    )
    _require_float_range(
        config.get("world_model_preview_imagination_discount"),
        "world_model_preview_imagination_discount",
        0.0,
        1.0,
        errors,
    )
    _require_int_range(
        config.get("world_model_preview_actor_critic_train_steps"),
        "world_model_preview_actor_critic_train_steps",
        1,
        128,
        errors,
    )
    _require_float_range(
        config.get("world_model_preview_actor_learning_rate"),
        "world_model_preview_actor_learning_rate",
        1e-6,
        0.1,
        errors,
    )
    _require_float_range(
        config.get("world_model_preview_critic_learning_rate"),
        "world_model_preview_critic_learning_rate",
        1e-6,
        0.1,
        errors,
    )
    _require_int_range(
        config.get("world_model_preview_imagination_batch_size"),
        "world_model_preview_imagination_batch_size",
        1,
        128,
        errors,
    )
    _require_float_range(
        config.get("world_model_preview_actor_critic_gradient_clip_norm"),
        "world_model_preview_actor_critic_gradient_clip_norm",
        0.1,
        100.0,
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

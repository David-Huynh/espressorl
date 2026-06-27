from __future__ import annotations

from typing import Any

from espresso_rl.domain.dreamer_control import DEFAULT_DREAMER_CONTROL_SPEC, DreamerControlSpec
from espresso_rl.domain.model_manifest import MODEL_FAMILY_DREAMER_V3

TRAINING_CONFIG_FORMAT = "espresso_rl_training_config_v1"
TRAINING_CONFIG_SCHEMA_VERSION = 1
TRAINER_AUDIT_REPORT_FORMAT = "espresso_rl_trainer_audit_report_v1"
TRAINER_ARTIFACT_STAGE_CONTRACT_ONLY = "artifact_contract_only"
TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE = "world_model_smoke"
TRAINER_ARTIFACT_STAGES = frozenset(
    {
        TRAINER_ARTIFACT_STAGE_CONTRACT_ONLY,
        TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE,
    }
)

_TRAINING_CONFIG_FIELDS = frozenset(
    {
        "format",
        "schema_version",
        "model_family",
        "artifact_stage",
        "dreamer_control_spec",
        "world_model_smoke_steps",
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
    try:
        DreamerControlSpec.from_dict(config.get("dreamer_control_spec"))
    except ValueError as exc:
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
        "seed": seed,
    }
    if artifact_stage == TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE:
        config["world_model_smoke_steps"] = 2
    errors = validate_training_config(config)
    if errors:
        raise ValueError("; ".join(errors[:10]))
    return config


def _reject_unknown_fields(value: dict[str, Any], allowed: frozenset[str], errors: list[str]) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        errors.append(f"training config contains unknown fields: {', '.join(unknown[:10])}")


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 32 and ch not in "\n\r\t" for ch in value) or any(ord(ch) == 127 for ch in value)

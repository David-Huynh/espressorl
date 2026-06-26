from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from espresso_rl.domain.dreamer_actions import DREAMER_ACTION_SCHEMA_VERSION
from espresso_rl.domain.optimization import OPTIMIZER_MODE_DREAMER_V3_SHADOW
from espresso_rl.domain.training import TRAINING_DATASET_FORMAT

MODEL_MANIFEST_FORMAT = "espresso_rl_model_manifest_v1"
MODEL_MANIFEST_SCHEMA_VERSION = 1
MODEL_FAMILY_DREAMER_V3 = "dreamer_v3"
MODEL_ARTIFACT_FORMAT_SAFETENSORS = "safetensors"
STATE_SCHEMA_VERSION = 1
ACTION_SCHEMA_VERSION = DREAMER_ACTION_SCHEMA_VERSION
REWARD_SCHEMA_VERSION = 1
RUNTIME_SCHEMA_VERSION = 1

ALLOWED_MODEL_FAMILIES = {MODEL_FAMILY_DREAMER_V3}
ALLOWED_MODEL_ARTIFACT_FORMATS = {MODEL_ARTIFACT_FORMAT_SAFETENSORS}
_HEX_CHARS = set("0123456789abcdefABCDEF")


@dataclass(frozen=True)
class ModelManifestValidation:
    verified: bool
    unavailable_reason: str | None = None
    model_family: str | None = None
    model_artifact_format: str | None = None
    model_artifact_sha256: str | None = None
    dataset_sha256: str | None = None
    dataset_manifest_sha256: str | None = None
    trainer_git_sha: str | None = None
    training_config_sha256: str | None = None
    state_schema_version: int | None = None
    action_schema_version: int | None = None
    reward_schema_version: int | None = None


def validate_model_manifest(
    manifest: Any,
    *,
    expected_model_sha256: str | None = None,
) -> ModelManifestValidation:
    if not isinstance(manifest, dict):
        return _invalid("DreamerV3 model manifest must be a JSON object.")

    if manifest.get("format") != MODEL_MANIFEST_FORMAT:
        return _invalid("DreamerV3 model manifest format is unsupported.")
    if manifest.get("schema_version") != MODEL_MANIFEST_SCHEMA_VERSION:
        return _invalid("DreamerV3 model manifest schema_version is unsupported.")

    model_family = manifest.get("model_family")
    if model_family not in ALLOWED_MODEL_FAMILIES:
        return _invalid("DreamerV3 model manifest model_family is unsupported.")

    artifact = manifest.get("model_artifact")
    if not isinstance(artifact, dict):
        return _invalid("DreamerV3 model manifest model_artifact is missing.")
    model_artifact_format = artifact.get("format")
    if model_artifact_format not in ALLOWED_MODEL_ARTIFACT_FORMATS:
        return _invalid("DreamerV3 model manifest model_artifact.format must be safetensors.")
    model_artifact_sha256 = _sha256(artifact.get("sha256"))
    if model_artifact_sha256 is None:
        return _invalid("DreamerV3 model manifest model_artifact.sha256 is invalid.")

    expected_digest = _sha256(expected_model_sha256)
    if expected_digest is not None and model_artifact_sha256 != expected_digest:
        return _invalid("DreamerV3 model manifest SHA-256 does not match configured model SHA.")

    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        return _invalid("DreamerV3 model manifest dataset is missing.")
    if dataset.get("format") != TRAINING_DATASET_FORMAT:
        return _invalid("DreamerV3 model manifest dataset format is unsupported.")
    dataset_sha256 = _sha256(dataset.get("sha256"))
    if dataset_sha256 is None:
        return _invalid("DreamerV3 model manifest dataset.sha256 is invalid.")
    dataset_manifest_sha256 = _sha256(dataset.get("manifest_sha256"))
    if dataset_manifest_sha256 is None:
        return _invalid("DreamerV3 model manifest dataset.manifest_sha256 is invalid.")

    trainer = manifest.get("trainer")
    if not isinstance(trainer, dict):
        return _invalid("DreamerV3 model manifest trainer is missing.")
    trainer_git_sha = _safe_nonempty_string(trainer.get("git_sha"), max_len=80)
    if trainer_git_sha is None:
        return _invalid("DreamerV3 model manifest trainer.git_sha is missing.")
    training_config_sha256 = _sha256(trainer.get("training_config_sha256"))
    if training_config_sha256 is None:
        return _invalid("DreamerV3 model manifest trainer.training_config_sha256 is invalid.")

    schemas = manifest.get("schemas")
    if not isinstance(schemas, dict):
        return _invalid("DreamerV3 model manifest schemas is missing.")
    state_schema_version = _int_value(schemas.get("state_schema_version"))
    action_schema_version = _int_value(schemas.get("action_schema_version"))
    reward_schema_version = _int_value(schemas.get("reward_schema_version"))
    if state_schema_version != STATE_SCHEMA_VERSION:
        return _invalid("DreamerV3 model manifest state schema is incompatible.")
    if action_schema_version != ACTION_SCHEMA_VERSION:
        return _invalid("DreamerV3 model manifest action schema is incompatible.")
    if reward_schema_version != REWARD_SCHEMA_VERSION:
        return _invalid("DreamerV3 model manifest reward schema is incompatible.")

    runtime = manifest.get("runtime_compatibility")
    if not isinstance(runtime, dict):
        return _invalid("DreamerV3 model manifest runtime_compatibility is missing.")
    if runtime.get("optimizer_mode") != OPTIMIZER_MODE_DREAMER_V3_SHADOW:
        return _invalid("DreamerV3 model manifest optimizer_mode is incompatible.")
    if _int_value(runtime.get("espresso_rl_runtime_schema_version")) != RUNTIME_SCHEMA_VERSION:
        return _invalid("DreamerV3 model manifest runtime schema is incompatible.")
    inference_ready = runtime.get("inference_ready")
    if inference_ready is False:
        return _invalid("DreamerV3 model manifest marks artifact as not inference-ready.")
    if inference_ready is not None and inference_ready is not True:
        return _invalid("DreamerV3 model manifest inference_ready is invalid.")

    return ModelManifestValidation(
        verified=True,
        model_family=model_family,
        model_artifact_format=model_artifact_format,
        model_artifact_sha256=model_artifact_sha256,
        dataset_sha256=dataset_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        trainer_git_sha=trainer_git_sha,
        training_config_sha256=training_config_sha256,
        state_schema_version=state_schema_version,
        action_schema_version=action_schema_version,
        reward_schema_version=reward_schema_version,
    )


def _invalid(reason: str) -> ModelManifestValidation:
    return ModelManifestValidation(verified=False, unavailable_reason=reason)


def _sha256(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if len(text) != 64 or any(ch not in _HEX_CHARS for ch in text):
        return None
    return text


def _safe_nonempty_string(value: object, *, max_len: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > max_len:
        return None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        return None
    return text


def _int_value(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value

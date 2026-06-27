from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from typing import Any

from espresso_rl.domain.dreamer_control import DreamerControlSpec
from espresso_rl.domain.dreamer_episodes import DREAMER_EPISODE_FORMAT, DREAMER_EPISODE_SCHEMA_VERSION
from espresso_rl.domain.model_manifest import (
    ACTION_SCHEMA_VERSION,
    MODEL_ARTIFACT_FORMAT_SAFETENSORS,
    MODEL_FAMILY_DREAMER_V3,
    MODEL_MANIFEST_FORMAT,
    MODEL_MANIFEST_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    RUNTIME_SCHEMA_VERSION,
    STATE_SCHEMA_VERSION,
)
from espresso_rl.domain.optimization import OPTIMIZER_MODE_DREAMER_V3_SHADOW
from espresso_rl.domain.trainer_artifacts import (
    TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW,
    TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE,
    TRAINER_AUDIT_REPORT_FORMAT,
    validate_training_config,
)
from espresso_rl.domain.training import TRAINING_DATASET_FORMAT, TRAINING_SCHEMA_VERSION, TRAINING_TRANSITION_FORMAT, validate_training_transition
from espresso_rl.dreamer.dataset import (
    DreamerEpisodeDatasetError,
    build_dreamer_episode_batch,
    build_dreamer_episodes_from_training_rows,
)
from espresso_rl.dreamer.reference_world_model import DreamerV3WorldModelConfig
from espresso_rl.dreamer.world_model_training import (
    FixedCadenceWorldModelTrainingError,
    WorldModelTrainPreviewConfig,
    run_fixed_cadence_world_model_train_preview,
    run_fixed_cadence_world_model_smoke_train,
)

MODEL_FILENAME = "dreamer_v3.safetensors"
MODEL_MANIFEST_FILENAME = "dreamer_v3_manifest.json"
TRAINING_CONFIG_FILENAME = "training_config.json"
CHECKSUMS_FILENAME = "checksums.txt"
AUDIT_REPORT_FILENAME = "audit_report.json"

DEFAULT_MAX_DATASET_BYTES = 8 * 1024 * 1024 * 1024
_MAX_DATASET_MANIFEST_BYTES = 256 * 1024
_MAX_TRAINING_CONFIG_BYTES = 128 * 1024
_DREAMER_TENSOR_KEYS = (
    "observations",
    "observed_profile_targets",
    "observed_profile_target_mask",
    "dynamic_actions",
    "dynamic_action_mask",
    "control_action_mask",
    "constraints",
    "decision_step_mask",
    "elapsed_seconds",
    "step_duration_seconds",
    "step_mask",
    "continuations",
    "rewards",
    "static_context",
    "terminal",
    "episode_weights",
    "source_training_row_ids",
)


@dataclass(frozen=True)
class TrainerArtifactFile:
    relative_path: str
    content_type: str
    size_bytes: int
    sha256: str
    content: bytes

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("content")
        return data


@dataclass(frozen=True)
class TrainerArtifactBuildResult:
    row_count: int
    dataset_sha256: str
    dataset_manifest_sha256: str
    training_config_sha256: str
    model_artifact_sha256: str
    model_manifest_sha256: str
    files: tuple[TrainerArtifactFile, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "dataset_sha256": self.dataset_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "training_config_sha256": self.training_config_sha256,
            "model_artifact_sha256": self.model_artifact_sha256,
            "model_manifest_sha256": self.model_manifest_sha256,
            "files": [file.to_dict() for file in self.files],
            "warnings": list(self.warnings),
        }


class TrainerArtifactError(ValueError):
    pass


def build_dreamer_trainer_artifacts(
    *,
    training_rows_jsonl: str,
    training_dataset_manifest_json: str,
    training_config_json: str,
    trainer_git_sha: str,
    created_at: int = 0,
    model_filename: str = MODEL_FILENAME,
    model_manifest_filename: str = MODEL_MANIFEST_FILENAME,
    max_dataset_bytes: int = DEFAULT_MAX_DATASET_BYTES,
) -> TrainerArtifactBuildResult:
    _validate_output_filename(model_filename, expected_name=MODEL_FILENAME, expected_suffix=".safetensors")
    _validate_output_filename(model_manifest_filename, expected_name=MODEL_MANIFEST_FILENAME, expected_suffix=".json")
    max_dataset_bytes = _positive_int(max_dataset_bytes, "max_dataset_bytes")
    trainer_git_sha = _safe_git_sha(trainer_git_sha)

    dataset_payload = training_rows_jsonl.encode("utf-8")
    dataset_manifest_payload = training_dataset_manifest_json.encode("utf-8")
    training_config_payload = training_config_json.encode("utf-8")
    _enforce_size("training_rows_jsonl", dataset_payload, max_dataset_bytes)
    _enforce_size("training_dataset_manifest_json", dataset_manifest_payload, _MAX_DATASET_MANIFEST_BYTES)
    _enforce_size("training_config_json", training_config_payload, _MAX_TRAINING_CONFIG_BYTES)

    dataset_manifest = _parse_json_object(training_dataset_manifest_json, "training dataset manifest")
    dataset_sha256 = _sha256_bytes(dataset_payload)
    dataset_manifest_sha256 = _sha256_bytes(dataset_manifest_payload)
    training_rows = _validate_dataset(training_rows_jsonl, dataset_manifest, expected_dataset_sha256=dataset_sha256)
    row_count = len(training_rows)

    training_config = _parse_json_object(training_config_json, "training config")
    config_errors = validate_training_config(training_config)
    if config_errors:
        raise TrainerArtifactError("; ".join(config_errors[:10]))
    artifact_stage = training_config["artifact_stage"]
    control_spec = DreamerControlSpec.from_dict(training_config["dreamer_control_spec"])
    dreamer_tensor_build = _dreamer_tensor_build(training_rows, control_spec=control_spec)
    dreamer_tensor_contract = dreamer_tensor_build["contract"]
    world_model_smoke = _world_model_smoke_metrics(
        dreamer_tensor_build["batch"],
        training_config=training_config,
    )
    world_model_train_preview = _world_model_train_preview_metrics(
        dreamer_tensor_build["episodes"],
        control_spec=control_spec,
        training_config=training_config,
    )
    canonical_training_config_text = _canonical_json(training_config) + "\n"
    canonical_training_config_payload = canonical_training_config_text.encode("utf-8")
    training_config_sha256 = _sha256_bytes(canonical_training_config_payload)

    model_payload = _placeholder_safetensors(
        dataset_sha256=dataset_sha256,
        training_config_sha256=training_config_sha256,
        dreamer_tensor_contract_sha256=dreamer_tensor_contract["tensor_contract_sha256"],
        world_model_smoke_sha256=_sha256_json(world_model_smoke) if world_model_smoke is not None else None,
        world_model_train_preview_sha256=(
            _sha256_json(world_model_train_preview) if world_model_train_preview is not None else None
        ),
        artifact_stage=artifact_stage,
        row_count=row_count,
        created_at=created_at,
    )
    model_sha256 = _sha256_bytes(model_payload)
    model_manifest = _model_manifest(
        model_sha256=model_sha256,
        dataset_sha256=dataset_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        training_config_sha256=training_config_sha256,
        trainer_git_sha=trainer_git_sha,
        artifact_stage=artifact_stage,
    )
    model_manifest_payload = (_canonical_json(model_manifest) + "\n").encode("utf-8")
    model_manifest_sha256 = _sha256_bytes(model_manifest_payload)

    audit_report_payload = (
        _canonical_json(
            _audit_report(
                created_at=created_at,
                row_count=row_count,
                dataset_sha256=dataset_sha256,
                dataset_manifest_sha256=dataset_manifest_sha256,
                training_config_sha256=training_config_sha256,
                model_artifact_sha256=model_sha256,
                model_manifest_sha256=model_manifest_sha256,
                trainer_git_sha=trainer_git_sha,
                dreamer_tensor_contract=dreamer_tensor_contract,
                world_model_smoke=world_model_smoke,
                world_model_train_preview=world_model_train_preview,
                artifact_stage=artifact_stage,
            )
        )
        + "\n"
    ).encode("utf-8")
    checksums_payload = _checksums_payload(
        {
            model_filename: model_payload,
            model_manifest_filename: model_manifest_payload,
            TRAINING_CONFIG_FILENAME: canonical_training_config_payload,
            AUDIT_REPORT_FILENAME: audit_report_payload,
        }
    )

    files = (
        _artifact_file(model_filename, model_payload, content_type="application/octet-stream"),
        _artifact_file(model_manifest_filename, model_manifest_payload, content_type="application/json; charset=utf-8"),
        _artifact_file(TRAINING_CONFIG_FILENAME, canonical_training_config_payload, content_type="application/json; charset=utf-8"),
        _artifact_file(AUDIT_REPORT_FILENAME, audit_report_payload, content_type="application/json; charset=utf-8"),
        _artifact_file(CHECKSUMS_FILENAME, checksums_payload, content_type="text/plain; charset=utf-8"),
    )
    return TrainerArtifactBuildResult(
        row_count=row_count,
        dataset_sha256=dataset_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        training_config_sha256=training_config_sha256,
        model_artifact_sha256=model_sha256,
        model_manifest_sha256=model_manifest_sha256,
        files=files,
        warnings=(
            f"{artifact_stage} output is a placeholder safetensors file and is not inference-ready",
            "DreamerV3 active inference and training are not implemented in this command",
        ),
    )


def _validate_dataset(
    training_rows_jsonl: str,
    dataset_manifest: dict[str, Any],
    *,
    expected_dataset_sha256: str,
) -> list[dict[str, Any]]:
    if dataset_manifest.get("format") != TRAINING_DATASET_FORMAT:
        raise TrainerArtifactError("training dataset manifest format is unsupported")
    if dataset_manifest.get("schema_version") != TRAINING_SCHEMA_VERSION:
        raise TrainerArtifactError("training dataset manifest schema_version is unsupported")
    if dataset_manifest.get("canonical_row_format") != TRAINING_TRANSITION_FORMAT:
        raise TrainerArtifactError("training dataset manifest canonical_row_format is unsupported")
    if dataset_manifest.get("dataset_sha256") != expected_dataset_sha256:
        raise TrainerArtifactError("training dataset manifest dataset_sha256 does not match training_rows.jsonl")
    if dataset_manifest.get("zero_trust", {}).get("canonical_transitions_only") is not True:
        raise TrainerArtifactError("training dataset manifest must declare canonical_transitions_only")
    if dataset_manifest.get("zero_trust", {}).get("absolute_grinder_fields_included") is not False:
        raise TrainerArtifactError("training dataset manifest must exclude absolute grinder fields")

    rows = _parse_training_rows(training_rows_jsonl)
    expected_row_count = dataset_manifest.get("row_count")
    if expected_row_count != len(rows):
        raise TrainerArtifactError("training dataset manifest row_count does not match training_rows.jsonl")
    if not rows:
        raise TrainerArtifactError("training dataset must contain at least one row")
    previous_id = 0
    for line_number, row in rows:
        errors = validate_training_transition(row)
        if errors:
            raise TrainerArtifactError(f"training row {line_number} failed validation: {'; '.join(errors[:10])}")
        row_id = int(row["training_row_id"])
        if row_id <= previous_id:
            raise TrainerArtifactError("training rows must be strictly ordered by training_row_id")
        previous_id = row_id
    return [row for _, row in rows]


def _dreamer_tensor_build(
    training_rows: list[dict[str, Any]],
    *,
    control_spec: DreamerControlSpec,
) -> dict[str, Any]:
    try:
        episodes = build_dreamer_episodes_from_training_rows(training_rows)
        batch = build_dreamer_episode_batch(episodes, control_spec=control_spec)
    except DreamerEpisodeDatasetError as exc:
        raise TrainerArtifactError(f"Dreamer tensor contract validation failed: {exc}") from exc

    lengths = [len(episode["steps"]) for episode in episodes]
    feature_names = _feature_names(batch)
    control_spec_dict = control_spec.to_dict()
    feature_layout = {
        "episode_format": DREAMER_EPISODE_FORMAT,
        "episode_schema_version": DREAMER_EPISODE_SCHEMA_VERSION,
        "feature_names": feature_names,
    }
    control_spec_sha256 = _sha256_json(control_spec_dict)
    feature_layout_sha256 = _sha256_json(feature_layout)
    tensor_contract_base = {
        "episode_format": DREAMER_EPISODE_FORMAT,
        "episode_schema_version": DREAMER_EPISODE_SCHEMA_VERSION,
        "episode_count": len(episodes),
        "episode_length_steps": {
            "min": min(lengths),
            "max": max(lengths),
            "avg": round(sum(lengths) / len(lengths), 4),
        },
        "observation_interval_ms": control_spec.observation_interval_ms,
        "decision_interval_ms": control_spec.decision_interval_ms,
        "decision_step_count": control_spec.decision_step_count,
        "tensor_shapes": _tensor_shapes(batch),
        "feature_names": feature_names,
        "feature_layout_sha256": feature_layout_sha256,
        "control_spec_sha256": control_spec_sha256,
        "control_spec": control_spec_dict,
    }
    return {
        "contract": {
            **tensor_contract_base,
            "tensor_contract_sha256": _sha256_json(tensor_contract_base),
        },
        "batch": batch,
        "episodes": episodes,
    }


def _world_model_smoke_metrics(
    batch: dict[str, Any],
    *,
    training_config: dict[str, Any],
) -> dict[str, Any] | None:
    if training_config.get("artifact_stage") != TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE:
        return None
    try:
        return run_fixed_cadence_world_model_smoke_train(
            batch,
            seed=int(training_config["seed"]),
            train_steps=int(training_config.get("world_model_smoke_steps", 2)),
        ).to_dict()
    except FixedCadenceWorldModelTrainingError as exc:
        raise TrainerArtifactError(f"Dreamer world-model smoke training failed: {exc}") from exc


def _world_model_train_preview_metrics(
    episodes: list[dict[str, Any]],
    *,
    control_spec: DreamerControlSpec,
    training_config: dict[str, Any],
) -> dict[str, Any] | None:
    if training_config.get("artifact_stage") != TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW:
        return None
    split = _split_episodes_for_preview(
        episodes,
        validation_split=float(training_config["world_model_preview_validation_split"]),
    )
    train_batch = build_dreamer_episode_batch(split["train_episodes"], control_spec=control_spec)
    validation_batch = build_dreamer_episode_batch(split["validation_episodes"], control_spec=control_spec)
    try:
        return run_fixed_cadence_world_model_train_preview(
            train_batch=train_batch,
            validation_batch=validation_batch,
            config=WorldModelTrainPreviewConfig(
                seed=int(training_config["seed"]),
                epochs=int(training_config["world_model_preview_epochs"]),
                batch_size=int(training_config["world_model_preview_batch_size"]),
                learning_rate=float(training_config["world_model_preview_learning_rate"]),
                gradient_steps_per_epoch=int(training_config["world_model_preview_gradient_steps_per_epoch"]),
                model=DreamerV3WorldModelConfig(
                    model_preset=str(training_config["world_model_preview_model_preset"]),
                    deter_dim=int(training_config["world_model_preview_deter_dim"]),
                    hidden_dim=int(training_config["world_model_preview_hidden_dim"]),
                    stoch_size=int(training_config["world_model_preview_stoch_size"]),
                    class_size=int(training_config["world_model_preview_class_size"]),
                    action_embed_dim=int(training_config["world_model_preview_action_embed_dim"]),
                    reward_bins=int(training_config["world_model_preview_reward_bins"]),
                    unimix=float(training_config["world_model_preview_unimix"]),
                    free_nats=float(training_config["world_model_preview_free_nats"]),
                    dyn_loss_scale=1.0,
                    rep_loss_scale=0.1,
                    observation_loss_scale=1.0,
                    reward_loss_scale=1.0,
                    continuation_loss_scale=1.0,
                ),
                validation_split=float(training_config["world_model_preview_validation_split"]),
                early_stop_patience=int(training_config["world_model_preview_early_stop_patience"]),
            ),
            dataset_split=split["summary"],
        ).to_dict()
    except FixedCadenceWorldModelTrainingError as exc:
        raise TrainerArtifactError(f"Dreamer world-model train preview failed: {exc}") from exc


def _split_episodes_for_preview(
    episodes: list[dict[str, Any]],
    *,
    validation_split: float,
) -> dict[str, Any]:
    if len(episodes) < 2:
        raise TrainerArtifactError("Dreamer world-model train preview requires at least two episodes")
    validation_count = max(1, min(len(episodes) - 1, round(len(episodes) * validation_split)))
    train_episodes = episodes[:-validation_count]
    validation_episodes = episodes[-validation_count:]
    summary = {
        "strategy": "sorted_context_time_tail_validation",
        "validation_split": validation_split,
        "train_episode_count": len(train_episodes),
        "validation_episode_count": len(validation_episodes),
        "train_source_training_row_ids": [int(episode["source_training_row_id"]) for episode in train_episodes],
        "validation_source_training_row_ids": [
            int(episode["source_training_row_id"]) for episode in validation_episodes
        ],
    }
    summary["dataset_split_sha256"] = _sha256_json(summary)
    return {
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "summary": summary,
    }


def _feature_names(batch: dict[str, Any]) -> dict[str, list[str]]:
    names = batch.get("feature_names")
    if not isinstance(names, dict):
        raise TrainerArtifactError("Dreamer tensor contract validation failed: missing feature names")
    return {str(key): [str(item) for item in value] for key, value in sorted(names.items())}


def _tensor_shapes(batch: dict[str, Any]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for key in _DREAMER_TENSOR_KEYS:
        tensor = batch.get(key)
        shape = getattr(tensor, "shape", None)
        if shape is None:
            raise TrainerArtifactError(f"Dreamer tensor contract validation failed: missing tensor {key}")
        shapes[key] = [int(dimension) for dimension in shape]
    return shapes


def _parse_training_rows(training_rows_jsonl: str) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(training_rows_jsonl.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainerArtifactError(f"training row {line_number} is not valid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise TrainerArtifactError(f"training row {line_number} must be an object")
        rows.append((line_number, row))
    return rows


def _model_manifest(
    *,
    model_sha256: str,
    dataset_sha256: str,
    dataset_manifest_sha256: str,
    training_config_sha256: str,
    trainer_git_sha: str,
    artifact_stage: str,
) -> dict[str, Any]:
    return {
        "format": MODEL_MANIFEST_FORMAT,
        "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY_DREAMER_V3,
        "model_artifact": {
            "format": MODEL_ARTIFACT_FORMAT_SAFETENSORS,
            "sha256": model_sha256,
        },
        "dataset": {
            "format": TRAINING_DATASET_FORMAT,
            "sha256": dataset_sha256,
            "manifest_sha256": dataset_manifest_sha256,
        },
        "trainer": {
            "git_sha": trainer_git_sha,
            "training_config_sha256": training_config_sha256,
            "artifact_stage": artifact_stage,
        },
        "schemas": {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "reward_schema_version": REWARD_SCHEMA_VERSION,
        },
        "runtime_compatibility": {
            "optimizer_mode": OPTIMIZER_MODE_DREAMER_V3_SHADOW,
            "espresso_rl_runtime_schema_version": RUNTIME_SCHEMA_VERSION,
            "inference_ready": False,
        },
    }


def _audit_report(
    *,
    created_at: int,
    row_count: int,
    dataset_sha256: str,
    dataset_manifest_sha256: str,
    training_config_sha256: str,
    model_artifact_sha256: str,
    model_manifest_sha256: str,
    trainer_git_sha: str,
    dreamer_tensor_contract: dict[str, Any],
    world_model_smoke: dict[str, Any] | None,
    world_model_train_preview: dict[str, Any] | None,
    artifact_stage: str,
) -> dict[str, Any]:
    return {
        "format": TRAINER_AUDIT_REPORT_FORMAT,
        "schema_version": 1,
        "created_at": int(created_at),
        "model_family": MODEL_FAMILY_DREAMER_V3,
        "artifact_stage": artifact_stage,
        "inference_ready": False,
        "row_count": row_count,
        "dataset_sha256": dataset_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "training_config_sha256": training_config_sha256,
        "model_artifact_sha256": model_artifact_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "trainer_git_sha": trainer_git_sha,
        "dreamer_tensor_contract": dreamer_tensor_contract,
        "world_model_smoke": world_model_smoke,
        "world_model_train_preview": world_model_train_preview,
        "zero_trust": {
            "dataset_manifest_hash_verified": True,
            "training_rows_revalidated": True,
            "dreamer_tensors_revalidated": True,
            "world_model_smoke_trained": world_model_smoke is not None,
            "world_model_train_preview_trained": world_model_train_preview is not None,
            "absolute_grinder_fields_allowed": False,
            "pickle_outputs_allowed": False,
            "runtime_inference_enabled": False,
        },
    }


def _placeholder_safetensors(
    *,
    dataset_sha256: str,
    training_config_sha256: str,
    dreamer_tensor_contract_sha256: str,
    world_model_smoke_sha256: str | None,
    world_model_train_preview_sha256: str | None,
    artifact_stage: str,
    row_count: int,
    created_at: int,
) -> bytes:
    header = {
        "__metadata__": {
            "format": "espresso_rl_placeholder_safetensors_v1",
            "model_family": MODEL_FAMILY_DREAMER_V3,
            "artifact_stage": artifact_stage,
            "inference_ready": "false",
            "dataset_sha256": dataset_sha256,
            "training_config_sha256": training_config_sha256,
            "dreamer_tensor_contract_sha256": dreamer_tensor_contract_sha256,
            "world_model_smoke_sha256": world_model_smoke_sha256 or "",
            "world_model_train_preview_sha256": world_model_train_preview_sha256 or "",
            "row_count": str(row_count),
            "created_at": str(int(created_at)),
        }
    }
    header_bytes = _canonical_json(header).encode("utf-8")
    return struct.pack("<Q", len(header_bytes)) + header_bytes


def _artifact_file(relative_path: str, content: bytes, *, content_type: str) -> TrainerArtifactFile:
    return TrainerArtifactFile(
        relative_path=relative_path,
        content_type=content_type,
        size_bytes=len(content),
        sha256=_sha256_bytes(content),
        content=content,
    )


def _checksums_payload(files: dict[str, bytes]) -> bytes:
    text = "".join(f"{_sha256_bytes(content)}  {name}\n" for name, content in sorted(files.items()))
    return text.encode("utf-8")


def _parse_json_object(text: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TrainerArtifactError(f"{label} is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise TrainerArtifactError(f"{label} must be a JSON object")
    return value


def _validate_output_filename(value: str, *, expected_name: str, expected_suffix: str) -> None:
    if value != expected_name:
        raise TrainerArtifactError(f"unsupported output filename: {value}")
    if not value.endswith(expected_suffix):
        raise TrainerArtifactError(f"output filename must end with {expected_suffix}")


def _safe_git_sha(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise TrainerArtifactError("trainer_git_sha is required")
    if len(text) > 80 or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise TrainerArtifactError("trainer_git_sha must be a safe short string")
    return text


def _enforce_size(label: str, payload: bytes, maximum: int) -> None:
    if len(payload) <= 0:
        raise TrainerArtifactError(f"{label} is empty")
    if len(payload) > maximum:
        raise TrainerArtifactError(f"{label} is too large")


def _positive_int(value: int, label: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise TrainerArtifactError(f"{label} must be positive")
    return parsed


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))

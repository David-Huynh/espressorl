from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from typing import Any

import torch

from espresso_rl.application.checkpoint_loading import load_verified_dreamer_checkpoint
from espresso_rl.domain.model_checkpoint import DreamerCheckpointCompatibility
from espresso_rl.domain.dreamer_control import DreamerControlSpec
from espresso_rl.domain.dreamer_episodes import DREAMER_EPISODE_FORMAT, DREAMER_EPISODE_SCHEMA_VERSION
from espresso_rl.domain.dreamer_live_action import DreamerLiveActionSpec
from espresso_rl.domain.dreamer_pre_shot import DreamerPreShotActionSpec
from espresso_rl.domain.dreamer_taste import DreamerTasteObjectiveSpec
from espresso_rl.domain.model_manifest import (
    ACTION_SCHEMA_VERSION,
    CHECKPOINT_ARTIFACT_FORMAT,
    CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
    CHECKPOINT_TENSOR_MANIFEST_FORMAT,
    CHECKPOINT_TENSOR_MANIFEST_SCHEMA_VERSION,
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
    TRAINER_ARTIFACT_STAGE_WORLD_MODEL_RELEASE_CANDIDATE,
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
from espresso_rl.dreamer.checkpoint_inference import (
    dreamer_batch_inference_sha256,
    materialize_verified_dreamer_checkpoint,
)
from espresso_rl.dreamer.world_model_training import (
    FixedCadenceWorldModelTrainingError,
    WorldModelReleaseCandidateConfig,
    WorldModelTrainPreviewConfig,
    run_fixed_cadence_world_model_release_candidate,
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
    "resolved_controls",
    "resolved_control_mask",
    "pre_shot_actions",
    "pre_shot_action_indexes",
    "pre_shot_action_mask",
    "pre_shot_capability_mask",
    "control_action_mask",
    "constraints",
    "decision_step_mask",
    "elapsed_seconds",
    "step_duration_seconds",
    "step_mask",
    "continuations",
    "rewards",
    "static_context",
    "taste_objective",
    "terminal",
    "context_static",
    "context_terminal",
    "context_time",
    "context_trajectory_embedding",
    "context_mask",
    "context_source_training_row_ids",
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
    pre_shot_action_spec = DreamerPreShotActionSpec.from_dict(
        training_config["dreamer_pre_shot_action_spec"]
    )
    live_action_spec = DreamerLiveActionSpec.from_dict(
        training_config["dreamer_live_action_spec"]
    )
    taste_objective_spec = DreamerTasteObjectiveSpec.from_dict(
        training_config["dreamer_taste_objective_spec"]
    )
    dreamer_tensor_build = _dreamer_tensor_build(
        training_rows,
        control_spec=control_spec,
        pre_shot_action_spec=pre_shot_action_spec,
        live_action_spec=live_action_spec,
        taste_objective_spec=taste_objective_spec,
    )
    dreamer_tensor_contract = dreamer_tensor_build["contract"]
    world_model_smoke = _world_model_smoke_metrics(
        dreamer_tensor_build["batch"],
        training_config=training_config,
    )
    world_model_train_preview_result = _world_model_train_preview_metrics(
        dreamer_tensor_build["episodes"],
        control_spec=control_spec,
        pre_shot_action_spec=pre_shot_action_spec,
        live_action_spec=live_action_spec,
        taste_objective_spec=taste_objective_spec,
        training_config=training_config,
    )
    world_model_release_candidate_result = _world_model_release_candidate_metrics(
        dreamer_tensor_build["episodes"],
        control_spec=control_spec,
        pre_shot_action_spec=pre_shot_action_spec,
        live_action_spec=live_action_spec,
        taste_objective_spec=taste_objective_spec,
        training_config=training_config,
    )
    world_model_train_preview = (
        world_model_train_preview_result["metrics"] if world_model_train_preview_result is not None else None
    )
    world_model_release_candidate = (
        world_model_release_candidate_result["metrics"] if world_model_release_candidate_result is not None else None
    )
    world_model_training_result = world_model_release_candidate_result or world_model_train_preview_result
    evaluation_report = (
        world_model_training_result["evaluation_report"] if world_model_training_result is not None else None
    )
    checkpoint_tensors = (
        world_model_training_result["checkpoint_tensors"] if world_model_training_result is not None else {}
    )
    checkpoint_architecture = (
        world_model_training_result["checkpoint_architecture"]
        if world_model_training_result is not None
        else {}
    )
    checkpoint_architecture_sha256 = _sha256_json(checkpoint_architecture)
    inference_probe_sha256 = (
        world_model_training_result["inference_probe_sha256"]
        if world_model_training_result is not None
        else None
    )
    heldout_inference_sha256 = (
        world_model_training_result["heldout_inference_sha256"]
        if world_model_training_result is not None
        else None
    )
    parity_batch = (
        world_model_training_result["parity_batch"]
        if world_model_training_result is not None
        else None
    )
    canonical_training_config_text = _canonical_json(training_config) + "\n"
    canonical_training_config_payload = canonical_training_config_text.encode("utf-8")
    training_config_sha256 = _sha256_bytes(canonical_training_config_payload)

    world_model_smoke_sha256 = _sha256_json(world_model_smoke) if world_model_smoke is not None else None
    world_model_train_preview_sha256 = (
        _sha256_json(world_model_train_preview) if world_model_train_preview is not None else None
    )
    world_model_release_candidate_sha256 = (
        _sha256_json(world_model_release_candidate) if world_model_release_candidate is not None else None
    )
    evaluation_report_sha256 = _sha256_json(evaluation_report) if evaluation_report is not None else None
    checkpoint_tensor_manifest = _checkpoint_tensor_manifest(checkpoint_tensors)
    checkpoint_tensor_manifest_sha256 = _sha256_json(checkpoint_tensor_manifest)
    checkpoint_metadata = _checkpoint_metadata(
        dataset_sha256=dataset_sha256,
        training_config_sha256=training_config_sha256,
        dreamer_tensor_contract_sha256=dreamer_tensor_contract["tensor_contract_sha256"],
        feature_layout_sha256=dreamer_tensor_contract["feature_layout_sha256"],
        control_spec_sha256=dreamer_tensor_contract["control_spec_sha256"],
        pre_shot_action_spec_sha256=dreamer_tensor_contract["pre_shot_action_spec_sha256"],
        live_action_spec_sha256=dreamer_tensor_contract["live_action_spec_sha256"],
        taste_objective_spec_sha256=dreamer_tensor_contract["taste_objective_spec_sha256"],
        checkpoint_tensor_manifest_sha256=checkpoint_tensor_manifest_sha256,
        checkpoint_architecture_sha256=checkpoint_architecture_sha256,
        inference_probe_sha256=inference_probe_sha256,
        heldout_inference_sha256=heldout_inference_sha256,
        evaluation_report_sha256=evaluation_report_sha256,
        world_model_smoke_sha256=world_model_smoke_sha256,
        world_model_train_preview_sha256=world_model_train_preview_sha256,
        world_model_release_candidate_sha256=world_model_release_candidate_sha256,
        artifact_stage=artifact_stage,
        row_count=row_count,
        created_at=created_at,
    )
    model_payload = _checkpoint_safetensors(checkpoint_tensors, metadata=checkpoint_metadata)
    model_sha256 = _sha256_bytes(model_payload)
    model_manifest = _model_manifest(
        model_sha256=model_sha256,
        dataset_sha256=dataset_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        training_config_sha256=training_config_sha256,
        trainer_git_sha=trainer_git_sha,
        artifact_stage=artifact_stage,
        checkpoint_tensor_manifest=checkpoint_tensor_manifest,
        checkpoint_tensor_manifest_sha256=checkpoint_tensor_manifest_sha256,
        checkpoint_architecture=checkpoint_architecture,
        checkpoint_architecture_sha256=checkpoint_architecture_sha256,
        inference_probe_sha256=inference_probe_sha256,
        heldout_inference_sha256=heldout_inference_sha256,
        evaluation_report_sha256=evaluation_report_sha256,
        dreamer_tensor_contract=dreamer_tensor_contract,
    )
    validate_dreamer_checkpoint_safetensors(model_payload, model_manifest)
    model_manifest_payload = (_canonical_json(model_manifest) + "\n").encode("utf-8")
    model_manifest_sha256 = _sha256_bytes(model_manifest_payload)
    if parity_batch is not None and heldout_inference_sha256 is not None:
        _validate_serialized_checkpoint_parity(
            model_payload=model_payload,
            model_manifest_payload=model_manifest_payload,
            expected_model_sha256=model_sha256,
            parity_batch=parity_batch,
            expected_heldout_sha256=heldout_inference_sha256,
            expected_pre_shot_action_spec_sha256=dreamer_tensor_contract["pre_shot_action_spec_sha256"],
            expected_live_action_spec_sha256=dreamer_tensor_contract["live_action_spec_sha256"],
            expected_taste_objective_spec_sha256=dreamer_tensor_contract["taste_objective_spec_sha256"],
        )

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
                checkpoint_tensor_manifest_sha256=checkpoint_tensor_manifest_sha256,
                checkpoint_architecture_sha256=checkpoint_architecture_sha256,
                inference_probe_sha256=inference_probe_sha256,
                heldout_inference_sha256=heldout_inference_sha256,
                evaluation_report_sha256=evaluation_report_sha256,
                world_model_smoke=world_model_smoke,
                world_model_train_preview=world_model_train_preview,
                world_model_release_candidate=world_model_release_candidate,
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
            f"{artifact_stage} output is a non-runtime safetensors checkpoint and is not inference-ready",
            "DreamerV3 active inference requires a separate explicit release artifact",
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
    pre_shot_action_spec: DreamerPreShotActionSpec,
    live_action_spec: DreamerLiveActionSpec,
    taste_objective_spec: DreamerTasteObjectiveSpec,
) -> dict[str, Any]:
    try:
        episodes = build_dreamer_episodes_from_training_rows(
            training_rows,
            pre_shot_action_spec=pre_shot_action_spec,
        )
        batch = build_dreamer_episode_batch(
            episodes,
            control_spec=control_spec,
            pre_shot_action_spec=pre_shot_action_spec,
            live_action_spec=live_action_spec,
        )
    except DreamerEpisodeDatasetError as exc:
        raise TrainerArtifactError(f"Dreamer tensor contract validation failed: {exc}") from exc

    lengths = [len(episode["steps"]) for episode in episodes]
    feature_names = _feature_names(batch)
    control_spec_dict = control_spec.to_dict()
    pre_shot_action_spec_dict = pre_shot_action_spec.to_dict()
    live_action_spec_dict = live_action_spec.to_dict()
    taste_objective_spec_dict = taste_objective_spec.to_dict()
    feature_layout = {
        "episode_format": DREAMER_EPISODE_FORMAT,
        "episode_schema_version": DREAMER_EPISODE_SCHEMA_VERSION,
        "feature_names": feature_names,
    }
    control_spec_sha256 = _sha256_json(control_spec_dict)
    pre_shot_action_spec_sha256 = _sha256_json(pre_shot_action_spec_dict)
    live_action_spec_sha256 = _sha256_json(live_action_spec_dict)
    taste_objective_spec_sha256 = _sha256_json(taste_objective_spec_dict)
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
        "context_window_size": int(batch["context_window_size"]),
        "tensor_shapes": _tensor_shapes(batch),
        "feature_names": feature_names,
        "feature_layout_sha256": feature_layout_sha256,
        "control_spec_sha256": control_spec_sha256,
        "control_spec": control_spec_dict,
        "pre_shot_action_spec_sha256": pre_shot_action_spec_sha256,
        "pre_shot_action_spec": pre_shot_action_spec_dict,
        "live_action_spec_sha256": live_action_spec_sha256,
        "live_action_spec": live_action_spec_dict,
        "taste_objective_spec_sha256": taste_objective_spec_sha256,
        "taste_objective_spec": taste_objective_spec_dict,
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
    pre_shot_action_spec: DreamerPreShotActionSpec,
    live_action_spec: DreamerLiveActionSpec,
    taste_objective_spec: DreamerTasteObjectiveSpec,
    training_config: dict[str, Any],
) -> dict[str, Any] | None:
    if training_config.get("artifact_stage") != TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW:
        return None
    split = _split_episodes_for_training(
        episodes,
        validation_split=float(training_config["world_model_preview_validation_split"]),
        stage_label="train preview",
    )
    train_batch = build_dreamer_episode_batch(
        split["train_episodes"],
        control_spec=control_spec,
        pre_shot_action_spec=pre_shot_action_spec,
        live_action_spec=live_action_spec,
    )
    validation_batch = build_dreamer_episode_batch(
        split["validation_episodes"],
        control_spec=control_spec,
        pre_shot_action_spec=pre_shot_action_spec,
        live_action_spec=live_action_spec,
    )
    try:
        result = run_fixed_cadence_world_model_train_preview(
            train_batch=train_batch,
            validation_batch=validation_batch,
            config=_world_model_run_config(
                "world_model_preview",
                training_config,
                control_spec=control_spec,
                pre_shot_action_spec=pre_shot_action_spec,
                live_action_spec=live_action_spec,
                taste_objective_spec=taste_objective_spec,
                config_type=WorldModelTrainPreviewConfig,
            ),
            dataset_split=split["summary"],
        )
        return {
            "metrics": result.to_dict(),
            "evaluation_report": result.evaluation_report,
            "checkpoint_tensors": result.checkpoint_tensors,
            "checkpoint_architecture": result.checkpoint_architecture.to_dict(),
            "inference_probe_sha256": result.inference_probe_sha256,
            "heldout_inference_sha256": result.heldout_inference_sha256,
            "parity_batch": result.parity_batch,
        }
    except FixedCadenceWorldModelTrainingError as exc:
        raise TrainerArtifactError(f"Dreamer world-model train preview failed: {exc}") from exc


def _world_model_release_candidate_metrics(
    episodes: list[dict[str, Any]],
    *,
    control_spec: DreamerControlSpec,
    pre_shot_action_spec: DreamerPreShotActionSpec,
    live_action_spec: DreamerLiveActionSpec,
    taste_objective_spec: DreamerTasteObjectiveSpec,
    training_config: dict[str, Any],
) -> dict[str, Any] | None:
    if training_config.get("artifact_stage") != TRAINER_ARTIFACT_STAGE_WORLD_MODEL_RELEASE_CANDIDATE:
        return None
    split = _split_episodes_for_training(
        episodes,
        validation_split=float(training_config["world_model_release_validation_split"]),
        stage_label="release candidate",
    )
    train_batch = build_dreamer_episode_batch(
        split["train_episodes"],
        control_spec=control_spec,
        pre_shot_action_spec=pre_shot_action_spec,
        live_action_spec=live_action_spec,
    )
    validation_batch = build_dreamer_episode_batch(
        split["validation_episodes"],
        control_spec=control_spec,
        pre_shot_action_spec=pre_shot_action_spec,
        live_action_spec=live_action_spec,
    )
    try:
        result = run_fixed_cadence_world_model_release_candidate(
            train_batch=train_batch,
            validation_batch=validation_batch,
            config=_world_model_run_config(
                "world_model_release",
                training_config,
                control_spec=control_spec,
                pre_shot_action_spec=pre_shot_action_spec,
                live_action_spec=live_action_spec,
                taste_objective_spec=taste_objective_spec,
                config_type=WorldModelReleaseCandidateConfig,
            ),
            dataset_split=split["summary"],
        )
        return {
            "metrics": result.to_dict(),
            "evaluation_report": result.evaluation_report,
            "checkpoint_tensors": result.checkpoint_tensors,
            "checkpoint_architecture": result.checkpoint_architecture.to_dict(),
            "inference_probe_sha256": result.inference_probe_sha256,
            "heldout_inference_sha256": result.heldout_inference_sha256,
            "parity_batch": result.parity_batch,
        }
    except FixedCadenceWorldModelTrainingError as exc:
        raise TrainerArtifactError(f"Dreamer world-model release candidate training failed: {exc}") from exc


def _world_model_run_config(
    prefix: str,
    training_config: dict[str, Any],
    *,
    control_spec: DreamerControlSpec,
    pre_shot_action_spec: DreamerPreShotActionSpec,
    live_action_spec: DreamerLiveActionSpec,
    taste_objective_spec: DreamerTasteObjectiveSpec,
    config_type,
):
    kwargs = {
        "seed": int(training_config["seed"]),
        "epochs": int(training_config[f"{prefix}_epochs"]),
        "batch_size": int(training_config[f"{prefix}_batch_size"]),
        "learning_rate": float(training_config[f"{prefix}_learning_rate"]),
        "gradient_steps_per_epoch": int(training_config[f"{prefix}_gradient_steps_per_epoch"]),
        "model": DreamerV3WorldModelConfig(
            model_preset=str(training_config[f"{prefix}_model_preset"]),
            deter_dim=int(training_config[f"{prefix}_deter_dim"]),
            hidden_dim=int(training_config[f"{prefix}_hidden_dim"]),
            stoch_size=int(training_config[f"{prefix}_stoch_size"]),
            class_size=int(training_config[f"{prefix}_class_size"]),
            action_embed_dim=int(training_config[f"{prefix}_action_embed_dim"]),
            reward_bins=int(training_config[f"{prefix}_reward_bins"]),
            unimix=float(training_config[f"{prefix}_unimix"]),
            free_nats=float(training_config[f"{prefix}_free_nats"]),
            dyn_loss_scale=1.0,
            rep_loss_scale=0.1,
            observation_loss_scale=1.0,
            reward_loss_scale=1.0,
            continuation_loss_scale=1.0,
        ),
        "validation_split": float(training_config[f"{prefix}_validation_split"]),
        "early_stop_patience": int(training_config[f"{prefix}_early_stop_patience"]),
        "control_spec": control_spec,
        "pre_shot_action_spec": pre_shot_action_spec,
        "live_action_spec": live_action_spec,
        "taste_objective_spec": taste_objective_spec,
        "imagination_horizon": int(training_config[f"{prefix}_imagination_horizon"]),
        "imagination_actor_hidden_dim": int(training_config[f"{prefix}_imagination_actor_hidden_dim"]),
        "imagination_critic_hidden_dim": int(training_config[f"{prefix}_imagination_critic_hidden_dim"]),
        "imagination_actor_entropy_scale": float(training_config[f"{prefix}_imagination_actor_entropy_scale"]),
        "pre_shot_behavior_loss_scale": float(training_config[f"{prefix}_pre_shot_behavior_loss_scale"]),
        "imagination_lambda_return": float(training_config[f"{prefix}_imagination_lambda_return"]),
        "imagination_discount": float(training_config[f"{prefix}_imagination_discount"]),
        "actor_critic_train_steps": int(training_config[f"{prefix}_actor_critic_train_steps"]),
        "actor_learning_rate": float(training_config[f"{prefix}_actor_learning_rate"]),
        "critic_learning_rate": float(training_config[f"{prefix}_critic_learning_rate"]),
        "imagination_batch_size": int(training_config[f"{prefix}_imagination_batch_size"]),
        "actor_critic_gradient_clip_norm": float(training_config[f"{prefix}_actor_critic_gradient_clip_norm"]),
    }
    if config_type is WorldModelReleaseCandidateConfig:
        kwargs["min_train_episodes"] = int(training_config["world_model_release_min_train_episodes"])
        kwargs["min_validation_episodes"] = int(training_config["world_model_release_min_validation_episodes"])
    return config_type(**kwargs)


def _split_episodes_for_training(
    episodes: list[dict[str, Any]],
    *,
    validation_split: float,
    stage_label: str,
) -> dict[str, Any]:
    if len(episodes) < 2:
        raise TrainerArtifactError(f"Dreamer world-model {stage_label} requires at least two episodes")
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
    checkpoint_tensor_manifest: dict[str, Any],
    checkpoint_tensor_manifest_sha256: str,
    checkpoint_architecture: dict[str, Any],
    checkpoint_architecture_sha256: str,
    inference_probe_sha256: str | None,
    heldout_inference_sha256: str | None,
    evaluation_report_sha256: str | None,
    dreamer_tensor_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": MODEL_MANIFEST_FORMAT,
        "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY_DREAMER_V3,
        "model_artifact": {
            "format": MODEL_ARTIFACT_FORMAT_SAFETENSORS,
            "sha256": model_sha256,
            "checkpoint_format": CHECKPOINT_ARTIFACT_FORMAT,
            "checkpoint_schema_version": CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
            "tensor_manifest_sha256": checkpoint_tensor_manifest_sha256,
            "architecture": checkpoint_architecture,
            "architecture_sha256": checkpoint_architecture_sha256,
            "inference_probe_sha256": inference_probe_sha256 or "",
            "heldout_inference_sha256": heldout_inference_sha256 or "",
            "evaluation_report_sha256": evaluation_report_sha256 or "",
            "tensor_manifest": checkpoint_tensor_manifest,
            "tensor_count": checkpoint_tensor_manifest["tensor_count"],
            "component_count": checkpoint_tensor_manifest["component_count"],
            "component_names": checkpoint_tensor_manifest["component_names"],
            "dreamer_tensor_contract_sha256": dreamer_tensor_contract["tensor_contract_sha256"],
            "feature_layout_sha256": dreamer_tensor_contract["feature_layout_sha256"],
            "control_spec_sha256": dreamer_tensor_contract["control_spec_sha256"],
            "pre_shot_action_spec_sha256": dreamer_tensor_contract["pre_shot_action_spec_sha256"],
            "live_action_spec_sha256": dreamer_tensor_contract["live_action_spec_sha256"],
            "taste_objective_spec_sha256": dreamer_tensor_contract["taste_objective_spec_sha256"],
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
    checkpoint_tensor_manifest_sha256: str,
    checkpoint_architecture_sha256: str,
    inference_probe_sha256: str | None,
    heldout_inference_sha256: str | None,
    evaluation_report_sha256: str | None,
    world_model_smoke: dict[str, Any] | None,
    world_model_train_preview: dict[str, Any] | None,
    world_model_release_candidate: dict[str, Any] | None,
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
        "checkpoint_tensor_manifest_sha256": checkpoint_tensor_manifest_sha256,
        "checkpoint_architecture_sha256": checkpoint_architecture_sha256,
        "inference_probe_sha256": inference_probe_sha256,
        "heldout_inference_sha256": heldout_inference_sha256,
        "evaluation_report_sha256": evaluation_report_sha256,
        "world_model_smoke": world_model_smoke,
        "world_model_train_preview": world_model_train_preview,
        "world_model_release_candidate": world_model_release_candidate,
        "zero_trust": {
            "dataset_manifest_hash_verified": True,
            "training_rows_revalidated": True,
            "dreamer_tensors_revalidated": True,
            "world_model_smoke_trained": world_model_smoke is not None,
            "world_model_train_preview_trained": world_model_train_preview is not None,
            "world_model_release_candidate_trained": world_model_release_candidate is not None,
            "checkpoint_safetensors_validated": True,
            "deterministic_inference_probe_recorded": inference_probe_sha256 is not None,
            "heldout_inference_parity_verified": heldout_inference_sha256 is not None,
            "absolute_grinder_fields_allowed": False,
            "pickle_outputs_allowed": False,
            "explicit_release_required": world_model_release_candidate is not None,
            "runtime_inference_enabled": False,
        },
    }


def _checkpoint_metadata(
    *,
    dataset_sha256: str,
    training_config_sha256: str,
    dreamer_tensor_contract_sha256: str,
    feature_layout_sha256: str,
    control_spec_sha256: str,
    pre_shot_action_spec_sha256: str,
    live_action_spec_sha256: str,
    taste_objective_spec_sha256: str,
    checkpoint_tensor_manifest_sha256: str,
    checkpoint_architecture_sha256: str,
    inference_probe_sha256: str | None,
    heldout_inference_sha256: str | None,
    evaluation_report_sha256: str | None,
    world_model_smoke_sha256: str | None,
    world_model_train_preview_sha256: str | None,
    world_model_release_candidate_sha256: str | None,
    artifact_stage: str,
    row_count: int,
    created_at: int,
) -> dict[str, str]:
    return {
        "format": CHECKPOINT_ARTIFACT_FORMAT,
        "schema_version": str(CHECKPOINT_ARTIFACT_SCHEMA_VERSION),
        "model_family": MODEL_FAMILY_DREAMER_V3,
        "artifact_stage": artifact_stage,
        "inference_ready": "false",
        "dataset_sha256": dataset_sha256,
        "training_config_sha256": training_config_sha256,
        "dreamer_tensor_contract_sha256": dreamer_tensor_contract_sha256,
        "feature_layout_sha256": feature_layout_sha256,
        "control_spec_sha256": control_spec_sha256,
        "pre_shot_action_spec_sha256": pre_shot_action_spec_sha256,
        "live_action_spec_sha256": live_action_spec_sha256,
        "taste_objective_spec_sha256": taste_objective_spec_sha256,
        "tensor_manifest_sha256": checkpoint_tensor_manifest_sha256,
        "architecture_sha256": checkpoint_architecture_sha256,
        "inference_probe_sha256": inference_probe_sha256 or "",
        "heldout_inference_sha256": heldout_inference_sha256 or "",
        "evaluation_report_sha256": evaluation_report_sha256 or "",
        "world_model_smoke_sha256": world_model_smoke_sha256 or "",
        "world_model_train_preview_sha256": world_model_train_preview_sha256 or "",
        "world_model_release_candidate_sha256": world_model_release_candidate_sha256 or "",
        "row_count": str(row_count),
        "created_at": str(int(created_at)),
    }


def _checkpoint_safetensors(tensors: dict[str, torch.Tensor], *, metadata: dict[str, str]) -> bytes:
    header: dict[str, Any] = {"__metadata__": {key: str(value) for key, value in sorted(metadata.items())}}
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        tensor = _checkpoint_tensor(tensors[name])
        raw = _tensor_bytes(tensor)
        next_offset = offset + len(raw)
        header[name] = {
            "dtype": "F32",
            "shape": [int(dimension) for dimension in tensor.shape],
            "data_offsets": [offset, next_offset],
        }
        chunks.append(raw)
        offset = next_offset
    header_bytes = _canonical_json(header).encode("utf-8")
    return struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(chunks)


def _checkpoint_tensor_manifest(tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    components: dict[str, dict[str, int]] = {}
    tensor_entries: dict[str, dict[str, Any]] = {}
    for name in sorted(tensors):
        component_name = name.split(".", 1)[0]
        tensor = _checkpoint_tensor(tensors[name])
        raw = _tensor_bytes(tensor)
        entry = {
            "component": component_name,
            "dtype": "F32",
            "shape": [int(dimension) for dimension in tensor.shape],
            "element_count": int(tensor.numel()),
            "sha256": _sha256_bytes(raw),
        }
        tensor_entries[name] = entry
        component = components.setdefault(component_name, {"tensor_count": 0, "element_count": 0})
        component["tensor_count"] += 1
        component["element_count"] += entry["element_count"]
    return {
        "format": CHECKPOINT_TENSOR_MANIFEST_FORMAT,
        "schema_version": CHECKPOINT_TENSOR_MANIFEST_SCHEMA_VERSION,
        "tensor_count": len(tensor_entries),
        "component_count": len(components),
        "component_names": sorted(components),
        "components": {key: components[key] for key in sorted(components)},
        "tensors": tensor_entries,
    }


def validate_dreamer_checkpoint_safetensors(payload: bytes, model_manifest: dict[str, Any]) -> None:
    artifact = model_manifest.get("model_artifact")
    if not isinstance(artifact, dict):
        raise TrainerArtifactError("checkpoint manifest is missing model_artifact")
    if artifact.get("format") != MODEL_ARTIFACT_FORMAT_SAFETENSORS:
        raise TrainerArtifactError("checkpoint artifact format must be safetensors")
    if artifact.get("sha256") != _sha256_bytes(payload):
        raise TrainerArtifactError("checkpoint artifact sha256 does not match payload")
    if artifact.get("checkpoint_format") != CHECKPOINT_ARTIFACT_FORMAT:
        raise TrainerArtifactError("checkpoint artifact format metadata is unsupported")
    if artifact.get("checkpoint_schema_version") != CHECKPOINT_ARTIFACT_SCHEMA_VERSION:
        raise TrainerArtifactError("checkpoint artifact schema version is unsupported")
    tensor_manifest = artifact.get("tensor_manifest")
    if not isinstance(tensor_manifest, dict):
        raise TrainerArtifactError("checkpoint tensor manifest is missing")
    expected_manifest_sha = artifact.get("tensor_manifest_sha256")
    if expected_manifest_sha != _sha256_json(tensor_manifest):
        raise TrainerArtifactError("checkpoint tensor manifest hash does not match")

    header, data = _safetensors_header(payload)
    data_length = len(data)
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise TrainerArtifactError("checkpoint safetensors metadata is missing")
    _require_metadata(metadata, "format", CHECKPOINT_ARTIFACT_FORMAT)
    _require_metadata(metadata, "schema_version", str(CHECKPOINT_ARTIFACT_SCHEMA_VERSION))
    _require_metadata(metadata, "model_family", MODEL_FAMILY_DREAMER_V3)
    _require_metadata(metadata, "inference_ready", "false")
    _require_metadata(metadata, "tensor_manifest_sha256", expected_manifest_sha)
    _require_metadata(metadata, "evaluation_report_sha256", artifact.get("evaluation_report_sha256", ""))
    architecture = artifact.get("architecture")
    if not isinstance(architecture, dict) or artifact.get("architecture_sha256") != _sha256_json(architecture):
        raise TrainerArtifactError("checkpoint runtime architecture hash does not match")
    for key in (
        "dreamer_tensor_contract_sha256",
        "feature_layout_sha256",
        "control_spec_sha256",
        "pre_shot_action_spec_sha256",
        "live_action_spec_sha256",
        "taste_objective_spec_sha256",
        "architecture_sha256",
        "inference_probe_sha256",
        "heldout_inference_sha256",
    ):
        _require_metadata(metadata, key, artifact.get(key))

    tensors = tensor_manifest.get("tensors")
    if not isinstance(tensors, dict):
        raise TrainerArtifactError("checkpoint tensor manifest tensors must be an object")
    expected_names = sorted(tensors)
    actual_names = sorted(key for key in header if key != "__metadata__")
    if actual_names != expected_names:
        raise TrainerArtifactError("checkpoint safetensors tensor names do not match manifest")
    if artifact.get("tensor_count") != len(expected_names) or tensor_manifest.get("tensor_count") != len(expected_names):
        raise TrainerArtifactError("checkpoint tensor count does not match manifest")
    if artifact.get("component_names") != tensor_manifest.get("component_names"):
        raise TrainerArtifactError("checkpoint component names do not match manifest")
    for name in expected_names:
        expected = tensors[name]
        actual = header[name]
        if actual.get("dtype") != expected.get("dtype"):
            raise TrainerArtifactError(f"checkpoint tensor {name} dtype does not match manifest")
        if actual.get("shape") != expected.get("shape"):
            raise TrainerArtifactError(f"checkpoint tensor {name} shape does not match manifest")
        offsets = actual.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= data_length
        ):
            raise TrainerArtifactError(f"checkpoint tensor {name} data offsets are invalid")
        tensor_bytes = data[offsets[0] : offsets[1]]
        if _sha256_bytes(tensor_bytes) != expected.get("sha256"):
            raise TrainerArtifactError(f"checkpoint tensor {name} sha256 does not match manifest")


def _checkpoint_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TrainerArtifactError("checkpoint tensors must be torch tensors")
    if not torch.isfinite(tensor.detach()).all():
        raise TrainerArtifactError("checkpoint tensor contains non-finite values")
    return tensor.detach().cpu().contiguous().to(dtype=torch.float32)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.numpy().tobytes(order="C")


def _safetensors_header(payload: bytes) -> tuple[dict[str, Any], bytes]:
    if not isinstance(payload, bytes) or len(payload) < 8:
        raise TrainerArtifactError("checkpoint safetensors payload is truncated")
    header_length = struct.unpack("<Q", payload[:8])[0]
    if header_length <= 0 or 8 + header_length > len(payload):
        raise TrainerArtifactError("checkpoint safetensors header length is invalid")
    try:
        header = json.loads(payload[8 : 8 + header_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainerArtifactError("checkpoint safetensors header is not valid UTF-8 JSON") from exc
    if not isinstance(header, dict):
        raise TrainerArtifactError("checkpoint safetensors header must be a JSON object")
    return header, payload[8 + header_length :]


def _require_metadata(metadata: dict[str, Any], key: str, expected: object) -> None:
    if metadata.get(key) != str(expected):
        raise TrainerArtifactError(f"checkpoint safetensors metadata {key} does not match manifest")


def _validate_serialized_checkpoint_parity(
    *,
    model_payload: bytes,
    model_manifest_payload: bytes,
    expected_model_sha256: str,
    parity_batch: dict[str, torch.Tensor],
    expected_heldout_sha256: str,
    expected_pre_shot_action_spec_sha256: str,
    expected_live_action_spec_sha256: str,
    expected_taste_objective_spec_sha256: str,
) -> None:
    class MemoryModelStore:
        def read_bytes(self, reference: str, *, max_bytes: int) -> bytes:
            payload = model_payload if reference == MODEL_FILENAME else model_manifest_payload
            if len(payload) > max_bytes:
                raise ValueError("checkpoint parity artifact exceeds limit")
            return payload

    try:
        checkpoint = load_verified_dreamer_checkpoint(
            MemoryModelStore(),
            artifact_reference=MODEL_FILENAME,
            manifest_reference=MODEL_MANIFEST_FILENAME,
            expected_artifact_sha256=expected_model_sha256,
            compatibility=DreamerCheckpointCompatibility(
                pre_shot_action_spec_sha256=expected_pre_shot_action_spec_sha256,
                live_action_spec_sha256=expected_live_action_spec_sha256,
                taste_objective_spec_sha256=expected_taste_objective_spec_sha256,
            ),
        )
        models = materialize_verified_dreamer_checkpoint(checkpoint)
        actual_heldout_sha256 = dreamer_batch_inference_sha256(
            world_model=models.world_model,
            context_encoder=models.context_encoder,
            actor=models.actor,
            critic=models.critic,
            batch=parity_batch,
        )
    except ValueError as exc:
        raise TrainerArtifactError(f"serialized checkpoint parity validation failed: {exc}") from exc
    if actual_heldout_sha256 != expected_heldout_sha256:
        raise TrainerArtifactError("serialized checkpoint heldout inference parity does not match")


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

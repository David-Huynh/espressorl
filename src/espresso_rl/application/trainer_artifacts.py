from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from typing import Any

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
    TRAINER_ARTIFACT_STAGE_CONTRACT_ONLY,
    TRAINER_AUDIT_REPORT_FORMAT,
    validate_training_config,
)
from espresso_rl.domain.training import TRAINING_DATASET_FORMAT, TRAINING_SCHEMA_VERSION, TRAINING_TRANSITION_FORMAT, validate_training_transition

MODEL_FILENAME = "dreamer_v3.safetensors"
MODEL_MANIFEST_FILENAME = "dreamer_v3_manifest.json"
TRAINING_CONFIG_FILENAME = "training_config.json"
CHECKSUMS_FILENAME = "checksums.txt"
AUDIT_REPORT_FILENAME = "audit_report.json"

DEFAULT_MAX_DATASET_BYTES = 8 * 1024 * 1024 * 1024
_MAX_DATASET_MANIFEST_BYTES = 256 * 1024
_MAX_TRAINING_CONFIG_BYTES = 128 * 1024


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
    row_count = _validate_dataset(training_rows_jsonl, dataset_manifest, expected_dataset_sha256=dataset_sha256)

    training_config = _parse_json_object(training_config_json, "training config")
    config_errors = validate_training_config(training_config)
    if config_errors:
        raise TrainerArtifactError("; ".join(config_errors[:10]))
    canonical_training_config_text = _canonical_json(training_config) + "\n"
    canonical_training_config_payload = canonical_training_config_text.encode("utf-8")
    training_config_sha256 = _sha256_bytes(canonical_training_config_payload)

    model_payload = _placeholder_safetensors(
        dataset_sha256=dataset_sha256,
        training_config_sha256=training_config_sha256,
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
            "artifact_contract_only output is a placeholder safetensors file and is not inference-ready",
            "DreamerV3 active inference and training are not implemented in this command",
        ),
    )


def _validate_dataset(
    training_rows_jsonl: str,
    dataset_manifest: dict[str, Any],
    *,
    expected_dataset_sha256: str,
) -> int:
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
    return len(rows)


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
            "artifact_stage": TRAINER_ARTIFACT_STAGE_CONTRACT_ONLY,
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
) -> dict[str, Any]:
    return {
        "format": TRAINER_AUDIT_REPORT_FORMAT,
        "schema_version": 1,
        "created_at": int(created_at),
        "model_family": MODEL_FAMILY_DREAMER_V3,
        "artifact_stage": TRAINER_ARTIFACT_STAGE_CONTRACT_ONLY,
        "inference_ready": False,
        "row_count": row_count,
        "dataset_sha256": dataset_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "training_config_sha256": training_config_sha256,
        "model_artifact_sha256": model_artifact_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "trainer_git_sha": trainer_git_sha,
        "zero_trust": {
            "dataset_manifest_hash_verified": True,
            "training_rows_revalidated": True,
            "absolute_grinder_fields_allowed": False,
            "pickle_outputs_allowed": False,
            "runtime_inference_enabled": False,
        },
    }


def _placeholder_safetensors(
    *,
    dataset_sha256: str,
    training_config_sha256: str,
    row_count: int,
    created_at: int,
) -> bytes:
    header = {
        "__metadata__": {
            "format": "espresso_rl_placeholder_safetensors_v1",
            "model_family": MODEL_FAMILY_DREAMER_V3,
            "artifact_stage": TRAINER_ARTIFACT_STAGE_CONTRACT_ONLY,
            "inference_ready": "false",
            "dataset_sha256": dataset_sha256,
            "training_config_sha256": training_config_sha256,
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

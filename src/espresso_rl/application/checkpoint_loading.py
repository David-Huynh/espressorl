from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from espresso_rl.domain.model_checkpoint import (
    DREAMER_CHECKPOINT_ARCHITECTURE_FORMAT,
    DREAMER_CHECKPOINT_ARCHITECTURE_SCHEMA_VERSION,
    DreamerCheckpointArchitecture,
    DreamerCheckpointCompatibility,
    DreamerCheckpointTensor,
    DreamerContextEncoderArchitecture,
    DreamerImaginationArchitecture,
    DreamerWorldModelArchitecture,
    VerifiedDreamerCheckpoint,
)
from espresso_rl.domain.dreamer_control import DreamerControlSpec
from espresso_rl.domain.model_manifest import (
    CHECKPOINT_ARTIFACT_FORMAT,
    CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
    MODEL_ARTIFACT_FORMAT_SAFETENSORS,
    MODEL_FAMILY_DREAMER_V3,
    validate_model_manifest,
)
from espresso_rl.domain.trainer_artifacts import (
    TRAINER_ARTIFACT_STAGE_CONTRACT_ONLY,
    TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE,
    TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW,
)
from espresso_rl.ports.model_store import ModelArtifactStore

DEFAULT_MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
MAX_MODEL_MANIFEST_BYTES = 256 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 8 * 1024 * 1024

_CHECKPOINT_STAGES = frozenset(
    {
        TRAINER_ARTIFACT_STAGE_CONTRACT_ONLY,
        TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE,
        TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW,
    }
)
_PREVIEW_COMPONENTS = ("actor", "context_encoder", "critic", "world_model")
_SAFE_TENSOR_NAME = re.compile(r"^[A-Za-z0-9_.]{1,200}$")
_HEX_CHARS = frozenset("0123456789abcdef")
_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "schema_version",
        "model_family",
        "model_artifact",
        "dataset",
        "trainer",
        "schemas",
        "runtime_compatibility",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "format",
        "sha256",
        "checkpoint_format",
        "checkpoint_schema_version",
        "tensor_manifest_sha256",
        "evaluation_report_sha256",
        "tensor_manifest",
        "tensor_count",
        "component_count",
        "component_names",
        "dreamer_tensor_contract_sha256",
        "feature_layout_sha256",
        "control_spec_sha256",
        "architecture",
        "architecture_sha256",
        "inference_probe_sha256",
        "heldout_inference_sha256",
    }
)
_METADATA_FIELDS = frozenset(
    {
        "format",
        "schema_version",
        "model_family",
        "artifact_stage",
        "inference_ready",
        "dataset_sha256",
        "training_config_sha256",
        "dreamer_tensor_contract_sha256",
        "feature_layout_sha256",
        "control_spec_sha256",
        "tensor_manifest_sha256",
        "architecture_sha256",
        "inference_probe_sha256",
        "heldout_inference_sha256",
        "evaluation_report_sha256",
        "world_model_smoke_sha256",
        "world_model_train_preview_sha256",
        "row_count",
        "created_at",
    }
)


class CheckpointLoadError(ValueError):
    pass


def load_verified_dreamer_checkpoint(
    store: ModelArtifactStore,
    *,
    artifact_reference: str,
    manifest_reference: str,
    expected_artifact_sha256: str,
    compatibility: DreamerCheckpointCompatibility | None = None,
    max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
) -> VerifiedDreamerCheckpoint:
    """Authenticate and inspect a checkpoint without constructing executable model objects."""

    expected_digest = _require_sha256(expected_artifact_sha256, "expected checkpoint artifact SHA-256")
    max_checkpoint_bytes = _positive_int(max_checkpoint_bytes, "max_checkpoint_bytes")
    manifest_payload = _read_store_bytes(
        store,
        manifest_reference,
        max_bytes=MAX_MODEL_MANIFEST_BYTES,
        label="checkpoint manifest",
    )
    artifact_payload = _read_store_bytes(
        store,
        artifact_reference,
        max_bytes=max_checkpoint_bytes,
        label="checkpoint artifact",
    )

    artifact_sha256 = _sha256_bytes(artifact_payload)
    if artifact_sha256 != expected_digest:
        raise CheckpointLoadError("checkpoint artifact SHA-256 does not match the configured digest")

    manifest = _parse_json_object(manifest_payload, "checkpoint manifest")
    _require_exact_fields(manifest, _MANIFEST_FIELDS, "checkpoint manifest")
    validation = validate_model_manifest(
        manifest,
        expected_model_sha256=expected_digest,
        require_checkpoint_metadata=True,
        require_inference_ready=False,
    )
    if not validation.verified:
        raise CheckpointLoadError(validation.unavailable_reason or "checkpoint manifest is invalid")

    artifact = _require_object(manifest.get("model_artifact"), "checkpoint manifest model_artifact")
    _require_exact_fields(artifact, _ARTIFACT_FIELDS, "checkpoint manifest model_artifact")
    dataset = _require_object(manifest.get("dataset"), "checkpoint manifest dataset")
    _require_exact_fields(dataset, frozenset({"format", "sha256", "manifest_sha256"}), "checkpoint manifest dataset")
    trainer = _require_object(manifest.get("trainer"), "checkpoint manifest trainer")
    _require_exact_fields(
        trainer,
        frozenset({"git_sha", "training_config_sha256", "artifact_stage"}),
        "checkpoint manifest trainer",
    )
    schemas = _require_object(manifest.get("schemas"), "checkpoint manifest schemas")
    _require_exact_fields(
        schemas,
        frozenset({"state_schema_version", "action_schema_version", "reward_schema_version"}),
        "checkpoint manifest schemas",
    )
    runtime = _require_object(manifest.get("runtime_compatibility"), "checkpoint manifest runtime_compatibility")
    _require_exact_fields(
        runtime,
        frozenset({"optimizer_mode", "espresso_rl_runtime_schema_version", "inference_ready"}),
        "checkpoint manifest runtime_compatibility",
    )

    tensor_manifest = _require_object(artifact.get("tensor_manifest"), "checkpoint tensor manifest")
    expected_tensor_manifest_sha256 = _require_sha256(
        artifact.get("tensor_manifest_sha256"),
        "checkpoint tensor manifest SHA-256",
    )
    if _sha256_json(tensor_manifest) != expected_tensor_manifest_sha256:
        raise CheckpointLoadError("checkpoint tensor manifest SHA-256 does not match its content")

    header, data_start = _parse_safetensors_header(artifact_payload)
    metadata = _require_object(header.get("__metadata__"), "checkpoint safetensors metadata")
    _validate_metadata(metadata, manifest=manifest, tensor_manifest_sha256=expected_tensor_manifest_sha256)
    artifact_stage = str(metadata["artifact_stage"])
    tensors = _validate_tensors(
        header,
        tensor_manifest=tensor_manifest,
        artifact=artifact,
        artifact_payload=artifact_payload,
        data_start=data_start,
        artifact_stage=artifact_stage,
    )
    architecture_payload = _require_object(artifact.get("architecture"), "checkpoint runtime architecture")
    architecture_sha256 = _require_sha256(
        artifact.get("architecture_sha256"),
        "checkpoint runtime architecture SHA-256",
    )
    if _sha256_json(architecture_payload) != architecture_sha256:
        raise CheckpointLoadError("checkpoint runtime architecture SHA-256 does not match its content")
    architecture = _parse_architecture(architecture_payload) if tensors else None
    if architecture is not None and _sha256_json(architecture.control_spec.to_dict()) != artifact.get(
        "control_spec_sha256"
    ):
        raise CheckpointLoadError("checkpoint runtime control spec does not match control_spec_sha256")
    inference_probe_sha256 = _optional_sha256(
        artifact.get("inference_probe_sha256"),
        "checkpoint inference probe SHA-256",
    )
    if tensors and inference_probe_sha256 is None:
        raise CheckpointLoadError("checkpoint inference probe SHA-256 is missing")
    heldout_inference_sha256 = _optional_sha256(
        artifact.get("heldout_inference_sha256"),
        "checkpoint heldout inference SHA-256",
    )
    if tensors and heldout_inference_sha256 is None:
        raise CheckpointLoadError("checkpoint heldout inference SHA-256 is missing")
    _validate_compatibility(artifact, compatibility or DreamerCheckpointCompatibility())

    return VerifiedDreamerCheckpoint(
        artifact_reference=artifact_reference,
        manifest_reference=manifest_reference,
        artifact_sha256=artifact_sha256,
        manifest_sha256=_sha256_bytes(manifest_payload),
        dataset_sha256=_require_sha256(dataset.get("sha256"), "checkpoint dataset SHA-256"),
        dataset_manifest_sha256=_require_sha256(
            dataset.get("manifest_sha256"),
            "checkpoint dataset manifest SHA-256",
        ),
        training_config_sha256=_require_sha256(
            trainer.get("training_config_sha256"),
            "checkpoint training config SHA-256",
        ),
        tensor_contract_sha256=_require_sha256(
            artifact.get("dreamer_tensor_contract_sha256"),
            "checkpoint tensor contract SHA-256",
        ),
        feature_layout_sha256=_require_sha256(
            artifact.get("feature_layout_sha256"),
            "checkpoint feature layout SHA-256",
        ),
        control_spec_sha256=_require_sha256(
            artifact.get("control_spec_sha256"),
            "checkpoint control spec SHA-256",
        ),
        evaluation_report_sha256=_optional_sha256(
            artifact.get("evaluation_report_sha256"),
            "checkpoint evaluation report SHA-256",
        ),
        architecture_sha256=architecture_sha256,
        inference_probe_sha256=inference_probe_sha256,
        heldout_inference_sha256=heldout_inference_sha256,
        architecture=architecture,
        artifact_stage=artifact_stage,
        inference_ready=bool(validation.inference_ready),
        tensors=tensors,
        payload=artifact_payload,
    )


def _read_store_bytes(store: ModelArtifactStore, reference: str, *, max_bytes: int, label: str) -> bytes:
    if not isinstance(reference, str) or not reference.strip():
        raise CheckpointLoadError(f"{label} reference must be non-empty")
    try:
        payload = store.read_bytes(reference, max_bytes=max_bytes)
    except (OSError, ValueError) as exc:
        raise CheckpointLoadError(f"{label} could not be read: {exc}") from exc
    if not isinstance(payload, bytes):
        raise CheckpointLoadError(f"{label} store returned a non-bytes payload")
    if not payload:
        raise CheckpointLoadError(f"{label} is empty")
    if len(payload) > max_bytes:
        raise CheckpointLoadError(f"{label} exceeds the configured size limit")
    return payload


def _parse_json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: _reject_json_constant(constant),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, CheckpointLoadError) as exc:
        if isinstance(exc, CheckpointLoadError):
            raise
        raise CheckpointLoadError(f"{label} is not valid strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CheckpointLoadError(f"{label} must be a JSON object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointLoadError(f"JSON object contains duplicate field {key}")
        result[key] = value
    return result


def _reject_json_constant(constant: str) -> None:
    raise CheckpointLoadError(f"JSON constant {constant} is not allowed")


def _parse_safetensors_header(payload: bytes) -> tuple[dict[str, Any], int]:
    if len(payload) < 10:
        raise CheckpointLoadError("checkpoint safetensors payload is truncated")
    header_length = int.from_bytes(payload[:8], "little")
    if header_length < 2 or header_length > MAX_SAFETENSORS_HEADER_BYTES:
        raise CheckpointLoadError("checkpoint safetensors header length is invalid")
    data_start = 8 + header_length
    if data_start > len(payload):
        raise CheckpointLoadError("checkpoint safetensors header is truncated")
    header = _parse_json_object(payload[8:data_start], "checkpoint safetensors header")
    return header, data_start


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    manifest: dict[str, Any],
    tensor_manifest_sha256: str,
) -> None:
    _require_exact_fields(metadata, _METADATA_FIELDS, "checkpoint safetensors metadata")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()):
        raise CheckpointLoadError("checkpoint safetensors metadata keys and values must be strings")

    artifact = manifest["model_artifact"]
    dataset = manifest["dataset"]
    trainer = manifest["trainer"]
    expected = {
        "format": CHECKPOINT_ARTIFACT_FORMAT,
        "schema_version": str(CHECKPOINT_ARTIFACT_SCHEMA_VERSION),
        "model_family": MODEL_FAMILY_DREAMER_V3,
        "inference_ready": "true" if manifest["runtime_compatibility"]["inference_ready"] else "false",
        "dataset_sha256": dataset["sha256"],
        "training_config_sha256": trainer["training_config_sha256"],
        "dreamer_tensor_contract_sha256": artifact["dreamer_tensor_contract_sha256"],
        "feature_layout_sha256": artifact["feature_layout_sha256"],
        "control_spec_sha256": artifact["control_spec_sha256"],
        "tensor_manifest_sha256": tensor_manifest_sha256,
        "architecture_sha256": artifact["architecture_sha256"],
        "inference_probe_sha256": artifact["inference_probe_sha256"],
        "heldout_inference_sha256": artifact["heldout_inference_sha256"],
        "evaluation_report_sha256": artifact["evaluation_report_sha256"],
        "artifact_stage": trainer["artifact_stage"],
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise CheckpointLoadError(f"checkpoint safetensors metadata {key} does not match manifest")
    if metadata["artifact_stage"] not in _CHECKPOINT_STAGES:
        raise CheckpointLoadError("checkpoint artifact stage is unsupported")
    _nonnegative_decimal(metadata["row_count"], "checkpoint metadata row_count")
    _nonnegative_decimal(metadata["created_at"], "checkpoint metadata created_at")
    for key in ("world_model_smoke_sha256", "world_model_train_preview_sha256"):
        _optional_sha256(metadata[key], f"checkpoint metadata {key}")
    if metadata["artifact_stage"] == TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW:
        _require_sha256(metadata["evaluation_report_sha256"], "checkpoint evaluation report SHA-256")


def _validate_tensors(
    header: dict[str, Any],
    *,
    tensor_manifest: dict[str, Any],
    artifact: dict[str, Any],
    artifact_payload: bytes,
    data_start: int,
    artifact_stage: str,
) -> tuple[DreamerCheckpointTensor, ...]:
    _require_exact_fields(
        tensor_manifest,
        frozenset({"format", "schema_version", "tensor_count", "component_count", "component_names", "components", "tensors"}),
        "checkpoint tensor manifest",
    )
    tensor_entries = _require_object(tensor_manifest.get("tensors"), "checkpoint tensor manifest tensors")
    components = _require_object(tensor_manifest.get("components"), "checkpoint tensor manifest components")
    expected_names = sorted(tensor_entries)
    actual_names = sorted(name for name in header if name != "__metadata__")
    if actual_names != expected_names:
        raise CheckpointLoadError("checkpoint safetensors tensor names do not match manifest")

    expected_count = len(expected_names)
    if tensor_manifest.get("tensor_count") != expected_count or artifact.get("tensor_count") != expected_count:
        raise CheckpointLoadError("checkpoint tensor count does not match manifest")
    component_names = sorted(components)
    if tensor_manifest.get("component_names") != component_names or artifact.get("component_names") != component_names:
        raise CheckpointLoadError("checkpoint component names do not match manifest")
    if tensor_manifest.get("component_count") != len(component_names) or artifact.get("component_count") != len(component_names):
        raise CheckpointLoadError("checkpoint component count does not match manifest")
    if artifact_stage == TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW:
        if tuple(component_names) != _PREVIEW_COMPONENTS or not expected_names:
            raise CheckpointLoadError("checkpoint train-preview components are incomplete")
    elif expected_names:
        raise CheckpointLoadError("checkpoint stage must not contain runtime tensors")

    data_length = len(artifact_payload) - data_start
    descriptors: list[DreamerCheckpointTensor] = []
    component_totals = {name: {"tensor_count": 0, "element_count": 0} for name in component_names}
    ranges: list[tuple[int, int, str]] = []
    for name in expected_names:
        if not isinstance(name, str) or _SAFE_TENSOR_NAME.fullmatch(name) is None:
            raise CheckpointLoadError("checkpoint tensor name is invalid")
        expected = _require_object(tensor_entries[name], f"checkpoint tensor manifest entry {name}")
        _require_exact_fields(
            expected,
            frozenset({"component", "dtype", "shape", "element_count", "sha256"}),
            f"checkpoint tensor manifest entry {name}",
        )
        actual = _require_object(header.get(name), f"checkpoint safetensors tensor {name}")
        _require_exact_fields(actual, frozenset({"dtype", "shape", "data_offsets"}), f"checkpoint safetensors tensor {name}")
        component = expected.get("component")
        if not isinstance(component, str) or component not in components or name.split(".", 1)[0] != component:
            raise CheckpointLoadError(f"checkpoint tensor {name} component is invalid")
        if expected.get("dtype") != "F32" or actual.get("dtype") != "F32":
            raise CheckpointLoadError(f"checkpoint tensor {name} dtype is incompatible")
        shape = _tensor_shape(expected.get("shape"), name)
        if actual.get("shape") != list(shape):
            raise CheckpointLoadError(f"checkpoint tensor {name} shape does not match manifest")
        element_count = math.prod(shape)
        if _nonnegative_int_value(expected.get("element_count")) != element_count:
            raise CheckpointLoadError(f"checkpoint tensor {name} element count does not match shape")
        offsets = actual.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= data_length
            or offsets[1] - offsets[0] != element_count * 4
        ):
            raise CheckpointLoadError(f"checkpoint tensor {name} data offsets are invalid")
        tensor_bytes = artifact_payload[data_start + offsets[0] : data_start + offsets[1]]
        tensor_sha256 = _require_sha256(expected.get("sha256"), f"checkpoint tensor {name} SHA-256")
        if _sha256_bytes(tensor_bytes) != tensor_sha256:
            raise CheckpointLoadError(f"checkpoint tensor {name} SHA-256 does not match manifest")
        ranges.append((offsets[0], offsets[1], name))
        component_totals[component]["tensor_count"] += 1
        component_totals[component]["element_count"] += element_count
        descriptors.append(
            DreamerCheckpointTensor(
                name=name,
                component=component,
                dtype="F32",
                shape=shape,
                element_count=element_count,
                sha256=tensor_sha256,
                data_start=data_start + offsets[0],
                data_end=data_start + offsets[1],
            )
        )

    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            raise CheckpointLoadError(f"checkpoint tensor {name} data offsets overlap or contain gaps")
        cursor = end
    if cursor != data_length:
        raise CheckpointLoadError("checkpoint safetensors data contains unreferenced bytes")
    for component_name, totals in component_totals.items():
        summary = _require_object(components.get(component_name), f"checkpoint component {component_name}")
        _require_exact_fields(summary, frozenset({"tensor_count", "element_count"}), f"checkpoint component {component_name}")
        normalized_summary = {
            "tensor_count": _nonnegative_int_value(summary.get("tensor_count")),
            "element_count": _nonnegative_int_value(summary.get("element_count")),
        }
        if None in normalized_summary.values() or normalized_summary != totals:
            raise CheckpointLoadError(f"checkpoint component {component_name} totals do not match tensors")
    return tuple(descriptors)


def _validate_compatibility(artifact: dict[str, Any], compatibility: DreamerCheckpointCompatibility) -> None:
    checks = (
        ("feature_layout_sha256", compatibility.feature_layout_sha256, "feature layout"),
        ("control_spec_sha256", compatibility.control_spec_sha256, "control spec"),
        ("dreamer_tensor_contract_sha256", compatibility.tensor_contract_sha256, "tensor contract"),
    )
    for manifest_key, expected, label in checks:
        if expected is not None and artifact.get(manifest_key) != expected:
            raise CheckpointLoadError(f"checkpoint {label} is incompatible with this runtime")


def _parse_architecture(value: dict[str, Any]) -> DreamerCheckpointArchitecture:
    _require_exact_fields(
        value,
        frozenset(
            {
                "format",
                "schema_version",
                "observation_dim",
                "behavior_dim",
                "static_dim",
                "dynamic_action_dim",
                "control_spec",
                "world_model",
                "context_encoder",
                "imagination",
            }
        ),
        "checkpoint runtime architecture",
    )
    if value.get("format") != DREAMER_CHECKPOINT_ARCHITECTURE_FORMAT:
        raise CheckpointLoadError("checkpoint runtime architecture format is unsupported")
    if value.get("schema_version") != DREAMER_CHECKPOINT_ARCHITECTURE_SCHEMA_VERSION:
        raise CheckpointLoadError("checkpoint runtime architecture schema version is unsupported")
    world_model = _require_object(value.get("world_model"), "checkpoint world-model architecture")
    _require_exact_fields(
        world_model,
        frozenset(
            {
                "model_preset",
                "deter_dim",
                "hidden_dim",
                "stoch_size",
                "class_size",
                "action_embed_dim",
                "reward_bins",
                "unimix",
                "free_nats",
                "dyn_loss_scale",
                "rep_loss_scale",
                "observation_loss_scale",
                "reward_loss_scale",
                "continuation_loss_scale",
            }
        ),
        "checkpoint world-model architecture",
    )
    imagination = _require_object(value.get("imagination"), "checkpoint imagination architecture")
    _require_exact_fields(
        imagination,
        frozenset(
            {
                "horizon",
                "actor_hidden_dim",
                "critic_hidden_dim",
                "value_bins",
                "discount",
                "lambda_return",
                "actor_entropy_scale",
            }
        ),
        "checkpoint imagination architecture",
    )
    context_encoder = _require_object(value.get("context_encoder"), "checkpoint context-encoder architecture")
    _require_exact_fields(
        context_encoder,
        frozenset(
            {
                "static_dim",
                "terminal_dim",
                "time_dim",
                "trajectory_dim",
                "hidden_dim",
                "context_dim",
            }
        ),
        "checkpoint context-encoder architecture",
    )
    try:
        return DreamerCheckpointArchitecture(
            format=value["format"],
            schema_version=value["schema_version"],
            observation_dim=value["observation_dim"],
            behavior_dim=value["behavior_dim"],
            static_dim=value["static_dim"],
            dynamic_action_dim=value["dynamic_action_dim"],
            control_spec=DreamerControlSpec.from_dict(value.get("control_spec")),
            world_model=DreamerWorldModelArchitecture(**world_model),
            context_encoder=DreamerContextEncoderArchitecture(**context_encoder),
            imagination=DreamerImaginationArchitecture(**imagination),
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointLoadError(f"checkpoint runtime architecture is invalid: {exc}") from exc


def _tensor_shape(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise CheckpointLoadError(f"checkpoint tensor {name} shape is invalid")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise CheckpointLoadError(f"checkpoint tensor {name} shape is invalid")
    if len(value) > 8:
        raise CheckpointLoadError(f"checkpoint tensor {name} rank is unsupported")
    return tuple(value)


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointLoadError(f"{label} must be an object")
    return value


def _require_exact_fields(value: dict[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing[:5])}")
        if unknown:
            details.append(f"unknown {', '.join(unknown[:5])}")
        raise CheckpointLoadError(f"{label} fields are invalid: {'; '.join(details)}")


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CheckpointLoadError(f"{label} must be a positive integer")
    return value


def _nonnegative_int_value(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _nonnegative_decimal(value: object, label: str) -> int:
    if not isinstance(value, str) or not value.isdecimal():
        raise CheckpointLoadError(f"{label} must be a non-negative decimal integer")
    return int(value)


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in _HEX_CHARS for character in value):
        raise CheckpointLoadError(f"{label} is invalid")
    return value


def _optional_sha256(value: object, label: str) -> str | None:
    if value == "":
        return None
    return _require_sha256(value, label)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)

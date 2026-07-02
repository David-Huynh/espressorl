from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from espresso_rl.application.checkpoint_loading import (
    DEFAULT_MAX_CHECKPOINT_BYTES,
    MAX_SAFETENSORS_HEADER_BYTES,
    CheckpointLoadError,
    load_verified_dreamer_checkpoint,
)
from espresso_rl.domain.model_release import (
    DREAMER_RELEASE_RECORD_FORMAT,
    DREAMER_RELEASE_RECORD_SCHEMA_VERSION,
    DreamerReleaseAuthorization,
)
from espresso_rl.domain.optimization import OPTIMIZER_MODE_DREAMER_V3_ACTIVE
from espresso_rl.domain.trainer_artifacts import TRAINER_ARTIFACT_STAGE_WORLD_MODEL_RELEASE_CANDIDATE
from espresso_rl.dreamer.checkpoint_inference import (
    DreamerCheckpointMaterializationError,
    materialize_verified_dreamer_checkpoint,
)
from espresso_rl.ports.model_store import ModelArtifactStore


MODEL_FILENAME = "dreamer_v3.safetensors"
MODEL_MANIFEST_FILENAME = "dreamer_v3_manifest.json"
RELEASE_RECORD_FILENAME = "release_record.json"
CHECKSUMS_FILENAME = "checksums.txt"


class DreamerModelReleaseError(ValueError):
    pass


@dataclass(frozen=True)
class DreamerReleaseFile:
    relative_path: str
    content_type: str
    size_bytes: int
    sha256: str
    content: bytes


@dataclass(frozen=True)
class DreamerModelReleaseResult:
    candidate_artifact_sha256: str
    candidate_manifest_sha256: str
    released_artifact_sha256: str
    released_manifest_sha256: str
    release_authorization_sha256: str
    files: tuple[DreamerReleaseFile, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "released_artifact_sha256": self.released_artifact_sha256,
            "released_manifest_sha256": self.released_manifest_sha256,
            "release_authorization_sha256": self.release_authorization_sha256,
            "files": [
                {
                    "relative_path": item.relative_path,
                    "content_type": item.content_type,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in self.files
            ],
        }


def release_dreamer_checkpoint(
    store: ModelArtifactStore,
    *,
    candidate_artifact_reference: str,
    candidate_manifest_reference: str,
    authorization: DreamerReleaseAuthorization,
    max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
) -> DreamerModelReleaseResult:
    """Authorize one verified candidate without retraining or changing tensor bytes."""

    if not isinstance(authorization, DreamerReleaseAuthorization):
        raise DreamerModelReleaseError("release authorization is invalid")
    captured_store = _CapturedModelStore(store)
    try:
        candidate = load_verified_dreamer_checkpoint(
            captured_store,
            artifact_reference=candidate_artifact_reference,
            manifest_reference=candidate_manifest_reference,
            expected_artifact_sha256=authorization.candidate_artifact_sha256,
            max_checkpoint_bytes=max_checkpoint_bytes,
        )
    except (CheckpointLoadError, ValueError) as exc:
        raise DreamerModelReleaseError(f"release candidate verification failed: {exc}") from exc
    if candidate.manifest_sha256 != authorization.candidate_manifest_sha256:
        raise DreamerModelReleaseError("release candidate manifest SHA-256 does not match authorization")
    if candidate.artifact_stage != TRAINER_ARTIFACT_STAGE_WORLD_MODEL_RELEASE_CANDIDATE:
        raise DreamerModelReleaseError("only a world-model release candidate can be authorized")
    if candidate.inference_ready or candidate.release_authorization is not None:
        raise DreamerModelReleaseError("release candidate is already inference-ready")

    try:
        candidate_models = materialize_verified_dreamer_checkpoint(candidate)
    except DreamerCheckpointMaterializationError as exc:
        raise DreamerModelReleaseError(f"release candidate inference parity failed: {exc}") from exc

    candidate_artifact_payload = captured_store.payload(candidate_artifact_reference)
    candidate_manifest_payload = captured_store.payload(candidate_manifest_reference)
    manifest = _parse_json_object(candidate_manifest_payload, "release candidate manifest")
    header, tensor_data = _split_safetensors(candidate_artifact_payload)
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise DreamerModelReleaseError("release candidate safetensors metadata is missing")

    authorization_payload = authorization.to_dict()
    authorization_sha256 = _sha256_json(authorization_payload)
    released_manifest = json.loads(_canonical_json(manifest))
    released_manifest["release_authorization"] = authorization_payload
    released_manifest["model_artifact"]["release_authorization_sha256"] = authorization_sha256
    released_manifest["runtime_compatibility"] = {
        "optimizer_mode": OPTIMIZER_MODE_DREAMER_V3_ACTIVE,
        "espresso_rl_runtime_schema_version": manifest["runtime_compatibility"][
            "espresso_rl_runtime_schema_version"
        ],
        "inference_ready": True,
    }

    released_header = json.loads(_canonical_json(header))
    released_metadata = released_header["__metadata__"]
    released_metadata.update(
        {
            "inference_ready": "true",
            "release_authorization_sha256": authorization_sha256,
            "release_candidate_artifact_sha256": authorization.candidate_artifact_sha256,
            "release_candidate_manifest_sha256": authorization.candidate_manifest_sha256,
        }
    )
    released_artifact_payload = _encode_safetensors(released_header, tensor_data)
    released_artifact_sha256 = _sha256_bytes(released_artifact_payload)
    released_manifest["model_artifact"]["sha256"] = released_artifact_sha256
    released_manifest_payload = (_canonical_json(released_manifest) + "\n").encode("utf-8")
    released_manifest_sha256 = _sha256_bytes(released_manifest_payload)

    release_store = _MemoryModelStore(
        {
            MODEL_FILENAME: released_artifact_payload,
            MODEL_MANIFEST_FILENAME: released_manifest_payload,
        }
    )
    try:
        released = load_verified_dreamer_checkpoint(
            release_store,
            artifact_reference=MODEL_FILENAME,
            manifest_reference=MODEL_MANIFEST_FILENAME,
            expected_artifact_sha256=released_artifact_sha256,
            max_checkpoint_bytes=max_checkpoint_bytes,
        )
        released_models = materialize_verified_dreamer_checkpoint(released)
    except (CheckpointLoadError, DreamerCheckpointMaterializationError, ValueError) as exc:
        raise DreamerModelReleaseError(f"released checkpoint verification failed: {exc}") from exc
    if not released.inference_ready or released.release_authorization != authorization:
        raise DreamerModelReleaseError("released checkpoint did not preserve its authorization")
    if candidate_models.inference_probe_sha256 != released_models.inference_probe_sha256:
        raise DreamerModelReleaseError("released checkpoint inference parity differs from candidate")
    candidate_tensors = tuple((item.name, item.dtype, item.shape, item.sha256) for item in candidate.tensors)
    released_tensors = tuple((item.name, item.dtype, item.shape, item.sha256) for item in released.tensors)
    if candidate_tensors != released_tensors or tensor_data != _split_safetensors(released_artifact_payload)[1]:
        raise DreamerModelReleaseError("released checkpoint tensor payloads differ from candidate")

    release_record = {
        "format": DREAMER_RELEASE_RECORD_FORMAT,
        "schema_version": DREAMER_RELEASE_RECORD_SCHEMA_VERSION,
        "authorization": authorization_payload,
        "authorization_sha256": authorization_sha256,
        "released_artifact": {
            "filename": MODEL_FILENAME,
            "sha256": released_artifact_sha256,
        },
        "released_manifest": {
            "filename": MODEL_MANIFEST_FILENAME,
            "sha256": released_manifest_sha256,
        },
        "verification": {
            "candidate_checkpoint_verified": True,
            "candidate_inference_probe_sha256": candidate_models.inference_probe_sha256,
            "released_checkpoint_verified": True,
            "released_inference_probe_sha256": released_models.inference_probe_sha256,
            "tensor_payloads_preserved": True,
            "pickle_content_allowed": False,
        },
    }
    release_record_payload = (_canonical_json(release_record) + "\n").encode("utf-8")
    checksums_payload = _checksums_payload(
        {
            MODEL_FILENAME: released_artifact_payload,
            MODEL_MANIFEST_FILENAME: released_manifest_payload,
            RELEASE_RECORD_FILENAME: release_record_payload,
        }
    )
    files = (
        _release_file(MODEL_FILENAME, released_artifact_payload, "application/octet-stream"),
        _release_file(
            MODEL_MANIFEST_FILENAME,
            released_manifest_payload,
            "application/json; charset=utf-8",
        ),
        _release_file(RELEASE_RECORD_FILENAME, release_record_payload, "application/json; charset=utf-8"),
        _release_file(CHECKSUMS_FILENAME, checksums_payload, "text/plain; charset=utf-8"),
    )
    return DreamerModelReleaseResult(
        candidate_artifact_sha256=authorization.candidate_artifact_sha256,
        candidate_manifest_sha256=authorization.candidate_manifest_sha256,
        released_artifact_sha256=released_artifact_sha256,
        released_manifest_sha256=released_manifest_sha256,
        release_authorization_sha256=authorization_sha256,
        files=files,
    )


class _CapturedModelStore:
    def __init__(self, delegate: ModelArtifactStore) -> None:
        self._delegate = delegate
        self._payloads: dict[str, bytes] = {}

    def read_bytes(self, reference: str, *, max_bytes: int) -> bytes:
        if reference not in self._payloads:
            self._payloads[reference] = self._delegate.read_bytes(reference, max_bytes=max_bytes)
        payload = self._payloads[reference]
        if len(payload) > max_bytes:
            raise ValueError("captured model artifact exceeds the configured size limit")
        return payload

    def payload(self, reference: str) -> bytes:
        try:
            return self._payloads[reference]
        except KeyError as exc:
            raise DreamerModelReleaseError("release candidate payload was not captured") from exc


class _MemoryModelStore:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self._payloads = payloads

    def read_bytes(self, reference: str, *, max_bytes: int) -> bytes:
        payload = self._payloads[reference]
        if len(payload) > max_bytes:
            raise ValueError("released model artifact exceeds the configured size limit")
        return payload


def _parse_json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DreamerModelReleaseError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise DreamerModelReleaseError(f"{label} must be an object")
    return value


def _split_safetensors(payload: bytes) -> tuple[dict[str, Any], bytes]:
    if len(payload) < 10:
        raise DreamerModelReleaseError("release candidate safetensors payload is truncated")
    header_length = int.from_bytes(payload[:8], "little")
    if header_length < 2 or header_length > MAX_SAFETENSORS_HEADER_BYTES:
        raise DreamerModelReleaseError("release candidate safetensors header length is invalid")
    data_start = 8 + header_length
    if data_start > len(payload):
        raise DreamerModelReleaseError("release candidate safetensors header is truncated")
    return _parse_json_object(payload[8:data_start], "release candidate safetensors header"), payload[data_start:]


def _encode_safetensors(header: dict[str, Any], tensor_data: bytes) -> bytes:
    header_payload = _canonical_json(header).encode("utf-8")
    if len(header_payload) > MAX_SAFETENSORS_HEADER_BYTES:
        raise DreamerModelReleaseError("released safetensors header is too large")
    return len(header_payload).to_bytes(8, "little") + header_payload + tensor_data


def _release_file(relative_path: str, content: bytes, content_type: str) -> DreamerReleaseFile:
    return DreamerReleaseFile(
        relative_path=relative_path,
        content_type=content_type,
        size_bytes=len(content),
        sha256=_sha256_bytes(content),
        content=content,
    )


def _checksums_payload(files: dict[str, bytes]) -> bytes:
    return "".join(f"{_sha256_bytes(content)}  {name}\n" for name, content in sorted(files.items())).encode(
        "utf-8"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock

from espresso_rl.domain.dreamer_runtime_audit import (
    DREAMER_FALLBACK_REASON_ACTIVE_UNAVAILABLE,
    DREAMER_FALLBACK_REASON_CANDIDATE_REJECTED,
    DreamerRuntimeAuditSummary,
)
from espresso_rl.domain.model_checkpoint import VerifiedDreamerCheckpoint
from espresso_rl.domain.model_manifest import ModelManifestValidation, validate_model_manifest
from espresso_rl.domain.optimization import (
    DEFAULT_OPTIMIZER_MODE,
    OPTIMIZER_MODE_DREAMER_V3_ACTIVE,
    OPTIMIZER_MODE_DREAMER_V3_SHADOW,
    OptimizationContext,
    normalize_optimizer_mode,
)
from espresso_rl.domain.models import Recommendation
from espresso_rl.optimizers.conservative_bo import ConservativeBOOptimizer
from espresso_rl.ports.optimizers import Optimizer

_DREAMER_SHADOW_FALLBACK_REASON = (
    "DreamerV3 shadow mode is not active inference; Bayesian Optimization is serving recommendations."
)
_DREAMER_ACTIVE_FALLBACK_REASON = (
    "DreamerV3 active mode is unavailable; Bayesian Optimization is serving recommendations."
)
_HASH_CHUNK_BYTES = 1024 * 1024
_MANIFEST_MAX_BYTES = 256 * 1024
_HEX_CHARS = set("0123456789abcdefABCDEF")


@dataclass(frozen=True)
class ModelArtifactStatus:
    path: str | None
    expected_sha256: str | None
    actual_sha256: str | None = None
    size_bytes: int | None = None
    verified: bool = False
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class ModelManifestStatus:
    path: str | None
    actual_sha256: str | None = None
    size_bytes: int | None = None
    verified: bool = False
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


@dataclass(frozen=True)
class RuntimeOptimizerStatus:
    configured_mode: str
    effective_mode: str
    model_artifact_path: str | None = None
    model_artifact_sha256: str | None = None
    model_artifact_actual_sha256: str | None = None
    model_artifact_size_bytes: int | None = None
    model_artifact_verified: bool = False
    model_artifact_unavailable_reason: str | None = None
    model_manifest_path: str | None = None
    model_manifest_sha256: str | None = None
    model_manifest_size_bytes: int | None = None
    model_manifest_verified: bool = False
    model_manifest_unavailable_reason: str | None = None
    model_manifest_model_family: str | None = None
    model_manifest_artifact_format: str | None = None
    model_manifest_dataset_sha256: str | None = None
    model_manifest_dataset_manifest_sha256: str | None = None
    model_manifest_trainer_git_sha: str | None = None
    model_manifest_training_config_sha256: str | None = None
    model_manifest_state_schema_version: int | None = None
    model_manifest_action_schema_version: int | None = None
    model_manifest_reward_schema_version: int | None = None
    checkpoint_verified: bool = False
    checkpoint_inference_ready: bool = False
    checkpoint_tensor_count: int = 0
    checkpoint_component_names: tuple[str, ...] = ()
    checkpoint_architecture_sha256: str | None = None
    checkpoint_inference_probe_sha256: str | None = None
    checkpoint_heldout_inference_sha256: str | None = None
    checkpoint_unavailable_reason: str | None = None
    checkpoint_inference_parity_verified: bool = False
    checkpoint_inference_parity_reason: str | None = None
    dreamer_v3_available: bool = False
    dreamer_v3_shadow_available: bool = False
    dreamer_v3_active_available: bool = False
    available_modes: tuple[str, ...] = (DEFAULT_OPTIMIZER_MODE,)
    unavailable_modes: dict[str, str] | None = None
    fallback_reason: str | None = None
    dreamer_v3_active_recommendation_count: int = 0
    dreamer_v3_bo_fallback_count: int = 0
    dreamer_v3_bo_fallback_reason_counts: dict[str, int] | None = None
    dreamer_v3_last_runtime_event: str | None = None
    dreamer_v3_last_bo_fallback_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "configured_mode": self.configured_mode,
            "effective_mode": self.effective_mode,
            "model_artifact_path": self.model_artifact_path,
            "model_artifact_sha256": self.model_artifact_sha256,
            "model_artifact_actual_sha256": self.model_artifact_actual_sha256,
            "model_artifact_size_bytes": self.model_artifact_size_bytes,
            "model_artifact_verified": self.model_artifact_verified,
            "model_artifact_unavailable_reason": self.model_artifact_unavailable_reason,
            "model_manifest_path": self.model_manifest_path,
            "model_manifest_sha256": self.model_manifest_sha256,
            "model_manifest_size_bytes": self.model_manifest_size_bytes,
            "model_manifest_verified": self.model_manifest_verified,
            "model_manifest_unavailable_reason": self.model_manifest_unavailable_reason,
            "model_manifest_model_family": self.model_manifest_model_family,
            "model_manifest_artifact_format": self.model_manifest_artifact_format,
            "model_manifest_dataset_sha256": self.model_manifest_dataset_sha256,
            "model_manifest_dataset_manifest_sha256": self.model_manifest_dataset_manifest_sha256,
            "model_manifest_trainer_git_sha": self.model_manifest_trainer_git_sha,
            "model_manifest_training_config_sha256": self.model_manifest_training_config_sha256,
            "model_manifest_state_schema_version": self.model_manifest_state_schema_version,
            "model_manifest_action_schema_version": self.model_manifest_action_schema_version,
            "model_manifest_reward_schema_version": self.model_manifest_reward_schema_version,
            "checkpoint_verified": self.checkpoint_verified,
            "checkpoint_inference_ready": self.checkpoint_inference_ready,
            "checkpoint_tensor_count": self.checkpoint_tensor_count,
            "checkpoint_component_names": list(self.checkpoint_component_names),
            "checkpoint_architecture_sha256": self.checkpoint_architecture_sha256,
            "checkpoint_inference_probe_sha256": self.checkpoint_inference_probe_sha256,
            "checkpoint_heldout_inference_sha256": self.checkpoint_heldout_inference_sha256,
            "checkpoint_unavailable_reason": self.checkpoint_unavailable_reason,
            "checkpoint_inference_parity_verified": self.checkpoint_inference_parity_verified,
            "checkpoint_inference_parity_reason": self.checkpoint_inference_parity_reason,
            "dreamer_v3_available": self.dreamer_v3_available,
            "dreamer_v3_shadow_available": self.dreamer_v3_shadow_available,
            "dreamer_v3_active_available": self.dreamer_v3_active_available,
            "available_modes": list(self.available_modes),
            "unavailable_modes": dict(self.unavailable_modes or {}),
            "fallback_reason": self.fallback_reason,
            "dreamer_v3_active_recommendation_count": self.dreamer_v3_active_recommendation_count,
            "dreamer_v3_bo_fallback_count": self.dreamer_v3_bo_fallback_count,
            "dreamer_v3_bo_fallback_reason_counts": dict(
                self.dreamer_v3_bo_fallback_reason_counts or {}
            ),
            "dreamer_v3_last_runtime_event": self.dreamer_v3_last_runtime_event,
            "dreamer_v3_last_bo_fallback_reason": self.dreamer_v3_last_bo_fallback_reason,
        }


class RuntimeOptimizer:
    """Runtime optimizer selector behind the optimizer port."""

    def __init__(
        self,
        optimizer_mode: str = DEFAULT_OPTIMIZER_MODE,
        *,
        model_artifact_path: str | None = None,
        model_artifact_sha256: str | None = None,
        model_manifest_path: str | None = None,
        model_artifact_max_bytes: int = 512 * 1024 * 1024,
        verified_checkpoint: VerifiedDreamerCheckpoint | None = None,
        checkpoint_unavailable_reason: str | None = None,
        checkpoint_inference_parity_verified: bool = False,
        checkpoint_inference_parity_reason: str | None = None,
        bo_optimizer: Optimizer | None = None,
        dreamer_optimizer: Optimizer | None = None,
    ) -> None:
        if model_artifact_max_bytes <= 0:
            raise ValueError("model_artifact_max_bytes must be positive")
        self._lock = RLock()
        self._bo_optimizer = bo_optimizer or ConservativeBOOptimizer()
        self._dreamer_optimizer = dreamer_optimizer
        self._model_artifact_max_bytes = model_artifact_max_bytes
        self._verified_checkpoint = verified_checkpoint
        self._checkpoint_unavailable_reason = _clean_optional_text(checkpoint_unavailable_reason)
        self._checkpoint_inference_parity_verified = bool(checkpoint_inference_parity_verified)
        self._checkpoint_inference_parity_reason = _clean_optional_text(checkpoint_inference_parity_reason)
        self._dreamer_runtime_audit = DreamerRuntimeAuditSummary()
        self._status = self.configure(
            optimizer_mode=optimizer_mode,
            model_artifact_path=model_artifact_path,
            model_artifact_sha256=model_artifact_sha256,
            model_manifest_path=model_manifest_path,
        )

    def configure(
        self,
        *,
        optimizer_mode: str,
        model_artifact_path: str | None = None,
        model_artifact_sha256: str | None = None,
        model_manifest_path: str | None = None,
        model_artifact_max_bytes: int | None = None,
    ) -> RuntimeOptimizerStatus:
        with self._lock:
            previous_status = getattr(self, "_status", None)
            if model_artifact_max_bytes is not None:
                if model_artifact_max_bytes <= 0:
                    raise ValueError("model_artifact_max_bytes must be positive")
                self._model_artifact_max_bytes = model_artifact_max_bytes
            max_bytes = self._model_artifact_max_bytes
        artifact_path = _clean_optional_text(model_artifact_path)
        artifact_sha256 = _clean_optional_sha256(model_artifact_sha256)
        manifest_path = _clean_optional_text(model_manifest_path)
        if previous_status is not None:
            artifact_path = artifact_path or previous_status.model_artifact_path
            artifact_sha256 = artifact_sha256 or previous_status.model_artifact_sha256
            manifest_path = manifest_path or previous_status.model_manifest_path

        requested_mode = normalize_optimizer_mode(optimizer_mode)
        manifest_status = verify_model_manifest_file(
            manifest_path,
            expected_model_sha256=artifact_sha256,
        )
        artifact_status = verify_model_artifact(
            artifact_path,
            artifact_sha256 or manifest_status.model_artifact_sha256,
            max_bytes=max_bytes,
        )
        checkpoint = self._verified_checkpoint
        checkpoint_verified = bool(
            checkpoint is not None
            and artifact_status.verified
            and checkpoint.artifact_reference == artifact_status.path
            and checkpoint.manifest_reference == manifest_status.path
            and checkpoint.artifact_sha256 == artifact_status.actual_sha256
            and checkpoint.manifest_sha256 == manifest_status.actual_sha256
        )
        checkpoint_inference_ready = bool(checkpoint_verified and checkpoint and checkpoint.inference_ready)
        checkpoint_reason = self._checkpoint_unavailable_reason
        if checkpoint_verified and not checkpoint_inference_ready:
            checkpoint_reason = "DreamerV3 checkpoint is verified but runtime inference is not enabled."
        elif not checkpoint_verified and checkpoint_reason is None:
            checkpoint_reason = "DreamerV3 checkpoint has not passed strict tensor verification."

        dreamer_v3_shadow_available = bool(
            checkpoint_verified and self._checkpoint_inference_parity_verified
        )
        dreamer_v3_active_available = bool(
            dreamer_v3_shadow_available
            and checkpoint_inference_ready
            and self._dreamer_optimizer is not None
        )
        dreamer_v3_available = dreamer_v3_active_available
        shadow_unavailable_reason = (
            manifest_status.unavailable_reason
            or artifact_status.unavailable_reason
            or self._checkpoint_inference_parity_reason
            or checkpoint_reason
            or "DreamerV3 shadow model artifact is not verified."
        )
        active_unavailable_reason = (
            "DreamerV3 active optimizer is not wired in."
            if checkpoint_inference_ready and dreamer_v3_shadow_available and self._dreamer_optimizer is None
            else manifest_status.unavailable_reason
            or artifact_status.unavailable_reason
            or self._checkpoint_inference_parity_reason
            or checkpoint_reason
            or "DreamerV3 active model artifact is not verified."
        )

        modes = [DEFAULT_OPTIMIZER_MODE]
        unavailable_modes: dict[str, str] = {}
        if dreamer_v3_shadow_available:
            modes.append(OPTIMIZER_MODE_DREAMER_V3_SHADOW)
        else:
            unavailable_modes[OPTIMIZER_MODE_DREAMER_V3_SHADOW] = shadow_unavailable_reason
        if dreamer_v3_active_available:
            modes.append(OPTIMIZER_MODE_DREAMER_V3_ACTIVE)
        else:
            unavailable_modes[OPTIMIZER_MODE_DREAMER_V3_ACTIVE] = active_unavailable_reason
        available_modes = tuple(modes)

        configured_mode = requested_mode
        effective_mode = DEFAULT_OPTIMIZER_MODE
        fallback_reason = None
        if requested_mode == OPTIMIZER_MODE_DREAMER_V3_ACTIVE and dreamer_v3_active_available:
            effective_mode = OPTIMIZER_MODE_DREAMER_V3_ACTIVE
        elif requested_mode == OPTIMIZER_MODE_DREAMER_V3_ACTIVE:
            fallback_reason = _DREAMER_ACTIVE_FALLBACK_REASON
        elif requested_mode == OPTIMIZER_MODE_DREAMER_V3_SHADOW and dreamer_v3_shadow_available:
            fallback_reason = _DREAMER_SHADOW_FALLBACK_REASON
        elif requested_mode == OPTIMIZER_MODE_DREAMER_V3_SHADOW:
            fallback_reason = _DREAMER_SHADOW_FALLBACK_REASON

        audit = self._dreamer_runtime_audit
        status = RuntimeOptimizerStatus(
            configured_mode=configured_mode,
            effective_mode=effective_mode,
            model_artifact_path=artifact_status.path,
            model_artifact_sha256=artifact_status.expected_sha256,
            model_artifact_actual_sha256=artifact_status.actual_sha256,
            model_artifact_size_bytes=artifact_status.size_bytes,
            model_artifact_verified=artifact_status.verified,
            model_artifact_unavailable_reason=artifact_status.unavailable_reason,
            model_manifest_path=manifest_status.path,
            model_manifest_sha256=manifest_status.actual_sha256,
            model_manifest_size_bytes=manifest_status.size_bytes,
            model_manifest_verified=manifest_status.verified,
            model_manifest_unavailable_reason=manifest_status.unavailable_reason,
            model_manifest_model_family=manifest_status.model_family,
            model_manifest_artifact_format=manifest_status.model_artifact_format,
            model_manifest_dataset_sha256=manifest_status.dataset_sha256,
            model_manifest_dataset_manifest_sha256=manifest_status.dataset_manifest_sha256,
            model_manifest_trainer_git_sha=manifest_status.trainer_git_sha,
            model_manifest_training_config_sha256=manifest_status.training_config_sha256,
            model_manifest_state_schema_version=manifest_status.state_schema_version,
            model_manifest_action_schema_version=manifest_status.action_schema_version,
            model_manifest_reward_schema_version=manifest_status.reward_schema_version,
            checkpoint_verified=checkpoint_verified,
            checkpoint_inference_ready=checkpoint_inference_ready,
            checkpoint_tensor_count=len(checkpoint.tensors) if checkpoint_verified and checkpoint else 0,
            checkpoint_component_names=checkpoint.component_names if checkpoint_verified and checkpoint else (),
            checkpoint_architecture_sha256=(
                checkpoint.architecture_sha256 if checkpoint_verified and checkpoint else None
            ),
            checkpoint_inference_probe_sha256=(
                checkpoint.inference_probe_sha256 if checkpoint_verified and checkpoint else None
            ),
            checkpoint_heldout_inference_sha256=(
                checkpoint.heldout_inference_sha256 if checkpoint_verified and checkpoint else None
            ),
            checkpoint_unavailable_reason=checkpoint_reason,
            checkpoint_inference_parity_verified=(
                checkpoint_verified and self._checkpoint_inference_parity_verified
            ),
            checkpoint_inference_parity_reason=self._checkpoint_inference_parity_reason,
            dreamer_v3_available=dreamer_v3_available,
            dreamer_v3_shadow_available=dreamer_v3_shadow_available,
            dreamer_v3_active_available=dreamer_v3_active_available,
            available_modes=available_modes,
            unavailable_modes=unavailable_modes,
            fallback_reason=fallback_reason,
            dreamer_v3_active_recommendation_count=audit.active_recommendation_count,
            dreamer_v3_bo_fallback_count=audit.bo_fallback_count,
            dreamer_v3_bo_fallback_reason_counts=audit.fallback_reason_counts_dict(),
            dreamer_v3_last_runtime_event=audit.last_runtime_event,
            dreamer_v3_last_bo_fallback_reason=audit.last_bo_fallback_reason,
        )
        with self._lock:
            self._status = status
        return status

    def status(self) -> RuntimeOptimizerStatus:
        with self._lock:
            return self._status

    def recommend(self, context: OptimizationContext) -> Recommendation:
        with self._lock:
            configured_mode = self._status.configured_mode
            effective_mode = self._status.effective_mode
            dreamer_optimizer = self._dreamer_optimizer
        if effective_mode == OPTIMIZER_MODE_DREAMER_V3_ACTIVE and dreamer_optimizer is not None:
            try:
                recommendation = dreamer_optimizer.recommend(context)
            except ValueError:
                self._record_dreamer_bo_fallback(DREAMER_FALLBACK_REASON_CANDIDATE_REJECTED)
                return self._bo_optimizer.recommend(context)
            self._record_dreamer_active_recommendation()
            return recommendation
        if configured_mode == OPTIMIZER_MODE_DREAMER_V3_ACTIVE:
            self._record_dreamer_bo_fallback(DREAMER_FALLBACK_REASON_ACTIVE_UNAVAILABLE)
        return self._bo_optimizer.recommend(context)

    def _record_dreamer_active_recommendation(self) -> None:
        with self._lock:
            self._dreamer_runtime_audit = self._dreamer_runtime_audit.record_active_recommendation()
            self._status = self._status_with_runtime_audit(self._status)

    def _record_dreamer_bo_fallback(self, reason: str) -> None:
        with self._lock:
            self._dreamer_runtime_audit = self._dreamer_runtime_audit.record_bo_fallback(reason)
            self._status = self._status_with_runtime_audit(self._status)

    def _status_with_runtime_audit(self, status: RuntimeOptimizerStatus) -> RuntimeOptimizerStatus:
        audit = self._dreamer_runtime_audit
        return replace(
            status,
            dreamer_v3_active_recommendation_count=audit.active_recommendation_count,
            dreamer_v3_bo_fallback_count=audit.bo_fallback_count,
            dreamer_v3_bo_fallback_reason_counts=audit.fallback_reason_counts_dict(),
            dreamer_v3_last_runtime_event=audit.last_runtime_event,
            dreamer_v3_last_bo_fallback_reason=audit.last_bo_fallback_reason,
        )


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_optional_sha256(value: str | None) -> str | None:
    text = _clean_optional_text(value)
    if text is None:
        return None
    if len(text) != 64 or any(ch not in _HEX_CHARS for ch in text):
        return None
    return text.lower()


def verify_model_artifact(
    model_artifact_path: str | None,
    expected_sha256: str | None,
    *,
    max_bytes: int,
) -> ModelArtifactStatus:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    path_text = _clean_optional_text(model_artifact_path)
    expected_digest = _clean_optional_sha256(expected_sha256)
    if not path_text:
        return ModelArtifactStatus(
            path=None,
            expected_sha256=expected_digest,
            unavailable_reason="DreamerV3 model artifact path is not configured.",
        )
    if expected_digest is None:
        return ModelArtifactStatus(
            path=path_text,
            expected_sha256=None,
            unavailable_reason="DreamerV3 model artifact SHA-256 is not configured or invalid.",
        )

    path = Path(path_text)
    try:
        stat = path.stat()
    except OSError:
        return ModelArtifactStatus(
            path=path_text,
            expected_sha256=expected_digest,
            unavailable_reason="DreamerV3 model artifact file does not exist or is unreadable.",
        )
    if not path.is_file():
        return ModelArtifactStatus(
            path=path_text,
            expected_sha256=expected_digest,
            unavailable_reason="DreamerV3 model artifact path is not a file.",
        )
    if stat.st_size <= 0:
        return ModelArtifactStatus(
            path=path_text,
            expected_sha256=expected_digest,
            size_bytes=stat.st_size,
            unavailable_reason="DreamerV3 model artifact file is empty.",
        )
    if stat.st_size > max_bytes:
        return ModelArtifactStatus(
            path=path_text,
            expected_sha256=expected_digest,
            size_bytes=stat.st_size,
            unavailable_reason="DreamerV3 model artifact file is larger than the configured limit.",
        )

    sha256 = hashlib.sha256()
    try:
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(_HASH_CHUNK_BYTES), b""):
                sha256.update(chunk)
    except OSError:
        return ModelArtifactStatus(
            path=path_text,
            expected_sha256=expected_digest,
            size_bytes=stat.st_size,
            unavailable_reason="DreamerV3 model artifact file could not be read.",
        )

    actual_digest = sha256.hexdigest()
    if actual_digest != expected_digest:
        return ModelArtifactStatus(
            path=path_text,
            expected_sha256=expected_digest,
            actual_sha256=actual_digest,
            size_bytes=stat.st_size,
            unavailable_reason="DreamerV3 model artifact SHA-256 does not match.",
        )
    return ModelArtifactStatus(
        path=path_text,
        expected_sha256=expected_digest,
        actual_sha256=actual_digest,
        size_bytes=stat.st_size,
        verified=True,
    )


def verify_model_manifest_file(
    model_manifest_path: str | None,
    *,
    expected_model_sha256: str | None = None,
) -> ModelManifestStatus:
    path_text = _clean_optional_text(model_manifest_path)
    expected_digest = _clean_optional_sha256(expected_model_sha256)
    if not path_text:
        return ModelManifestStatus(
            path=None,
            unavailable_reason="DreamerV3 model manifest path is not configured.",
        )

    path = Path(path_text)
    try:
        stat = path.stat()
    except OSError:
        return ModelManifestStatus(
            path=path_text,
            unavailable_reason="DreamerV3 model manifest file does not exist or is unreadable.",
        )
    if not path.is_file():
        return ModelManifestStatus(
            path=path_text,
            unavailable_reason="DreamerV3 model manifest path is not a file.",
        )
    if stat.st_size <= 0:
        return ModelManifestStatus(
            path=path_text,
            size_bytes=stat.st_size,
            unavailable_reason="DreamerV3 model manifest file is empty.",
        )
    if stat.st_size > _MANIFEST_MAX_BYTES:
        return ModelManifestStatus(
            path=path_text,
            size_bytes=stat.st_size,
            unavailable_reason="DreamerV3 model manifest file is larger than the configured limit.",
        )

    try:
        payload = path.read_bytes()
    except OSError:
        return ModelManifestStatus(
            path=path_text,
            size_bytes=stat.st_size,
            unavailable_reason="DreamerV3 model manifest file could not be read.",
        )
    actual_digest = hashlib.sha256(payload).hexdigest()
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ModelManifestStatus(
            path=path_text,
            actual_sha256=actual_digest,
            size_bytes=stat.st_size,
            unavailable_reason="DreamerV3 model manifest is not valid UTF-8 JSON.",
        )

    validation = validate_model_manifest(
        manifest,
        expected_model_sha256=expected_digest,
    )
    return _manifest_status_from_validation(
        path=path_text,
        actual_sha256=actual_digest,
        size_bytes=stat.st_size,
        validation=validation,
    )


def _manifest_status_from_validation(
    *,
    path: str,
    actual_sha256: str,
    size_bytes: int,
    validation: ModelManifestValidation,
) -> ModelManifestStatus:
    return ModelManifestStatus(
        path=path,
        actual_sha256=actual_sha256,
        size_bytes=size_bytes,
        verified=validation.verified,
        unavailable_reason=validation.unavailable_reason,
        model_family=validation.model_family,
        model_artifact_format=validation.model_artifact_format,
        model_artifact_sha256=validation.model_artifact_sha256,
        dataset_sha256=validation.dataset_sha256,
        dataset_manifest_sha256=validation.dataset_manifest_sha256,
        trainer_git_sha=validation.trainer_git_sha,
        training_config_sha256=validation.training_config_sha256,
        state_schema_version=validation.state_schema_version,
        action_schema_version=validation.action_schema_version,
        reward_schema_version=validation.reward_schema_version,
    )

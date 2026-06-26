from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from espresso_rl.domain.optimization import (
    DEFAULT_OPTIMIZER_MODE,
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
_HASH_CHUNK_BYTES = 1024 * 1024
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
class RuntimeOptimizerStatus:
    configured_mode: str
    effective_mode: str
    model_artifact_path: str | None = None
    model_artifact_sha256: str | None = None
    model_artifact_actual_sha256: str | None = None
    model_artifact_size_bytes: int | None = None
    model_artifact_verified: bool = False
    model_artifact_unavailable_reason: str | None = None
    dreamer_v3_available: bool = False
    available_modes: tuple[str, ...] = (DEFAULT_OPTIMIZER_MODE,)
    unavailable_modes: dict[str, str] | None = None
    fallback_reason: str | None = None

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
            "dreamer_v3_available": self.dreamer_v3_available,
            "available_modes": list(self.available_modes),
            "unavailable_modes": dict(self.unavailable_modes or {}),
            "fallback_reason": self.fallback_reason,
        }


class RuntimeOptimizer:
    """Runtime optimizer selector behind the optimizer port."""

    def __init__(
        self,
        optimizer_mode: str = DEFAULT_OPTIMIZER_MODE,
        *,
        model_artifact_path: str | None = None,
        model_artifact_sha256: str | None = None,
        model_artifact_max_bytes: int = 512 * 1024 * 1024,
        bo_optimizer: Optimizer | None = None,
    ) -> None:
        if model_artifact_max_bytes <= 0:
            raise ValueError("model_artifact_max_bytes must be positive")
        self._lock = RLock()
        self._bo_optimizer = bo_optimizer or ConservativeBOOptimizer()
        self._model_artifact_max_bytes = model_artifact_max_bytes
        self._status = self.configure(
            optimizer_mode=optimizer_mode,
            model_artifact_path=model_artifact_path,
            model_artifact_sha256=model_artifact_sha256,
        )

    def configure(
        self,
        *,
        optimizer_mode: str,
        model_artifact_path: str | None = None,
        model_artifact_sha256: str | None = None,
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
        if previous_status is not None:
            artifact_path = artifact_path or previous_status.model_artifact_path
            artifact_sha256 = artifact_sha256 or previous_status.model_artifact_sha256

        requested_mode = normalize_optimizer_mode(optimizer_mode)
        artifact_status = verify_model_artifact(
            artifact_path,
            artifact_sha256,
            max_bytes=max_bytes,
        )
        dreamer_v3_available = artifact_status.verified
        available_modes = (
            (DEFAULT_OPTIMIZER_MODE, OPTIMIZER_MODE_DREAMER_V3_SHADOW)
            if dreamer_v3_available
            else (DEFAULT_OPTIMIZER_MODE,)
        )
        unavailable_modes = (
            {}
            if dreamer_v3_available
            else {
                OPTIMIZER_MODE_DREAMER_V3_SHADOW: artifact_status.unavailable_reason
                or "DreamerV3 model artifact is not verified."
            }
        )
        configured_mode = requested_mode
        effective_mode = DEFAULT_OPTIMIZER_MODE
        fallback_reason = None
        if requested_mode == OPTIMIZER_MODE_DREAMER_V3_SHADOW and not dreamer_v3_available:
            configured_mode = DEFAULT_OPTIMIZER_MODE
        elif requested_mode == OPTIMIZER_MODE_DREAMER_V3_SHADOW:
            fallback_reason = _DREAMER_SHADOW_FALLBACK_REASON

        status = RuntimeOptimizerStatus(
            configured_mode=configured_mode,
            effective_mode=effective_mode,
            model_artifact_path=artifact_status.path,
            model_artifact_sha256=artifact_status.expected_sha256,
            model_artifact_actual_sha256=artifact_status.actual_sha256,
            model_artifact_size_bytes=artifact_status.size_bytes,
            model_artifact_verified=artifact_status.verified,
            model_artifact_unavailable_reason=artifact_status.unavailable_reason,
            dreamer_v3_available=dreamer_v3_available,
            available_modes=available_modes,
            unavailable_modes=unavailable_modes,
            fallback_reason=fallback_reason,
        )
        with self._lock:
            self._status = status
        return status

    def status(self) -> RuntimeOptimizerStatus:
        with self._lock:
            return self._status

    def recommend(self, context: OptimizationContext) -> Recommendation:
        with self._lock:
            effective_mode = self._status.effective_mode
        if effective_mode == DEFAULT_OPTIMIZER_MODE:
            return self._bo_optimizer.recommend(context)
        return self._bo_optimizer.recommend(context)


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

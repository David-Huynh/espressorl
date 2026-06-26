from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class RuntimeOptimizerStatus:
    configured_mode: str
    effective_mode: str
    model_artifact_path: str | None = None
    model_artifact_sha256: str | None = None
    dreamer_v3_available: bool = False
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "configured_mode": self.configured_mode,
            "effective_mode": self.effective_mode,
            "model_artifact_path": self.model_artifact_path,
            "model_artifact_sha256": self.model_artifact_sha256,
            "dreamer_v3_available": self.dreamer_v3_available,
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
        bo_optimizer: Optimizer | None = None,
    ) -> None:
        self._lock = RLock()
        self._bo_optimizer = bo_optimizer or ConservativeBOOptimizer()
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
    ) -> RuntimeOptimizerStatus:
        with self._lock:
            previous_status = getattr(self, "_status", None)
        artifact_path = _clean_optional_text(model_artifact_path)
        artifact_sha256 = _clean_optional_text(model_artifact_sha256)
        if previous_status is not None:
            artifact_path = artifact_path or previous_status.model_artifact_path
            artifact_sha256 = artifact_sha256 or previous_status.model_artifact_sha256

        requested_mode = normalize_optimizer_mode(optimizer_mode)
        dreamer_v3_available = bool(artifact_path and artifact_sha256)
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
            model_artifact_path=artifact_path,
            model_artifact_sha256=artifact_sha256,
            dreamer_v3_available=dreamer_v3_available,
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

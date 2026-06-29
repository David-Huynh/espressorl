from __future__ import annotations

from dataclasses import dataclass

from espresso_rl.domain.model_checkpoint import VerifiedDreamerCheckpoint
from espresso_rl.domain.shadow_contract import SHADOW_INFERENCE_CONTRACT_LEARNED_CONTEXT_ENCODER_V1
from espresso_rl.dreamer.checkpoint_inference import (
    DreamerCheckpointMaterializationError,
    DreamerShadowModels,
    materialize_verified_dreamer_checkpoint,
)


@dataclass(frozen=True)
class DreamerShadowInferenceStatus:
    checkpoint_artifact_sha256: str
    checkpoint_manifest_sha256: str
    inference_probe_sha256: str
    heldout_inference_sha256: str
    inference_contract_id: str
    tensor_count: int
    component_names: tuple[str, ...]
    parity_verified: bool = True
    inference_ready: bool = False
    recommendation_enabled: bool = False
    machine_control_enabled: bool = False


class DreamerShadowInferenceError(ValueError):
    pass


@dataclass(frozen=True)
class DreamerShadowInferenceSession:
    """Materialized models for offline parity checks only."""

    models: DreamerShadowModels
    status: DreamerShadowInferenceStatus
    checkpoint: VerifiedDreamerCheckpoint


def build_dreamer_shadow_inference_session(
    checkpoint: VerifiedDreamerCheckpoint,
) -> DreamerShadowInferenceSession:
    try:
        models = materialize_verified_dreamer_checkpoint(checkpoint)
    except DreamerCheckpointMaterializationError as exc:
        raise DreamerShadowInferenceError(str(exc)) from exc
    return DreamerShadowInferenceSession(
        models=models,
        status=DreamerShadowInferenceStatus(
            checkpoint_artifact_sha256=checkpoint.artifact_sha256,
            checkpoint_manifest_sha256=checkpoint.manifest_sha256,
            inference_probe_sha256=models.inference_probe_sha256,
            heldout_inference_sha256=checkpoint.heldout_inference_sha256 or "",
            inference_contract_id=SHADOW_INFERENCE_CONTRACT_LEARNED_CONTEXT_ENCODER_V1,
            tensor_count=len(checkpoint.tensors),
            component_names=checkpoint.component_names,
        ),
        checkpoint=checkpoint,
    )

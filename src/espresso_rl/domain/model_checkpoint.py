from __future__ import annotations

from dataclasses import dataclass
import math

from espresso_rl.domain.dreamer_control import DreamerControlSpec
from espresso_rl.domain.dreamer_live_action import DreamerLiveActionSpec
from espresso_rl.domain.dreamer_pre_shot import DreamerPreShotActionSpec
from espresso_rl.domain.dreamer_taste import (
    DREAMER_TASTE_OBJECTIVE_ATTRIBUTES,
    DreamerTasteObjectiveSpec,
)
from espresso_rl.domain.model_release import DreamerReleaseAuthorization

DREAMER_CHECKPOINT_ARCHITECTURE_FORMAT = "espresso_rl_dreamer_v3_checkpoint_architecture_v3"
DREAMER_CHECKPOINT_ARCHITECTURE_SCHEMA_VERSION = 3
DREAMER_INFERENCE_PROBE_FORMAT = "espresso_rl_dreamer_v3_inference_probe_v1"

_HEX_CHARS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class DreamerCheckpointCompatibility:
    """Runtime-owned hashes used to reject structurally valid but incompatible models."""

    feature_layout_sha256: str | None = None
    control_spec_sha256: str | None = None
    pre_shot_action_spec_sha256: str | None = None
    live_action_spec_sha256: str | None = None
    taste_objective_spec_sha256: str | None = None
    tensor_contract_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "feature_layout_sha256",
            "control_spec_sha256",
            "pre_shot_action_spec_sha256",
            "live_action_spec_sha256",
            "taste_objective_spec_sha256",
            "tensor_contract_sha256",
        ):
            value = getattr(self, field_name)
            if value is not None and not _is_sha256(value):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class DreamerCheckpointTensor:
    name: str
    component: str
    dtype: str
    shape: tuple[int, ...]
    element_count: int
    sha256: str
    data_start: int
    data_end: int


@dataclass(frozen=True)
class DreamerWorldModelArchitecture:
    model_preset: str
    deter_dim: int
    hidden_dim: int
    stoch_size: int
    class_size: int
    action_embed_dim: int
    reward_bins: int
    unimix: float
    free_nats: float
    dyn_loss_scale: float
    rep_loss_scale: float
    observation_loss_scale: float
    reward_loss_scale: float
    continuation_loss_scale: float

    def __post_init__(self) -> None:
        if self.model_preset not in {"espresso_debug", "espresso_small", "espresso_medium"}:
            raise ValueError("checkpoint world-model preset is unsupported")
        _bounded_int(self.deter_dim, 8, 2048, "deter_dim")
        _bounded_int(self.hidden_dim, 8, 2048, "hidden_dim")
        _bounded_int(self.stoch_size, 2, 64, "stoch_size")
        _bounded_int(self.class_size, 2, 128, "class_size")
        _bounded_int(self.action_embed_dim, 4, 256, "action_embed_dim")
        _bounded_int(self.reward_bins, 3, 255, "reward_bins")
        if self.reward_bins % 2 == 0:
            raise ValueError("checkpoint reward_bins must be odd")
        _bounded_float(self.unimix, 0.0, 0.2, "unimix")
        _bounded_float(self.free_nats, 0.0, 10.0, "free_nats")
        for field_name in (
            "dyn_loss_scale",
            "rep_loss_scale",
            "observation_loss_scale",
            "reward_loss_scale",
            "continuation_loss_scale",
        ):
            _bounded_float(getattr(self, field_name), 0.0, 100.0, field_name)

    def to_dict(self) -> dict[str, object]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class DreamerImaginationArchitecture:
    horizon: int
    actor_hidden_dim: int
    critic_hidden_dim: int
    value_bins: int
    discount: float
    lambda_return: float
    actor_entropy_scale: float
    pre_shot_behavior_loss_scale: float

    def __post_init__(self) -> None:
        _bounded_int(self.horizon, 1, 32, "horizon")
        _bounded_int(self.actor_hidden_dim, 8, 2048, "actor_hidden_dim")
        _bounded_int(self.critic_hidden_dim, 8, 2048, "critic_hidden_dim")
        _bounded_int(self.value_bins, 3, 255, "value_bins")
        if self.value_bins % 2 == 0:
            raise ValueError("checkpoint value_bins must be odd")
        _bounded_float(self.discount, 0.0, 1.0, "discount")
        _bounded_float(self.lambda_return, 0.0, 1.0, "lambda_return")
        _bounded_float(self.actor_entropy_scale, 0.0, 1.0, "actor_entropy_scale")
        _bounded_float(
            self.pre_shot_behavior_loss_scale,
            0.0,
            100.0,
            "pre_shot_behavior_loss_scale",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class DreamerContextEncoderArchitecture:
    static_dim: int
    terminal_dim: int
    time_dim: int
    trajectory_dim: int
    hidden_dim: int
    context_dim: int

    def __post_init__(self) -> None:
        _bounded_int(self.static_dim, 1, 256, "context_encoder.static_dim")
        _bounded_int(self.terminal_dim, 1, 256, "context_encoder.terminal_dim")
        _bounded_int(self.time_dim, 1, 16, "context_encoder.time_dim")
        _bounded_int(self.trajectory_dim, 1, 1024, "context_encoder.trajectory_dim")
        _bounded_int(self.hidden_dim, 8, 2048, "context_encoder.hidden_dim")
        _bounded_int(self.context_dim, 8, 2048, "context_encoder.context_dim")

    def to_dict(self) -> dict[str, object]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class DreamerCheckpointArchitecture:
    observation_dim: int
    behavior_dim: int
    static_dim: int
    live_action_dim: int
    taste_objective_dim: int
    control_spec: DreamerControlSpec
    pre_shot_action_spec: DreamerPreShotActionSpec
    live_action_spec: DreamerLiveActionSpec
    taste_objective_spec: DreamerTasteObjectiveSpec
    world_model: DreamerWorldModelArchitecture
    context_encoder: DreamerContextEncoderArchitecture
    imagination: DreamerImaginationArchitecture
    format: str = DREAMER_CHECKPOINT_ARCHITECTURE_FORMAT
    schema_version: int = DREAMER_CHECKPOINT_ARCHITECTURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.format != DREAMER_CHECKPOINT_ARCHITECTURE_FORMAT:
            raise ValueError("checkpoint architecture format is unsupported")
        if self.schema_version != DREAMER_CHECKPOINT_ARCHITECTURE_SCHEMA_VERSION:
            raise ValueError("checkpoint architecture schema version is unsupported")
        _bounded_int(self.observation_dim, 1, 256, "observation_dim")
        _bounded_int(self.behavior_dim, 1, 512, "behavior_dim")
        _bounded_int(self.static_dim, 1, 256, "static_dim")
        _bounded_int(self.live_action_dim, 1, 64, "live_action_dim")
        _bounded_int(self.taste_objective_dim, 1, 64, "taste_objective_dim")
        if not isinstance(self.world_model, DreamerWorldModelArchitecture):
            raise ValueError("checkpoint world-model architecture is invalid")
        if not isinstance(self.context_encoder, DreamerContextEncoderArchitecture):
            raise ValueError("checkpoint context-encoder architecture is invalid")
        if not isinstance(self.imagination, DreamerImaginationArchitecture):
            raise ValueError("checkpoint imagination architecture is invalid")
        if not isinstance(self.control_spec, DreamerControlSpec):
            raise ValueError("checkpoint control spec is invalid")
        if not isinstance(self.pre_shot_action_spec, DreamerPreShotActionSpec):
            raise ValueError("checkpoint pre-shot action spec is invalid")
        if not isinstance(self.live_action_spec, DreamerLiveActionSpec):
            raise ValueError("checkpoint live action spec is invalid")
        if not isinstance(self.taste_objective_spec, DreamerTasteObjectiveSpec):
            raise ValueError("checkpoint taste-objective spec is invalid")
        if self.live_action_dim != len(self.live_action_spec.bins):
            raise ValueError("checkpoint live_action_dim does not match live-action spec")
        if self.taste_objective_dim != 1 + len(DREAMER_TASTE_OBJECTIVE_ATTRIBUTES):
            raise ValueError("checkpoint taste_objective_dim does not match taste-objective spec")
        if self.world_model.reward_bins != self.imagination.value_bins:
            raise ValueError("checkpoint reward and value bin counts must match")
        if self.context_encoder.static_dim != self.static_dim:
            raise ValueError("checkpoint context encoder static_dim must match static_dim")
        if self.context_encoder.context_dim != self.world_model.deter_dim:
            raise ValueError("checkpoint context encoder context_dim must match world-model deter_dim")

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "observation_dim": self.observation_dim,
            "behavior_dim": self.behavior_dim,
            "static_dim": self.static_dim,
            "live_action_dim": self.live_action_dim,
            "taste_objective_dim": self.taste_objective_dim,
            "control_spec": self.control_spec.to_dict(),
            "pre_shot_action_spec": self.pre_shot_action_spec.to_dict(),
            "live_action_spec": self.live_action_spec.to_dict(),
            "taste_objective_spec": self.taste_objective_spec.to_dict(),
            "world_model": self.world_model.to_dict(),
            "context_encoder": self.context_encoder.to_dict(),
            "imagination": self.imagination.to_dict(),
        }


@dataclass(frozen=True)
class VerifiedDreamerCheckpoint:
    """Authenticated checkpoint bytes plus release-declared runtime readiness."""

    artifact_reference: str
    manifest_reference: str
    artifact_sha256: str
    manifest_sha256: str
    dataset_sha256: str
    dataset_manifest_sha256: str
    training_config_sha256: str
    tensor_contract_sha256: str
    feature_layout_sha256: str
    control_spec_sha256: str
    pre_shot_action_spec_sha256: str
    live_action_spec_sha256: str
    taste_objective_spec_sha256: str
    evaluation_report_sha256: str | None
    architecture_sha256: str
    inference_probe_sha256: str | None
    heldout_inference_sha256: str | None
    architecture: DreamerCheckpointArchitecture | None
    artifact_stage: str
    inference_ready: bool
    release_authorization: DreamerReleaseAuthorization | None
    tensors: tuple[DreamerCheckpointTensor, ...]
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.inference_ready, bool):
            raise ValueError("checkpoint inference_ready must be boolean")
        if self.inference_ready != (self.release_authorization is not None):
            raise ValueError("checkpoint release authorization must match inference readiness")

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(sorted({tensor.component for tensor in self.tensors}))

    def tensor(self, name: str) -> DreamerCheckpointTensor:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)

    def tensor_bytes(self, name: str) -> memoryview:
        tensor = self.tensor(name)
        return memoryview(self.payload)[tensor.data_start : tensor.data_end]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_CHARS for character in value)
    )


def _bounded_int(value: object, minimum: int, maximum: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"checkpoint {label} is invalid")


def _bounded_float(value: object, minimum: float, maximum: float, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ValueError(f"checkpoint {label} is invalid")

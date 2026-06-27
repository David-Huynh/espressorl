from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from espresso_rl.domain.model_checkpoint import (
    DREAMER_INFERENCE_PROBE_FORMAT,
    DreamerCheckpointArchitecture,
    DreamerImaginationArchitecture,
    DreamerWorldModelArchitecture,
    VerifiedDreamerCheckpoint,
)
from espresso_rl.domain.dreamer_control import DreamerControlSpec
from espresso_rl.domain.trainer_artifacts import TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW
from espresso_rl.dreamer.imagination import (
    DreamerV3ImaginationActor,
    DreamerV3ImaginationConfig,
    DreamerV3ImaginationCritic,
)
from espresso_rl.dreamer.reference_world_model import (
    DreamerV3VectorWorldModel,
    DreamerV3WorldModelConfig,
    behavior_tensor_from_parts,
)

_COMPONENT_BUFFER_NAMES = {
    "world_model": "reward_bins",
    "actor": "static_action_bins",
    "critic": "value_bins",
}


class DreamerCheckpointMaterializationError(ValueError):
    pass


@dataclass(frozen=True)
class DreamerShadowModels:
    world_model: DreamerV3VectorWorldModel
    actor: DreamerV3ImaginationActor
    critic: DreamerV3ImaginationCritic
    imagination_config: DreamerV3ImaginationConfig
    inference_probe_sha256: str


def checkpoint_architecture_from_models(
    *,
    world_model: DreamerV3VectorWorldModel,
    actor: DreamerV3ImaginationActor,
    critic: DreamerV3ImaginationCritic,
    observation_dim: int,
    behavior_dim: int,
    static_dim: int,
    dynamic_action_dim: int,
    control_spec: DreamerControlSpec,
) -> DreamerCheckpointArchitecture:
    if actor.config != critic.config:
        raise DreamerCheckpointMaterializationError("actor and critic imagination configs do not match")
    model = world_model.config
    imagination = actor.config
    return DreamerCheckpointArchitecture(
        observation_dim=observation_dim,
        behavior_dim=behavior_dim,
        static_dim=static_dim,
        dynamic_action_dim=dynamic_action_dim,
        control_spec=control_spec,
        world_model=DreamerWorldModelArchitecture(
            model_preset=model.model_preset,
            deter_dim=model.deter_dim,
            hidden_dim=model.hidden_dim,
            stoch_size=model.stoch_size,
            class_size=model.class_size,
            action_embed_dim=model.action_embed_dim,
            reward_bins=model.reward_bins,
            unimix=model.unimix,
            free_nats=model.free_nats,
            dyn_loss_scale=model.dyn_loss_scale,
            rep_loss_scale=model.rep_loss_scale,
            observation_loss_scale=model.observation_loss_scale,
            reward_loss_scale=model.reward_loss_scale,
            continuation_loss_scale=model.continuation_loss_scale,
        ),
        imagination=DreamerImaginationArchitecture(
            horizon=imagination.horizon,
            actor_hidden_dim=imagination.actor_hidden_dim,
            critic_hidden_dim=imagination.critic_hidden_dim,
            value_bins=imagination.value_bins,
            discount=imagination.discount,
            lambda_return=imagination.lambda_return,
            actor_entropy_scale=imagination.actor_entropy_scale,
        ),
    )


def materialize_verified_dreamer_checkpoint(
    checkpoint: VerifiedDreamerCheckpoint,
) -> DreamerShadowModels:
    if checkpoint.artifact_stage != TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW:
        raise DreamerCheckpointMaterializationError("only train-preview checkpoints contain model tensors")
    if sys.byteorder != "little":
        raise DreamerCheckpointMaterializationError("checkpoint materialization requires a little-endian runtime")

    architecture = checkpoint.architecture
    if architecture is None or checkpoint.inference_probe_sha256 is None:
        raise DreamerCheckpointMaterializationError("checkpoint runtime architecture or inference probe is missing")
    world_config = _world_model_config(architecture.world_model)
    imagination_config = _imagination_config(architecture.imagination)
    world_model = DreamerV3VectorWorldModel(
        observation_dim=architecture.observation_dim,
        behavior_dim=architecture.behavior_dim,
        static_dim=architecture.static_dim,
        config=world_config,
    )
    actor = DreamerV3ImaginationActor(
        feature_dim=world_model.feature_dim,
        dynamic_action_dim=architecture.dynamic_action_dim,
        config=imagination_config,
    )
    critic = DreamerV3ImaginationCritic(
        feature_dim=world_model.feature_dim,
        config=imagination_config,
    )

    modules = {
        "world_model": world_model,
        "actor": actor,
        "critic": critic,
    }
    expected_names: set[str] = set()
    for component_name, module in modules.items():
        expected_names.update(f"{component_name}.{name}" for name in module.state_dict())
        expected_names.add(f"{component_name}.{_COMPONENT_BUFFER_NAMES[component_name]}")
    actual_names = {tensor.name for tensor in checkpoint.tensors}
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing[:5])}")
        if extra:
            details.append(f"extra {', '.join(extra[:5])}")
        raise DreamerCheckpointMaterializationError(
            f"checkpoint parameter names are incompatible: {'; '.join(details)}"
        )

    for component_name, module in modules.items():
        state = {}
        for parameter_name, expected_tensor in module.state_dict().items():
            checkpoint_name = f"{component_name}.{parameter_name}"
            loaded = _decode_tensor(checkpoint, checkpoint_name)
            if tuple(loaded.shape) != tuple(expected_tensor.shape):
                raise DreamerCheckpointMaterializationError(
                    f"checkpoint parameter {checkpoint_name} shape is incompatible"
                )
            state[parameter_name] = loaded
        try:
            module.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise DreamerCheckpointMaterializationError(
                f"checkpoint {component_name} state dictionary is incompatible"
            ) from exc
        expected_buffer = getattr(module, _COMPONENT_BUFFER_NAMES[component_name])
        stored_buffer = _decode_tensor(
            checkpoint,
            f"{component_name}.{_COMPONENT_BUFFER_NAMES[component_name]}",
        )
        if not torch.equal(stored_buffer, expected_buffer.detach().cpu()):
            raise DreamerCheckpointMaterializationError(
                f"checkpoint {component_name} fixed bins are incompatible"
            )
        module.requires_grad_(False)
        module.eval()

    actual_probe_sha256 = dreamer_inference_probe_sha256(
        world_model=world_model,
        actor=actor,
        critic=critic,
        architecture=architecture,
    )
    if actual_probe_sha256 != checkpoint.inference_probe_sha256:
        raise DreamerCheckpointMaterializationError("checkpoint deterministic inference probe does not match")
    return DreamerShadowModels(
        world_model=world_model,
        actor=actor,
        critic=critic,
        imagination_config=imagination_config,
        inference_probe_sha256=actual_probe_sha256,
    )


@torch.no_grad()
def dreamer_inference_probe_sha256(
    *,
    world_model: DreamerV3VectorWorldModel,
    actor: DreamerV3ImaginationActor,
    critic: DreamerV3ImaginationCritic,
    architecture: DreamerCheckpointArchitecture,
) -> str:
    with _deterministic_cpu_execution():
        world_model.eval()
        actor.eval()
        critic.eval()
        batch = _probe_batch(architecture)
        observed = world_model.observe(batch, sample=False)
        features = observed["features"][:, -1]
        control_mask = batch["control_action_mask"][:, -1]
        actor_output = actor(features, control_mask)
        behavior = behavior_tensor_from_parts(
            observed_profile_targets=batch["observed_profile_targets"][:, -1],
            observed_profile_target_mask=batch["observed_profile_target_mask"][:, -1],
            dynamic_actions=actor_output["dynamic_actions"],
            dynamic_action_mask=actor_output["dynamic_action_mask"],
            control_action_mask=control_mask,
            constraints=batch["constraints"][:, -1],
            decision_step_mask=batch["decision_step_mask"][:, -1],
        )
        imagined = world_model.imagine_step(
            observed["deter"][:, -1],
            observed["stoch"][:, -1],
            behavior,
            sample=False,
        )
        outputs = {
            "actor.dynamic_actions": actor_output["dynamic_actions"],
            "actor.static_actions": actor_output["static_actions"],
            "actor.static_logits": actor_output["static_logits"],
            "critic.logits": critic(features),
            "critic.value": critic.value(features),
            "probe.imagine_features": imagined["features"],
            "probe.imagine_prior_logits": imagined["prior_logits"],
            "world.continuation": world_model.continuation_probability(features),
            "world.features": observed["features"],
            "world.posterior_logits": observed["posterior_logits"],
            "world.prior_logits": observed["prior_logits"],
            "world.reward": world_model.reward_prediction(features),
        }
    return _tensor_output_sha256(DREAMER_INFERENCE_PROBE_FORMAT, outputs)


@torch.no_grad()
def dreamer_batch_inference_sha256(
    *,
    world_model: DreamerV3VectorWorldModel,
    actor: DreamerV3ImaginationActor,
    critic: DreamerV3ImaginationCritic,
    batch: dict[str, torch.Tensor],
) -> str:
    with _deterministic_cpu_execution():
        world_model.eval()
        actor.eval()
        critic.eval()
        observed = world_model.observe(batch, sample=False)
        features = observed["features"]
        actor_output = actor(features, batch["control_action_mask"])
        outputs = {
            "actor.dynamic_actions": actor_output["dynamic_actions"],
            "actor.static_actions": actor_output["static_actions"],
            "actor.static_logits": actor_output["static_logits"],
            "critic.logits": critic(features),
            "critic.value": critic.value(features),
            "world.continuation": world_model.continuation_probability(features),
            "world.features": features,
            "world.observation": world_model.observation_decoder(features),
            "world.posterior_logits": observed["posterior_logits"],
            "world.prior_logits": observed["prior_logits"],
            "world.reward": world_model.reward_prediction(features),
        }
    return _tensor_output_sha256("espresso_rl_dreamer_v3_heldout_inference_v1", outputs)


def _tensor_output_sha256(format_name: str, outputs: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(format_name.encode("ascii"))
    for name in sorted(outputs):
        tensor = outputs[name].detach().cpu().contiguous().to(dtype=torch.float32)
        descriptor = json.dumps(
            {"name": name, "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(descriptor).to_bytes(4, "little"))
        digest.update(descriptor)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@contextmanager
def _deterministic_cpu_execution():
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(old_deterministic)
        torch.set_num_threads(old_threads)


def _probe_batch(architecture: DreamerCheckpointArchitecture) -> dict[str, torch.Tensor]:
    batch_size = 2
    step_count = 3
    observation_dim = architecture.observation_dim
    dynamic_dim = architecture.dynamic_action_dim
    expected_behavior_dim = observation_dim * 2 + dynamic_dim * 4 + 1
    if expected_behavior_dim != architecture.behavior_dim:
        raise DreamerCheckpointMaterializationError(
            "checkpoint behavior dimension is incompatible with the probe feature layout"
        )

    observations = _sequence_tensor(batch_size, step_count, observation_dim, scale=0.03125)
    observed_targets = _sequence_tensor(batch_size, step_count, observation_dim, scale=0.015625)
    dynamic_actions = _sequence_tensor(batch_size, step_count, dynamic_dim, scale=0.0078125)
    control_mask = torch.ones((batch_size, step_count, dynamic_dim), dtype=torch.float32)
    control_mask[:, 1, 1::2] = 0.0
    dynamic_action_mask = control_mask.clone()
    constraints = torch.ones((batch_size, step_count, dynamic_dim), dtype=torch.float32)
    return {
        "observations": observations,
        "observed_profile_targets": observed_targets,
        "observed_profile_target_mask": torch.ones_like(observed_targets),
        "dynamic_actions": dynamic_actions * dynamic_action_mask,
        "dynamic_action_mask": dynamic_action_mask,
        "control_action_mask": control_mask,
        "constraints": constraints,
        "decision_step_mask": torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32),
        "rewards": torch.zeros((batch_size, step_count), dtype=torch.float32),
        "continuations": torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32),
        "step_mask": torch.ones((batch_size, step_count), dtype=torch.float32),
        "static_context": _sequence_tensor(batch_size, 1, architecture.static_dim, scale=0.00390625)[:, 0],
    }


def _sequence_tensor(batch_size: int, step_count: int, feature_dim: int, *, scale: float) -> torch.Tensor:
    count = batch_size * step_count * feature_dim
    values = torch.arange(1, count + 1, dtype=torch.float32) * scale
    return values.reshape(batch_size, step_count, feature_dim)


def _decode_tensor(checkpoint: VerifiedDreamerCheckpoint, name: str) -> torch.Tensor:
    descriptor = checkpoint.tensor(name)
    if descriptor.dtype != "F32":
        raise DreamerCheckpointMaterializationError(f"checkpoint parameter {name} dtype is incompatible")
    raw = bytearray(checkpoint.tensor_bytes(name))
    tensor = torch.frombuffer(raw, dtype=torch.float32).clone()
    try:
        return tensor.reshape(descriptor.shape)
    except RuntimeError as exc:
        raise DreamerCheckpointMaterializationError(
            f"checkpoint parameter {name} element count is incompatible"
        ) from exc


def _world_model_config(value: DreamerWorldModelArchitecture) -> DreamerV3WorldModelConfig:
    return DreamerV3WorldModelConfig(
        model_preset=value.model_preset,
        deter_dim=value.deter_dim,
        hidden_dim=value.hidden_dim,
        stoch_size=value.stoch_size,
        class_size=value.class_size,
        action_embed_dim=value.action_embed_dim,
        reward_bins=value.reward_bins,
        unimix=value.unimix,
        free_nats=value.free_nats,
        dyn_loss_scale=value.dyn_loss_scale,
        rep_loss_scale=value.rep_loss_scale,
        observation_loss_scale=value.observation_loss_scale,
        reward_loss_scale=value.reward_loss_scale,
        continuation_loss_scale=value.continuation_loss_scale,
    )


def _imagination_config(value: DreamerImaginationArchitecture) -> DreamerV3ImaginationConfig:
    return DreamerV3ImaginationConfig(
        horizon=value.horizon,
        actor_hidden_dim=value.actor_hidden_dim,
        critic_hidden_dim=value.critic_hidden_dim,
        value_bins=value.value_bins,
        discount=value.discount,
        lambda_return=value.lambda_return,
        actor_entropy_scale=value.actor_entropy_scale,
    )

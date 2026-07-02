from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from espresso_rl.domain.model_checkpoint import (
    DREAMER_INFERENCE_PROBE_FORMAT,
    DreamerContextEncoderArchitecture,
    DreamerCheckpointArchitecture,
    DreamerImaginationArchitecture,
    DreamerWorldModelArchitecture,
    VerifiedDreamerCheckpoint,
)
from espresso_rl.domain.dreamer_control import DreamerControlSpec
from espresso_rl.domain.dreamer_live_action import DreamerLiveActionSpec
from espresso_rl.domain.trainer_artifacts import (
    TRAINER_ARTIFACT_STAGE_WORLD_MODEL_RELEASE_CANDIDATE,
    TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW,
)
from espresso_rl.dreamer.imagination import (
    DreamerV3ImaginationActor,
    DreamerV3ImaginationConfig,
    DreamerV3ImaginationCritic,
)
from espresso_rl.dreamer.context_encoder import DreamerContextEncoder, DreamerContextEncoderConfig
from espresso_rl.dreamer.reference_world_model import (
    DreamerV3VectorWorldModel,
    DreamerV3WorldModelConfig,
    behavior_tensor_from_parts,
)
from espresso_rl.dreamer.dataset import (
    DREAMER_CONTEXT_WINDOW_SIZE,
)

_COMPONENT_BUFFER_NAMES = {
    "world_model": ("reward_bins",),
    "actor": (
        "pre_shot_action_bins",
        "pre_shot_action_bin_counts",
        "live_action_bins",
        "live_action_bin_counts",
    ),
    "critic": ("value_bins",),
}
_TENSOR_BEARING_STAGES = frozenset(
    {
        TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW,
        TRAINER_ARTIFACT_STAGE_WORLD_MODEL_RELEASE_CANDIDATE,
    }
)


class DreamerCheckpointMaterializationError(ValueError):
    pass


@dataclass(frozen=True)
class DreamerShadowModels:
    world_model: DreamerV3VectorWorldModel
    context_encoder: DreamerContextEncoder
    actor: DreamerV3ImaginationActor
    critic: DreamerV3ImaginationCritic
    imagination_config: DreamerV3ImaginationConfig
    inference_probe_sha256: str


def checkpoint_architecture_from_models(
    *,
    world_model: DreamerV3VectorWorldModel,
    context_encoder: DreamerContextEncoder,
    actor: DreamerV3ImaginationActor,
    critic: DreamerV3ImaginationCritic,
    observation_dim: int,
    behavior_dim: int,
    static_dim: int,
    live_action_dim: int,
    control_spec: DreamerControlSpec,
    live_action_spec: DreamerLiveActionSpec,
) -> DreamerCheckpointArchitecture:
    if actor.config != critic.config:
        raise DreamerCheckpointMaterializationError("actor and critic imagination configs do not match")
    if actor.taste_objective_spec != critic.taste_objective_spec:
        raise DreamerCheckpointMaterializationError("actor and critic taste-objective specs do not match")
    if actor.control_spec != control_spec:
        raise DreamerCheckpointMaterializationError("actor and checkpoint control specs do not match")
    if actor.live_action_spec != live_action_spec:
        raise DreamerCheckpointMaterializationError("actor and checkpoint live-action specs do not match")
    model = world_model.config
    imagination = actor.config
    return DreamerCheckpointArchitecture(
        observation_dim=observation_dim,
        behavior_dim=behavior_dim,
        static_dim=static_dim,
        live_action_dim=live_action_dim,
        taste_objective_dim=actor.taste_objective_dim,
        control_spec=control_spec,
        pre_shot_action_spec=actor.pre_shot_action_spec,
        live_action_spec=actor.live_action_spec,
        taste_objective_spec=critic.taste_objective_spec,
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
        context_encoder=DreamerContextEncoderArchitecture(
            static_dim=context_encoder.static_dim,
            terminal_dim=context_encoder.terminal_dim,
            time_dim=context_encoder.time_dim,
            trajectory_dim=context_encoder.trajectory_dim,
            hidden_dim=context_encoder.config.hidden_dim,
            context_dim=context_encoder.config.context_dim,
        ),
        imagination=DreamerImaginationArchitecture(
            horizon=imagination.horizon,
            actor_hidden_dim=imagination.actor_hidden_dim,
            critic_hidden_dim=imagination.critic_hidden_dim,
            value_bins=imagination.value_bins,
            discount=imagination.discount,
            lambda_return=imagination.lambda_return,
            actor_entropy_scale=imagination.actor_entropy_scale,
            pre_shot_behavior_loss_scale=imagination.pre_shot_behavior_loss_scale,
        ),
    )


def materialize_verified_dreamer_checkpoint(
    checkpoint: VerifiedDreamerCheckpoint,
) -> DreamerShadowModels:
    if checkpoint.artifact_stage not in _TENSOR_BEARING_STAGES:
        raise DreamerCheckpointMaterializationError("checkpoint does not contain model tensors")
    if sys.byteorder != "little":
        raise DreamerCheckpointMaterializationError("checkpoint materialization requires a little-endian runtime")

    architecture = checkpoint.architecture
    if architecture is None or checkpoint.inference_probe_sha256 is None:
        raise DreamerCheckpointMaterializationError("checkpoint runtime architecture or inference probe is missing")
    world_config = _world_model_config(architecture.world_model)
    context_config = _context_encoder_config(architecture.context_encoder)
    imagination_config = _imagination_config(architecture.imagination)
    world_model = DreamerV3VectorWorldModel(
        observation_dim=architecture.observation_dim,
        behavior_dim=architecture.behavior_dim,
        static_dim=architecture.static_dim,
        config=world_config,
    )
    context_encoder = DreamerContextEncoder(
        static_dim=architecture.context_encoder.static_dim,
        terminal_dim=architecture.context_encoder.terminal_dim,
        time_dim=architecture.context_encoder.time_dim,
        trajectory_dim=architecture.context_encoder.trajectory_dim,
        config=context_config,
    )
    actor = DreamerV3ImaginationActor(
        feature_dim=world_model.feature_dim,
        taste_objective_dim=architecture.taste_objective_dim,
        config=imagination_config,
        control_spec=architecture.control_spec,
        pre_shot_action_spec=architecture.pre_shot_action_spec,
        live_action_spec=architecture.live_action_spec,
        taste_objective_spec=architecture.taste_objective_spec,
    )
    critic = DreamerV3ImaginationCritic(
        feature_dim=world_model.feature_dim,
        taste_objective_dim=architecture.taste_objective_dim,
        config=imagination_config,
        taste_objective_spec=architecture.taste_objective_spec,
    )

    modules = {
        "world_model": world_model,
        "context_encoder": context_encoder,
        "actor": actor,
        "critic": critic,
    }
    expected_names: set[str] = set()
    for component_name, module in modules.items():
        expected_names.update(f"{component_name}.{name}" for name in module.state_dict())
        for buffer_name in _COMPONENT_BUFFER_NAMES.get(component_name, ()):
            expected_names.add(f"{component_name}.{buffer_name}")
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
        for buffer_name in _COMPONENT_BUFFER_NAMES.get(component_name, ()):
            expected_buffer = getattr(module, buffer_name)
            stored_buffer = _decode_tensor(checkpoint, f"{component_name}.{buffer_name}")
            if not torch.equal(stored_buffer, expected_buffer.detach().cpu()):
                raise DreamerCheckpointMaterializationError(
                    f"checkpoint {component_name} fixed buffer {buffer_name} shape is incompatible"
                )
        module.requires_grad_(False)
        module.eval()

    actual_probe_sha256 = dreamer_inference_probe_sha256(
        world_model=world_model,
        context_encoder=context_encoder,
        actor=actor,
        critic=critic,
        architecture=architecture,
    )
    if actual_probe_sha256 != checkpoint.inference_probe_sha256:
        raise DreamerCheckpointMaterializationError("checkpoint deterministic inference probe does not match")
    return DreamerShadowModels(
        world_model=world_model,
        context_encoder=context_encoder,
        actor=actor,
        critic=critic,
        imagination_config=imagination_config,
        inference_probe_sha256=actual_probe_sha256,
    )


@torch.no_grad()
def dreamer_inference_probe_sha256(
    *,
    world_model: DreamerV3VectorWorldModel,
    context_encoder: DreamerContextEncoder,
    actor: DreamerV3ImaginationActor,
    critic: DreamerV3ImaginationCritic,
    architecture: DreamerCheckpointArchitecture,
) -> str:
    with _deterministic_cpu_execution():
        world_model.eval()
        context_encoder.eval()
        actor.eval()
        critic.eval()
        batch = _probe_batch(architecture)
        context_state = context_encoder(batch)
        observed = world_model.observe(batch, context_state=context_state, sample=False)
        features = observed["features"][:, -1]
        control_mask = batch["control_action_mask"][:, -1]
        taste_objective = batch["taste_objective"]
        actor_output = actor(
            features,
            taste_objective,
            batch["pre_shot_capability_mask"],
            control_mask,
            batch["resolved_controls"][:, -1],
            batch["resolved_control_mask"][:, -1],
        )
        behavior = behavior_tensor_from_parts(
            resolved_controls=actor_output["resolved_controls"],
            resolved_control_mask=actor_output["resolved_control_mask"],
            control_action_mask=control_mask,
            constraints=batch["constraints"][:, -1],
            decision_step_mask=batch["decision_step_mask"][:, -1],
            pre_shot_actions=actor_output["pre_shot_actions"],
            pre_shot_action_mask=actor_output["pre_shot_action_mask"],
            pre_shot_capability_mask=batch["pre_shot_capability_mask"],
        )
        imagined = world_model.imagine_step(
            observed["deter"][:, -1],
            observed["stoch"][:, -1],
            behavior,
            sample=False,
        )
        outputs = {
            "actor.resolved_controls": actor_output["resolved_controls"],
            "actor.live_action_choices": actor_output["live_action_choices"],
            "actor.live_action_logits": actor_output["live_action_logits"],
            "actor.pre_shot_actions": actor_output["pre_shot_actions"],
            "actor.pre_shot_logits": actor_output["pre_shot_logits"],
            "context.mask": batch["context_mask"],
            "context.source_training_row_ids": batch["context_source_training_row_ids"].to(dtype=torch.float32),
            "context.static": batch["context_static"],
            "context.terminal": batch["context_terminal"],
            "context.time": batch["context_time"],
            "context.trajectory_embedding": batch["context_trajectory_embedding"],
            "context.encoded_state": context_state,
            "critic.logits": critic(features, taste_objective),
            "critic.value": critic.value(features, taste_objective),
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
    context_encoder: DreamerContextEncoder,
    actor: DreamerV3ImaginationActor,
    critic: DreamerV3ImaginationCritic,
    batch: dict[str, torch.Tensor],
) -> str:
    with _deterministic_cpu_execution():
        world_model.eval()
        context_encoder.eval()
        actor.eval()
        critic.eval()
        context_state = context_encoder(batch)
        observed = world_model.observe(batch, context_state=context_state, sample=False)
        features = observed["features"]
        taste_objective = batch["taste_objective"].unsqueeze(1).expand(
            -1,
            features.shape[1],
            -1,
        )
        dynamic_output = actor.select_dynamic(
            features,
            taste_objective,
            batch["control_action_mask"],
            batch["resolved_controls"],
            batch["resolved_control_mask"],
        )
        pre_shot_output = actor.select_pre_shot(
            features[:, -1],
            batch["taste_objective"],
            batch["pre_shot_capability_mask"],
        )
        outputs = {
            "actor.resolved_controls": dynamic_output["resolved_controls"],
            "actor.live_action_choices": dynamic_output["live_action_choices"],
            "actor.live_action_logits": dynamic_output["live_action_logits"],
            "actor.pre_shot_actions": pre_shot_output["pre_shot_actions"],
            "actor.pre_shot_logits": pre_shot_output["pre_shot_logits"],
            "critic.logits": critic(features, taste_objective),
            "critic.value": critic.value(features, taste_objective),
            "context.encoded_state": context_state,
            "world.continuation": world_model.continuation_probability(features),
            "world.features": features,
            "world.observation": world_model.observation_decoder(features),
            "world.posterior_logits": observed["posterior_logits"],
            "world.prior_logits": observed["prior_logits"],
            "world.reward": world_model.reward_prediction(features),
        }
        for key in (
            "context_mask",
            "context_source_training_row_ids",
            "context_static",
            "context_terminal",
            "context_time",
            "context_trajectory_embedding",
        ):
            if key in batch:
                outputs[f"context.{key}"] = batch[key].to(dtype=torch.float32)
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
    dynamic_dim = architecture.live_action_dim
    pre_shot_dim = len(architecture.pre_shot_action_spec.bins)
    expected_behavior_dim = dynamic_dim * 4 + 1 + pre_shot_dim * 3
    if expected_behavior_dim != architecture.behavior_dim:
        raise DreamerCheckpointMaterializationError(
            "checkpoint behavior dimension is incompatible with the probe feature layout"
        )

    observations = _sequence_tensor(batch_size, step_count, observation_dim, scale=0.03125)
    resolved_controls = torch.tensor(
        [
            [1.0, 8.0, 0.0, 1.0, 93.0, 36.0, 0.0],
            [1.0, 8.5, 0.0, 1.0, 93.0, 36.0, 0.0],
            [2.0, 0.0, 2.5, 1.0, 92.5, 38.0, 0.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0).expand(batch_size, -1, -1).clone()
    resolved_control_mask = torch.ones_like(resolved_controls)
    resolved_control_mask[:, :2, 2] = 0.0
    resolved_control_mask[:, 2, 1] = 0.0
    control_mask = torch.ones((batch_size, step_count, dynamic_dim), dtype=torch.float32)
    control_mask[:, 1, 1::2] = 0.0
    constraints = torch.ones((batch_size, step_count, dynamic_dim), dtype=torch.float32)
    pre_shot_actions = _sequence_tensor(batch_size, 1, pre_shot_dim, scale=0.00390625)[:, 0]
    pre_shot_capability_mask = torch.ones_like(pre_shot_actions)
    taste_objective = torch.zeros((batch_size, architecture.taste_objective_dim), dtype=torch.float32)
    taste_objective[:, 0] = 1.0
    return {
        "observations": observations,
        "resolved_controls": resolved_controls * resolved_control_mask,
        "resolved_control_mask": resolved_control_mask,
        "control_action_mask": control_mask,
        "constraints": constraints,
        "pre_shot_actions": pre_shot_actions,
        "pre_shot_action_indexes": torch.zeros((batch_size, pre_shot_dim), dtype=torch.long),
        "pre_shot_action_mask": pre_shot_capability_mask.clone(),
        "pre_shot_capability_mask": pre_shot_capability_mask,
        "taste_objective": taste_objective,
        "decision_step_mask": torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32),
        "rewards": torch.zeros((batch_size, step_count), dtype=torch.float32),
        "continuations": torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32),
        "step_mask": torch.ones((batch_size, step_count), dtype=torch.float32),
        "static_context": _sequence_tensor(batch_size, 1, architecture.static_dim, scale=0.00390625)[:, 0],
        "context_static": _sequence_tensor(
            batch_size,
            DREAMER_CONTEXT_WINDOW_SIZE,
            architecture.static_dim,
            scale=0.001953125,
        ),
        "context_terminal": _sequence_tensor(
            batch_size,
            DREAMER_CONTEXT_WINDOW_SIZE,
            architecture.context_encoder.terminal_dim,
            scale=0.0009765625,
        ),
        "context_time": _sequence_tensor(
            batch_size,
            DREAMER_CONTEXT_WINDOW_SIZE,
            architecture.context_encoder.time_dim,
            scale=0.25,
        ),
        "context_trajectory_embedding": _sequence_tensor(
            batch_size,
            DREAMER_CONTEXT_WINDOW_SIZE,
            architecture.context_encoder.trajectory_dim,
            scale=0.00048828125,
        ),
        "context_mask": _probe_context_mask(batch_size),
        "context_source_training_row_ids": torch.arange(
            1,
            batch_size * DREAMER_CONTEXT_WINDOW_SIZE + 1,
            dtype=torch.long,
        ).reshape(batch_size, DREAMER_CONTEXT_WINDOW_SIZE),
    }


def _sequence_tensor(batch_size: int, step_count: int, feature_dim: int, *, scale: float) -> torch.Tensor:
    count = batch_size * step_count * feature_dim
    values = torch.arange(1, count + 1, dtype=torch.float32) * scale
    return values.reshape(batch_size, step_count, feature_dim)


def _probe_context_mask(batch_size: int) -> torch.Tensor:
    row = [
        1.0 if index < DREAMER_CONTEXT_WINDOW_SIZE // 2 else 0.0
        for index in range(DREAMER_CONTEXT_WINDOW_SIZE)
    ]
    return torch.tensor([row] * batch_size, dtype=torch.float32)


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


def _context_encoder_config(value: DreamerContextEncoderArchitecture) -> DreamerContextEncoderConfig:
    return DreamerContextEncoderConfig(
        hidden_dim=value.hidden_dim,
        context_dim=value.context_dim,
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
        pre_shot_behavior_loss_scale=value.pre_shot_behavior_loss_scale,
    )

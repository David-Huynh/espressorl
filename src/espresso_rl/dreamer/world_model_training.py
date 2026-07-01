from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from espresso_rl.domain.dreamer_control import DreamerControlSpec
from espresso_rl.domain.dreamer_pre_shot import (
    DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC,
    DreamerPreShotActionSpec,
)
from espresso_rl.domain.dreamer_taste import (
    DEFAULT_DREAMER_TASTE_OBJECTIVE_SPEC,
    DreamerTasteObjectiveSpec,
)
from espresso_rl.domain.model_checkpoint import DreamerCheckpointArchitecture
from espresso_rl.dreamer.checkpoint_inference import (
    checkpoint_architecture_from_models,
    dreamer_inference_probe_sha256,
    dreamer_batch_inference_sha256,
)
from espresso_rl.dreamer.context_encoder import DreamerContextEncoder, DreamerContextEncoderConfig
from espresso_rl.dreamer.imagination import (
    DreamerV3ImaginationActor,
    DreamerV3ImaginationConfig,
    DreamerV3ImaginationCritic,
    dreamer_v3_imagination_rollout,
    run_dreamer_v3_imagination_preview,
)
from espresso_rl.dreamer.reference_world_model import (
    DreamerV3VectorWorldModel,
    DreamerV3WorldModelConfig,
    default_world_model_config,
)

WORLD_MODEL_SMOKE_FORMAT = "espresso_rl_world_model_smoke_v1"
WORLD_MODEL_SMOKE_SCHEMA_VERSION = 1
WORLD_MODEL_TRAIN_PREVIEW_FORMAT = "espresso_rl_world_model_train_preview_v1"
WORLD_MODEL_TRAIN_PREVIEW_SCHEMA_VERSION = 1
DREAMER_V3_EVALUATION_REPORT_FORMAT = "espresso_rl_dreamer_v3_offline_evaluation_report_v1"
DREAMER_V3_EVALUATION_REPORT_SCHEMA_VERSION = 1
_EVAL_WORLD_MODEL_LOSS_MAX = 1_000_000.0
_EVAL_REWARD_RMSE_MAX = 10.0
_EVAL_CRITIC_VALUE_RMSE_MAX = 10.0
_EVAL_UNSUPPORTED_DYNAMIC_ACTION_MAX = 0.0


@dataclass(frozen=True)
class WorldModelSmokeResult:
    seed: int
    train_steps: int
    model: DreamerV3WorldModelConfig
    initial: dict[str, float]
    final: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": WORLD_MODEL_SMOKE_FORMAT,
            "schema_version": WORLD_MODEL_SMOKE_SCHEMA_VERSION,
            "seed": self.seed,
            "train_steps": self.train_steps,
            "device": "cpu",
            "dtype": "float32",
            "model_config": self.model.to_dict(),
            "initial": self.initial,
            "final": self.final,
            "loss_delta_total": round(self.initial["loss_total"] - self.final["loss_total"], 8),
        }


@dataclass(frozen=True)
class WorldModelTrainPreviewConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    gradient_steps_per_epoch: int
    model: DreamerV3WorldModelConfig
    validation_split: float
    early_stop_patience: int
    control_spec: DreamerControlSpec
    pre_shot_action_spec: DreamerPreShotActionSpec = DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC
    taste_objective_spec: DreamerTasteObjectiveSpec = DEFAULT_DREAMER_TASTE_OBJECTIVE_SPEC
    imagination_horizon: int = 3
    imagination_actor_hidden_dim: int = 32
    imagination_critic_hidden_dim: int = 32
    imagination_actor_entropy_scale: float = 0.0003
    pre_shot_behavior_loss_scale: float = 1.0
    imagination_lambda_return: float = 0.95
    imagination_discount: float = 0.997
    actor_critic_train_steps: int = 3
    actor_learning_rate: float = 0.0003
    critic_learning_rate: float = 0.0003
    imagination_batch_size: int = 4
    actor_critic_gradient_clip_norm: float = 10.0


@dataclass(frozen=True)
class WorldModelTrainPreviewResult:
    config: WorldModelTrainPreviewConfig
    dataset_split: dict[str, Any]
    train_loss_curve: tuple[dict[str, float], ...]
    validation_loss_curve: tuple[dict[str, float], ...]
    best_epoch: int
    epochs_completed: int
    early_stopped: bool
    actor_critic_train_curve: tuple[dict[str, float], ...]
    imagination_preview: dict[str, Any]
    evaluation_report: dict[str, Any]
    checkpoint_tensors: dict[str, torch.Tensor]
    checkpoint_architecture: DreamerCheckpointArchitecture
    inference_probe_sha256: str
    heldout_inference_sha256: str
    parity_batch: dict[str, torch.Tensor]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": WORLD_MODEL_TRAIN_PREVIEW_FORMAT,
            "schema_version": WORLD_MODEL_TRAIN_PREVIEW_SCHEMA_VERSION,
            "seed": self.config.seed,
            "device": "cpu",
            "dtype": "float32",
            "epochs_requested": self.config.epochs,
            "epochs_completed": self.epochs_completed,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "gradient_steps_per_epoch": self.config.gradient_steps_per_epoch,
            "model_config": self.config.model.to_dict(),
            "validation_split": self.config.validation_split,
            "early_stop_patience": self.config.early_stop_patience,
            "imagination_horizon": self.config.imagination_horizon,
            "imagination_actor_hidden_dim": self.config.imagination_actor_hidden_dim,
            "imagination_critic_hidden_dim": self.config.imagination_critic_hidden_dim,
            "imagination_actor_entropy_scale": self.config.imagination_actor_entropy_scale,
            "pre_shot_behavior_loss_scale": self.config.pre_shot_behavior_loss_scale,
            "imagination_lambda_return": self.config.imagination_lambda_return,
            "imagination_discount": self.config.imagination_discount,
            "actor_critic_train_steps": self.config.actor_critic_train_steps,
            "actor_learning_rate": self.config.actor_learning_rate,
            "critic_learning_rate": self.config.critic_learning_rate,
            "imagination_batch_size": self.config.imagination_batch_size,
            "actor_critic_gradient_clip_norm": self.config.actor_critic_gradient_clip_norm,
            "best_epoch": self.best_epoch,
            "early_stopped": self.early_stopped,
            "dataset_split": self.dataset_split,
            "dataset_split_sha256": str(
                self.dataset_split.get("dataset_split_sha256") or _sha256_json(self.dataset_split)
            ),
            "train_loss_curve": list(self.train_loss_curve),
            "validation_loss_curve": list(self.validation_loss_curve),
            "actor_critic_train_curve": list(self.actor_critic_train_curve),
            "imagination_preview": self.imagination_preview,
            "evaluation_report": self.evaluation_report,
            "checkpoint_tensor_names": sorted(self.checkpoint_tensors),
            "checkpoint_architecture": self.checkpoint_architecture.to_dict(),
            "inference_probe_sha256": self.inference_probe_sha256,
            "heldout_inference_sha256": self.heldout_inference_sha256,
        }


class FixedCadenceWorldModelTrainingError(ValueError):
    pass


FixedCadenceWorldModelSmokeError = FixedCadenceWorldModelTrainingError


def run_fixed_cadence_world_model_smoke_train(
    batch: dict[str, Any],
    *,
    seed: int,
    train_steps: int,
    learning_rate: float = 1e-3,
    model_config: DreamerV3WorldModelConfig | None = None,
) -> WorldModelSmokeResult:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise FixedCadenceWorldModelTrainingError("world model smoke seed must be a uint32 integer")
    if isinstance(train_steps, bool) or not isinstance(train_steps, int) or not 1 <= train_steps <= 20:
        raise FixedCadenceWorldModelTrainingError("world model smoke train_steps must be 1..20")
    _validate_learning_rate(learning_rate, "world model smoke")
    model_config = model_config or default_world_model_config("espresso_debug")
    _validate_reference_model_config(model_config)

    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        tensors = _cpu_float_batch(batch)
        _validate_batch_shapes(tensors)
        model = DreamerV3VectorWorldModel(
            observation_dim=tensors["observations"].shape[-1],
            behavior_dim=_behavior_tensor(tensors).shape[-1],
            static_dim=tensors["static_context"].shape[-1],
            config=model_config,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        initial = _evaluate(model, None, tensors)
        for _ in range(train_steps):
            _train_epoch(model, None, optimizer, tensors, batch_size=tensors["observations"].shape[0])
        final = _evaluate(model, None, tensors)
    finally:
        torch.use_deterministic_algorithms(old_deterministic)
        torch.set_num_threads(old_threads)
    return WorldModelSmokeResult(seed=seed, train_steps=train_steps, model=model_config, initial=initial, final=final)


def run_fixed_cadence_world_model_train_preview(
    *,
    train_batch: dict[str, Any],
    validation_batch: dict[str, Any],
    config: WorldModelTrainPreviewConfig,
    dataset_split: dict[str, Any],
) -> WorldModelTrainPreviewResult:
    _validate_preview_config(config)
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config.seed)
        train_tensors = _cpu_float_batch(train_batch)
        validation_tensors = _cpu_float_batch(validation_batch)
        _validate_batch_shapes(train_tensors)
        _validate_batch_shapes(validation_tensors)
        model = DreamerV3VectorWorldModel(
            observation_dim=train_tensors["observations"].shape[-1],
            behavior_dim=_behavior_tensor(train_tensors).shape[-1],
            static_dim=train_tensors["static_context"].shape[-1],
            config=config.model,
        )
        context_encoder = _context_encoder_for_batch(train_tensors, config.model)
        optimizer = torch.optim.Adam(
            [*model.parameters(), *context_encoder.parameters()],
            lr=config.learning_rate,
        )
        train_curve: list[dict[str, float]] = []
        validation_curve: list[dict[str, float]] = []
        best_epoch = 0
        best_validation_loss = math.inf
        epochs_without_improvement = 0
        early_stopped = False

        for epoch_index in range(1, config.epochs + 1):
            repeated_losses = [
                _train_epoch(model, context_encoder, optimizer, train_tensors, batch_size=config.batch_size)
                for _ in range(config.gradient_steps_per_epoch)
            ]
            epoch_losses = _mean_loss_dicts(repeated_losses)
            validation_losses = _evaluate(model, context_encoder, validation_tensors)
            train_curve.append({"epoch": epoch_index, **epoch_losses})
            validation_curve.append({"epoch": epoch_index, **validation_losses})

            validation_total = validation_losses["loss_total"]
            if validation_total < best_validation_loss - 1e-8:
                best_validation_loss = validation_total
                best_epoch = epoch_index
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stop_patience:
                    early_stopped = True
                    break
        imagination_config = DreamerV3ImaginationConfig(
            horizon=config.imagination_horizon,
            actor_hidden_dim=config.imagination_actor_hidden_dim,
            critic_hidden_dim=config.imagination_critic_hidden_dim,
            value_bins=config.model.reward_bins,
            discount=config.imagination_discount,
            lambda_return=config.imagination_lambda_return,
            actor_entropy_scale=config.imagination_actor_entropy_scale,
            pre_shot_behavior_loss_scale=config.pre_shot_behavior_loss_scale,
        )
        torch.manual_seed((config.seed + 0x9E3779B9) % (2**32))
        actor = DreamerV3ImaginationActor(
            feature_dim=model.feature_dim,
            dynamic_action_dim=validation_tensors["dynamic_actions"].shape[-1],
            taste_objective_dim=validation_tensors["taste_objective"].shape[-1],
            config=imagination_config,
            pre_shot_action_spec=config.pre_shot_action_spec,
            taste_objective_spec=config.taste_objective_spec,
        )
        critic = DreamerV3ImaginationCritic(
            feature_dim=model.feature_dim,
            taste_objective_dim=validation_tensors["taste_objective"].shape[-1],
            config=imagination_config,
            taste_objective_spec=config.taste_objective_spec,
        )
        actor_critic_curve = _train_actor_critic(
            world_model=model,
            context_encoder=context_encoder,
            actor=actor,
            critic=critic,
            batch=train_tensors,
            config=config,
            imagination_config=imagination_config,
        )
        imagination_preview = run_dreamer_v3_imagination_preview(
            world_model=model,
            context_encoder=context_encoder,
            batch=validation_tensors,
            config=imagination_config,
            actor=actor,
            critic=critic,
        )
        evaluation_report = _offline_evaluation_report(
            world_model=model,
            context_encoder=context_encoder,
            actor=actor,
            critic=critic,
            batch=validation_tensors,
            config=config,
            imagination_config=imagination_config,
        )
        checkpoint_tensors = _checkpoint_tensors(model, context_encoder, actor, critic)
        _freeze_runtime_modules(model, context_encoder, actor, critic)
        checkpoint_architecture = checkpoint_architecture_from_models(
            world_model=model,
            context_encoder=context_encoder,
            actor=actor,
            critic=critic,
            observation_dim=int(train_tensors["observations"].shape[-1]),
            behavior_dim=int(_behavior_tensor(train_tensors).shape[-1]),
            static_dim=int(train_tensors["static_context"].shape[-1]),
            dynamic_action_dim=int(train_tensors["dynamic_actions"].shape[-1]),
            control_spec=config.control_spec,
        )
        inference_probe_sha256 = dreamer_inference_probe_sha256(
            world_model=model,
            context_encoder=context_encoder,
            actor=actor,
            critic=critic,
            architecture=checkpoint_architecture,
        )
        heldout_inference_sha256 = dreamer_batch_inference_sha256(
            world_model=model,
            context_encoder=context_encoder,
            actor=actor,
            critic=critic,
            batch=validation_tensors,
        )
    finally:
        torch.use_deterministic_algorithms(old_deterministic)
        torch.set_num_threads(old_threads)

    return WorldModelTrainPreviewResult(
        config=config,
        dataset_split=dataset_split,
        train_loss_curve=tuple(train_curve),
        validation_loss_curve=tuple(validation_curve),
        best_epoch=best_epoch,
        epochs_completed=len(train_curve),
        early_stopped=early_stopped,
        actor_critic_train_curve=tuple(actor_critic_curve),
        imagination_preview=imagination_preview,
        evaluation_report=evaluation_report,
        checkpoint_tensors=checkpoint_tensors,
        checkpoint_architecture=checkpoint_architecture,
        inference_probe_sha256=inference_probe_sha256,
        heldout_inference_sha256=heldout_inference_sha256,
        parity_batch=validation_tensors,
    )


def _cpu_float_batch(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    required = (
        "observations",
        "observed_profile_targets",
        "observed_profile_target_mask",
        "dynamic_actions",
        "dynamic_action_mask",
        "pre_shot_actions",
        "pre_shot_action_mask",
        "pre_shot_capability_mask",
        "control_action_mask",
        "constraints",
        "decision_step_mask",
        "rewards",
        "continuations",
        "step_mask",
        "static_context",
        "taste_objective",
    )
    converted: dict[str, torch.Tensor] = {}
    for key in required:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise FixedCadenceWorldModelTrainingError(f"world model training batch is missing tensor {key}")
        converted[key] = value.detach().to(device="cpu", dtype=torch.float32)
    pre_shot_indexes = batch.get("pre_shot_action_indexes")
    if not isinstance(pre_shot_indexes, torch.Tensor):
        raise FixedCadenceWorldModelTrainingError(
            "world model training batch is missing tensor pre_shot_action_indexes"
        )
    converted["pre_shot_action_indexes"] = pre_shot_indexes.detach().to(device="cpu", dtype=torch.long)
    for key in (
        "context_static",
        "context_terminal",
        "context_time",
        "context_trajectory_embedding",
        "context_mask",
        "context_source_training_row_ids",
    ):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            converted[key] = value.detach().to(device="cpu", dtype=torch.float32)
    return converted


def _validate_batch_shapes(batch: dict[str, torch.Tensor]) -> None:
    observations = batch["observations"]
    if observations.ndim != 3:
        raise FixedCadenceWorldModelTrainingError(
            "world model training observations must have shape (batch, steps, features)"
        )
    batch_size, step_count, _ = observations.shape
    if batch_size <= 0 or step_count <= 1:
        raise FixedCadenceWorldModelTrainingError("world model training batch must contain at least two valid steps")
    for key in (
        "observed_profile_targets",
        "observed_profile_target_mask",
        "dynamic_actions",
        "dynamic_action_mask",
        "control_action_mask",
        "constraints",
    ):
        tensor = batch[key]
        if tensor.ndim != 3 or tensor.shape[:2] != (batch_size, step_count):
            raise FixedCadenceWorldModelTrainingError(f"world model training {key} shape is invalid")
    for key in ("decision_step_mask", "rewards", "continuations", "step_mask"):
        tensor = batch[key]
        if tensor.ndim != 2 or tensor.shape != (batch_size, step_count):
            raise FixedCadenceWorldModelTrainingError(f"world model training {key} shape is invalid")
    static_context = batch["static_context"]
    if static_context.ndim != 2 or static_context.shape[0] != batch_size:
        raise FixedCadenceWorldModelTrainingError("world model training static_context shape is invalid")
    taste_objective = batch["taste_objective"]
    if taste_objective.ndim != 2 or taste_objective.shape[0] != batch_size:
        raise FixedCadenceWorldModelTrainingError("world model training taste_objective shape is invalid")
    pre_shot_actions = batch["pre_shot_actions"]
    for key in (
        "pre_shot_action_indexes",
        "pre_shot_action_mask",
        "pre_shot_capability_mask",
    ):
        tensor = batch[key]
        if tensor.ndim != 2 or tensor.shape != pre_shot_actions.shape:
            raise FixedCadenceWorldModelTrainingError(f"world model training {key} shape is invalid")
    if pre_shot_actions.ndim != 2 or pre_shot_actions.shape[0] != batch_size:
        raise FixedCadenceWorldModelTrainingError("world model training pre_shot_actions shape is invalid")
    if float(batch["step_mask"].sum().item()) < 2.0:
        raise FixedCadenceWorldModelTrainingError("world model training batch must contain at least two valid steps")


def _context_encoder_for_batch(
    batch: dict[str, torch.Tensor],
    model_config: DreamerV3WorldModelConfig,
) -> DreamerContextEncoder:
    for key in (
        "context_static",
        "context_terminal",
        "context_time",
        "context_trajectory_embedding",
        "context_mask",
    ):
        if key not in batch:
            raise FixedCadenceWorldModelTrainingError(f"world model preview batch is missing tensor {key}")
    return DreamerContextEncoder(
        static_dim=int(batch["context_static"].shape[-1]),
        terminal_dim=int(batch["context_terminal"].shape[-1]),
        time_dim=int(batch["context_time"].shape[-1]),
        trajectory_dim=int(batch["context_trajectory_embedding"].shape[-1]),
        config=DreamerContextEncoderConfig(
            hidden_dim=model_config.hidden_dim,
            context_dim=model_config.deter_dim,
        ),
    )


def _validate_preview_config(config: WorldModelTrainPreviewConfig) -> None:
    if isinstance(config.seed, bool) or not isinstance(config.seed, int) or not 0 <= config.seed <= 2**32 - 1:
        raise FixedCadenceWorldModelTrainingError("world model preview seed must be a uint32 integer")
    if isinstance(config.epochs, bool) or not isinstance(config.epochs, int) or not 1 <= config.epochs <= 50:
        raise FixedCadenceWorldModelTrainingError("world model preview epochs must be 1..50")
    if isinstance(config.batch_size, bool) or not isinstance(config.batch_size, int) or not 1 <= config.batch_size <= 128:
        raise FixedCadenceWorldModelTrainingError("world model preview batch_size must be 1..128")
    if (
        isinstance(config.gradient_steps_per_epoch, bool)
        or not isinstance(config.gradient_steps_per_epoch, int)
        or not 1 <= config.gradient_steps_per_epoch <= 32
    ):
        raise FixedCadenceWorldModelTrainingError("world model preview gradient_steps_per_epoch must be 1..32")
    if (
        isinstance(config.validation_split, bool)
        or not isinstance(config.validation_split, (int, float))
        or not math.isfinite(float(config.validation_split))
        or not 0.05 <= float(config.validation_split) <= 0.5
    ):
        raise FixedCadenceWorldModelTrainingError("world model preview validation_split must be 0.05..0.5")
    if (
        isinstance(config.early_stop_patience, bool)
        or not isinstance(config.early_stop_patience, int)
        or not 1 <= config.early_stop_patience <= 20
    ):
        raise FixedCadenceWorldModelTrainingError("world model preview early_stop_patience must be 1..20")
    _range_int(config.imagination_horizon, "imagination_horizon", 1, 32)
    _range_int(config.imagination_actor_hidden_dim, "imagination_actor_hidden_dim", 8, 2048)
    _range_int(config.imagination_critic_hidden_dim, "imagination_critic_hidden_dim", 8, 2048)
    _range_float(config.imagination_actor_entropy_scale, "imagination_actor_entropy_scale", 0.0, 1.0)
    _range_float(config.imagination_lambda_return, "imagination_lambda_return", 0.0, 1.0)
    _range_float(config.imagination_discount, "imagination_discount", 0.0, 1.0)
    _range_int(config.actor_critic_train_steps, "actor_critic_train_steps", 1, 128)
    _range_int(config.imagination_batch_size, "imagination_batch_size", 1, 128)
    _range_float(config.actor_critic_gradient_clip_norm, "actor_critic_gradient_clip_norm", 0.1, 100.0)
    _validate_learning_rate(config.actor_learning_rate, "actor")
    _validate_learning_rate(config.critic_learning_rate, "critic")
    _validate_reference_model_config(config.model)
    _validate_learning_rate(config.learning_rate, "world model preview")


def _behavior_tensor(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    step_count = batch["observed_profile_targets"].shape[1]
    return torch.cat(
        [
            batch["observed_profile_targets"],
            batch["observed_profile_target_mask"],
            batch["dynamic_actions"],
            batch["dynamic_action_mask"],
            batch["control_action_mask"],
            batch["constraints"],
            batch["decision_step_mask"].unsqueeze(-1),
            batch["pre_shot_actions"].unsqueeze(1).expand(-1, step_count, -1),
            batch["pre_shot_action_mask"].unsqueeze(1).expand(-1, step_count, -1),
            batch["pre_shot_capability_mask"].unsqueeze(1).expand(-1, step_count, -1),
        ],
        dim=-1,
    )


def _train_epoch(
    model: DreamerV3VectorWorldModel,
    context_encoder: DreamerContextEncoder | None,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> dict[str, float]:
    model.train()
    if context_encoder is not None:
        context_encoder.train()
    batch_count = batch["observations"].shape[0]
    accumulated: dict[str, float] = {}
    minibatches = 0
    for start in range(0, batch_count, batch_size):
        end = min(start + batch_size, batch_count)
        minibatch = _slice_batch(batch, start, end)
        optimizer.zero_grad()
        losses = model.losses(
            minibatch,
            context_state=_context_state(context_encoder, minibatch),
        )
        losses["loss_total"].backward()
        parameters = [*model.parameters()]
        if context_encoder is not None:
            parameters.extend(context_encoder.parameters())
        nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        for key in losses:
            accumulated.setdefault(key, 0.0)
            accumulated[key] += float(losses[key].item())
        minibatches += 1
    return {key: round(value / max(minibatches, 1), 8) for key, value in accumulated.items()}


def _slice_batch(batch: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    return {key: value[start:end] for key, value in batch.items()}


@torch.no_grad()
def _evaluate(
    model: DreamerV3VectorWorldModel,
    context_encoder: DreamerContextEncoder | None,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    model.eval()
    if context_encoder is not None:
        context_encoder.eval()
    losses = model.losses(
        batch,
        context_state=_context_state(context_encoder, batch),
        sample=False,
    )
    model.train()
    if context_encoder is not None:
        context_encoder.train()
    return {key: round(float(value.item()), 8) for key, value in losses.items()}


def _mean_loss_dicts(losses: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in losses for key in item})
    return {key: round(sum(item[key] for item in losses) / len(losses), 8) for key in keys}


def _train_actor_critic(
    *,
    world_model: DreamerV3VectorWorldModel,
    context_encoder: DreamerContextEncoder,
    actor: DreamerV3ImaginationActor,
    critic: DreamerV3ImaginationCritic,
    batch: dict[str, torch.Tensor],
    config: WorldModelTrainPreviewConfig,
    imagination_config: DreamerV3ImaginationConfig,
) -> list[dict[str, float]]:
    world_model.eval()
    context_encoder.eval()
    actor.train()
    critic.train()
    world_model_requires_grad = [parameter.requires_grad for parameter in world_model.parameters()]
    context_encoder_requires_grad = [parameter.requires_grad for parameter in context_encoder.parameters()]
    try:
        for parameter in world_model.parameters():
            parameter.requires_grad_(False)
        for parameter in context_encoder.parameters():
            parameter.requires_grad_(False)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate)
        curve: list[dict[str, float]] = []
        for step_index in range(1, config.actor_critic_train_steps + 1):
            minibatch = _cyclic_batch(
                batch,
                start=(step_index - 1) * config.imagination_batch_size,
                batch_size=config.imagination_batch_size,
            )
            critic_metrics = _train_critic_step(
                world_model=world_model,
                context_encoder=context_encoder,
                actor=actor,
                critic=critic,
                optimizer=critic_optimizer,
                batch=minibatch,
                imagination_config=imagination_config,
                gradient_clip_norm=config.actor_critic_gradient_clip_norm,
            )
            actor_metrics = _train_actor_step(
                world_model=world_model,
                context_encoder=context_encoder,
                actor=actor,
                critic=critic,
                optimizer=actor_optimizer,
                batch=minibatch,
                imagination_config=imagination_config,
                gradient_clip_norm=config.actor_critic_gradient_clip_norm,
            )
            curve.append(
                {
                    "step": float(step_index),
                    **critic_metrics,
                    **actor_metrics,
                }
            )
    finally:
        for parameter, requires_grad in zip(world_model.parameters(), world_model_requires_grad, strict=True):
            parameter.requires_grad_(requires_grad)
        for parameter, requires_grad in zip(context_encoder.parameters(), context_encoder_requires_grad, strict=True):
            parameter.requires_grad_(requires_grad)
    return curve


@torch.no_grad()
def _offline_evaluation_report(
    *,
    world_model: DreamerV3VectorWorldModel,
    context_encoder: DreamerContextEncoder,
    actor: DreamerV3ImaginationActor,
    critic: DreamerV3ImaginationCritic,
    batch: dict[str, torch.Tensor],
    config: WorldModelTrainPreviewConfig,
    imagination_config: DreamerV3ImaginationConfig,
) -> dict[str, Any]:
    world_model.eval()
    context_encoder.eval()
    actor.eval()
    critic.eval()
    validation_losses = _evaluate(world_model, context_encoder, batch)
    context_state = _context_state(context_encoder, batch)
    observed = world_model.observe(batch, context_state=context_state, sample=False)
    step_mask = batch["step_mask"]
    valid_count = step_mask.sum().clamp_min(1.0)
    features = observed["features"]
    reward_prediction = world_model.reward_prediction(features)
    reward_error = (reward_prediction - batch["rewards"]) * step_mask
    reward_mae = reward_error.abs().sum() / valid_count
    reward_rmse = torch.sqrt((reward_error.square().sum() / valid_count).clamp_min(0.0))
    continuation_prediction = world_model.continuation_probability(features)
    continuation_error = (continuation_prediction - batch["continuations"]) * step_mask
    continuation_mae = continuation_error.abs().sum() / valid_count
    continuation_rmse = torch.sqrt((continuation_error.square().sum() / valid_count).clamp_min(0.0))

    rollout = dreamer_v3_imagination_rollout(
        world_model=world_model,
        context_encoder=context_encoder,
        batch=batch,
        config=imagination_config,
        actor=actor,
        critic=critic,
    )
    critic_error = rollout["values"][:, :-1] - rollout["lambda_returns"]
    critic_value_mae = critic_error.abs().mean()
    critic_value_rmse = torch.sqrt(critic_error.square().mean().clamp_min(0.0))
    dynamic_actions = rollout["dynamic_actions"]
    control_mask = rollout["control_action_mask"]
    unsupported_dynamic = dynamic_actions * (1.0 - control_mask.unsqueeze(1))
    unsupported_abs_max = unsupported_dynamic.abs().max()
    imagined_returns = rollout["lambda_returns"]

    world_model_loss_ok = _finite_bounded(validation_losses["loss_total"], _EVAL_WORLD_MODEL_LOSS_MAX)
    reward_calibration_ok = _finite_bounded(reward_rmse, _EVAL_REWARD_RMSE_MAX)
    critic_value_ok = _finite_bounded(critic_value_rmse, _EVAL_CRITIC_VALUE_RMSE_MAX)
    action_mask_ok = bool(unsupported_abs_max.item() <= _EVAL_UNSUPPORTED_DYNAMIC_ACTION_MAX)

    return {
        "format": DREAMER_V3_EVALUATION_REPORT_FORMAT,
        "schema_version": DREAMER_V3_EVALUATION_REPORT_SCHEMA_VERSION,
        "inference_ready": False,
        "contract_only": True,
        "device": "cpu",
        "dtype": "float32",
        "seed": config.seed,
        "validation_batch_size": int(batch["observations"].shape[0]),
        "imagination_horizon": config.imagination_horizon,
        "world_model_validation": validation_losses,
        "reward_prediction": {
            "mae": _rounded_scalar(reward_mae),
            "rmse": _rounded_scalar(reward_rmse),
        },
        "continuation_prediction": {
            "mae": _rounded_scalar(continuation_mae),
            "rmse": _rounded_scalar(continuation_rmse),
        },
        "critic_value": {
            "mae": _rounded_scalar(critic_value_mae),
            "rmse": _rounded_scalar(critic_value_rmse),
        },
        "actor": {
            "entropy_mean": _rounded_scalar(rollout["pre_shot_entropy"].mean()),
            "pre_shot_behavior_loss": _rounded_scalar(rollout["pre_shot_behavior_loss"]),
            "imagined_return_mean": _rounded_scalar(imagined_returns.mean()),
            "imagined_return_std": _rounded_scalar(imagined_returns.std(unbiased=False)),
            "supported_dynamic_action_count": _rounded_scalar(control_mask.sum()),
            "unsupported_dynamic_action_abs_max": _rounded_scalar(unsupported_abs_max),
        },
        "gates": {
            "world_model_loss_ok": world_model_loss_ok,
            "reward_calibration_ok": reward_calibration_ok,
            "critic_value_ok": critic_value_ok,
            "action_mask_ok": action_mask_ok,
            "evaluation_passed": (
                world_model_loss_ok
                and reward_calibration_ok
                and critic_value_ok
                and action_mask_ok
            ),
        },
        "thresholds": {
            "world_model_loss_total_max": _EVAL_WORLD_MODEL_LOSS_MAX,
            "reward_rmse_max": _EVAL_REWARD_RMSE_MAX,
            "critic_value_rmse_max": _EVAL_CRITIC_VALUE_RMSE_MAX,
            "unsupported_dynamic_action_abs_max": _EVAL_UNSUPPORTED_DYNAMIC_ACTION_MAX,
        },
    }


def _train_critic_step(
    *,
    world_model: DreamerV3VectorWorldModel,
    context_encoder: DreamerContextEncoder,
    actor: DreamerV3ImaginationActor,
    critic: DreamerV3ImaginationCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    imagination_config: DreamerV3ImaginationConfig,
    gradient_clip_norm: float,
) -> dict[str, float]:
    actor.eval()
    critic.train()
    with torch.no_grad():
        rollout = dreamer_v3_imagination_rollout(
            world_model=world_model,
            context_encoder=context_encoder,
            batch=batch,
            config=imagination_config,
            actor=actor,
            critic=critic,
        )
    optimizer.zero_grad()
    step_mask = torch.ones_like(rollout["lambda_returns"])
    critic_loss = critic.loss(
        rollout["imagined_features"].detach(),
        rollout["taste_objective"].detach(),
        rollout["lambda_returns"].detach(),
        step_mask,
    )
    critic_loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), gradient_clip_norm)
    optimizer.step()
    return {
        "critic_loss": _rounded_scalar(critic_loss),
        "critic_grad_norm": _rounded_scalar(grad_norm),
    }


def _train_actor_step(
    *,
    world_model: DreamerV3VectorWorldModel,
    context_encoder: DreamerContextEncoder,
    actor: DreamerV3ImaginationActor,
    critic: DreamerV3ImaginationCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    imagination_config: DreamerV3ImaginationConfig,
    gradient_clip_norm: float,
) -> dict[str, float]:
    actor.train()
    critic.eval()
    critic_requires_grad = [parameter.requires_grad for parameter in critic.parameters()]
    try:
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
        optimizer.zero_grad()
        rollout = dreamer_v3_imagination_rollout(
            world_model=world_model,
            context_encoder=context_encoder,
            batch=batch,
            config=imagination_config,
            actor=actor,
            critic=critic,
        )
        advantage = (rollout["lambda_returns"] - rollout["values"][:, :-1]).detach()
        pre_shot_policy_loss = -(rollout["pre_shot_log_prob"] * advantage[:, 0]).mean()
        pre_shot_behavior_loss = rollout["pre_shot_behavior_loss"]
        imagined_return_loss = -rollout["lambda_returns"].mean()
        entropy = rollout["pre_shot_entropy"].mean()
        actor_loss = (
            imagined_return_loss
            + pre_shot_policy_loss
            + imagination_config.pre_shot_behavior_loss_scale * pre_shot_behavior_loss
            - imagination_config.actor_entropy_scale * entropy
        )
        actor_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), gradient_clip_norm)
        optimizer.step()
    finally:
        for parameter, requires_grad in zip(critic.parameters(), critic_requires_grad, strict=True):
            parameter.requires_grad_(requires_grad)

    dynamic_actions = rollout["dynamic_actions"]
    control_mask = rollout["control_action_mask"]
    unsupported_dynamic = dynamic_actions * (1.0 - control_mask.unsqueeze(1))
    return {
        "actor_loss": _rounded_scalar(actor_loss),
        "pre_shot_policy_loss": _rounded_scalar(pre_shot_policy_loss),
        "pre_shot_behavior_loss": _rounded_scalar(pre_shot_behavior_loss),
        "imagined_return_loss": _rounded_scalar(imagined_return_loss),
        "actor_entropy_mean": _rounded_scalar(entropy),
        "imagined_return_mean": _rounded_scalar(rollout["lambda_returns"].mean()),
        "actor_grad_norm": _rounded_scalar(grad_norm),
        "supported_dynamic_action_count": _rounded_scalar(control_mask.sum()),
        "unsupported_dynamic_action_abs_max": _rounded_scalar(unsupported_dynamic.abs().max()),
    }


def _cyclic_batch(batch: dict[str, torch.Tensor], *, start: int, batch_size: int) -> dict[str, torch.Tensor]:
    count = batch["observations"].shape[0]
    indexes = (torch.arange(batch_size, dtype=torch.long) + start) % count
    return {key: value.index_select(0, indexes) for key, value in batch.items()}


def _rounded_scalar(value: torch.Tensor) -> float:
    return round(float(value.detach().cpu().item()), 8)


def _finite_bounded(value: float | torch.Tensor, maximum: float) -> bool:
    parsed = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
    return math.isfinite(parsed) and parsed <= maximum


def _checkpoint_tensors(
    world_model: DreamerV3VectorWorldModel,
    context_encoder: DreamerContextEncoder,
    actor: DreamerV3ImaginationActor,
    critic: DreamerV3ImaginationCritic,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for component_name, module in (
        ("world_model", world_model),
        ("context_encoder", context_encoder),
        ("actor", actor),
        ("critic", critic),
    ):
        for tensor_name, tensor in module.state_dict().items():
            tensors[f"{component_name}.{tensor_name}"] = tensor.detach().cpu().contiguous().to(dtype=torch.float32)
    tensors["world_model.reward_bins"] = world_model.reward_bins.detach().cpu().contiguous().to(dtype=torch.float32)
    tensors["actor.pre_shot_action_bins"] = (
        actor.pre_shot_action_bins.detach().cpu().contiguous().to(dtype=torch.float32)
    )
    tensors["actor.pre_shot_action_bin_counts"] = (
        actor.pre_shot_action_bin_counts.detach().cpu().contiguous().to(dtype=torch.float32)
    )
    tensors["critic.value_bins"] = critic.value_bins.detach().cpu().contiguous().to(dtype=torch.float32)
    return tensors


def _freeze_runtime_modules(*modules: nn.Module) -> None:
    for module in modules:
        module.requires_grad_(False)
        module.eval()


def _context_state(
    context_encoder: DreamerContextEncoder | None,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    if context_encoder is None:
        return None
    return context_encoder(batch)


def _validate_reference_model_config(config: DreamerV3WorldModelConfig) -> None:
    if not isinstance(config, DreamerV3WorldModelConfig):
        raise FixedCadenceWorldModelTrainingError("world model preview model config is invalid")
    _range_int(config.deter_dim, "deter_dim", 8, 2048)
    _range_int(config.hidden_dim, "hidden_dim", 8, 2048)
    _range_int(config.stoch_size, "stoch_size", 2, 64)
    _range_int(config.class_size, "class_size", 2, 128)
    _range_int(config.action_embed_dim, "action_embed_dim", 4, 256)
    _range_int(config.reward_bins, "reward_bins", 3, 255)
    if config.reward_bins % 2 == 0:
        raise FixedCadenceWorldModelTrainingError("world model preview reward_bins must be odd")
    _range_float(config.unimix, "unimix", 0.0, 0.2)
    _range_float(config.free_nats, "free_nats", 0.0, 10.0)
    _range_float(config.dyn_loss_scale, "dyn_loss_scale", 0.0, 10.0)
    _range_float(config.rep_loss_scale, "rep_loss_scale", 0.0, 10.0)
    _range_float(config.observation_loss_scale, "observation_loss_scale", 0.0, 10.0)
    _range_float(config.reward_loss_scale, "reward_loss_scale", 0.0, 10.0)
    _range_float(config.continuation_loss_scale, "continuation_loss_scale", 0.0, 10.0)


def _validate_learning_rate(value: object, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 1e-6 <= float(value) <= 0.1
    ):
        raise FixedCadenceWorldModelTrainingError(f"{label} learning_rate is invalid")


def _range_int(value: object, label: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise FixedCadenceWorldModelTrainingError(f"world model preview {label} is invalid")


def _range_float(value: object, label: str, minimum: float, maximum: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise FixedCadenceWorldModelTrainingError(f"world model preview {label} is invalid")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()

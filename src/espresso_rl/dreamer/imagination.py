from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from espresso_rl.dreamer.dataset import DREAMER_DYNAMIC_ACTION_FEATURES
from espresso_rl.dreamer.reference_world_model import (
    DreamerV3VectorWorldModel,
    behavior_tensor_from_parts,
    symexp_twohot_bins,
    twohot_cross_entropy,
    twohot_prediction,
)

DREAMER_V3_IMAGINATION_PREVIEW_FORMAT = "espresso_rl_dreamer_v3_imagination_preview_v1"
DREAMER_V3_IMAGINATION_PREVIEW_SCHEMA_VERSION = 1
DREAMER_STATIC_RECIPE_ACTION_HEADS = (
    "grind_delta_steps_from_current",
    "dose_delta_g",
    "yield_delta_g",
)
DREAMER_STATIC_RECIPE_ACTION_BINS = {
    "grind_delta_steps_from_current": (-4.0, -2.0, 0.0, 2.0, 4.0),
    "dose_delta_g": (-1.0, -0.5, 0.0, 0.5, 1.0),
    "yield_delta_g": (-6.0, -3.0, 0.0, 3.0, 6.0),
}


@dataclass(frozen=True)
class DreamerV3ImaginationConfig:
    horizon: int = 3
    actor_hidden_dim: int = 32
    critic_hidden_dim: int = 32
    value_bins: int = 41
    discount: float = 0.997
    lambda_return: float = 0.95
    actor_entropy_scale: float = 0.0003

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "actor_hidden_dim": self.actor_hidden_dim,
            "critic_hidden_dim": self.critic_hidden_dim,
            "value_bins": self.value_bins,
            "discount": self.discount,
            "lambda_return": self.lambda_return,
            "actor_entropy_scale": self.actor_entropy_scale,
        }


class DreamerV3ImaginationActor(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        dynamic_action_dim: int,
        config: DreamerV3ImaginationConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.dynamic_action_dim = dynamic_action_dim
        self.static_head_count = len(DREAMER_STATIC_RECIPE_ACTION_HEADS)
        self.static_bin_count = len(next(iter(DREAMER_STATIC_RECIPE_ACTION_BINS.values())))
        self.trunk = _mlp(feature_dim, config.actor_hidden_dim, config.actor_hidden_dim)
        self.static_head = nn.Linear(config.actor_hidden_dim, self.static_head_count * self.static_bin_count)
        self.dynamic_head = nn.Linear(config.actor_hidden_dim, dynamic_action_dim)
        self.register_buffer("static_action_bins", _static_action_bin_tensor(), persistent=False)

    def forward(self, features: torch.Tensor, control_action_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(features)
        static_logits = self.static_head(hidden).reshape(
            *features.shape[:-1],
            self.static_head_count,
            self.static_bin_count,
        )
        static_indexes = torch.argmax(static_logits, dim=-1)
        static_actions = _gather_static_action_bins(self.static_action_bins, static_indexes)
        static_log_probs = F.log_softmax(static_logits, dim=-1)
        selected_static_log_prob = static_log_probs.gather(-1, static_indexes.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        static_entropy = -(F.softmax(static_logits, dim=-1) * static_log_probs).sum(dim=(-1, -2))

        dynamic_action_mask = (control_action_mask > 0.5).to(dtype=features.dtype)
        dynamic_actions = torch.tanh(self.dynamic_head(hidden)) * dynamic_action_mask
        return {
            "static_logits": static_logits,
            "static_action_indexes": static_indexes,
            "static_actions": static_actions,
            "static_log_prob": selected_static_log_prob,
            "static_entropy": static_entropy,
            "dynamic_actions": dynamic_actions,
            "dynamic_action_mask": dynamic_action_mask,
        }


class DreamerV3ImaginationCritic(nn.Module):
    def __init__(self, *, feature_dim: int, config: DreamerV3ImaginationConfig) -> None:
        super().__init__()
        self.config = config
        self.net = _mlp(feature_dim, config.critic_hidden_dim, config.value_bins)
        self.register_buffer("value_bins", symexp_twohot_bins(config.value_bins), persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return twohot_prediction(self(features), self.value_bins)

    def loss(
        self,
        features: torch.Tensor,
        target_returns: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> torch.Tensor:
        losses = twohot_cross_entropy(self(features), target_returns, self.value_bins)
        return (losses * step_mask).sum() / step_mask.sum().clamp_min(1.0)


def lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continuations: torch.Tensor,
    *,
    discount: float,
    lambda_return: float,
) -> torch.Tensor:
    if rewards.ndim != 2 or continuations.shape != rewards.shape:
        raise ValueError("lambda returns require rewards and continuations with shape (batch, horizon)")
    if values.shape != (rewards.shape[0], rewards.shape[1] + 1):
        raise ValueError("lambda returns require values with shape (batch, horizon + 1)")
    next_return = values[:, -1]
    targets: list[torch.Tensor] = []
    for step_index in range(rewards.shape[1] - 1, -1, -1):
        bootstrap = (1.0 - lambda_return) * values[:, step_index + 1] + lambda_return * next_return
        next_return = rewards[:, step_index] + discount * continuations[:, step_index] * bootstrap
        targets.append(next_return)
    return torch.stack(list(reversed(targets)), dim=1)


@torch.no_grad()
def run_dreamer_v3_imagination_preview(
    *,
    world_model: DreamerV3VectorWorldModel,
    batch: dict[str, torch.Tensor],
    config: DreamerV3ImaginationConfig,
    actor: DreamerV3ImaginationActor | None = None,
    critic: DreamerV3ImaginationCritic | None = None,
) -> dict[str, Any]:
    validate_imagination_config(config)
    world_model.eval()
    if actor is None:
        actor = DreamerV3ImaginationActor(
            feature_dim=world_model.feature_dim,
            dynamic_action_dim=batch["dynamic_actions"].shape[-1],
            config=config,
        )
    if critic is None:
        critic = DreamerV3ImaginationCritic(feature_dim=world_model.feature_dim, config=config)
    observed = world_model.observe(batch, sample=False)
    start_indexes = _last_decision_indexes(batch)
    batch_indexes = torch.arange(batch["observations"].shape[0], device=batch["observations"].device)
    deter = observed["deter"][batch_indexes, start_indexes]
    stoch = observed["stoch"][batch_indexes, start_indexes]
    features = observed["features"][batch_indexes, start_indexes]
    control_mask = _decision_control_mask(batch)
    constraints = batch["constraints"].amax(dim=1)

    imagined_features: list[torch.Tensor] = []
    dynamic_actions: list[torch.Tensor] = []
    static_actions: list[torch.Tensor] = []
    static_logits: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    continuations: list[torch.Tensor] = []
    values: list[torch.Tensor] = [critic.value(features)]
    value_logits: list[torch.Tensor] = [critic(features)]

    for _ in range(config.horizon):
        imagined_features.append(features)
        actor_output = actor(features, control_mask)
        static_actions.append(actor_output["static_actions"])
        static_logits.append(actor_output["static_logits"])
        dynamic_actions.append(actor_output["dynamic_actions"])
        log_probs.append(actor_output["static_log_prob"])
        entropies.append(actor_output["static_entropy"])
        behavior = behavior_tensor_from_parts(
            observed_profile_targets=torch.zeros(
                features.shape[0],
                5,
                dtype=features.dtype,
                device=features.device,
            ),
            observed_profile_target_mask=torch.zeros(
                features.shape[0],
                5,
                dtype=features.dtype,
                device=features.device,
            ),
            dynamic_actions=actor_output["dynamic_actions"],
            dynamic_action_mask=actor_output["dynamic_action_mask"],
            control_action_mask=control_mask,
            constraints=constraints,
            decision_step_mask=torch.ones(features.shape[0], dtype=features.dtype, device=features.device),
        )
        imagined = world_model.imagine_step(deter, stoch, behavior, sample=False)
        deter = imagined["deter"]
        stoch = imagined["stoch"]
        features = imagined["features"]
        rewards.append(world_model.reward_prediction(features))
        continuations.append(world_model.continuation_probability(features))
        values.append(critic.value(features))
        value_logits.append(critic(features))

    imagined_feature_tensor = torch.stack(imagined_features, dim=1)
    reward_tensor = torch.stack(rewards, dim=1)
    continuation_tensor = torch.stack(continuations, dim=1)
    value_tensor = torch.stack(values, dim=1)
    lambda_target_tensor = lambda_returns(
        reward_tensor,
        value_tensor,
        continuation_tensor,
        discount=config.discount,
        lambda_return=config.lambda_return,
    )
    step_mask = torch.ones_like(lambda_target_tensor)
    critic_loss = critic.loss(imagined_feature_tensor, lambda_target_tensor, step_mask)
    advantage = lambda_target_tensor - value_tensor[:, :-1]
    log_prob_tensor = torch.stack(log_probs, dim=1)
    entropy_tensor = torch.stack(entropies, dim=1)
    actor_loss = -(log_prob_tensor * advantage).mean() - config.actor_entropy_scale * entropy_tensor.mean()
    dynamic_action_tensor = torch.stack(dynamic_actions, dim=1)
    unsupported_dynamic = dynamic_action_tensor * (1.0 - control_mask.unsqueeze(1))

    return {
        "format": DREAMER_V3_IMAGINATION_PREVIEW_FORMAT,
        "schema_version": DREAMER_V3_IMAGINATION_PREVIEW_SCHEMA_VERSION,
        "inference_ready": False,
        "contract_only": True,
        "config": config.to_dict(),
        "static_action_heads": list(DREAMER_STATIC_RECIPE_ACTION_HEADS),
        "static_action_bins": {key: list(value) for key, value in DREAMER_STATIC_RECIPE_ACTION_BINS.items()},
        "dynamic_action_features": list(DREAMER_DYNAMIC_ACTION_FEATURES),
        "start_count": int(features.shape[0]),
        "feature_dim": int(world_model.feature_dim),
        "static_logits_shape": _shape(torch.stack(static_logits, dim=1)),
        "static_action_shape": _shape(torch.stack(static_actions, dim=1)),
        "dynamic_action_shape": _shape(dynamic_action_tensor),
        "control_action_mask_shape": _shape(control_mask),
        "supported_dynamic_action_count": int(control_mask.sum().item()),
        "unsupported_dynamic_action_abs_max": _round_float(unsupported_dynamic.abs().max()),
        "reward_prediction_shape": _shape(reward_tensor),
        "continuation_shape": _shape(continuation_tensor),
        "critic_value_logits_shape": _shape(torch.stack(value_logits, dim=1)),
        "lambda_return_shape": _shape(lambda_target_tensor),
        "actor_entropy_mean": _round_float(entropy_tensor.mean()),
        "actor_loss_preview": _round_float(actor_loss),
        "critic_loss_preview": _round_float(critic_loss),
    }


def validate_imagination_config(config: DreamerV3ImaginationConfig) -> None:
    if not isinstance(config, DreamerV3ImaginationConfig):
        raise ValueError("Dreamer imagination config is invalid")
    _range_int(config.horizon, "horizon", 1, 32)
    _range_int(config.actor_hidden_dim, "actor_hidden_dim", 8, 2048)
    _range_int(config.critic_hidden_dim, "critic_hidden_dim", 8, 2048)
    _range_int(config.value_bins, "value_bins", 3, 255)
    if config.value_bins % 2 == 0:
        raise ValueError("Dreamer imagination value_bins must be odd")
    _range_float(config.discount, "discount", 0.0, 1.0)
    _range_float(config.lambda_return, "lambda_return", 0.0, 1.0)
    _range_float(config.actor_entropy_scale, "actor_entropy_scale", 0.0, 1.0)


def _last_decision_indexes(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    decision_valid = batch["decision_step_mask"] * batch["step_mask"]
    step_indexes = torch.arange(decision_valid.shape[1], device=decision_valid.device, dtype=decision_valid.dtype)
    return (decision_valid * step_indexes.unsqueeze(0)).argmax(dim=1).to(dtype=torch.long)


def _decision_control_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    decision_valid = (batch["decision_step_mask"] * batch["step_mask"]).unsqueeze(-1)
    return (batch["control_action_mask"] * decision_valid).amax(dim=1)


def _static_action_bin_tensor() -> torch.Tensor:
    return torch.tensor(
        [DREAMER_STATIC_RECIPE_ACTION_BINS[name] for name in DREAMER_STATIC_RECIPE_ACTION_HEADS],
        dtype=torch.float32,
    )


def _gather_static_action_bins(action_bins: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
    expanded_bins = action_bins.to(device=indexes.device).expand(*indexes.shape[:-1], -1, -1)
    return expanded_bins.gather(-1, indexes.unsqueeze(-1)).squeeze(-1)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _range_int(value: object, label: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"Dreamer imagination {label} is invalid")


def _range_float(value: object, label: str, minimum: float, maximum: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ValueError(f"Dreamer imagination {label} is invalid")


def _shape(tensor: torch.Tensor) -> list[int]:
    return [int(dimension) for dimension in tensor.shape]


def _round_float(value: torch.Tensor) -> float:
    return round(float(value.item()), 8)

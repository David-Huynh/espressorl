from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from espresso_rl.dreamer.dataset import DREAMER_DYNAMIC_ACTION_FEATURES, DREAMER_OBSERVED_TARGET_FEATURES
from espresso_rl.domain.dreamer_control import DEFAULT_DREAMER_CONTROL_SPEC, DreamerControlSpec
from espresso_rl.domain.dreamer_live_action import (
    DEFAULT_DREAMER_LIVE_ACTION_SPEC,
    DREAMER_LIVE_ACTION_FIELDS,
    DreamerLiveActionSpec,
)
from espresso_rl.domain.dreamer_pre_shot import (
    DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC,
    DREAMER_PRE_SHOT_ACTION_FIELDS,
    DreamerPreShotActionSpec,
)
from espresso_rl.domain.dreamer_taste import (
    DEFAULT_DREAMER_TASTE_OBJECTIVE_SPEC,
    DREAMER_TASTE_OBJECTIVE_ATTRIBUTES,
    DreamerTasteObjectiveSpec,
)
from espresso_rl.dreamer.context_encoder import DreamerContextEncoder
from espresso_rl.dreamer.reference_world_model import (
    DreamerV3VectorWorldModel,
    behavior_tensor_from_parts,
    symexp_twohot_bins,
    twohot_cross_entropy,
    twohot_prediction,
)

DREAMER_V3_IMAGINATION_PREVIEW_FORMAT = "espresso_rl_dreamer_v3_imagination_preview_v1"
DREAMER_V3_IMAGINATION_PREVIEW_SCHEMA_VERSION = 1
@dataclass(frozen=True)
class DreamerV3ImaginationConfig:
    horizon: int = 3
    actor_hidden_dim: int = 32
    critic_hidden_dim: int = 32
    value_bins: int = 41
    discount: float = 0.997
    lambda_return: float = 0.95
    actor_entropy_scale: float = 0.0003
    pre_shot_behavior_loss_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "actor_hidden_dim": self.actor_hidden_dim,
            "critic_hidden_dim": self.critic_hidden_dim,
            "value_bins": self.value_bins,
            "discount": self.discount,
            "lambda_return": self.lambda_return,
            "actor_entropy_scale": self.actor_entropy_scale,
            "pre_shot_behavior_loss_scale": self.pre_shot_behavior_loss_scale,
        }


class DreamerV3ImaginationActor(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        taste_objective_dim: int,
        config: DreamerV3ImaginationConfig,
        control_spec: DreamerControlSpec = DEFAULT_DREAMER_CONTROL_SPEC,
        pre_shot_action_spec: DreamerPreShotActionSpec = DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC,
        live_action_spec: DreamerLiveActionSpec = DEFAULT_DREAMER_LIVE_ACTION_SPEC,
        taste_objective_spec: DreamerTasteObjectiveSpec = DEFAULT_DREAMER_TASTE_OBJECTIVE_SPEC,
    ) -> None:
        super().__init__()
        self.config = config
        self.control_spec = control_spec
        self.dynamic_target_dim = len(DREAMER_DYNAMIC_ACTION_FEATURES)
        self.taste_objective_dim = taste_objective_dim
        self.pre_shot_action_spec = pre_shot_action_spec
        self.live_action_spec = live_action_spec
        self.taste_objective_spec = taste_objective_spec
        if not isinstance(control_spec, DreamerControlSpec):
            raise ValueError("Dreamer actor control spec is invalid")
        if taste_objective_dim != 1 + len(DREAMER_TASTE_OBJECTIVE_ATTRIBUTES):
            raise ValueError("Dreamer actor taste-objective dimension does not match its spec")
        self.pre_shot_head_count = len(DREAMER_PRE_SHOT_ACTION_FIELDS)
        self.pre_shot_bin_counts_tuple = tuple(
            len(pre_shot_action_spec.bins[name]) for name in DREAMER_PRE_SHOT_ACTION_FIELDS
        )
        self.pre_shot_max_bin_count = max(self.pre_shot_bin_counts_tuple)
        self.live_head_count = len(DREAMER_LIVE_ACTION_FIELDS)
        self.live_bin_counts_tuple = tuple(
            len(live_action_spec.bins[name]) for name in DREAMER_LIVE_ACTION_FIELDS
        )
        self.live_max_bin_count = max(self.live_bin_counts_tuple)
        self.trunk = _mlp(
            feature_dim + taste_objective_dim,
            config.actor_hidden_dim,
            config.actor_hidden_dim,
        )
        self.pre_shot_heads = nn.ModuleList(
            nn.Linear(config.actor_hidden_dim, bin_count)
            for bin_count in self.pre_shot_bin_counts_tuple
        )
        self.live_heads = nn.ModuleList(
            nn.Linear(config.actor_hidden_dim, bin_count)
            for bin_count in self.live_bin_counts_tuple
        )
        self.register_buffer(
            "pre_shot_action_bins",
            _pre_shot_action_bin_tensor(pre_shot_action_spec),
            persistent=False,
        )
        self.register_buffer(
            "pre_shot_action_bin_counts",
            torch.tensor(self.pre_shot_bin_counts_tuple, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "live_action_bins",
            _live_action_bin_tensor(live_action_spec),
            persistent=False,
        )
        self.register_buffer(
            "live_action_bin_counts",
            torch.tensor(self.live_bin_counts_tuple, dtype=torch.float32),
            persistent=False,
        )
        dynamic_low, dynamic_high = _dynamic_action_bound_tensors(control_spec)
        self.register_buffer("dynamic_action_low", dynamic_low, persistent=False)
        self.register_buffer("dynamic_action_high", dynamic_high, persistent=False)

    def forward(
        self,
        features: torch.Tensor,
        taste_objective: torch.Tensor,
        pre_shot_capability_mask: torch.Tensor,
        control_action_mask: torch.Tensor,
        target_state: torch.Tensor | None = None,
        pump_measurements: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self._hidden(features, taste_objective)
        if target_state is None:
            target_state = torch.zeros(
                (*hidden.shape[:-1], self.dynamic_target_dim),
                dtype=hidden.dtype,
                device=hidden.device,
            )
        return {
            **self._pre_shot_output(hidden, pre_shot_capability_mask),
            **self._dynamic_output(hidden, control_action_mask, target_state, pump_measurements),
        }

    def select_pre_shot(
        self,
        features: torch.Tensor,
        taste_objective: torch.Tensor,
        capability_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self._pre_shot_output(self._hidden(features, taste_objective), capability_mask)

    def select_dynamic(
        self,
        features: torch.Tensor,
        taste_objective: torch.Tensor,
        control_action_mask: torch.Tensor,
        target_state: torch.Tensor,
        pump_measurements: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self._dynamic_output(
            self._hidden(features, taste_objective),
            control_action_mask,
            target_state,
            pump_measurements,
        )

    def _hidden(self, features: torch.Tensor, taste_objective: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2 or taste_objective.shape != (*features.shape[:-1], self.taste_objective_dim):
            raise ValueError("Dreamer actor taste-objective shape is incompatible with features")
        return self.trunk(torch.cat([features, taste_objective.to(dtype=features.dtype)], dim=-1))

    def _pre_shot_output(
        self,
        hidden: torch.Tensor,
        capability_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expected_shape = (*hidden.shape[:-1], self.pre_shot_head_count)
        if capability_mask.shape != expected_shape:
            raise ValueError("Dreamer pre-shot capability mask shape is invalid")
        capability_mask = (capability_mask > 0.5).to(dtype=hidden.dtype)
        padded_logits = torch.full(
            (*hidden.shape[:-1], self.pre_shot_head_count, self.pre_shot_max_bin_count),
            torch.finfo(hidden.dtype).min,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        indexes: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        selected_log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for head_index, (head, bin_count) in enumerate(
            zip(self.pre_shot_heads, self.pre_shot_bin_counts_tuple, strict=True)
        ):
            logits = head(hidden)
            padded_logits[..., head_index, :bin_count] = logits
            index = torch.argmax(logits, dim=-1)
            bins = self.pre_shot_action_bins[head_index, :bin_count].to(
                dtype=hidden.dtype,
                device=hidden.device,
            )
            indexes.append(index)
            values.append(bins[index])
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs.append(log_probs.gather(-1, index.unsqueeze(-1)).squeeze(-1))
            entropies.append(-(F.softmax(logits, dim=-1) * log_probs).sum(dim=-1))
        index_tensor = torch.stack(indexes, dim=-1)
        value_tensor = torch.stack(values, dim=-1) * capability_mask
        log_prob_tensor = torch.stack(selected_log_probs, dim=-1) * capability_mask
        entropy_tensor = torch.stack(entropies, dim=-1) * capability_mask
        capability_count = capability_mask.sum(dim=-1).clamp_min(1.0)
        return {
            "pre_shot_logits": padded_logits,
            "pre_shot_action_indexes": index_tensor,
            "pre_shot_actions": value_tensor,
            "pre_shot_action_mask": capability_mask,
            "pre_shot_log_prob": log_prob_tensor.sum(dim=-1),
            "pre_shot_entropy": entropy_tensor.sum(dim=-1) / capability_count,
        }

    def _dynamic_output(
        self,
        hidden: torch.Tensor,
        control_action_mask: torch.Tensor,
        target_state: torch.Tensor,
        pump_measurements: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if control_action_mask.shape != (*hidden.shape[:-1], self.dynamic_target_dim):
            raise ValueError("Dreamer dynamic action capability mask shape is invalid")
        if target_state.shape != (*hidden.shape[:-1], self.dynamic_target_dim):
            raise ValueError("Dreamer dynamic target state shape is invalid")
        if pump_measurements is None:
            pressure_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("pressure_target_bar")
            flow_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("flow_target_ml_s")
            pump_measurements = target_state[..., [pressure_index, flow_index]]
        if pump_measurements.shape != (*hidden.shape[:-1], 2):
            raise ValueError("Dreamer pump measurement shape is invalid")
        capability_mask = (control_action_mask > 0.5).to(dtype=hidden.dtype)
        dynamic_low = self.dynamic_action_low.to(dtype=hidden.dtype, device=hidden.device)
        dynamic_high = self.dynamic_action_high.to(dtype=hidden.dtype, device=hidden.device)
        padded_logits = torch.full(
            (*hidden.shape[:-1], self.live_head_count, self.live_max_bin_count),
            torch.finfo(hidden.dtype).min,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        indexes: list[torch.Tensor] = []
        choices: list[torch.Tensor] = []
        selected_log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for head_index, (head, bin_count) in enumerate(
            zip(self.live_heads, self.live_bin_counts_tuple, strict=True)
        ):
            logits = head(hidden)
            bins = self.live_action_bins[head_index, :bin_count].to(
                dtype=hidden.dtype,
                device=hidden.device,
            )
            if DREAMER_LIVE_ACTION_FIELDS[head_index] == "pump_target_mode":
                pressure_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("pressure_target_bar")
                flow_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("flow_target_ml_s")
                pressure_allowed = control_action_mask[..., pressure_index] > 0.5
                flow_allowed = control_action_mask[..., flow_index] > 0.5
                allowed = torch.stack([pressure_allowed, flow_allowed], dim=-1)
                any_allowed = allowed.any(dim=-1, keepdim=True)
                logits = torch.where(
                    allowed | ~any_allowed,
                    logits,
                    torch.full_like(logits, torch.finfo(logits.dtype).min),
                )
            padded_logits[..., head_index, :bin_count] = logits
            index = torch.argmax(logits, dim=-1)
            indexes.append(index)
            choices.append(bins[index])
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs.append(log_probs.gather(-1, index.unsqueeze(-1)).squeeze(-1))
            entropies.append(-(F.softmax(logits, dim=-1) * log_probs).sum(dim=-1))
        index_tensor = torch.stack(indexes, dim=-1)
        choice_tensor = torch.stack(choices, dim=-1)
        mode_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("pump_target_mode")
        pressure_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("pressure_target_bar")
        flow_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("flow_target_ml_s")
        selected_mode = choice_tensor[..., mode_index]
        dynamic_action_mask = capability_mask.clone()
        dynamic_action_mask[..., pressure_index] *= (selected_mode == 1.0).to(dtype=hidden.dtype)
        dynamic_action_mask[..., flow_index] *= (selected_mode == 2.0).to(dtype=hidden.dtype)
        choice_tensor = choice_tensor * dynamic_action_mask
        next_target_state = _apply_live_action_delta(
            target_state.to(dtype=hidden.dtype),
            choice_tensor,
            dynamic_low,
            dynamic_high,
            pump_measurements.to(dtype=hidden.dtype),
        )
        dynamic_actions = next_target_state * dynamic_action_mask
        log_prob_tensor = torch.stack(selected_log_probs, dim=-1) * dynamic_action_mask
        entropy_tensor = torch.stack(entropies, dim=-1) * dynamic_action_mask
        capability_count = dynamic_action_mask.sum(dim=-1).clamp_min(1.0)
        return {
            "live_action_logits": padded_logits,
            "live_action_indexes": index_tensor,
            "live_action_choices": choice_tensor,
            "live_action_mask": dynamic_action_mask,
            "live_log_prob": log_prob_tensor.sum(dim=-1),
            "live_entropy": entropy_tensor.sum(dim=-1) / capability_count,
            "dynamic_actions": dynamic_actions,
            "dynamic_action_mask": dynamic_action_mask,
            "dynamic_target_state": next_target_state,
        }


class DreamerV3ImaginationCritic(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        taste_objective_dim: int,
        config: DreamerV3ImaginationConfig,
        taste_objective_spec: DreamerTasteObjectiveSpec = DEFAULT_DREAMER_TASTE_OBJECTIVE_SPEC,
    ) -> None:
        super().__init__()
        self.config = config
        self.taste_objective_dim = taste_objective_dim
        self.taste_objective_spec = taste_objective_spec
        if taste_objective_dim != 1 + len(DREAMER_TASTE_OBJECTIVE_ATTRIBUTES):
            raise ValueError("Dreamer critic taste-objective dimension does not match its spec")
        self.net = _mlp(feature_dim + taste_objective_dim, config.critic_hidden_dim, config.value_bins)
        self.register_buffer("value_bins", symexp_twohot_bins(config.value_bins), persistent=False)

    def forward(self, features: torch.Tensor, taste_objective: torch.Tensor) -> torch.Tensor:
        return self.net(_condition_on_taste(features, taste_objective, self.taste_objective_dim))

    def value(self, features: torch.Tensor, taste_objective: torch.Tensor) -> torch.Tensor:
        return twohot_prediction(self(features, taste_objective), self.value_bins)

    def loss(
        self,
        features: torch.Tensor,
        taste_objective: torch.Tensor,
        target_returns: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> torch.Tensor:
        losses = twohot_cross_entropy(self(features, taste_objective), target_returns, self.value_bins)
        return (losses * step_mask).sum() / step_mask.sum().clamp_min(1.0)


def masked_pre_shot_behavior_loss(
    actor_output: dict[str, torch.Tensor],
    target_indexes: torch.Tensor,
    target_mask: torch.Tensor,
    bin_counts: tuple[int, ...],
) -> torch.Tensor:
    logits = actor_output["pre_shot_logits"]
    capability_mask = actor_output["pre_shot_action_mask"]
    if target_indexes.shape != target_mask.shape or target_mask.shape != capability_mask.shape:
        raise ValueError("Dreamer pre-shot behavior targets have incompatible shapes")
    if target_indexes.dtype != torch.long:
        raise ValueError("Dreamer pre-shot behavior indexes must be int64")
    observed_mask = (target_mask > 0.5).to(dtype=logits.dtype)
    if torch.any(observed_mask > capability_mask):
        raise ValueError("Dreamer pre-shot behavior target exceeds capability mask")
    losses: list[torch.Tensor] = []
    for head_index, bin_count in enumerate(bin_counts):
        indexes = target_indexes[..., head_index]
        active = observed_mask[..., head_index]
        if torch.any((indexes < 0) | ((indexes >= bin_count) & (active > 0.0))):
            raise ValueError("Dreamer pre-shot behavior index is outside action bins")
        log_probs = F.log_softmax(logits[..., head_index, :bin_count], dim=-1)
        losses.append(-log_probs.gather(-1, indexes.clamp(0, bin_count - 1).unsqueeze(-1)).squeeze(-1) * active)
    loss_tensor = torch.stack(losses, dim=-1)
    return loss_tensor.sum() / observed_mask.sum().clamp_min(1.0)


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
    context_encoder: DreamerContextEncoder | None = None,
    actor: DreamerV3ImaginationActor | None = None,
    critic: DreamerV3ImaginationCritic | None = None,
) -> dict[str, Any]:
    validate_imagination_config(config)
    world_model.eval()
    if actor is None:
        pre_shot_spec = _pre_shot_spec_from_batch(batch)
        actor = DreamerV3ImaginationActor(
            feature_dim=world_model.feature_dim,
            taste_objective_dim=batch["taste_objective"].shape[-1],
            config=config,
            control_spec=_control_spec_from_batch(batch),
            pre_shot_action_spec=pre_shot_spec,
            live_action_spec=_live_action_spec_from_batch(batch),
            taste_objective_spec=_taste_objective_spec_from_batch(batch),
        )
    if critic is None:
        critic = DreamerV3ImaginationCritic(
            feature_dim=world_model.feature_dim,
            taste_objective_dim=batch["taste_objective"].shape[-1],
            config=config,
            taste_objective_spec=_taste_objective_spec_from_batch(batch),
        )
    rollout = dreamer_v3_imagination_rollout(
        world_model=world_model,
        context_encoder=context_encoder,
        batch=batch,
        config=config,
        actor=actor,
        critic=critic,
    )
    dynamic_action_tensor = rollout["dynamic_actions"]
    control_mask = rollout["control_action_mask"]
    unsupported_dynamic = dynamic_action_tensor * (1.0 - control_mask.unsqueeze(1))

    return {
        "format": DREAMER_V3_IMAGINATION_PREVIEW_FORMAT,
        "schema_version": DREAMER_V3_IMAGINATION_PREVIEW_SCHEMA_VERSION,
        "inference_ready": False,
        "contract_only": True,
        "config": config.to_dict(),
        "pre_shot_action_heads": list(DREAMER_PRE_SHOT_ACTION_FIELDS),
        "pre_shot_action_bins": {
            key: list(actor.pre_shot_action_spec.bins[key]) for key in DREAMER_PRE_SHOT_ACTION_FIELDS
        },
        "live_action_heads": list(DREAMER_LIVE_ACTION_FIELDS),
        "live_action_bins": {
            key: list(actor.live_action_spec.bins[key]) for key in DREAMER_LIVE_ACTION_FIELDS
        },
        "dynamic_action_features": list(DREAMER_DYNAMIC_ACTION_FEATURES),
        "start_count": int(rollout["imagined_features"].shape[0]),
        "feature_dim": int(world_model.feature_dim),
        "pre_shot_logits_shape": _shape(rollout["pre_shot_logits"]),
        "pre_shot_action_shape": _shape(rollout["pre_shot_actions"]),
        "pre_shot_held_action_shape": _shape(rollout["pre_shot_actions_held"]),
        "live_action_logits_shape": _shape(rollout["live_action_logits"]),
        "live_action_choice_shape": _shape(rollout["live_action_choices"]),
        "dynamic_action_shape": _shape(dynamic_action_tensor),
        "control_action_mask_shape": _shape(control_mask),
        "supported_dynamic_action_count": int(control_mask.sum().item()),
        "unsupported_dynamic_action_abs_max": _round_float(unsupported_dynamic.abs().max()),
        "reward_prediction_shape": _shape(rollout["rewards"]),
        "continuation_shape": _shape(rollout["continuations"]),
        "critic_value_logits_shape": _shape(rollout["value_logits"]),
        "lambda_return_shape": _shape(rollout["lambda_returns"]),
        "actor_entropy_mean": _round_float(rollout["pre_shot_entropy"].mean()),
        "pre_shot_behavior_loss": _round_float(rollout["pre_shot_behavior_loss"]),
        "actor_loss_preview": _round_float(rollout["actor_loss_preview"]),
        "critic_loss_preview": _round_float(rollout["critic_loss_preview"]),
    }


def dreamer_v3_imagination_rollout(
    *,
    world_model: DreamerV3VectorWorldModel,
    context_encoder: DreamerContextEncoder | None = None,
    batch: dict[str, torch.Tensor],
    config: DreamerV3ImaginationConfig,
    actor: DreamerV3ImaginationActor,
    critic: DreamerV3ImaginationCritic,
) -> dict[str, torch.Tensor]:
    validate_imagination_config(config)
    world_model.eval()
    with torch.no_grad():
        observed = world_model.observe(
            batch,
            context_state=_context_state(context_encoder, batch),
            sample=False,
        )
    start_indexes = _last_decision_indexes(batch)
    batch_indexes = torch.arange(batch["observations"].shape[0], device=batch["observations"].device)
    deter = observed["deter"][batch_indexes, start_indexes].detach()
    stoch = observed["stoch"][batch_indexes, start_indexes].detach()
    features = observed["features"][batch_indexes, start_indexes].detach()
    control_mask = _decision_control_mask(batch)
    constraints = batch["constraints"].amax(dim=1)
    taste_objective = batch["taste_objective"]
    pre_shot_capability_mask = batch["pre_shot_capability_mask"]
    pre_shot_output = actor.select_pre_shot(
        features,
        taste_objective,
        pre_shot_capability_mask,
    )
    pre_shot_behavior_loss = masked_pre_shot_behavior_loss(
        pre_shot_output,
        batch["pre_shot_action_indexes"],
        batch["pre_shot_action_mask"],
        actor.pre_shot_bin_counts_tuple,
    )
    target_state = _initial_dynamic_target_state(
        batch,
        batch_indexes=batch_indexes,
        start_indexes=start_indexes,
        pre_shot_actions=pre_shot_output["pre_shot_actions"],
    )

    imagined_features: list[torch.Tensor] = []
    dynamic_actions: list[torch.Tensor] = []
    live_action_logits: list[torch.Tensor] = []
    live_action_indexes: list[torch.Tensor] = []
    live_action_choices: list[torch.Tensor] = []
    live_log_probs: list[torch.Tensor] = []
    live_entropies: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    continuations: list[torch.Tensor] = []
    values: list[torch.Tensor] = [critic.value(features, taste_objective)]
    value_logits: list[torch.Tensor] = [critic(features, taste_objective)]

    for _ in range(config.horizon):
        imagined_features.append(features)
        predicted_observation = world_model.observation_prediction(features)
        dynamic_output = actor.select_dynamic(
            features,
            taste_objective,
            control_mask,
            target_state,
            predicted_observation[..., :2],
        )
        target_state = dynamic_output["dynamic_target_state"]
        dynamic_actions.append(dynamic_output["dynamic_actions"])
        live_action_logits.append(dynamic_output["live_action_logits"])
        live_action_indexes.append(dynamic_output["live_action_indexes"])
        live_action_choices.append(dynamic_output["live_action_choices"])
        live_log_probs.append(dynamic_output["live_log_prob"])
        live_entropies.append(dynamic_output["live_entropy"])
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
            dynamic_actions=dynamic_output["dynamic_actions"],
            dynamic_action_mask=dynamic_output["dynamic_action_mask"],
            control_action_mask=control_mask,
            constraints=constraints,
            decision_step_mask=torch.ones(features.shape[0], dtype=features.dtype, device=features.device),
            pre_shot_actions=pre_shot_output["pre_shot_actions"],
            pre_shot_action_mask=pre_shot_output["pre_shot_action_mask"],
            pre_shot_capability_mask=pre_shot_capability_mask,
        )
        imagined = world_model.imagine_step(deter, stoch, behavior, sample=False)
        deter = imagined["deter"]
        stoch = imagined["stoch"]
        features = imagined["features"]
        rewards.append(world_model.reward_prediction(features))
        continuations.append(world_model.continuation_probability(features))
        values.append(critic.value(features, taste_objective))
        value_logits.append(critic(features, taste_objective))

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
    critic_loss = critic.loss(imagined_feature_tensor, taste_objective, lambda_target_tensor, step_mask)
    advantage = lambda_target_tensor - value_tensor[:, :-1]
    pre_shot_policy_loss = -(pre_shot_output["pre_shot_log_prob"] * advantage[:, 0].detach()).mean()
    live_log_prob_tensor = torch.stack(live_log_probs, dim=1)
    live_entropy_tensor = torch.stack(live_entropies, dim=1)
    live_policy_loss = -(live_log_prob_tensor * advantage.detach()).mean()
    entropy = 0.5 * (pre_shot_output["pre_shot_entropy"].mean() + live_entropy_tensor.mean())
    actor_loss = (
        -lambda_target_tensor.mean()
        + pre_shot_policy_loss
        + live_policy_loss
        + config.pre_shot_behavior_loss_scale * pre_shot_behavior_loss
        - config.actor_entropy_scale * entropy
    )
    dynamic_action_tensor = torch.stack(dynamic_actions, dim=1)
    return {
        "imagined_features": imagined_feature_tensor,
        "rewards": reward_tensor,
        "continuations": continuation_tensor,
        "values": value_tensor,
        "value_logits": torch.stack(value_logits, dim=1),
        "lambda_returns": lambda_target_tensor,
        "pre_shot_logits": pre_shot_output["pre_shot_logits"],
        "pre_shot_action_indexes": pre_shot_output["pre_shot_action_indexes"],
        "pre_shot_actions": pre_shot_output["pre_shot_actions"],
        "pre_shot_actions_held": pre_shot_output["pre_shot_actions"].unsqueeze(1).expand(
            -1, config.horizon, -1
        ),
        "pre_shot_action_mask": pre_shot_output["pre_shot_action_mask"],
        "pre_shot_log_prob": pre_shot_output["pre_shot_log_prob"],
        "pre_shot_entropy": pre_shot_output["pre_shot_entropy"],
        "pre_shot_policy_loss": pre_shot_policy_loss,
        "pre_shot_behavior_loss": pre_shot_behavior_loss,
        "live_action_logits": torch.stack(live_action_logits, dim=1),
        "live_action_indexes": torch.stack(live_action_indexes, dim=1),
        "live_action_choices": torch.stack(live_action_choices, dim=1),
        "live_log_prob": live_log_prob_tensor,
        "live_entropy": live_entropy_tensor,
        "live_policy_loss": live_policy_loss,
        "actor_entropy": entropy,
        "taste_objective": taste_objective,
        "dynamic_actions": dynamic_action_tensor,
        "control_action_mask": control_mask,
        "actor_loss_preview": actor_loss,
        "critic_loss_preview": critic_loss,
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
    _range_float(config.pre_shot_behavior_loss_scale, "pre_shot_behavior_loss_scale", 0.0, 100.0)


def _last_decision_indexes(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    decision_valid = batch["decision_step_mask"] * batch["step_mask"]
    step_indexes = torch.arange(decision_valid.shape[1], device=decision_valid.device, dtype=decision_valid.dtype)
    return (decision_valid * step_indexes.unsqueeze(0)).argmax(dim=1).to(dtype=torch.long)


def _decision_control_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    decision_valid = (batch["decision_step_mask"] * batch["step_mask"]).unsqueeze(-1)
    return (batch["control_action_mask"] * decision_valid).amax(dim=1)


def _context_state(
    context_encoder: DreamerContextEncoder | None,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    if context_encoder is None:
        return None
    return context_encoder(batch)


def _pre_shot_spec_from_batch(batch: dict[str, Any]) -> DreamerPreShotActionSpec:
    value = batch.get("pre_shot_action_spec")
    return DreamerPreShotActionSpec.from_dict(value) if isinstance(value, dict) else DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC


def _live_action_spec_from_batch(batch: dict[str, Any]) -> DreamerLiveActionSpec:
    value = batch.get("live_action_spec")
    return DreamerLiveActionSpec.from_dict(value) if isinstance(value, dict) else DEFAULT_DREAMER_LIVE_ACTION_SPEC


def _control_spec_from_batch(batch: dict[str, Any]) -> DreamerControlSpec:
    value = batch.get("control_spec")
    return DreamerControlSpec.from_dict(value) if isinstance(value, dict) else DEFAULT_DREAMER_CONTROL_SPEC


def _taste_objective_spec_from_batch(batch: dict[str, Any]) -> DreamerTasteObjectiveSpec:
    value = batch.get("taste_objective_spec")
    return (
        DreamerTasteObjectiveSpec.from_dict(value)
        if isinstance(value, dict)
        else DEFAULT_DREAMER_TASTE_OBJECTIVE_SPEC
    )


def _pre_shot_action_bin_tensor(spec: DreamerPreShotActionSpec) -> torch.Tensor:
    maximum = max(len(spec.bins[name]) for name in DREAMER_PRE_SHOT_ACTION_FIELDS)
    tensor = torch.zeros((len(DREAMER_PRE_SHOT_ACTION_FIELDS), maximum), dtype=torch.float32)
    for index, name in enumerate(DREAMER_PRE_SHOT_ACTION_FIELDS):
        values = torch.tensor(spec.bins[name], dtype=torch.float32)
        tensor[index, : values.shape[0]] = values
    return tensor


def _live_action_bin_tensor(spec: DreamerLiveActionSpec) -> torch.Tensor:
    maximum = max(len(spec.bins[name]) for name in DREAMER_LIVE_ACTION_FIELDS)
    tensor = torch.zeros((len(DREAMER_LIVE_ACTION_FIELDS), maximum), dtype=torch.float32)
    for index, name in enumerate(DREAMER_LIVE_ACTION_FIELDS):
        values = torch.tensor(spec.bins[name], dtype=torch.float32)
        tensor[index, : values.shape[0]] = values
    return tensor


def _dynamic_action_bound_tensors(control_spec: DreamerControlSpec) -> tuple[torch.Tensor, torch.Tensor]:
    limits = control_spec.safety_limits
    ranges = {
        "pump_target_mode": (1.0, 2.0),
        "pressure_target_bar": (limits.min_pressure_bar, limits.max_pressure_bar),
        "flow_target_ml_s": (limits.min_flow_ml_s, limits.max_flow_ml_s),
        "valve_position": (limits.min_valve_position, limits.max_valve_position),
        "temperature_target_c": (limits.min_temperature_c, limits.max_temperature_c),
        "yield_stop_target_g": (limits.min_yield_stop_target_g, limits.max_yield_stop_target_g),
        "stop": (0.0, 1.0),
    }
    low = torch.tensor([ranges[name][0] for name in DREAMER_DYNAMIC_ACTION_FEATURES], dtype=torch.float32)
    high = torch.tensor([ranges[name][1] for name in DREAMER_DYNAMIC_ACTION_FEATURES], dtype=torch.float32)
    return low, high


def _apply_live_action_delta(
    target_state: torch.Tensor,
    live_action_choices: torch.Tensor,
    dynamic_low: torch.Tensor,
    dynamic_high: torch.Tensor,
    pump_measurements: torch.Tensor,
) -> torch.Tensor:
    mode_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("pump_target_mode")
    pressure_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("pressure_target_bar")
    flow_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("flow_target_ml_s")
    selected_mode = live_action_choices[..., mode_index]
    previous_mode = target_state[..., mode_index]
    base_target = target_state.clone()
    base_target[..., pressure_index] = torch.where(
        (selected_mode == 1.0) & (previous_mode != 1.0),
        pump_measurements[..., 0],
        target_state[..., pressure_index],
    )
    base_target[..., flow_index] = torch.where(
        (selected_mode == 2.0) & (previous_mode != 2.0),
        pump_measurements[..., 1],
        target_state[..., flow_index],
    )
    delta_mask = torch.ones_like(live_action_choices)
    delta_mask[..., mode_index] = 0.0
    next_target = base_target + live_action_choices * delta_mask
    next_target[..., mode_index] = torch.where(
        live_action_choices[..., mode_index] > 0.0,
        live_action_choices[..., mode_index],
        target_state[..., mode_index],
    )
    stop_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("stop")
    next_target = torch.maximum(torch.minimum(next_target, dynamic_high), dynamic_low)
    requested_stop = (live_action_choices[..., stop_index] > 0.5).to(dtype=next_target.dtype)
    next_target[..., stop_index] = torch.maximum(target_state[..., stop_index], requested_stop)
    return next_target


def _initial_dynamic_target_state(
    batch: dict[str, torch.Tensor],
    *,
    batch_indexes: torch.Tensor,
    start_indexes: torch.Tensor,
    pre_shot_actions: torch.Tensor,
) -> torch.Tensor:
    target_state = torch.zeros(
        batch_indexes.shape[0],
        len(DREAMER_DYNAMIC_ACTION_FEATURES),
        dtype=pre_shot_actions.dtype,
        device=pre_shot_actions.device,
    )
    observed_targets = batch["observed_profile_targets"][batch_indexes, start_indexes].to(dtype=pre_shot_actions.dtype)
    observed_mask = batch["observed_profile_target_mask"][batch_indexes, start_indexes].to(dtype=pre_shot_actions.dtype)
    profile_to_dynamic = {
        "pressure_target_bar": "pressure_target_bar",
        "flow_target_ml_s": "flow_target_ml_s",
        "temperature_target_c": "temperature_target_c",
        "valve_open": "valve_position",
    }
    for observed_name, dynamic_name in profile_to_dynamic.items():
        observed_index = DREAMER_OBSERVED_TARGET_FEATURES.index(observed_name)
        dynamic_index = DREAMER_DYNAMIC_ACTION_FEATURES.index(dynamic_name)
        target_state[:, dynamic_index] = observed_targets[:, observed_index] * observed_mask[:, observed_index]
    observed_pump_mode_index = DREAMER_OBSERVED_TARGET_FEATURES.index("pump_target_mode")
    dynamic_pump_mode_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("pump_target_mode")
    target_state[:, dynamic_pump_mode_index] = observed_targets[:, observed_pump_mode_index]
    yield_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("yield_stop_target_g")
    pre_shot_yield_index = DREAMER_PRE_SHOT_ACTION_FIELDS.index("yield_target_g")
    target_state[:, yield_index] = pre_shot_actions[:, pre_shot_yield_index]
    logged_actions = batch["dynamic_actions"][batch_indexes, start_indexes].to(dtype=pre_shot_actions.dtype)
    logged_mask = batch["dynamic_action_mask"][batch_indexes, start_indexes].to(dtype=pre_shot_actions.dtype)
    return torch.where(logged_mask > 0.5, logged_actions, target_state)


def _condition_on_taste(
    features: torch.Tensor,
    taste_objective: torch.Tensor,
    taste_objective_dim: int,
) -> torch.Tensor:
    if taste_objective.ndim == 2 and features.ndim > 2:
        taste_objective = taste_objective.unsqueeze(1).expand(
            *features.shape[:-1],
            taste_objective_dim,
        )
    if taste_objective.shape != (*features.shape[:-1], taste_objective_dim):
        raise ValueError("Dreamer critic taste-objective shape is incompatible with features")
    return torch.cat([features, taste_objective.to(dtype=features.dtype)], dim=-1)


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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DREAMER_V3_REFERENCE_WORLD_MODEL_FORMAT = "espresso_rl_dreamer_v3_reference_world_model_v1"
DREAMER_V3_REFERENCE_WORLD_MODEL_SCHEMA_VERSION = 1
DREAMER_V3_MODEL_PRESETS = frozenset({"espresso_debug", "espresso_small", "espresso_medium"})


@dataclass(frozen=True)
class DreamerV3WorldModelConfig:
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

    @property
    def stoch_dim(self) -> int:
        return self.stoch_size * self.class_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": DREAMER_V3_REFERENCE_WORLD_MODEL_FORMAT,
            "schema_version": DREAMER_V3_REFERENCE_WORLD_MODEL_SCHEMA_VERSION,
            "model_preset": self.model_preset,
            "deter_dim": self.deter_dim,
            "hidden_dim": self.hidden_dim,
            "stoch_size": self.stoch_size,
            "class_size": self.class_size,
            "stoch_dim": self.stoch_dim,
            "action_embed_dim": self.action_embed_dim,
            "reward_bins": self.reward_bins,
            "unimix": self.unimix,
            "free_nats": self.free_nats,
            "dyn_loss_scale": self.dyn_loss_scale,
            "rep_loss_scale": self.rep_loss_scale,
            "observation_loss_scale": self.observation_loss_scale,
            "reward_loss_scale": self.reward_loss_scale,
            "continuation_loss_scale": self.continuation_loss_scale,
        }


def default_world_model_config(model_preset: str = "espresso_debug") -> DreamerV3WorldModelConfig:
    if model_preset == "espresso_debug":
        return DreamerV3WorldModelConfig(
            model_preset=model_preset,
            deter_dim=32,
            hidden_dim=32,
            stoch_size=4,
            class_size=4,
            action_embed_dim=16,
            reward_bins=41,
            unimix=0.01,
            free_nats=1.0,
            dyn_loss_scale=1.0,
            rep_loss_scale=0.1,
            observation_loss_scale=1.0,
            reward_loss_scale=1.0,
            continuation_loss_scale=1.0,
        )
    if model_preset == "espresso_small":
        return DreamerV3WorldModelConfig(
            model_preset=model_preset,
            deter_dim=128,
            hidden_dim=128,
            stoch_size=8,
            class_size=8,
            action_embed_dim=32,
            reward_bins=41,
            unimix=0.01,
            free_nats=1.0,
            dyn_loss_scale=1.0,
            rep_loss_scale=0.1,
            observation_loss_scale=1.0,
            reward_loss_scale=1.0,
            continuation_loss_scale=1.0,
        )
    if model_preset == "espresso_medium":
        return DreamerV3WorldModelConfig(
            model_preset=model_preset,
            deter_dim=256,
            hidden_dim=256,
            stoch_size=16,
            class_size=16,
            action_embed_dim=64,
            reward_bins=101,
            unimix=0.01,
            free_nats=1.0,
            dyn_loss_scale=1.0,
            rep_loss_scale=0.1,
            observation_loss_scale=1.0,
            reward_loss_scale=1.0,
            continuation_loss_scale=1.0,
        )
    raise ValueError(f"unsupported DreamerV3 world model preset: {model_preset}")


def symlog(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(torch.abs(value))


def symexp(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.expm1(torch.abs(value))


def symexp_twohot_bins(bin_count: int, *, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count < 3:
        raise ValueError("bin_count must be an integer >= 3")
    if bin_count % 2 == 1:
        half = torch.linspace(-20.0, 0.0, (bin_count - 1) // 2 + 1, dtype=torch.float32, device=device)
        half = symexp(half)
        return torch.cat([half, -half[:-1].flip(0)], dim=0)
    half = torch.linspace(-20.0, 0.0, bin_count // 2, dtype=torch.float32, device=device)
    half = symexp(half)
    return torch.cat([half, -half.flip(0)], dim=0)


def twohot_targets(target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    target = target.to(dtype=torch.float32)
    bins = bins.to(device=target.device, dtype=torch.float32)
    below = (bins <= target.unsqueeze(-1)).to(torch.int64).sum(dim=-1) - 1
    above = bins.shape[-1] - (bins > target.unsqueeze(-1)).to(torch.int64).sum(dim=-1)
    below = below.clamp(0, bins.shape[-1] - 1)
    above = above.clamp(0, bins.shape[-1] - 1)
    equal = below == above
    dist_to_below = torch.where(equal, torch.ones_like(target), torch.abs(bins[below] - target))
    dist_to_above = torch.where(equal, torch.ones_like(target), torch.abs(bins[above] - target))
    total = (dist_to_below + dist_to_above).clamp_min(1e-8)
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    return (
        F.one_hot(below, bins.shape[-1]).to(dtype=torch.float32) * weight_below.unsqueeze(-1)
        + F.one_hot(above, bins.shape[-1]).to(dtype=torch.float32) * weight_above.unsqueeze(-1)
    )


def twohot_cross_entropy(logits: torch.Tensor, target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    target_dist = twohot_targets(target, bins)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1)


def twohot_prediction(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    bins = bins.to(device=logits.device, dtype=torch.float32)
    return (probs * bins).sum(dim=-1)


def unimix_logits(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    if unimix <= 0.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    uniform = torch.full_like(probs, 1.0 / logits.shape[-1])
    mixed = (1.0 - unimix) * probs + unimix * uniform
    return torch.log(mixed.clamp_min(1e-8))


def straight_through_onehot(
    logits: torch.Tensor,
    *,
    unimix: float = 0.0,
    sample: bool = True,
) -> torch.Tensor:
    mixed_logits = unimix_logits(logits, unimix)
    probs = F.softmax(mixed_logits, dim=-1)
    if sample:
        flat = probs.detach().reshape(-1, probs.shape[-1])
        indexes = torch.multinomial(flat, num_samples=1).reshape(*probs.shape[:-1])
    else:
        indexes = torch.argmax(probs, dim=-1)
    hard = F.one_hot(indexes, probs.shape[-1]).to(dtype=probs.dtype)
    return hard.detach() + probs - probs.detach()


def categorical_kl_logits(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    *,
    unimix: float = 0.0,
) -> torch.Tensor:
    first_logits = unimix_logits(first_logits, unimix)
    second_logits = unimix_logits(second_logits, unimix)
    first_log_probs = F.log_softmax(first_logits, dim=-1)
    second_log_probs = F.log_softmax(second_logits, dim=-1)
    first_probs = torch.exp(first_log_probs)
    return (first_probs * (first_log_probs - second_log_probs)).sum(dim=(-1, -2))


class DreamerV3VectorWorldModel(nn.Module):
    def __init__(
        self,
        *,
        observation_dim: int,
        behavior_dim: int,
        static_dim: int,
        config: DreamerV3WorldModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.observation_encoder = _mlp(observation_dim + static_dim, config.hidden_dim, config.hidden_dim)
        self.behavior_encoder = nn.Sequential(
            nn.Linear(behavior_dim, config.action_embed_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(config.stoch_dim + config.action_embed_dim, config.deter_dim)
        self.prior = _mlp(config.deter_dim, config.hidden_dim, config.stoch_size * config.class_size)
        self.posterior = _mlp(
            config.deter_dim + config.hidden_dim,
            config.hidden_dim,
            config.stoch_size * config.class_size,
        )
        feature_dim = config.deter_dim + config.stoch_dim
        self.observation_decoder = _mlp(feature_dim, config.hidden_dim, observation_dim)
        self.reward_decoder = _mlp(feature_dim, config.hidden_dim, config.reward_bins)
        self.continuation_decoder = _mlp(feature_dim, config.hidden_dim, 1)
        self.register_buffer("reward_bins", symexp_twohot_bins(config.reward_bins), persistent=False)

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        deter = torch.zeros(batch_size, self.config.deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.config.stoch_size, self.config.class_size, device=device)
        return deter, stoch

    def observe(
        self,
        batch: dict[str, torch.Tensor],
        *,
        is_first: torch.Tensor | None = None,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        observations = batch["observations"]
        behavior = _behavior_tensor(batch)
        static_context = batch["static_context"]
        step_mask = batch["step_mask"]
        batch_size, step_count, _ = observations.shape
        deter, stoch = self.initial_state(batch_size, observations.device)
        if is_first is None:
            is_first = torch.zeros_like(step_mask, dtype=torch.bool)
            is_first[:, 0] = True
        else:
            is_first = is_first.to(device=observations.device, dtype=torch.bool)

        deter_states: list[torch.Tensor] = []
        stoch_states: list[torch.Tensor] = []
        prior_logits: list[torch.Tensor] = []
        posterior_logits: list[torch.Tensor] = []

        for step_index in range(step_count):
            reset_mask = (~is_first[:, step_index]).to(dtype=observations.dtype).unsqueeze(-1)
            valid_mask = step_mask[:, step_index].to(dtype=observations.dtype).unsqueeze(-1)
            deter = deter * reset_mask
            stoch = stoch * reset_mask.unsqueeze(-1)
            action = _squash_action(behavior[:, step_index])
            action = action * valid_mask
            action_embed = self.behavior_encoder(action)
            deter = self.gru(torch.cat([stoch.flatten(start_dim=-2), action_embed], dim=-1), deter)
            prior = self.prior(deter).reshape(batch_size, self.config.stoch_size, self.config.class_size)
            obs_embed = self.observation_encoder(
                torch.cat([symlog(observations[:, step_index]), symlog(static_context)], dim=-1)
            )
            post = self.posterior(torch.cat([deter, obs_embed], dim=-1)).reshape(
                batch_size,
                self.config.stoch_size,
                self.config.class_size,
            )
            stoch = straight_through_onehot(post, unimix=self.config.unimix, sample=sample)
            deter = deter * valid_mask
            stoch = stoch * valid_mask.unsqueeze(-1)
            deter_states.append(deter)
            stoch_states.append(stoch)
            prior_logits.append(prior)
            posterior_logits.append(post)

        deter_tensor = torch.stack(deter_states, dim=1)
        stoch_tensor = torch.stack(stoch_states, dim=1)
        features = torch.cat([deter_tensor, stoch_tensor.flatten(start_dim=-2)], dim=-1)
        return {
            "deter": deter_tensor,
            "stoch": stoch_tensor,
            "features": features,
            "prior_logits": torch.stack(prior_logits, dim=1),
            "posterior_logits": torch.stack(posterior_logits, dim=1),
        }

    def losses(self, batch: dict[str, torch.Tensor], *, sample: bool = True) -> dict[str, torch.Tensor]:
        observed = self.observe(batch, sample=sample)
        features = observed["features"]
        step_mask = batch["step_mask"]
        valid_count = step_mask.sum().clamp_min(1.0)
        observation_prediction = self.observation_decoder(features)
        observation_loss = (
            ((observation_prediction - symlog(batch["observations"])) ** 2) * step_mask.unsqueeze(-1)
        ).sum() / (valid_count * batch["observations"].shape[-1])
        reward_logits = self.reward_decoder(features)
        reward_loss = (
            twohot_cross_entropy(reward_logits, batch["rewards"], self.reward_bins) * step_mask
        ).sum() / valid_count
        continuation_logits = self.continuation_decoder(features).squeeze(-1)
        continuation_loss = (
            F.binary_cross_entropy_with_logits(continuation_logits, batch["continuations"], reduction="none")
            * step_mask
        ).sum() / valid_count
        dyn = categorical_kl_logits(
            observed["posterior_logits"].detach(),
            observed["prior_logits"],
            unimix=self.config.unimix,
        ).clamp_min(self.config.free_nats)
        rep = categorical_kl_logits(
            observed["posterior_logits"],
            observed["prior_logits"].detach(),
            unimix=self.config.unimix,
        ).clamp_min(self.config.free_nats)
        dyn_loss = (dyn * step_mask).sum() / valid_count
        rep_loss = (rep * step_mask).sum() / valid_count
        total = (
            self.config.observation_loss_scale * observation_loss
            + self.config.reward_loss_scale * reward_loss
            + self.config.continuation_loss_scale * continuation_loss
            + self.config.dyn_loss_scale * dyn_loss
            + self.config.rep_loss_scale * rep_loss
        )
        return {
            "loss_total": total,
            "loss_observation": observation_loss,
            "loss_reward": reward_loss,
            "loss_continuation": continuation_loss,
            "loss_dyn": dyn_loss,
            "loss_rep": rep_loss,
        }


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _behavior_tensor(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            batch["observed_profile_targets"],
            batch["observed_profile_target_mask"],
            batch["dynamic_actions"],
            batch["dynamic_action_mask"],
            batch["control_action_mask"],
            batch["constraints"],
            batch["decision_step_mask"].unsqueeze(-1),
        ],
        dim=-1,
    )


def _squash_action(value: torch.Tensor) -> torch.Tensor:
    return value / torch.maximum(torch.ones_like(value), torch.abs(value))

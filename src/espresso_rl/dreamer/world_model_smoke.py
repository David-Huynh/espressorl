from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

WORLD_MODEL_SMOKE_FORMAT = "espresso_rl_world_model_smoke_v1"
WORLD_MODEL_SMOKE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WorldModelSmokeResult:
    seed: int
    train_steps: int
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
            "initial": self.initial,
            "final": self.final,
            "loss_delta_total": round(self.initial["loss_total"] - self.final["loss_total"], 8),
        }


class FixedCadenceWorldModelSmokeError(ValueError):
    pass


class _FixedCadenceSmokeWorldModel(nn.Module):
    def __init__(
        self,
        *,
        observation_dim: int,
        behavior_dim: int,
        static_dim: int,
        hidden_dim: int = 32,
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        self.observation_encoder = nn.Sequential(
            nn.Linear(observation_dim + static_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.behavior_encoder = nn.Sequential(nn.Linear(behavior_dim, 16), nn.ELU())
        self.gru = nn.GRUCell(latent_dim + 16, hidden_dim)
        self.prior = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, latent_dim))
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.observation_decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, observation_dim),
        )
        self.reward_decoder = nn.Sequential(nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))
        self.continuation_decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        observations = batch["observations"]
        behavior = _behavior_tensor(batch)
        static_context = batch["static_context"]
        batch_size, step_count, _ = observations.shape
        h = observations.new_zeros(batch_size, self.gru.hidden_size)
        z = observations.new_zeros(batch_size, self.prior[-1].out_features)
        observation_predictions: list[torch.Tensor] = []
        reward_predictions: list[torch.Tensor] = []
        continuation_logits: list[torch.Tensor] = []
        prior_states: list[torch.Tensor] = []
        posterior_states: list[torch.Tensor] = []

        for step_index in range(step_count):
            behavior_embed = self.behavior_encoder(behavior[:, step_index])
            h = self.gru(torch.cat([z, behavior_embed], dim=-1), h)
            prior_state = self.prior(h)
            observation_embed = self.observation_encoder(
                torch.cat([observations[:, step_index], static_context], dim=-1)
            )
            posterior_state = self.posterior(torch.cat([h, observation_embed], dim=-1))
            z = torch.tanh(posterior_state)
            state = torch.cat([h, z], dim=-1)
            observation_predictions.append(self.observation_decoder(state))
            reward_predictions.append(self.reward_decoder(state).squeeze(-1))
            continuation_logits.append(self.continuation_decoder(state).squeeze(-1))
            prior_states.append(prior_state)
            posterior_states.append(posterior_state)

        return {
            "observations": torch.stack(observation_predictions, dim=1),
            "rewards": torch.stack(reward_predictions, dim=1),
            "continuation_logits": torch.stack(continuation_logits, dim=1),
            "prior_states": torch.stack(prior_states, dim=1),
            "posterior_states": torch.stack(posterior_states, dim=1),
        }


def run_fixed_cadence_world_model_smoke_train(
    batch: dict[str, Any],
    *,
    seed: int,
    train_steps: int,
) -> WorldModelSmokeResult:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise FixedCadenceWorldModelSmokeError("world model smoke seed must be a uint32 integer")
    if isinstance(train_steps, bool) or not isinstance(train_steps, int) or not 1 <= train_steps <= 20:
        raise FixedCadenceWorldModelSmokeError("world model smoke train_steps must be 1..20")

    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        tensors = _cpu_float_batch(batch)
        _validate_batch_shapes(tensors)
        model = _FixedCadenceSmokeWorldModel(
            observation_dim=tensors["observations"].shape[-1],
            behavior_dim=_behavior_tensor(tensors).shape[-1],
            static_dim=tensors["static_context"].shape[-1],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        initial = _evaluate(model, tensors)
        for _ in range(train_steps):
            optimizer.zero_grad()
            losses = _losses(model, tensors)
            losses["loss_total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        final = _evaluate(model, tensors)
    finally:
        torch.use_deterministic_algorithms(old_deterministic)
        torch.set_num_threads(old_threads)
    return WorldModelSmokeResult(seed=seed, train_steps=train_steps, initial=initial, final=final)


def _cpu_float_batch(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    required = (
        "observations",
        "observed_profile_targets",
        "observed_profile_target_mask",
        "dynamic_actions",
        "dynamic_action_mask",
        "control_action_mask",
        "constraints",
        "decision_step_mask",
        "rewards",
        "continuations",
        "step_mask",
        "static_context",
    )
    converted: dict[str, torch.Tensor] = {}
    for key in required:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise FixedCadenceWorldModelSmokeError(f"world model smoke batch is missing tensor {key}")
        converted[key] = value.detach().to(device="cpu", dtype=torch.float32)
    return converted


def _validate_batch_shapes(batch: dict[str, torch.Tensor]) -> None:
    observations = batch["observations"]
    if observations.ndim != 3:
        raise FixedCadenceWorldModelSmokeError("world model smoke observations must have shape (batch, steps, features)")
    batch_size, step_count, _ = observations.shape
    if batch_size <= 0 or step_count <= 1:
        raise FixedCadenceWorldModelSmokeError("world model smoke batch must contain at least two valid steps")
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
            raise FixedCadenceWorldModelSmokeError(f"world model smoke {key} shape is invalid")
    for key in ("decision_step_mask", "rewards", "continuations", "step_mask"):
        tensor = batch[key]
        if tensor.ndim != 2 or tensor.shape != (batch_size, step_count):
            raise FixedCadenceWorldModelSmokeError(f"world model smoke {key} shape is invalid")
    static_context = batch["static_context"]
    if static_context.ndim != 2 or static_context.shape[0] != batch_size:
        raise FixedCadenceWorldModelSmokeError("world model smoke static_context shape is invalid")
    if float(batch["step_mask"].sum().item()) < 2.0:
        raise FixedCadenceWorldModelSmokeError("world model smoke batch must contain at least two valid steps")


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


def _losses(model: _FixedCadenceSmokeWorldModel, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = model(batch)
    step_mask = batch["step_mask"]
    observation_weight = step_mask.unsqueeze(-1)
    valid_count = step_mask.sum().clamp_min(1.0)
    observation_loss = (
        ((output["observations"] - batch["observations"]) ** 2) * observation_weight
    ).sum() / (valid_count * batch["observations"].shape[-1])
    reward_loss = (((output["rewards"] - batch["rewards"]) ** 2) * step_mask).sum() / valid_count
    continuation_loss = (
        F.binary_cross_entropy_with_logits(output["continuation_logits"], batch["continuations"], reduction="none")
        * step_mask
    ).sum() / valid_count
    kl_loss = (((output["posterior_states"] - output["prior_states"]) ** 2).mean(dim=-1) * step_mask).sum() / valid_count
    total = observation_loss + reward_loss + continuation_loss + 0.1 * kl_loss
    return {
        "loss_total": total,
        "loss_observation": observation_loss,
        "loss_reward": reward_loss,
        "loss_continuation": continuation_loss,
        "loss_kl": kl_loss,
    }


@torch.no_grad()
def _evaluate(model: _FixedCadenceSmokeWorldModel, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    model.eval()
    losses = _losses(model, batch)
    model.train()
    return {key: round(float(value.item()), 8) for key, value in losses.items()}

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from espresso_rl.dreamer.reference_world_model import symlog


DREAMER_CONTEXT_ENCODER_FORMAT = "espresso_rl_dreamer_v3_context_encoder_v1"
DREAMER_CONTEXT_ENCODER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DreamerContextEncoderConfig:
    hidden_dim: int
    context_dim: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": DREAMER_CONTEXT_ENCODER_FORMAT,
            "schema_version": DREAMER_CONTEXT_ENCODER_SCHEMA_VERSION,
            "hidden_dim": self.hidden_dim,
            "context_dim": self.context_dim,
        }


class DreamerContextEncoder(nn.Module):
    def __init__(
        self,
        *,
        static_dim: int,
        terminal_dim: int,
        time_dim: int,
        trajectory_dim: int,
        config: DreamerContextEncoderConfig,
    ) -> None:
        super().__init__()
        self.static_dim = static_dim
        self.terminal_dim = terminal_dim
        self.time_dim = time_dim
        self.trajectory_dim = trajectory_dim
        self.config = config
        input_dim = static_dim + terminal_dim + time_dim + trajectory_dim
        self.row_encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.recurrent = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(config.hidden_dim, config.context_dim),
            nn.SiLU(),
        )

    @property
    def input_dim(self) -> int:
        return self.static_dim + self.terminal_dim + self.time_dim + self.trajectory_dim

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        context_static = _required_tensor(batch, "context_static")
        context_terminal = _required_tensor(batch, "context_terminal")
        context_time = _required_tensor(batch, "context_time")
        context_trajectory = _required_tensor(batch, "context_trajectory_embedding")
        context_mask = _required_tensor(batch, "context_mask")
        if context_static.ndim != 3:
            raise ValueError("context_static must have shape (batch, context, features)")
        batch_size, context_window_size, _ = context_static.shape
        expected_shapes = {
            "context_terminal": (batch_size, context_window_size, self.terminal_dim),
            "context_time": (batch_size, context_window_size, self.time_dim),
            "context_trajectory_embedding": (batch_size, context_window_size, self.trajectory_dim),
            "context_mask": (batch_size, context_window_size),
        }
        actual_shapes = {
            "context_terminal": tuple(context_terminal.shape),
            "context_time": tuple(context_time.shape),
            "context_trajectory_embedding": tuple(context_trajectory.shape),
            "context_mask": tuple(context_mask.shape),
        }
        if tuple(context_static.shape) != (batch_size, context_window_size, self.static_dim):
            raise ValueError("context_static feature dimension is incompatible with context encoder")
        for key, expected in expected_shapes.items():
            if actual_shapes[key] != expected:
                raise ValueError(f"{key} shape is incompatible with context encoder")

        mask = (context_mask > 0.5).to(dtype=context_static.dtype)
        rows = torch.cat(
            [
                symlog(context_static),
                symlog(context_terminal),
                symlog(context_time),
                symlog(context_trajectory),
            ],
            dim=-1,
        )
        rows = rows * mask.unsqueeze(-1)
        encoded_rows = self.row_encoder(rows.reshape(batch_size * context_window_size, self.input_dim))
        encoded_rows = encoded_rows.reshape(batch_size, context_window_size, self.config.hidden_dim)
        recurrent_output, _ = self.recurrent(encoded_rows)

        lengths = mask.sum(dim=1).to(dtype=torch.long)
        gather_indexes = (lengths - 1).clamp_min(0)
        batch_indexes = torch.arange(batch_size, device=context_static.device)
        summary = recurrent_output[batch_indexes, gather_indexes]
        context = self.output(summary)
        nonempty = (lengths > 0).to(dtype=context.dtype).unsqueeze(-1)
        return context * nonempty


def _required_tensor(batch: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"context encoder batch is missing tensor {key}")
    return value

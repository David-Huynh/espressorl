"""
Symlog critic with two-hot discrete regression (DreamerV3 §3).

The critic predicts the expected lambda-return in symlog space using a
categorical distribution over N_BINS equally-spaced bins. This handles
mixed reward scales without manual normalisation tuning.

Value estimate = symexp( sum( softmax(logits) * bins ) )
Training loss  = cross-entropy( logits, two_hot(target, bins) )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rssm import STATE_DIM
from .utils import symexp, two_hot_loss

N_BINS   = 41
BIN_MIN  = -5.0   # symlog space — covers rewards compressed by symlog
BIN_MAX  =  5.0


class SymlogCritic(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 256, n_bins: int = N_BINS) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden),    nn.ELU(),
            nn.Linear(hidden, n_bins),
        )
        self.register_buffer("bins", torch.linspace(BIN_MIN, BIN_MAX, n_bins))

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (value, logits):
            value  : (*)  — expected return in original space (via symexp)
            logits : (*, N_BINS)
        """
        logits = self.net(torch.cat([h, z], dim=-1))  # (*, N_BINS)
        probs  = F.softmax(logits, dim=-1)
        value  = symexp((probs * self.bins).sum(-1))   # (*)
        return value, logits

    def loss(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        target_returns: torch.Tensor,  # (*) — lambda returns in original space
    ) -> torch.Tensor:
        """Two-hot cross-entropy loss against lambda returns."""
        logits = self.net(torch.cat([h, z], dim=-1))
        return two_hot_loss(logits, target_returns, self.bins)

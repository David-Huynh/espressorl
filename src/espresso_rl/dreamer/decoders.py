"""
Decoder heads for the DreamerV3 world model.

All decoders receive the full latent state s = concat(h, z) — shape (*, STATE_DIM).

ProfileDecoder   : s → (5, 100) reconstructed shot profile
RewardDecoder    : s → scalar reward (two-hot bins in symlog space)
ContinueDecoder  : s → binary continuation probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rssm import STATE_DIM  # 512

# Reward bins spanning ±~20 in symlog space (covers rewards 0–1 with margin)
N_REWARD_BINS = 41
REWARD_BIN_MIN = -5.0
REWARD_BIN_MAX = 5.0

PROFILE_CHANNELS = 5
PROFILE_POINTS   = 100
PROFILE_DIM      = PROFILE_CHANNELS * PROFILE_POINTS  # 500


def _mlp(in_dim: int, *hidden_dims: int, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ELU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ProfileDecoder(nn.Module):
    """Reconstruct the shot profile from the latent state."""

    def __init__(self, state_dim: int = STATE_DIM) -> None:
        super().__init__()
        self.net = _mlp(state_dim, 512, 512, out_dim=PROFILE_DIM)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        h, z : (*, h_dim), (*, z_dim)
        returns: (*, 5, 100)
        """
        s = torch.cat([h, z], dim=-1)
        out = self.net(s)  # (*, 500)
        return out.unflatten(-1, (PROFILE_CHANNELS, PROFILE_POINTS))

    def loss(self, h: torch.Tensor, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction loss."""
        pred = self.forward(h, z)
        return F.mse_loss(pred, target)


class RewardDecoder(nn.Module):
    """
    Predict scalar reward via two-hot discrete regression in symlog space.
    Returns logits over N_REWARD_BINS bins; expected value is computed by the caller.
    """

    def __init__(self, state_dim: int = STATE_DIM, n_bins: int = N_REWARD_BINS) -> None:
        super().__init__()
        self.net = _mlp(state_dim, 256, 256, out_dim=n_bins)
        self.register_buffer(
            "bins",
            torch.linspace(REWARD_BIN_MIN, REWARD_BIN_MAX, n_bins),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Returns logits (*, N_BINS)."""
        s = torch.cat([h, z], dim=-1)
        return self.net(s)

    def expected_value(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute E[reward] = symexp(sum(softmax(logits) * bins)).
        """
        from .utils import symexp
        probs = F.softmax(logits, dim=-1)
        return symexp((probs * self.bins).sum(-1))

    def loss(self, h: torch.Tensor, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        from .utils import two_hot_loss
        logits = self.forward(h, z)
        return two_hot_loss(logits, target, self.bins)


class ContinueDecoder(nn.Module):
    """
    Binary classifier: does the 'episode' continue after this state?
    In espresso context this is almost always 1, but included for DreamerV3 completeness.
    """

    def __init__(self, state_dim: int = STATE_DIM) -> None:
        super().__init__()
        self.net = _mlp(state_dim, 256, out_dim=1)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Returns log-odds (*, 1); sigmoid gives P(continue)."""
        s = torch.cat([h, z], dim=-1)
        return self.net(s)

    def loss(self, h: torch.Tensor, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        target: (*, 1) float — 1.0 if episode continues, 0.0 at terminal.
        """
        logits = self.forward(h, z)
        return F.binary_cross_entropy_with_logits(logits, target)

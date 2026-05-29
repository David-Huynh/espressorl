"""
Factored categorical actor for DreamerV3.

Two independent discrete heads:
    grind_head : 11 logits → delta in {-5,-4,-3,-2,-1,0,+1,+2,+3,+4,+5} grinder steps
                             ≈ ±50μm at 10μm/step, ±62.5μm at 12.5μm/step
    dose_head  : 11 logits → absolute dose in {15.0, 15.5, …, 20.0} g (0.5g steps)

Factored (11+11=22 logits) vs joint (121) — each head has an independent entropy
bonus so the agent can learn "go finer" without confounding dose, and vice-versa.

Training is done exclusively on imagined rollouts from the RSSM — never on real
transitions directly.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .rssm import STATE_DIM

GRIND_BINS = 11                                        # -5 … +5 steps
DOSE_BINS  = 11                                        # 15.0 … 20.0 g

# Look-up tables: action index → physical value
GRIND_DELTA_STEPS: list[int]  = list(range(-5, 6))    # [-5, -4, ..., +5]
DOSE_TARGETS_G:   list[float] = [15.0 + i * 0.5 for i in range(11)]


class FactoredCategoricalActor(nn.Module):
    """
    Actor outputting two independent categorical distributions over
    grind delta (steps) and dose target (grams).
    """

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden),    nn.ELU(),
        )
        self.grind_head = nn.Linear(hidden, GRIND_BINS)
        self.dose_head  = nn.Linear(hidden, DOSE_BINS)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (grind_logits, dose_logits) — (*, GRIND_BINS), (*, DOSE_BINS)."""
        features = self.shared(torch.cat([h, z], dim=-1))
        return self.grind_head(features), self.dose_head(features)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self, h: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample discrete action indices with log-probabilities.

        Returns:
            grind_idx     : (B,) int64  — index into GRIND_DELTA_STEPS
            dose_idx      : (B,) int64  — index into DOSE_TARGETS_G
            grind_log_prob: (B,)
            dose_log_prob : (B,)
        """
        g_logits, d_logits = self.forward(h, z)
        g_dist, d_dist = Categorical(logits=g_logits), Categorical(logits=d_logits)
        g_idx, d_idx   = g_dist.sample(), d_dist.sample()
        return g_idx, d_idx, g_dist.log_prob(g_idx), d_dist.log_prob(d_idx)

    def entropy(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Sum of entropy of both heads. (B,)"""
        g_logits, d_logits = self.forward(h, z)
        return Categorical(logits=g_logits).entropy() + Categorical(logits=d_logits).entropy()

    def log_prob(
        self,
        h: torch.Tensor, z: torch.Tensor,
        grind_idx: torch.Tensor, dose_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Log probability of a given (grind_idx, dose_idx) pair. (B,)"""
        g_logits, d_logits = self.forward(h, z)
        return (
            Categorical(logits=g_logits).log_prob(grind_idx)
            + Categorical(logits=d_logits).log_prob(dose_idx)
        )

    # ------------------------------------------------------------------
    # Action decoding (index → physical quantity)
    # ------------------------------------------------------------------

    @staticmethod
    def decode_grind(grind_idx: torch.Tensor, step_size_um: float) -> torch.Tensor:
        """grind action index → delta μm. (B,) → (B,)"""
        lut = torch.tensor(GRIND_DELTA_STEPS, dtype=torch.float32, device=grind_idx.device)
        return lut[grind_idx] * step_size_um

    @staticmethod
    def decode_dose(dose_idx: torch.Tensor) -> torch.Tensor:
        """dose action index → grams. (B,) → (B,)"""
        lut = torch.tensor(DOSE_TARGETS_G, dtype=torch.float32, device=dose_idx.device)
        return lut[dose_idx]

    # ------------------------------------------------------------------
    # Action encoding (physical quantity → index)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_grind(delta_steps: int) -> int:
        """Clamp delta_steps to valid range and return index."""
        clamped = max(-5, min(5, round(delta_steps)))
        return clamped + 5  # shift: -5→0, 0→5, +5→10

    @staticmethod
    def encode_dose(dose_g: float) -> int:
        return max(0, min(10, round((dose_g - 15.0) / 0.5)))

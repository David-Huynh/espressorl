"""
Recurrent State Space Model (RSSM) — the world model backbone.

Architecture (DreamerV3, espresso-adapted):
    Deterministic path : h_t = GRU(h_{t-1}, concat(z_{t-1}, a_emb_{t-1}))
    Stochastic path    : z_t ~ Categorical(posterior(h_t, e_t))   [training]
                         z_t ~ Categorical(prior(h_t))             [imagination]

State representation:
    h_t  : (B, H)         deterministic hidden — H = 256
    z_t  : (B, K*M)       flattened one-hot stochastic — K=16 categories × M=16 classes
    s_t  = concat(h_t, z_t)  → 512-dim latent used by decoders/actor/critic

Action encoding:
    action = concat(one_hot(grind_idx, GRIND_BINS), one_hot(dose_idx, DOSE_BINS))  → 18-dim
"""

import torch
import torch.nn as nn

from .utils import straight_through_categorical

H_DIM    = 256   # GRU hidden size
Z_CATS   = 16    # number of categorical distributions
Z_CLS    = 16    # classes per distribution
Z_DIM    = Z_CATS * Z_CLS   # = 256  (flattened stochastic state)
STATE_DIM = H_DIM + Z_DIM   # = 512  (full latent state for decoders/policy)

GRIND_BINS = 11  # grind delta discrete actions: -5 … +5 steps
DOSE_BINS  = 11  # dose discrete actions: 15.0 … 20.0 g
ACTION_DIM = GRIND_BINS + DOSE_BINS  # = 22


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ELU(),
        nn.Linear(hidden, out_dim),
    )


class RSSM(nn.Module):
    def __init__(
        self,
        h_dim:     int = H_DIM,
        z_cats:    int = Z_CATS,
        z_cls:     int = Z_CLS,
        embed_dim: int = 256,           # observation embedding from encoder
        action_dim: int = ACTION_DIM,
        hidden:    int = 256,
    ) -> None:
        super().__init__()
        self.h_dim  = h_dim
        self.z_cats = z_cats
        self.z_cls  = z_cls
        self.z_dim  = z_cats * z_cls

        # Action embedding before feeding into GRU
        self.action_embed = nn.Sequential(nn.Linear(action_dim, 32), nn.ELU())

        # GRU: input = concat(z_{t-1}, a_emb_{t-1})
        self.gru = nn.GRUCell(self.z_dim + 32, h_dim)

        # Prior: h_t → z_t logits (no observation)
        self.prior_net = _mlp(h_dim, hidden, z_cats * z_cls)

        # Posterior: (h_t, e_t) → z_t logits (with observation embedding)
        self.post_net = _mlp(h_dim + embed_dim, hidden, z_cats * z_cls)

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.h_dim, device=device)
        z = torch.zeros(batch_size, self.z_dim, device=device)
        return h, z

    # ------------------------------------------------------------------
    # Training step (with observation — uses posterior)
    # ------------------------------------------------------------------

    def observe_step(
        self,
        h: torch.Tensor,           # (B, H)
        z: torch.Tensor,           # (B, Z_DIM)
        action: torch.Tensor,      # (B, ACTION_DIM) one-hot
        embed: torch.Tensor,       # (B, embed_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (h_new, z_new, post_logits, prior_logits).
        z_new sampled from posterior via straight-through estimator.
        """
        a_emb   = self.action_embed(action)                                  # (B, 32)
        h_new   = self.gru(torch.cat([z, a_emb], dim=-1), h)                # (B, H)

        prior_logits = self.prior_net(h_new).view(-1, self.z_cats, self.z_cls)
        post_logits  = self.post_net(torch.cat([h_new, embed], dim=-1)).view(-1, self.z_cats, self.z_cls)

        z_new = straight_through_categorical(post_logits)                    # (B, Z_DIM)
        return h_new, z_new, post_logits, prior_logits

    # ------------------------------------------------------------------
    # Imagination step (without observation — uses prior)
    # ------------------------------------------------------------------

    def imagine_step(
        self,
        h: torch.Tensor,      # (B, H)
        z: torch.Tensor,      # (B, Z_DIM)
        action: torch.Tensor, # (B, ACTION_DIM) one-hot
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (h_new, z_new) sampled from prior — no real observation needed."""
        a_emb  = self.action_embed(action)
        h_new  = self.gru(torch.cat([z, a_emb], dim=-1), h)
        prior_logits = self.prior_net(h_new).view(-1, self.z_cats, self.z_cls)
        z_new  = straight_through_categorical(prior_logits)
        return h_new, z_new

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def encode_action(grind_idx: torch.Tensor, dose_idx: torch.Tensor) -> torch.Tensor:
        """
        grind_idx : (B,) int64, values 0–6
        dose_idx  : (B,) int64, values 0–10
        returns   : (B, 18) float one-hot
        """
        g = torch.nn.functional.one_hot(grind_idx, GRIND_BINS).float()
        d = torch.nn.functional.one_hot(dose_idx, DOSE_BINS).float()
        return torch.cat([g, d], dim=-1)

"""
WorldModel: encoder + RSSM + decoder heads + one-step training.

Training loss (ELBO-style):
    L = recon_loss(profile) + reward_loss + cont_loss + kl_scale * KL(post || prior)
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from .decoders import ContinueDecoder, ProfileDecoder, RewardDecoder
from .encoder import ObservationEncoder
from .rssm import RSSM
from .utils import kl_categorical

logger = logging.getLogger(__name__)

KL_SCALE     = 0.1
KL_FREE_BITS = 1.0
GRAD_CLIP    = 100.0
LR           = 3e-4


@dataclass
class WMTrainMetrics:
    loss_total:  float
    loss_recon:  float
    loss_reward: float
    loss_cont:   float
    loss_kl:     float


class WorldModel(nn.Module):
    """
    Wraps all world model components and exposes:
        encode()      — single-observation embedding
        observe_seq() — run RSSM over a sequence, returns latent states
        train_step()  — one gradient step on a batch
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder  = ObservationEncoder()
        self.rssm     = RSSM()
        self.profile_dec = ProfileDecoder()
        self.reward_dec  = RewardDecoder()
        self.cont_dec    = ContinueDecoder()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def encode(
        self,
        profile:      torch.Tensor,   # (B, 5, 100)
        grind_um:     torch.Tensor,   # (B,)
        dose_g:       torch.Tensor,   # (B,)
        step_size_um: torch.Tensor,   # (B,)
    ) -> torch.Tensor:                # (B, embed_dim)
        return self.encoder(profile, grind_um, dose_g, step_size_um)

    def observe_sequence(
        self,
        batch: dict[str, torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Roll the RSSM over a batch of sequences (training mode — uses posterior).

        batch keys:
            obs     : (B, T, 5, 100)
            actions : (B, T, 2)  [grind_idx, dose_idx]

        Returns:
            h_seq         : (B, T, H)
            z_seq         : (B, T, Z_DIM)
            post_logits   : (B, T, K, M)
            prior_logits  : (B, T, K, M)
        """
        obs     = batch["obs"]         # (B, T, 5, 100)
        actions = batch["actions"]     # (B, T, 2)
        B, T    = obs.shape[:2]

        # Use the scalar fields from the first shot in each sequence as proxies
        # (in practice these change slowly and the encoder is robust to this)
        grind_um     = torch.zeros(B, device=device)
        dose_g       = torch.zeros(B, device=device)
        step_size_um = torch.ones(B, device=device) * 10.0  # placeholder

        h, z = self.rssm.initial_state(B, device)

        h_list, z_list, post_list, prior_list = [], [], [], []

        for t in range(T):
            obs_t   = obs[:, t]                       # (B, 5, 100)
            act_t   = actions[:, t]                   # (B, 2)

            embed_t = self.encoder(obs_t, grind_um, dose_g, step_size_um)  # (B, E)
            act_enc = self.rssm.encode_action(act_t[:, 0], act_t[:, 1])    # (B, 22)

            h, z, post_logits, prior_logits = self.rssm.observe_step(h, z, act_enc, embed_t)

            h_list.append(h)
            z_list.append(z)
            post_list.append(post_logits)
            prior_list.append(prior_logits)

        return (
            torch.stack(h_list, dim=1),     # (B, T, H)
            torch.stack(z_list, dim=1),     # (B, T, Z_DIM)
            torch.stack(post_list, dim=1),  # (B, T, K, M)
            torch.stack(prior_list, dim=1), # (B, T, K, M)
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch:  dict[str, torch.Tensor],
        device: torch.device,
    ) -> WMTrainMetrics:
        self.train()
        self.optimizer.zero_grad()

        obs     = batch["obs"]      # (B, T, 5, 100)
        rewards = batch["rewards"]  # (B, T)
        conts   = batch["conts"]    # (B, T)
        B, T    = obs.shape[:2]

        h_seq, z_seq, post_logits, prior_logits = self.observe_sequence(batch, device)

        # Flatten batch+time for decoder calls: (B*T, ...)
        h_flat = h_seq.reshape(B * T, -1)
        z_flat = z_seq.reshape(B * T, -1)

        # ---- Reconstruction loss ----
        obs_flat  = obs.reshape(B * T, 5, 100)
        loss_recon = self.profile_dec.loss(h_flat, z_flat, obs_flat)

        # ---- Reward prediction loss ----
        loss_reward = self.reward_dec.loss(h_flat, z_flat, rewards.reshape(B * T))

        # ---- Continue prediction loss ----
        loss_cont = self.cont_dec.loss(
            h_flat, z_flat,
            conts.reshape(B * T, 1),
        )

        # ---- KL divergence ----
        # post_logits/prior_logits: (B, T, K, M) → (B*T, K, M)
        kl = kl_categorical(
            post_logits.reshape(B * T, *post_logits.shape[2:]),
            prior_logits.reshape(B * T, *prior_logits.shape[2:]),
            free_bits=KL_FREE_BITS,
        )
        loss_kl = kl.mean()

        loss = loss_recon + loss_reward + loss_cont + KL_SCALE * loss_kl

        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), GRAD_CLIP)
        self.optimizer.step()

        return WMTrainMetrics(
            loss_total  = loss.item(),
            loss_recon  = loss_recon.item(),
            loss_reward = loss_reward.item(),
            loss_cont   = loss_cont.item(),
            loss_kl     = loss_kl.item(),
        )

    # ------------------------------------------------------------------
    # Validation (no gradients)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validation_loss(
        self,
        batch:  dict[str, torch.Tensor],
        device: torch.device,
    ) -> float:
        """Reconstruction loss on a held-out batch — used for convergence detection."""
        self.eval()
        obs = batch["obs"]
        B, T = obs.shape[:2]

        h_seq, z_seq, _, _ = self.observe_sequence(batch, device)
        h_flat = h_seq.reshape(B * T, -1)
        z_flat = z_seq.reshape(B * T, -1)
        obs_flat = obs.reshape(B * T, 5, 100)

        return self.profile_dec.loss(h_flat, z_flat, obs_flat).item()

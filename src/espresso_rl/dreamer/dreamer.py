"""
DreamerV3 top-level class.

Responsibilities:
  1. World model training on real shot sequences (ELBO)
  2. Actor-critic training on imagined rollouts (ÃŽÂ»-returns, REINFORCE)
  3. BOÃ¢â€ â€™DreamerV3 transition: convergence detection via held-out reconstruction loss
  4. Checkpoint save / load
  5. Recommendation inference from latest latent state
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..models import ShotRecord
from .actor import FactoredCategoricalActor
from .critic import SymlogCritic
from .dataset import sample_batch
from .rssm import STATE_DIM, GRIND_BINS, DOSE_BINS
from .utils import lambda_returns
from .world_model import WorldModel

logger = logging.getLogger(__name__)

# ---- Hyper-parameters --------------------------------------------------------
IMAGINATION_HORIZON = 15
GAMMA               = 0.997
LAMBDA_             = 0.95
ENTROPY_SCALE       = 3e-4     # DreamerV3 default
GRAD_CLIP           = 100.0
LR_ACTOR            = 3e-5
LR_CRITIC           = 3e-4

# Steps per shot arrival
WM_STEPS_PER_SHOT   = 50   # world model gradient steps
AC_STEPS_PER_SHOT   = 25   # actor-critic gradient steps

# Convergence detection
VAL_WINDOW          = 5    # evaluate held-out loss every this many WM steps
PLATEAU_CHECKS      = 5    # consecutive checks with <PLATEAU_THRESH improvement
PLATEAU_THRESH      = 0.01 # 1% relative improvement needed

# Minimum rated shots before even trying to switch
MIN_SHOTS_FOR_SWITCH = 25

BATCH_SIZE          = 16
HELD_OUT            = 4    # shots reserved for validation


@dataclass
class DreamerMetrics:
    wm_loss:    float = 0.0
    ac_loss:    float = 0.0
    val_loss:   float | None = None
    mode:       str = "bo"


class DreamerV3:
    def __init__(self, weights_path: Path | None = None) -> None:
        self.device = torch.device("cpu")

        self.world_model = WorldModel().to(self.device)
        self.actor  = FactoredCategoricalActor().to(self.device)
        self.critic = SymlogCritic().to(self.device)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Running convergence tracker
        self._val_losses: deque[float] = deque(maxlen=PLATEAU_CHECKS + 1)
        self._plateau_count = 0
        self._wm_steps_since_eval = 0

        # Most-recent latent state (updated after each real shot)
        self._last_h: torch.Tensor | None = None
        self._last_z: torch.Tensor | None = None

        if weights_path and weights_path.exists():
            self.load(weights_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on_shots(self, shots: list[ShotRecord]) -> DreamerMetrics:
        """
        Full training cycle after a new shot arrives.
        Runs WM_STEPS_PER_SHOT world-model steps then AC_STEPS_PER_SHOT
        actor-critic imagination steps.
        """
        rated = [s for s in shots if s.reward is not None]
        metrics = DreamerMetrics()

        if len(rated) < 4:
            return metrics  # not enough data yet

        # Split train / validation
        train_shots = rated[:-HELD_OUT] if len(rated) > HELD_OUT else rated
        val_shots   = rated[-HELD_OUT:] if len(rated) > HELD_OUT else rated

        # --- World model ---
        wm_total = 0.0
        for _ in range(WM_STEPS_PER_SHOT):
            batch = sample_batch(train_shots, BATCH_SIZE, device=self.device)
            if batch is None:
                break
            m = self.world_model.train_step(batch, self.device)
            wm_total += m.loss_total
            self._wm_steps_since_eval += 1

            if self._wm_steps_since_eval >= VAL_WINDOW:
                val_batch = sample_batch(val_shots, min(4, len(val_shots)), device=self.device)
                if val_batch is not None:
                    vl = self.world_model.validation_loss(val_batch, self.device)
                    self._val_losses.append(vl)
                    metrics.val_loss = vl
                    logger.debug("WM val_loss=%.4f", vl)
                self._wm_steps_since_eval = 0

        metrics.wm_loss = wm_total / max(WM_STEPS_PER_SHOT, 1)

        # --- Actor-critic ---
        ac_total = 0.0
        for _ in range(AC_STEPS_PER_SHOT):
            seed_batch = sample_batch(train_shots, BATCH_SIZE, device=self.device)
            if seed_batch is None:
                break
            loss = self._train_actor_critic_step(seed_batch)
            ac_total += loss

        metrics.ac_loss = ac_total / max(AC_STEPS_PER_SHOT, 1)

        # Update last known latent state from the most recent real shot
        self._update_last_state(rated[-1])

        return metrics

    # ------------------------------------------------------------------
    # Imagination rollout + actor-critic update
    # ------------------------------------------------------------------

    def _train_actor_critic_step(self, seed_batch: dict[str, torch.Tensor]) -> float:
        """One imagination rollout + combined actor + critic gradient step."""
        self.world_model.eval()
        self.actor.train()
        self.critic.train()

        # Encode seed states (use last time-step of each sequence as starting point)
        with torch.no_grad():
            h_seq, z_seq, _, _ = self.world_model.observe_sequence(seed_batch, self.device)

        h = h_seq[:, -1].detach()   # (B, H)
        z = z_seq[:, -1].detach()   # (B, Z_DIM)

        # Imagine H steps
        traj_h, traj_z, traj_g_idx, traj_d_idx, traj_g_lp, traj_d_lp = [], [], [], [], [], []
        traj_rewards, traj_values = [], []

        for _ in range(IMAGINATION_HORIZON):
            g_idx, d_idx, g_lp, d_lp = self.actor.sample(h, z)
            act = self.world_model.rssm.encode_action(g_idx, d_idx)

            # World model prior step
            h, z = self.world_model.rssm.imagine_step(h, z, act)

            r_logits = self.world_model.reward_dec.forward(h, z)
            r_pred   = self.world_model.reward_dec.expected_value(r_logits)
            v, _     = self.critic(h, z)

            traj_h.append(h); traj_z.append(z)
            traj_g_idx.append(g_idx); traj_d_idx.append(d_idx)
            traj_g_lp.append(g_lp);   traj_d_lp.append(d_lp)
            traj_rewards.append(r_pred)
            traj_values.append(v)

        # Bootstrap value from final imagined state
        with torch.no_grad():
            v_last, _ = self.critic(traj_h[-1], traj_z[-1])

        rewards  = torch.stack(traj_rewards)                       # (T, B)
        values   = torch.stack(traj_values + [v_last.detach()])    # (T+1, B)
        conts    = torch.ones_like(rewards)

        returns = lambda_returns(rewards, values, conts, GAMMA, LAMBDA_).detach()  # (T, B)

        # ---- Critic loss ----
        self.critic_opt.zero_grad()
        critic_loss = sum(
            self.critic.loss(traj_h[t], traj_z[t], returns[t])
            for t in range(IMAGINATION_HORIZON)
        ) / IMAGINATION_HORIZON  # type: ignore[assignment]
        critic_loss.backward(retain_graph=True)  # type: ignore[union-attr]
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.critic_opt.step()

        # ---- Actor loss (REINFORCE + entropy) ----
        self.actor_opt.zero_grad()
        log_probs = [traj_g_lp[t] + traj_d_lp[t] for t in range(IMAGINATION_HORIZON)]
        entropy   = sum(
            self.actor.entropy(traj_h[t], traj_z[t])
            for t in range(IMAGINATION_HORIZON)
        ) / IMAGINATION_HORIZON  # type: ignore[assignment]

        advantages = (returns - values[:-1]).detach()
        actor_loss = -sum(
            (lp * adv).mean()
            for lp, adv in zip(log_probs, advantages)
        ) / IMAGINATION_HORIZON - ENTROPY_SCALE * entropy  # type: ignore[operator]

        actor_loss.backward()  # type: ignore[union-attr]
        nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.actor_opt.step()

        return float(critic_loss) + float(actor_loss)

    # ------------------------------------------------------------------
    # Inference Ã¢â‚¬â€ get recommendation from policy
    # ------------------------------------------------------------------

    @torch.no_grad()
    def recommend(
        self,
        shot: ShotRecord,
        microns_per_step: float,
        current_relative_grind_um_from_reference: float,
        current_dose_g: float,
    ) -> dict[str, Any]:
        """
        Encode the most recent shot, advance RSSM with a 'no-op' action,
        then ask the actor for the best action.
        """
        self.world_model.eval()
        self.actor.eval()

        profile = torch.tensor(shot.shot_profile, dtype=torch.float32).unsqueeze(0)
        grind   = torch.tensor([current_relative_grind_um_from_reference], dtype=torch.float32)
        dose    = torch.tensor([current_dose_g], dtype=torch.float32)
        step    = torch.tensor([microns_per_step], dtype=torch.float32)

        embed = self.world_model.encode(profile, grind, dose, step)  # (1, E)

        if self._last_h is None:
            h, z = self.world_model.rssm.initial_state(1, self.device)
        else:
            h, z = self._last_h, self._last_z

        # No-op action (hold current grind, hold current dose)
        g_no_op = torch.tensor([FactoredCategoricalActor.encode_grind(0)])
        d_no_op = torch.tensor([FactoredCategoricalActor.encode_dose(current_dose_g)])
        act = self.world_model.rssm.encode_action(g_no_op, d_no_op)

        h_new, z_new, _, _ = self.world_model.rssm.observe_step(h, z, act, embed)
        self._last_h, self._last_z = h_new, z_new

        # Sample action from policy
        g_idx, d_idx, _, _ = self.actor.sample(h_new, z_new)

        delta_um  = FactoredCategoricalActor.decode_grind(g_idx, microns_per_step).item()
        next_dose = FactoredCategoricalActor.decode_dose(d_idx).item()
        projected_relative_grind_um_from_reference = current_relative_grind_um_from_reference + delta_um

        return {
            "grind_delta_um_from_current":    delta_um,
            "grind_delta_steps_from_current": round(delta_um / microns_per_step),
            "projected_relative_grind_um_from_reference":     projected_relative_grind_um_from_reference,
            "next_dose_g":       next_dose,
            "mode":              "dreamerv3",
        }

    # ------------------------------------------------------------------
    # Convergence detection
    # ------------------------------------------------------------------

    @property
    def has_converged(self) -> bool:
        """
        True when the held-out reconstruction loss has plateaued for
        PLATEAU_CHECKS consecutive evaluations (improvement < PLATEAU_THRESH).
        """
        if len(self._val_losses) < PLATEAU_CHECKS + 1:
            return False
        oldest = list(self._val_losses)[0]
        best_recent = min(list(self._val_losses)[1:])
        return best_recent >= oldest * (1.0 - PLATEAU_THRESH)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        torch.save({
            "world_model": self.world_model.state_dict(),
            "actor":       self.actor.state_dict(),
            "critic":      self.critic.state_dict(),
        }, path)
        logger.info("DreamerV3 weights saved to %s", path)

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(ckpt["world_model"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        logger.info("DreamerV3 weights loaded from %s", path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_last_state(self, shot: ShotRecord) -> None:
        """Encode the most recent real shot and cache the resulting latent state."""
        with torch.no_grad():
            self.world_model.eval()
            profile = torch.tensor(shot.shot_profile, dtype=torch.float32).unsqueeze(0)
            grind   = torch.tensor([shot.relative_grind_um_from_reference],     dtype=torch.float32)
            dose    = torch.tensor([shot.dose_g],        dtype=torch.float32)
            step    = torch.tensor([shot.microns_per_step],  dtype=torch.float32)
            embed   = self.world_model.encode(profile, grind, dose, step)

            if self._last_h is None:
                h, z = self.world_model.rssm.initial_state(1, self.device)
            else:
                h, z = self._last_h, self._last_z

            g_idx = torch.tensor([FactoredCategoricalActor.encode_grind(
                round(shot.action_grind_delta_um_from_current / max(shot.microns_per_step, 1e-6))
            )])
            d_idx = torch.tensor([FactoredCategoricalActor.encode_dose(shot.action_dose_g)])
            act   = self.world_model.rssm.encode_action(g_idx, d_idx)

            h_new, z_new, _, _ = self.world_model.rssm.observe_step(h, z, act, embed)
            self._last_h, self._last_z = h_new, z_new

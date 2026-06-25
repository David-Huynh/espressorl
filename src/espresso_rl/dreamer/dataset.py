"""
Build fixed-length training sequences from the replay buffer for RSSM training.

Each espresso shot is one "time-step" in the world model's sequence.
Consecutive rated shots form the trajectory.

Sequence layout (length T):
    obs[t]     : (5, 100)  Ã¢â‚¬â€ shot profile
    actions[t] : (2,)      Ã¢â‚¬â€ [grind_idx, dose_idx] that produced obs[t]
                             (stored in ShotRecord.action_grind_delta_um_from_current / action_dose_g)
    rewards[t] : scalar    Ã¢â‚¬â€ composite reward for obs[t]
    conts[t]   : 1.0       Ã¢â‚¬â€ episode continues (always; espresso has no hard resets)
"""

import random

import numpy as np
import torch

from ..models import ShotRecord
from .actor import FactoredCategoricalActor

SEQ_LEN    = 8   # steps per training sequence
MIN_SHOTS  = 4   # minimum rated shots needed before we can build any batch


def _encode_action(shot: ShotRecord) -> tuple[int, int]:
    """Convert a ShotRecord's stored action back to discrete indices."""
    delta_steps = shot.action_grind_delta_um_from_current / max(shot.microns_per_step, 1e-6)
    grind_idx = FactoredCategoricalActor.encode_grind(round(delta_steps))
    dose_idx  = FactoredCategoricalActor.encode_dose(shot.action_dose_g)
    return grind_idx, dose_idx


def _pad_window(window: list[ShotRecord], target_len: int) -> list[ShotRecord]:
    """Left-pad a window shorter than target_len by repeating the first element."""
    if len(window) >= target_len:
        return window
    pad = [window[0]] * (target_len - len(window))
    return pad + window


def sample_batch(
    shots: list[ShotRecord],
    batch_size: int = 16,
    seq_len: int    = SEQ_LEN,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor] | None:
    """
    Sample a random batch of contiguous shot sequences.

    Parameters
    ----------
    shots       : rated ShotRecords in chronological order
    batch_size  : number of sequences per batch
    seq_len     : steps per sequence
    device      : torch device

    Returns None if there are not enough shots to build a batch.
    """
    if len(shots) < MIN_SHOTS:
        return None

    obs_list, action_list, reward_list, cont_list = [], [], [], []

    for _ in range(batch_size):
        n = len(shots)
        if n >= seq_len:
            start = random.randint(0, n - seq_len)
            window = shots[start : start + seq_len]
        else:
            # Fewer shots than seq_len Ã¢â‚¬â€ use all and pad
            window = _pad_window(shots, seq_len)

        obs     = np.stack([s.shot_profile for s in window])            # (T, 5, 100)
        rewards = np.array([s.reward or 0.0 for s in window], dtype=np.float32)
        conts   = np.ones(seq_len, dtype=np.float32)                    # always continue
        actions = [_encode_action(s) for s in window]                   # [(g_idx, d_idx), Ã¢â‚¬Â¦]

        obs_list.append(obs)
        action_list.append(actions)
        reward_list.append(rewards)
        cont_list.append(conts)

    return {
        "obs":     torch.tensor(np.stack(obs_list),    dtype=torch.float32,  device=device),  # (B, T, 5, 100)
        "actions": torch.tensor(action_list,            dtype=torch.long,     device=device),  # (B, T, 2)
        "rewards": torch.tensor(np.stack(reward_list), dtype=torch.float32,  device=device),  # (B, T)
        "conts":   torch.tensor(np.stack(cont_list),   dtype=torch.float32,  device=device),  # (B, T)
    }

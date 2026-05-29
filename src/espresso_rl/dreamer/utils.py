"""
DreamerV3 mathematical primitives.

References:
  Hafner et al. 2023 — "Mastering Diverse Domains through World Models"
  https://arxiv.org/abs/2301.04104
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Symlog / symexp
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log: sign(x) * log(|x| + 1). Compresses large values."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# ---------------------------------------------------------------------------
# Two-hot encoding for discrete value regression
# ---------------------------------------------------------------------------

def two_hot(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Encode scalar x as a soft two-hot distribution over bins.
    Operates in symlog space (x is first passed through symlog).

    x    : (*,)
    bins : (B,)
    returns: (*, B)
    """
    x_sl = symlog(x)
    B = bins.shape[0]
    # Index of the bin just below x
    below = (bins <= x_sl.unsqueeze(-1)).sum(-1) - 1
    below = below.clamp(0, B - 2)
    above = below + 1

    b_lo = bins[below]          # (*,)
    b_hi = bins[above]          # (*,)
    frac = ((x_sl - b_lo) / (b_hi - b_lo + 1e-8)).clamp(0.0, 1.0)

    target = torch.zeros(*x.shape, B, device=x.device, dtype=x.dtype)
    target.scatter_(-1, below.unsqueeze(-1), (1.0 - frac).unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), frac.unsqueeze(-1))
    return target


def two_hot_loss(logits: torch.Tensor, targets: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss between predicted logits and two-hot encoded targets.

    logits  : (*, B)
    targets : (*,)  — raw scalar values (will be two-hot encoded internally)
    bins    : (B,)
    """
    target_dist = two_hot(targets, bins)           # (*, B)
    log_probs = F.log_softmax(logits, dim=-1)      # (*, B)
    return -(target_dist * log_probs).sum(-1).mean()


# ---------------------------------------------------------------------------
# Categorical latents (straight-through estimator)
# ---------------------------------------------------------------------------

def straight_through_categorical(logits: torch.Tensor) -> torch.Tensor:
    """
    Sample a one-hot vector from categorical logits with a straight-through
    gradient estimator so gradients flow through the discrete sample.

    logits  : (*, K, M)  — K categories, M classes each
    returns : (*, K*M)   — flattened one-hot with straight-through grad
    """
    *batch, K, M = logits.shape
    probs = F.softmax(logits, dim=-1)                     # (*, K, M)

    # Sample (detached so multinomial doesn't interfere with autograd)
    flat = probs.detach().view(-1, M)
    indices = torch.multinomial(flat, num_samples=1).squeeze(-1)  # (prod(*batch)*K,)
    indices = indices.view(*batch, K)

    hard = F.one_hot(indices, M).float()                  # (*, K, M)

    # Straight-through: use hard in forward pass, probs in backward pass
    z = hard + probs - probs.detach()
    return z.flatten(-2)                                  # (*, K*M)


# ---------------------------------------------------------------------------
# KL divergence between two categorical distributions
# ---------------------------------------------------------------------------

def kl_categorical(
    post_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    free_bits: float = 1.0,
) -> torch.Tensor:
    """
    KL(posterior || prior) per category, summed over categories.
    Applies "free bits" clipping from DreamerV3 (avoids collapsing priors).

    post_logits  : (*, K, M)
    prior_logits : (*, K, M)
    returns      : (*,)
    """
    post_probs = F.softmax(post_logits, dim=-1)    # (*, K, M)
    prior_probs = F.softmax(prior_logits, dim=-1)

    # KL per category: (*, K)
    kl = (post_probs * (torch.log(post_probs + 1e-8) - torch.log(prior_probs + 1e-8))).sum(-1)

    # Free bits: clip minimum per category then sum
    kl = kl.clamp(min=free_bits / post_logits.shape[-2])  # distribute free_bits over K cats
    return kl.sum(-1)                                       # (*,)


# ---------------------------------------------------------------------------
# Lambda returns
# ---------------------------------------------------------------------------

def lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    gamma: float = 0.997,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """
    Generalised lambda returns (TD(λ)).

    rewards   : (T, B)   — rewards at steps 1..T
    values    : (T+1, B) — bootstrapped values at steps 0..T (values[T] = bootstrap)
    continues : (T, B)   — 1 if episode continues, 0 at terminal
    returns   : (T, B)
    """
    G = values[-1]
    returns = []
    for t in reversed(range(rewards.shape[0])):
        G = rewards[t] + gamma * continues[t] * ((1.0 - lambda_) * values[t + 1] + lambda_ * G)
        returns.insert(0, G)
    return torch.stack(returns)  # (T, B)

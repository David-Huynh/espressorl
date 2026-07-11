from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from espresso_rl.domain.cpbo import PreferenceLabel


LABEL_TO_INDEX = {
    PreferenceLabel.NEW_BETTER: 0,
    PreferenceLabel.TIE: 1,
    PreferenceLabel.ANCHOR_BETTER: 2,
}
INDEX_TO_LABEL = tuple(LABEL_TO_INDEX)


def standard_normal_cdf(value: Tensor) -> Tensor:
    return torch.special.ndtr(value)


def jnd_probabilities(
    difference: Tensor,
    *,
    gamma: Tensor | float,
    sigma_pref: Tensor | float,
) -> Tensor:
    """Return [NEW_BETTER, TIE, ANCHOR_BETTER] probabilities.

    `difference` is always f(new) - f(anchor). No semantic probability
    clamping is applied here.
    """

    if not torch.is_floating_point(difference):
        difference = difference.to(dtype=torch.float64)
    gamma_tensor = torch.as_tensor(gamma, dtype=difference.dtype, device=difference.device)
    sigma_tensor = torch.as_tensor(sigma_pref, dtype=difference.dtype, device=difference.device)
    if torch.any(~torch.isfinite(difference)):
        raise ValueError("utility differences must be finite")
    if torch.any(~torch.isfinite(gamma_tensor)) or torch.any(gamma_tensor < 0):
        raise ValueError("gamma must be finite and nonnegative")
    if torch.any(~torch.isfinite(sigma_tensor)) or torch.any(sigma_tensor <= 0):
        raise ValueError("sigma_pref must be finite and positive")

    scale = math.sqrt(2.0) * sigma_tensor
    p_new = standard_normal_cdf((difference - gamma_tensor) / scale)
    p_anchor = standard_normal_cdf((-difference - gamma_tensor) / scale)
    p_tie = standard_normal_cdf((gamma_tensor - difference) / scale) - standard_normal_cdf(
        (-gamma_tensor - difference) / scale
    )
    probabilities = torch.stack((p_new, p_tie, p_anchor), dim=-1)
    if torch.any(~torch.isfinite(probabilities)):
        raise FloatingPointError("JND likelihood produced non-finite probabilities")
    return probabilities


def jnd_log_probabilities(
    difference: Tensor,
    *,
    gamma: Tensor | float,
    sigma_pref: Tensor | float,
    epsilon: float,
) -> Tensor:
    if not 0 < epsilon < 1e-3:
        raise ValueError("epsilon must be small and positive")
    probabilities = jnd_probabilities(
        difference,
        gamma=gamma,
        sigma_pref=sigma_pref,
    )
    return torch.log(probabilities.clamp_min(epsilon))


class ThreeOutcomeJNDLikelihood(nn.Module):
    """Learnable JND threshold with fixed per-run perceptual noise."""

    def __init__(
        self,
        *,
        sigma_pref: float,
        initial_gamma: float,
        learn_gamma: bool = True,
        probability_epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        if not math.isfinite(sigma_pref) or sigma_pref <= 0:
            raise ValueError("sigma_pref must be positive and finite")
        if not math.isfinite(initial_gamma) or initial_gamma < 0:
            raise ValueError("initial_gamma must be nonnegative and finite")
        if not 0 < probability_epsilon < 1e-3:
            raise ValueError("probability_epsilon must be small and positive")
        self.register_buffer("sigma_pref", torch.tensor(float(sigma_pref), dtype=torch.float64))
        raw_gamma = _inverse_softplus(max(initial_gamma, 1e-12))
        self.raw_gamma = nn.Parameter(
            torch.tensor(raw_gamma, dtype=torch.float64),
            requires_grad=bool(learn_gamma),
        )
        self.probability_epsilon = float(probability_epsilon)

    @property
    def gamma(self) -> Tensor:
        return F.softplus(self.raw_gamma)

    def probabilities(self, difference: Tensor) -> Tensor:
        return jnd_probabilities(
            difference,
            gamma=self.gamma,
            sigma_pref=self.sigma_pref,
        )

    def expected_log_prob(self, difference_samples: Tensor, labels: Tensor) -> Tensor:
        if labels.ndim != 1:
            raise ValueError("preference labels must be one-dimensional")
        if difference_samples.shape[-1] != labels.shape[0]:
            raise ValueError("difference samples and labels disagree")
        if torch.any((labels < 0) | (labels > 2)):
            raise ValueError("preference labels must use indices 0..2")
        log_probabilities = jnd_log_probabilities(
            difference_samples,
            gamma=self.gamma,
            sigma_pref=self.sigma_pref,
            epsilon=self.probability_epsilon,
        )
        gather_index = labels.reshape((1,) * (difference_samples.ndim - 1) + (-1, 1))
        gather_index = gather_index.expand(*difference_samples.shape, 1)
        selected = torch.gather(log_probabilities, dim=-1, index=gather_index).squeeze(-1)
        return selected.mean(dim=tuple(range(selected.ndim - 1)))


def preference_label_indices(labels: list[PreferenceLabel]) -> Tensor:
    return torch.tensor([LABEL_TO_INDEX[PreferenceLabel(label)] for label in labels], dtype=torch.long)


def _inverse_softplus(value: float) -> float:
    if value > 20:
        return value
    return math.log(math.expm1(value))

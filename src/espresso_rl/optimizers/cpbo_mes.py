from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from espresso_rl.domain.cpbo import PreferenceLabel
from espresso_rl.optimizers.cpbo_config import MESConfig
from espresso_rl.optimizers.cpbo_jnd import jnd_probabilities, standard_normal_cdf
from espresso_rl.optimizers.cpbo_truncation import sample_upper_truncated_bivariate_gaussian


@dataclass(frozen=True)
class MaximumDistributionApproximation:
    representative_values: Tensor
    weights: Tensor
    strategy: str
    gumbel_location: float | None = None
    gumbel_scale: float | None = None


@dataclass(frozen=True)
class CPBOMESResult:
    candidate_index: int
    acquisition_value: float
    unclipped_acquisition_value: float
    outcome_probabilities: dict[str, float]
    all_acquisition_values: tuple[float, ...]
    maximum_distribution: MaximumDistributionApproximation
    truncation_fallback_count: int


def evaluate_cpbo_mes(
    *,
    posterior_mean: Tensor,
    posterior_covariance: Tensor,
    candidate_indices: tuple[int, ...],
    maximum_indices: tuple[int, ...] | None = None,
    anchor_index: int,
    gamma: float,
    sigma_pref: float,
    config: MESConfig,
    seed: int,
    covariance_jitter: float,
) -> CPBOMESResult:
    mean = torch.as_tensor(posterior_mean, dtype=torch.float64).reshape(-1)
    covariance = torch.as_tensor(posterior_covariance, dtype=torch.float64)
    _validate_posterior(mean, covariance, candidate_indices, anchor_index)
    feasible_maximum_indices = (
        tuple(range(len(mean))) if maximum_indices is None else maximum_indices
    )
    if not feasible_maximum_indices:
        raise ValueError("CPBO MES maximum support cannot be empty")
    if len(set(feasible_maximum_indices)) != len(feasible_maximum_indices):
        raise ValueError("CPBO MES maximum indices must be unique")
    if any(index < 0 or index >= len(mean) for index in feasible_maximum_indices):
        raise ValueError("CPBO MES maximum index is out of range")
    centered_mean, centered_covariance = centered_posterior_moments(mean, covariance)
    maximum_index_tensor = torch.tensor(feasible_maximum_indices, dtype=torch.long)
    anchor_is_feasible = anchor_index in feasible_maximum_indices
    maximum_distribution = approximate_maximum_distribution(
        centered_mean[maximum_index_tensor],
        centered_covariance[maximum_index_tensor][:, maximum_index_tensor],
        config=config,
        seed=seed,
        jitter=covariance_jitter,
    )

    acquisition_values: list[float] = []
    unclipped_values: list[float] = []
    probabilities_by_candidate: list[Tensor] = []
    fallback_count = 0
    for chunk_start in range(0, len(candidate_indices), config.candidate_chunk_size):
        chunk = candidate_indices[chunk_start : chunk_start + config.candidate_chunk_size]
        chunk_probabilities = predictive_outcome_probabilities(
            mean,
            covariance,
            candidate_indices=chunk,
            anchor_index=anchor_index,
            gamma=gamma,
            sigma_pref=sigma_pref,
            variance_roundoff_floor=config.variance_roundoff_floor,
        )
        probabilities_by_candidate.extend(chunk_probabilities)
        unconditional_entropies = categorical_entropy(
            chunk_probabilities,
            epsilon=config.entropy_epsilon,
        )
        for local_index, candidate_index in enumerate(chunk):
            pair_mean = centered_mean[[candidate_index, anchor_index]]
            pair_covariance = centered_covariance[
                [candidate_index, anchor_index]
            ][:, [candidate_index, anchor_index]]
            conditional_entropy = 0.0
            for maximum_index, (maximum_value, weight) in enumerate(
                zip(
                    maximum_distribution.representative_values,
                    maximum_distribution.weights,
                )
            ):
                samples, diagnostics = sample_upper_truncated_bivariate_gaussian(
                    mean=pair_mean,
                    covariance=pair_covariance,
                    upper=torch.stack(
                        (
                            maximum_value,
                            maximum_value
                            if anchor_is_feasible
                            else torch.tensor(float("inf"), dtype=torch.float64),
                        )
                    ),
                    sample_count=config.truncated_samples_per_bin,
                    seed=seed + candidate_index * 100_003 + maximum_index * 997,
                    rejection_batch_size=config.rejection_batch_size,
                    rejection_max_batches=config.rejection_max_batches,
                    rejection_min_acceptance=config.rejection_min_acceptance,
                    gibbs_burn_in=config.gibbs_burn_in,
                    gibbs_thinning=config.gibbs_thinning,
                    jitter=covariance_jitter,
                )
                if diagnostics.method != "rejection":
                    fallback_count += 1
                differences = samples[:, 0] - samples[:, 1]
                response_probabilities = jnd_probabilities(
                    differences,
                    gamma=gamma,
                    sigma_pref=sigma_pref,
                ).mean(dim=0)
                entropy = categorical_entropy(
                    response_probabilities.unsqueeze(0),
                    epsilon=config.entropy_epsilon,
                )[0]
                conditional_entropy += float(weight) * float(entropy)
            unconditional_entropy = float(unconditional_entropies[local_index])
            raw_acquisition = unconditional_entropy - conditional_entropy
            unclipped_values.append(raw_acquisition)
            acquisition_values.append(
                0.0 if -1e-6 <= raw_acquisition < 0.0 else raw_acquisition
            )

    if not acquisition_values or not all(math.isfinite(value) for value in acquisition_values):
        raise FloatingPointError("CPBO MES produced no finite acquisition values")
    best_position = max(range(len(acquisition_values)), key=acquisition_values.__getitem__)
    if acquisition_values[best_position] < 0.0:
        raise FloatingPointError(
            "all CPBO MES candidates had materially negative approximate information gain"
        )
    selected_probabilities = probabilities_by_candidate[best_position]
    return CPBOMESResult(
        candidate_index=candidate_indices[best_position],
        acquisition_value=acquisition_values[best_position],
        unclipped_acquisition_value=unclipped_values[best_position],
        outcome_probabilities={
            PreferenceLabel.NEW_BETTER.value: float(selected_probabilities[0]),
            PreferenceLabel.TIE.value: float(selected_probabilities[1]),
            PreferenceLabel.ANCHOR_BETTER.value: float(selected_probabilities[2]),
        },
        all_acquisition_values=tuple(acquisition_values),
        maximum_distribution=maximum_distribution,
        truncation_fallback_count=fallback_count,
    )


def predictive_outcome_probabilities(
    mean: Tensor,
    covariance: Tensor,
    *,
    candidate_indices: tuple[int, ...],
    anchor_index: int,
    gamma: float,
    sigma_pref: float,
    variance_roundoff_floor: float,
) -> Tensor:
    indices = torch.tensor(candidate_indices, dtype=torch.long)
    mean_difference = mean[indices] - mean[anchor_index]
    difference_variance = (
        covariance[indices, indices]
        + covariance[anchor_index, anchor_index]
        - 2.0 * covariance[indices, anchor_index]
    )
    if torch.any(difference_variance < -variance_roundoff_floor):
        raise FloatingPointError("candidate-anchor difference variance is materially negative")
    difference_variance = difference_variance.clamp_min(0.0)
    total_sd = torch.sqrt(difference_variance + 2.0 * sigma_pref**2)
    p_new = standard_normal_cdf((mean_difference - gamma) / total_sd)
    p_anchor = standard_normal_cdf((-mean_difference - gamma) / total_sd)
    p_tie = standard_normal_cdf((gamma - mean_difference) / total_sd) - standard_normal_cdf(
        (-gamma - mean_difference) / total_sd
    )
    probabilities = torch.stack((p_new, p_tie, p_anchor), dim=-1)
    if torch.any(~torch.isfinite(probabilities)):
        raise FloatingPointError("predictive preference probabilities are non-finite")
    totals = probabilities.sum(dim=-1)
    if not torch.allclose(totals, torch.ones_like(totals), atol=1e-8, rtol=1e-8):
        raise FloatingPointError("predictive preference probabilities do not sum to one")
    return probabilities


def categorical_entropy(probabilities: Tensor, *, epsilon: float) -> Tensor:
    if probabilities.shape[-1] != 3:
        raise ValueError("CPBO entropy requires exactly three outcomes")
    return -torch.sum(probabilities * torch.log(probabilities.clamp_min(epsilon)), dim=-1)


def centered_posterior_moments(mean: Tensor, covariance: Tensor) -> tuple[Tensor, Tensor]:
    n = len(mean)
    centering = torch.eye(n, dtype=torch.float64) - torch.full(
        (n, n),
        1.0 / n,
        dtype=torch.float64,
    )
    centered_mean = centering @ mean
    centered_covariance = centering @ covariance @ centering.T
    centered_covariance = 0.5 * (centered_covariance + centered_covariance.T)
    return centered_mean, centered_covariance


def approximate_maximum_distribution(
    centered_mean: Tensor,
    centered_covariance: Tensor,
    *,
    config: MESConfig,
    seed: int,
    jitter: float,
) -> MaximumDistributionApproximation:
    direct_maxima = sample_zero_centered_function_maxima(
        centered_mean,
        centered_covariance,
        sample_count=config.posterior_max_function_samples,
        seed=seed + 17,
        jitter=jitter,
    )
    if config.maximum_strategy == "direct_max_samples":
        representatives, weights = _weighted_bins(direct_maxima, config.maximum_value_bins)
        return MaximumDistributionApproximation(
            representative_values=representatives,
            weights=weights,
            strategy="direct_max_samples",
        )

    probabilities = torch.tensor([0.25, 0.50, 0.75], dtype=torch.float64)
    quantiles = torch.quantile(direct_maxima, probabilities)
    gumbel_coordinates = -torch.log(-torch.log(probabilities))
    design = torch.stack((torch.ones_like(gumbel_coordinates), gumbel_coordinates), dim=-1)
    solution = torch.linalg.lstsq(design, quantiles.unsqueeze(-1)).solution.squeeze(-1)
    location = solution[0]
    scale = solution[1]
    if not torch.isfinite(scale) or scale <= 1e-9:
        scale = torch.std(direct_maxima, unbiased=False).clamp_min(1e-6) * math.sqrt(6.0) / math.pi
        location = torch.median(direct_maxima) + scale * math.log(math.log(2.0))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 29)
    ranks = torch.rand(config.gumbel_maximum_samples, dtype=torch.float64, generator=generator)
    ranks = config.rank_epsilon + ranks * (1.0 - 2.0 * config.rank_epsilon)
    maxima = location - scale * torch.log(-torch.log(ranks))
    representatives, weights = _weighted_bins(maxima, config.maximum_value_bins)
    return MaximumDistributionApproximation(
        representative_values=representatives,
        weights=weights,
        strategy="paper_gumbel",
        gumbel_location=float(location),
        gumbel_scale=float(scale),
    )


def sample_zero_centered_function_maxima(
    mean: Tensor,
    covariance: Tensor,
    *,
    sample_count: int,
    seed: int,
    jitter: float,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cholesky = _stable_cholesky(covariance, jitter)
    noise = torch.randn((sample_count, len(mean)), dtype=torch.float64, generator=generator)
    samples = mean + noise @ cholesky.T
    samples = samples - samples.mean(dim=-1, keepdim=True)
    maxima = torch.max(samples, dim=-1).values
    if torch.any(~torch.isfinite(maxima)):
        raise FloatingPointError("posterior maximum samples are non-finite")
    return maxima


def _weighted_bins(samples: Tensor, bin_count: int) -> tuple[Tensor, Tensor]:
    if len(samples) < 1:
        raise ValueError("maximum approximation requires samples")
    minimum = torch.min(samples)
    maximum = torch.max(samples)
    if float(maximum - minimum) <= 1e-12:
        return samples[:1], torch.ones(1, dtype=torch.float64)
    edges = torch.linspace(float(minimum), float(maximum), bin_count + 1, dtype=torch.float64)
    assignments = torch.bucketize(samples, edges[1:-1])
    representatives: list[Tensor] = []
    counts: list[int] = []
    for index in range(bin_count):
        values = samples[assignments == index]
        if len(values):
            representatives.append(torch.mean(values))
            counts.append(len(values))
    weights = torch.tensor(counts, dtype=torch.float64)
    weights /= weights.sum()
    return torch.stack(representatives), weights


def _stable_cholesky(covariance: Tensor, jitter: float) -> Tensor:
    identity = torch.eye(covariance.shape[0], dtype=torch.float64)
    current = jitter
    for _ in range(8):
        cholesky, info = torch.linalg.cholesky_ex(covariance + identity * current)
        if int(info) == 0:
            return cholesky
        current *= 10.0
    raise FloatingPointError("posterior covariance could not be stabilized")


def _validate_posterior(
    mean: Tensor,
    covariance: Tensor,
    candidate_indices: tuple[int, ...],
    anchor_index: int,
) -> None:
    if covariance.shape != (len(mean), len(mean)) or len(mean) < 2:
        raise ValueError("CPBO MES posterior dimensions are invalid")
    if not candidate_indices:
        raise ValueError("CPBO MES requires at least one candidate")
    if not 0 <= anchor_index < len(mean):
        raise ValueError("CPBO MES anchor index is invalid")
    if any(index == anchor_index or not 0 <= index < len(mean) for index in candidate_indices):
        raise ValueError("CPBO MES candidate indices are invalid")
    if torch.any(~torch.isfinite(mean)) or torch.any(~torch.isfinite(covariance)):
        raise ValueError("CPBO MES posterior must be finite")
    if not torch.allclose(covariance, covariance.T, atol=1e-8, rtol=1e-8):
        raise ValueError("CPBO MES covariance must be symmetric")

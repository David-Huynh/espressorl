from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TruncatedGaussianDiagnostics:
    method: str
    rejection_draws: int
    rejection_accepts: int
    acceptance_rate: float
    jitter_used: float


def sample_upper_truncated_bivariate_gaussian(
    *,
    mean: Tensor,
    covariance: Tensor,
    upper: Tensor | float,
    sample_count: int,
    seed: int,
    rejection_batch_size: int,
    rejection_max_batches: int,
    rejection_min_acceptance: float,
    gibbs_burn_in: int,
    gibbs_thinning: int,
    jitter: float,
) -> tuple[Tensor, TruncatedGaussianDiagnostics]:
    mean = torch.as_tensor(mean, dtype=torch.float64).reshape(-1)
    covariance = torch.as_tensor(covariance, dtype=torch.float64)
    upper = torch.as_tensor(upper, dtype=torch.float64).reshape(-1)
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("truncated Gaussian requires a bivariate mean and covariance")
    if upper.numel() == 1:
        upper = upper.repeat(2)
    if upper.shape != (2,):
        raise ValueError("truncation upper bound must be scalar or length two")
    if sample_count < 1 or rejection_batch_size < 1 or rejection_max_batches < 1:
        raise ValueError("truncation sample counts must be positive")
    if gibbs_burn_in < 0 or gibbs_thinning < 1:
        raise ValueError("Gibbs burn-in and thinning are invalid")
    if not 0 < rejection_min_acceptance <= 1:
        raise ValueError("minimum rejection acceptance is invalid")
    if torch.any(~torch.isfinite(mean)) or torch.any(~torch.isfinite(covariance)):
        raise ValueError("truncated Gaussian mean and covariance must be finite")
    if torch.any(torch.isnan(upper)) or torch.any(torch.isneginf(upper)):
        raise ValueError("truncation upper bounds must be finite or positive infinity")
    if not torch.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-10):
        raise ValueError("truncated Gaussian covariance must be symmetric")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cholesky, jitter_used = _stable_cholesky(covariance, jitter)
    accepted: list[Tensor] = []
    accepted_count = 0
    draw_count = 0
    for _ in range(rejection_max_batches):
        noise = torch.randn((rejection_batch_size, 2), dtype=torch.float64, generator=generator)
        draws = mean + noise @ cholesky.T
        mask = torch.all(draws <= upper, dim=-1)
        current = draws[mask]
        draw_count += rejection_batch_size
        if len(current):
            accepted.append(current)
            accepted_count += len(current)
        if accepted_count >= sample_count:
            samples = torch.cat(accepted, dim=0)[:sample_count]
            return samples, TruncatedGaussianDiagnostics(
                method="rejection",
                rejection_draws=draw_count,
                rejection_accepts=accepted_count,
                acceptance_rate=accepted_count / draw_count,
                jitter_used=jitter_used,
            )
        if draw_count >= rejection_batch_size * 2:
            rate = accepted_count / draw_count
            if rate < rejection_min_acceptance:
                break

    prefix = torch.cat(accepted, dim=0) if accepted else torch.empty((0, 2), dtype=torch.float64)
    remaining = sample_count - len(prefix)
    gibbs = _sample_gibbs_upper_truncated(
        mean=mean,
        covariance=covariance + torch.eye(2, dtype=torch.float64) * jitter_used,
        upper=upper,
        sample_count=remaining,
        generator=generator,
        burn_in=gibbs_burn_in,
        thinning=gibbs_thinning,
    )
    samples = torch.cat((prefix[:sample_count], gibbs), dim=0)[:sample_count]
    if samples.shape != (sample_count, 2) or torch.any(~torch.isfinite(samples)):
        raise FloatingPointError("upper-truncated Gaussian sampler failed")
    if torch.any(samples > upper + 1e-10):
        raise FloatingPointError("upper-truncated Gaussian sampler violated a bound")
    return samples, TruncatedGaussianDiagnostics(
        method="gibbs" if len(prefix) == 0 else "rejection_then_gibbs",
        rejection_draws=draw_count,
        rejection_accepts=accepted_count,
        acceptance_rate=(accepted_count / draw_count if draw_count else 0.0),
        jitter_used=jitter_used,
    )


def _sample_gibbs_upper_truncated(
    *,
    mean: Tensor,
    covariance: Tensor,
    upper: Tensor,
    sample_count: int,
    generator: torch.Generator,
    burn_in: int,
    thinning: int,
) -> Tensor:
    if sample_count == 0:
        return torch.empty((0, 2), dtype=torch.float64)
    variances = torch.diagonal(covariance).clamp_min(1e-14)
    covariance_01 = covariance[0, 1]
    conditional_variance_0 = (variances[0] - covariance_01.square() / variances[1]).clamp_min(1e-14)
    conditional_variance_1 = (variances[1] - covariance_01.square() / variances[0]).clamp_min(1e-14)
    state = torch.minimum(mean, upper - 0.1 * torch.sqrt(variances))
    state = torch.minimum(state, upper - 1e-12)
    output = torch.empty((sample_count, 2), dtype=torch.float64)
    total_steps = burn_in + sample_count * thinning
    output_index = 0
    for step in range(total_steps):
        conditional_mean_0 = mean[0] + covariance_01 / variances[1] * (state[1] - mean[1])
        state[0] = _sample_upper_truncated_normal(
            conditional_mean_0,
            torch.sqrt(conditional_variance_0),
            upper[0],
            generator,
        )
        conditional_mean_1 = mean[1] + covariance_01 / variances[0] * (state[0] - mean[0])
        state[1] = _sample_upper_truncated_normal(
            conditional_mean_1,
            torch.sqrt(conditional_variance_1),
            upper[1],
            generator,
        )
        if step >= burn_in and (step - burn_in) % thinning == 0:
            output[output_index] = state
            output_index += 1
    return output


def _sample_upper_truncated_normal(
    mean: Tensor,
    standard_deviation: Tensor,
    upper: Tensor,
    generator: torch.Generator,
) -> Tensor:
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    alpha = (upper - mean) / standard_deviation
    if float(alpha) < -5.0:
        # Inverse-CDF sampling underflows in the far lower tail. By symmetry,
        # sample Z >= -alpha with Robert's exponential rejection sampler and
        # return -Z. The retry cap keeps pathological numerical input finite.
        lower = -float(alpha)
        rate = 0.5 * (lower + math.sqrt(lower * lower + 4.0))
        for _ in range(10_000):
            uniform = torch.rand((), dtype=torch.float64, generator=generator).clamp_min(1e-15)
            proposal = lower - math.log(float(uniform)) / rate
            accept_rank = torch.rand((), dtype=torch.float64, generator=generator).clamp_min(1e-15)
            if math.log(float(accept_rank)) <= -0.5 * (proposal - rate) ** 2:
                return mean - standard_deviation * proposal
        raise FloatingPointError("far-tail truncated-normal sampler exhausted retries")
    upper_probability = normal.cdf(alpha).clamp(min=1e-15, max=1.0)
    rank = torch.rand((), dtype=torch.float64, generator=generator) * upper_probability
    rank = rank.clamp(min=1e-15, max=1.0 - 1e-15)
    return torch.minimum(mean + standard_deviation * normal.icdf(rank), upper)


def _stable_cholesky(covariance: Tensor, jitter: float) -> tuple[Tensor, float]:
    if jitter <= 0 or not math.isfinite(jitter):
        raise ValueError("covariance jitter must be positive and finite")
    identity = torch.eye(2, dtype=torch.float64)
    current = jitter
    for _ in range(8):
        cholesky, info = torch.linalg.cholesky_ex(covariance + identity * current)
        if int(info) == 0:
            return cholesky, current
        current *= 10.0
    raise ValueError("truncated Gaussian covariance is not positive semidefinite")

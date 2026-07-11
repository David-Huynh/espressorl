from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Sequence

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.models import ExactGP
from torch import Tensor

from espresso_rl.domain.models import FixedCadenceShotSequence
from espresso_rl.optimizers.cpbo_config import TraceSurrogateConfig


TRACE_FEATURE_NAMES = (
    "duration_s",
    "time_to_first_flow_s",
    "mean_pressure_bar",
    "peak_pressure_bar",
    "mean_beverage_flow_g_s",
    "peak_beverage_flow_g_s",
    "pressure_integral_bar_s",
    "flow_integral_g",
    "early_flow_slope_g_s2",
    "middle_flow_slope_g_s2",
    "late_flow_slope_g_s2",
    "early_to_late_flow_ratio",
    "median_pressure_over_peak_flow",
    "resistance_trend",
    "flow_variability",
)


@dataclass(frozen=True)
class TraceFeatureVector:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(TRACE_FEATURE_NAMES):
            raise ValueError("trace feature vector has an unexpected size")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("trace features must be finite")


@dataclass(frozen=True)
class TracePrediction:
    mean: Tensor
    variance: Tensor
    enabled: bool
    warnings: tuple[str, ...] = ()


class _ExactTraceGP(ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: GaussianLikelihood) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[-1]))

    def forward(self, x: Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class IndependentTraceSurrogate:
    """Independent exact GPs over standardized fixed-length trace summaries."""

    def __init__(self, config: TraceSurrogateConfig) -> None:
        self.config = config
        self.models: list[_ExactTraceGP] = []
        self.likelihoods: list[GaussianLikelihood] = []
        self.feature_median: Tensor | None = None
        self.feature_scale: Tensor | None = None
        self.enabled = False
        self.warnings: tuple[str, ...] = ()

    def fit(
        self,
        train_x: Tensor,
        trace_rows: Tensor,
        *,
        warm_start_checkpoint: str | None = None,
    ) -> bool:
        train_x = torch.as_tensor(train_x, dtype=torch.float64)
        trace_rows = torch.as_tensor(trace_rows, dtype=torch.float64)
        self.models = []
        self.likelihoods = []
        warnings: list[str] = []
        if train_x.ndim != 2 or train_x.shape[-1] != 3:
            raise ValueError("trace surrogate inputs must have shape [n, 3]")
        if trace_rows.ndim != 2 or trace_rows.shape != (train_x.shape[0], len(TRACE_FEATURE_NAMES)):
            raise ValueError("trace surrogate feature matrix has an unexpected shape")
        if train_x.shape[0] < self.config.minimum_valid_telemetry_shots:
            self.enabled = False
            self.warnings = ("trace_kernel_waiting_for_minimum_telemetry_shots",)
            return False
        if torch.any(~torch.isfinite(train_x)) or torch.any(~torch.isfinite(trace_rows)):
            self.enabled = False
            self.warnings = ("trace_kernel_non_finite_training_data",)
            return False

        median = torch.quantile(trace_rows, 0.5, dim=0)
        q1 = torch.quantile(trace_rows, 0.25, dim=0)
        q3 = torch.quantile(trace_rows, 0.75, dim=0)
        scale = (q3 - q1).clamp_min(self.config.feature_epsilon)
        standardized = (trace_rows - median) / scale
        self.feature_median = median
        self.feature_scale = scale
        try:
            warm_states = _decode_trace_checkpoint(warm_start_checkpoint) if warm_start_checkpoint else []
        except ValueError:
            warm_states = []
            warnings.append("trace_warm_start_checkpoint_rejected")

        for feature_index in range(standardized.shape[1]):
            target = standardized[:, feature_index]
            likelihood = GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(
                    self.config.observation_noise_floor
                )
            ).to(dtype=torch.float64)
            model = _ExactTraceGP(train_x, target, likelihood).to(dtype=torch.float64)
            if feature_index < len(warm_states):
                try:
                    _load_matching_finite_state(model, warm_states[feature_index])
                except ValueError:
                    warnings.append(f"trace_feature_{feature_index}_warm_start_rejected")
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
            )
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            best_loss = math.inf
            best_model: dict[str, Tensor] | None = None
            best_likelihood: dict[str, Tensor] | None = None
            stale_steps = 0
            with gpytorch.settings.cholesky_jitter(self.config.jitter):
                for _ in range(self.config.fit_steps):
                    optimizer.zero_grad(set_to_none=True)
                    output = model(train_x)
                    loss = -mll(output, target)
                    if not torch.isfinite(loss):
                        warnings.append(f"trace_feature_{feature_index}_non_finite_loss")
                        break
                    loss.backward()
                    optimizer.step()
                    numeric_loss = float(loss.detach())
                    if numeric_loss < best_loss - 1e-7:
                        best_loss = numeric_loss
                        best_model = copy.deepcopy(model.state_dict())
                        best_likelihood = copy.deepcopy(likelihood.state_dict())
                        stale_steps = 0
                    else:
                        stale_steps += 1
                    if stale_steps >= self.config.early_stopping_patience:
                        break
            if best_model is None or best_likelihood is None:
                self.enabled = False
                self.warnings = tuple(sorted(set(warnings or ["trace_kernel_fit_failed"])))
                return False
            model.load_state_dict(best_model)
            likelihood.load_state_dict(best_likelihood)
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(
                self.config.jitter
            ):
                training_prediction = likelihood(model(train_x)).mean
            rmse = torch.sqrt(torch.mean((training_prediction - target) ** 2))
            if not torch.isfinite(rmse) or float(rmse) > self.config.validation_max_standardized_rmse:
                warnings.append(f"trace_feature_{feature_index}_validation_failed")
                self.enabled = False
                self.warnings = tuple(sorted(set(warnings)))
                return False
            self.models.append(model)
            self.likelihoods.append(likelihood)

        self.enabled = len(self.models) == len(TRACE_FEATURE_NAMES)
        self.warnings = tuple(sorted(set(warnings)))
        return self.enabled

    def checkpoint_json(self) -> str | None:
        if not self.enabled:
            return None
        payload = {
            "schema_version": 1,
            "models": [_finite_state_to_json(model.state_dict()) for model in self.models],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def predict(self, x: Tensor) -> TracePrediction:
        x = torch.as_tensor(x, dtype=torch.float64)
        if x.ndim != 2 or x.shape[-1] != 3:
            raise ValueError("trace prediction inputs must have shape [n, 3]")
        if not self.enabled:
            empty = torch.empty((x.shape[0], 0), dtype=torch.float64, device=x.device)
            return TracePrediction(empty, empty, False, self.warnings)
        means: list[Tensor] = []
        variances: list[Tensor] = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(
            self.config.jitter
        ):
            for model, likelihood in zip(self.models, self.likelihoods):
                predictive = likelihood(model(x))
                means.append(predictive.mean)
                variances.append(predictive.variance.clamp_min(self.config.observation_noise_floor))
        mean = torch.stack(means, dim=-1)
        variance = torch.stack(variances, dim=-1)
        if torch.any(~torch.isfinite(mean)) or torch.any(~torch.isfinite(variance)):
            raise FloatingPointError("trace surrogate prediction is non-finite")
        return TracePrediction(mean, variance, True, self.warnings)


def extract_trace_features(
    sequence: FixedCadenceShotSequence,
    config: TraceSurrogateConfig,
) -> TraceFeatureVector:
    dt = sequence.sample_interval_ms / 1000.0
    pressure = torch.as_tensor(sequence.pressure_bar, dtype=torch.float64)
    flow = torch.as_tensor(sequence.beverage_flow_g_s, dtype=torch.float64)
    duration = (sequence.step_count - 1) * dt
    flowing = torch.nonzero(flow >= config.first_flow_threshold_g_s, as_tuple=False)
    first_flow = float(flowing[0, 0]) * dt if len(flowing) else duration
    third = max(2, sequence.step_count // 3)
    early = flow[:third]
    middle = flow[third : min(2 * third, sequence.step_count)]
    late = flow[min(2 * third, sequence.step_count) :]
    if len(middle) < 2:
        middle = flow
    if len(late) < 2:
        late = flow
    pressure_integral = float(torch.trapezoid(pressure, dx=dt))
    flow_integral = float(torch.trapezoid(flow, dx=dt))
    pressure_over_flow = pressure / flow.clamp_min(config.feature_epsilon)
    resistance_third = max(2, len(pressure_over_flow) // 3)
    resistance_early = torch.median(pressure_over_flow[:resistance_third])
    resistance_late = torch.median(pressure_over_flow[-resistance_third:])
    values = (
        duration,
        first_flow,
        float(torch.mean(pressure)),
        float(torch.max(pressure)),
        float(torch.mean(flow)),
        float(torch.max(flow)),
        pressure_integral,
        flow_integral,
        _linear_slope(early, dt),
        _linear_slope(middle, dt),
        _linear_slope(late, dt),
        float(torch.mean(early) / torch.mean(late).clamp_min(config.feature_epsilon)),
        float(torch.median(pressure) / torch.max(flow).clamp_min(config.feature_epsilon)),
        float(resistance_late - resistance_early),
        float(torch.std(flow, unbiased=False)),
    )
    return TraceFeatureVector(values)


def expected_uncertain_rbf(
    mean_x: Tensor,
    variance_x: Tensor,
    mean_y: Tensor,
    variance_y: Tensor,
    lengthscales: Tensor,
) -> Tensor:
    mean_x = torch.as_tensor(mean_x, dtype=torch.float64)
    variance_x = torch.as_tensor(variance_x, dtype=torch.float64)
    mean_y = torch.as_tensor(mean_y, dtype=torch.float64)
    variance_y = torch.as_tensor(variance_y, dtype=torch.float64)
    lengthscales = torch.as_tensor(lengthscales, dtype=torch.float64).reshape(-1)
    if mean_x.ndim != 2 or mean_y.ndim != 2 or mean_x.shape[1] != mean_y.shape[1]:
        raise ValueError("uncertain RBF means must be compatible matrices")
    if variance_x.shape != mean_x.shape or variance_y.shape != mean_y.shape:
        raise ValueError("uncertain RBF variances must match means")
    if lengthscales.shape[0] != mean_x.shape[1] or torch.any(lengthscales <= 0):
        raise ValueError("uncertain RBF lengthscales are invalid")
    if torch.any(variance_x < 0) or torch.any(variance_y < 0):
        raise ValueError("uncertain RBF variances must be nonnegative")
    lambda_diag = lengthscales.square()
    combined_variance = variance_x[:, None, :] + variance_y[None, :, :]
    denominator = lambda_diag + combined_variance
    determinant_factor = torch.prod(1.0 + combined_variance / lambda_diag, dim=-1).rsqrt()
    mean_delta = mean_x[:, None, :] - mean_y[None, :, :]
    exponent = -0.5 * torch.sum(mean_delta.square() / denominator, dim=-1)
    result = determinant_factor * torch.exp(exponent)
    if torch.any(~torch.isfinite(result)):
        raise FloatingPointError("uncertain RBF kernel produced non-finite covariance")
    return result


def _linear_slope(values: Tensor, dt: float) -> float:
    if len(values) < 2:
        return 0.0
    x = torch.arange(len(values), dtype=torch.float64) * dt
    centered_x = x - torch.mean(x)
    denominator = torch.sum(centered_x.square()).clamp_min(1e-12)
    return float(torch.sum(centered_x * (values - torch.mean(values))) / denominator)


def _finite_state_to_json(state: dict[str, Tensor]) -> dict[str, dict[str, object]]:
    encoded: dict[str, dict[str, object]] = {}
    for name, tensor in state.items():
        value = tensor.detach().cpu().to(torch.float64)
        if torch.any(~torch.isfinite(value)):
            if "_constraint." in name:
                continue
            raise FloatingPointError(f"trace checkpoint tensor {name} is non-finite")
        encoded[name] = {"shape": list(value.shape), "values": value.reshape(-1).tolist()}
    return encoded


def _decode_trace_checkpoint(value: str) -> list[dict[str, object]]:
    if not isinstance(value, str) or len(value.encode("utf-8")) > 4 * 1024 * 1024:
        raise ValueError("trace checkpoint is invalid or too large")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("trace checkpoint is invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "models"}:
        raise ValueError("trace checkpoint fields are invalid")
    if payload["schema_version"] != 1 or not isinstance(payload["models"], list):
        raise ValueError("trace checkpoint schema is unsupported")
    return payload["models"]


def _load_matching_finite_state(model: torch.nn.Module, raw_state: object) -> None:
    if not isinstance(raw_state, dict):
        raise ValueError("trace checkpoint model state is invalid")
    current = model.state_dict()
    unknown = set(raw_state) - set(current)
    if unknown:
        raise ValueError("trace checkpoint contains unknown tensors")
    updated = {name: value.detach().clone() for name, value in current.items()}
    for name, encoded in raw_state.items():
        if not isinstance(encoded, dict) or set(encoded) != {"shape", "values"}:
            raise ValueError("trace checkpoint tensor encoding is invalid")
        shape = encoded["shape"]
        values = encoded["values"]
        if not isinstance(shape, list) or not isinstance(values, list) or math.prod(shape) != len(values):
            raise ValueError("trace checkpoint tensor shape is invalid")
        if tuple(shape) != tuple(updated[name].shape):
            continue
        tensor = torch.tensor(values, dtype=updated[name].dtype, device=updated[name].device).reshape(shape)
        if torch.any(~torch.isfinite(tensor)):
            raise ValueError("trace checkpoint tensor is non-finite")
        updated[name] = tensor
    model.load_state_dict(updated)

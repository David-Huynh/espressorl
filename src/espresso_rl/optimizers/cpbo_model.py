from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ZeroMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import Tensor

from espresso_rl.optimizers.cpbo_config import PreferenceGPConfig
from espresso_rl.optimizers.cpbo_jnd import ThreeOutcomeJNDLikelihood
from espresso_rl.optimizers.cpbo_kernel import PhysicsInformedAdditiveKernel, assert_fixed_output_scale


CHECKPOINT_SCHEMA_VERSION = 1


class VariationalPreferenceGP(ApproximateGP):
    def __init__(
        self,
        inducing_inputs: Tensor,
        *,
        physics_dimensions: int,
        trace_dimensions: int,
        config: PreferenceGPConfig,
    ) -> None:
        inducing_inputs = torch.as_tensor(inducing_inputs, dtype=torch.float64)
        if inducing_inputs.ndim != 2 or len(inducing_inputs) < 1:
            raise ValueError("preference GP requires a nonempty inducing-input matrix")
        distribution = CholeskyVariationalDistribution(inducing_inputs.shape[0])
        strategy = VariationalStrategy(
            self,
            inducing_inputs,
            distribution,
            learn_inducing_locations=False,
        )
        super().__init__(strategy)
        self.mean_module = ZeroMean()
        self.covar_module = PhysicsInformedAdditiveKernel(
            physics_dimensions=physics_dimensions,
            trace_dimensions=trace_dimensions,
            model_config=config,
        )
        self.to(dtype=torch.float64)
        assert_fixed_output_scale(self.covar_module)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean = self.mean_module(x)
        covariance = self.covar_module(x)
        return MultivariateNormal(mean, covariance)


@dataclass(frozen=True)
class PreferenceGPFitResult:
    model: VariationalPreferenceGP
    likelihood: ThreeOutcomeJNDLikelihood
    checkpoint_json: str
    best_loss: float | None
    steps_completed: int
    converged: bool
    warnings: tuple[str, ...]


def fit_preference_gp(
    *,
    train_inputs: Tensor,
    comparison_indices: Tensor,
    labels: Tensor,
    physics_dimensions: int,
    trace_dimensions: int,
    config: PreferenceGPConfig,
    inducing_inputs: Tensor | None = None,
    warm_start_checkpoint: str | None = None,
    random_seed: int,
) -> PreferenceGPFitResult:
    train_inputs = torch.as_tensor(train_inputs, dtype=torch.float64)
    comparison_indices = torch.as_tensor(comparison_indices, dtype=torch.long)
    labels = torch.as_tensor(labels, dtype=torch.long)
    _validate_training_data(train_inputs, comparison_indices, labels)
    if inducing_inputs is None:
        inducing_inputs = select_inducing_inputs(train_inputs, config.inducing_point_cap)
    model = VariationalPreferenceGP(
        inducing_inputs,
        physics_dimensions=physics_dimensions,
        trace_dimensions=trace_dimensions,
        config=config,
    )
    likelihood = ThreeOutcomeJNDLikelihood(
        sigma_pref=config.sigma_pref,
        initial_gamma=config.initial_gamma,
        learn_gamma=config.learn_gamma,
        probability_epsilon=config.probability_epsilon,
    ).to(dtype=torch.float64)
    warnings: list[str] = []
    if warm_start_checkpoint:
        try:
            load_safe_checkpoint(model, likelihood, warm_start_checkpoint)
        except ValueError as exc:
            warnings.append(f"warm_start_rejected:{exc}")

    if comparison_indices.shape[0] == 0:
        model.eval()
        likelihood.eval()
        checkpoint = serialize_safe_checkpoint(model, likelihood)
        return PreferenceGPFitResult(
            model=model,
            likelihood=likelihood,
            checkpoint_json=checkpoint,
            best_loss=None,
            steps_completed=0,
            converged=True,
            warnings=tuple(warnings),
        )

    torch.manual_seed(random_seed)
    model.train()
    likelihood.train()
    parameters = list(model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    best_loss = math.inf
    best_model_state: dict[str, Tensor] | None = None
    best_likelihood_state: dict[str, Tensor] | None = None
    stale_steps = 0
    steps_completed = 0
    converged = False
    with gpytorch.settings.cholesky_jitter(config.covariance_jitter):
        for step in range(config.fit_steps):
            optimizer.zero_grad(set_to_none=True)
            posterior = model(train_inputs)
            function_samples = posterior.rsample(torch.Size((config.likelihood_samples,)))
            new_values = function_samples[:, comparison_indices[:, 0]]
            anchor_values = function_samples[:, comparison_indices[:, 1]]
            differences = new_values - anchor_values
            expected_log_likelihood = likelihood.expected_log_prob(differences, labels).sum()
            kl = model.variational_strategy.kl_divergence().sum()
            loss = -(expected_log_likelihood - kl)
            if not torch.isfinite(loss):
                warnings.append("non_finite_training_loss")
                break
            loss.backward()
            _ensure_finite_gradients(parameters)
            optimizer.step()
            steps_completed = step + 1
            numeric_loss = float(loss.detach())
            if numeric_loss < best_loss - config.minimum_improvement:
                best_loss = numeric_loss
                best_model_state = copy.deepcopy(model.state_dict())
                best_likelihood_state = copy.deepcopy(likelihood.state_dict())
                stale_steps = 0
            else:
                stale_steps += 1
            if stale_steps >= config.early_stopping_patience:
                converged = True
                break
    if best_model_state is None or best_likelihood_state is None:
        raise FloatingPointError("preference GP failed before producing a finite checkpoint")
    model.load_state_dict(best_model_state)
    likelihood.load_state_dict(best_likelihood_state)
    model.eval()
    likelihood.eval()
    assert_fixed_output_scale(model.covar_module)
    checkpoint = serialize_safe_checkpoint(model, likelihood)
    return PreferenceGPFitResult(
        model=model,
        likelihood=likelihood,
        checkpoint_json=checkpoint,
        best_loss=best_loss,
        steps_completed=steps_completed,
        converged=converged or steps_completed == config.fit_steps,
        warnings=tuple(warnings),
    )


def select_inducing_inputs(train_inputs: Tensor, cap: int) -> Tensor:
    if cap < 1:
        raise ValueError("inducing-point cap must be positive")
    if len(train_inputs) <= cap:
        return train_inputs.detach().clone()
    raw = train_inputs[:, :3]
    selected = [0]
    minimum_distance = torch.sum((raw - raw[0]) ** 2, dim=-1)
    while len(selected) < cap:
        index = int(torch.argmax(minimum_distance))
        selected.append(index)
        distance = torch.sum((raw - raw[index]) ** 2, dim=-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        minimum_distance[selected] = -1.0
    return train_inputs[selected].detach().clone()


def posterior_at(
    model: VariationalPreferenceGP,
    inputs: Tensor,
    *,
    jitter: float,
) -> MultivariateNormal:
    inputs = torch.as_tensor(inputs, dtype=torch.float64)
    if inputs.ndim != 2 or torch.any(~torch.isfinite(inputs)):
        raise ValueError("posterior inputs must be a finite matrix")
    model.eval()
    with torch.no_grad(), gpytorch.settings.cholesky_jitter(jitter):
        posterior = model(inputs)
    if torch.any(~torch.isfinite(posterior.mean)) or torch.any(~torch.isfinite(posterior.covariance_matrix)):
        raise FloatingPointError("preference GP posterior is non-finite")
    return posterior


def serialize_safe_checkpoint(
    model: VariationalPreferenceGP,
    likelihood: ThreeOutcomeJNDLikelihood,
) -> str:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": _state_dict_to_json(model.state_dict()),
        "likelihood": _state_dict_to_json(likelihood.state_dict()),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_safe_checkpoint(
    model: VariationalPreferenceGP,
    likelihood: ThreeOutcomeJNDLikelihood,
    checkpoint_json: str,
) -> None:
    if not isinstance(checkpoint_json, str) or not checkpoint_json:
        raise ValueError("checkpoint must be nonempty JSON text")
    try:
        payload = json.loads(checkpoint_json)
    except json.JSONDecodeError as exc:
        raise ValueError("checkpoint is not valid JSON") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "model", "likelihood"}:
        raise ValueError("checkpoint top-level fields are invalid")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema version is unsupported")
    _load_compatible_state(model, payload["model"], allow_variational_expansion=True)
    _load_compatible_state(likelihood, payload["likelihood"], allow_variational_expansion=False)


def _validate_training_data(
    train_inputs: Tensor,
    comparison_indices: Tensor,
    labels: Tensor,
) -> None:
    if train_inputs.ndim != 2 or len(train_inputs) < 1:
        raise ValueError("preference GP training inputs must be a nonempty matrix")
    if torch.any(~torch.isfinite(train_inputs)):
        raise ValueError("preference GP training inputs must be finite")
    if comparison_indices.ndim != 2 or comparison_indices.shape[1] != 2:
        raise ValueError("comparison indices must have shape [n, 2]")
    if labels.ndim != 1 or len(labels) != len(comparison_indices):
        raise ValueError("comparison labels do not match comparison indices")
    if len(comparison_indices):
        if torch.any(comparison_indices < 0) or torch.any(comparison_indices >= len(train_inputs)):
            raise ValueError("comparison index is outside the recipe matrix")
    if torch.any((labels < 0) | (labels > 2)):
        raise ValueError("preference labels must use indices 0..2")


def _ensure_finite_gradients(parameters: Sequence[Tensor]) -> None:
    for parameter in parameters:
        if parameter.grad is not None and torch.any(~torch.isfinite(parameter.grad)):
            raise FloatingPointError("preference GP produced a non-finite gradient")


def _state_dict_to_json(state: Mapping[str, Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, tensor in state.items():
        value = tensor.detach().cpu().to(dtype=torch.float64)
        if torch.any(~torch.isfinite(value)):
            if "_constraint." in name:
                continue
            raise FloatingPointError(f"checkpoint tensor {name} is non-finite")
        result[name] = {
            "shape": list(value.shape),
            "values": value.reshape(-1).tolist(),
        }
    return result


def _load_compatible_state(
    module: torch.nn.Module,
    raw_state: Any,
    *,
    allow_variational_expansion: bool,
) -> None:
    if not isinstance(raw_state, Mapping):
        raise ValueError("checkpoint state must be an object")
    current = module.state_dict()
    unknown = sorted(set(raw_state) - set(current))
    if unknown:
        raise ValueError(f"checkpoint contains unknown tensors: {', '.join(unknown[:3])}")
    updated = {name: tensor.detach().clone() for name, tensor in current.items()}
    for name, encoded in raw_state.items():
        if not isinstance(encoded, Mapping) or set(encoded) != {"shape", "values"}:
            raise ValueError(f"checkpoint tensor {name} has invalid encoding")
        shape = encoded["shape"]
        values = encoded["values"]
        if not isinstance(shape, list) or not all(isinstance(size, int) and size >= 0 for size in shape):
            raise ValueError(f"checkpoint tensor {name} has invalid shape")
        if not isinstance(values, list) or math.prod(shape) != len(values):
            raise ValueError(f"checkpoint tensor {name} has invalid value count")
        source = torch.tensor(values, dtype=updated[name].dtype, device=updated[name].device).reshape(shape)
        if torch.any(~torch.isfinite(source)):
            raise ValueError(f"checkpoint tensor {name} is non-finite")
        target = updated[name]
        if source.shape == target.shape:
            updated[name] = source
            continue
        expandable = (
            allow_variational_expansion
            and "_variational_distribution" in name
            and source.ndim == target.ndim
            and all(old <= new for old, new in zip(source.shape, target.shape))
        )
        if expandable:
            slices = tuple(slice(0, size) for size in source.shape)
            target[slices] = source
            updated[name] = target
    module.load_state_dict(updated)

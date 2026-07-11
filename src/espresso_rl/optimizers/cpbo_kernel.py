from __future__ import annotations

import math
from typing import Mapping

import gpytorch
import torch
from gpytorch.kernels import Kernel, MaternKernel
from torch import Tensor, nn
from torch.nn import functional as F

from espresso_rl.optimizers.cpbo_config import PreferenceGPConfig
from espresso_rl.optimizers.cpbo_trace import expected_uncertain_rbf


class PhysicsInformedAdditiveKernel(Kernel):
    """Fixed-scale convex mixture over raw, proxy-physics, and trace kernels."""

    has_lengthscale = False

    def __init__(
        self,
        *,
        physics_dimensions: int,
        trace_dimensions: int,
        model_config: PreferenceGPConfig,
    ) -> None:
        super().__init__()
        if physics_dimensions < 1 or trace_dimensions < 0:
            raise ValueError("kernel feature dimensions are invalid")
        self.physics_dimensions = physics_dimensions
        self.trace_dimensions = trace_dimensions
        self.raw_kernel = MaternKernel(nu=2.5, ard_num_dims=3)
        self.physics_kernel = MaternKernel(nu=1.5, ard_num_dims=physics_dimensions)
        self.trace_enabled = trace_dimensions > 0
        initial_weights = [
            model_config.initial_raw_kernel_weight,
            model_config.initial_physics_kernel_weight,
        ]
        if self.trace_enabled:
            initial_weights.append(max(model_config.initial_trace_kernel_weight, 1e-6))
            self.raw_trace_lengthscales = nn.Parameter(
                torch.zeros(trace_dimensions, dtype=torch.float64)
            )
        else:
            self.register_parameter("raw_trace_lengthscales", None)
        weights = torch.tensor(initial_weights, dtype=torch.float64).clamp_min(1e-12)
        weights = weights / weights.sum()
        self.weight_logits = nn.Parameter(torch.log(weights))

    @property
    def mixture_weights(self) -> Tensor:
        return torch.softmax(self.weight_logits, dim=0)

    @property
    def trace_lengthscales(self) -> Tensor:
        if self.raw_trace_lengthscales is None:
            return torch.empty(0, dtype=torch.float64, device=self.weight_logits.device)
        return F.softplus(self.raw_trace_lengthscales) + 1e-6

    def weights_dict(self) -> Mapping[str, float]:
        weights = self.mixture_weights.detach().cpu()
        result = {
            "raw": float(weights[0]),
            "physics": float(weights[1]),
            "trace": 0.0,
        }
        if self.trace_enabled:
            result["trace"] = float(weights[2])
        return result

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **params,
    ) -> Tensor:
        if x1.ndim != 2 or x2.ndim != 2:
            raise ValueError("CPBO additive kernel expects two-dimensional input matrices")
        expected_dimensions = 3 + self.physics_dimensions + 2 * self.trace_dimensions
        if x1.shape[-1] != expected_dimensions or x2.shape[-1] != expected_dimensions:
            raise ValueError("CPBO additive kernel input has an unexpected feature count")

        raw_x1, physics_x1, trace_mean_x1, trace_var_x1 = self._split(x1)
        raw_x2, physics_x2, trace_mean_x2, trace_var_x2 = self._split(x2)
        raw = self.raw_kernel(raw_x1, raw_x2, diag=diag, **params)
        physics = self.physics_kernel(physics_x1, physics_x2, diag=diag, **params)
        if not isinstance(raw, Tensor):
            raw = raw.to_dense()
        if not isinstance(physics, Tensor):
            physics = physics.to_dense()
        raw = _normalize_kernel_output(raw, diag=diag)
        physics = _normalize_kernel_output(physics, diag=diag)
        components = [raw, physics]
        if self.trace_enabled:
            trace = _normalized_expected_trace_kernel(
                trace_mean_x1,
                trace_var_x1,
                trace_mean_x2,
                trace_var_x2,
                self.trace_lengthscales,
            )
            if diag:
                trace = torch.diagonal(trace, dim1=-2, dim2=-1)
            components.append(trace)
        weights = self.mixture_weights.to(dtype=x1.dtype, device=x1.device)
        result = sum(weight * component for weight, component in zip(weights, components))
        if torch.any(~torch.isfinite(result)):
            raise FloatingPointError("CPBO additive kernel produced non-finite covariance")
        return result

    def _split(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        raw = x[:, :3]
        physics_end = 3 + self.physics_dimensions
        physics = x[:, 3:physics_end]
        if not self.trace_enabled:
            empty = x.new_empty((x.shape[0], 0))
            return raw, physics, empty, empty
        trace_end = physics_end + self.trace_dimensions
        trace_mean = x[:, physics_end:trace_end]
        trace_variance = x[:, trace_end : trace_end + self.trace_dimensions].clamp_min(0.0)
        return raw, physics, trace_mean, trace_variance


def _normalize_kernel_output(value: Tensor, *, diag: bool) -> Tensor:
    # Matern base kernels have exact unit marginal variance. Keeping this helper
    # explicit makes the fixed latent output-scale invariant auditable.
    return value


def _normalized_expected_trace_kernel(
    mean_x: Tensor,
    variance_x: Tensor,
    mean_y: Tensor,
    variance_y: Tensor,
    lengthscales: Tensor,
) -> Tensor:
    covariance = expected_uncertain_rbf(
        mean_x,
        variance_x,
        mean_y,
        variance_y,
        lengthscales,
    )
    self_x = torch.diagonal(
        expected_uncertain_rbf(mean_x, variance_x, mean_x, variance_x, lengthscales)
    )
    self_y = torch.diagonal(
        expected_uncertain_rbf(mean_y, variance_y, mean_y, variance_y, lengthscales)
    )
    denominator = torch.sqrt(self_x[:, None] * self_y[None, :]).clamp_min(1e-12)
    return covariance / denominator


def assert_fixed_output_scale(kernel: PhysicsInformedAdditiveKernel) -> None:
    if any(isinstance(module, gpytorch.kernels.ScaleKernel) for module in kernel.modules()):
        raise AssertionError("CPBO utility kernel must not contain a ScaleKernel")
    if not math.isclose(sum(kernel.weights_dict().values()), 1.0, abs_tol=1e-8):
        raise AssertionError("CPBO kernel weights must sum to one")

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor

from espresso_rl.domain.cpbo import RecipePoint
from espresso_rl.optimizers.cpbo_config import PhysicsProxyConfig


PHYSICS_FEATURE_NAMES = (
    "dose_g",
    "target_output_g",
    "brew_ratio",
    "fineness",
    "log_particle_size_proxy",
    "log_bed_depth_proxy",
    "log_resistance_proxy",
    "log_flow_proxy",
    "log_expected_duration_proxy",
)


@dataclass(frozen=True)
class PhysicsFeatureResult:
    values: tuple[float, ...]
    fallback_bed_depth: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.values) != len(PHYSICS_FEATURE_NAMES):
            raise ValueError("physics feature vector has an unexpected size")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("physics features must be finite")


@dataclass(frozen=True)
class RobustFeatureScaler:
    median: tuple[float, ...]
    scale: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.median) != len(self.scale) or not self.median:
            raise ValueError("physics scaler dimensions are invalid")
        if not all(math.isfinite(value) for value in self.median):
            raise ValueError("physics scaler median must be finite")
        if not all(math.isfinite(value) and value > 0 for value in self.scale):
            raise ValueError("physics scaler scale must be positive and finite")

    @classmethod
    def fit(
        cls,
        rows: Sequence[Sequence[float]],
        *,
        scale_floor: float,
    ) -> "RobustFeatureScaler":
        if not rows:
            raise ValueError("cannot fit physics scaler without feature rows")
        values = torch.as_tensor(rows, dtype=torch.float64)
        if values.ndim != 2 or torch.any(~torch.isfinite(values)):
            raise ValueError("physics scaler input must be a finite matrix")
        median = torch.quantile(values, 0.5, dim=0)
        q1 = torch.quantile(values, 0.25, dim=0)
        q3 = torch.quantile(values, 0.75, dim=0)
        iqr = q3 - q1
        mad = torch.quantile(torch.abs(values - median), 0.5, dim=0) * 1.4826
        scale = torch.where(iqr > scale_floor, iqr, mad)
        scale = scale.clamp_min(scale_floor)
        return cls(
            median=tuple(float(value) for value in median),
            scale=tuple(float(value) for value in scale),
        )

    def transform(self, rows: Sequence[Sequence[float]] | Tensor) -> Tensor:
        values = torch.as_tensor(rows, dtype=torch.float64)
        median = torch.tensor(self.median, dtype=torch.float64, device=values.device)
        scale = torch.tensor(self.scale, dtype=torch.float64, device=values.device)
        transformed = (values - median) / scale
        if torch.any(~torch.isfinite(transformed)):
            raise FloatingPointError("standardized physics features are non-finite")
        return transformed


def phi0(
    recipe: RecipePoint,
    config: PhysicsProxyConfig,
) -> PhysicsFeatureResult:
    """Configurable recipe-only proxy map used solely as kernel coordinates."""

    fineness = recipe.normalized_x[0]
    diagnostics: list[str] = []
    fallback_bed_depth = config.basket_diameter_mm is None
    if fallback_bed_depth:
        basket_area_cm2 = 1.0
        bed_depth = recipe.dose_g
        diagnostics.append("bed_depth_proxy_fallback_standardized_dose")
    else:
        basket_diameter_cm = config.basket_diameter_mm / 10.0
        basket_area_cm2 = math.pi * (basket_diameter_cm / 2.0) ** 2
        bed_depth = recipe.dose_g / (
            basket_area_cm2 * config.bulk_density_g_cm3 * config.packing_fraction
        )

    if config.calibrated_coarse_particle_size_um is not None:
        coarse = config.calibrated_coarse_particle_size_um
        fine = config.calibrated_fine_particle_size_um
        particle_size = math.exp(
            (1.0 - fineness) * math.log(coarse) + fineness * math.log(fine)
        )
        diagnostics.append("particle_size_proxy_calibrated_endpoint_mapping")
    else:
        particle_size = math.exp(-config.particle_size_beta * fineness)
        diagnostics.append("particle_size_proxy_monotone_fallback")
    permeability = particle_size**2
    resistance = bed_depth / (permeability + config.epsilon)
    flow = config.nominal_pressure_bar / (resistance + config.epsilon)
    expected_duration = recipe.target_output_g / (
        flow
        * basket_area_cm2
        * config.nominal_water_density_g_cm3
        + config.epsilon
    )
    values = (
        recipe.dose_g,
        recipe.target_output_g,
        recipe.brew_ratio,
        fineness,
        math.log(particle_size + config.epsilon),
        math.log(bed_depth + config.epsilon),
        math.log(resistance + config.epsilon),
        math.log(flow + config.epsilon),
        math.log(expected_duration + config.epsilon),
    )
    return PhysicsFeatureResult(
        values=values,
        fallback_bed_depth=fallback_bed_depth,
        diagnostics=tuple(diagnostics),
    )


def physics_feature_matrix(
    recipes: Iterable[RecipePoint],
    config: PhysicsProxyConfig,
) -> tuple[Tensor, tuple[str, ...]]:
    rows: list[tuple[float, ...]] = []
    diagnostics: set[str] = set()
    for recipe in recipes:
        result = phi0(recipe, config)
        rows.append(result.values)
        diagnostics.update(result.diagnostics)
    if not rows:
        return torch.empty((0, len(PHYSICS_FEATURE_NAMES)), dtype=torch.float64), tuple()
    values = torch.tensor(rows, dtype=torch.float64)
    if torch.any(~torch.isfinite(values)):
        raise FloatingPointError("physics feature matrix is non-finite")
    return values, tuple(sorted(diagnostics))

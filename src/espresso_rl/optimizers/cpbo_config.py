from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from espresso_rl.domain.cpbo import ComparisonMode


@dataclass(frozen=True)
class PhysicsProxyConfig:
    basket_diameter_mm: float | None = None
    calibrated_coarse_particle_size_um: float | None = None
    calibrated_fine_particle_size_um: float | None = None
    bulk_density_g_cm3: float = 0.35
    packing_fraction: float = 0.60
    particle_size_beta: float = 3.0
    nominal_pressure_bar: float = 9.0
    nominal_water_density_g_cm3: float = 1.0
    epsilon: float = 1e-9
    robust_scale_floor: float = 1e-6

    def __post_init__(self) -> None:
        positive = (
            self.bulk_density_g_cm3,
            self.packing_fraction,
            self.particle_size_beta,
            self.nominal_pressure_bar,
            self.nominal_water_density_g_cm3,
            self.epsilon,
            self.robust_scale_floor,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("physics proxy constants must be positive and finite")
        if self.basket_diameter_mm is not None and (
            not math.isfinite(self.basket_diameter_mm) or self.basket_diameter_mm <= 0
        ):
            raise ValueError("basket_diameter_mm must be positive when configured")
        particle_mapping = (
            self.calibrated_coarse_particle_size_um,
            self.calibrated_fine_particle_size_um,
        )
        if (particle_mapping[0] is None) != (particle_mapping[1] is None):
            raise ValueError("calibrated particle-size endpoints must be configured together")
        if particle_mapping[0] is not None:
            coarse, fine = particle_mapping
            if not (
                math.isfinite(coarse)
                and math.isfinite(fine)
                and coarse > fine > 0
            ):
                raise ValueError("calibrated particle-size endpoints must satisfy coarse > fine > 0")


@dataclass(frozen=True)
class TraceSurrogateConfig:
    minimum_valid_telemetry_shots: int = 8
    fit_steps: int = 150
    learning_rate: float = 0.05
    early_stopping_patience: int = 25
    jitter: float = 1e-6
    observation_noise_floor: float = 1e-4
    validation_max_standardized_rmse: float = 3.0
    first_flow_threshold_g_s: float = 0.05
    feature_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.minimum_valid_telemetry_shots < 2:
            raise ValueError("trace activation requires at least two telemetry shots")
        if self.fit_steps < 1 or self.early_stopping_patience < 1:
            raise ValueError("trace fit steps and patience must be positive")
        for name in (
            "learning_rate",
            "jitter",
            "observation_noise_floor",
            "validation_max_standardized_rmse",
            "first_flow_threshold_g_s",
            "feature_epsilon",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class PreferenceGPConfig:
    sigma_pref: float = 0.20
    initial_gamma: float = 0.20
    learn_gamma: bool = True
    fit_steps: int = 300
    learning_rate: float = 0.03
    likelihood_samples: int = 128
    early_stopping_patience: int = 50
    minimum_improvement: float = 1e-6
    inducing_point_cap: int = 128
    covariance_jitter: float = 1e-6
    probability_epsilon: float = 1e-12
    initial_raw_kernel_weight: float = 0.80
    initial_physics_kernel_weight: float = 0.20
    initial_trace_kernel_weight: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma_pref) or self.sigma_pref <= 0:
            raise ValueError("sigma_pref must be positive and fixed within a run")
        if not math.isfinite(self.initial_gamma) or self.initial_gamma < 0:
            raise ValueError("initial_gamma must be nonnegative")
        if self.fit_steps < 1 or self.likelihood_samples < 1:
            raise ValueError("GP fit steps and likelihood samples must be positive")
        if self.early_stopping_patience < 1 or self.inducing_point_cap < 1:
            raise ValueError("GP patience and inducing cap must be positive")
        if not 0 < self.probability_epsilon < 1e-3:
            raise ValueError("probability_epsilon must be small and positive")
        weights = (
            self.initial_raw_kernel_weight,
            self.initial_physics_kernel_weight,
            self.initial_trace_kernel_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("initial kernel weights must be finite and nonnegative")
        if sum(weights) <= 0:
            raise ValueError("at least one initial kernel weight must be positive")


@dataclass(frozen=True)
class MESConfig:
    maximum_strategy: str = "paper_gumbel"
    sobol_candidate_count: int = 192
    local_candidate_count: int = 48
    posterior_max_function_samples: int = 512
    gumbel_maximum_samples: int = 2_000
    maximum_value_bins: int = 10
    truncated_samples_per_bin: int = 256
    rank_epsilon: float = 1e-4
    near_duplicate_tolerance: float = 1e-5
    allow_repeat_recipes: bool = False
    candidate_chunk_size: int = 32
    rejection_batch_size: int = 2_048
    rejection_max_batches: int = 8
    rejection_min_acceptance: float = 0.01
    gibbs_burn_in: int = 96
    gibbs_thinning: int = 2
    variance_roundoff_floor: float = 1e-12
    entropy_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.maximum_strategy not in {"paper_gumbel", "direct_max_samples"}:
            raise ValueError("maximum_strategy must be paper_gumbel or direct_max_samples")
        integer_fields = (
            "sobol_candidate_count",
            "local_candidate_count",
            "posterior_max_function_samples",
            "gumbel_maximum_samples",
            "maximum_value_bins",
            "truncated_samples_per_bin",
            "candidate_chunk_size",
            "rejection_batch_size",
            "rejection_max_batches",
            "gibbs_burn_in",
            "gibbs_thinning",
        )
        if any(getattr(self, name) < 1 for name in integer_fields):
            raise ValueError("MES sample counts and batch sizes must be positive")
        if self.maximum_value_bins > self.gumbel_maximum_samples:
            raise ValueError("maximum_value_bins exceeds maximum samples")
        if not 0 < self.rank_epsilon < 0.25:
            raise ValueError("rank_epsilon must be in (0, 0.25)")
        if self.near_duplicate_tolerance < 0:
            raise ValueError("near_duplicate_tolerance must be nonnegative")
        if not 0 < self.rejection_min_acceptance <= 1:
            raise ValueError("rejection_min_acceptance must be in (0, 1]")
        if self.variance_roundoff_floor <= 0 or self.entropy_epsilon <= 0:
            raise ValueError("MES numerical floors must be positive")


@dataclass(frozen=True)
class TrustRegionConfig:
    initial_length: float = 0.8
    minimum_length: float = 0.5**7
    maximum_length: float = 1.6
    success_tolerance: int = 3
    failure_tolerance: int = 4
    shape_factor_min: float = 0.20
    shape_factor_max: float = 5.0

    def __post_init__(self) -> None:
        if not 0 < self.minimum_length < self.initial_length <= self.maximum_length:
            raise ValueError("trust-region length bounds are invalid")
        if self.success_tolerance != 3 or self.failure_tolerance != 4:
            raise ValueError("q=1 three-dimensional CPBO requires success=3 and failure=4")
        if not 0 < self.shape_factor_min <= 1 <= self.shape_factor_max:
            raise ValueError("trust-region shape factor bounds are invalid")


@dataclass(frozen=True)
class CPBOConfig:
    profile_name: str = "application"
    comparison_mode: ComparisonMode = ComparisonMode.BEST_INCUMBENT
    random_seed: int = 17
    model: PreferenceGPConfig = field(default_factory=PreferenceGPConfig)
    acquisition: MESConfig = field(default_factory=MESConfig)
    trust_region: TrustRegionConfig = field(default_factory=TrustRegionConfig)
    physics: PhysicsProxyConfig = field(default_factory=PhysicsProxyConfig)
    trace: TraceSurrogateConfig = field(default_factory=TraceSurrogateConfig)
    grind_domain_radius_steps: float = 10.0
    stepped_grind_resolution: float = 1.0
    stepless_grind_resolution: float = 0.1
    dose_resolution_g: float = 0.1
    target_output_resolution_g: float = 0.1
    checkpoint_max_bytes: int = 4 * 1024 * 1024
    feature_version: str = "espresso_physics_trace_v1"
    configuration_version: str = "cpbo_config_v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparison_mode", ComparisonMode(self.comparison_mode))
        if self.profile_name not in {"application", "paper_fidelity"}:
            raise ValueError("CPBO profile_name must be application or paper_fidelity")
        if not self.feature_version.strip() or not self.configuration_version.strip():
            raise ValueError("CPBO profile and version fields are required")
        if self.random_seed < 0 or self.checkpoint_max_bytes < 1:
            raise ValueError("CPBO seed and checkpoint limit are invalid")
        for name in (
            "grind_domain_radius_steps",
            "stepped_grind_resolution",
            "stepless_grind_resolution",
            "dose_resolution_g",
            "target_output_resolution_g",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["comparison_mode"] = self.comparison_mode.value
        return value

    @property
    def effective_configuration_version(self) -> str:
        payload = self.to_dict()
        declared_version = str(payload.pop("configuration_version"))
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
        return f"{declared_version}:{digest}"


def application_cpbo_config() -> CPBOConfig:
    """Runtime defaults calibrated for espresso latency, not paper reproduction."""

    return CPBOConfig()


def paper_fidelity_cpbo_config() -> CPBOConfig:
    """CPBO paper-scale Monte Carlo settings; sigma remains explicit and configurable."""

    return CPBOConfig(
        profile_name="paper_fidelity",
        comparison_mode=ComparisonMode.GLOBAL_PREVIOUS,
        model=PreferenceGPConfig(
            sigma_pref=0.10,
            initial_gamma=0.10,
            fit_steps=2_000,
            learning_rate=0.03,
            likelihood_samples=1_000,
            early_stopping_patience=250,
        ),
        acquisition=MESConfig(
            sobol_candidate_count=1_024,
            local_candidate_count=128,
            posterior_max_function_samples=1_000,
            gumbel_maximum_samples=25_000,
            maximum_value_bins=20,
            truncated_samples_per_bin=1_000,
            rank_epsilon=1e-4,
            candidate_chunk_size=16,
        ),
        trace=TraceSurrogateConfig(
            minimum_valid_telemetry_shots=8,
            fit_steps=500,
            learning_rate=0.03,
            early_stopping_patience=75,
        ),
    )


def cpbo_config_from_dict(value: Mapping[str, Any] | None) -> CPBOConfig:
    if value is None:
        return application_cpbo_config()
    if not isinstance(value, Mapping):
        raise ValueError("cpbo configuration must be an object")
    allowed = set(CPBOConfig.__dataclass_fields__)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown CPBO configuration fields: {', '.join(unknown)}")
    profile_name = str(value.get("profile_name") or "application")
    base = paper_fidelity_cpbo_config() if profile_name == "paper_fidelity" else application_cpbo_config()
    kwargs: dict[str, Any] = {}
    nested = {
        "model": PreferenceGPConfig,
        "acquisition": MESConfig,
        "trust_region": TrustRegionConfig,
        "physics": PhysicsProxyConfig,
        "trace": TraceSurrogateConfig,
    }
    for key, raw in value.items():
        if key in nested:
            if not isinstance(raw, Mapping):
                raise ValueError(f"cpbo.{key} must be an object")
            current = getattr(base, key)
            field_names = set(current.__dataclass_fields__)
            nested_unknown = sorted(set(raw) - field_names)
            if nested_unknown:
                raise ValueError(f"unknown cpbo.{key} fields: {', '.join(nested_unknown)}")
            kwargs[key] = replace(current, **dict(raw))
        elif key == "comparison_mode":
            kwargs[key] = ComparisonMode(raw)
        else:
            kwargs[key] = raw
    return replace(base, **kwargs)

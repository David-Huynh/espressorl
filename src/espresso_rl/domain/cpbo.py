from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from espresso_rl.domain.models import GrinderStepDirection
from espresso_rl.domain.recipe_limits import (
    RECIPE_DOMAIN_DOSE_MAX_G,
    RECIPE_DOMAIN_DOSE_MIN_G,
    RECIPE_DOMAIN_GRIND_RADIUS_MAX_STEPS,
    RECIPE_DOMAIN_GRIND_RADIUS_MIN_STEPS,
    RECIPE_DOMAIN_OUTPUT_MAX_G,
    RECIPE_DOMAIN_OUTPUT_MIN_G,
)
from espresso_rl.domain.taste_goal import TasteGoal


CPBO_MODEL_VERSION = "cpbo_jnd_mes_v1"
CPBO_CONFIGURATION_VERSION = "cpbo_config_v1"
CPBO_FEATURE_VERSION = "espresso_physics_trace_v1"
RECIPE_DOMAIN_VERSION = "recipe_domain_v1"
_NORMALIZED_DIMENSION = 3
_FLOAT_TOLERANCE = 1e-9


class PreferenceLabel(str, Enum):
    """The label orientation is always new shot relative to anchor shot."""

    NEW_BETTER = "new_better"
    ANCHOR_BETTER = "anchor_better"
    TIE = "tie"


class ComparisonMode(str, Enum):
    GLOBAL_PREVIOUS = "global_previous"
    BEST_INCUMBENT = "best_incumbent"


class CPBOProfile(str, Enum):
    APPLICATION = "application"
    PAPER_FIDELITY = "paper_fidelity"


class PhysicalShotStatus(str, Enum):
    VALID = "valid"
    MACHINE_FAILURE = "machine_failure"
    ABORTED = "aborted"


@dataclass(frozen=True)
class RecipeDomain:
    """Physical search limits used to normalize one CPBO optimization run."""

    grind_radius_steps: float = 10.0
    dose_min_g: float = 6.0
    dose_max_g: float = 30.0
    target_output_min_g: float = 5.0
    target_output_max_g: float = 250.0

    def __post_init__(self) -> None:
        values = (
            self.grind_radius_steps,
            self.dose_min_g,
            self.dose_max_g,
            self.target_output_min_g,
            self.target_output_max_g,
        )
        if any(isinstance(value, bool) for value in values):
            raise ValueError("recipe domain values must be numeric")
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in values):
            raise ValueError("recipe domain values must be positive and finite")
        bounded_values = (
            (
                "grind_radius_steps",
                self.grind_radius_steps,
                RECIPE_DOMAIN_GRIND_RADIUS_MIN_STEPS,
                RECIPE_DOMAIN_GRIND_RADIUS_MAX_STEPS,
            ),
            ("dose_min_g", self.dose_min_g, RECIPE_DOMAIN_DOSE_MIN_G, RECIPE_DOMAIN_DOSE_MAX_G),
            ("dose_max_g", self.dose_max_g, RECIPE_DOMAIN_DOSE_MIN_G, RECIPE_DOMAIN_DOSE_MAX_G),
            (
                "target_output_min_g",
                self.target_output_min_g,
                RECIPE_DOMAIN_OUTPUT_MIN_G,
                RECIPE_DOMAIN_OUTPUT_MAX_G,
            ),
            (
                "target_output_max_g",
                self.target_output_max_g,
                RECIPE_DOMAIN_OUTPUT_MIN_G,
                RECIPE_DOMAIN_OUTPUT_MAX_G,
            ),
        )
        for name, value, minimum, maximum in bounded_values:
            if not minimum <= float(value) <= maximum:
                raise ValueError(f"recipe domain {name} is outside the integrity envelope")
        if self.dose_max_g <= self.dose_min_g:
            raise ValueError("recipe domain dose_max_g must exceed dose_min_g")
        if self.target_output_max_g <= self.target_output_min_g:
            raise ValueError(
                "recipe domain target_output_max_g must exceed target_output_min_g"
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "grind_radius_steps": float(self.grind_radius_steps),
            "dose_min_g": float(self.dose_min_g),
            "dose_max_g": float(self.dose_max_g),
            "target_output_min_g": float(self.target_output_min_g),
            "target_output_max_g": float(self.target_output_max_g),
        }

    @property
    def effective_version(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
        return f"{RECIPE_DOMAIN_VERSION}:{digest}"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecipeDomain":
        if not isinstance(value, Mapping):
            raise ValueError("recipe domain must be an object")
        allowed = {
            "grind_radius_steps",
            "dose_min_g",
            "dose_max_g",
            "target_output_min_g",
            "target_output_max_g",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown recipe domain fields: {', '.join(unknown)}")
        if any(isinstance(value.get(name), bool) for name in value):
            raise ValueError("recipe domain values must be numeric")
        defaults = cls()
        return cls(
            grind_radius_steps=float(value.get("grind_radius_steps", defaults.grind_radius_steps)),
            dose_min_g=float(value.get("dose_min_g", defaults.dose_min_g)),
            dose_max_g=float(value.get("dose_max_g", defaults.dose_max_g)),
            target_output_min_g=float(
                value.get("target_output_min_g", defaults.target_output_min_g)
            ),
            target_output_max_g=float(
                value.get("target_output_max_g", defaults.target_output_max_g)
            ),
        )


@dataclass(frozen=True)
class RecipeParameter:
    name: str
    physical_min: float
    physical_max: float
    resolution: float
    unit: str
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("recipe parameter name is required")
        if not self.unit.strip():
            raise ValueError("recipe parameter unit is required")
        values = (self.physical_min, self.physical_max, self.resolution)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"{self.name} bounds and resolution must be finite")
        if self.physical_max <= self.physical_min:
            raise ValueError(f"{self.name} physical_max must exceed physical_min")
        if self.resolution <= 0:
            raise ValueError(f"{self.name} resolution must be positive")
        if self.resolution > self.physical_max - self.physical_min:
            raise ValueError(f"{self.name} resolution exceeds its physical range")

    def normalize(self, value: float) -> float:
        value = self.validate_physical(value)
        return (value - self.physical_min) / (self.physical_max - self.physical_min)

    def normalize_observation(self, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{self.name} observation must be finite")
        return (value - self.physical_min) / (self.physical_max - self.physical_min)

    def inverse(self, normalized: float) -> float:
        normalized = _normalized_scalar(normalized, self.name)
        return self.physical_min + normalized * (self.physical_max - self.physical_min)

    def quantize(self, value: float) -> float:
        value = self.validate_physical(value, allow_roundoff=True)
        clipped = min(self.physical_max, max(self.physical_min, value))
        step_index = math.floor(
            ((clipped - self.physical_min) / self.resolution) + 0.5 + 1e-12
        )
        quantized = self.physical_min + step_index * self.resolution
        quantized = min(self.physical_max, max(self.physical_min, quantized))
        return float(round(quantized, _decimal_places(self.resolution) + 2))

    def quantize_observation(self, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{self.name} observation must be finite")
        step_index = math.floor(
            ((value - self.physical_min) / self.resolution) + 0.5 + 1e-12
        )
        quantized = self.physical_min + step_index * self.resolution
        return float(round(quantized, _decimal_places(self.resolution) + 2))

    def validate_physical(self, value: float, *, allow_roundoff: bool = False) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{self.name} must be finite")
        tolerance = _FLOAT_TOLERANCE if allow_roundoff else 0.0
        if value < self.physical_min - tolerance or value > self.physical_max + tolerance:
            raise ValueError(f"{self.name} is outside physical bounds")
        return value


@dataclass(frozen=True)
class RecipeSpace:
    grind: RecipeParameter
    dose: RecipeParameter
    target_output: RecipeParameter
    grinder_step_direction: GrinderStepDirection
    version: str = CPBO_CONFIGURATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "grinder_step_direction",
            GrinderStepDirection(self.grinder_step_direction),
        )
        if self.grind.name != "grind_size":
            raise ValueError("grind parameter must be named grind_size")
        if self.dose.name != "dose_g":
            raise ValueError("dose parameter must be named dose_g")
        if self.target_output.name != "target_output_g":
            raise ValueError("target output parameter must be named target_output_g")
        if not self.version.strip():
            raise ValueError("recipe space version is required")

    def quantize_recipe(
        self,
        grind_size: float,
        dose_g: float,
        target_output_g: float,
    ) -> tuple[float, float, float]:
        recipe = (
            self.grind.quantize(grind_size),
            self.dose.quantize(dose_g),
            self.target_output.quantize(target_output_g),
        )
        self.validate_recipe(*recipe)
        return recipe

    def quantize_observation(
        self,
        grind_size: float,
        dose_g: float,
        target_output_g: float,
    ) -> tuple[float, float, float]:
        return (
            self.grind.quantize_observation(grind_size),
            self.dose.quantize_observation(dose_g),
            self.target_output.quantize_observation(target_output_g),
        )

    def validate_recipe(
        self,
        grind_size: float,
        dose_g: float,
        target_output_g: float,
    ) -> None:
        self.grind.validate_physical(grind_size, allow_roundoff=True)
        self.dose.validate_physical(dose_g, allow_roundoff=True)
        self.target_output.validate_physical(target_output_g, allow_roundoff=True)

    def normalize_recipe(
        self,
        grind_size: float,
        dose_g: float,
        target_output_g: float,
    ) -> tuple[float, float, float]:
        self.validate_recipe(grind_size, dose_g, target_output_g)
        physical_grind = self.grind.normalize(grind_size)
        fineness = (
            physical_grind
            if self.grinder_step_direction == GrinderStepDirection.HIGHER_IS_FINER
            else 1.0 - physical_grind
        )
        return (
            float(fineness),
            float(self.dose.normalize(dose_g)),
            float(self.target_output.normalize(target_output_g)),
        )

    def normalize_observation(
        self,
        grind_size: float,
        dose_g: float,
        target_output_g: float,
    ) -> tuple[float, float, float]:
        physical_grind = self.grind.normalize_observation(grind_size)
        fineness = (
            physical_grind
            if self.grinder_step_direction == GrinderStepDirection.HIGHER_IS_FINER
            else 1.0 - physical_grind
        )
        return (
            float(fineness),
            float(self.dose.normalize_observation(dose_g)),
            float(self.target_output.normalize_observation(target_output_g)),
        )

    def inverse_recipe(
        self,
        normalized_x: Sequence[float],
        *,
        quantize: bool = True,
    ) -> tuple[float, float, float]:
        normalized = normalized_recipe(normalized_x)
        physical_grind = (
            normalized[0]
            if self.grinder_step_direction == GrinderStepDirection.HIGHER_IS_FINER
            else 1.0 - normalized[0]
        )
        recipe = (
            self.grind.inverse(physical_grind),
            self.dose.inverse(normalized[1]),
            self.target_output.inverse(normalized[2]),
        )
        if quantize:
            return self.quantize_recipe(*recipe)
        self.validate_recipe(*recipe)
        return recipe

    def is_feasible_normalized(self, normalized_x: Sequence[float]) -> bool:
        try:
            self.inverse_recipe(normalized_x, quantize=True)
        except ValueError:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "grind": _parameter_to_dict(self.grind),
            "dose": _parameter_to_dict(self.dose),
            "target_output": _parameter_to_dict(self.target_output),
            "grinder_step_direction": self.grinder_step_direction.value,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecipeSpace":
        if not isinstance(value, Mapping):
            raise ValueError("recipe space must be an object")
        return cls(
            grind=_parameter_from_dict(value.get("grind")),
            dose=_parameter_from_dict(value.get("dose")),
            target_output=_parameter_from_dict(value.get("target_output")),
            grinder_step_direction=GrinderStepDirection(value.get("grinder_step_direction")),
            version=str(value.get("version") or ""),
        )


@dataclass(frozen=True)
class OptimizationRunContext:
    install_id: str
    machine_id: str
    bean_context_id: str | None
    grinder_context_id: str | None
    profile_id: str | None
    raw_profile_hash: str | None = None
    basket_id: str | None = None
    water_id: str | None = None
    user_id: str | None = None
    temperature_profile_id: str | None = None
    pressure_profile_id: str | None = None
    taste_goal: TasteGoal = field(default_factory=TasteGoal.balanced)

    def __post_init__(self) -> None:
        if not self.install_id.strip() or not self.machine_id.strip():
            raise ValueError("run context requires install_id and machine_id")
        if self.profile_id is not None and self.raw_profile_hash is not None:
            object.__setattr__(self, "raw_profile_hash", None)
        for field_name in (
            "install_id",
            "machine_id",
            "bean_context_id",
            "grinder_context_id",
            "profile_id",
            "raw_profile_hash",
            "basket_id",
            "water_id",
            "user_id",
            "temperature_profile_id",
            "pressure_profile_id",
        ):
            value = getattr(self, field_name)
            if value is not None and len(value) > 256:
                raise ValueError(f"{field_name} is too long")
        if not isinstance(self.taste_goal, TasteGoal):
            object.__setattr__(self, "taste_goal", TasteGoal.from_dict(self.taste_goal))

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "install_id": self.install_id,
            "machine_id": self.machine_id,
            "bean_context_id": self.bean_context_id,
            "grinder_context_id": self.grinder_context_id,
            "profile_id": self.profile_id,
            "raw_profile_hash": self.raw_profile_hash,
            "basket_id": self.basket_id,
            "water_id": self.water_id,
            "user_id": self.user_id,
            "temperature_profile_id": self.temperature_profile_id,
            "pressure_profile_id": self.pressure_profile_id,
            "taste_goal": self.taste_goal.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OptimizationRunContext":
        if not isinstance(value, Mapping):
            raise ValueError("optimization run context must be an object")
        fields = {
            key: value.get(key)
            for key in cls.__dataclass_fields__
            if key != "taste_goal"
        }
        fields["taste_goal"] = TasteGoal.from_dict(value.get("taste_goal"))
        return cls(**fields)


@dataclass(frozen=True)
class OptimizationRun:
    run_id: str
    context: OptimizationRunContext
    comparison_mode: ComparisonMode
    recipe_space: RecipeSpace
    created_at: int
    configuration_version: str = CPBO_CONFIGURATION_VERSION
    active: bool = True

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        object.__setattr__(self, "comparison_mode", ComparisonMode(self.comparison_mode))
        if self.created_at < 0:
            raise ValueError("created_at must be nonnegative")
        if not self.configuration_version.strip():
            raise ValueError("configuration_version is required")


@dataclass(frozen=True)
class ObservedRecipe:
    """Physical recipe observed for a shot, independent of optimizer policy bounds."""

    grind_size: float
    dose_g: float
    target_output_g: float

    def __post_init__(self) -> None:
        values = (float(self.grind_size), float(self.dose_g), float(self.target_output_g))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("observed recipe values must be finite")
        if abs(values[0]) > RECIPE_DOMAIN_GRIND_RADIUS_MAX_STEPS:
            raise ValueError("observed grind is outside the integrity envelope")
        if not RECIPE_DOMAIN_DOSE_MIN_G <= values[1] <= RECIPE_DOMAIN_DOSE_MAX_G:
            raise ValueError("observed dose is outside the integrity envelope")
        if not RECIPE_DOMAIN_OUTPUT_MIN_G <= values[2] <= RECIPE_DOMAIN_OUTPUT_MAX_G:
            raise ValueError("observed output is outside the integrity envelope")
        object.__setattr__(self, "grind_size", values[0])
        object.__setattr__(self, "dose_g", values[1])
        object.__setattr__(self, "target_output_g", values[2])

    @property
    def brew_ratio(self) -> float:
        return self.target_output_g / self.dose_g


@dataclass(frozen=True)
class RecipePoint:
    recipe_id: str
    optimization_run_id: str
    grind_size: float
    dose_g: float
    target_output_g: float
    brew_ratio: float
    normalized_x: tuple[float, float, float]
    created_at: int

    def __post_init__(self) -> None:
        if not self.recipe_id.strip() or not self.optimization_run_id.strip():
            raise ValueError("recipe requires recipe_id and optimization_run_id")
        if self.created_at < 0:
            raise ValueError("recipe created_at must be nonnegative")
        values = (self.grind_size, self.dose_g, self.target_output_g, self.brew_ratio)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("recipe physical values must be finite")
        if self.dose_g <= 0 or self.target_output_g <= 0 or self.brew_ratio <= 0:
            raise ValueError("recipe dose, output, and ratio must be positive")
        if not math.isclose(
            self.brew_ratio,
            self.target_output_g / self.dose_g,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("brew_ratio must be derived from target_output_g / dose_g")
        object.__setattr__(self, "normalized_x", recipe_coordinates(self.normalized_x))

    @property
    def inside_search_space(self) -> bool:
        return all(
            -_FLOAT_TOLERANCE <= value <= 1.0 + _FLOAT_TOLERANCE
            for value in self.normalized_x
        )

    @classmethod
    def create(
        cls,
        run_id: str,
        recipe_space: RecipeSpace,
        grind_size: float,
        dose_g: float,
        target_output_g: float,
        *,
        created_at: int | None = None,
    ) -> "RecipePoint":
        quantized = recipe_space.quantize_recipe(grind_size, dose_g, target_output_g)
        normalized_x = recipe_space.normalize_recipe(*quantized)
        key = json.dumps(
            [run_id, recipe_space.version, *quantized],
            separators=(",", ":"),
        )
        recipe_id = f"recipe_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]}"
        return cls(
            recipe_id=recipe_id,
            optimization_run_id=run_id,
            grind_size=quantized[0],
            dose_g=quantized[1],
            target_output_g=quantized[2],
            brew_ratio=quantized[2] / quantized[1],
            normalized_x=normalized_x,
            created_at=_now() if created_at is None else int(created_at),
        )

    @classmethod
    def observe(
        cls,
        run_id: str,
        recipe_space: RecipeSpace,
        grind_size: float,
        dose_g: float,
        target_output_g: float,
        *,
        created_at: int | None = None,
    ) -> "RecipePoint":
        observation = ObservedRecipe(grind_size, dose_g, target_output_g)
        quantized = recipe_space.quantize_observation(
            observation.grind_size,
            observation.dose_g,
            observation.target_output_g,
        )
        normalized_x = recipe_space.normalize_observation(*quantized)
        key = json.dumps(
            [run_id, recipe_space.version, *quantized],
            separators=(",", ":"),
        )
        recipe_id = f"recipe_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]}"
        return cls(
            recipe_id=recipe_id,
            optimization_run_id=run_id,
            grind_size=quantized[0],
            dose_g=quantized[1],
            target_output_g=quantized[2],
            brew_ratio=quantized[2] / quantized[1],
            normalized_x=normalized_x,
            created_at=_now() if created_at is None else int(created_at),
        )


@dataclass(frozen=True)
class PreferenceShot:
    shot_id: str
    recipe_id: str
    optimization_run_id: str
    sequence_number: int
    started_at: int
    completed_at: int | None
    status: PhysicalShotStatus
    telemetry_available: bool
    observed_recipe: ObservedRecipe | None = None
    raw_telemetry_reference: str | None = None
    trace_feature_names: tuple[str, ...] = ()
    trace_features: tuple[float, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.shot_id.strip() or not self.recipe_id.strip() or not self.optimization_run_id.strip():
            raise ValueError("shot identifiers are required")
        if self.sequence_number < 1:
            raise ValueError("shot sequence_number must be positive")
        if self.started_at < 0 or (self.completed_at is not None and self.completed_at < self.started_at):
            raise ValueError("shot timestamps are invalid")
        object.__setattr__(self, "status", PhysicalShotStatus(self.status))
        if self.observed_recipe is not None and not isinstance(self.observed_recipe, ObservedRecipe):
            raise ValueError("observed_recipe must be an ObservedRecipe")
        if self.status == PhysicalShotStatus.VALID and self.completed_at is None:
            raise ValueError("valid shot requires completed_at")
        if self.trace_features is not None:
            features = tuple(float(value) for value in self.trace_features)
            if len(features) != len(self.trace_feature_names):
                raise ValueError("trace feature names and values must have equal length")
            if not all(math.isfinite(value) for value in features):
                raise ValueError("trace features must be finite")
            object.__setattr__(self, "trace_features", features)
            object.__setattr__(self, "telemetry_available", True)
        elif self.trace_feature_names:
            raise ValueError("trace feature names require trace feature values")
        _validate_metadata(self.metadata)


@dataclass(frozen=True)
class PreferenceComparison:
    comparison_id: str
    optimization_run_id: str
    new_shot_id: str
    anchor_shot_id: str
    label: PreferenceLabel
    comparison_mode: ComparisonMode
    created_at: int
    taste_goal: TasteGoal = field(default_factory=TasteGoal.balanced)

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.comparison_id,
                self.optimization_run_id,
                self.new_shot_id,
                self.anchor_shot_id,
            )
        ):
            raise ValueError("comparison identifiers are required")
        if self.new_shot_id == self.anchor_shot_id:
            raise ValueError("comparison requires two distinct physical shots")
        object.__setattr__(self, "label", PreferenceLabel(self.label))
        object.__setattr__(self, "comparison_mode", ComparisonMode(self.comparison_mode))
        if not isinstance(self.taste_goal, TasteGoal):
            object.__setattr__(self, "taste_goal", TasteGoal.from_dict(self.taste_goal))
        if self.created_at < 0:
            raise ValueError("comparison created_at must be nonnegative")


@dataclass(frozen=True)
class TrustRegionState:
    center: tuple[float, float, float]
    length: float = 0.8
    success_count: int = 0
    failure_count: int = 0
    restart_pending: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", normalized_recipe(self.center))
        if not math.isfinite(self.length) or self.length <= 0:
            raise ValueError("trust-region length must be positive and finite")
        if self.success_count < 0 or self.failure_count < 0:
            raise ValueError("trust-region counters must be nonnegative")
        if self.success_count and self.failure_count:
            raise ValueError("success and failure counters cannot both be nonzero")


@dataclass(frozen=True)
class OptimizerState:
    optimization_run_id: str
    previous_valid_shot_id: str | None
    incumbent_shot_id: str | None
    iteration: int
    trust_region_state: TrustRegionState
    model_checkpoint: str | None
    trace_model_checkpoint: str | None
    random_seed: int
    configuration_version: str
    pending_recipe_id: str | None = None
    pending_anchor_shot_id: str | None = None
    pending_shot_id: str | None = None
    pending_suggestion_json: str | None = None
    updated_at: int = field(default_factory=lambda: int(time.time()))

    def __post_init__(self) -> None:
        if not self.optimization_run_id.strip():
            raise ValueError("optimizer state requires optimization_run_id")
        if self.iteration < 0 or self.random_seed < 0 or self.updated_at < 0:
            raise ValueError("optimizer state counters and timestamps must be nonnegative")
        if not self.configuration_version.strip():
            raise ValueError("optimizer state configuration_version is required")
        if (self.pending_recipe_id is None) != (self.pending_anchor_shot_id is None):
            raise ValueError("pending recipe and anchor must be set together")
        if self.pending_shot_id is not None and self.pending_recipe_id is None:
            raise ValueError("pending shot requires a pending recipe")


@dataclass(frozen=True)
class AcquisitionDiagnostics:
    acquisition_value: float
    unclipped_acquisition_value: float
    outcome_probabilities: Mapping[str, float]
    learned_gamma: float
    kernel_weights: Mapping[str, float]
    raw_kernel_lengthscales: tuple[float, float, float]
    physics_kernel_lengthscales: tuple[float, ...]
    trace_kernel_enabled: bool
    fit_warnings: tuple[str, ...]
    maximum_strategy: str
    truncation_fallback_count: int
    random_seed: int = 0
    trace_kernel_lengthscales: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        numeric = (
            self.acquisition_value,
            self.unclipped_acquisition_value,
            self.learned_gamma,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("acquisition diagnostics must be finite")
        if self.acquisition_value < 0 or self.learned_gamma < 0:
            raise ValueError("acquisition value and gamma must be nonnegative")
        if self.random_seed < 0:
            raise ValueError("acquisition random seed must be nonnegative")
        lengthscales = (
            *self.raw_kernel_lengthscales,
            *self.physics_kernel_lengthscales,
            *self.trace_kernel_lengthscales,
        )
        if any(not math.isfinite(float(value)) or value <= 0 for value in lengthscales):
            raise ValueError("kernel lengthscales must be positive and finite")
        if len(self.raw_kernel_lengthscales) != 3 or not self.physics_kernel_lengthscales:
            raise ValueError("kernel diagnostics have invalid dimensions")
        if self.trace_kernel_enabled != bool(self.trace_kernel_lengthscales):
            raise ValueError("trace-kernel diagnostics disagree with activation state")
        weights = {str(key): float(value) for key, value in self.kernel_weights.items()}
        if set(weights) != {"raw", "physics", "trace"}:
            raise ValueError("kernel diagnostics require raw, physics, and trace weights")
        if any(not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("kernel weights must be nonnegative and finite")
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-8):
            raise ValueError("kernel weights must sum to one")
        if not self.trace_kernel_enabled and weights["trace"] != 0.0:
            raise ValueError("disabled trace kernel must have exactly zero weight")
        object.__setattr__(self, "kernel_weights", weights)
        if self.maximum_strategy not in {"paper_gumbel", "direct_max_samples"}:
            raise ValueError("maximum strategy is invalid")
        if self.truncation_fallback_count < 0:
            raise ValueError("truncation fallback count must be nonnegative")
        probabilities = {str(key): float(value) for key, value in self.outcome_probabilities.items()}
        expected = {label.value for label in PreferenceLabel}
        if set(probabilities) != expected:
            raise ValueError("diagnostics must contain all three preference probabilities")
        if any(not 0.0 <= value <= 1.0 for value in probabilities.values()):
            raise ValueError("preference probabilities must be within [0, 1]")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
            raise ValueError("preference probabilities must sum to one")
        object.__setattr__(self, "outcome_probabilities", probabilities)


@dataclass(frozen=True)
class TrustRegionDiagnostics:
    length: float
    lower_bounds: tuple[float, float, float]
    upper_bounds: tuple[float, float, float]
    success_count: int
    failure_count: int
    restart_pending: bool
    full_domain_proposal: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower_bounds", normalized_recipe(self.lower_bounds))
        object.__setattr__(self, "upper_bounds", normalized_recipe(self.upper_bounds))
        if any(low > high for low, high in zip(self.lower_bounds, self.upper_bounds)):
            raise ValueError("trust-region lower bounds exceed upper bounds")


@dataclass(frozen=True)
class Suggestion:
    suggestion_id: str
    optimization_run_id: str
    recipe: RecipePoint
    anchor_shot_id: str
    comparison_mode: ComparisonMode
    acquisition: AcquisitionDiagnostics
    trust_region: TrustRegionDiagnostics
    model_version: str
    iteration: int
    created_at: int

    def __post_init__(self) -> None:
        if self.recipe.optimization_run_id != self.optimization_run_id:
            raise ValueError("suggestion recipe belongs to another optimization run")
        if not self.recipe.inside_search_space:
            raise ValueError("suggestion recipe must be inside the search space")
        if not self.anchor_shot_id.strip():
            raise ValueError("suggestion anchor_shot_id is required")
        object.__setattr__(self, "comparison_mode", ComparisonMode(self.comparison_mode))
        if self.iteration < 1:
            raise ValueError("suggestion iteration must be positive")


@dataclass(frozen=True)
class SuggestionComputation:
    suggestion: Suggestion
    model_checkpoint: str
    trace_model_checkpoint: str | None


@dataclass(frozen=True)
class ShotRequest:
    optimization_run_id: str
    recipe: RecipePoint
    anchor_shot_id: str | None
    comparison_mode: ComparisonMode
    is_baseline: bool
    model_version: str = CPBO_MODEL_VERSION

    def __post_init__(self) -> None:
        if self.recipe.optimization_run_id != self.optimization_run_id:
            raise ValueError("shot request recipe belongs to another optimization run")
        if not self.recipe.inside_search_space:
            raise ValueError("shot request recipe must be inside the search space")
        object.__setattr__(self, "comparison_mode", ComparisonMode(self.comparison_mode))
        if self.is_baseline != (self.anchor_shot_id is None):
            raise ValueError("only a baseline shot request may omit its anchor")


@dataclass(frozen=True)
class ModelRecommendation:
    optimization_run_id: str
    recipe: RecipePoint
    source: str
    directly_established: bool
    incumbent_shot_id: str | None
    model_version: str = CPBO_MODEL_VERSION

    def __post_init__(self) -> None:
        if self.recipe.optimization_run_id != self.optimization_run_id:
            raise ValueError("recommended recipe belongs to another optimization run")
        if not self.recipe.inside_search_space:
            raise ValueError("recommended recipe must be inside the search space")
        if not self.source.strip():
            raise ValueError("recommendation source is required")


def new_cpbo_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def normalized_recipe(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != _NORMALIZED_DIMENSION:
        raise ValueError("normalized recipe must have exactly three dimensions")
    return tuple(_normalized_scalar(item, "normalized recipe") for item in value)  # type: ignore[return-value]


def recipe_coordinates(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != _NORMALIZED_DIMENSION:
        raise ValueError("recipe coordinates must have exactly three dimensions")
    coordinates = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in coordinates):
        raise ValueError("recipe coordinates must be finite")
    return coordinates  # type: ignore[return-value]


def _normalized_scalar(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < -_FLOAT_TOLERANCE or value > 1.0 + _FLOAT_TOLERANCE:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return min(1.0, max(0.0, value))


def _parameter_to_dict(parameter: RecipeParameter) -> dict[str, Any]:
    return {
        "name": parameter.name,
        "physical_min": parameter.physical_min,
        "physical_max": parameter.physical_max,
        "resolution": parameter.resolution,
        "unit": parameter.unit,
        "constraints": list(parameter.constraints),
    }


def _parameter_from_dict(value: Any) -> RecipeParameter:
    if not isinstance(value, Mapping):
        raise ValueError("recipe parameter must be an object")
    return RecipeParameter(
        name=str(value.get("name") or ""),
        physical_min=float(value.get("physical_min")),
        physical_max=float(value.get("physical_max")),
        resolution=float(value.get("resolution")),
        unit=str(value.get("unit") or ""),
        constraints=tuple(
            str(item)
            for item in (value.get("constraints") or value.get("safety_constraints") or ())
        ),
    )


def _decimal_places(value: float) -> int:
    text = f"{value:.12f}".rstrip("0")
    return len(text.partition(".")[2])


def _validate_metadata(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("shot metadata must be an object")
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("shot metadata must be finite JSON data") from exc
    if len(encoded.encode("utf-8")) > 64 * 1024:
        raise ValueError("shot metadata exceeds 64 KiB")


def _now() -> int:
    return int(time.time())

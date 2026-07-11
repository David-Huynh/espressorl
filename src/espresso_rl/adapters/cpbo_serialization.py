from __future__ import annotations

import json
from typing import Any, Mapping

from espresso_rl.domain.cpbo import (
    AcquisitionDiagnostics,
    ComparisonMode,
    OptimizationRun,
    OptimizationRunContext,
    OptimizerState,
    PhysicalShotStatus,
    PreferenceComparison,
    PreferenceLabel,
    PreferenceShot,
    RecipePoint,
    RecipeSpace,
    Suggestion,
    TrustRegionDiagnostics,
    TrustRegionState,
)


MAX_CPBO_JSON_BYTES = 8 * 1024 * 1024


def encode_cpbo_json(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("CPBO persistence payload must be finite JSON") from exc
    if len(encoded.encode("utf-8")) > MAX_CPBO_JSON_BYTES:
        raise ValueError("CPBO persistence payload exceeds 8 MiB")
    return encoded


def decode_cpbo_json(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, str) or not value:
        raise ValueError("CPBO persistence payload must be nonempty JSON text")
    if len(value.encode("utf-8")) > MAX_CPBO_JSON_BYTES:
        raise ValueError("CPBO persistence payload exceeds 8 MiB")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("CPBO persistence payload is invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("CPBO persistence payload must be an object")
    return decoded


def run_to_json(run: OptimizationRun) -> str:
    return encode_cpbo_json(
        {
            "run_id": run.run_id,
            "context": run.context.to_dict(),
            "comparison_mode": run.comparison_mode.value,
            "recipe_space": run.recipe_space.to_dict(),
            "created_at": run.created_at,
            "configuration_version": run.configuration_version,
            "active": run.active,
        }
    )


def run_from_json(value: Any) -> OptimizationRun:
    row = decode_cpbo_json(value)
    _require_keys(
        row,
        {
            "run_id",
            "context",
            "comparison_mode",
            "recipe_space",
            "created_at",
            "configuration_version",
            "active",
        },
        "optimization run",
    )
    return OptimizationRun(
        run_id=str(row["run_id"]),
        context=OptimizationRunContext.from_dict(row["context"]),
        comparison_mode=ComparisonMode(row["comparison_mode"]),
        recipe_space=RecipeSpace.from_dict(row["recipe_space"]),
        created_at=int(row["created_at"]),
        configuration_version=str(row["configuration_version"]),
        active=_strict_bool(row["active"], "active"),
    )


def recipe_to_json(recipe: RecipePoint) -> str:
    return encode_cpbo_json(
        {
            "recipe_id": recipe.recipe_id,
            "optimization_run_id": recipe.optimization_run_id,
            "grind_size": recipe.grind_size,
            "dose_g": recipe.dose_g,
            "target_output_g": recipe.target_output_g,
            "brew_ratio": recipe.brew_ratio,
            "normalized_x": list(recipe.normalized_x),
            "created_at": recipe.created_at,
        }
    )


def recipe_from_json(value: Any) -> RecipePoint:
    row = decode_cpbo_json(value)
    _require_keys(
        row,
        {
            "recipe_id",
            "optimization_run_id",
            "grind_size",
            "dose_g",
            "target_output_g",
            "brew_ratio",
            "normalized_x",
            "created_at",
        },
        "recipe",
    )
    return RecipePoint(
        recipe_id=str(row["recipe_id"]),
        optimization_run_id=str(row["optimization_run_id"]),
        grind_size=float(row["grind_size"]),
        dose_g=float(row["dose_g"]),
        target_output_g=float(row["target_output_g"]),
        brew_ratio=float(row["brew_ratio"]),
        normalized_x=tuple(row["normalized_x"]),
        created_at=int(row["created_at"]),
    )


def shot_to_json(shot: PreferenceShot) -> str:
    return encode_cpbo_json(
        {
            "shot_id": shot.shot_id,
            "recipe_id": shot.recipe_id,
            "optimization_run_id": shot.optimization_run_id,
            "sequence_number": shot.sequence_number,
            "started_at": shot.started_at,
            "completed_at": shot.completed_at,
            "status": shot.status.value,
            "telemetry_available": shot.telemetry_available,
            "raw_telemetry_reference": shot.raw_telemetry_reference,
            "trace_feature_names": list(shot.trace_feature_names),
            "trace_features": list(shot.trace_features) if shot.trace_features is not None else None,
            "metadata": dict(shot.metadata),
        }
    )


def shot_from_json(value: Any) -> PreferenceShot:
    row = decode_cpbo_json(value)
    _require_keys(
        row,
        {
            "shot_id",
            "recipe_id",
            "optimization_run_id",
            "sequence_number",
            "started_at",
            "completed_at",
            "status",
            "telemetry_available",
            "raw_telemetry_reference",
            "trace_feature_names",
            "trace_features",
            "metadata",
        },
        "physical shot",
    )
    return PreferenceShot(
        shot_id=str(row["shot_id"]),
        recipe_id=str(row["recipe_id"]),
        optimization_run_id=str(row["optimization_run_id"]),
        sequence_number=int(row["sequence_number"]),
        started_at=int(row["started_at"]),
        completed_at=(int(row["completed_at"]) if row["completed_at"] is not None else None),
        status=PhysicalShotStatus(row["status"]),
        telemetry_available=_strict_bool(row["telemetry_available"], "telemetry_available"),
        raw_telemetry_reference=_optional_string(row["raw_telemetry_reference"]),
        trace_feature_names=tuple(str(item) for item in row["trace_feature_names"]),
        trace_features=(
            tuple(float(item) for item in row["trace_features"])
            if row["trace_features"] is not None
            else None
        ),
        metadata=row["metadata"],
    )


def comparison_to_json(comparison: PreferenceComparison) -> str:
    return encode_cpbo_json(
        {
            "comparison_id": comparison.comparison_id,
            "optimization_run_id": comparison.optimization_run_id,
            "new_shot_id": comparison.new_shot_id,
            "anchor_shot_id": comparison.anchor_shot_id,
            "label": comparison.label.value,
            "comparison_mode": comparison.comparison_mode.value,
            "created_at": comparison.created_at,
        }
    )


def comparison_from_json(value: Any) -> PreferenceComparison:
    row = decode_cpbo_json(value)
    _require_keys(
        row,
        {
            "comparison_id",
            "optimization_run_id",
            "new_shot_id",
            "anchor_shot_id",
            "label",
            "comparison_mode",
            "created_at",
        },
        "preference comparison",
    )
    return PreferenceComparison(
        comparison_id=str(row["comparison_id"]),
        optimization_run_id=str(row["optimization_run_id"]),
        new_shot_id=str(row["new_shot_id"]),
        anchor_shot_id=str(row["anchor_shot_id"]),
        label=PreferenceLabel(row["label"]),
        comparison_mode=ComparisonMode(row["comparison_mode"]),
        created_at=int(row["created_at"]),
    )


def state_to_json(state: OptimizerState) -> str:
    trust = state.trust_region_state
    return encode_cpbo_json(
        {
            "optimization_run_id": state.optimization_run_id,
            "previous_valid_shot_id": state.previous_valid_shot_id,
            "incumbent_shot_id": state.incumbent_shot_id,
            "iteration": state.iteration,
            "trust_region_state": {
                "center": list(trust.center),
                "length": trust.length,
                "success_count": trust.success_count,
                "failure_count": trust.failure_count,
                "restart_pending": trust.restart_pending,
            },
            "model_checkpoint": state.model_checkpoint,
            "trace_model_checkpoint": state.trace_model_checkpoint,
            "random_seed": state.random_seed,
            "configuration_version": state.configuration_version,
            "pending_recipe_id": state.pending_recipe_id,
            "pending_anchor_shot_id": state.pending_anchor_shot_id,
            "pending_shot_id": state.pending_shot_id,
            "pending_suggestion_json": state.pending_suggestion_json,
            "updated_at": state.updated_at,
        }
    )


def state_from_json(value: Any) -> OptimizerState:
    row = decode_cpbo_json(value)
    expected = {
        "optimization_run_id",
        "previous_valid_shot_id",
        "incumbent_shot_id",
        "iteration",
        "trust_region_state",
        "model_checkpoint",
        "trace_model_checkpoint",
        "random_seed",
        "configuration_version",
        "pending_recipe_id",
        "pending_anchor_shot_id",
        "pending_shot_id",
        "pending_suggestion_json",
        "updated_at",
    }
    _require_keys(row, expected, "optimizer state")
    trust = row["trust_region_state"]
    if not isinstance(trust, Mapping):
        raise ValueError("trust_region_state must be an object")
    _require_keys(
        trust,
        {"center", "length", "success_count", "failure_count", "restart_pending"},
        "trust-region state",
    )
    return OptimizerState(
        optimization_run_id=str(row["optimization_run_id"]),
        previous_valid_shot_id=_optional_string(row["previous_valid_shot_id"]),
        incumbent_shot_id=_optional_string(row["incumbent_shot_id"]),
        iteration=int(row["iteration"]),
        trust_region_state=TrustRegionState(
            center=tuple(trust["center"]),
            length=float(trust["length"]),
            success_count=int(trust["success_count"]),
            failure_count=int(trust["failure_count"]),
            restart_pending=_strict_bool(trust["restart_pending"], "restart_pending"),
        ),
        model_checkpoint=_optional_string(row["model_checkpoint"]),
        trace_model_checkpoint=_optional_string(row["trace_model_checkpoint"]),
        random_seed=int(row["random_seed"]),
        configuration_version=str(row["configuration_version"]),
        pending_recipe_id=_optional_string(row["pending_recipe_id"]),
        pending_anchor_shot_id=_optional_string(row["pending_anchor_shot_id"]),
        pending_shot_id=_optional_string(row["pending_shot_id"]),
        pending_suggestion_json=_optional_string(row["pending_suggestion_json"]),
        updated_at=int(row["updated_at"]),
    )


def suggestion_to_json(suggestion: Suggestion) -> str:
    acquisition = suggestion.acquisition
    trust = suggestion.trust_region
    return encode_cpbo_json(
        {
            "suggestion_id": suggestion.suggestion_id,
            "optimization_run_id": suggestion.optimization_run_id,
            "recipe": decode_cpbo_json(recipe_to_json(suggestion.recipe)),
            "anchor_shot_id": suggestion.anchor_shot_id,
            "comparison_mode": suggestion.comparison_mode.value,
            "acquisition": {
                "acquisition_value": acquisition.acquisition_value,
                "unclipped_acquisition_value": acquisition.unclipped_acquisition_value,
                "outcome_probabilities": dict(acquisition.outcome_probabilities),
                "learned_gamma": acquisition.learned_gamma,
                "kernel_weights": dict(acquisition.kernel_weights),
                "raw_kernel_lengthscales": list(acquisition.raw_kernel_lengthscales),
                "physics_kernel_lengthscales": list(acquisition.physics_kernel_lengthscales),
                "trace_kernel_enabled": acquisition.trace_kernel_enabled,
                "fit_warnings": list(acquisition.fit_warnings),
                "maximum_strategy": acquisition.maximum_strategy,
                "truncation_fallback_count": acquisition.truncation_fallback_count,
                "random_seed": acquisition.random_seed,
                "trace_kernel_lengthscales": list(acquisition.trace_kernel_lengthscales),
            },
            "trust_region": {
                "length": trust.length,
                "lower_bounds": list(trust.lower_bounds),
                "upper_bounds": list(trust.upper_bounds),
                "success_count": trust.success_count,
                "failure_count": trust.failure_count,
                "restart_pending": trust.restart_pending,
                "full_domain_proposal": trust.full_domain_proposal,
            },
            "model_version": suggestion.model_version,
            "iteration": suggestion.iteration,
            "created_at": suggestion.created_at,
        }
    )


def suggestion_from_json(value: Any) -> Suggestion:
    row = decode_cpbo_json(value)
    _require_keys(
        row,
        {
            "suggestion_id",
            "optimization_run_id",
            "recipe",
            "anchor_shot_id",
            "comparison_mode",
            "acquisition",
            "trust_region",
            "model_version",
            "iteration",
            "created_at",
        },
        "suggestion",
    )
    acquisition = row["acquisition"]
    trust = row["trust_region"]
    if not isinstance(acquisition, Mapping) or not isinstance(trust, Mapping):
        raise ValueError("suggestion diagnostics must be objects")
    _require_keys(
        acquisition,
        {
            "acquisition_value",
            "unclipped_acquisition_value",
            "outcome_probabilities",
            "learned_gamma",
            "kernel_weights",
            "raw_kernel_lengthscales",
            "physics_kernel_lengthscales",
            "trace_kernel_enabled",
            "trace_kernel_lengthscales",
            "fit_warnings",
            "maximum_strategy",
            "truncation_fallback_count",
            "random_seed",
        },
        "acquisition diagnostics",
    )
    _require_keys(
        trust,
        {
            "length",
            "lower_bounds",
            "upper_bounds",
            "success_count",
            "failure_count",
            "restart_pending",
            "full_domain_proposal",
        },
        "trust-region diagnostics",
    )
    return Suggestion(
        suggestion_id=str(row["suggestion_id"]),
        optimization_run_id=str(row["optimization_run_id"]),
        recipe=recipe_from_json(encode_cpbo_json(row["recipe"])),
        anchor_shot_id=str(row["anchor_shot_id"]),
        comparison_mode=ComparisonMode(row["comparison_mode"]),
        acquisition=AcquisitionDiagnostics(
            acquisition_value=float(acquisition["acquisition_value"]),
            unclipped_acquisition_value=float(acquisition["unclipped_acquisition_value"]),
            outcome_probabilities=acquisition["outcome_probabilities"],
            learned_gamma=float(acquisition["learned_gamma"]),
            kernel_weights=acquisition["kernel_weights"],
            raw_kernel_lengthscales=tuple(acquisition["raw_kernel_lengthscales"]),
            physics_kernel_lengthscales=tuple(acquisition["physics_kernel_lengthscales"]),
            trace_kernel_enabled=_strict_bool(acquisition["trace_kernel_enabled"], "trace_kernel_enabled"),
            fit_warnings=tuple(str(item) for item in acquisition["fit_warnings"]),
            maximum_strategy=str(acquisition["maximum_strategy"]),
            truncation_fallback_count=int(acquisition["truncation_fallback_count"]),
            random_seed=int(acquisition["random_seed"]),
            trace_kernel_lengthscales=tuple(acquisition["trace_kernel_lengthscales"]),
        ),
        trust_region=TrustRegionDiagnostics(
            length=float(trust["length"]),
            lower_bounds=tuple(trust["lower_bounds"]),
            upper_bounds=tuple(trust["upper_bounds"]),
            success_count=int(trust["success_count"]),
            failure_count=int(trust["failure_count"]),
            restart_pending=_strict_bool(trust["restart_pending"], "restart_pending"),
            full_domain_proposal=_strict_bool(trust["full_domain_proposal"], "full_domain_proposal"),
        ),
        model_version=str(row["model_version"]),
        iteration=int(row["iteration"]),
        created_at=int(row["created_at"]),
    )


def _require_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing={','.join(missing)}")
        if unknown:
            details.append(f"unknown={','.join(unknown)}")
        raise ValueError(f"{name} fields are invalid ({'; '.join(details)})")


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional identifier must be text or null")
    return value or None

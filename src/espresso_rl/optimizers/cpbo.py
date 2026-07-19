from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from espresso_rl.domain.cpbo import (
    CPBO_MODEL_VERSION,
    AcquisitionDiagnostics,
    ComparisonMode,
    LocalOptimizationConvergedError,
    ModelRecommendation,
    OptimizationRun,
    OptimizerState,
    PhysicalShotStatus,
    PreferenceComparison,
    PreferenceLabel,
    PreferenceShot,
    RecipePoint,
    Suggestion,
    SuggestionComputation,
    TrustRegionDiagnostics,
    new_cpbo_id,
)
from espresso_rl.optimizers.cpbo_candidates import build_candidate_domain
from espresso_rl.optimizers.cpbo_config import CPBOConfig
from espresso_rl.optimizers.cpbo_jnd import preference_label_indices
from espresso_rl.optimizers.cpbo_mes import evaluate_cpbo_mes
from espresso_rl.optimizers.cpbo_model import fit_preference_gp, posterior_at
from espresso_rl.optimizers.cpbo_physics import (
    PHYSICS_FEATURE_NAMES,
    RobustFeatureScaler,
    phi0,
)
from espresso_rl.optimizers.cpbo_trace import (
    TRACE_FEATURE_NAMES,
    IndependentTraceSurrogate,
)
from espresso_rl.optimizers.cpbo_trust_region import (
    resume_trust_region,
    trust_region_bounds,
    update_trust_region,
    validate_q_one,
)


class ConsecutivePreferentialBayesianOptimizer:
    """Machine-agnostic CPBO with fixed-anchor, three-outcome MES."""

    def __init__(self, config: CPBOConfig) -> None:
        self.config = config
        validate_q_one()

    def suggest(
        self,
        *,
        run: OptimizationRun,
        recipes: Sequence[RecipePoint],
        shots: Sequence[PreferenceShot],
        comparisons: Sequence[PreferenceComparison],
        state: OptimizerState,
        now: int,
    ) -> SuggestionComputation:
        self._validate_configuration_version(run, state)
        if (
            run.comparison_mode == ComparisonMode.BEST_INCUMBENT
            and state.trust_region_state.locally_converged
        ):
            raise LocalOptimizationConvergedError(
                "local CPBO converged; resume exploration before requesting another suggestion"
            )
        valid_shots, recipe_by_id, shot_by_id = _validate_run_data(
            run,
            recipes,
            shots,
            comparisons,
            state,
        )
        if state.pending_recipe_id is not None:
            raise ValueError("an unresolved CPBO suggestion already exists")
        anchor_shot_id = (
            state.previous_valid_shot_id
            if run.comparison_mode == ComparisonMode.GLOBAL_PREVIOUS
            else state.incumbent_shot_id
        )
        if anchor_shot_id is None:
            raise ValueError("CPBO requires a valid baseline shot before suggesting")
        anchor_shot = shot_by_id[anchor_shot_id]
        anchor_recipe = recipe_by_id[anchor_shot.recipe_id]

        training_recipes = _unique_valid_recipes(valid_shots, recipe_by_id)
        physics_scaler, physics_warnings = _fit_physics_scaler(
            run,
            training_recipes,
            self.config,
        )
        trace_surrogate, trace_warnings = _fit_trace_surrogate(
            valid_shots,
            recipe_by_id,
            self.config,
            warm_start_checkpoint=state.trace_model_checkpoint,
        )
        encoder = _FeatureEncoder(
            physics_scaler=physics_scaler,
            trace_surrogate=trace_surrogate,
            config=self.config,
        )
        train_inputs = encoder.encode(training_recipes)
        recipe_index = {recipe.recipe_id: index for index, recipe in enumerate(training_recipes)}
        comparison_indices = torch.tensor(
            [
                [
                    recipe_index[shot_by_id[row.new_shot_id].recipe_id],
                    recipe_index[shot_by_id[row.anchor_shot_id].recipe_id],
                ]
                for row in comparisons
            ],
            dtype=torch.long,
        ).reshape((-1, 2))
        labels = preference_label_indices([row.label for row in comparisons])
        fit = fit_preference_gp(
            train_inputs=train_inputs,
            comparison_indices=comparison_indices,
            labels=labels,
            physics_dimensions=len(PHYSICS_FEATURE_NAMES),
            trace_dimensions=(len(TRACE_FEATURE_NAMES) if trace_surrogate.enabled else 0),
            config=self.config.model,
            warm_start_checkpoint=state.model_checkpoint,
            random_seed=state.random_seed + state.iteration,
        )
        raw_lengthscales = (
            fit.model.covar_module.raw_kernel.lengthscale.detach().reshape(-1).to(torch.float64)
        )
        full_domain = run.comparison_mode == ComparisonMode.GLOBAL_PREVIOUS
        if full_domain:
            lower_bounds = (0.0, 0.0, 0.0)
            upper_bounds = (1.0, 1.0, 1.0)
        else:
            lower_bounds, upper_bounds = trust_region_bounds(
                state.trust_region_state,
                raw_lengthscales,
                self.config.trust_region,
            )
        candidate_domain = build_candidate_domain(
            run_id=run.run_id,
            recipe_space=run.recipe_space,
            evaluated_recipes=training_recipes,
            anchor_recipe=anchor_recipe,
            config=self.config.acquisition,
            seed=state.random_seed + state.iteration * 31,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            created_at=now,
        )
        encoded_domain = encoder.encode(candidate_domain.discretization_recipes)
        posterior = posterior_at(
            fit.model,
            encoded_domain,
            jitter=self.config.model.covariance_jitter,
        )
        mes = evaluate_cpbo_mes(
            posterior_mean=posterior.mean,
            posterior_covariance=posterior.covariance_matrix,
            candidate_indices=candidate_domain.proposal_indices,
            maximum_indices=candidate_domain.maximum_indices,
            anchor_index=candidate_domain.anchor_index,
            gamma=float(fit.likelihood.gamma.detach()),
            sigma_pref=self.config.model.sigma_pref,
            config=self.config.acquisition,
            seed=state.random_seed + state.iteration * 101,
            covariance_jitter=self.config.model.covariance_jitter,
        )
        selected_recipe = candidate_domain.discretization_recipes[mes.candidate_index]
        fit_warnings = tuple(
            sorted(set((*fit.warnings, *physics_warnings, *trace_warnings)))
        )
        acquisition = AcquisitionDiagnostics(
            acquisition_value=mes.acquisition_value,
            unclipped_acquisition_value=mes.unclipped_acquisition_value,
            outcome_probabilities=mes.outcome_probabilities,
            learned_gamma=float(fit.likelihood.gamma.detach()),
            kernel_weights=fit.model.covar_module.weights_dict(),
            raw_kernel_lengthscales=tuple(float(value) for value in raw_lengthscales),
            physics_kernel_lengthscales=tuple(
                float(value)
                for value in fit.model.covar_module.physics_kernel.lengthscale.detach().reshape(-1)
            ),
            trace_kernel_enabled=trace_surrogate.enabled,
            fit_warnings=fit_warnings,
            maximum_strategy=mes.maximum_distribution.strategy,
            truncation_fallback_count=mes.truncation_fallback_count,
            random_seed=state.random_seed + state.iteration * 101,
            trace_kernel_lengthscales=tuple(
                float(value)
                for value in fit.model.covar_module.trace_lengthscales.detach().reshape(-1)
            ),
        )
        trust_diagnostics = TrustRegionDiagnostics(
            length=state.trust_region_state.length,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            success_count=state.trust_region_state.success_count,
            failure_count=state.trust_region_state.failure_count,
            restart_pending=state.trust_region_state.restart_pending,
            full_domain_proposal=full_domain,
            locally_converged=state.trust_region_state.locally_converged,
            last_transition_action=(
                state.trust_region_state.transitions[-1].action
                if state.trust_region_state.transitions
                else None
            ),
        )
        suggestion = Suggestion(
            suggestion_id=new_cpbo_id("suggestion"),
            optimization_run_id=run.run_id,
            recipe=selected_recipe,
            anchor_shot_id=anchor_shot_id,
            comparison_mode=run.comparison_mode,
            acquisition=acquisition,
            trust_region=trust_diagnostics,
            model_version=CPBO_MODEL_VERSION,
            iteration=state.iteration + 1,
            created_at=now,
        )
        checkpoint = fit.checkpoint_json
        if len(checkpoint.encode("utf-8")) > self.config.checkpoint_max_bytes:
            raise ValueError("CPBO model checkpoint exceeds the configured size limit")
        trace_checkpoint = trace_surrogate.checkpoint_json()
        if trace_checkpoint is not None and len(trace_checkpoint.encode("utf-8")) > self.config.checkpoint_max_bytes:
            raise ValueError("CPBO trace checkpoint exceeds the configured size limit")
        return SuggestionComputation(
            suggestion=suggestion,
            model_checkpoint=checkpoint,
            trace_model_checkpoint=trace_checkpoint,
        )

    def recommend_evaluated(
        self,
        *,
        run: OptimizationRun,
        recipes: Sequence[RecipePoint],
        shots: Sequence[PreferenceShot],
        comparisons: Sequence[PreferenceComparison],
        state: OptimizerState,
    ) -> ModelRecommendation:
        self._validate_configuration_version(run, state)
        valid_shots, recipe_by_id, shot_by_id = _validate_run_data(
            run,
            recipes,
            shots,
            comparisons,
            state,
        )
        if run.comparison_mode == ComparisonMode.BEST_INCUMBENT:
            if state.incumbent_shot_id is None:
                raise ValueError("best-incumbent run has no incumbent")
            incumbent = shot_by_id[state.incumbent_shot_id]
            incumbent_recipe = recipe_by_id[incumbent.recipe_id]
            if incumbent_recipe.inside_search_space:
                return ModelRecommendation(
                    optimization_run_id=run.run_id,
                    recipe=incumbent_recipe,
                    source="direct_incumbent",
                    directly_established=True,
                    incumbent_shot_id=incumbent.shot_id,
                )

        training_recipes = _unique_valid_recipes(valid_shots, recipe_by_id)
        physics_scaler, _ = _fit_physics_scaler(run, training_recipes, self.config)
        trace_surrogate, _ = _fit_trace_surrogate(
            valid_shots,
            recipe_by_id,
            self.config,
            warm_start_checkpoint=state.trace_model_checkpoint,
        )
        encoder = _FeatureEncoder(physics_scaler, trace_surrogate, self.config)
        train_inputs = encoder.encode(training_recipes)
        recipe_index = {recipe.recipe_id: index for index, recipe in enumerate(training_recipes)}
        comparison_indices = torch.tensor(
            [
                [
                    recipe_index[shot_by_id[row.new_shot_id].recipe_id],
                    recipe_index[shot_by_id[row.anchor_shot_id].recipe_id],
                ]
                for row in comparisons
            ],
            dtype=torch.long,
        ).reshape((-1, 2))
        fit = fit_preference_gp(
            train_inputs=train_inputs,
            comparison_indices=comparison_indices,
            labels=preference_label_indices([row.label for row in comparisons]),
            physics_dimensions=len(PHYSICS_FEATURE_NAMES),
            trace_dimensions=(len(TRACE_FEATURE_NAMES) if trace_surrogate.enabled else 0),
            config=self.config.model,
            warm_start_checkpoint=state.model_checkpoint,
            random_seed=state.random_seed + state.iteration,
        )
        posterior = posterior_at(
            fit.model,
            train_inputs,
            jitter=self.config.model.covariance_jitter,
        )
        feasible_indices = [
            index for index, recipe in enumerate(training_recipes) if recipe.inside_search_space
        ]
        if not feasible_indices:
            raise ValueError("optimization run has no evaluated recipe inside the search space")
        best_index = max(feasible_indices, key=lambda index: float(posterior.mean[index]))
        return ModelRecommendation(
            optimization_run_id=run.run_id,
            recipe=training_recipes[best_index],
            source="maximum_posterior_mean_evaluated_recipe",
            directly_established=False,
            incumbent_shot_id=state.incumbent_shot_id,
        )

    def update_trust_region_state(
        self,
        state,
        label: PreferenceLabel,
        *,
        candidate_center: tuple[float, float, float],
    ):
        return update_trust_region(
            state,
            label,
            candidate_center=candidate_center,
            config=self.config.trust_region,
        )

    def resume_trust_region_state(
        self,
        state,
        *,
        center,
        after_comparison_id,
        incumbent_shot_id,
        created_at,
        control_event_id=None,
    ):
        return resume_trust_region(
            state,
            center=center,
            config=self.config.trust_region,
            after_comparison_id=after_comparison_id,
            incumbent_shot_id=incumbent_shot_id,
            created_at=created_at,
            control_event_id=control_event_id,
        )

    def _validate_configuration_version(
        self,
        run: OptimizationRun,
        state: OptimizerState,
    ) -> None:
        expected = self.config.effective_configuration_version
        if run.configuration_version != expected or state.configuration_version != expected:
            raise ValueError("CPBO run configuration version differs from the active engine")


class _FeatureEncoder:
    def __init__(
        self,
        physics_scaler: RobustFeatureScaler,
        trace_surrogate: IndependentTraceSurrogate,
        config: CPBOConfig,
    ) -> None:
        self.physics_scaler = physics_scaler
        self.trace_surrogate = trace_surrogate
        self.config = config

    def encode(self, recipes: Sequence[RecipePoint]) -> Tensor:
        raw = torch.tensor([recipe.normalized_x for recipe in recipes], dtype=torch.float64)
        physics_rows = [phi0(recipe, self.config.physics).values for recipe in recipes]
        physics = self.physics_scaler.transform(physics_rows)
        components = [raw, physics]
        trace_prediction = self.trace_surrogate.predict(raw)
        if trace_prediction.enabled:
            components.extend((trace_prediction.mean, trace_prediction.variance))
        encoded = torch.cat(components, dim=-1)
        if torch.any(~torch.isfinite(encoded)):
            raise FloatingPointError("CPBO feature encoding is non-finite")
        return encoded


def _fit_physics_scaler(
    run: OptimizationRun,
    training_recipes: Sequence[RecipePoint],
    config: CPBOConfig,
) -> tuple[RobustFeatureScaler, tuple[str, ...]]:
    reference_recipes = list(training_recipes)
    for fineness in (0.0, 0.25, 0.5, 0.75, 1.0):
        for dose in (0.0, 0.25, 0.5, 0.75, 1.0):
            for output in (0.0, 0.25, 0.5, 0.75, 1.0):
                try:
                    physical = run.recipe_space.inverse_recipe((fineness, dose, output), quantize=True)
                    reference_recipes.append(
                        RecipePoint.create(run.run_id, run.recipe_space, *physical, created_at=run.created_at)
                    )
                except ValueError:
                    continue
    unique = {recipe.recipe_id: recipe for recipe in reference_recipes}
    results = [phi0(recipe, config.physics) for recipe in unique.values()]
    scaler = RobustFeatureScaler.fit(
        [result.values for result in results],
        scale_floor=config.physics.robust_scale_floor,
    )
    warnings = tuple(sorted({warning for result in results for warning in result.diagnostics}))
    return scaler, warnings


def _fit_trace_surrogate(
    valid_shots: Sequence[PreferenceShot],
    recipe_by_id: dict[str, RecipePoint],
    config: CPBOConfig,
    *,
    warm_start_checkpoint: str | None,
) -> tuple[IndependentTraceSurrogate, tuple[str, ...]]:
    surrogate = IndependentTraceSurrogate(config.trace)
    eligible = [
        shot
        for shot in valid_shots
        if shot.trace_features is not None and tuple(shot.trace_feature_names) == TRACE_FEATURE_NAMES
    ]
    if len(eligible) < config.trace.minimum_valid_telemetry_shots:
        surrogate.warnings = ("trace_kernel_waiting_for_minimum_telemetry_shots",)
        return surrogate, surrogate.warnings
    train_x = torch.tensor(
        [recipe_by_id[shot.recipe_id].normalized_x for shot in eligible],
        dtype=torch.float64,
    )
    trace_rows = torch.tensor([shot.trace_features for shot in eligible], dtype=torch.float64)
    surrogate.fit(train_x, trace_rows, warm_start_checkpoint=warm_start_checkpoint)
    return surrogate, surrogate.warnings


def _validate_run_data(
    run: OptimizationRun,
    recipes: Sequence[RecipePoint],
    shots: Sequence[PreferenceShot],
    comparisons: Sequence[PreferenceComparison],
    state: OptimizerState,
) -> tuple[list[PreferenceShot], dict[str, RecipePoint], dict[str, PreferenceShot]]:
    if state.optimization_run_id != run.run_id:
        raise ValueError("optimizer state belongs to another run")
    recipe_by_id = {recipe.recipe_id: recipe for recipe in recipes}
    shot_by_id = {shot.shot_id: shot for shot in shots}
    if len(recipe_by_id) != len(recipes) or len(shot_by_id) != len(shots):
        raise ValueError("duplicate recipe or shot identifiers were loaded")
    if any(recipe.optimization_run_id != run.run_id for recipe in recipes):
        raise ValueError("recipe belongs to another optimization run")
    if any(shot.optimization_run_id != run.run_id for shot in shots):
        raise ValueError("shot belongs to another optimization run")
    if any(shot.recipe_id not in recipe_by_id for shot in shots):
        raise ValueError("shot references an unknown recipe")
    for shot in shots:
        observed = shot.observed_recipe
        if observed is None:
            if not recipe_by_id[shot.recipe_id].inside_search_space:
                raise ValueError("out-of-space shot is missing its observed recipe")
            continue
        canonical = RecipePoint.observe(
            run.run_id,
            run.recipe_space,
            observed.grind_size,
            observed.dose_g,
            observed.target_output_g,
            created_at=recipe_by_id[shot.recipe_id].created_at,
        )
        if canonical.recipe_id != shot.recipe_id:
            raise ValueError("shot recipe differs from its observed recipe")
    valid_shots = sorted(
        (shot for shot in shots if shot.status == PhysicalShotStatus.VALID),
        key=lambda shot: shot.sequence_number,
    )
    if not valid_shots:
        raise ValueError("optimization run has no valid physical shot")
    if len({shot.sequence_number for shot in shots}) != len(shots):
        raise ValueError("shot sequence numbers must be unique within a run")
    _validate_comparison_history(run, valid_shots, comparisons, shot_by_id)
    for identifier in (state.previous_valid_shot_id, state.incumbent_shot_id):
        if identifier is not None and (
            identifier not in shot_by_id
            or shot_by_id[identifier].status != PhysicalShotStatus.VALID
        ):
            raise ValueError("optimizer state references a missing or invalid shot")
    return valid_shots, recipe_by_id, shot_by_id


def _validate_comparison_history(
    run: OptimizationRun,
    valid_shots: Sequence[PreferenceShot],
    comparisons: Sequence[PreferenceComparison],
    shot_by_id: dict[str, PreferenceShot],
) -> None:
    ordered = sorted(comparisons, key=lambda row: (row.created_at, row.comparison_id))
    incumbent_id = valid_shots[0].shot_id
    previous_id = valid_shots[0].shot_id
    seen_new_shots: set[str] = set()
    for row in ordered:
        if row.optimization_run_id != run.run_id:
            raise ValueError("comparison belongs to another run")
        if row.new_shot_id not in shot_by_id or row.anchor_shot_id not in shot_by_id:
            raise ValueError("comparison references an unknown shot")
        new_shot = shot_by_id[row.new_shot_id]
        anchor_shot = shot_by_id[row.anchor_shot_id]
        if new_shot.status != PhysicalShotStatus.VALID or anchor_shot.status != PhysicalShotStatus.VALID:
            raise ValueError("failed or aborted shots cannot create preference comparisons")
        if row.new_shot_id in seen_new_shots:
            raise ValueError("a physical shot cannot have two preference labels")
        expected_anchor = previous_id if row.comparison_mode == ComparisonMode.GLOBAL_PREVIOUS else incumbent_id
        if row.anchor_shot_id != expected_anchor:
            raise ValueError("comparison orientation or anchor history is invalid")
        if new_shot.sequence_number <= anchor_shot.sequence_number:
            raise ValueError("new shot must occur after its anchor")
        seen_new_shots.add(row.new_shot_id)
        previous_id = row.new_shot_id
        if row.anchor_shot_id == incumbent_id and row.label == PreferenceLabel.NEW_BETTER:
            incumbent_id = row.new_shot_id


def _unique_valid_recipes(
    valid_shots: Sequence[PreferenceShot],
    recipe_by_id: dict[str, RecipePoint],
) -> list[RecipePoint]:
    result: list[RecipePoint] = []
    seen: set[str] = set()
    for shot in valid_shots:
        if shot.recipe_id not in seen:
            seen.add(shot.recipe_id)
            result.append(recipe_by_id[shot.recipe_id])
    return result

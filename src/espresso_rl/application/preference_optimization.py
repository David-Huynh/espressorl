from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable, Mapping

from espresso_rl.domain.cpbo import (
    CPBO_CONFIGURATION_VERSION,
    ComparisonMode,
    ModelRecommendation,
    OptimizationRun,
    OptimizationRunContext,
    OptimizerState,
    PhysicalShotStatus,
    PreferenceComparison,
    PreferenceLabel,
    PreferenceShot,
    RecipeDomain,
    RecipePoint,
    RecipeSpace,
    ShotRequest,
    Suggestion,
    TrustRegionState,
    new_cpbo_id,
)
from espresso_rl.domain.models import Recipe, Recommendation, RecommendationMode, RecommendationStatus
from espresso_rl.ports.preference_optimization import (
    PreferentialOptimizationRepository,
    PreferentialOptimizerEngine,
)


RecipeSpaceFactory = Callable[[Recipe, RecipeDomain], RecipeSpace]
TraceFeatureExtractor = Callable[[Any], tuple[tuple[str, ...], tuple[float, ...] | None]]


class ConsecutivePreferenceOptimizationService:
    """Stateful CPBO use cases over canonical domain models and repository ports."""

    def __init__(
        self,
        repository: PreferentialOptimizationRepository,
        optimizer: PreferentialOptimizerEngine,
        recipe_space_factory: RecipeSpaceFactory,
        *,
        random_seed: int,
        configuration_version: str = CPBO_CONFIGURATION_VERSION,
        recipe_domain: RecipeDomain = RecipeDomain(),
        initial_trust_region_length: float = 0.8,
        trace_feature_extractor: TraceFeatureExtractor | None = None,
        clock: Callable[[], int],
    ) -> None:
        if random_seed < 0:
            raise ValueError("CPBO random seed must be nonnegative")
        if not configuration_version.strip():
            raise ValueError("CPBO configuration version is required")
        if not math.isfinite(initial_trust_region_length) or initial_trust_region_length <= 0:
            raise ValueError("initial trust-region length must be positive and finite")
        self._repository = repository
        self._optimizer = optimizer
        self._recipe_space_factory = recipe_space_factory
        self._random_seed = random_seed
        self._configuration_version = configuration_version
        self._recipe_domain = recipe_domain
        self._initial_trust_region_length = initial_trust_region_length
        self._trace_feature_extractor = trace_feature_extractor
        self._clock = clock

    def initialize(
        self,
        run_context: OptimizationRunContext,
        baseline_recipe: Recipe,
        *,
        comparison_mode: ComparisonMode,
    ) -> ShotRequest:
        existing = self._compatible_active_run(run_context)
        if existing is not None:
            state = self._require_state(existing.run_id)
            pending = self._repository.get_pending_suggestion(existing.run_id)
            if pending is not None:
                if state.pending_shot_id is not None:
                    raise ValueError("the active CPBO run is awaiting preference feedback")
                return ShotRequest(
                    optimization_run_id=existing.run_id,
                    recipe=pending.recipe,
                    anchor_shot_id=pending.anchor_shot_id,
                    comparison_mode=existing.comparison_mode,
                    is_baseline=False,
                )
            shots = self._repository.list_shots(existing.run_id)
            if any(shot.status == PhysicalShotStatus.VALID for shot in shots):
                suggestion = self.suggest_next(existing.run_id)
                return ShotRequest(
                    optimization_run_id=existing.run_id,
                    recipe=suggestion.recipe,
                    anchor_shot_id=suggestion.anchor_shot_id,
                    comparison_mode=existing.comparison_mode,
                    is_baseline=False,
                )
            baseline = self._repository.list_recipes(existing.run_id)[0]
            return ShotRequest(
                optimization_run_id=existing.run_id,
                recipe=baseline,
                anchor_shot_id=None,
                comparison_mode=existing.comparison_mode,
                is_baseline=True,
            )

        now = self._clock()
        recipe_space = replace(
            self._recipe_space_factory(baseline_recipe, self._recipe_domain),
            version=self._recipe_domain.effective_version,
        )
        run = OptimizationRun(
            run_id=new_cpbo_id("run"),
            context=run_context,
            comparison_mode=ComparisonMode(comparison_mode),
            recipe_space=recipe_space,
            created_at=now,
            configuration_version=self._configuration_version,
        )
        baseline = RecipePoint.create(
            run.run_id,
            recipe_space,
            baseline_recipe.relative_grind_steps_from_reference,
            baseline_recipe.dose_g,
            baseline_recipe.target_yield_g,
            created_at=now,
        )
        state = OptimizerState(
            optimization_run_id=run.run_id,
            previous_valid_shot_id=None,
            incumbent_shot_id=None,
            iteration=0,
            trust_region_state=TrustRegionState(
                center=baseline.normalized_x,
                length=self._initial_trust_region_length,
            ),
            model_checkpoint=None,
            trace_model_checkpoint=None,
            random_seed=self._random_seed,
            configuration_version=self._configuration_version,
            updated_at=now,
        )
        self._repository.create_run(run, baseline, state)
        return ShotRequest(
            optimization_run_id=run.run_id,
            recipe=baseline,
            anchor_shot_id=None,
            comparison_mode=run.comparison_mode,
            is_baseline=True,
        )

    def suggest_next(self, run_id: str) -> Suggestion:
        run = self._require_run(run_id)
        state = self._require_state(run_id)
        pending = self._repository.get_pending_suggestion(run_id)
        if state.pending_recipe_id is not None:
            if state.pending_shot_id is not None:
                raise ValueError("the valid candidate shot still requires preference feedback")
            if pending is None:
                raise ValueError("optimizer state references a missing pending suggestion")
            return pending
        recipes = self._repository.list_recipes(run_id)
        shots = self._repository.list_shots(run_id)
        comparisons = self._repository.list_comparisons(run_id)
        computation = self._optimizer.suggest(
            run=run,
            recipes=recipes,
            shots=shots,
            comparisons=comparisons,
            state=state,
            now=self._clock(),
        )
        suggestion = computation.suggestion
        updated_state = replace(
            state,
            iteration=suggestion.iteration,
            model_checkpoint=computation.model_checkpoint,
            trace_model_checkpoint=computation.trace_model_checkpoint,
            pending_recipe_id=suggestion.recipe.recipe_id,
            pending_anchor_shot_id=suggestion.anchor_shot_id,
            pending_shot_id=None,
            pending_suggestion_json=None,
            updated_at=self._clock(),
        )
        self._repository.save_suggestion(suggestion.recipe, suggestion, updated_state)
        return suggestion

    def record_shot(
        self,
        run_id: str,
        recipe: Recipe | RecipePoint,
        status: PhysicalShotStatus,
        telemetry: Any = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        shot_id: str | None = None,
        started_at: int | None = None,
        completed_at: int | None = None,
        raw_telemetry_reference: str | None = None,
        allow_recipe_deviation: bool = False,
    ) -> PreferenceShot:
        run = self._require_run(run_id)
        state = self._require_state(run_id)
        status = PhysicalShotStatus(status)
        recipe_point = self._canonical_recipe(run, recipe)
        existing_shots = self._repository.list_shots(run_id)
        has_valid_baseline = any(shot.status == PhysicalShotStatus.VALID for shot in existing_shots)
        if has_valid_baseline:
            if state.pending_recipe_id is None:
                raise ValueError("a non-baseline shot requires a pending CPBO suggestion")
            if state.pending_shot_id is not None:
                raise ValueError("the pending CPBO suggestion already has a physical shot")
            if recipe_point.recipe_id != state.pending_recipe_id and not allow_recipe_deviation:
                raise ValueError("physical shot recipe does not match the pending quantized suggestion")
        else:
            baseline = self._repository.list_recipes(run_id)[0]
            if recipe_point.recipe_id != baseline.recipe_id:
                raise ValueError("first physical shot must use the configured baseline recipe")

        now = self._clock()
        started = now if started_at is None else int(started_at)
        completed = completed_at
        if status == PhysicalShotStatus.VALID and completed is None:
            completed = now
        trace_names, trace_values = _trace_features_from_telemetry(
            telemetry,
            self._trace_feature_extractor,
        )
        shot = PreferenceShot(
            shot_id=shot_id or new_cpbo_id("shot"),
            recipe_id=recipe_point.recipe_id,
            optimization_run_id=run_id,
            sequence_number=len(existing_shots) + 1,
            started_at=started,
            completed_at=completed,
            status=status,
            telemetry_available=trace_values is not None,
            raw_telemetry_reference=raw_telemetry_reference,
            trace_feature_names=trace_names,
            trace_features=trace_values,
            metadata=dict(metadata or {}),
        )
        if not has_valid_baseline and status == PhysicalShotStatus.VALID:
            updated_state = replace(
                state,
                previous_valid_shot_id=shot.shot_id,
                incumbent_shot_id=shot.shot_id,
                trust_region_state=replace(state.trust_region_state, center=recipe_point.normalized_x),
                updated_at=now,
            )
        elif status == PhysicalShotStatus.VALID:
            updated_state = replace(state, pending_shot_id=shot.shot_id, updated_at=now)
        else:
            updated_state = replace(
                state,
                pending_recipe_id=None,
                pending_anchor_shot_id=None,
                pending_shot_id=None,
                pending_suggestion_json=None,
                updated_at=now,
            )
        self._repository.record_shot(recipe_point, shot, updated_state)
        return shot

    def record_preference(
        self,
        run_id: str,
        new_shot_id: str,
        anchor_shot_id: str,
        label: PreferenceLabel,
    ) -> OptimizerState:
        run = self._require_run(run_id)
        state = self._require_state(run_id)
        label = PreferenceLabel(label)
        if state.pending_shot_id != new_shot_id:
            matching = [
                row
                for row in self._repository.list_comparisons(run_id)
                if row.new_shot_id == new_shot_id and row.anchor_shot_id == anchor_shot_id
            ]
            if matching:
                if any(row.label != label for row in matching):
                    raise ValueError("preference replay conflicts with the stored label")
                return state
            raise ValueError("new_shot_id is not the pending CPBO candidate shot")
        if state.pending_anchor_shot_id != anchor_shot_id:
            raise ValueError("anchor_shot_id reverses or changes the pending comparison orientation")
        new_shot = self._repository.get_shot(new_shot_id)
        anchor_shot = self._repository.get_shot(anchor_shot_id)
        if new_shot is None or anchor_shot is None:
            raise ValueError("preference references an unknown physical shot")
        if new_shot.optimization_run_id != run_id or anchor_shot.optimization_run_id != run_id:
            raise ValueError("preference shots belong to another optimization run")
        if new_shot.status != PhysicalShotStatus.VALID or anchor_shot.status != PhysicalShotStatus.VALID:
            raise ValueError("machine failure or aborted shot cannot create a preference")
        if new_shot.sequence_number <= anchor_shot.sequence_number:
            raise ValueError("new shot must be physically newer than its anchor")
        recipe = self._repository.get_recipe(new_shot.recipe_id)
        if recipe is None:
            raise ValueError("new shot references a missing recipe")

        now = self._clock()
        comparison = PreferenceComparison(
            comparison_id=new_cpbo_id("comparison"),
            optimization_run_id=run_id,
            new_shot_id=new_shot_id,
            anchor_shot_id=anchor_shot_id,
            label=label,
            comparison_mode=run.comparison_mode,
            created_at=now,
            taste_goal=run.context.taste_goal,
        )
        incumbent_shot_id = state.incumbent_shot_id
        trust_region_state = state.trust_region_state
        if run.comparison_mode == ComparisonMode.BEST_INCUMBENT:
            trust_region_state = self._optimizer.update_trust_region_state(
                trust_region_state,
                label,
                candidate_center=recipe.normalized_x,
            )
            if label == PreferenceLabel.NEW_BETTER:
                incumbent_shot_id = new_shot_id
        updated_state = replace(
            state,
            previous_valid_shot_id=new_shot_id,
            incumbent_shot_id=incumbent_shot_id,
            trust_region_state=trust_region_state,
            pending_recipe_id=None,
            pending_anchor_shot_id=None,
            pending_shot_id=None,
            pending_suggestion_json=None,
            updated_at=now,
        )
        self._repository.record_comparison(comparison, updated_state)
        return updated_state

    def get_state(self, run_id: str) -> OptimizerState:
        return self._require_state(run_id)

    def get_run(self, run_id: str) -> OptimizationRun:
        return self._require_run(run_id)

    def get_shot(self, shot_id: str) -> PreferenceShot:
        shot = self._repository.get_shot(shot_id)
        if shot is None:
            raise ValueError(f"unknown CPBO shot {shot_id}")
        return shot

    def find_shot(self, shot_id: str) -> PreferenceShot | None:
        return self._repository.get_shot(shot_id)

    def get_comparison(
        self,
        run_id: str,
        new_shot_id: str,
        anchor_shot_id: str,
    ) -> PreferenceComparison:
        matching = [
            comparison
            for comparison in self._repository.list_comparisons(run_id)
            if comparison.new_shot_id == new_shot_id
            and comparison.anchor_shot_id == anchor_shot_id
        ]
        if len(matching) != 1:
            raise ValueError("stored preference comparison is missing or ambiguous")
        return matching[0]

    def reset_owner(self, install_id: str, machine_id: str) -> dict[str, int]:
        return self._repository.reset_owner(install_id, machine_id)

    def get_recommendation(self, run_id: str) -> ModelRecommendation:
        run = self._require_run(run_id)
        return self._optimizer.recommend_evaluated(
            run=run,
            recipes=self._repository.list_recipes(run_id),
            shots=self._repository.list_shots(run_id),
            comparisons=self._repository.list_comparisons(run_id),
            state=self._require_state(run_id),
        )

    def active_run(self, context: OptimizationRunContext) -> OptimizationRun | None:
        return self._compatible_active_run(context)

    def _compatible_active_run(
        self,
        context: OptimizationRunContext,
    ) -> OptimizationRun | None:
        run = self._repository.find_active_run(context)
        if run is None:
            return None
        if (
            run.configuration_version == self._configuration_version
            and run.recipe_space.version == self._recipe_domain.effective_version
        ):
            return run
        self._repository.deactivate_run(run.run_id)
        return None

    def suggestion_to_machine_recommendation(
        self,
        suggestion: Suggestion,
        *,
        current_recipe: Recipe,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        grinder_context_id: str | None,
        profile_id: str | None,
        raw_profile_hash: str | None = None,
        now: int | None = None,
    ) -> Recommendation:
        timestamp = self._clock() if now is None else int(now)
        run = self._require_run(suggestion.optimization_run_id)
        candidate = suggestion.recipe
        run.recipe_space.validate_recipe(
            candidate.grind_size,
            candidate.dose_g,
            candidate.target_output_g,
        )
        grind_delta = candidate.grind_size - current_recipe.relative_grind_steps_from_reference
        mode = (
            RecommendationMode.CPBO_GLOBAL_PREVIOUS
            if suggestion.comparison_mode == ComparisonMode.GLOBAL_PREVIOUS
            else RecommendationMode.CPBO_BEST_INCUMBENT
        )
        recommendation = Recommendation(
            recommendation_id=suggestion.suggestion_id,
            created_at=timestamp,
            updated_at=timestamp,
            expires_at=None,
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            grinder_context_id=grinder_context_id,
            profile_id=profile_id,
            raw_profile_hash=raw_profile_hash,
            grind_delta_steps_from_current=grind_delta,
            grind_delta_um_from_current=(
                grind_delta * current_recipe.microns_per_step * current_recipe.grinder_direction_sign
            ),
            projected_relative_step_from_reference=candidate.grind_size,
            projected_relative_grind_um_from_reference=(
                candidate.grind_size
                * current_recipe.microns_per_step
                * current_recipe.grinder_direction_sign
            ),
            next_dose_g=candidate.dose_g,
            target_yield_g=candidate.target_output_g,
            target_ratio=candidate.brew_ratio,
            mode=mode,
            confidence=suggestion.acquisition.outcome_probabilities[PreferenceLabel.NEW_BETTER.value],
            reason="CPBO-MES selected one quantized recipe for comparison against the anchor shot.",
            status=RecommendationStatus.PENDING,
            source_shot_id=suggestion.anchor_shot_id,
            optimization_run_id=suggestion.optimization_run_id,
            comparison_anchor_shot_id=suggestion.anchor_shot_id,
            comparison_mode=suggestion.comparison_mode.value,
            preference_feedback_required=True,
            taste_goal=run.context.taste_goal,
            grinder_step_direction=current_recipe.grinder_step_direction,
            grinder_adjustment_mode=current_recipe.grinder_adjustment_mode,
        )
        return recommendation

    def configure_recipe_domain(self, recipe_domain: RecipeDomain) -> None:
        self._recipe_domain = recipe_domain

    def _canonical_recipe(self, run: OptimizationRun, recipe: Recipe | RecipePoint) -> RecipePoint:
        if isinstance(recipe, RecipePoint):
            if recipe.optimization_run_id != run.run_id:
                raise ValueError("recipe belongs to another optimization run")
            canonical = RecipePoint.create(
                run.run_id,
                run.recipe_space,
                recipe.grind_size,
                recipe.dose_g,
                recipe.target_output_g,
                created_at=recipe.created_at,
            )
            if canonical.recipe_id != recipe.recipe_id:
                raise ValueError("recipe is not canonical for the run recipe space")
            return recipe
        if recipe.grinder_step_direction != run.recipe_space.grinder_step_direction:
            raise ValueError("recipe grinder direction differs from the optimization run")
        return RecipePoint.create(
            run.run_id,
            run.recipe_space,
            recipe.relative_grind_steps_from_reference,
            recipe.dose_g,
            recipe.target_yield_g,
            created_at=self._clock(),
        )

    def _require_run(self, run_id: str) -> OptimizationRun:
        run = self._repository.get_run(run_id)
        if run is None:
            raise ValueError(f"unknown optimization run {run_id}")
        return run

    def _require_state(self, run_id: str) -> OptimizerState:
        state = self._repository.get_state(run_id)
        if state is None:
            raise ValueError(f"optimization run {run_id} has no persisted state")
        return state


def _trace_features_from_telemetry(
    telemetry: Any,
    extractor: TraceFeatureExtractor | None,
) -> tuple[tuple[str, ...], tuple[float, ...] | None]:
    if telemetry is None:
        return (), None
    if isinstance(telemetry, Mapping):
        names = tuple(str(name) for name in telemetry.get("feature_names") or ())
        values = telemetry.get("features")
        if values is None:
            return (), None
        return names, tuple(float(value) for value in values)
    if extractor is not None:
        return extractor(telemetry)
    raise ValueError("telemetry must be a fixed-cadence sequence or trace-feature object")

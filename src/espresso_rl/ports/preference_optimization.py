from __future__ import annotations

from typing import Protocol, Sequence

from espresso_rl.domain.cpbo import (
    ModelRecommendation,
    OptimizationRun,
    OptimizationRunContext,
    OptimizerState,
    PreferenceComparison,
    PreferenceLabel,
    PreferenceShot,
    RecipePoint,
    Suggestion,
    SuggestionComputation,
    TrustRegionState,
)


class PreferentialOptimizerEngine(Protocol):
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
        ...

    def recommend_evaluated(
        self,
        *,
        run: OptimizationRun,
        recipes: Sequence[RecipePoint],
        shots: Sequence[PreferenceShot],
        comparisons: Sequence[PreferenceComparison],
        state: OptimizerState,
    ) -> ModelRecommendation:
        ...

    def update_trust_region_state(
        self,
        state: TrustRegionState,
        label: PreferenceLabel,
        *,
        candidate_center: tuple[float, float, float],
    ) -> TrustRegionState:
        ...


class PreferentialOptimizationRepository(Protocol):
    def find_active_run(self, context: OptimizationRunContext) -> OptimizationRun | None:
        ...

    def get_run(self, run_id: str) -> OptimizationRun | None:
        ...

    def create_run(
        self,
        run: OptimizationRun,
        baseline_recipe: RecipePoint,
        state: OptimizerState,
    ) -> None:
        ...

    def deactivate_run(self, run_id: str) -> None:
        ...

    def update_run_configuration(
        self,
        run: OptimizationRun,
        state: OptimizerState,
    ) -> None:
        ...

    def migrate_run_recipe_space(
        self,
        run: OptimizationRun,
        recipes: Sequence[RecipePoint],
        shots: Sequence[PreferenceShot],
        state: OptimizerState,
    ) -> None:
        ...

    def get_recipe(self, recipe_id: str) -> RecipePoint | None:
        ...

    def list_recipes(self, run_id: str) -> list[RecipePoint]:
        ...

    def list_shots(self, run_id: str) -> list[PreferenceShot]:
        ...

    def get_shot(self, shot_id: str) -> PreferenceShot | None:
        ...

    def list_comparisons(self, run_id: str) -> list[PreferenceComparison]:
        ...

    def get_state(self, run_id: str) -> OptimizerState | None:
        ...

    def get_pending_suggestion(self, run_id: str) -> Suggestion | None:
        ...

    def save_suggestion(
        self,
        recipe: RecipePoint,
        suggestion: Suggestion,
        state: OptimizerState,
    ) -> None:
        ...

    def record_shot(
        self,
        recipe: RecipePoint,
        shot: PreferenceShot,
        state: OptimizerState,
    ) -> None:
        ...

    def replace_shot_observation(
        self,
        recipe: RecipePoint,
        shot: PreferenceShot,
        state: OptimizerState,
        *,
        invalidate_pending_suggestion: bool,
    ) -> None:
        ...

    def replace_history_after_shot_exclusion(
        self,
        shot: PreferenceShot,
        comparisons: Sequence[PreferenceComparison],
        state: OptimizerState,
    ) -> None:
        ...

    def record_comparison(
        self,
        comparison: PreferenceComparison,
        state: OptimizerState,
    ) -> None:
        ...

    def reset_owner(self, install_id: str, machine_id: str) -> dict[str, int]:
        ...

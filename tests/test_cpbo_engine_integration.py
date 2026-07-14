from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from espresso_rl.adapters.sqlite_repositories import (
    SQLitePreferentialOptimizationRepository,
    SQLiteStore,
)
from espresso_rl.application.preference_optimization import ConsecutivePreferenceOptimizationService
from espresso_rl.domain.cpbo import (
    ComparisonMode,
    OptimizationRunContext,
    PhysicalShotStatus,
    PreferenceLabel,
    RecipeParameter,
    RecipeSpace,
)
from espresso_rl.domain.models import GrinderStepDirection, Recipe
from espresso_rl.optimizers.cpbo import ConsecutivePreferentialBayesianOptimizer
from espresso_rl.optimizers.cpbo_config import application_cpbo_config
from espresso_rl.optimizers.cpbo_trace import TRACE_FEATURE_NAMES


class RealCPBOEngineIntegrationTests(unittest.TestCase):
    def test_real_engine_completes_preference_only_iterations(self) -> None:
        config = fast_cpbo_config()
        clock = CounterClock()
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "cpbo.db") as store:
                repository = SQLitePreferentialOptimizationRepository(store)
                service = ConsecutivePreferenceOptimizationService(
                    repository,
                    ConsecutivePreferentialBayesianOptimizer(config),
                    recipe_space_factory,
                    random_seed=config.random_seed,
                    configuration_version=config.effective_configuration_version,
                    clock=clock,
                )
                request = service.initialize(
                    OptimizationRunContext("install", "machine", "bean", "grinder", "profile"),
                    baseline_recipe(),
                    comparison_mode=ComparisonMode.BEST_INCUMBENT,
                )
                service.record_shot(
                    request.optimization_run_id,
                    request.recipe,
                    PhysicalShotStatus.VALID,
                    shot_id="baseline",
                    started_at=1,
                    completed_at=2,
                )

                first = service.suggest_next(request.optimization_run_id)
                self.assertEqual(first.anchor_shot_id, "baseline")
                self.assertEqual(first.iteration, 1)
                self.assertGreaterEqual(first.acquisition.acquisition_value, 0.0)
                self.assertEqual(len(first.recipe.normalized_x), 3)
                self.assertEqual(first.recipe.grind_size, round(first.recipe.grind_size))

                service.record_shot(
                    request.optimization_run_id,
                    first.recipe,
                    PhysicalShotStatus.VALID,
                    shot_id="candidate_1",
                    started_at=3,
                    completed_at=4,
                )
                service.record_preference(
                    request.optimization_run_id,
                    "candidate_1",
                    "baseline",
                    PreferenceLabel.TIE,
                )
                second = service.suggest_next(request.optimization_run_id)
                self.assertEqual(second.anchor_shot_id, "baseline")
                comparisons = repository.list_comparisons(request.optimization_run_id)
                self.assertEqual([row.label for row in comparisons], [PreferenceLabel.TIE])
                self.assertIsNone(repository.list_shots(request.optimization_run_id)[0].metadata.get("rating"))

    def test_seeded_synthetic_jnd_trials_improve_over_corner_baseline(self) -> None:
        for seed in (3, 7):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as tmp:
                config = fast_cpbo_config(seed=seed)
                clock = CounterClock()
                with SQLiteStore(Path(tmp) / "cpbo.db") as store:
                    repository = SQLitePreferentialOptimizationRepository(store)
                    service = ConsecutivePreferenceOptimizationService(
                        repository,
                        ConsecutivePreferentialBayesianOptimizer(config),
                        corner_recipe_space_factory,
                        random_seed=config.random_seed,
                        configuration_version=config.effective_configuration_version,
                        clock=clock,
                    )
                    request = service.initialize(
                        OptimizationRunContext(
                            "install",
                            f"machine_{seed}",
                            "bean",
                            "grinder",
                            "profile",
                        ),
                        corner_baseline_recipe(),
                        comparison_mode=ComparisonMode.BEST_INCUMBENT,
                    )
                    baseline = service.record_shot(
                        request.optimization_run_id,
                        request.recipe,
                        PhysicalShotStatus.VALID,
                        shot_id=f"baseline_{seed}",
                        started_at=1,
                        completed_at=2,
                    )
                    suggestion = service.suggest_next(request.optimization_run_id)
                    candidate = service.record_shot(
                        request.optimization_run_id,
                        suggestion.recipe,
                        PhysicalShotStatus.VALID,
                        shot_id=f"candidate_{seed}",
                        started_at=3,
                        completed_at=4,
                    )
                    label = synthetic_jnd_oracle(
                        suggestion.recipe.normalized_x,
                        request.recipe.normalized_x,
                        gamma=0.001,
                    )
                    self.assertEqual(label, PreferenceLabel.NEW_BETTER)
                    service.record_preference(
                        request.optimization_run_id,
                        candidate.shot_id,
                        baseline.shot_id,
                        label,
                    )

                    recommendation = service.get_recommendation(request.optimization_run_id)
                    self.assertTrue(recommendation.directly_established)
                    self.assertGreater(
                        synthetic_utility(recommendation.recipe.normalized_x),
                        synthetic_utility(request.recipe.normalized_x),
                    )

    def test_telemetry_can_activate_after_run_start_without_schema_change(self) -> None:
        config = fast_cpbo_config(trace_minimum=2)
        clock = CounterClock()
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "cpbo.db") as store:
                repository = SQLitePreferentialOptimizationRepository(store)
                service = ConsecutivePreferenceOptimizationService(
                    repository,
                    ConsecutivePreferentialBayesianOptimizer(config),
                    recipe_space_factory,
                    random_seed=config.random_seed,
                    configuration_version=config.effective_configuration_version,
                    trace_feature_extractor=lambda values: (
                        TRACE_FEATURE_NAMES,
                        tuple(float(value) for value in values),
                    ),
                    clock=clock,
                )
                request = service.initialize(
                    OptimizationRunContext("install", "machine", "bean", "grinder", "profile"),
                    baseline_recipe(),
                    comparison_mode=ComparisonMode.BEST_INCUMBENT,
                )
                service.record_shot(
                    request.optimization_run_id,
                    request.recipe,
                    PhysicalShotStatus.VALID,
                    shot_id="baseline",
                    started_at=1,
                    completed_at=2,
                )

                for index in range(1, 4):
                    suggestion = service.suggest_next(request.optimization_run_id)
                    telemetry = (
                        None
                        if index == 1
                        else tuple(float(feature + index) for feature in range(len(TRACE_FEATURE_NAMES)))
                    )
                    service.record_shot(
                        request.optimization_run_id,
                        suggestion.recipe,
                        PhysicalShotStatus.VALID,
                        telemetry=telemetry,
                        shot_id=f"candidate_{index}",
                        started_at=index * 2 + 1,
                        completed_at=index * 2 + 2,
                    )
                    service.record_preference(
                        request.optimization_run_id,
                        f"candidate_{index}",
                        "baseline",
                        PreferenceLabel.TIE,
                    )

                with_trace = service.suggest_next(request.optimization_run_id)
                self.assertTrue(with_trace.acquisition.trace_kernel_enabled)
                self.assertGreater(with_trace.acquisition.kernel_weights["trace"], 0.0)
                shots = repository.list_shots(request.optimization_run_id)
                self.assertEqual(
                    [shot.telemetry_available for shot in shots],
                    [False, False, True, True],
                )

    def test_real_engine_refits_mixed_anchor_policy_history(self) -> None:
        config = fast_cpbo_config()
        clock = CounterClock()
        context = OptimizationRunContext("install", "machine", "bean", "grinder", "profile")
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "cpbo.db") as store:
                repository = SQLitePreferentialOptimizationRepository(store)
                service = ConsecutivePreferenceOptimizationService(
                    repository,
                    ConsecutivePreferentialBayesianOptimizer(config),
                    recipe_space_factory,
                    random_seed=config.random_seed,
                    configuration_version=config.effective_configuration_version,
                    clock=clock,
                )
                baseline = service.initialize(
                    context,
                    baseline_recipe(),
                    comparison_mode=ComparisonMode.BEST_INCUMBENT,
                )
                service.record_shot(
                    baseline.optimization_run_id,
                    baseline.recipe,
                    PhysicalShotStatus.VALID,
                    shot_id="baseline",
                    started_at=1,
                    completed_at=2,
                )
                local = service.suggest_next(baseline.optimization_run_id)
                service.record_shot(
                    baseline.optimization_run_id,
                    local.recipe,
                    PhysicalShotStatus.VALID,
                    shot_id="local_candidate",
                    started_at=3,
                    completed_at=4,
                )
                service.record_preference(
                    baseline.optimization_run_id,
                    "local_candidate",
                    "baseline",
                    PreferenceLabel.TIE,
                )

                global_request = service.initialize(
                    context,
                    baseline_recipe(),
                    comparison_mode=ComparisonMode.GLOBAL_PREVIOUS,
                )
                self.assertEqual(global_request.optimization_run_id, baseline.optimization_run_id)
                self.assertEqual(global_request.anchor_shot_id, "local_candidate")
                service.record_shot(
                    baseline.optimization_run_id,
                    global_request.recipe,
                    PhysicalShotStatus.VALID,
                    shot_id="global_candidate",
                    started_at=5,
                    completed_at=6,
                )
                service.record_preference(
                    baseline.optimization_run_id,
                    "global_candidate",
                    "local_candidate",
                    PreferenceLabel.TIE,
                )

                local_again = service.initialize(
                    context,
                    baseline_recipe(),
                    comparison_mode=ComparisonMode.BEST_INCUMBENT,
                )
                self.assertEqual(local_again.optimization_run_id, baseline.optimization_run_id)
                self.assertEqual(local_again.anchor_shot_id, "baseline")
                self.assertEqual(len(repository.list_comparisons(baseline.optimization_run_id)), 2)


class CounterClock:
    def __init__(self) -> None:
        self.value = 100

    def __call__(self) -> int:
        self.value += 1
        return self.value


def baseline_recipe() -> Recipe:
    return Recipe(
        relative_grind_steps_from_reference=5.0,
        microns_per_step=10.0,
        dose_g=18.0,
        target_yield_g=36.0,
        grinder_step_direction=GrinderStepDirection.HIGHER_IS_FINER,
    )


def recipe_space_factory(recipe: Recipe, _recipe_domain: object) -> RecipeSpace:
    return RecipeSpace(
        RecipeParameter("grind_size", 0.0, 10.0, 1.0, "step"),
        RecipeParameter("dose_g", 16.0, 20.0, 0.1, "g"),
        RecipeParameter("target_output_g", 26.0, 46.0, 0.1, "g"),
        recipe.grinder_step_direction,
    )


def corner_baseline_recipe() -> Recipe:
    return Recipe(
        relative_grind_steps_from_reference=0.0,
        microns_per_step=10.0,
        dose_g=14.0,
        target_yield_g=20.0,
        grinder_step_direction=GrinderStepDirection.HIGHER_IS_FINER,
    )


def corner_recipe_space_factory(recipe: Recipe, _recipe_domain: object) -> RecipeSpace:
    return RecipeSpace(
        RecipeParameter("grind_size", 0.0, 10.0, 1.0, "step"),
        RecipeParameter("dose_g", 14.0, 22.0, 0.1, "g"),
        RecipeParameter("target_output_g", 20.0, 60.0, 0.1, "g"),
        recipe.grinder_step_direction,
    )


def synthetic_utility(normalized_x: tuple[float, float, float]) -> float:
    return sum(normalized_x)


def synthetic_jnd_oracle(
    new_x: tuple[float, float, float],
    anchor_x: tuple[float, float, float],
    *,
    gamma: float,
) -> PreferenceLabel:
    difference = synthetic_utility(new_x) - synthetic_utility(anchor_x)
    if difference > gamma:
        return PreferenceLabel.NEW_BETTER
    if difference < -gamma:
        return PreferenceLabel.ANCHOR_BETTER
    return PreferenceLabel.TIE


def fast_cpbo_config(*, seed: int = 1, trace_minimum: int = 8):
    base = application_cpbo_config()
    return replace(
        base,
        random_seed=seed,
        model=replace(
            base.model,
            fit_steps=8,
            likelihood_samples=12,
            early_stopping_patience=4,
        ),
        acquisition=replace(
            base.acquisition,
            sobol_candidate_count=24,
            local_candidate_count=8,
            posterior_max_function_samples=128,
            gumbel_maximum_samples=300,
            maximum_value_bins=5,
            truncated_samples_per_bin=96,
            candidate_chunk_size=4,
            rejection_batch_size=512,
            rejection_max_batches=4,
            gibbs_burn_in=30,
        ),
        trace=replace(
            base.trace,
            minimum_valid_telemetry_shots=trace_minimum,
            fit_steps=3,
            early_stopping_patience=2,
        ),
    )


if __name__ == "__main__":
    unittest.main()

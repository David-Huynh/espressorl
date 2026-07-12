from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from espresso_rl.adapters.sqlite_repositories import (
    SQLitePreferentialOptimizationRepository,
    SQLiteStore,
)
from espresso_rl.application.cpbo_runtime import CPBORuntimeBridge, strict_context_from_shot
from espresso_rl.application.preference_optimization import (
    ConsecutivePreferenceOptimizationService,
)
from espresso_rl.application.upload_payloads import recommendation_upload_payload
from espresso_rl.application.upload_validation import validate_upload_payload
from espresso_rl.domain.cpbo import (
    AcquisitionDiagnostics,
    ComparisonMode,
    ModelRecommendation,
    PreferenceLabel,
    RecipeParameter,
    RecipePoint,
    RecipeSpace,
    Suggestion,
    SuggestionComputation,
    TrustRegionDiagnostics,
)
from espresso_rl.domain.events import PreferenceFeedbackEvent
from espresso_rl.domain.models import GrinderStepDirection, Recipe, SafetyBounds, ShotRecord
from espresso_rl.domain.taste_goal import TasteGoal
from espresso_rl.optimizers.cpbo_config import TrustRegionConfig
from espresso_rl.optimizers.cpbo_trust_region import update_trust_region


class _ShotRepository:
    def __init__(self) -> None:
        self.rows: dict[str, ShotRecord] = {}

    def get(self, shot_id: str) -> ShotRecord | None:
        return self.rows.get(shot_id)


class _Engine:
    def __init__(self, grinds: list[float]) -> None:
        self.grinds = list(grinds)

    def suggest(self, *, run, recipes, shots, comparisons, state, now):
        anchor = (
            state.previous_valid_shot_id
            if run.comparison_mode == ComparisonMode.GLOBAL_PREVIOUS
            else state.incumbent_shot_id
        )
        recipe = RecipePoint.create(
            run.run_id,
            run.recipe_space,
            self.grinds.pop(0),
            18.0,
            36.0,
            created_at=now,
        )
        suggestion = Suggestion(
            suggestion_id=f"suggestion_{state.iteration + 1}",
            optimization_run_id=run.run_id,
            recipe=recipe,
            anchor_shot_id=anchor,
            comparison_mode=run.comparison_mode,
            acquisition=AcquisitionDiagnostics(
                acquisition_value=0.2,
                unclipped_acquisition_value=0.2,
                outcome_probabilities={
                    "new_better": 0.45,
                    "tie": 0.10,
                    "anchor_better": 0.45,
                },
                learned_gamma=0.2,
                kernel_weights={"raw": 0.8, "physics": 0.2, "trace": 0.0},
                raw_kernel_lengthscales=(1.0, 1.0, 1.0),
                physics_kernel_lengthscales=(1.0,),
                trace_kernel_enabled=False,
                fit_warnings=(),
                maximum_strategy="paper_gumbel",
                truncation_fallback_count=0,
            ),
            trust_region=TrustRegionDiagnostics(
                length=state.trust_region_state.length,
                lower_bounds=(0.0, 0.0, 0.0),
                upper_bounds=(1.0, 1.0, 1.0),
                success_count=state.trust_region_state.success_count,
                failure_count=state.trust_region_state.failure_count,
                restart_pending=state.trust_region_state.restart_pending,
                full_domain_proposal=False,
            ),
            model_version="test_cpbo",
            iteration=state.iteration + 1,
            created_at=now,
        )
        return SuggestionComputation(suggestion, '{"model":"safe"}', None)

    def recommend_evaluated(self, *, run, recipes, shots, comparisons, state):
        shot_id = state.incumbent_shot_id or state.previous_valid_shot_id
        shot = next(row for row in shots if row.shot_id == shot_id)
        recipe = next(row for row in recipes if row.recipe_id == shot.recipe_id)
        return ModelRecommendation(run.run_id, recipe, "test", True, shot_id)

    def update_trust_region_state(self, state, label, *, candidate_center):
        return update_trust_region(
            state,
            label,
            candidate_center=candidate_center,
            config=TrustRegionConfig(),
        )


class CPBORuntimeBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = SQLiteStore(Path(self.temp.name) / "runtime.db")
        self.repository = SQLitePreferentialOptimizationRepository(self.store)
        self.shots = _ShotRepository()
        self.recommendations = []
        self.clock_value = 100

    def tearDown(self) -> None:
        self.store.close()
        self.temp.cleanup()

    def bridge(self, grinds: list[float], comparison_sink=None) -> CPBORuntimeBridge:
        service = ConsecutivePreferenceOptimizationService(
            self.repository,
            _Engine(grinds),
            _recipe_space,
            random_seed=7,
            clock=self.clock,
        )
        return CPBORuntimeBridge(
            service,
            self.shots,
            self.recommendations.append,
            strict_context_from_shot,
            comparison_sink,
            comparison_mode=ComparisonMode.BEST_INCUMBENT,
            safety_bounds=SafetyBounds(),
        )

    def clock(self) -> int:
        self.clock_value += 1
        return self.clock_value

    def test_baseline_candidate_and_preference_advance_without_numeric_rating(self) -> None:
        uploaded_comparisons = []
        bridge = self.bridge([6.0, 7.0], uploaded_comparisons.append)
        baseline = _shot("baseline", grind=5.0)
        self.shots.rows[baseline.shot_id] = baseline

        baseline_outcome = bridge.handle_shot(baseline)

        self.assertIsNotNone(baseline_outcome.recommendation)
        first = baseline_outcome.recommendation
        self.assertTrue(first.preference_feedback_required)
        self.assertEqual(first.comparison_anchor_shot_id, "baseline")
        self.assertEqual(first.projected_relative_step_from_reference, 6.0)
        self.assertTrue(validate_upload_payload(recommendation_upload_payload(first)).ok)

        candidate = _shot("candidate", grind=6.0)
        self.shots.rows[candidate.shot_id] = candidate
        candidate_outcome = bridge.handle_shot(candidate)
        self.assertTrue(candidate_outcome.awaiting_preference)
        self.assertIsNone(candidate_outcome.recommendation)

        next_recommendation = bridge.handle_preference(
            PreferenceFeedbackEvent(
                optimization_run_id=first.optimization_run_id,
                new_shot_id="candidate",
                anchor_shot_id="baseline",
                label=PreferenceLabel.ANCHOR_BETTER,
                comparison_mode=ComparisonMode.BEST_INCUMBENT,
                install_id="install",
                machine_id="gaggimate:AA_BB",
                timestamp=200,
            )
        )

        comparisons = self.repository.list_comparisons(first.optimization_run_id)
        self.assertEqual([row.label for row in comparisons], [PreferenceLabel.ANCHOR_BETTER])
        self.assertEqual(len(uploaded_comparisons), 1)
        self.assertEqual(uploaded_comparisons[0].new_shot_id, "candidate")
        self.assertEqual(uploaded_comparisons[0].anchor_shot_id, "baseline")
        self.assertEqual(uploaded_comparisons[0].label, "anchor_better")
        self.assertEqual(uploaded_comparisons[0].taste_goal, TasteGoal.balanced())
        self.assertEqual(next_recommendation.comparison_anchor_shot_id, "baseline")
        self.assertEqual(next_recommendation.projected_relative_step_from_reference, 7.0)

    def test_preference_cannot_cross_taste_goal_contexts(self) -> None:
        goal = TasteGoal.custom({"sweet": "high", "bitter": "low"})
        bridge = self.bridge([6.0, 7.0])
        baseline = _shot("baseline", grind=5.0, taste_goal=goal)
        self.shots.rows[baseline.shot_id] = baseline
        first = bridge.handle_shot(baseline).recommendation
        candidate = _shot("candidate", grind=6.0, taste_goal=goal)
        self.shots.rows[candidate.shot_id] = candidate
        bridge.handle_shot(candidate)

        with self.assertRaisesRegex(ValueError, "taste goal"):
            bridge.handle_preference(
                PreferenceFeedbackEvent(
                    optimization_run_id=first.optimization_run_id,
                    new_shot_id="candidate",
                    anchor_shot_id="baseline",
                    label=PreferenceLabel.NEW_BETTER,
                    comparison_mode=ComparisonMode.BEST_INCUMBENT,
                    install_id="install",
                    machine_id="gaggimate:AA_BB",
                    timestamp=200,
                    taste_goal=TasteGoal.balanced(),
                )
            )
        self.assertEqual(self.repository.list_comparisons(first.optimization_run_id), [])

    def test_unknown_recipe_control_is_not_fabricated(self) -> None:
        bridge = self.bridge([6.0])
        shot = _shot("unknown", grind=None)
        self.shots.rows[shot.shot_id] = shot

        outcome = bridge.handle_shot(shot)

        self.assertEqual(outcome.skipped_reason, "recipe_controls_not_fully_known")
        self.assertEqual(self.recommendations, [])

    def test_confirmed_dose_target_allows_cpbo_without_a_measured_dose(self) -> None:
        bridge = self.bridge([6.0, 7.0])
        shot = _shot(
            "manual-dose",
            grind=5.0,
            dose_observed=False,
            dose_target_g=18.0,
            dose_target_confirmed=True,
        )
        self.shots.rows[shot.shot_id] = shot

        outcome = bridge.handle_shot(shot)

        self.assertIsNotNone(outcome.recommendation)
        stored = self.repository.get_shot("manual-dose")
        recipe = self.repository.get_recipe(stored.recipe_id)
        self.assertEqual(recipe.dose_g, 18.0)
        self.assertFalse(stored.metadata["dose_measured"])

        candidate = _shot(
            "manual-dose-candidate",
            grind=6.0,
            dose_observed=False,
            dose_target_g=18.0,
            dose_target_confirmed=True,
        )
        self.shots.rows[candidate.shot_id] = candidate
        candidate_outcome = bridge.handle_shot(candidate)
        self.assertTrue(candidate_outcome.awaiting_preference)

    def test_unconfirmed_manual_dose_is_not_a_cpbo_recipe_coordinate(self) -> None:
        bridge = self.bridge([6.0])
        shot = _shot(
            "unconfirmed-dose",
            grind=5.0,
            dose_observed=False,
            dose_target_g=18.0,
            dose_target_confirmed=False,
        )

        outcome = bridge.handle_shot(shot)

        self.assertEqual(outcome.skipped_reason, "recipe_controls_not_fully_known")

    def test_unmeasured_dose_without_an_explicit_target_is_not_fabricated(self) -> None:
        bridge = self.bridge([6.0])
        shot = _shot(
            "unknown-dose",
            grind=5.0,
            dose_observed=False,
            dose_target_g=None,
        )
        self.shots.rows[shot.shot_id] = shot

        outcome = bridge.handle_shot(shot)

        self.assertEqual(outcome.skipped_reason, "recipe_controls_not_fully_known")
        self.assertEqual(self.recommendations, [])

    def test_machine_failure_is_not_stored_as_tie(self) -> None:
        bridge = self.bridge([6.0, 7.0])
        baseline = _shot("baseline", grind=5.0)
        self.shots.rows[baseline.shot_id] = baseline
        first = bridge.handle_shot(baseline).recommendation
        failed = _shot("failed", grind=6.0, shot_end_state="machine_failure")
        self.shots.rows[failed.shot_id] = failed

        outcome = bridge.handle_shot(failed)

        self.assertIsNotNone(outcome.recommendation)
        self.assertEqual(self.repository.list_comparisons(first.optimization_run_id), [])
        self.assertEqual(
            self.repository.get_shot("failed").status.value,
            "machine_failure",
        )

    def test_new_shot_does_not_replace_an_unanswered_comparison(self) -> None:
        bridge = self.bridge([6.0])
        baseline = _shot("baseline", grind=5.0)
        self.shots.rows[baseline.shot_id] = baseline
        first = bridge.handle_shot(baseline).recommendation
        candidate = _shot("candidate", grind=6.0)
        self.shots.rows[candidate.shot_id] = candidate
        bridge.handle_shot(candidate)
        extra = _shot("extra", grind=7.0)
        self.shots.rows[extra.shot_id] = extra

        outcome = bridge.handle_shot(extra)

        self.assertTrue(outcome.awaiting_preference)
        self.assertEqual(outcome.skipped_reason, "preference_feedback_pending")
        self.assertIsNone(self.repository.get_shot("extra"))
        state = self.repository.get_state(first.optimization_run_id)
        self.assertEqual(state.pending_shot_id, "candidate")

    def test_profile_content_hash_is_part_of_run_context(self) -> None:
        first = _shot("first", grind=5.0, raw_profile_hash="a" * 64)
        changed = _shot("changed", grind=5.0, raw_profile_hash="b" * 64)
        self.assertNotEqual(
            strict_context_from_shot(first).fingerprint,
            strict_context_from_shot(changed).fingerprint,
        )


def _recipe_space(recipe: Recipe) -> RecipeSpace:
    return RecipeSpace(
        RecipeParameter("grind_size", 0.0, 10.0, 1.0, "step"),
        RecipeParameter("dose_g", 14.0, 22.0, 0.1, "g"),
        RecipeParameter("target_output_g", 20.0, 60.0, 0.1, "g"),
        recipe.grinder_step_direction,
        1.2,
        3.5,
    )


def _shot(
    shot_id: str,
    *,
    grind: float | None,
    shot_end_state: str = "finished",
    raw_profile_hash: str | None = None,
    taste_goal: TasteGoal | None = None,
    dose_observed: bool = True,
    dose_target_g: float | None = None,
    dose_target_confirmed: bool = False,
) -> ShotRecord:
    return ShotRecord(
        shot_id=shot_id,
        timestamp=100,
        install_id="install",
        machine_id="gaggimate:AA_BB",
        machine_adapter="gaggimate",
        profile=np.zeros((5, 100), dtype=np.float32),
        microns_per_step=10.0,
        relative_grind_steps_from_reference=grind,
        grind_observed=grind is not None,
        dose_in_g=18.0,
        dose_observed=dose_observed,
        dose_target_g=dose_target_g,
        dose_target_confirmed=dose_target_confirmed,
        target_yield_g=36.0,
        target_yield_observed=True,
        beverage_out_g=35.5,
        shot_time_s=30.0,
        bean_context_id="bean",
        grinder_context_id="grinder",
        taste_goal=taste_goal or TasteGoal.balanced(),
        profile_id="profile",
        raw_profile_hash=raw_profile_hash,
        grinder_step_direction=GrinderStepDirection.HIGHER_IS_FINER,
        shot_end_state=shot_end_state,
    )


if __name__ == "__main__":
    unittest.main()

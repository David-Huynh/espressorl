from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from espresso_rl.adapters.sqlite_repositories import (
    SQLitePreferentialOptimizationRepository,
    SQLiteStore,
)
from espresso_rl.adapters.cpbo_serialization import (
    shot_from_json,
    state_from_json,
    state_to_json,
)
from espresso_rl.application.preference_optimization import (
    ConsecutivePreferenceOptimizationService,
)
from espresso_rl.domain.cpbo import (
    AcquisitionDiagnostics,
    ComparisonMode,
    ModelRecommendation,
    OptimizationRunContext,
    PhysicalShotStatus,
    PreferenceLabel,
    RecipeDomain,
    RecipeParameter,
    RecipePoint,
    RecipeSpace,
    Suggestion,
    SuggestionComputation,
    TrustRegionDiagnostics,
)
from espresso_rl.domain.models import GrinderStepDirection, Recipe
from espresso_rl.domain.taste_goal import TasteGoal
from espresso_rl.optimizers.cpbo_config import TrustRegionConfig
from espresso_rl.optimizers.cpbo_trust_region import resume_trust_region, update_trust_region


class RecordingEngine:
    def __init__(
        self,
        proposed_grinds: list[float] | None = None,
        trust_config: TrustRegionConfig | None = None,
    ) -> None:
        self.proposed_grinds = list(proposed_grinds or [6.0, 7.0, 8.0])
        self.trust_config = trust_config or TrustRegionConfig()
        self.anchors: list[str] = []

    def suggest(self, *, run, recipes, shots, comparisons, state, now):
        anchor = (
            state.previous_valid_shot_id
            if run.comparison_mode == ComparisonMode.GLOBAL_PREVIOUS
            else state.incumbent_shot_id
        )
        self.anchors.append(anchor)
        grind = self.proposed_grinds.pop(0)
        recipe = RecipePoint.create(run.run_id, run.recipe_space, grind, 18.0, 36.0, created_at=now)
        suggestion = Suggestion(
            suggestion_id=f"suggestion_{state.iteration + 1}",
            optimization_run_id=run.run_id,
            recipe=recipe,
            anchor_shot_id=anchor,
            comparison_mode=run.comparison_mode,
            acquisition=AcquisitionDiagnostics(
                acquisition_value=0.1,
                unclipped_acquisition_value=0.1,
                outcome_probabilities={
                    "new_better": 0.4,
                    "tie": 0.2,
                    "anchor_better": 0.4,
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
                full_domain_proposal=run.comparison_mode == ComparisonMode.GLOBAL_PREVIOUS,
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
        return ModelRecommendation(
            run.run_id,
            recipe,
            "test",
            run.comparison_mode == ComparisonMode.BEST_INCUMBENT,
            state.incumbent_shot_id,
        )

    def update_trust_region_state(self, state, label, *, candidate_center):
        return update_trust_region(
            state,
            label,
            candidate_center=candidate_center,
            config=self.trust_config,
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
            config=self.trust_config,
            after_comparison_id=after_comparison_id,
            incumbent_shot_id=incumbent_shot_id,
            created_at=created_at,
            control_event_id=control_event_id,
        )


class CPBOApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = SQLiteStore(Path(self.temp.name) / "cpbo.db")
        self.repository = SQLitePreferentialOptimizationRepository(self.store)
        self.clock_value = 100

    def tearDown(self) -> None:
        self.store.close()
        self.temp.cleanup()

    def service(self, mode: ComparisonMode, engine: RecordingEngine | None = None):
        engine = engine or RecordingEngine()
        service = ConsecutivePreferenceOptimizationService(
            self.repository,
            engine,
            recipe_space_factory,
            random_seed=11,
            initial_trust_region_length=engine.trust_config.initial_length,
            clock=self.clock,
        )
        request = service.initialize(run_context(), baseline_recipe(), comparison_mode=mode)
        service.record_shot(
            request.optimization_run_id,
            baseline_recipe(),
            PhysicalShotStatus.VALID,
            shot_id="baseline",
            started_at=1,
            completed_at=2,
        )
        return service, engine, request.optimization_run_id

    def clock(self) -> int:
        self.clock_value += 1
        return self.clock_value

    def test_first_valid_shot_initializes_without_comparison(self) -> None:
        service, _, run_id = self.service(ComparisonMode.BEST_INCUMBENT)
        state = service.get_state(run_id)
        self.assertEqual(state.previous_valid_shot_id, "baseline")
        self.assertEqual(state.incumbent_shot_id, "baseline")
        self.assertEqual(self.repository.list_comparisons(run_id), [])

    def test_legacy_trust_region_state_decodes_without_restart_semantics(self) -> None:
        service, _, run_id = self.service(ComparisonMode.BEST_INCUMBENT)
        payload = json.loads(state_to_json(service.get_state(run_id)))
        trust = payload["trust_region_state"]
        trust.pop("locally_converged")
        trust.pop("transitions")

        decoded = state_from_json(json.dumps(payload))

        self.assertFalse(decoded.trust_region_state.locally_converged)
        self.assertEqual(decoded.trust_region_state.transitions, ())

    def test_global_mode_loss_still_advances_previous_anchor(self) -> None:
        service, engine, run_id = self.service(ComparisonMode.GLOBAL_PREVIOUS)
        suggestion = service.suggest_next(run_id)
        service.record_shot(
            run_id,
            suggestion.recipe,
            PhysicalShotStatus.VALID,
            shot_id="candidate_1",
            started_at=3,
            completed_at=4,
        )
        state = service.record_preference(
            run_id,
            "candidate_1",
            "baseline",
            PreferenceLabel.ANCHOR_BETTER,
        )
        self.assertEqual(state.previous_valid_shot_id, "candidate_1")
        service.suggest_next(run_id)
        self.assertEqual(engine.anchors, ["baseline", "candidate_1"])

    def test_best_mode_loss_and_tie_never_replace_incumbent(self) -> None:
        service, engine, run_id = self.service(ComparisonMode.BEST_INCUMBENT)
        for index, label in enumerate((PreferenceLabel.ANCHOR_BETTER, PreferenceLabel.TIE), start=1):
            suggestion = service.suggest_next(run_id)
            shot_id = f"candidate_{index}"
            service.record_shot(
                run_id,
                suggestion.recipe,
                PhysicalShotStatus.VALID,
                shot_id=shot_id,
                started_at=2 * index + 1,
                completed_at=2 * index + 2,
            )
            state = service.record_preference(run_id, shot_id, "baseline", label)
            self.assertEqual(state.incumbent_shot_id, "baseline")
        self.assertEqual(engine.anchors, ["baseline", "baseline"])
        trust = service.get_state(run_id).trust_region_state
        self.assertEqual(trust.failure_count, 0)
        self.assertEqual(trust.length, 0.4)

    def test_resume_preserves_evidence_incumbent_and_model_checkpoint(self) -> None:
        trust_config = TrustRegionConfig(initial_length=0.5**6)
        engine = RecordingEngine([6.0, 7.0, 8.0], trust_config)
        service, _, run_id = self.service(ComparisonMode.BEST_INCUMBENT, engine)
        for index in range(2):
            suggestion = service.suggest_next(run_id)
            shot_id = f"candidate_{index + 1}"
            service.record_shot(
                run_id,
                suggestion.recipe,
                PhysicalShotStatus.VALID,
                shot_id=shot_id,
                started_at=3 + index * 2,
                completed_at=4 + index * 2,
            )
            state = service.record_preference(
                run_id,
                shot_id,
                "baseline",
                PreferenceLabel.ANCHOR_BETTER,
            )

        self.assertTrue(state.trust_region_state.locally_converged)
        comparisons_before = self.repository.list_comparisons(run_id)
        checkpoint_before = state.model_checkpoint
        resumed = service.resume_local_exploration(
            run_id,
            control_event_id="resume_request_1",
        )
        self.assertFalse(resumed.trust_region_state.locally_converged)
        self.assertEqual(resumed.incumbent_shot_id, "baseline")
        self.assertEqual(resumed.model_checkpoint, checkpoint_before)
        self.assertEqual(self.repository.list_comparisons(run_id), comparisons_before)
        self.assertEqual(resumed.trust_region_state.length, trust_config.initial_length)
        self.assertEqual(
            resumed.trust_region_state.transitions[-1].after_comparison_id,
            comparisons_before[-1].comparison_id,
        )
        duplicate = service.resume_local_exploration(
            run_id,
            control_event_id="resume_request_1",
        )
        self.assertEqual(duplicate, resumed)
        with self.assertRaisesRegex(ValueError, "has not converged"):
            service.resume_local_exploration(
                run_id,
                control_event_id="resume_request_2",
            )
        self.assertIsNotNone(service.suggest_next(run_id))

    def test_best_mode_only_new_better_replaces_incumbent(self) -> None:
        service, _, run_id = self.service(ComparisonMode.BEST_INCUMBENT)
        suggestion = service.suggest_next(run_id)
        service.record_shot(
            run_id,
            suggestion.recipe,
            PhysicalShotStatus.VALID,
            shot_id="winner",
            started_at=3,
            completed_at=4,
        )
        state = service.record_preference(run_id, "winner", "baseline", PreferenceLabel.NEW_BETTER)
        self.assertEqual(state.incumbent_shot_id, "winner")
        self.assertEqual(state.trust_region_state.center, suggestion.recipe.normalized_x)

    def test_reversed_shot_ids_are_rejected(self) -> None:
        service, _, run_id = self.service(ComparisonMode.BEST_INCUMBENT)
        suggestion = service.suggest_next(run_id)
        service.record_shot(
            run_id,
            suggestion.recipe,
            PhysicalShotStatus.VALID,
            shot_id="candidate",
            started_at=3,
            completed_at=4,
        )
        with self.assertRaisesRegex(ValueError, "pending CPBO candidate"):
            service.record_preference(run_id, "baseline", "candidate", PreferenceLabel.NEW_BETTER)

    def test_failed_shot_creates_no_comparison_and_allows_another_suggestion(self) -> None:
        service, _, run_id = self.service(ComparisonMode.BEST_INCUMBENT)
        first = service.suggest_next(run_id)
        service.record_shot(
            run_id,
            first.recipe,
            PhysicalShotStatus.MACHINE_FAILURE,
            shot_id="failed",
            started_at=3,
            completed_at=4,
        )
        self.assertEqual(self.repository.list_comparisons(run_id), [])
        second = service.suggest_next(run_id)
        self.assertNotEqual(first.suggestion_id, second.suggestion_id)
        with self.assertRaisesRegex(ValueError, "pending CPBO candidate"):
            service.record_preference(run_id, "failed", "baseline", PreferenceLabel.TIE)

    def test_repeated_recipe_is_one_recipe_and_multiple_shots(self) -> None:
        engine = RecordingEngine([5.0])
        service, _, run_id = self.service(ComparisonMode.GLOBAL_PREVIOUS, engine)
        suggestion = service.suggest_next(run_id)
        self.assertEqual(suggestion.recipe.grind_size, baseline_recipe().relative_grind_steps_from_reference)
        service.record_shot(
            run_id,
            suggestion.recipe,
            PhysicalShotStatus.VALID,
            shot_id="repeat",
            started_at=3,
            completed_at=4,
        )
        service.record_preference(run_id, "repeat", "baseline", PreferenceLabel.TIE)
        self.assertEqual(len(self.repository.list_recipes(run_id)), 1)
        self.assertEqual(len(self.repository.list_shots(run_id)), 2)

    def test_contexts_are_isolated_in_persistence(self) -> None:
        service, _, first_run = self.service(ComparisonMode.BEST_INCUMBENT)
        second_request = service.initialize(
            replace(run_context(), bean_context_id="other_bean"),
            baseline_recipe(),
            comparison_mode=ComparisonMode.BEST_INCUMBENT,
        )
        self.assertNotEqual(first_run, second_request.optimization_run_id)

        sweet_context = replace(
            run_context(),
            taste_goal=TasteGoal.custom({"sweet": "high", "bitter": "low"}),
        )
        sweet_request = service.initialize(
            sweet_context,
            baseline_recipe(),
            comparison_mode=ComparisonMode.BEST_INCUMBENT,
        )
        self.assertNotEqual(first_run, sweet_request.optimization_run_id)
        self.assertEqual(service.active_run(sweet_context).run_id, sweet_request.optimization_run_id)
        self.assertEqual(service.active_run(run_context()).run_id, first_run)

    def test_changing_comparison_policy_reuses_the_active_run_evidence(self) -> None:
        service, _, first_run = self.service(ComparisonMode.BEST_INCUMBENT)
        suggestion = service.suggest_next(first_run)
        service.record_shot(
            first_run,
            suggestion.recipe,
            PhysicalShotStatus.VALID,
            shot_id="candidate_before_policy_change",
            started_at=3,
            completed_at=4,
        )
        service.record_preference(
            first_run,
            "candidate_before_policy_change",
            "baseline",
            PreferenceLabel.NEW_BETTER,
        )
        original_shots = self.repository.list_shots(first_run)
        original_comparisons = self.repository.list_comparisons(first_run)

        replacement = service.initialize(
            run_context(),
            baseline_recipe(),
            comparison_mode=ComparisonMode.GLOBAL_PREVIOUS,
        )

        self.assertEqual(replacement.optimization_run_id, first_run)
        self.assertTrue(self.repository.get_run(first_run).active)
        self.assertEqual(replacement.comparison_mode, ComparisonMode.GLOBAL_PREVIOUS)
        self.assertEqual(self.repository.list_shots(first_run), original_shots)
        self.assertEqual(self.repository.list_comparisons(first_run), original_comparisons)

    def test_policy_change_waits_for_pending_preference_without_losing_it(self) -> None:
        service, _, run_id = self.service(ComparisonMode.BEST_INCUMBENT)
        suggestion = service.suggest_next(run_id)
        service.record_shot(
            run_id,
            suggestion.recipe,
            PhysicalShotStatus.VALID,
            shot_id="candidate_before_policy_change",
            started_at=3,
            completed_at=4,
        )

        pending_run = service.active_run(
            run_context(),
            comparison_mode=ComparisonMode.GLOBAL_PREVIOUS,
        )
        self.assertEqual(pending_run.run_id, run_id)
        self.assertEqual(pending_run.comparison_mode, ComparisonMode.BEST_INCUMBENT)

        service.record_preference(
            run_id,
            "candidate_before_policy_change",
            "baseline",
            PreferenceLabel.TIE,
        )
        reconfigured_run = service.active_run(
            run_context(),
            comparison_mode=ComparisonMode.GLOBAL_PREVIOUS,
        )

        self.assertEqual(reconfigured_run.run_id, run_id)
        self.assertEqual(reconfigured_run.comparison_mode, ComparisonMode.GLOBAL_PREVIOUS)
        comparison = self.repository.list_comparisons(run_id)[0]
        self.assertEqual(comparison.comparison_mode, ComparisonMode.BEST_INCUMBENT)

    def test_initialize_resumes_with_a_new_suggestion_not_the_incumbent(self) -> None:
        service, engine, run_id = self.service(ComparisonMode.BEST_INCUMBENT)

        resumed = service.initialize(
            run_context(),
            baseline_recipe(),
            comparison_mode=ComparisonMode.BEST_INCUMBENT,
        )

        self.assertEqual(resumed.optimization_run_id, run_id)
        self.assertFalse(resumed.is_baseline)
        self.assertEqual(resumed.anchor_shot_id, "baseline")
        self.assertEqual(engine.anchors, ["baseline"])
        self.assertEqual(
            self.repository.get_state(run_id).pending_recipe_id,
            resumed.recipe.recipe_id,
        )

    def test_legacy_unscoped_run_is_migrated_to_balanced_goal(self) -> None:
        service, _, run_id = self.service(ComparisonMode.BEST_INCUMBENT)
        row = self.store.conn.execute(
            "SELECT payload_json FROM cpbo_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        payload = json.loads(row["payload_json"])
        legacy_context = payload["context"]
        legacy_context.pop("taste_goal")
        legacy_fingerprint = hashlib.sha256(
            json.dumps(legacy_context, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.store.conn.execute(
            "UPDATE cpbo_runs SET context_fingerprint=?, payload_json=? WHERE run_id=?",
            (legacy_fingerprint, json.dumps(payload, sort_keys=True, separators=(",", ":")), run_id),
        )
        self.store.conn.commit()

        migrated = service.active_run(run_context())

        self.assertEqual(migrated.run_id, run_id)
        stored = self.store.conn.execute(
            "SELECT context_fingerprint, payload_json FROM cpbo_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        self.assertEqual(stored["context_fingerprint"], run_context().fingerprint)
        self.assertEqual(
            json.loads(stored["payload_json"])["context"]["taste_goal"]["mode"],
            "balanced",
        )

    def test_legacy_profile_hash_fragments_choose_the_newest_run(self) -> None:
        service, _, older_run_id = self.service(ComparisonMode.BEST_INCUMBENT)
        older_row = self.store.conn.execute(
            "SELECT payload_json FROM cpbo_runs WHERE run_id=?",
            (older_run_id,),
        ).fetchone()
        older_payload = json.loads(older_row["payload_json"])
        older_payload["context"]["raw_profile_hash"] = "a" * 64
        older_fingerprint = hashlib.sha256(
            json.dumps(
                older_payload["context"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.store.conn.execute(
            "UPDATE cpbo_runs SET context_fingerprint=?, payload_json=? WHERE run_id=?",
            (
                older_fingerprint,
                json.dumps(older_payload, sort_keys=True, separators=(",", ":")),
                older_run_id,
            ),
        )

        newer_run_id = "run_newer_fragment"
        newer_payload = dict(older_payload)
        newer_payload["run_id"] = newer_run_id
        newer_payload["created_at"] = older_payload["created_at"] + 1
        newer_payload["context"] = dict(older_payload["context"])
        newer_payload["context"]["raw_profile_hash"] = "b" * 64
        newer_fingerprint = hashlib.sha256(
            json.dumps(
                newer_payload["context"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.store.conn.execute(
            """
            INSERT INTO cpbo_runs (
                run_id, context_fingerprint, install_id, machine_id,
                active, created_at, payload_json
            ) VALUES (?, ?, ?, ?, 1, ?, ?)
            """,
            (
                newer_run_id,
                newer_fingerprint,
                "install",
                "machine",
                newer_payload["created_at"],
                json.dumps(newer_payload, sort_keys=True, separators=(",", ":")),
            ),
        )
        self.store.conn.commit()

        migrated = self.repository.find_active_run(run_context())

        self.assertEqual(migrated.run_id, newer_run_id)
        rows = self.store.conn.execute(
            "SELECT run_id, context_fingerprint, active, payload_json FROM cpbo_runs ORDER BY run_id"
        ).fetchall()
        by_id = {row["run_id"]: row for row in rows}
        self.assertEqual(by_id[older_run_id]["active"], 0)
        self.assertEqual(by_id[newer_run_id]["active"], 1)
        self.assertEqual(
            by_id[newer_run_id]["context_fingerprint"],
            run_context().fingerprint,
        )
        self.assertIsNone(
            json.loads(by_id[newer_run_id]["payload_json"])["context"]["raw_profile_hash"]
        )

    def test_initialize_does_not_request_an_already_pulled_candidate_again(self) -> None:
        service, _, run_id = self.service(ComparisonMode.BEST_INCUMBENT)
        suggestion = service.suggest_next(run_id)
        service.record_shot(
            run_id,
            suggestion.recipe,
            PhysicalShotStatus.VALID,
            shot_id="candidate",
            started_at=3,
            completed_at=4,
        )

        with self.assertRaisesRegex(ValueError, "awaiting preference"):
            service.initialize(
                run_context(),
                baseline_recipe(),
                comparison_mode=ComparisonMode.BEST_INCUMBENT,
            )

    def test_configuration_change_refits_the_active_run_without_losing_history(self) -> None:
        first_service = ConsecutivePreferenceOptimizationService(
            self.repository,
            RecordingEngine(),
            recipe_space_factory,
            random_seed=11,
            configuration_version="config:v1",
            clock=self.clock,
        )
        first = first_service.initialize(
            run_context(),
            baseline_recipe(),
            comparison_mode=ComparisonMode.BEST_INCUMBENT,
        )
        first_recipe = self.repository.list_recipes(first.optimization_run_id)[0]

        second_service = ConsecutivePreferenceOptimizationService(
            self.repository,
            RecordingEngine(),
            recipe_space_factory,
            random_seed=11,
            configuration_version="config:v2",
            clock=self.clock,
        )
        active = second_service.active_run(run_context())
        self.assertEqual(active.run_id, first.optimization_run_id)
        self.assertTrue(self.repository.get_run(first.optimization_run_id).active)
        second = second_service.initialize(
            run_context(),
            baseline_recipe(),
            comparison_mode=ComparisonMode.BEST_INCUMBENT,
        )

        self.assertEqual(first.optimization_run_id, second.optimization_run_id)
        self.assertEqual(self.repository.get_recipe(first_recipe.recipe_id), first_recipe)
        self.assertEqual(
            self.repository.get_run(second.optimization_run_id).configuration_version,
            "config:v2",
        )

    def test_recipe_domain_change_migrates_active_evidence_in_place(self) -> None:
        service, _, run_id = self.service(
            ComparisonMode.BEST_INCUMBENT,
            RecordingEngine([6.0, 7.0]),
        )
        first = service.suggest_next(run_id)
        service.record_shot(
            run_id,
            first.recipe,
            PhysicalShotStatus.VALID,
            shot_id="winner",
            started_at=3,
            completed_at=4,
        )
        service.record_preference(
            run_id,
            "winner",
            "baseline",
            PreferenceLabel.NEW_BETTER,
        )
        stale = service.suggest_next(run_id)
        original_shots = self.repository.list_shots(run_id)
        original_comparisons = self.repository.list_comparisons(run_id)
        original_recipe_ids = {shot.recipe_id for shot in original_shots}
        original_state = self.repository.get_state(run_id)
        new_domain = RecipeDomain(
            grind_radius_steps=2.0,
            dose_min_g=19.0,
            dose_max_g=23.0,
            target_output_min_g=40.0,
            target_output_max_g=60.0,
        )
        replacement = ConsecutivePreferenceOptimizationService(
            self.repository,
            RecordingEngine([6.5]),
            recipe_space_factory,
            random_seed=19,
            configuration_version="config:v2",
            recipe_domain=new_domain,
            clock=self.clock,
        )

        migrated = replacement.active_run(
            run_context(),
            comparison_mode=ComparisonMode.BEST_INCUMBENT,
        )

        self.assertEqual(migrated.run_id, run_id)
        self.assertTrue(migrated.active)
        self.assertEqual(migrated.recipe_space.version, new_domain.effective_version)
        self.assertEqual(migrated.configuration_version, "config:v2")
        migrated_shots = self.repository.list_shots(run_id)
        self.assertEqual(
            [shot.shot_id for shot in migrated_shots],
            [shot.shot_id for shot in original_shots],
        )
        self.assertTrue(all(shot.observed_recipe is not None for shot in migrated_shots))
        self.assertTrue(
            original_recipe_ids.isdisjoint({shot.recipe_id for shot in migrated_shots})
        )
        self.assertEqual(self.repository.list_comparisons(run_id), original_comparisons)
        winner = self.repository.get_shot("winner")
        winner_recipe = self.repository.get_recipe(winner.recipe_id)
        self.assertEqual(
            (
                winner_recipe.grind_size,
                winner_recipe.dose_g,
                winner_recipe.target_output_g,
            ),
            (6.0, 18.0, 36.0),
        )
        self.assertEqual(winner_recipe.normalized_x, (0.75, -0.25, -0.2))
        self.assertFalse(winner_recipe.inside_search_space)
        state = self.repository.get_state(run_id)
        self.assertEqual(state.previous_valid_shot_id, "winner")
        self.assertEqual(state.incumbent_shot_id, "winner")
        self.assertEqual(state.iteration, original_state.iteration)
        self.assertEqual(state.trust_region_state.center, (0.75, 0.0, 0.0))
        self.assertEqual(state.random_seed, 19)
        self.assertEqual(state.configuration_version, "config:v2")
        self.assertIsNone(state.pending_recipe_id)
        self.assertIsNone(state.model_checkpoint)
        stale_status = self.store.conn.execute(
            "SELECT status FROM cpbo_suggestions WHERE suggestion_id=?",
            (stale.suggestion_id,),
        ).fetchone()["status"]
        self.assertEqual(stale_status, "superseded")
        stored_recipe_ids = {
            recipe.recipe_id for recipe in self.repository.list_recipes(run_id)
        }
        self.assertTrue(original_recipe_ids.issubset(stored_recipe_ids))

    def test_recipe_domain_change_waits_for_pending_preference(self) -> None:
        service, _, run_id = self.service(
            ComparisonMode.BEST_INCUMBENT,
            RecordingEngine([6.0]),
        )
        suggestion = service.suggest_next(run_id)
        service.record_shot(
            run_id,
            suggestion.recipe,
            PhysicalShotStatus.VALID,
            shot_id="pending_candidate",
            started_at=3,
            completed_at=4,
        )
        old_version = self.repository.get_run(run_id).recipe_space.version
        new_domain = RecipeDomain(
            grind_radius_steps=3.0,
            dose_min_g=16.0,
            dose_max_g=24.0,
            target_output_min_g=30.0,
            target_output_max_g=70.0,
        )
        replacement = ConsecutivePreferenceOptimizationService(
            self.repository,
            RecordingEngine([7.0]),
            recipe_space_factory,
            random_seed=11,
            recipe_domain=new_domain,
            clock=self.clock,
        )

        deferred = replacement.active_run(run_context())

        self.assertEqual(deferred.recipe_space.version, old_version)
        self.assertEqual(
            self.repository.get_state(run_id).pending_shot_id,
            "pending_candidate",
        )
        replacement.record_preference(
            run_id,
            "pending_candidate",
            "baseline",
            PreferenceLabel.TIE,
        )
        migrated = replacement.active_run(run_context())
        self.assertEqual(migrated.run_id, run_id)
        self.assertEqual(migrated.recipe_space.version, new_domain.effective_version)
        self.assertEqual(len(self.repository.list_comparisons(run_id)), 1)
        self.assertEqual(
            self.repository.list_comparisons(run_id)[0].label,
            PreferenceLabel.TIE,
        )

    def test_repository_rejects_partial_recipe_space_migration_atomically(self) -> None:
        service, _, run_id = self.service(ComparisonMode.BEST_INCUMBENT)
        original_run = self.repository.get_run(run_id)
        original_state = self.repository.get_state(run_id)
        replacement_run = replace(
            original_run,
            recipe_space=original_run.recipe_space.with_domain(
                RecipeDomain(
                    grind_radius_steps=3.0,
                    dose_min_g=16.0,
                    dose_max_g=24.0,
                    target_output_min_g=30.0,
                    target_output_max_g=70.0,
                )
            ),
        )

        with self.assertRaisesRegex(ValueError, "requires physical shots"):
            self.repository.migrate_run_recipe_space(
                replacement_run,
                (),
                (),
                original_state,
            )

        self.assertEqual(self.repository.get_run(run_id), original_run)
        self.assertEqual(self.repository.get_state(run_id), original_state)

    def test_state_survives_repository_reconstruction(self) -> None:
        service, _, run_id = self.service(ComparisonMode.BEST_INCUMBENT)
        suggestion = service.suggest_next(run_id)
        reconstructed = SQLitePreferentialOptimizationRepository(self.store)
        self.assertEqual(reconstructed.get_state(run_id).pending_recipe_id, suggestion.recipe.recipe_id)
        self.assertEqual(reconstructed.get_pending_suggestion(run_id).suggestion_id, suggestion.suggestion_id)

    def test_shot_payload_without_observed_recipe_remains_readable(self) -> None:
        shot = shot_from_json(
            json.dumps(
                {
                    "shot_id": "legacy-shot",
                    "recipe_id": "legacy-recipe",
                    "optimization_run_id": "legacy-run",
                    "sequence_number": 1,
                    "started_at": 10,
                    "completed_at": 20,
                    "status": "valid",
                    "telemetry_available": False,
                    "raw_telemetry_reference": None,
                    "trace_feature_names": [],
                    "trace_features": None,
                    "metadata": {},
                }
            )
        )

        self.assertIsNone(shot.observed_recipe)

    def test_reset_owner_removes_cpbo_records(self) -> None:
        _, _, run_id = self.service(ComparisonMode.BEST_INCUMBENT)
        counts = self.repository.reset_owner("install", "machine")
        self.assertEqual(counts["runs"], 1)
        self.assertIsNone(self.repository.get_run(run_id))


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
        RecipeParameter("dose_g", 14.0, 22.0, 0.1, "g"),
        RecipeParameter("target_output_g", 20.0, 60.0, 0.1, "g"),
        recipe.grinder_step_direction,
    )


def run_context() -> OptimizationRunContext:
    return OptimizationRunContext(
        install_id="install",
        machine_id="machine",
        bean_context_id="bean",
        grinder_context_id="grinder",
        profile_id="profile",
        basket_id="basket",
        water_id="water",
        user_id="user",
    )


if __name__ == "__main__":
    unittest.main()

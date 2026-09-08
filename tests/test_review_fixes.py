from __future__ import annotations

import unittest
from dataclasses import fields, replace
from unittest.mock import patch

import test_cpbo_runtime_bridge as runtime_fixtures
from test_cpbo_runtime_bridge import _shot
import test_gaggimate_adapter as mqtt_fixtures
from espresso_rl.adapters.cpbo_serialization import run_from_json, run_to_json
from espresso_rl.application.cpbo_runtime import strict_context_from_shot
from espresso_rl.config import Config
from espresso_rl.domain.cpbo import ComparisonMode, PreferenceLabel
from espresso_rl.domain.events import PreferenceFeedbackEvent
from espresso_rl.domain.models import ShotRecord


class ReviewRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = runtime_fixtures.CPBORuntimeBridgeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def prepare(self, mode=ComparisonMode.BEST_INCUMBENT):
        comparisons = []
        bridge = self.fixture.bridge([6., 7., 8.], comparisons.append)
        bridge._comparison_mode = mode
        baseline = _shot("baseline", grind=5.)
        baseline.profile_label = "Morning espresso"
        self.fixture.shots.rows[baseline.shot_id] = baseline
        first = bridge.handle_shot(baseline)
        candidate = _shot("candidate", grind=6.)
        self.fixture.shots.rows[candidate.shot_id] = candidate
        outcome = bridge.handle_shot(candidate)
        event = PreferenceFeedbackEvent(
            optimization_run_id=first.optimization_run_id,
            new_shot_id=candidate.shot_id, anchor_shot_id=baseline.shot_id,
            label=None, abstained=True, install_id=baseline.install_id,
            machine_id=baseline.machine_id, timestamp=200, comparison_mode=mode,
        )
        return bridge, event, outcome, comparisons

    def test_abstention_keeps_physical_shot_without_label_or_trust_region_update(self):
        bridge, event, outcome, uploads = self.prepare()
        before = bridge._optimizer.get_state(event.optimization_run_id)
        recommendation = bridge.handle_preference(event)
        after = bridge._optimizer.get_state(event.optimization_run_id)
        self.assertEqual(after.incumbent_shot_id, before.incumbent_shot_id)
        self.assertEqual(after.previous_valid_shot_id, before.previous_valid_shot_id)
        self.assertEqual(after.trust_region_state, before.trust_region_state)
        self.assertIsNone(after.pending_shot_id)
        self.assertEqual(recommendation.comparison_anchor_shot_id, "baseline")
        self.assertEqual(self.fixture.repository.list_comparisons(event.optimization_run_id), [])
        self.assertEqual(uploads, [])
        stored = self.fixture.repository.get_shot("candidate")
        self.assertEqual(stored.status.value, "valid")
        self.assertEqual(stored.metadata["preference_abstention"]["anchor_shot_id"], "baseline")
        self.assertEqual(bridge.handle_preference(event).recommendation_id, recommendation.recommendation_id)
        # Recreating application services must preserve the resolution.
        restarted = self.fixture.bridge([8.])
        self.assertFalse(restarted.handle_shot(self.fixture.shots.rows["candidate"]).awaiting_preference)
        with self.assertRaises(ValueError):
            bridge.handle_preference(replace(event, label=PreferenceLabel.TIE, abstained=False))
        self.assertIsNone(bridge.handle_shot(self.fixture.shots.rows["candidate"]).preference_request)

    def test_global_previous_abstention_keeps_last_comparable_anchor(self):
        bridge, event, _, _ = self.prepare(ComparisonMode.GLOBAL_PREVIOUS)
        recommendation = bridge.handle_preference(event)
        self.assertEqual(recommendation.comparison_anchor_shot_id, "baseline")
        next_shot = _shot("next", grind=7.)
        self.fixture.shots.rows["next"] = next_shot
        self.assertEqual(bridge.handle_shot(next_shot).preference_request.anchor_shot_id, "baseline")

    def test_abstention_rejects_wrong_owner_anchor_and_label(self):
        bridge, event, _, _ = self.prepare()
        for changes in ({"install_id": "another"}, {"machine_id": "another"}, {"anchor_shot_id": "unknown"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                bridge.handle_preference(replace(event, **changes))
        for changes in ({"label": PreferenceLabel.TIE}, {"abstained": False}, {"abstained": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(event, **changes)
        self.assertEqual(bridge._optimizer.get_state(event.optimization_run_id).pending_shot_id, "candidate")

    def test_abstention_retry_recovers_after_suggestion_failure(self):
        bridge, event, _, _ = self.prepare()
        with patch.object(bridge._optimizer._optimizer, "suggest", side_effect=RuntimeError("fit failed")):
            with self.assertRaises(RuntimeError):
                bridge.handle_preference(event)
        recommendation = bridge.handle_preference(event)
        self.assertIsNotNone(recommendation)
        self.assertEqual(self.fixture.repository.list_comparisons(event.optimization_run_id), [])

    def test_labeled_feedback_retry_recovers_without_duplicate_comparison(self):
        bridge, event, _, _ = self.prepare()
        event = replace(event, abstained=False, label=PreferenceLabel.NEW_BETTER)
        with patch.object(bridge._optimizer._optimizer, "suggest", side_effect=RuntimeError("fit failed")):
            with self.assertRaises(RuntimeError):
                bridge.handle_preference(event)
        recommendation = bridge.handle_preference(event)
        self.assertIsNotNone(recommendation)
        self.assertEqual(len(self.fixture.repository.list_comparisons(event.optimization_run_id)), 1)

    def test_correction_after_abstention_does_not_reopen_comparison(self):
        bridge, event, _, _ = self.prepare()
        bridge.handle_preference(event)
        corrected = _shot("candidate", grind=7.)
        self.fixture.shots.rows[corrected.shot_id] = corrected
        result = bridge.handle_shot_correction(corrected)
        self.assertFalse(result.awaiting_preference)
        self.assertIsNone(bridge._optimizer.get_state(event.optimization_run_id).pending_shot_id)
        self.assertEqual(self.fixture.repository.list_comparisons(event.optimization_run_id), [])
        self.assertIn("preference_abstention", self.fixture.repository.get_shot("candidate").metadata)

    def test_temperature_is_a_variable_within_the_same_profile_context(self):
        first = _shot("first", grind=5.)
        warmer = replace(first, shot_id="warmer", profile_temperature_c=96., final_phase_temperature_c=96.)
        self.assertEqual(strict_context_from_shot(first), strict_context_from_shot(warmer))

    def test_anchor_details_describe_stored_physical_shot(self):
        _, _, outcome, _ = self.prepare()
        anchor = outcome.preference_request.anchor
        self.assertEqual(anchor.timestamp, 100)
        self.assertEqual(anchor.relative_grind_steps_from_reference, 5.)
        self.assertEqual(anchor.dose_g, 18.)
        self.assertEqual(anchor.target_yield_g, 36.)
        self.assertEqual(anchor.beverage_out_g, 35.5)
        self.assertEqual(anchor.profile_label, "Morning espresso")
        for changes in ({"timestamp": True}, {"dose_g": 0}, {"dose_g": 101}, {"target_yield_g": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(anchor, **changes)

    def test_synthetic_basket_migration_preserves_run_and_evidence(self):
        bridge, event, _, _ = self.prepare()
        original = bridge._optimizer.get_run(event.optimization_run_id)
        old = replace(original, context=replace(original.context, basket_id="basket_ml:18"))
        with self.fixture.store.conn:
            self.fixture.store.conn.execute(
                "UPDATE cpbo_runs SET context_fingerprint=?, payload_json=? WHERE run_id=?",
                (old.context.fingerprint, run_to_json(old), old.run_id),
            )
        migrated = self.fixture.repository.find_active_run(original.context)
        self.assertEqual(migrated.run_id, original.run_id)
        self.assertIsNone(migrated.context.basket_id)
        self.assertEqual(len(self.fixture.repository.list_shots(original.run_id)), 2)
        self.assertEqual(bridge._optimizer.get_state(original.run_id).pending_shot_id, "candidate")
        real = replace(original, context=replace(original.context, basket_id="basket-real-18g"))
        self.assertEqual(run_from_json(run_to_json(real)).context.basket_id, "basket-real-18g")
        self.assertNotIn("basket_size_ml", {f.name for f in fields(Config)})
        self.assertNotIn("basket_size_ml", {f.name for f in fields(ShotRecord)})
        self.assertNotIn("machine_pressure_bar", {f.name for f in fields(Config)})
        self.assertNotIn("grinder_model", {f.name for f in fields(Config)})


class ReviewWireTests(unittest.TestCase):
    def test_abstain_wire_value_is_not_a_preference_label(self):
        fixture = mqtt_fixtures.GaggimateAdapterTests()
        fixture.setUp()
        payload = dict(event_type="preference_feedback", schema_version=1,
            optimization_run_id="run", new_shot_id="new", anchor_shot_id="anchor", label="abstain",
            comparison_mode="best_incumbent", install_id="install", machine_id="gaggimate:AA_BB",
            timestamp=200, source="webui", taste_goal={"schema_version": 1, "mode": "balanced", "targets": {}})
        event = fixture.client.translate_preference_payload(payload, "AA_BB")
        self.assertTrue(event.abstained)
        self.assertIsNone(event.label)
        with self.assertRaises(ValueError):
            fixture.client.translate_preference_payload({**payload, "label": "skip_maybe"}, "AA_BB")


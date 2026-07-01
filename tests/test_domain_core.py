from __future__ import annotations

import unittest

import numpy as np

from espresso_rl.domain.events import OptimizerSettingsEvent, ShotFeedbackEvent, ShotProfileEvent
from espresso_rl.domain.follow_through import infer_follow_through
from espresso_rl.domain.models import (
    FollowThroughState,
    Recipe,
    Recommendation,
    RecommendationDecision,
    RecommendationMode,
    ShotRecord,
)
from espresso_rl.domain.profile import (
    build_fixed_cadence_sequence,
    profile_mse,
    profile_score,
    resample_profile,
    resample_profile_with_quality,
    resample_shot_metadata,
)
from espresso_rl.domain.reward import compute_reward
from espresso_rl.domain.staleness import check_recommendation_staleness


def event(**overrides) -> ShotProfileEvent:
    base = {
        "shot_id": "shot_1",
        "install_id": "install_1",
        "machine_id": "machine_1",
        "machine_adapter": "gaggimate",
        "timestamp": 1000,
        "time_ms": [0, 500, 1000],
        "pressure": [0.0, 8.0, 9.0],
        "target_pressure": [0.0, 8.0, 9.0],
        "pump_flow": [0.0, 2.0, 2.0],
        "target_flow": [0.0, 2.0, 2.0],
        "beverage_flow": [0.0, 1.8, 2.0],
        "weight": [0.0, 10.0, 36.0],
        "microns_per_step": 12.5,
        "relative_grind_steps_from_reference": 42,
        "dose_in_g": 18.0,
        "target_yield_g": 36.0,
        "beverage_out_g": 36.0,
        "shot_time_s": 30.0,
    }
    base.update(overrides)
    return ShotProfileEvent(**base)


class DomainCoreTests(unittest.TestCase):
    def test_optimizer_settings_event_normalizes_safe_modes(self) -> None:
        event = OptimizerSettingsEvent(
            install_id="install_1",
            machine_id="machine_1",
            timestamp=1,
            optimizer_mode="bo",
        )

        self.assertEqual(event.optimizer_mode, "bayesian_optimization")
        self.assertEqual(event.taste_objective, {"mode": "auto"})

        custom = OptimizerSettingsEvent(
            install_id="install_1",
            machine_id="machine_1",
            timestamp=1,
            taste_objective={"mode": "custom", "sweetness": "high", "clarity": "unspecified"},
        )
        self.assertEqual(custom.taste_objective, {"mode": "custom", "sweetness": "high"})

    def test_optimizer_settings_event_rejects_invalid_mode_and_digest(self) -> None:
        with self.assertRaises(ValueError):
            OptimizerSettingsEvent(
                install_id="install_1",
                machine_id="machine_1",
                timestamp=1,
                optimizer_mode="remote_exec",
            )
        with self.assertRaises(ValueError):
            OptimizerSettingsEvent(
                install_id="install_1",
                machine_id="machine_1",
                timestamp=1,
                optimizer_mode="bayesian_optimization",
                model_artifact_sha256="not-a-digest",
            )
        with self.assertRaisesRegex(ValueError, "custom mode requires"):
            OptimizerSettingsEvent(
                install_id="install_1",
                machine_id="machine_1",
                timestamp=1,
                taste_objective={"mode": "custom"},
            )

    def test_feedback_requires_rating_or_explicit_skip(self) -> None:
        base = {
            "shot_id": "shot_1",
            "install_id": "install_1",
            "machine_id": "machine_1",
            "timestamp": 1,
        }
        with self.assertRaisesRegex(ValueError, "rating is required"):
            ShotFeedbackEvent(**base)
        with self.assertRaisesRegex(ValueError, "skipped feedback cannot include"):
            ShotFeedbackEvent(**base, skipped=True, rating=4)

    def test_canonical_profile_requires_aligned_arrays(self) -> None:
        with self.assertRaises(ValueError):
            event(pump_flow=[0.0, 1.0])

    def test_canonical_profile_rejects_unbounded_sample_count(self) -> None:
        samples = [0.0] * 501
        with self.assertRaisesRegex(ValueError, "must not exceed 500"):
            event(
                time_ms=list(range(501)),
                pressure=samples,
                target_pressure=samples,
                pump_flow=samples,
                target_flow=samples,
                beverage_flow=samples,
                weight=samples,
            )

    def test_canonical_profile_rejects_nonfinite_profile_values(self) -> None:
        with self.assertRaises(ValueError):
            event(pump_flow=[0.0, float("nan"), 2.0])

    def test_canonical_profile_rejects_nonfinite_scalar_values(self) -> None:
        with self.assertRaises(ValueError):
            event(dose_in_g=float("inf"))

    def test_canonical_profile_rejects_boolean_numeric_values(self) -> None:
        with self.assertRaises(ValueError):
            event(pump_flow=[0.0, True, 2.0])
        with self.assertRaisesRegex(ValueError, "valve_open"):
            event(valve_open=[False, 1, True])

    def test_canonical_profile_accepts_sanitized_execution_metadata(self) -> None:
        parsed = event(
            profile_id="profile_1",
            profile_label="Cremina lever machine",
            profile_phase_count=5,
            final_phase_index=3,
            final_phase_name="ramp",
            final_phase_type="brew",
            final_phase_elapsed_s=8.5,
            final_pump_target="pressure",
            final_target_pressure=9.0,
            final_target_flow=0.0,
            final_valve_open=True,
            profile_temperature_c=86.5,
            final_phase_temperature_c=86.5,
            shot_end_state="manual_or_interrupted",
        )

        self.assertEqual(parsed.final_phase_name, "ramp")
        self.assertEqual(parsed.final_pump_target, "pressure")
        self.assertEqual(parsed.shot_end_state, "manual_or_interrupted")

    def test_canonical_profile_rejects_invalid_execution_metadata(self) -> None:
        with self.assertRaises(ValueError):
            event(final_phase_type="steam")
        with self.assertRaises(ValueError):
            event(final_target_pressure=99.0)
        with self.assertRaises(ValueError):
            event(profile_phase_count=1.5)

    def test_profile_resamples_to_fixed_shape(self) -> None:
        profile = resample_profile(event())
        self.assertEqual(profile.shape, (5, 100))
        self.assertEqual(profile.dtype, np.float32)
        self.assertAlmostEqual(float(profile[4, -1]), 36.0)

    def test_scalar_temperature_metadata_is_not_fabricated_as_live_telemetry(self) -> None:
        metadata = resample_shot_metadata(
            event(
                profile_temperature_c=93.0,
                final_phase_temperature_c=92.0,
            )
        )

        self.assertIsNone(metadata.temperature_profile)
        self.assertIsNone(metadata.target_temperature_profile)
        self.assertIsNone(metadata.pump_target_mode_profile)
        self.assertIsNotNone(metadata.beverage_flow_profile)

    def test_sampled_temperature_and_pump_mode_are_resampled(self) -> None:
        metadata = resample_shot_metadata(
            event(
                temperature=[91.0, 92.0, 93.0],
                target_temperature=[93.0, 93.0, 92.0],
                pump_target_mode=[1, 1, 2],
            )
        )

        self.assertEqual(metadata.temperature_profile.shape, (100,))  # type: ignore[union-attr]
        self.assertAlmostEqual(float(metadata.beverage_flow_profile[-1]), 2.0)  # type: ignore[index]
        self.assertAlmostEqual(float(metadata.temperature_profile[-1]), 93.0)  # type: ignore[index]
        self.assertAlmostEqual(float(metadata.target_temperature_profile[-1]), 92.0)  # type: ignore[index]
        self.assertEqual(int(metadata.pump_target_mode_profile[-1]), 2)  # type: ignore[index]

    def test_fixed_cadence_sequence_uses_exact_250ms_transitions(self) -> None:
        sequence = build_fixed_cadence_sequence(
            event(
                time_ms=[240, 510, 745, 1010],
                pressure=[0.0, 2.0, 4.0, 6.0],
                target_pressure=[1.0, 2.0, 3.0, 4.0],
                pump_flow=[0.0, 1.0, 2.0, 3.0],
                target_flow=[0.0, 1.5, 2.0, 2.5],
                beverage_flow=[0.0, 0.5, 1.0, 1.5],
                weight=[0.0, 1.0, 3.0, 6.0],
                temperature=[90.0, 90.5, 91.0, 91.5],
                target_temperature=[92.0, 92.0, 92.5, 92.5],
                pump_target_mode=[1, 1, 2, 2],
                valve_open=[False, True, True, False],
            )
        )

        self.assertIsNotNone(sequence)
        self.assertEqual(sequence.sample_interval_ms, 250)  # type: ignore[union-attr]
        self.assertEqual(sequence.step_count, 4)  # type: ignore[union-attr]
        self.assertEqual(sequence.pump_target_mode.tolist(), [1, 1, 1, 2])  # type: ignore[union-attr]
        self.assertEqual(sequence.valve_open.tolist(), [0, 0, 1, 1])  # type: ignore[union-attr]
        self.assertAlmostEqual(float(sequence.pressure_bar[-1]), 5.8491, places=3)  # type: ignore[union-attr]

    def test_fixed_cadence_sequence_requires_complete_control_telemetry(self) -> None:
        self.assertIsNone(build_fixed_cadence_sequence(event()))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            build_fixed_cadence_sequence(
                event(
                    time_ms=[0, 500, 400],
                    temperature=[90.0, 91.0, 92.0],
                    target_temperature=[92.0, 92.0, 92.0],
                    pump_target_mode=[1, 1, 1],
                    valve_open=[True, True, True],
                )
            )

    def test_profile_mse_ignores_inactive_zero_flow_target(self) -> None:
        profile = resample_profile(
            event(pump_flow=[100_000.0, 100_000.0, 100_000.0], target_flow=[0.0, 0.0, 0.0])
        )

        self.assertEqual(profile_mse(profile), 0.0)
        self.assertEqual(profile_score(profile), 1.0)

    def test_profile_mse_uses_only_active_valid_target_channels(self) -> None:
        profile = resample_profile(
            event(
                pressure=[3.0, 3.0, 3.0],
                target_pressure=[0.0, 0.0, 0.0],
                pump_flow=[1.0, 2.0, 3.0],
                target_flow=[1.0, 2.0, 3.0],
            )
        )

        self.assertEqual(profile_mse(profile), 0.0)

    def test_sanitized_profile_masks_invalid_active_flow_pair(self) -> None:
        quality = resample_profile_with_quality(
            event(pump_flow=[100_000.0, 100_000.0, 100_000.0], target_flow=[2.0, 2.0, 2.0])
        )

        self.assertFalse(quality.flow_valid)
        self.assertTrue(quality.flow_masked)
        self.assertEqual(float(quality.profile[2].max()), 0.0)
        self.assertEqual(float(quality.profile[3].max()), 0.0)
        self.assertEqual(profile_mse(quality.profile), 0.0)

    def test_sanitized_profile_masks_invalid_target_flow(self) -> None:
        quality = resample_profile_with_quality(
            event(pump_flow=[1.0, 2.0, 2.0], target_flow=[100_000.0, 100_000.0, 100_000.0])
        )

        self.assertTrue(quality.flow_valid)
        self.assertTrue(quality.flow_masked)
        self.assertEqual(float(quality.profile[2].max()), 0.0)
        self.assertEqual(float(quality.profile[3].max()), 0.0)

    def test_uncalibrated_pump_flow_is_masked_without_discarding_beverage_flow(self) -> None:
        source = event(pump_flow_calibration_required=True)
        quality = resample_profile_with_quality(source)
        metadata = resample_shot_metadata(source)

        self.assertFalse(quality.flow_valid)
        self.assertTrue(quality.flow_masked)
        self.assertEqual(float(quality.profile[2].max()), 0.0)
        self.assertEqual(float(quality.profile[3].max()), 0.0)
        self.assertAlmostEqual(float(metadata.beverage_flow_profile[-1]), 2.0)  # type: ignore[index]

    def test_ignored_recommendation_is_not_followed(self) -> None:
        rec = Recommendation(
            recommendation_id="rec_1",
            created_at=1,
            updated_at=1,
            expires_at=None,
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id=None,
            grind_delta_steps_from_current=1,
            grind_delta_um_from_current=12.5,
            projected_relative_step_from_reference=43,
            projected_relative_grind_um_from_reference=537.5,
            next_dose_g=18.0,
            target_yield_g=36.0,
            target_ratio=2.0,
            mode=RecommendationMode.ZERO_IMMEDIATE_BO,
            confidence=0.3,
            reason="test",
        )
        shot = ShotRecord(
            shot_id="shot_1",
            timestamp=2,
            install_id="install_1",
            machine_id="machine_1",
            machine_adapter="gaggimate",
            profile=np.zeros((5, 100), dtype=np.float32),
            microns_per_step=12.5,
            relative_grind_steps_from_reference=43,
            dose_in_g=18.0,
            target_yield_g=36.0,
            beverage_out_g=36.0,
        )
        result = infer_follow_through(shot, rec, RecommendationDecision.IGNORED)
        self.assertEqual(result.state, FollowThroughState.NOT_FOLLOWED)
        self.assertEqual(result.attribution_weight, 0.0)

    def test_not_followed_recommendation_does_not_reduce_actual_shot_rating_confidence(self) -> None:
        result = compute_reward(
            human_rating=5,
            profile_score=0.1,
            follow_through=FollowThroughState.NOT_FOLLOWED,
            taste_tags=["balanced"],
        )
        self.assertGreater(result.reward, 0.8)
        self.assertEqual(result.confidence, 1.0)

    def test_reward_confidence_uses_tags_and_channeling_signal(self) -> None:
        untagged = compute_reward(
            human_rating=None,
            profile_score=0.8,
            follow_through=FollowThroughState.UNKNOWN,
        )
        tagged = compute_reward(
            human_rating=None,
            profile_score=0.8,
            follow_through=FollowThroughState.UNKNOWN,
            taste_tags=["balanced"],
        )
        channeling = compute_reward(
            human_rating=None,
            profile_score=0.8,
            follow_through=FollowThroughState.UNKNOWN,
            taste_tags=["channeling_suspected"],
        )

        self.assertGreater(tagged.confidence, untagged.confidence)
        self.assertLess(channeling.confidence, tagged.confidence)

    def test_stale_rules_detect_expiry_bean_change_and_manual_recipe_change(self) -> None:
        rec = Recommendation(
            recommendation_id="rec_1",
            created_at=1,
            updated_at=1,
            expires_at=10,
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_a",
            grind_delta_steps_from_current=1,
            grind_delta_um_from_current=12.5,
            projected_relative_step_from_reference=43,
            projected_relative_grind_um_from_reference=537.5,
            next_dose_g=18.0,
            target_yield_g=36.0,
            target_ratio=2.0,
            mode=RecommendationMode.ZERO_IMMEDIATE_BO,
            confidence=0.3,
            reason="test",
        )
        self.assertTrue(check_recommendation_staleness(rec, now=10, bean_context_id="bean_a").stale)
        self.assertTrue(check_recommendation_staleness(rec, now=5, bean_context_id="bean_b").stale)
        self.assertTrue(
            check_recommendation_staleness(
                rec,
                now=5,
                bean_context_id="bean_a",
                current_recipe=Recipe(
                    relative_grind_steps_from_reference=48,
                    microns_per_step=12.5,
                    dose_g=18.0,
                    target_yield_g=36.0,
                ),
            ).stale
        )
        rec.grinder_context_id = "grinder_a"
        grinder_stale = check_recommendation_staleness(
            rec,
            now=5,
            bean_context_id="bean_a",
            grinder_context_id="grinder_b",
        )
        self.assertTrue(grinder_stale.stale)
        self.assertEqual(grinder_stale.reason, "grinder_context_changed")


if __name__ == "__main__":
    unittest.main()

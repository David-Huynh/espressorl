from __future__ import annotations

import unittest

import numpy as np

from espresso_rl.domain.events import OptimizerSettingsEvent, ShotProfileEvent
from espresso_rl.domain.follow_through import infer_follow_through
from espresso_rl.domain.models import (
    FollowThroughState,
    Recipe,
    Recommendation,
    RecommendationDecision,
    RecommendationMode,
    RecommendationStatus,
    ShotRecord,
)
from espresso_rl.domain.profile import (
    build_fixed_cadence_sequence,
    resample_profile,
    resample_profile_with_quality,
    resample_shot_metadata,
)
from espresso_rl.domain.staleness import check_recommendation_staleness


class DomainCoreTests(unittest.TestCase):
    def test_optimizer_settings_migrates_old_bo_name_but_rejects_removed_model(self) -> None:
        event = OptimizerSettingsEvent("install", "machine", 1, optimizer_mode="bayesian_optimization")
        self.assertEqual(event.optimizer_mode, "cpbo")
        with self.assertRaises(ValueError):
            OptimizerSettingsEvent("install", "machine", 1, optimizer_mode="dreamer_v3")

    def test_profile_event_normalizes_hash_and_validates_arrays(self) -> None:
        shot = _event(raw_profile_hash="A" * 64)
        self.assertEqual(shot.raw_profile_hash, "a" * 64)
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            _event(weight=[0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            _event(pressure=[0.0, float("nan"), 9.0])

    def test_profile_resampling_and_metadata_are_fixed_shape(self) -> None:
        shot = _event(
            temperature=[90.0, 91.0, 92.0],
            target_temperature=[92.0, 92.0, 92.0],
            pump_target_mode=[1, 1, 2],
            valve_open=[True, True, False],
        )
        self.assertEqual(resample_profile(shot).shape, (5, 100))
        metadata = resample_shot_metadata(shot)
        self.assertEqual(metadata.temperature_profile.shape, (100,))  # type: ignore[union-attr]
        self.assertEqual(metadata.pump_target_mode_profile.shape, (100,))  # type: ignore[union-attr]

    def test_fixed_cadence_requires_complete_control_telemetry(self) -> None:
        self.assertIsNone(build_fixed_cadence_sequence(_event()))
        sequence = build_fixed_cadence_sequence(
            _event(
                time_ms=[0, 250, 500],
                temperature=[90.0, 90.5, 91.0],
                target_temperature=[91.0, 91.0, 91.0],
                pump_target_mode=[1, 1, 2],
                valve_open=[True, True, False],
            )
        )
        self.assertIsNotNone(sequence)
        self.assertEqual(sequence.sample_interval_ms, 250)  # type: ignore[union-attr]

    def test_invalid_pump_flow_is_masked_without_losing_beverage_flow(self) -> None:
        shot = _event(pump_flow=[0.0, 500.0, 500.0])
        quality = resample_profile_with_quality(shot)
        self.assertTrue(quality.flow_masked)
        self.assertTrue(np.all(quality.profile[2] == 0.0))
        self.assertFalse(np.all(resample_shot_metadata(shot).beverage_flow_profile == 0.0))

    def test_ignored_recommendation_is_never_followed(self) -> None:
        result = infer_follow_through(
            _shot_record(),
            _recommendation(),
            RecommendationDecision.IGNORED,
        )
        self.assertEqual(result.state, FollowThroughState.NOT_FOLLOWED)
        self.assertEqual(result.attribution_weight, 0.0)

    def test_staleness_detects_expiry_and_context_change(self) -> None:
        recommendation = _recommendation()
        self.assertTrue(
            check_recommendation_staleness(
                recommendation,
                now=200,
                bean_context_id="bean",
                grinder_context_id="grinder",
            ).stale
        )
        recommendation.expires_at = 500
        self.assertTrue(
            check_recommendation_staleness(
                recommendation,
                now=100,
                bean_context_id="other",
                grinder_context_id="grinder",
            ).stale
        )


def _event(**overrides: object) -> ShotProfileEvent:
    values: dict[str, object] = {
        "shot_id": "shot",
        "install_id": "install",
        "machine_id": "machine",
        "machine_adapter": "gaggimate",
        "timestamp": 100,
        "time_ms": [0, 500, 1000],
        "pressure": [0.0, 8.0, 9.0],
        "target_pressure": [0.0, 8.0, 9.0],
        "pump_flow": [0.0, 2.0, 2.0],
        "target_flow": [0.0, 2.0, 2.0],
        "beverage_flow": [0.0, 1.8, 2.0],
        "weight": [0.0, 10.0, 36.0],
        "microns_per_step": 12.5,
        "relative_grind_steps_from_reference": 2.0,
        "dose_in_g": 18.0,
        "target_yield_g": 36.0,
        "beverage_out_g": 36.0,
        "shot_time_s": 30.0,
    }
    values.update(overrides)
    return ShotProfileEvent(**values)  # type: ignore[arg-type]


def _shot_record() -> ShotRecord:
    return ShotRecord(
        shot_id="shot",
        timestamp=100,
        install_id="install",
        machine_id="machine",
        machine_adapter="gaggimate",
        profile=np.zeros((5, 100), dtype=np.float32),
        microns_per_step=12.5,
        relative_grind_steps_from_reference=3.0,
        dose_in_g=18.0,
        target_yield_g=36.0,
        beverage_out_g=36.0,
    )


def _recommendation() -> Recommendation:
    return Recommendation(
        recommendation_id="rec",
        created_at=50,
        updated_at=50,
        expires_at=100,
        install_id="install",
        machine_id="machine",
        bean_context_id="bean",
        grinder_context_id="grinder",
        grind_delta_steps_from_current=1.0,
        grind_delta_um_from_current=12.5,
        projected_relative_step_from_reference=3.0,
        projected_relative_grind_um_from_reference=37.5,
        next_dose_g=18.0,
        target_yield_g=36.0,
        target_ratio=2.0,
        mode=RecommendationMode.CPBO_BEST_INCUMBENT,
        confidence=0.5,
        reason="test",
        status=RecommendationStatus.PENDING,
    )


if __name__ == "__main__":
    unittest.main()

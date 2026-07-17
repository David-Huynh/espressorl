from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from espresso_rl.adapters.sqlite_repositories import (
    SQLiteRecommendationRepository,
    SQLiteShotRepository,
    SQLiteStore,
    SQLiteUploadQueueRepository,
)
from espresso_rl.application.services import EspressoRLService
from espresso_rl.domain.events import RecommendationApplyEvent, ShotCorrectionEvent, ShotProfileEvent
from espresso_rl.domain.models import (
    Recommendation,
    RecommendationApplyStatus,
    RecommendationMode,
    RecommendationStatus,
    UploadQueueStatus,
)


class ApplicationServiceTests(unittest.TestCase):
    def test_ingest_stores_physical_shot_without_scalar_taste_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                service = EspressoRLService(
                    shots,
                    SQLiteRecommendationRepository(store),
                    clock=lambda: 200,
                )

                result = service.ingest_shot_profile(_shot_event())

                self.assertTrue(result.stored)
                self.assertIsNone(result.recommendation)
                stored = shots.get("shot_1")
                self.assertIsNotNone(stored)
                self.assertEqual(stored.beverage_out_g, 35.8)  # type: ignore[union-attr]
                self.assertFalse(hasattr(stored, "human_rating"))
                self.assertFalse(hasattr(stored, "reward"))

    def test_local_optimization_disabled_drops_before_storage_or_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                queue = SQLiteUploadQueueRepository(store)
                service = EspressoRLService(
                    shots,
                    SQLiteRecommendationRepository(store),
                    upload_queue=queue,
                    clock=lambda: 200,
                    community_upload_enabled_default=True,
                )

                result = service.ingest_shot_profile(
                    _shot_event(local_optimization_enabled=False)
                )

                self.assertFalse(result.stored)
                self.assertEqual(result.dropped_reason, "local_optimization_disabled")
                self.assertIsNone(shots.get("shot_1"))
                self.assertEqual(queue.count_by_status(), {})

    def test_community_upload_is_independent_and_requires_explicit_consent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                service = EspressoRLService(
                    SQLiteShotRepository(store),
                    SQLiteRecommendationRepository(store),
                    upload_queue=queue,
                    clock=lambda: 200,
                )

                service.ingest_shot_profile(_shot_event(community_upload_enabled=True))

                counts = queue.count_by_status()
                self.assertEqual(counts.get(UploadQueueStatus.PENDING), 1)

    def test_exact_shot_replay_is_idempotent_and_does_not_requeue_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                service = EspressoRLService(
                    SQLiteShotRepository(store),
                    SQLiteRecommendationRepository(store),
                    upload_queue=queue,
                    clock=lambda: 200,
                )
                event = _shot_event(community_upload_enabled=True)

                first = service.ingest_shot_profile(event)
                replay = service.ingest_shot_profile(event)

                self.assertFalse(first.replayed)
                self.assertTrue(replay.replayed)
                self.assertEqual(queue.count_by_status().get(UploadQueueStatus.PENDING), 1)

    def test_conflicting_shot_replay_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                service = EspressoRLService(
                    SQLiteShotRepository(store),
                    SQLiteRecommendationRepository(store),
                    clock=lambda: 200,
                )
                service.ingest_shot_profile(_shot_event())

                with self.assertRaisesRegex(ValueError, "conflicts with an existing immutable shot"):
                    service.ingest_shot_profile(_shot_event(beverage_out_g=34.0))

    def test_predictive_cutoff_diagnostics_round_trip_without_replacing_measured_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                service = EspressoRLService(
                    shots,
                    SQLiteRecommendationRepository(store),
                    clock=lambda: 200,
                )

                service.ingest_shot_profile(
                    _shot_event(
                        beverage_out_g=33.2,
                        beverage_out_observation="control_cutoff",
                        predicted_final_beverage_out_g=36.0,
                        predictive_stop_applied=True,
                        predictive_stop_delay_ms=800.0,
                        predictive_stop_rate_g_per_s=3.5,
                        predictive_stop_lead_g=2.8,
                    )
                )

                stored = shots.get("shot_1")
                self.assertEqual(stored.beverage_out_g, 33.2)  # type: ignore[union-attr]
                self.assertEqual(stored.predicted_final_beverage_out_g, 36.0)  # type: ignore[union-attr]
                self.assertTrue(stored.predictive_stop_applied)  # type: ignore[union-attr]

    def test_recommendation_apply_rejects_cross_machine_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                recommendations = SQLiteRecommendationRepository(store)
                recommendations.upsert(_recommendation())
                service = EspressoRLService(
                    SQLiteShotRepository(store),
                    recommendations,
                    clock=lambda: 200,
                )

                with self.assertRaisesRegex(ValueError, "owner"):
                    service.record_recommendation_apply(
                        RecommendationApplyEvent(
                            recommendation_id="rec_1",
                            status=RecommendationApplyStatus.APPLIED,
                            timestamp=200,
                            install_id="install_1",
                            machine_id="machine_other",
                        )
                    )

    def test_shot_parameter_correction_recomputes_recipe_and_replaces_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                queue = SQLiteUploadQueueRepository(store)
                service = EspressoRLService(
                    shots,
                    SQLiteRecommendationRepository(store),
                    upload_queue=queue,
                    clock=lambda: 250,
                )
                service.ingest_shot_profile(_shot_event(community_upload_enabled=True))
                original_upload = queue.list_by_status(UploadQueueStatus.PENDING)[0]
                queue.update_status(
                    original_upload.upload_id,
                    UploadQueueStatus.UPLOADED,
                    now=210,
                )

                corrected = service.record_shot_correction(
                    ShotCorrectionEvent(
                        shot_id="shot_1",
                        install_id="install_1",
                        machine_id="machine_1",
                        timestamp=240,
                        relative_grind_steps_from_reference=3.0,
                        dose_in_g=17.5,
                        target_yield_g=42.0,
                        beverage_out_g=41.5,
                        source="gaggimate_shot_history",
                    )
                )

                self.assertEqual(corrected.relative_grind_steps_from_reference, 3.0)
                self.assertEqual(corrected.relative_grind_um_from_reference, 37.5)
                self.assertEqual(corrected.dose_in_g, 17.5)
                self.assertEqual(corrected.dose_target_g, 17.5)
                self.assertAlmostEqual(corrected.target_ratio, 42.0 / 17.5)
                self.assertAlmostEqual(corrected.brew_ratio, 41.5 / 17.5)
                self.assertEqual(corrected.beverage_out_observation, "user_corrected")
                pending = queue.list_by_status(UploadQueueStatus.PENDING)
                self.assertEqual(len(pending), 1)
                self.assertEqual(
                    len(queue.list_by_status(UploadQueueStatus.UPLOADED)),
                    1,
                )
                payload = json.loads(pending[0].payload_json)
                self.assertEqual(payload["dose_in_g"], 17.5)
                self.assertEqual(payload["target_yield_g"], 42.0)

    def test_cross_owner_correction_does_not_mutate_canonical_shot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                service = EspressoRLService(
                    shots,
                    SQLiteRecommendationRepository(store),
                    clock=lambda: 250,
                )
                service.ingest_shot_profile(_shot_event())
                with self.assertRaisesRegex(ValueError, "owner"):
                    service.record_shot_correction(
                        ShotCorrectionEvent(
                            shot_id="shot_1",
                            install_id="other_install",
                            machine_id="machine_1",
                            timestamp=240,
                            dose_in_g=17.5,
                            source="test",
                        )
                    )
                self.assertEqual(shots.get("shot_1").dose_in_g, 18.0)  # type: ignore[union-attr]

    def test_measured_dose_correction_overrides_stale_followed_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                recommendations = SQLiteRecommendationRepository(store)
                recommendations.upsert(_recommendation())
                service = EspressoRLService(
                    shots,
                    recommendations,
                    clock=lambda: 250,
                )
                service.ingest_shot_profile(_shot_event(recommendation_id="rec_1"))
                service.record_shot_correction(
                    ShotCorrectionEvent(
                        shot_id="shot_1",
                        install_id="install_1",
                        machine_id="machine_1",
                        timestamp=230,
                        dose_followed=True,
                        source="gaggimate_dose_confirmation",
                    )
                )

                corrected = service.record_shot_correction(
                    ShotCorrectionEvent(
                        shot_id="shot_1",
                        install_id="install_1",
                        machine_id="machine_1",
                        timestamp=240,
                        dose_in_g=16.3,
                        source="gaggimate_shot_history",
                    )
                )

                self.assertEqual(corrected.dose_in_g, 16.3)
                self.assertTrue(corrected.dose_observed)
                self.assertFalse(corrected.dose_target_confirmed)
                self.assertFalse(corrected.dose_followed)
                self.assertEqual(corrected.dose_recommendation_trust, 0.0)

    def test_correction_inside_integrity_envelope_is_not_limited_by_optimizer_space(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                service = EspressoRLService(
                    shots,
                    SQLiteRecommendationRepository(store),
                    clock=lambda: 250,
                )
                service.ingest_shot_profile(_shot_event())

                corrected = service.record_shot_correction(
                    ShotCorrectionEvent(
                        shot_id="shot_1",
                        install_id="install_1",
                        machine_id="machine_1",
                        timestamp=240,
                        dose_in_g=80.0,
                        target_yield_g=300.0,
                        source="test",
                    )
                )

                self.assertEqual(corrected.dose_in_g, 80.0)
                self.assertEqual(corrected.target_yield_g, 300.0)
                self.assertEqual(shots.get("shot_1").dose_in_g, 80.0)  # type: ignore[union-attr]


def _shot_event(**overrides: object) -> ShotProfileEvent:
    values: dict[str, object] = {
        "shot_id": "shot_1",
        "install_id": "install_1",
        "machine_id": "machine_1",
        "machine_adapter": "gaggimate",
        "timestamp": 150,
        "time_ms": [0, 500, 1000],
        "pressure": [0.0, 8.0, 9.0],
        "target_pressure": [0.0, 8.0, 9.0],
        "pump_flow": [0.0, 2.0, 2.0],
        "target_flow": [0.0, 2.0, 2.0],
        "beverage_flow": [0.0, 1.8, 2.0],
        "weight": [0.0, 10.0, 35.8],
        "microns_per_step": 12.5,
        "relative_grind_steps_from_reference": 2.0,
        "dose_in_g": 18.0,
        "target_yield_g": 36.0,
        "beverage_out_g": 35.8,
        "shot_time_s": 30.0,
        "bean_context_id": "bean_1",
        "grinder_context_id": "grinder_1",
        "profile_id": "profile_1",
    }
    values.update(overrides)
    return ShotProfileEvent(**values)  # type: ignore[arg-type]


def _recommendation() -> Recommendation:
    return Recommendation(
        recommendation_id="rec_1",
        created_at=100,
        updated_at=100,
        expires_at=500,
        install_id="install_1",
        machine_id="machine_1",
        bean_context_id="bean_1",
        grinder_context_id="grinder_1",
        profile_id="profile_1",
        grind_delta_steps_from_current=1.0,
        grind_delta_um_from_current=12.5,
        projected_relative_step_from_reference=3.0,
        projected_relative_grind_um_from_reference=37.5,
        next_dose_g=18.0,
        target_yield_g=36.0,
        target_ratio=2.0,
        mode=RecommendationMode.CPBO_BEST_INCUMBENT,
        confidence=0.5,
        reason="CPBO-MES suggestion",
        status=RecommendationStatus.PENDING,
    )


if __name__ == "__main__":
    unittest.main()

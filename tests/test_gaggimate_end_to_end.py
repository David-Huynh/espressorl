from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from espresso_rl.adapters.gaggimate_mqtt import GaggimateMQTTClient
from espresso_rl.adapters.supabase_upload import UploadQueueWorker
from espresso_rl.adapters.sqlite_repositories import (
    SQLiteRecommendationRepository,
    SQLiteShotRepository,
    SQLiteStore,
    SQLiteUploadQueueRepository,
)
from espresso_rl.application.services import EspressoRLService
from espresso_rl.application.runtime_coordinator import AutoTuningRuntimeCoordinator
from espresso_rl.application.upload_maintenance import UploadQueueMaintenanceService
from espresso_rl.config import Config
from espresso_rl.domain.models import UploadQueueStatus
from espresso_rl.domain.optimization import OptimizationContext
from espresso_rl.main import build_status_payload, upload_queue_for_service
from espresso_rl.optimizers.conservative_bo import ConservativeBOOptimizer

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class FakeMQTTTransport:
    def __init__(self) -> None:
        self.published: list[tuple[str, str, int, bool]] = []

    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> None:
        self.published.append((topic, payload, qos, retain))


class FakeMQTTMessage:
    def __init__(self, topic: str, payload: dict) -> None:
        self.topic = topic
        self.payload = json.dumps(payload).encode("utf-8")


class RecordingBOOptimizer(ConservativeBOOptimizer):
    def __init__(self) -> None:
        self.contexts: list[OptimizationContext] = []

    def recommend(self, context: OptimizationContext):
        self.contexts.append(context)
        return super().recommend(context)


class FailOnceUploadClient:
    def __init__(self) -> None:
        self.attempted_upload_ids: list[str] = []

    def upload(self, item) -> None:
        self.attempted_upload_ids.append(item.upload_id)
        if len(self.attempted_upload_ids) == 1:
            raise RuntimeError("temporary ingest outage")


class GaggimateEndToEndHarness:
    def __init__(self, store: SQLiteStore, data_dir: Path) -> None:
        self.now = 1_700_000_100
        self.config = Config(
            mqtt_host="localhost",
            data_dir=data_dir,
            install_id="install_integration_1",
            machine_id="gaggimate:AA_BB",
            bean_context_id="bean_integration_1",
            grinder_context_id="grinder_integration_1",
            community_upload_enabled=True,
            supabase_ingest_url="https://example.invalid/functions/v1/espresso-rl-ingest",
            upload_secret="s" * 32,
        )
        self.config.now = lambda: self.now  # type: ignore[method-assign]
        self.shots = SQLiteShotRepository(store)
        self.recommendations = SQLiteRecommendationRepository(store)
        self.uploads = SQLiteUploadQueueRepository(store)
        self.upload_maintenance = UploadQueueMaintenanceService(self.uploads, clock=self.config.now)
        self.optimizer = RecordingBOOptimizer()
        self.service = EspressoRLService(
            shots=self.shots,
            recommendations=self.recommendations,
            optimizer=self.optimizer,
            upload_queue=upload_queue_for_service(self.config, self.uploads),
            clock=self.config.now,
            community_upload_enabled_default=False,
        )
        self.transport = FakeMQTTTransport()
        self.last_ingest_result = None
        self.last_feedback_result = None
        self.last_machine_state_recommendation = None
        self.runtime_coordinator = AutoTuningRuntimeCoordinator(
            service=self.service,
            publisher=self,
        )
        self.client = GaggimateMQTTClient(
            config=self.config,
            on_shot=self._on_shot,
            on_feedback=self._on_feedback,
            on_correction=lambda event: None,
            on_upload_maintenance=lambda event: None,
            on_decision=lambda event: None,
            on_apply=lambda event: None,
            on_machine_state=self._on_machine_state,
        )
        self.client._client = self.transport  # type: ignore[assignment]

    def send(self, topic: str, payload: dict) -> None:
        self.client._on_message(  # type: ignore[arg-type]
            self.transport,
            None,
            FakeMQTTMessage(topic, payload),
        )

    def publications(self, topic: str) -> list[tuple[dict, int, bool]]:
        return [
            (json.loads(payload), qos, retain)
            for published_topic, payload, qos, retain in self.transport.published
            if published_topic == topic
        ]

    def publish_recommendation(self, recommendation) -> None:
        self.client.publish_recommendation(recommendation)

    def publish_status(
        self,
        machine_id: str,
        bean_context_id: str | None,
        grinder_context_id: str | None,
        *,
        profile_id: str | None = None,
        profile_label: str | None = None,
        last_shot_id: str | None = None,
        last_shot_at: int | None = None,
        last_recommendation_id: str | None = None,
        last_recommendation_at: int | None = None,
        mode: str | None = None,
    ) -> None:
        status = build_status_payload(
            config=self.config,
            service=self.service,
            upload_maintenance=self.upload_maintenance,
            shot_repo=self.shots,
            upload_queue_repo=self.uploads,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            grinder_context_id=grinder_context_id,
            profile_id=profile_id,
            profile_label=profile_label,
            last_shot_id=last_shot_id,
            last_shot_at=last_shot_at,
            last_recommendation_id=last_recommendation_id,
            last_recommendation_at=last_recommendation_at,
            mode=mode,
        )
        self.client.publish_status(machine_id, status)

    def _on_shot(self, event) -> None:
        self.last_ingest_result = self.runtime_coordinator.handle_shot(event)

    def _on_feedback(self, event) -> None:
        self.last_feedback_result = self.runtime_coordinator.handle_feedback(event)

    def _on_machine_state(self, event) -> None:
        self.last_machine_state_recommendation = self.runtime_coordinator.handle_machine_state(event)


class GaggimateEndToEndTests(unittest.TestCase):
    def shot_payload(self, **overrides) -> dict:
        payload = json.loads((FIXTURE_DIR / "gaggimate_shot_profile.json").read_text())
        payload.update(overrides)
        return payload

    def rating_payload(self, shot_id: str, **overrides) -> dict:
        payload = json.loads((FIXTURE_DIR / "gaggimate_shot_rating.json").read_text())
        payload.update({"shot_id": shot_id, **overrides})
        return payload

    def test_shot_rating_recommendation_upload_and_status_chain(self) -> None:
        shot_payload = self.shot_payload()
        rating_payload = self.rating_payload("shot_integration_1")

        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                harness = GaggimateEndToEndHarness(store, Path(tmp))
                harness.send("gaggimate/AA_BB/shot/profile", shot_payload)

                stored = harness.shots.get("shot_integration_1")
                self.assertIsNotNone(stored)
                self.assertEqual(stored.bean_context_id, "bean_integration_1")
                self.assertEqual(stored.grinder_context_id, "grinder_integration_1")
                self.assertEqual(stored.profile_id, "integration_lever")
                self.assertEqual(stored.beverage_out_g, 38.0)
                self.assertEqual(stored.relative_grind_steps_from_reference, 2.0)
                self.assertIsNotNone(stored.temperature_profile)
                self.assertIsNone(harness.last_ingest_result.recommendation)

                pending = harness.uploads.list_by_status(UploadQueueStatus.PENDING)
                self.assertEqual([(item.local_record_type, item.local_record_id) for item in pending], [
                    ("shot", "shot_integration_1")
                ])

                first_statuses = harness.publications("gaggimate/AA_BB/rl/status")
                self.assertEqual(len(first_statuses), 1)
                first_status, qos, retain = first_statuses[0]
                self.assertEqual(qos, 1)
                self.assertTrue(retain)
                first_steps = {
                    step["key"]: step for step in first_status["auto_tuning_diagnostic_steps"]
                }
                self.assertEqual(first_steps["shot_observed"]["state"], "ok")
                self.assertEqual(first_steps["shot_stored"]["state"], "ok")
                self.assertEqual(first_steps["shot_usable"]["state"], "ok")
                self.assertEqual(first_steps["rating"]["state"], "waiting")
                self.assertEqual(first_steps["recommendation"]["state"], "waiting")
                self.assertEqual(first_steps["community_upload"]["state"], "waiting")

                harness.send("gaggimate/AA_BB/rl/rating", rating_payload)

                rated = harness.shots.get("shot_integration_1")
                self.assertEqual(rated.human_rating, 4)
                self.assertEqual(rated.taste_tags, ["sweet", "balanced"])
                self.assertTrue(rated.feedback_recorded)

                recommendation = harness.last_feedback_result.recommendation
                self.assertIsNotNone(recommendation)
                self.assertEqual(recommendation.source_shot_id, "shot_integration_1")
                self.assertIsNotNone(harness.recommendations.get(recommendation.recommendation_id))

                pending = harness.uploads.list_by_status(UploadQueueStatus.PENDING)
                self.assertEqual(
                    {(item.local_record_type, item.local_record_id) for item in pending},
                    {
                        ("shot", "shot_integration_1"),
                        ("recommendation", recommendation.recommendation_id),
                    },
                )
                shot_upload = next(item for item in pending if item.local_record_type == "shot")
                queued_shot = json.loads(shot_upload.payload_json)
                self.assertEqual(queued_shot["human_rating"], 4)
                self.assertEqual(queued_shot["taste_tags"], ["sweet", "balanced"])

                recommendations = harness.publications("gaggimate/AA_BB/rl/recommendation")
                self.assertEqual(len(recommendations), 1)
                published_recommendation, qos, retain = recommendations[0]
                self.assertEqual(qos, 1)
                self.assertTrue(retain)
                self.assertEqual(
                    published_recommendation["recommendation_id"],
                    recommendation.recommendation_id,
                )

                statuses = harness.publications("gaggimate/AA_BB/rl/status")
                self.assertEqual(len(statuses), 2)
                final_status, qos, retain = statuses[-1]
                self.assertEqual(qos, 1)
                self.assertTrue(retain)
                self.assertEqual(final_status["event_type"], "espresso_rl_status")
                self.assertEqual(final_status["optimizer_profile_id"], "integration_lever")
                self.assertEqual(final_status["optimizer_profile_label"], "Integration Lever Profile")
                self.assertEqual(final_status["last_shot_id"], "shot_integration_1")
                self.assertEqual(final_status["last_shot_human_rating"], 4)
                self.assertEqual(
                    final_status["last_recommendation_id"],
                    recommendation.recommendation_id,
                )
                self.assertEqual(final_status["upload_queue_pending_count"], 2)
                self.assertTrue(final_status["community_upload_enabled"])
                final_steps = {
                    step["key"]: step for step in final_status["auto_tuning_diagnostic_steps"]
                }
                self.assertEqual(final_steps["rating"]["state"], "ok")
                self.assertEqual(final_steps["recommendation"]["state"], "ok")
                self.assertEqual(final_steps["community_upload"]["state"], "waiting")
                self.assertEqual(final_steps["status_published"]["state"], "ok")

    def test_partial_action_and_actual_yield_remain_optimizable_and_uploadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                harness = GaggimateEndToEndHarness(store, Path(tmp))
                harness.send("gaggimate/AA_BB/shot/profile", self.shot_payload())
                harness.send(
                    "gaggimate/AA_BB/rl/rating",
                    self.rating_payload("shot_integration_1"),
                )
                prior_recommendation = harness.last_feedback_result.recommendation
                self.assertIsNotNone(prior_recommendation)

                partial = self.shot_payload(
                    shot_id="shot_partial_action",
                    timestamp=1_700_000_200,
                    beverage_out_g=31.5,
                    weight=[0.0, 0.0, 1.2, 5.5, 12.5, 20.5, 27.0, 31.5],
                    grind_observed=False,
                    dose_observed=False,
                )
                for field_name in (
                    "relative_grind_steps_from_reference",
                    "current_absolute_step",
                    "absolute_reference_step",
                    "dose_in_g",
                ):
                    partial.pop(field_name, None)

                harness.send("gaggimate/AA_BB/shot/profile", partial)
                harness.send(
                    "gaggimate/AA_BB/rl/rating",
                    self.rating_payload(
                        "shot_partial_action",
                        timestamp=1_700_000_210,
                        rating=3,
                        taste_tags=["thin"],
                    ),
                )

                stored = harness.shots.get("shot_partial_action")
                self.assertIsNotNone(stored)
                self.assertFalse(stored.grind_observed)
                self.assertFalse(stored.dose_observed)
                self.assertTrue(stored.target_yield_observed)
                self.assertEqual(stored.action_observed, {
                    "grind": False,
                    "dose": False,
                    "target_yield": True,
                })
                self.assertEqual(stored.beverage_out_g, 31.5)
                self.assertEqual(stored.realized_yield_g, 31.5)
                self.assertNotEqual(stored.beverage_out_g, stored.recommended_target_yield_g)
                self.assertFalse(stored.exclude_from_local_optimization)
                self.assertGreater(stored.optimization_weight, 0.0)

                recommendation = harness.last_feedback_result.recommendation
                self.assertIsNotNone(recommendation)
                self.assertEqual(recommendation.source_shot_id, "shot_partial_action")
                optimizer_context = harness.optimizer.contexts[-1]
                self.assertEqual(optimizer_context.current_recipe.relative_grind_steps_from_reference, 2.0)
                self.assertEqual(optimizer_context.current_recipe.dose_g, 18.0)
                self.assertEqual(optimizer_context.current_recipe.target_yield_g, 31.5)
                partial_observation = optimizer_context.shots[-1]
                self.assertEqual(partial_observation.shot_id, "shot_partial_action")
                self.assertEqual(partial_observation.action_observed, stored.action_observed)

                pending = harness.uploads.list_by_status(UploadQueueStatus.PENDING)
                shot_upload = next(
                    item
                    for item in pending
                    if item.local_record_type == "shot"
                    and item.local_record_id == "shot_partial_action"
                )
                queued_shot = json.loads(shot_upload.payload_json)
                self.assertEqual(queued_shot["action_observed"], stored.action_observed)
                self.assertIsNone(queued_shot["relative_grind_steps_from_reference"])
                self.assertEqual(queued_shot["dose_in_g"], 18.0)
                self.assertEqual(queued_shot["beverage_out_g"], 31.5)
                self.assertEqual(queued_shot["human_rating"], 3)

    def test_optimizer_context_isolated_by_profile_bean_and_grinder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                harness = GaggimateEndToEndHarness(store, Path(tmp))

                harness.send("gaggimate/AA_BB/shot/profile", self.shot_payload())
                harness.send(
                    "gaggimate/AA_BB/rl/rating",
                    self.rating_payload("shot_integration_1"),
                )
                lever_recommendation = harness.last_feedback_result.recommendation
                self.assertEqual(lever_recommendation.profile_id, "integration_lever")

                turbo = self.shot_payload(
                    shot_id="shot_turbo",
                    timestamp=1_700_000_200,
                    profile_id="integration_turbo",
                    profile_label="Integration Turbo Profile",
                )
                harness.send("gaggimate/AA_BB/shot/profile", turbo)
                harness.send(
                    "gaggimate/AA_BB/rl/rating",
                    self.rating_payload("shot_turbo", timestamp=1_700_000_210),
                )
                turbo_recommendation = harness.last_feedback_result.recommendation
                self.assertEqual(turbo_recommendation.profile_id, "integration_turbo")
                self.assertEqual(
                    [shot.shot_id for shot in harness.optimizer.contexts[-1].shots],
                    ["shot_turbo"],
                )

                other_context = self.shot_payload(
                    shot_id="shot_other_context",
                    timestamp=1_700_000_300,
                    bean_context_id="bean_integration_2",
                    bean_context_name="Different Coffee",
                    grinder_context_id="grinder_integration_2",
                    profile_id="integration_turbo",
                    profile_label="Integration Turbo Profile",
                )
                harness.send("gaggimate/AA_BB/shot/profile", other_context)
                harness.send(
                    "gaggimate/AA_BB/rl/rating",
                    self.rating_payload("shot_other_context", timestamp=1_700_000_310),
                )
                isolated_recommendation = harness.last_feedback_result.recommendation
                self.assertEqual(isolated_recommendation.bean_context_id, "bean_integration_2")
                self.assertEqual(isolated_recommendation.grinder_context_id, "grinder_integration_2")
                self.assertEqual(isolated_recommendation.profile_id, "integration_turbo")
                self.assertEqual(
                    [shot.shot_id for shot in harness.optimizer.contexts[-1].shots],
                    ["shot_other_context"],
                )
                self.assertNotEqual(
                    isolated_recommendation.recommendation_id,
                    turbo_recommendation.recommendation_id,
                )

                harness.send(
                    "gaggimate/AA_BB/machine/state",
                    {
                        "event_type": "machine_state",
                        "schema_version": 1,
                        "machine_id": "gaggimate:AA_BB",
                        "timestamp": 1_700_000_400,
                        "state": "idle",
                        "bean_context_id": "bean_integration_1",
                        "bean_context_name": "Integration Coffee",
                        "grinder_context_id": "grinder_integration_1",
                        "grinder_calibration_mode": "absolute_display_calibrated",
                        "microns_per_step": 12.5,
                        "step_direction": "higher_is_coarser",
                        "relative_grind_steps_from_reference": 2,
                        "current_absolute_step": 42,
                        "absolute_reference_step": 40,
                        "dose_in_g": 18.0,
                        "target_yield_g": 38.0,
                        "profile_id": "integration_lever",
                        "profile_label": "Integration Lever Profile",
                    },
                )

                restored = harness.last_machine_state_recommendation
                self.assertIsNotNone(restored)
                self.assertEqual(restored.profile_id, "integration_lever")
                self.assertEqual(
                    [shot.shot_id for shot in harness.optimizer.contexts[-1].shots],
                    ["shot_integration_1"],
                )
                restored_status = harness.publications("gaggimate/AA_BB/rl/status")[-1][0]
                self.assertEqual(restored_status["optimizer_profile_id"], "integration_lever")
                self.assertEqual(
                    restored_status["optimizer_profile_label"],
                    "Integration Lever Profile",
                )
                self.assertEqual(restored_status["local_shot_count"], 1)
                self.assertEqual(restored_status["rated_shot_count"], 1)

    def test_community_upload_opt_out_keeps_local_optimization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                harness = GaggimateEndToEndHarness(store, Path(tmp))
                harness.send(
                    "gaggimate/AA_BB/shot/profile",
                    self.shot_payload(community_upload_enabled=False),
                )
                harness.send(
                    "gaggimate/AA_BB/rl/rating",
                    self.rating_payload("shot_integration_1"),
                )

                self.assertIsNotNone(harness.shots.get("shot_integration_1"))
                self.assertIsNotNone(harness.last_feedback_result.recommendation)
                self.assertEqual(harness.uploads.count_by_status(), {})
                self.assertFalse(
                    harness.service.community_upload_enabled_for(
                        "install_integration_1",
                        "gaggimate:AA_BB",
                    )
                )
                final_status = harness.publications("gaggimate/AA_BB/rl/status")[-1][0]
                self.assertFalse(final_status["community_upload_enabled"])
                final_steps = {
                    step["key"]: step for step in final_status["auto_tuning_diagnostic_steps"]
                }
                self.assertEqual(final_steps["community_upload"]["state"], "off")

    def test_transient_upload_failure_retries_without_losing_local_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                harness = GaggimateEndToEndHarness(store, Path(tmp))
                harness.send("gaggimate/AA_BB/shot/profile", self.shot_payload())
                harness.send(
                    "gaggimate/AA_BB/rl/rating",
                    self.rating_payload("shot_integration_1"),
                )
                client = FailOnceUploadClient()
                worker = UploadQueueWorker(harness.uploads, client, clock=harness.config.now)

                self.assertEqual(worker.run_once(limit=1), 0)
                failed = harness.uploads.list_by_status(UploadQueueStatus.FAILED)
                self.assertEqual(len(failed), 1)
                self.assertEqual(failed[0].attempt_count, 1)
                self.assertGreater(failed[0].next_retry_at, harness.now)
                self.assertIsNotNone(harness.shots.get("shot_integration_1"))

                harness.publish_status(
                    "gaggimate:AA_BB",
                    "bean_integration_1",
                    "grinder_integration_1",
                    profile_id="integration_lever",
                    profile_label="Integration Lever Profile",
                    last_shot_id="shot_integration_1",
                )
                retry_status = harness.publications("gaggimate/AA_BB/rl/status")[-1][0]
                retry_steps = {
                    step["key"]: step for step in retry_status["auto_tuning_diagnostic_steps"]
                }
                self.assertEqual(retry_steps["community_upload"]["state"], "attention")

                failed_upload_id = failed[0].upload_id
                harness.now = failed[0].next_retry_at
                self.assertEqual(worker.run_once(limit=1), 1)
                self.assertEqual(harness.uploads.list_by_status(UploadQueueStatus.FAILED), [])
                uploaded = harness.uploads.list_by_status(UploadQueueStatus.UPLOADED)
                self.assertIn(failed_upload_id, [item.upload_id for item in uploaded])
                self.assertEqual(client.attempted_upload_ids, [failed_upload_id, failed_upload_id])
                self.assertIsNotNone(harness.shots.get("shot_integration_1"))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from espresso_rl.adapters.gaggimate_mqtt import GaggimateMQTTClient
from espresso_rl.adapters.sqlite_repositories import (
    SQLiteRecommendationRepository,
    SQLiteShotRepository,
    SQLiteStore,
    SQLiteUploadQueueRepository,
)
from espresso_rl.application.services import EspressoRLService
from espresso_rl.application.upload_maintenance import UploadQueueMaintenanceService
from espresso_rl.config import Config
from espresso_rl.domain.models import UploadQueueStatus
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
        self.service = EspressoRLService(
            shots=self.shots,
            recommendations=self.recommendations,
            optimizer=ConservativeBOOptimizer(),
            upload_queue=upload_queue_for_service(self.config, self.uploads),
            clock=self.config.now,
            community_upload_enabled_default=False,
        )
        self.transport = FakeMQTTTransport()
        self.last_ingest_result = None
        self.last_feedback_result = None
        self.client = GaggimateMQTTClient(
            config=self.config,
            on_shot=self._on_shot,
            on_feedback=self._on_feedback,
            on_correction=lambda event: None,
            on_upload_maintenance=lambda event: None,
            on_decision=lambda event: None,
            on_apply=lambda event: None,
            on_machine_state=lambda event: None,
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

    def _publish_status(
        self,
        machine_id: str,
        bean_context_id: str | None,
        grinder_context_id: str | None,
        *,
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
            last_shot_id=last_shot_id,
            last_shot_at=last_shot_at,
            last_recommendation_id=last_recommendation_id,
            last_recommendation_at=last_recommendation_at,
            mode=mode,
        )
        self.client.publish_status(machine_id, status)

    def _on_shot(self, event) -> None:
        result = self.service.ingest_shot_profile(event)
        self.last_ingest_result = result
        if result.shot is None:
            return
        if result.recommendation is not None:
            self.client.publish_recommendation(result.recommendation)
        self._publish_status(
            event.machine_id,
            event.bean_context_id,
            event.grinder_context_id,
            last_shot_id=result.shot.shot_id,
            last_shot_at=result.shot.timestamp,
            last_recommendation_id=(
                result.recommendation.recommendation_id if result.recommendation else None
            ),
            last_recommendation_at=(
                result.recommendation.created_at if result.recommendation else None
            ),
            mode=result.recommendation.mode.value if result.recommendation else None,
        )

    def _on_feedback(self, event) -> None:
        result = self.service.record_feedback(event)
        self.last_feedback_result = result
        if result.recommendation is not None:
            self.client.publish_recommendation(result.recommendation)
        self._publish_status(
            result.shot.machine_id,
            result.shot.bean_context_id,
            result.shot.grinder_context_id,
            last_shot_id=result.shot.shot_id,
            last_shot_at=result.shot.timestamp,
            last_recommendation_id=(
                result.recommendation.recommendation_id if result.recommendation else None
            ),
            last_recommendation_at=(
                result.recommendation.created_at if result.recommendation else None
            ),
            mode=result.recommendation.mode.value if result.recommendation else None,
        )


class GaggimateEndToEndTests(unittest.TestCase):
    def test_shot_rating_recommendation_upload_and_status_chain(self) -> None:
        shot_payload = json.loads((FIXTURE_DIR / "gaggimate_shot_profile.json").read_text())
        rating_payload = json.loads((FIXTURE_DIR / "gaggimate_shot_rating.json").read_text())

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


if __name__ == "__main__":
    unittest.main()

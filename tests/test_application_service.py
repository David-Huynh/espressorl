from __future__ import annotations

import unittest

from espresso_rl.application.services import EspressoRLService
from espresso_rl.domain.events import (
    MachineStateEvent,
    RecommendationDecisionEvent,
    ShotFeedbackEvent,
    ShotProfileEvent,
)
from espresso_rl.domain.models import (
    FollowThroughState,
    MachineState,
    Recommendation,
    RecommendationDecision,
    RecommendationMode,
    RecommendationStatus,
    ShotRecord,
    UploadQueueItem,
    UploadQueueStatus,
)
from espresso_rl.optimizers.conservative_bo import ConservativeBOOptimizer


class MemoryShotRepository:
    def __init__(self) -> None:
        self.rows: dict[str, ShotRecord] = {}

    def upsert(self, shot: ShotRecord) -> None:
        self.rows[shot.shot_id] = shot

    def get(self, shot_id: str) -> ShotRecord | None:
        return self.rows.get(shot_id)

    def list_recent(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None = None,
        limit: int = 200,
    ) -> list[ShotRecord]:
        rows = [
            row
            for row in self.rows.values()
            if row.install_id == install_id
            and row.machine_id == machine_id
            and row.bean_context_id == bean_context_id
        ]
        return sorted(rows, key=lambda row: row.timestamp)[-limit:]


class MemoryRecommendationRepository:
    def __init__(self) -> None:
        self.rows: dict[str, Recommendation] = {}

    def upsert(self, recommendation: Recommendation) -> None:
        self.rows[recommendation.recommendation_id] = recommendation

    def get(self, recommendation_id: str) -> Recommendation | None:
        return self.rows.get(recommendation_id)

    def get_current(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        now: int,
    ) -> Recommendation | None:
        rows = [
            row
            for row in self.rows.values()
            if row.install_id == install_id
            and row.machine_id == machine_id
            and row.bean_context_id == bean_context_id
            and row.active_at(now)
        ]
        return max(rows, key=lambda row: row.created_at) if rows else None

    def supersede_active(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        now: int,
        except_recommendation_id: str | None = None,
    ) -> None:
        for row in self.rows.values():
            if row.recommendation_id == except_recommendation_id:
                continue
            if row.install_id == install_id and row.machine_id == machine_id and row.bean_context_id == bean_context_id:
                if row.status in {RecommendationStatus.PENDING, RecommendationStatus.SHOWN}:
                    row.status = RecommendationStatus.SUPERSEDED
                    row.superseded_at = now


class MemoryUploadQueue:
    def __init__(self) -> None:
        self.rows: dict[str, UploadQueueItem] = {}

    def enqueue(self, item: UploadQueueItem) -> None:
        self.rows[item.upload_id] = item

    def list_ready(self, now: int, limit: int = 100) -> list[UploadQueueItem]:
        return [
            item
            for item in self.rows.values()
            if item.status in {UploadQueueStatus.PENDING, UploadQueueStatus.FAILED}
            and (item.next_retry_at is None or item.next_retry_at <= now)
        ][:limit]

    def update_status(
        self,
        upload_id: str,
        status: UploadQueueStatus,
        now: int,
        error_message: str | None = None,
        next_retry_at: int | None = None,
    ) -> None:
        item = self.rows[upload_id]
        item.status = status
        item.error_message = error_message
        item.next_retry_at = next_retry_at
        item.updated_at = now


def shot_event(shot_id: str, timestamp: int, **overrides) -> ShotProfileEvent:
    base = {
        "shot_id": shot_id,
        "install_id": "install_1",
        "machine_id": "machine_1",
        "machine_adapter": "gaggimate",
        "timestamp": timestamp,
        "time_ms": [0, 500, 1000],
        "pressure": [0.0, 8.0, 9.0],
        "target_pressure": [0.0, 8.0, 9.0],
        "flow": [0.0, 2.0, 2.0],
        "target_flow": [0.0, 2.0, 2.0],
        "weight": [0.0, 10.0, 36.0],
        "grinder_step_size_um": 12.5,
        "grind_steps": 42,
        "dose_in_g": 18.0,
        "target_yield_g": 36.0,
        "beverage_out_g": 36.0,
        "shot_time_s": 30.0,
    }
    base.update(overrides)
    return ShotProfileEvent(**base)


class ApplicationServiceTests(unittest.TestCase):
    def test_zero_start_generates_bounded_second_shot_recommendation(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        result = service.ingest_shot_profile(shot_event("shot_1", 1))

        self.assertEqual(result.recommendation.mode, RecommendationMode.ZERO_IMMEDIATE_BO)
        self.assertEqual(result.recommendation.next_dose_g, 18.0)
        self.assertLessEqual(abs(result.recommendation.grind_delta_steps), 2)
        self.assertLessEqual(abs(result.recommendation.target_yield_g - 36.0), 4.0)
        self.assertLess(shots.get("shot_1").reward_confidence, 1.0)  # type: ignore[union-attr]

    def test_decision_and_actual_shot_data_drive_follow_through(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        first = service.ingest_shot_profile(shot_event("shot_1", 1)).recommendation
        service.record_recommendation_decision(
            RecommendationDecisionEvent(
                recommendation_id=first.recommendation_id,
                decision=RecommendationDecision.ACCEPTED,
                timestamp=2,
            )
        )

        service.ingest_shot_profile(
            shot_event(
                "shot_2",
                3,
                recommendation_id=first.recommendation_id,
                grind_steps=first.next_grind_steps,
                dose_in_g=first.next_dose_g,
                target_yield_g=first.target_yield_g,
                beverage_out_g=first.target_yield_g,
            )
        )
        shot = shots.get("shot_2")
        self.assertIsNotNone(shot)
        self.assertEqual(shot.recommendation_followed, FollowThroughState.FOLLOWED)  # type: ignore[union-attr]
        self.assertEqual(recs.get(first.recommendation_id).status, RecommendationStatus.USED)  # type: ignore[union-attr]

        updated = service.record_feedback(
            ShotFeedbackEvent(
                shot_id="shot_2",
                install_id="install_1",
                machine_id="machine_1",
                timestamp=4,
                recommendation_id=first.recommendation_id,
                rating=5,
                taste_tags=["balanced"],
            )
        )
        self.assertGreater(updated.reward or 0.0, 0.8)
        self.assertEqual(updated.reward_confidence, 1.0)

    def test_machine_idle_shows_current_recommendation(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        rec = service.ingest_shot_profile(shot_event("shot_1", 1)).recommendation
        shown = service.handle_machine_state(
            MachineStateEvent(
                install_id="install_1",
                machine_id="machine_1",
                machine_adapter="gaggimate",
                timestamp=2,
                state=MachineState.IDLE,
                grind_steps=42,
                grinder_step_size_um=12.5,
                dose_in_g=18.0,
                target_yield_g=36.0,
            )
        )

        self.assertIsNotNone(shown)
        self.assertEqual(shown.recommendation_id, rec.recommendation_id)  # type: ignore[union-attr]
        self.assertEqual(shown.status, RecommendationStatus.SHOWN)  # type: ignore[union-attr]
        self.assertEqual(shown.shown_count, 1)  # type: ignore[union-attr]

    def test_stale_manual_recipe_change_expires_old_recommendation(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        old = service.ingest_shot_profile(shot_event("shot_1", 1)).recommendation
        new = service.handle_machine_state(
            MachineStateEvent(
                install_id="install_1",
                machine_id="machine_1",
                machine_adapter="gaggimate",
                timestamp=2,
                state=MachineState.IDLE,
                grind_steps=old.next_grind_steps + 10,
                grinder_step_size_um=12.5,
                dose_in_g=18.0,
                target_yield_g=36.0,
            )
        )

        self.assertIsNotNone(new)
        self.assertNotEqual(new.recommendation_id, old.recommendation_id)  # type: ignore[union-attr]
        self.assertEqual(recs.get(old.recommendation_id).status, RecommendationStatus.EXPIRED)  # type: ignore[union-attr]

    def test_accepted_recommendation_is_not_reprompted_as_shown_on_wake(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        rec = service.ingest_shot_profile(shot_event("shot_1", 1)).recommendation
        accepted = service.record_recommendation_decision(
            RecommendationDecisionEvent(
                recommendation_id=rec.recommendation_id,
                decision=RecommendationDecision.ACCEPTED,
                timestamp=2,
            )
        )
        shown = service.handle_machine_state(
            MachineStateEvent(
                install_id="install_1",
                machine_id="machine_1",
                machine_adapter="gaggimate",
                timestamp=3,
                state=MachineState.IDLE,
                grind_steps=rec.next_grind_steps,
                grinder_step_size_um=12.5,
                dose_in_g=rec.next_dose_g,
                target_yield_g=rec.target_yield_g,
            )
        )

        self.assertEqual(shown.recommendation_id, rec.recommendation_id)  # type: ignore[union-attr]
        self.assertEqual(shown.status, RecommendationStatus.ACCEPTED)  # type: ignore[union-attr]
        self.assertEqual(accepted.status, RecommendationStatus.ACCEPTED)

    def test_no_answer_reprompts_until_user_decides(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        rec = service.ingest_shot_profile(shot_event("shot_1", 1)).recommendation
        latest = rec
        for shown_count in range(1, 8):
            latest = service.handle_machine_state(
                MachineStateEvent(
                    install_id="install_1",
                    machine_id="machine_1",
                    machine_adapter="gaggimate",
                    timestamp=shown_count + 1,
                    state=MachineState.IDLE,
                    grind_steps=42,
                    grinder_step_size_um=12.5,
                    dose_in_g=18.0,
                    target_yield_g=36.0,
                )
            )
            self.assertIsNotNone(latest)
            self.assertEqual(latest.recommendation_id, rec.recommendation_id)  # type: ignore[union-attr]
            self.assertEqual(latest.status, RecommendationStatus.SHOWN)  # type: ignore[union-attr]
            self.assertEqual(latest.shown_count, shown_count)  # type: ignore[union-attr]

        accepted = service.record_recommendation_decision(
            RecommendationDecisionEvent(
                recommendation_id=rec.recommendation_id,
                decision=RecommendationDecision.ACCEPTED,
                timestamp=20,
            )
        )
        self.assertEqual(accepted.status, RecommendationStatus.ACCEPTED)

    def test_expired_recommendation_is_not_treated_as_followed_observation(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        rec = service.ingest_shot_profile(shot_event("shot_1", 1)).recommendation
        stored = recs.get(rec.recommendation_id)
        stored.status = RecommendationStatus.EXPIRED  # type: ignore[union-attr]
        recs.upsert(stored)  # type: ignore[arg-type]

        service.ingest_shot_profile(
            shot_event(
                "shot_2",
                2,
                recommendation_id=rec.recommendation_id,
                grind_steps=rec.next_grind_steps,
                dose_in_g=rec.next_dose_g,
                target_yield_g=rec.target_yield_g,
                beverage_out_g=rec.target_yield_g,
            )
        )
        shot = shots.get("shot_2")
        self.assertEqual(shot.recommendation_followed, FollowThroughState.UNKNOWN)  # type: ignore[union-attr]
        self.assertEqual(shot.recommendation_attribution_weight, 0.0)  # type: ignore[union-attr]

    def test_upload_queue_receives_shot_and_recommendation_snapshots_when_enabled(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        uploads = MemoryUploadQueue()
        service = EspressoRLService(
            shots,
            recs,
            ConservativeBOOptimizer(),
            upload_queue=uploads,
            clock=lambda: 10,
        )

        result = service.ingest_shot_profile(shot_event("shot_1", 1))

        self.assertTrue(any(item.local_record_type == "shot" for item in uploads.rows.values()))
        self.assertTrue(any(item.local_record_type == "recommendation" for item in uploads.rows.values()))
        payloads = [item.payload_json for item in uploads.rows.values()]
        self.assertTrue(any('"shot_id":"shot_1"' in payload for payload in payloads))
        self.assertTrue(
            any(result.recommendation.recommendation_id in payload for payload in payloads)
        )


if __name__ == "__main__":
    unittest.main()

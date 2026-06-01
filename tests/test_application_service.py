from __future__ import annotations

import copy
import unittest

from espresso_rl.application.services import EspressoRLService, _recommendation_signature
from espresso_rl.domain.events import (
    MachineStateEvent,
    RecommendationApplyEvent,
    RecommendationDecisionEvent,
    ShotFeedbackEvent,
    ShotProfileEvent,
)
from espresso_rl.domain.models import (
    FollowThroughState,
    MachineState,
    Recommendation,
    RecommendationApplyStatus,
    RecommendationDecision,
    RecommendationMode,
    RecommendationStatus,
    ShotRecord,
    ShotType,
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

    def get_latest(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
    ) -> Recommendation | None:
        rows = [
            row
            for row in self.rows.values()
            if row.install_id == install_id
            and row.machine_id == machine_id
            and row.bean_context_id == bean_context_id
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


class RecordingUploadQueue:
    """Upload queue stub that records every enqueue call (without coalescing), so
    a test can assert exactly which lifecycle transitions produced an upload."""

    def __init__(self) -> None:
        self.calls: list[UploadQueueItem] = []

    def enqueue(self, item: UploadQueueItem) -> None:
        self.calls.append(item)

    def list_ready(self, now: int, limit: int = 100) -> list[UploadQueueItem]:
        return []

    def update_status(self, *args, **kwargs) -> None:
        pass

    def for_record(self, local_record_type: str, local_record_id: str | None = None) -> list[UploadQueueItem]:
        return [
            item
            for item in self.calls
            if item.local_record_type == local_record_type
            and (local_record_id is None or item.local_record_id == local_record_id)
        ]


def idle_event(timestamp: int, **overrides) -> MachineStateEvent:
    base = {
        "install_id": "install_1",
        "machine_id": "machine_1",
        "machine_adapter": "gaggimate",
        "timestamp": timestamp,
        "state": MachineState.IDLE,
        "bean_context_id": "bean_1",
        "grind_steps": 42,
        "grinder_step_size_um": 12.5,
        "dose_in_g": 18.0,
        "target_yield_g": 36.0,
    }
    base.update(overrides)
    return MachineStateEvent(**base)


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
        "bean_context_id": "bean_1",
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

    def test_utility_flush_does_not_consume_or_train_active_recommendation(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        active = service.ingest_shot_profile(shot_event("shot_1", 1)).recommendation
        result = service.ingest_shot_profile(
            shot_event(
                "flush_1",
                2,
                recommendation_id=active.recommendation_id,
                shot_time_s=5.0,
                beverage_out_g=1.0,
                weight=[0.0, 0.5, 1.0],
            )
        )

        self.assertIsNone(result.recommendation)
        self.assertEqual(result.shot.shot_type, ShotType.UTILITY_FLUSH)
        self.assertTrue(result.shot.exclude_from_local_optimization)
        self.assertEqual(result.shot.optimization_weight, 0.0)
        self.assertFalse(result.shot.rating_prompt_allowed)
        self.assertEqual(result.shot.recommendation_attribution_weight, 0.0)
        self.assertEqual(recs.get(active.recommendation_id).status, RecommendationStatus.PENDING)  # type: ignore[union-attr]

    def test_local_optimization_disabled_stores_shot_without_new_recommendation(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        result = service.ingest_shot_profile(
            shot_event(
                "shot_1",
                1,
                local_optimization_enabled=False,
                exclude_from_local_optimization=True,
            )
        )

        self.assertIsNone(result.recommendation)
        self.assertEqual(result.shot.shot_type, ShotType.ESPRESSO)
        self.assertTrue(result.shot.exclude_from_local_optimization)
        self.assertEqual(result.shot.optimization_weight, 0.0)
        self.assertTrue(result.shot.rating_prompt_allowed)
        self.assertEqual(recs.rows, {})

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
                bean_context_id="bean_1",
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
                bean_context_id="bean_1",
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
                bean_context_id="bean_1",
                grind_steps=rec.next_grind_steps,
                grinder_step_size_um=12.5,
                dose_in_g=rec.next_dose_g,
                target_yield_g=rec.target_yield_g,
            )
        )

        self.assertEqual(shown.recommendation_id, rec.recommendation_id)  # type: ignore[union-attr]
        self.assertEqual(shown.status, RecommendationStatus.ACCEPTED)  # type: ignore[union-attr]
        self.assertEqual(accepted.status, RecommendationStatus.ACCEPTED)

    def test_apply_acknowledgement_does_not_mark_recommendation_followed(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        rec = service.ingest_shot_profile(shot_event("shot_1", 1)).recommendation
        service.record_recommendation_decision(
            RecommendationDecisionEvent(
                recommendation_id=rec.recommendation_id,
                decision=RecommendationDecision.ACCEPTED,
                timestamp=2,
            )
        )
        applied = service.record_recommendation_apply(
            RecommendationApplyEvent(
                recommendation_id=rec.recommendation_id,
                status=RecommendationApplyStatus.PARTIALLY_APPLIED,
                timestamp=3,
                applied_fields={"target_yield_g": rec.target_yield_g},
                manual_fields=["next_grind_steps", "next_dose_g"],
                message="Target yield applied; grind and dose are manual.",
            )
        )

        self.assertEqual(applied.status, RecommendationStatus.ACCEPTED)
        self.assertEqual(applied.apply_status, RecommendationApplyStatus.PARTIALLY_APPLIED)
        self.assertEqual(applied.applied_fields["target_yield_g"], rec.target_yield_g)
        self.assertEqual(applied.manual_fields, ["next_grind_steps", "next_dose_g"])
        self.assertIsNone(applied.used_at)
        self.assertEqual(recs.get(rec.recommendation_id).status, RecommendationStatus.ACCEPTED)  # type: ignore[union-attr]

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
                    bean_context_id="bean_1",
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

    def test_idle_reshows_do_not_reenqueue_recommendation(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        queue = RecordingUploadQueue()
        service = EspressoRLService(
            shots, recs, ConservativeBOOptimizer(), upload_queue=queue, clock=lambda: 10
        )

        rec = service.ingest_shot_profile(shot_event("shot_1", 1)).recommendation
        for ts in range(2, 9):
            service.handle_machine_state(idle_event(ts))

        # Only the create and the first show are meaningful; the six later idle
        # re-marks (shown_count 2..7) are incidental churn and must not upload.
        self.assertEqual(len(queue.for_record("recommendation", rec.recommendation_id)), 2)
        # The shot uploads exactly once (at ingest); idle pings never touch it.
        self.assertEqual(len(queue.for_record("shot")), 1)

    def test_recommendation_lifecycle_transitions_each_enqueue_once(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        queue = RecordingUploadQueue()
        service = EspressoRLService(
            shots, recs, ConservativeBOOptimizer(), upload_queue=queue, clock=lambda: 10
        )

        rec = service.ingest_shot_profile(shot_event("shot_1", 1)).recommendation  # created
        service.handle_machine_state(idle_event(2))  # first shown
        service.record_recommendation_decision(
            RecommendationDecisionEvent(
                recommendation_id=rec.recommendation_id,
                decision=RecommendationDecision.ACCEPTED,
                timestamp=3,
            )
        )  # accepted
        service.record_recommendation_apply(
            RecommendationApplyEvent(
                recommendation_id=rec.recommendation_id,
                status=RecommendationApplyStatus.PARTIALLY_APPLIED,
                timestamp=4,
                applied_fields={"target_yield_g": rec.target_yield_g},
                manual_fields=["next_grind_steps"],
            )
        )  # applied

        calls = queue.for_record("recommendation", rec.recommendation_id)
        # created, first-shown, accepted, applied -> four uploads, each a new snapshot.
        self.assertEqual(len(calls), 4)
        self.assertEqual(len({item.payload_hash for item in calls}), 4)

    def test_recommendation_signature_ignores_incidental_churn(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        rec = service.ingest_shot_profile(shot_event("shot_1", 1)).recommendation
        base = recs.get(rec.recommendation_id)
        self.assertEqual(base.shown_count, 0)

        shown_once = copy.copy(base)
        shown_once.shown_count = 1
        shown_more = copy.copy(base)
        shown_more.shown_count = 7
        shown_more.updated_at = base.updated_at + 9999

        # First show flips was_shown (and the signature)...
        self.assertNotEqual(_recommendation_signature(base), _recommendation_signature(shown_once))
        # ...but later shown_count bumps and updated_at-only changes do not.
        self.assertEqual(_recommendation_signature(shown_once), _recommendation_signature(shown_more))

        # A changed recommended value is meaningful even if status is unchanged.
        different_dose = copy.copy(base)
        different_dose.next_dose_g = base.next_dose_g + 1.0
        self.assertNotEqual(_recommendation_signature(base), _recommendation_signature(different_dose))

        # A lifecycle status change is meaningful.
        accepted = copy.copy(base)
        accepted.status = RecommendationStatus.ACCEPTED
        self.assertNotEqual(_recommendation_signature(base), _recommendation_signature(accepted))


if __name__ == "__main__":
    unittest.main()

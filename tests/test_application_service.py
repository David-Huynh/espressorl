from __future__ import annotations

import copy
import unittest

from espresso_rl.application.services import EspressoRLService, _recommendation_signature
from espresso_rl.domain.events import (
    MachineStateEvent,
    RecommendationApplyEvent,
    RecommendationDecisionEvent,
    ShotCorrectionEvent,
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
from espresso_rl.domain.optimization import OptimizationContext
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
        grinder_context_id: str | None = None,
    ) -> list[ShotRecord]:
        rows = [
            row
            for row in self.rows.values()
            if row.install_id == install_id
            and row.machine_id == machine_id
            and row.bean_context_id == bean_context_id
            and row.grinder_context_id == grinder_context_id
        ]
        return sorted(rows, key=lambda row: row.timestamp)[-limit:]

    def list_machine_shots(
        self,
        install_id: str,
        machine_id: str,
        limit: int = 500,
    ) -> list[ShotRecord]:
        rows = [
            row
            for row in self.rows.values()
            if row.install_id == install_id and row.machine_id == machine_id
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
        grinder_context_id: str | None = None,
    ) -> Recommendation | None:
        rows = [
            row
            for row in self.rows.values()
            if row.install_id == install_id
            and row.machine_id == machine_id
            and row.bean_context_id == bean_context_id
            and row.grinder_context_id == grinder_context_id
            and row.active_at(now)
        ]
        return max(rows, key=lambda row: row.created_at) if rows else None

    def get_latest(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        grinder_context_id: str | None = None,
    ) -> Recommendation | None:
        rows = [
            row
            for row in self.rows.values()
            if row.install_id == install_id
            and row.machine_id == machine_id
            and row.bean_context_id == bean_context_id
            and row.grinder_context_id == grinder_context_id
        ]
        return max(rows, key=lambda row: row.created_at) if rows else None

    def supersede_active(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        now: int,
        except_recommendation_id: str | None = None,
        grinder_context_id: str | None = None,
    ) -> None:
        for row in self.rows.values():
            if row.recommendation_id == except_recommendation_id:
                continue
            if (
                row.install_id == install_id
                and row.machine_id == machine_id
                and row.bean_context_id == bean_context_id
                and row.grinder_context_id == grinder_context_id
            ):
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

    def count_by_status(self) -> dict[UploadQueueStatus, int]:
        counts: dict[UploadQueueStatus, int] = {}
        for item in self.rows.values():
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def list_by_status(self, status: UploadQueueStatus, limit: int = 100) -> list[UploadQueueItem]:
        return [
            item
            for item in sorted(self.rows.values(), key=lambda row: row.updated_at, reverse=True)
            if item.status == status
        ][:limit]

    def requeue(
        self,
        upload_id: str,
        now: int,
        error_message: str | None = None,
    ) -> None:
        item = self.rows[upload_id]
        item.status = UploadQueueStatus.PENDING
        item.attempt_count = 0
        item.last_attempt_at = None
        item.next_retry_at = None
        item.error_message = error_message
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

    def count_by_status(self) -> dict[UploadQueueStatus, int]:
        counts: dict[UploadQueueStatus, int] = {}
        for item in self.calls:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def list_by_status(self, status: UploadQueueStatus, limit: int = 100) -> list[UploadQueueItem]:
        return [item for item in self.calls if item.status == status][:limit]

    def requeue(
        self,
        upload_id: str,
        now: int,
        error_message: str | None = None,
    ) -> None:
        for item in self.calls:
            if item.upload_id == upload_id:
                item.status = UploadQueueStatus.PENDING
                item.attempt_count = 0
                item.last_attempt_at = None
                item.next_retry_at = None
                item.error_message = error_message
                item.updated_at = now
                return
        raise ValueError(f"unknown upload_id {upload_id}")

    def for_record(self, local_record_type: str, local_record_id: str | None = None) -> list[UploadQueueItem]:
        return [
            item
            for item in self.calls
            if item.local_record_type == local_record_type
            and (local_record_id is None or item.local_record_id == local_record_id)
        ]


class CapturingOptimizer:
    def __init__(self) -> None:
        self.contexts: list[OptimizationContext] = []

    def recommend(self, context: OptimizationContext) -> Recommendation:
        self.contexts.append(context)
        current = context.current_recipe
        index = len(self.contexts)
        return Recommendation(
            recommendation_id=f"rec_capture_{index}",
            created_at=context.now,
            updated_at=context.now,
            expires_at=context.now + 3600,
            install_id=context.install_id,
            machine_id=context.machine_id,
            bean_context_id=context.bean_context_id,
            grinder_context_id=context.grinder_context_id,
            grind_delta_steps_from_current=0,
            grind_delta_um_from_current=0.0,
            projected_relative_step_from_reference=current.relative_grind_steps_from_reference,
            projected_relative_grind_um_from_reference=current.relative_grind_um_from_reference,
            next_dose_g=current.dose_g,
            target_yield_g=current.target_yield_g,
            target_ratio=current.target_ratio or current.target_yield_g / current.dose_g,
            mode=RecommendationMode.ZERO_IMMEDIATE_BO,
            confidence=0.5,
            reason="captured",
            status=RecommendationStatus.PENDING,
            source_shot_id=context.shots[-1].shot_id if context.shots else None,
        )


def idle_event(timestamp: int, **overrides) -> MachineStateEvent:
    base = {
        "install_id": "install_1",
        "machine_id": "machine_1",
        "machine_adapter": "gaggimate",
        "timestamp": timestamp,
        "state": MachineState.IDLE,
        "bean_context_id": "bean_1",
        "relative_grind_steps_from_reference": 42,
        "microns_per_step": 12.5,
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
        "microns_per_step": 12.5,
        "relative_grind_steps_from_reference": 42,
        "dose_in_g": 18.0,
        "target_yield_g": 36.0,
        "beverage_out_g": 36.0,
        "shot_time_s": 30.0,
        "bean_context_id": "bean_1",
    }
    base.update(overrides)
    return ShotProfileEvent(**base)


def feedback_event(shot_id: str, timestamp: int, **overrides) -> ShotFeedbackEvent:
    base = {
        "shot_id": shot_id,
        "install_id": "install_1",
        "machine_id": "machine_1",
        "timestamp": timestamp,
        "rating": 3,
    }
    base.update(overrides)
    return ShotFeedbackEvent(**base)


def ingest_and_feedback(
    service: EspressoRLService,
    event: ShotProfileEvent,
    *,
    rating: int = 3,
    taste_tags: list[str] | None = None,
) -> Recommendation:
    result = service.ingest_shot_profile(event)
    if result.shot is None:
        raise AssertionError("expected shot to be stored")
    feedback = service.record_feedback(
        feedback_event(
            event.shot_id,
            event.timestamp + 1,
            recommendation_id=event.recommendation_id,
            rating=rating,
            taste_tags=taste_tags or [],
        )
    )
    if feedback.recommendation is None:
        raise AssertionError("expected latest-shot feedback to generate a recommendation")
    return feedback.recommendation


class ApplicationServiceTests(unittest.TestCase):
    def test_idle_without_rated_shot_does_not_create_baseline_recommendation(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        self.assertIsNone(service.handle_machine_state(idle_event(1)))
        self.assertEqual(recs.rows, {})

    def test_feedback_generates_bounded_second_shot_recommendation(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        result = service.ingest_shot_profile(shot_event("shot_1", 1))

        self.assertIsNone(result.recommendation)
        self.assertIsNone(service.handle_machine_state(idle_event(2)))
        feedback = service.record_feedback(feedback_event("shot_1", 3, rating=4))
        self.assertEqual(feedback.recommendation.mode, RecommendationMode.ZERO_IMMEDIATE_BO)
        self.assertEqual(feedback.recommendation.next_dose_g, 18.0)
        self.assertLessEqual(abs(feedback.recommendation.grind_delta_steps_from_current), 2)
        self.assertLessEqual(abs(feedback.recommendation.target_yield_g - 36.0), 4.0)
        self.assertTrue(feedback.shot.feedback_recorded)
        self.assertLess(shots.get("shot_1").reward_confidence, 1.0)  # type: ignore[union-attr]

    def test_previous_bag_same_bean_history_becomes_optimizer_prior_points(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        optimizer = CapturingOptimizer()
        service = EspressoRLService(shots, recs, optimizer, clock=lambda: 10)

        ingest_and_feedback(
            service,
            shot_event(
                "old_bag_good",
                1,
                bean_context_id="bean_lavazza_super_crema_100_001",
                grinder_context_id="grinder_jx_pro",
                relative_grind_steps_from_reference=40,
                target_yield_g=38.0,
            ),
            rating=5,
        )
        ingest_and_feedback(
            service,
            shot_event(
                "current_bag_first",
                10,
                bean_context_id="bean_lavazza_super_crema_200_001",
                bean_context_name="Lavazza Super Crema",
                grinder_context_id="grinder_jx_pro",
                relative_grind_steps_from_reference=42,
                target_yield_g=36.0,
            ),
            rating=3,
        )

        context = optimizer.contexts[-1]
        self.assertEqual([shot.shot_id for shot in context.shots], ["current_bag_first"])
        self.assertEqual(len(context.prior_points), 1)
        point = context.prior_points[0]
        self.assertEqual(point.source, "local_bean_history")
        self.assertEqual(point.grind_delta_um_from_current, -25.0)
        self.assertEqual(point.target_yield_g, 38.0)

    def test_previous_bag_priors_are_bean_and_grinder_isolated(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        optimizer = CapturingOptimizer()
        service = EspressoRLService(shots, recs, optimizer, clock=lambda: 10)

        ingest_and_feedback(
            service,
            shot_event(
                "same_bean_other_grinder",
                1,
                bean_context_id="bean_lavazza_100_001",
                bean_context_name="Lavazza",
                grinder_context_id="grinder_other",
            ),
            rating=5,
        )
        ingest_and_feedback(
            service,
            shot_event(
                "other_bean_same_grinder",
                3,
                bean_context_id="bean_onyx_100_001",
                bean_context_name="Onyx",
                grinder_context_id="grinder_jx_pro",
            ),
            rating=5,
        )
        ingest_and_feedback(
            service,
            shot_event(
                "current_bag_first",
                10,
                bean_context_id="bean_lavazza_200_001",
                bean_context_name="Lavazza",
                grinder_context_id="grinder_jx_pro",
            ),
            rating=3,
        )

        self.assertEqual(list(optimizer.contexts[-1].prior_points), [])

    def test_skipped_feedback_is_complete_and_generates_from_the_latest_shot(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)
        service.ingest_shot_profile(shot_event("shot_1", 1))

        result = service.record_feedback(
            feedback_event("shot_1", 2, rating=None, skipped=True)
        )

        self.assertTrue(result.shot.feedback_recorded)
        self.assertIsNone(result.shot.human_rating)
        self.assertEqual(result.recommendation.source_shot_id, "shot_1")

    def test_rating_opt_out_uses_profile_evidence_without_waiting_for_feedback(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        result = service.ingest_shot_profile(
            shot_event("shot_1", 1, rating_prompt_allowed=False)
        )

        self.assertTrue(result.shot.feedback_recorded)
        self.assertIsNotNone(result.recommendation)
        self.assertEqual(result.recommendation.source_shot_id, "shot_1")

    def test_feedback_owner_mismatch_is_rejected_without_training(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)
        service.ingest_shot_profile(shot_event("shot_1", 1))

        with self.assertRaisesRegex(ValueError, "does not match the stored shot owner"):
            service.record_feedback(
                feedback_event("shot_1", 2, machine_id="other_machine", rating=5)
            )

        self.assertFalse(shots.get("shot_1").feedback_recorded)  # type: ignore[union-attr]
        self.assertEqual(recs.rows, {})

    def test_late_feedback_updates_history_without_replacing_latest_shot_recommendation(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)
        service.ingest_shot_profile(shot_event("shot_1", 1))
        service.ingest_shot_profile(shot_event("shot_2", 3))

        result = service.record_feedback(feedback_event("shot_1", 4, rating=5))

        self.assertTrue(result.shot.feedback_recorded)
        self.assertIsNone(result.recommendation)
        self.assertEqual(recs.rows, {})

    def test_duplicate_unchanged_feedback_reuses_the_same_recommendation(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)
        service.ingest_shot_profile(shot_event("shot_1", 1))

        first = service.record_feedback(feedback_event("shot_1", 2, rating=4))
        duplicate = service.record_feedback(feedback_event("shot_1", 2, rating=4))

        self.assertEqual(
            duplicate.recommendation.recommendation_id,
            first.recommendation.recommendation_id,
        )
        self.assertEqual(len(recs.rows), 1)

    def test_same_bean_different_grinders_have_isolated_bo_contexts(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        service.ingest_shot_profile(shot_event("shot_a", 1, grinder_context_id="grinder_a"))
        rec_a = service.record_feedback(feedback_event("shot_a", 2, rating=4)).recommendation
        service.ingest_shot_profile(
            shot_event("shot_b", 3, grinder_context_id="grinder_b", relative_grind_steps_from_reference=52)
        )
        rec_b = service.record_feedback(feedback_event("shot_b", 4, rating=2)).recommendation

        self.assertIsNotNone(rec_a)
        self.assertIsNotNone(rec_b)
        self.assertEqual(rec_a.grinder_context_id, "grinder_a")  # type: ignore[union-attr]
        self.assertEqual(rec_b.grinder_context_id, "grinder_b")  # type: ignore[union-attr]
        self.assertEqual(rec_a.source_shot_id, "shot_a")  # type: ignore[union-attr]
        self.assertEqual(rec_b.source_shot_id, "shot_b")  # type: ignore[union-attr]
        self.assertEqual(
            [shot.shot_id for shot in shots.list_recent("install_1", "machine_1", "bean_1", grinder_context_id="grinder_a")],
            ["shot_a"],
        )
        self.assertEqual(
            [shot.shot_id for shot in shots.list_recent("install_1", "machine_1", "bean_1", grinder_context_id="grinder_b")],
            ["shot_b"],
        )
        current_a = recs.get_current("install_1", "machine_1", "bean_1", 20, grinder_context_id="grinder_a")
        current_b = recs.get_current("install_1", "machine_1", "bean_1", 20, grinder_context_id="grinder_b")
        self.assertEqual(current_a.recommendation_id, rec_a.recommendation_id)  # type: ignore[union-attr]
        self.assertEqual(current_b.recommendation_id, rec_b.recommendation_id)  # type: ignore[union-attr]
        self.assertNotEqual(current_a.recommendation_id, current_b.recommendation_id)  # type: ignore[union-attr]

    def test_relative_only_grinder_context_emits_relative_recommendation(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        service.ingest_shot_profile(
            shot_event(
                "shot_1",
                1,
                grinder_calibration_mode="relative_calibrated",
                grinder_context_id="grinder_a",
                relative_grind_steps_from_reference=3,
                current_absolute_step=None,
                absolute_reference_step=None,
            )
        )
        rec = service.record_feedback(feedback_event("shot_1", 2, rating=4)).recommendation

        self.assertIsNotNone(rec)
        self.assertIsNone(rec.current_absolute_step)  # type: ignore[union-attr]
        self.assertIsNone(rec.projected_absolute_step)  # type: ignore[union-attr]
        self.assertEqual(
            rec.projected_relative_step_from_reference,  # type: ignore[union-attr]
            3 + rec.grind_delta_steps_from_current,  # type: ignore[union-attr]
        )

    def test_absolute_display_grinder_context_keeps_optimizer_input_relative(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        service.ingest_shot_profile(
            shot_event(
                "shot_1",
                1,
                grinder_calibration_mode="absolute_display_calibrated",
                grinder_context_id="grinder_a",
                relative_grind_steps_from_reference=3,
                current_absolute_step=42,
                absolute_reference_step=39,
            )
        )
        rec = service.record_feedback(feedback_event("shot_1", 2, rating=4)).recommendation

        self.assertIsNotNone(rec)
        self.assertEqual(rec.current_absolute_step, 42)  # type: ignore[union-attr]
        self.assertEqual(rec.absolute_reference_step, 39)  # type: ignore[union-attr]
        self.assertEqual(
            rec.projected_relative_step_from_reference,  # type: ignore[union-attr]
            3 + rec.grind_delta_steps_from_current,  # type: ignore[union-attr]
        )
        self.assertEqual(
            rec.projected_absolute_step,  # type: ignore[union-attr]
            42 + rec.grind_delta_steps_from_current,  # type: ignore[union-attr]
        )

    def test_feedback_rejects_a_recommendation_id_mismatch(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)
        recommendation = ingest_and_feedback(service, shot_event("shot_1", 1))
        service.ingest_shot_profile(
            shot_event("shot_2", 3, recommendation_id=recommendation.recommendation_id)
        )

        with self.assertRaisesRegex(ValueError, "recommendation_id does not match"):
            service.record_feedback(
                feedback_event("shot_2", 4, recommendation_id="rec_wrong", rating=4)
            )

    def test_ingest_masks_invalid_flow_without_dropping_espresso_shot(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        result = service.ingest_shot_profile(
            shot_event(
                "shot_1",
                1,
                flow=[100_000.0, 100_000.0, 100_000.0],
                target_flow=[2.0, 2.0, 2.0],
            )
        )

        self.assertTrue(result.stored)
        shot = shots.get("shot_1")
        self.assertIsNotNone(shot)
        self.assertFalse(shot.profile_flow_valid)  # type: ignore[union-attr]
        self.assertTrue(shot.profile_flow_masked)  # type: ignore[union-attr]
        self.assertEqual(float(shot.profile[2].max()), 0.0)  # type: ignore[union-attr]
        self.assertEqual(float(shot.profile[3].max()), 0.0)  # type: ignore[union-attr]

    def test_utility_flush_does_not_consume_or_train_active_recommendation(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        active = ingest_and_feedback(service, shot_event("shot_1", 1))
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
        self.assertIsNone(result.shot)
        self.assertFalse(result.stored)
        self.assertEqual(result.dropped_reason, "not_locally_optimizable:utility_flush")
        self.assertIsNone(shots.get("flush_1"))
        self.assertEqual(recs.get(active.recommendation_id).status, RecommendationStatus.PENDING)  # type: ignore[union-attr]

    def test_local_optimization_disabled_drops_shot_without_new_recommendation(self) -> None:
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
        self.assertIsNone(result.shot)
        self.assertFalse(result.stored)
        self.assertEqual(result.dropped_reason, "local_optimization_disabled")
        self.assertEqual(shots.rows, {})
        self.assertEqual(recs.rows, {})

    def test_local_optimization_disabled_shot_is_not_queued_for_community_upload(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        uploads = MemoryUploadQueue()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), upload_queue=uploads, clock=lambda: 10)

        result = service.ingest_shot_profile(
            shot_event(
                "hot_water_1",
                1,
                local_optimization_enabled=False,
                exclude_from_local_optimization=True,
            )
        )

        self.assertIsNone(result.shot)
        self.assertEqual(shots.rows, {})
        self.assertFalse(any(item.local_record_id == "hot_water_1" for item in uploads.rows.values()))

    def test_shot_correction_excludes_stored_shot_from_local_optimization(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        uploads = MemoryUploadQueue()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), upload_queue=uploads, clock=lambda: 10)

        service.ingest_shot_profile(shot_event("shot_1", 1))
        corrected = service.record_shot_correction(
            ShotCorrectionEvent(
                shot_id="shot_1",
                install_id="install_1",
                machine_id="machine_1",
                timestamp=2,
                exclude_from_local_optimization=True,
                correction_tags=["bad_puck_prep"],
            )
        )

        self.assertTrue(corrected.exclude_from_local_optimization)
        self.assertEqual(corrected.optimization_weight, 0.0)
        self.assertEqual(corrected.recommendation_attribution_weight, 0.0)
        self.assertEqual(corrected.recommendation_followed, FollowThroughState.NOT_FOLLOWED)
        self.assertIn("channeling_suspected", corrected.taste_tags)
        self.assertEqual(shots.get("shot_1").optimization_weight, 0.0)  # type: ignore[union-attr]
        self.assertTrue(uploads.list_ready(now=20))  # corrected espresso snapshot is still uploadable

    def test_shot_correction_records_variable_specific_not_followed(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        rec = ingest_and_feedback(service, shot_event("shot_1", 1))
        service.record_recommendation_decision(
            RecommendationDecisionEvent(
                recommendation_id=rec.recommendation_id,
                decision=RecommendationDecision.ACCEPTED,
                timestamp=2,
            )
        )
        service.ingest_shot_profile(
            shot_event(
                "shot_2",
                3,
                recommendation_id=rec.recommendation_id,
                relative_grind_steps_from_reference=rec.projected_relative_step_from_reference,
                dose_in_g=rec.next_dose_g,
                target_yield_g=rec.target_yield_g,
                beverage_out_g=rec.target_yield_g,
            )
        )

        corrected = service.record_shot_correction(
            ShotCorrectionEvent(
                shot_id="shot_2",
                install_id="install_1",
                machine_id="machine_1",
                timestamp=4,
                grind_followed=False,
                dose_followed=True,
                yield_followed=True,
                correction_tags=["did_not_follow_grind", "changed_manually"],
            )
        )

        self.assertFalse(corrected.exclude_from_local_optimization)
        self.assertFalse(corrected.grind_followed)
        self.assertTrue(corrected.dose_followed)
        self.assertTrue(corrected.yield_followed)
        self.assertEqual(corrected.grind_recommendation_trust, 0.0)
        self.assertGreater(corrected.recommendation_attribution_weight, 0.0)
        self.assertLess(corrected.recommendation_attribution_weight, 1.0)
        self.assertEqual(corrected.recommendation_followed, FollowThroughState.PARTIALLY_FOLLOWED)

    def test_utility_shot_is_not_queued_for_community_upload(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        uploads = MemoryUploadQueue()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), upload_queue=uploads, clock=lambda: 10)

        result = service.ingest_shot_profile(
            shot_event(
                "flush_1",
                1,
                shot_time_s=5.0,
                beverage_out_g=1.0,
                weight=[0.0, 0.5, 1.0],
            )
        )

        self.assertIsNone(result.shot)
        self.assertEqual(shots.rows, {})
        self.assertFalse(any(item.local_record_id == "flush_1" for item in uploads.rows.values()))

    def test_decision_and_actual_shot_data_drive_follow_through(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        first = ingest_and_feedback(service, shot_event("shot_1", 1))
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
                relative_grind_steps_from_reference=first.projected_relative_step_from_reference,
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
        self.assertGreater(updated.shot.reward or 0.0, 0.8)
        self.assertEqual(updated.shot.reward_confidence, 1.0)

    def test_close_yield_and_small_negative_tare_remain_valid_training_data(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)
        first = ingest_and_feedback(
            service,
            shot_event("shot_1", 1, target_yield_g=38.0, beverage_out_g=38.0),
        )
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
                relative_grind_steps_from_reference=first.projected_relative_step_from_reference,
                dose_in_g=first.next_dose_g,
                target_yield_g=38.0,
                beverage_out_g=37.5,
                weight=[-0.1, 10.0, 37.5],
            )
        )
        stored = shots.get("shot_2")

        self.assertEqual(stored.recommendation_followed, FollowThroughState.FOLLOWED)
        feedback = service.record_feedback(feedback_event("shot_2", 4, rating=5))
        self.assertTrue(feedback.shot.feedback_recorded)
        self.assertEqual(feedback.recommendation.source_shot_id, "shot_2")

    def test_machine_idle_shows_current_recommendation(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        rec = ingest_and_feedback(service, shot_event("shot_1", 1))
        shown = service.handle_machine_state(
            MachineStateEvent(
                install_id="install_1",
                machine_id="machine_1",
                machine_adapter="gaggimate",
                timestamp=2,
                state=MachineState.IDLE,
                bean_context_id="bean_1",
                relative_grind_steps_from_reference=42,
                microns_per_step=12.5,
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

        old = ingest_and_feedback(service, shot_event("shot_1", 1))
        new = service.handle_machine_state(
            MachineStateEvent(
                install_id="install_1",
                machine_id="machine_1",
                machine_adapter="gaggimate",
                timestamp=2,
                state=MachineState.IDLE,
                bean_context_id="bean_1",
                relative_grind_steps_from_reference=old.projected_relative_step_from_reference + 10,
                microns_per_step=12.5,
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

        rec = ingest_and_feedback(service, shot_event("shot_1", 1))
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
                relative_grind_steps_from_reference=rec.projected_relative_step_from_reference,
                microns_per_step=12.5,
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

        rec = ingest_and_feedback(service, shot_event("shot_1", 1))
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
                manual_fields=["projected_relative_step_from_reference", "next_dose_g"],
                message="Target yield applied; grind and dose are manual.",
            )
        )

        self.assertEqual(applied.status, RecommendationStatus.ACCEPTED)
        self.assertEqual(applied.apply_status, RecommendationApplyStatus.PARTIALLY_APPLIED)
        self.assertEqual(applied.applied_fields["target_yield_g"], rec.target_yield_g)
        self.assertEqual(applied.manual_fields, ["projected_relative_step_from_reference", "next_dose_g"])
        self.assertIsNone(applied.used_at)
        self.assertEqual(recs.get(rec.recommendation_id).status, RecommendationStatus.ACCEPTED)  # type: ignore[union-attr]

    def test_no_answer_reprompts_until_user_decides(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

        rec = ingest_and_feedback(service, shot_event("shot_1", 1))
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
                    relative_grind_steps_from_reference=42,
                    microns_per_step=12.5,
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

        rec = ingest_and_feedback(service, shot_event("shot_1", 1))
        stored = recs.get(rec.recommendation_id)
        stored.status = RecommendationStatus.EXPIRED  # type: ignore[union-attr]
        recs.upsert(stored)  # type: ignore[arg-type]

        service.ingest_shot_profile(
            shot_event(
                "shot_2",
                2,
                recommendation_id=rec.recommendation_id,
                relative_grind_steps_from_reference=rec.projected_relative_step_from_reference,
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
        self.assertFalse(any(item.local_record_type == "recommendation" for item in uploads.rows.values()))
        feedback = service.record_feedback(feedback_event("shot_1", 2, rating=4))
        self.assertTrue(any(item.local_record_type == "recommendation" for item in uploads.rows.values()))
        payloads = [item.payload_json for item in uploads.rows.values()]
        self.assertTrue(any('"shot_id":"shot_1"' in payload for payload in payloads))
        self.assertTrue(
            any(feedback.recommendation.recommendation_id in payload for payload in payloads)
        )

    def test_idle_reshows_do_not_reenqueue_recommendation(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        queue = RecordingUploadQueue()
        service = EspressoRLService(
            shots, recs, ConservativeBOOptimizer(), upload_queue=queue, clock=lambda: 10
        )

        rec = ingest_and_feedback(service, shot_event("shot_1", 1))
        for ts in range(2, 9):
            service.handle_machine_state(idle_event(ts))

        # Only the create and the first show are meaningful; the six later idle
        # re-marks (shown_count 2..7) are incidental churn and must not upload.
        self.assertEqual(len(queue.for_record("recommendation", rec.recommendation_id)), 2)
        # The shot uploads at ingest and after feedback; idle pings never touch it.
        self.assertEqual(len(queue.for_record("shot")), 2)

    def test_recommendation_lifecycle_transitions_each_enqueue_once(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        queue = RecordingUploadQueue()
        service = EspressoRLService(
            shots, recs, ConservativeBOOptimizer(), upload_queue=queue, clock=lambda: 10
        )

        rec = ingest_and_feedback(service, shot_event("shot_1", 1))  # created after feedback
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
                manual_fields=["projected_relative_step_from_reference"],
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

        rec = ingest_and_feedback(service, shot_event("shot_1", 1))
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

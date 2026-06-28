from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from espresso_rl.adapters.sqlite_repositories import (
    SQLiteLocalDataRepository,
    SQLiteRecommendationRepository,
    SQLiteShotRepository,
    SQLiteStore,
    SQLiteUploadQueueRepository,
)
from espresso_rl.application.local_data import LocalDataService
from espresso_rl.domain.models import (
    Recommendation,
    RecommendationMode,
    RecommendationStatus,
    ShotRecord,
    ShotType,
    UploadQueueItem,
    UploadQueueStatus,
)


class LocalDataManagementTests(unittest.TestCase):
    def test_status_marks_only_real_optimizer_shots_as_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                local = SQLiteLocalDataRepository(store)
                queue = SQLiteUploadQueueRepository(store)
                shots.upsert(_shot("espresso_1", shot_type=ShotType.ESPRESSO, optimization_weight=1.0))
                shots.upsert(_shot("flush_1", shot_type=ShotType.UTILITY_FLUSH, exclude=True))
                queue.enqueue(_upload("flush_upload", "flush_1"))
                service = LocalDataService(local, install_id="install_1", machine_id="machine_1", clock=lambda: 20)

                status = service.status(limit=10).to_dict()

                recent = {shot["shot_id"]: shot for shot in status["recent_shots"]}
                self.assertTrue(recent["espresso_1"]["included_in_optimizer"])
                self.assertFalse(recent["flush_1"]["included_in_optimizer"])
                self.assertTrue(recent["flush_1"]["rejected_upload"])
                self.assertEqual(status["contexts"][0]["optimizer_shot_count"], 1)

    def test_purge_useless_shots_keeps_optimizer_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                local = SQLiteLocalDataRepository(store)
                queue = SQLiteUploadQueueRepository(store)
                shots.upsert(_shot("espresso_1", shot_type=ShotType.ESPRESSO, optimization_weight=1.0))
                shots.upsert(_shot("excluded_1", shot_type=ShotType.ESPRESSO, exclude=True))
                shots.upsert(_shot("flush_1", shot_type=ShotType.UTILITY_FLUSH, exclude=True))
                queue.enqueue(_upload("flush_upload", "flush_1"))
                service = LocalDataService(local, install_id="install_1", machine_id="machine_1", clock=lambda: 20)

                dry_run = service.purge_useless_shots(dry_run=True).to_dict()
                result = service.purge_useless_shots(dry_run=False).to_dict()

                self.assertEqual(dry_run["counts"]["shots"], 2)
                self.assertEqual(result["counts"]["shots"], 2)
                self.assertIsNotNone(shots.get("espresso_1"))
                self.assertIsNone(shots.get("excluded_1"))
                self.assertIsNone(shots.get("flush_1"))
                self.assertEqual(store.conn.execute("SELECT COUNT(*) AS count FROM upload_queue").fetchone()["count"], 0)

    def test_reset_optimizer_context_excludes_shots_and_supersedes_active_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                recommendations = SQLiteRecommendationRepository(store)
                local = SQLiteLocalDataRepository(store)
                shots.upsert(_shot("espresso_1", shot_type=ShotType.ESPRESSO, optimization_weight=1.0))
                recommendations.upsert(_recommendation("rec_1"))
                service = LocalDataService(local, install_id="install_1", machine_id="machine_1", clock=lambda: 20)

                result = service.reset_optimizer_context("bean_1").to_dict()

                self.assertEqual(result["counts"]["shots"], 1)
                self.assertEqual(result["counts"]["recommendations"], 1)
                shot = shots.get("espresso_1")
                rec = recommendations.get("rec_1")
                self.assertTrue(shot.exclude_from_local_optimization)  # type: ignore[union-attr]
                self.assertEqual(shot.optimization_weight, 0.0)  # type: ignore[union-attr]
                self.assertEqual(rec.status, RecommendationStatus.SUPERSEDED)  # type: ignore[union-attr]

    def test_reset_all_deletes_machine_local_data_and_queued_uploads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                recommendations = SQLiteRecommendationRepository(store)
                local = SQLiteLocalDataRepository(store)
                queue = SQLiteUploadQueueRepository(store)
                shots.upsert(_shot("espresso_1", shot_type=ShotType.ESPRESSO, optimization_weight=1.0))
                recommendations.upsert(_recommendation("rec_1"))
                queue.enqueue(_upload("shot_upload", "espresso_1", local_record_type="shot"))
                queue.enqueue(_upload("rec_upload", "rec_1", local_record_type="recommendation"))
                store.conn.execute(
                    """
                    INSERT INTO dreamer_shadow_evaluations (
                        evaluation_id, install_id, machine_id, bean_context_id, grinder_context_id,
                        source_timestamp, status, payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "eval_1",
                        "install_1",
                        "machine_1",
                        "bean_1",
                        "",
                        10,
                        "ok",
                        "{}",
                        10,
                        10,
                    ),
                )
                store.conn.execute(
                    """
                    INSERT INTO dreamer_shadow_quality_reports (
                        report_id, install_id, machine_id, bean_context_id, grinder_context_id,
                        checkpoint_artifact_sha256, checkpoint_inference_probe_sha256,
                        overall_status, payload_json, generated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "report_1",
                        "install_1",
                        "machine_1",
                        "bean_1",
                        "",
                        "a" * 64,
                        "b" * 64,
                        "insufficient_data",
                        "{}",
                        10,
                    ),
                )
                store.conn.commit()
                service = LocalDataService(local, install_id="install_1", machine_id="machine_1", clock=lambda: 20)

                dry_run = service.reset_all(dry_run=True).to_dict()
                result = service.reset_all().to_dict()

                self.assertEqual(dry_run["counts"]["shots"], 1)
                self.assertEqual(dry_run["counts"]["recommendations"], 1)
                self.assertEqual(dry_run["counts"]["upload_queue"], 2)
                self.assertEqual(dry_run["counts"]["dreamer_shadow_evaluations"], 1)
                self.assertEqual(dry_run["counts"]["dreamer_shadow_quality_reports"], 1)
                self.assertEqual(result["counts"], dry_run["counts"])
                self.assertIsNone(shots.get("espresso_1"))
                self.assertIsNone(recommendations.get("rec_1"))
                self.assertEqual(store.conn.execute("SELECT COUNT(*) AS count FROM upload_queue").fetchone()["count"], 0)
                self.assertEqual(
                    store.conn.execute("SELECT COUNT(*) AS count FROM dreamer_shadow_evaluations").fetchone()["count"],
                    0,
                )
                self.assertEqual(
                    store.conn.execute("SELECT COUNT(*) AS count FROM dreamer_shadow_quality_reports").fetchone()["count"],
                    0,
                )


def _shot(
    shot_id: str,
    *,
    shot_type: ShotType,
    exclude: bool = False,
    optimization_weight: float = 1.0,
) -> ShotRecord:
    return ShotRecord(
        shot_id=shot_id,
        timestamp=10,
        install_id="install_1",
        machine_id="machine_1",
        machine_adapter="gaggimate",
        bean_context_id="bean_1",
        profile=np.zeros((5, 100), dtype=np.float32),
        microns_per_step=12.5,
        dose_in_g=18.0,
        target_yield_g=36.0,
        shot_type=shot_type,
        exclude_from_local_optimization=exclude,
        optimization_weight=optimization_weight,
        feedback_recorded=True,
        reward=0.5,
        reward_confidence=0.5,
        created_at=10,
        updated_at=10,
    )


def _recommendation(recommendation_id: str) -> Recommendation:
    return Recommendation(
        recommendation_id=recommendation_id,
        created_at=10,
        updated_at=10,
        expires_at=100,
        install_id="install_1",
        machine_id="machine_1",
        bean_context_id="bean_1",
        grind_delta_steps_from_current=0,
        grind_delta_um_from_current=0,
        projected_relative_step_from_reference=42,
        projected_relative_grind_um_from_reference=525,
        next_dose_g=18,
        target_yield_g=36,
        target_ratio=2,
        mode=RecommendationMode.LOCAL_BO,
        confidence=0.5,
        reason="test",
        status=RecommendationStatus.PENDING,
    )


def _upload(upload_id: str, local_record_id: str, *, local_record_type: str = "shot") -> UploadQueueItem:
    return UploadQueueItem(
        upload_id=upload_id,
        local_record_type=local_record_type,
        local_record_id=local_record_id,
        payload_hash=f"{upload_id}_hash",
        payload_json='{"event_type":"shot_record"}',
        status=UploadQueueStatus.REJECTED,
        attempt_count=1,
        error_message="preflight failed",
        created_at=10,
        updated_at=10,
    )


if __name__ == "__main__":
    unittest.main()

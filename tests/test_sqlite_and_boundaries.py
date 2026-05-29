from __future__ import annotations

import ast
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
from espresso_rl.domain.events import RecommendationApplyEvent, ShotProfileEvent
from espresso_rl.domain.models import RecommendationApplyStatus, UploadQueueItem, UploadQueueStatus
from espresso_rl.optimizers.conservative_bo import ConservativeBOOptimizer
from espresso_rl.adapters.supabase_upload import UploadQueueWorker


def shot_event() -> ShotProfileEvent:
    return ShotProfileEvent(
        shot_id="shot_1",
        install_id="install_1",
        machine_id="machine_1",
        machine_adapter="gaggimate",
        timestamp=1,
        time_ms=[0, 500, 1000],
        pressure=[0.0, 8.0, 9.0],
        target_pressure=[0.0, 8.0, 9.0],
        flow=[0.0, 2.0, 2.0],
        target_flow=[0.0, 2.0, 2.0],
        weight=[0.0, 10.0, 36.0],
        grinder_step_size_um=12.5,
        grind_steps=42,
        dose_in_g=18.0,
        target_yield_g=36.0,
        beverage_out_g=36.0,
        shot_time_s=30.0,
    )


class SQLiteAndBoundaryTests(unittest.TestCase):
    def test_sqlite_repositories_round_trip_core_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            shots = SQLiteShotRepository(store)
            recs = SQLiteRecommendationRepository(store)
            service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

            result = service.ingest_shot_profile(shot_event())
            service.record_recommendation_apply(
                RecommendationApplyEvent(
                    recommendation_id=result.recommendation.recommendation_id,
                    status=RecommendationApplyStatus.MANUAL_REQUIRED,
                    timestamp=2,
                    manual_fields=["next_grind_steps", "next_dose_g"],
                )
            )
            stored_shot = shots.get("shot_1")
            stored_rec = recs.get(result.recommendation.recommendation_id)

            self.assertIsNotNone(stored_shot)
            self.assertIsNotNone(stored_rec)
            self.assertEqual(stored_shot.profile.shape, (5, 100))  # type: ignore[union-attr]
            self.assertEqual(stored_rec.reason, result.recommendation.reason)  # type: ignore[union-attr]
            self.assertEqual(stored_rec.apply_status, RecommendationApplyStatus.MANUAL_REQUIRED)  # type: ignore[union-attr]
            self.assertEqual(stored_rec.manual_fields, ["next_grind_steps", "next_dose_g"])  # type: ignore[union-attr]

    def test_sqlite_upload_queue_tracks_retry_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            queue = SQLiteUploadQueueRepository(store)
            queue.enqueue(
                UploadQueueItem(
                    upload_id="upload_1",
                    local_record_type="shot",
                    local_record_id="shot_1",
                    payload_hash="abc",
                    payload_json='{"event_type":"shot_record"}',
                    status=UploadQueueStatus.PENDING,
                    created_at=1,
                    updated_at=1,
                )
            )

            self.assertEqual([item.upload_id for item in queue.list_ready(now=2)], ["upload_1"])
            queue.update_status(
                upload_id="upload_1",
                status=UploadQueueStatus.FAILED,
                now=3,
                error_message="network",
                next_retry_at=10,
            )
            self.assertEqual(queue.list_ready(now=4), [])
            ready = queue.list_ready(now=10)
            self.assertEqual(ready[0].attempt_count, 1)
            self.assertEqual(ready[0].error_message, "network")

    def test_upload_worker_marks_successful_records_uploaded(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.uploaded: list[str] = []

            def upload(self, item: UploadQueueItem) -> None:
                self.uploaded.append(item.upload_id)

        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            queue = SQLiteUploadQueueRepository(store)
            queue.enqueue(
                UploadQueueItem(
                    upload_id="upload_1",
                    local_record_type="shot",
                    local_record_id="shot_1",
                    payload_hash="abc",
                    payload_json='{"event_type":"shot_record"}',
                    status=UploadQueueStatus.PENDING,
                    created_at=1,
                    updated_at=1,
                )
            )
            client = FakeClient()
            worker = UploadQueueWorker(queue, client, clock=lambda: 5)

            self.assertEqual(worker.run_once(), 1)
            self.assertEqual(client.uploaded, ["upload_1"])
            self.assertEqual(queue.list_ready(now=6), [])

    def test_core_layers_do_not_import_adapters_or_infrastructure(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "espresso_rl"
        core_dirs = ["domain", "application", "optimizers", "ports"]
        forbidden = {
            "espresso_rl.adapters",
            "paho",
            "sqlite3",
            "supabase",
        }
        violations: list[str] = []
        for dirname in core_dirs:
            for path in (root / dirname).rglob("*.py"):
                tree = ast.parse(path.read_text(), filename=str(path))
                for node in ast.walk(tree):
                    module = None
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module = alias.name
                            if any(module == item or module.startswith(f"{item}.") for item in forbidden):
                                violations.append(f"{path}: import {module}")
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        module = node.module
                        if any(module == item or module.startswith(f"{item}.") for item in forbidden):
                            violations.append(f"{path}: from {module}")
        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()

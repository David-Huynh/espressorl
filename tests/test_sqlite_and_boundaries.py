from __future__ import annotations

import ast
import tempfile
import threading
import unittest
from pathlib import Path

from espresso_rl.config import Config
from espresso_rl.adapters.sqlite_repositories import (
    SQLiteRecommendationRepository,
    SQLiteShotRepository,
    SQLiteStore,
    SQLiteUploadQueueRepository,
    _shot_to_row,
)
from espresso_rl.adapters.postgres_repositories import _upsert
from espresso_rl.application.services import EspressoRLService
from espresso_rl.domain.events import RecommendationApplyEvent, ShotProfileEvent
from espresso_rl.domain.models import RecommendationApplyStatus, UploadQueueItem, UploadQueueStatus
from espresso_rl.optimizers.conservative_bo import ConservativeBOOptimizer
from espresso_rl.adapters.supabase_upload import (
    MAX_UPLOAD_ATTEMPTS,
    UploadQueueWorker,
    UploadRateLimited,
)
from espresso_rl.main import maybe_start_upload_worker, upload_queue_for_service


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


def queue_item(
    upload_id: str,
    payload_hash: str,
    *,
    status: UploadQueueStatus = UploadQueueStatus.PENDING,
    local_record_type: str = "shot",
    local_record_id: str = "shot_1",
    attempt_count: int = 0,
) -> UploadQueueItem:
    return UploadQueueItem(
        upload_id=upload_id,
        local_record_type=local_record_type,
        local_record_id=local_record_id,
        payload_hash=payload_hash,
        payload_json='{"event_type":"shot_record"}',
        status=status,
        attempt_count=attempt_count,
        created_at=1,
        updated_at=1,
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

    def test_shared_shot_row_uses_boolean_for_postgres_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            service = EspressoRLService(
                SQLiteShotRepository(store),
                SQLiteRecommendationRepository(store),
                ConservativeBOOptimizer(),
                clock=lambda: 10,
            )

            result = service.ingest_shot_profile(shot_event())
            row = _shot_to_row(result.shot)

            self.assertIs(row["raw_profile_available"], True)

    def test_postgres_upsert_rolls_back_failed_transaction(self) -> None:
        class FailingConnection:
            def __init__(self) -> None:
                self.rolled_back = False
                self.committed = False

            def execute(self, *_args: object, **_kwargs: object) -> None:
                raise RuntimeError("database rejected row")

            def commit(self) -> None:
                self.committed = True

            def rollback(self) -> None:
                self.rolled_back = True

        conn = FailingConnection()

        with self.assertRaises(RuntimeError):
            _upsert(conn, "shots", "shot_id", {"shot_id": "shot_1"})

        self.assertTrue(conn.rolled_back)
        self.assertFalse(conn.committed)

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

    def test_enqueue_coalesces_pending_versions_of_same_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            queue = SQLiteUploadQueueRepository(store)
            for payload_hash in ("h1", "h2", "h3"):
                queue.enqueue(queue_item(f"shot_shot_1_{payload_hash}", payload_hash))
            ready = queue.list_ready(now=10)
            self.assertEqual(len(ready), 1)
            self.assertEqual(ready[0].payload_hash, "h3")  # newest queued state wins

    def test_enqueue_skips_content_already_uploaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            queue = SQLiteUploadQueueRepository(store)
            queue.enqueue(queue_item("u_abc", "abc"))
            queue.update_status("u_abc", UploadQueueStatus.UPLOADED, now=2)
            queue.enqueue(queue_item("u_abc", "abc"))  # identical content already sent
            self.assertEqual(queue.list_ready(now=10), [])
            count = store.conn.execute(
                "SELECT COUNT(*) AS c FROM upload_queue WHERE local_record_id='shot_1'"
            ).fetchone()["c"]
            self.assertEqual(count, 1)

    def test_enqueue_rearms_when_content_changes_after_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            queue = SQLiteUploadQueueRepository(store)
            queue.enqueue(queue_item("u_a", "a"))
            queue.update_status("u_a", UploadQueueStatus.UPLOADED, now=2)
            queue.enqueue(queue_item("u_b", "b"))  # e.g. a rating added later
            self.assertEqual([item.upload_id for item in queue.list_ready(now=10)], ["u_b"])
            count = store.conn.execute(
                "SELECT COUNT(*) AS c FROM upload_queue WHERE local_record_id='shot_1'"
            ).fetchone()["c"]
            self.assertEqual(count, 2)  # uploaded 'a' kept as memory + pending 'b'

    def test_enqueue_never_deletes_in_flight_uploading_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            queue = SQLiteUploadQueueRepository(store)
            queue.enqueue(queue_item("u_a", "a"))
            queue.update_status("u_a", UploadQueueStatus.UPLOADING, now=2)
            queue.enqueue(queue_item("u_b", "b"))  # coalesce must leave u_a alone
            statuses = {
                row["upload_id"]: row["status"]
                for row in store.conn.execute(
                    "SELECT upload_id, status FROM upload_queue WHERE local_record_id='shot_1'"
                ).fetchall()
            }
            self.assertEqual(statuses, {"u_a": "uploading", "u_b": "pending"})

    def test_uploading_transition_is_not_counted_as_an_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            queue = SQLiteUploadQueueRepository(store)
            queue.enqueue(queue_item("u_a", "a"))
            queue.update_status("u_a", UploadQueueStatus.UPLOADING, now=2)
            queue.update_status("u_a", UploadQueueStatus.FAILED, now=3, next_retry_at=10)
            ready = queue.list_ready(now=10)
            self.assertEqual(ready[0].attempt_count, 1)  # one failed try, not two

    def test_worker_dead_letters_after_max_attempts(self) -> None:
        class BoomClient:
            def upload(self, item: UploadQueueItem) -> None:
                raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            queue = SQLiteUploadQueueRepository(store)
            queue.enqueue(queue_item("u_a", "a", attempt_count=MAX_UPLOAD_ATTEMPTS - 1))
            worker = UploadQueueWorker(queue, BoomClient(), clock=lambda: 100)
            worker.run_once()
            status = store.conn.execute(
                "SELECT status FROM upload_queue WHERE upload_id='u_a'"
            ).fetchone()["status"]
            self.assertEqual(status, "rejected")
            self.assertEqual(queue.list_ready(now=10_000), [])

    def test_worker_defers_rate_limited_upload_without_charging_attempt(self) -> None:
        class LimitedClient:
            def upload(self, item: UploadQueueItem) -> None:
                raise UploadRateLimited(retry_after=120)

        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            queue = SQLiteUploadQueueRepository(store)
            queue.enqueue(queue_item("u_a", "a"))
            worker = UploadQueueWorker(queue, LimitedClient(), clock=lambda: 1000)
            worker.run_once()
            self.assertEqual(queue.list_ready(now=1001), [])  # deferred past now
            ready = queue.list_ready(now=2000)
            self.assertEqual(len(ready), 1)
            self.assertEqual(ready[0].attempt_count, 0)  # rate limiting never charges an attempt
            self.assertEqual(ready[0].status, UploadQueueStatus.PENDING)

    def test_worker_rate_limited_without_header_defers_to_utc_day_reset(self) -> None:
        class LimitedClient:
            def upload(self, item: UploadQueueItem) -> None:
                raise UploadRateLimited(retry_after=None)

        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            queue = SQLiteUploadQueueRepository(store)
            queue.enqueue(queue_item("u_a", "a"))
            worker = UploadQueueWorker(queue, LimitedClient(), clock=lambda: 1000)
            worker.run_once()
            next_retry_at = store.conn.execute(
                "SELECT next_retry_at FROM upload_queue WHERE upload_id='u_a'"
            ).fetchone()["next_retry_at"]
            self.assertEqual(next_retry_at, 1000 + (86_400 - (1000 % 86_400)))

    def test_admin_role_never_pushes_to_community_upload_queue(self) -> None:
        config = Config(
            mqtt_host="localhost",
            community_upload_enabled=True,
            supabase_ingest_url="https://example.invalid/ingest",
            upload_secret="x" * 32,
            deployment_role="admin",
        )

        self.assertFalse(config.should_enqueue_community_uploads())
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            queue = SQLiteUploadQueueRepository(store)
            self.assertIsNone(upload_queue_for_service(config, queue))
            worker = maybe_start_upload_worker(
                config,
                queue,
                threading.Event(),
            )
        self.assertIsNone(worker)

    def test_postgres_schema_defines_public_and_admin_storage_tables(self) -> None:
        schema = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "espresso_rl"
            / "adapters"
            / "postgres_schema.sql"
        ).read_text()

        for table_name in (
            "shots",
            "recommendations",
            "upload_queue",
            "community_raw_uploads",
            "community_validated_shots",
            "community_recommendations",
            "install_trust_scores",
            "abuse_events",
            "training_dataset",
            "community_priors",
        ):
            self.assertIn(f"CREATE TABLE IF NOT EXISTS {table_name}", schema)

        self.assertIn("PRIMARY KEY (install_id, upload_id)", schema)
        self.assertIn("UNIQUE (install_id, payload_hash)", schema)

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

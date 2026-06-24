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
from espresso_rl.domain.events import RecommendationApplyEvent, ShotFeedbackEvent, ShotProfileEvent
from espresso_rl.domain.models import RecommendationApplyStatus, UploadQueueItem, UploadQueueStatus
from espresso_rl.optimizers.conservative_bo import ConservativeBOOptimizer
from espresso_rl.adapters.supabase_upload import (
    MAX_UPLOAD_ATTEMPTS,
    UploadQueueWorker,
    UploadRateLimited,
)
from espresso_rl.main import build_status_payload, maybe_start_upload_worker, upload_queue_for_service


def shot_event(**overrides) -> ShotProfileEvent:
    base = {
        "shot_id": "shot_1",
        "install_id": "install_1",
        "machine_id": "machine_1",
        "machine_adapter": "gaggimate",
        "timestamp": 1,
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
        "weight_source": "hardware_scale",
        "flow_source": "beverage_weight_derivative",
        "flow_units": "g_per_s",
        "pump_flow_source": "gaggimate_pump_model",
        "pump_flow_units": "ml_per_s",
        "pump_flow_calibration_required": False,
        "profile_id": "profile_1",
        "profile_label": "Cremina lever machine",
        "profile_type": "pro",
        "profile_phase_count": 5,
        "final_phase_index": 3,
        "final_phase_name": "ramp",
        "final_phase_type": "brew",
        "final_phase_elapsed_s": 8.5,
        "final_pump_target": "pressure",
        "final_target_pressure": 9.0,
        "final_target_flow": 0.0,
        "final_valve_open": True,
        "profile_temperature_c": 86.5,
        "final_phase_temperature_c": 86.5,
        "shot_end_state": "manual_or_interrupted",
    }
    base.update(overrides)
    return ShotProfileEvent(**base)


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
            feedback = service.record_feedback(
                ShotFeedbackEvent(
                    shot_id="shot_1",
                    install_id="install_1",
                    machine_id="machine_1",
                    timestamp=2,
                    rating=4,
                )
            )
            service.record_recommendation_apply(
                RecommendationApplyEvent(
                    recommendation_id=feedback.recommendation.recommendation_id,
                    status=RecommendationApplyStatus.MANUAL_REQUIRED,
                    timestamp=2,
                    manual_fields=["next_grind_steps", "next_dose_g"],
                )
            )
            stored_shot = shots.get("shot_1")
            stored_rec = recs.get(feedback.recommendation.recommendation_id)

            self.assertIsNotNone(stored_shot)
            self.assertIsNotNone(stored_rec)
            self.assertEqual(stored_shot.profile.shape, (5, 100))  # type: ignore[union-attr]
            self.assertEqual(stored_shot.weight_source, "hardware_scale")  # type: ignore[union-attr]
            self.assertEqual(stored_shot.profile_label, "Cremina lever machine")  # type: ignore[union-attr]
            self.assertEqual(stored_shot.final_phase_name, "ramp")  # type: ignore[union-attr]
            self.assertTrue(stored_shot.final_valve_open)  # type: ignore[union-attr]
            self.assertEqual(stored_shot.shot_end_state, "manual_or_interrupted")  # type: ignore[union-attr]
            self.assertTrue(stored_shot.feedback_recorded)  # type: ignore[union-attr]
            self.assertEqual(stored_rec.reason, feedback.recommendation.reason)  # type: ignore[union-attr]
            self.assertEqual(stored_rec.apply_status, RecommendationApplyStatus.MANUAL_REQUIRED)  # type: ignore[union-attr]
            self.assertEqual(stored_rec.manual_fields, ["next_grind_steps", "next_dose_g"])  # type: ignore[union-attr]

    def test_sqlite_scopes_recent_shots_and_current_recommendations_by_grinder_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            shots = SQLiteShotRepository(store)
            recs = SQLiteRecommendationRepository(store)
            service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

            service.ingest_shot_profile(
                shot_event(shot_id="shot_a", bean_context_id="bean_1", grinder_context_id="grinder_a")
            )
            rec_a = service.record_feedback(
                ShotFeedbackEvent(
                    shot_id="shot_a",
                    install_id="install_1",
                    machine_id="machine_1",
                    timestamp=2,
                    rating=4,
                )
            ).recommendation
            service.ingest_shot_profile(
                shot_event(
                    shot_id="shot_b",
                    timestamp=3,
                    bean_context_id="bean_1",
                    grinder_context_id="grinder_b",
                    grind_steps=52,
                )
            )
            rec_b = service.record_feedback(
                ShotFeedbackEvent(
                    shot_id="shot_b",
                    install_id="install_1",
                    machine_id="machine_1",
                    timestamp=4,
                    rating=2,
                )
            ).recommendation

            self.assertEqual(
                [shot.shot_id for shot in shots.list_recent("install_1", "machine_1", "bean_1", grinder_context_id="grinder_a")],
                ["shot_a"],
            )
            self.assertEqual(
                [shot.shot_id for shot in shots.list_recent("install_1", "machine_1", "bean_1", grinder_context_id="grinder_b")],
                ["shot_b"],
            )
            self.assertEqual(
                recs.get_current("install_1", "machine_1", "bean_1", 20, grinder_context_id="grinder_a").recommendation_id,
                rec_a.recommendation_id,
            )
            self.assertEqual(
                recs.get_current("install_1", "machine_1", "bean_1", 20, grinder_context_id="grinder_b").recommendation_id,
                rec_b.recommendation_id,
            )

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

    def test_status_payload_includes_sanitized_recent_shot_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(mqtt_host="localhost", data_dir=Path(tmp), install_id="install_1")
            store = SQLiteStore(Path(tmp) / "espresso.db")
            shots = SQLiteShotRepository(store)
            recs = SQLiteRecommendationRepository(store)
            service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

            service.ingest_shot_profile(shot_event())
            status = build_status_payload(
                config=config,
                service=service,
                shot_repo=shots,
                upload_maintenance=None,
                upload_queue_repo=None,
                machine_id="machine_1",
                bean_context_id=None,
            )

            recent = status["recent_shots"]
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0]["shot_id"], "shot_1")
            self.assertEqual(recent[0]["profile_label"], "Cremina lever machine")
            self.assertEqual(recent[0]["final_phase_name"], "ramp")
            self.assertEqual(recent[0]["shot_end_state"], "manual_or_interrupted")
            self.assertNotIn("profile_resampled", recent[0])

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

    def test_enqueue_rearms_rejected_record_when_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            queue = SQLiteUploadQueueRepository(store)
            queue.enqueue(queue_item("u_a", "a"))
            queue.update_status("u_a", UploadQueueStatus.REJECTED, now=2, error_message="schema")
            queue.enqueue(queue_item("u_b", "b"))

            ready = queue.list_ready(now=10)
            self.assertEqual([item.upload_id for item in ready], ["u_b"])
            statuses = {
                row["upload_id"]: row["status"]
                for row in store.conn.execute(
                    "SELECT upload_id, status FROM upload_queue WHERE local_record_id='shot_1'"
                ).fetchall()
            }
            self.assertEqual(statuses, {"u_b": "pending"})

    def test_upload_queue_counts_by_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            queue = SQLiteUploadQueueRepository(store)
            queue.enqueue(queue_item("u_a", "a"))
            queue.enqueue(queue_item("u_b", "b", local_record_id="shot_2"))
            queue.update_status("u_b", UploadQueueStatus.REJECTED, now=2, error_message="schema")

            counts = queue.count_by_status()

            self.assertEqual(counts[UploadQueueStatus.PENDING], 1)
            self.assertEqual(counts[UploadQueueStatus.REJECTED], 1)

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
        self.assertIn("validation_summary JSONB", schema)
        self.assertIn("validation_errors JSONB", schema)
        self.assertIn("UNIQUE (source_validation_id)", schema)
        self.assertIn("idx_community_priors_context_key", schema)
        self.assertIn("feedback_recorded BOOLEAN NOT NULL DEFAULT FALSE", schema)
        self.assertIn("ADD COLUMN IF NOT EXISTS feedback_recorded", schema)

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

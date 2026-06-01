from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from espresso_rl.adapters.sqlite_repositories import SQLiteStore, SQLiteUploadQueueRepository
from espresso_rl.application.upload_maintenance import UploadQueueMaintenanceService
from espresso_rl.application.upload_validation import validate_upload_payload_json
from espresso_rl.domain.models import UploadQueueItem, UploadQueueStatus


def payload(**overrides) -> str:
    data = {
        "event_type": "shot_record",
        "schema_version": 1,
        "shot_id": "shot_1",
        "install_id": "install_1",
        "machine_id": "machine_1",
        "timestamp": 1,
        "dose_in_g": 18.0,
        "target_yield_g": 36.0,
        "target_ratio": 2.0,
        "beverage_out_g": 36.0,
        "shot_time_s": 30.0,
    }
    data.update(overrides)
    return json.dumps(data, sort_keys=True)


def queue_item(
    upload_id: str,
    payload_json: str,
    *,
    local_record_id: str = "shot_1",
    status: UploadQueueStatus = UploadQueueStatus.REJECTED,
) -> UploadQueueItem:
    return UploadQueueItem(
        upload_id=upload_id,
        local_record_type="shot",
        local_record_id=local_record_id,
        payload_hash=upload_id,
        payload_json=payload_json,
        status=status,
        attempt_count=3,
        error_message="HTTP 400: old validation error",
        created_at=1,
        updated_at=2,
    )


class UploadMaintenanceTests(unittest.TestCase):
    def test_preflight_accepts_valid_shot_payload(self) -> None:
        result = validate_upload_payload_json(payload())

        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])

    def test_preflight_rejects_utility_flush_payload(self) -> None:
        result = validate_upload_payload_json(
            payload(
                shot_id="flush_1",
                shot_type="utility_flush",
                beverage_out_g=1.0,
                shot_time_s=3.0,
            )
        )

        self.assertFalse(result.ok)
        self.assertIn("beverage_out_g out of range", result.errors)
        self.assertIn("shot_time_s out of range", result.errors)

    def test_requeue_valid_rejected_uploads_leaves_invalid_rows_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            queue = SQLiteUploadQueueRepository(store)
            queue.enqueue(queue_item("valid", payload(), local_record_id="shot_valid"))
            queue.enqueue(
                queue_item(
                    "invalid",
                    payload(shot_id="flush_1", beverage_out_g=1.0, shot_time_s=3.0),
                    local_record_id="shot_invalid",
                )
            )
            service = UploadQueueMaintenanceService(queue, clock=lambda: 10)

            result = service.requeue_valid_rejected(limit=10)

            self.assertEqual(result.inspected, 2)
            self.assertEqual(result.requeued, 1)
            self.assertEqual(result.skipped, 1)
            statuses = {
                row["upload_id"]: row["status"]
                for row in store.conn.execute("SELECT upload_id, status FROM upload_queue").fetchall()
            }
            self.assertEqual(statuses["valid"], "pending")
            self.assertEqual(statuses["invalid"], "rejected")

    def test_latest_rejected_summary_does_not_expose_payload_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            queue = SQLiteUploadQueueRepository(store)
            queue.enqueue(queue_item("rejected", payload(), local_record_id="shot_1"))
            service = UploadQueueMaintenanceService(queue, clock=lambda: 10)

            summary = service.latest_rejected()

            self.assertEqual(summary.upload_id, "rejected")  # type: ignore[union-attr]
            self.assertEqual(summary.local_record_id, "shot_1")  # type: ignore[union-attr]
            self.assertFalse(hasattr(summary, "payload_json"))


if __name__ == "__main__":
    unittest.main()

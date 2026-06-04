from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from espresso_rl.adapters.sqlite_repositories import SQLiteShotRepository, SQLiteStore, SQLiteUploadQueueRepository
from espresso_rl.application.upload_maintenance import UploadQueueMaintenanceService
from espresso_rl.application.upload_validation import (
    mask_untrusted_profile_channels,
    validate_upload_payload_json,
)
from espresso_rl.domain.models import ShotRecord, ShotType, UploadQueueItem, UploadQueueStatus


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


def payload_dict(**overrides) -> dict:
    data = json.loads(payload())
    data.update(overrides)
    return data


def valid_profile() -> list[list[float]]:
    profile = [[0.0 for _ in range(100)] for _ in range(5)]
    profile[0] = [9.0 for _ in range(100)]
    profile[1] = [9.0 for _ in range(100)]
    profile[2] = [2.0 for _ in range(100)]
    profile[3] = [0.0 for _ in range(100)]
    profile[4] = [i * 0.36 for i in range(100)]
    profile[4][-1] = 36.0
    return profile


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

    def test_preflight_allows_invalid_flow_when_flow_target_is_inactive(self) -> None:
        profile = valid_profile()
        profile[2] = [100_000.0 for _ in range(100)]

        result = validate_upload_payload_json(payload(profile_resampled=profile))

        self.assertTrue(result.ok)

    def test_preflight_allows_invalid_flow_when_flow_target_is_active(self) -> None:
        profile = valid_profile()
        profile[2] = [100_000.0 for _ in range(100)]
        profile[3] = [2.0 for _ in range(100)]

        result = validate_upload_payload_json(payload(profile_resampled=profile))

        self.assertTrue(result.ok)

    def test_preflight_rejects_nonfinite_flow_even_when_maskable(self) -> None:
        profile = valid_profile()
        profile[2] = [float("inf") for _ in range(100)]

        result = validate_upload_payload_json(payload(profile_resampled=profile))

        self.assertFalse(result.ok)
        self.assertIn("profile_resampled flow contains non-finite or nonnumeric values", result.errors)

    def test_trusted_payload_copy_masks_invalid_inactive_flow(self) -> None:
        profile = valid_profile()
        profile[2] = [100_000.0 for _ in range(100)]
        raw = payload_dict(profile_resampled=profile)

        trusted = mask_untrusted_profile_channels(raw)

        self.assertEqual(trusted["profile_resampled"][2], [0.0 for _ in range(100)])
        self.assertEqual(trusted["profile_resampled"][3], [0.0 for _ in range(100)])
        self.assertFalse(trusted["profile_flow_valid"])
        self.assertTrue(trusted["profile_flow_masked"])

    def test_trusted_payload_copy_masks_invalid_active_flow_pair(self) -> None:
        profile = valid_profile()
        profile[2] = [100_000.0 for _ in range(100)]
        profile[3] = [2.0 for _ in range(100)]
        raw = payload_dict(profile_resampled=profile)

        trusted = mask_untrusted_profile_channels(raw)

        self.assertEqual(trusted["profile_resampled"][2], [0.0 for _ in range(100)])
        self.assertEqual(trusted["profile_resampled"][3], [0.0 for _ in range(100)])
        self.assertFalse(trusted["profile_flow_valid"])
        self.assertTrue(trusted["profile_flow_masked"])

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
            invalid = store.conn.execute(
                "SELECT attempt_count, error_message, updated_at FROM upload_queue WHERE upload_id=?",
                ("invalid",),
            ).fetchone()
            self.assertEqual(statuses["valid"], "pending")
            self.assertEqual(statuses["invalid"], "rejected")
            self.assertEqual(invalid["attempt_count"], 3)
            self.assertEqual(invalid["updated_at"], 10)
            self.assertIn("preflight failed", invalid["error_message"])
            self.assertIn("beverage_out_g out of range", invalid["error_message"])

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

    def test_purge_rejected_deletes_useless_shots_but_keeps_local_optimizer_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / "espresso.db")
            shots = SQLiteShotRepository(store)
            queue = SQLiteUploadQueueRepository(store)
            profile = np.zeros((5, 100), dtype=np.float32)
            shots.upsert(
                ShotRecord(
                    shot_id="flush_1",
                    timestamp=1,
                    install_id="install_1",
                    machine_id="machine_1",
                    machine_adapter="gaggimate",
                    profile=profile,
                    grinder_step_size_um=12.5,
                    dose_in_g=18.0,
                    target_yield_g=36.0,
                    shot_type=ShotType.UTILITY_FLUSH,
                    exclude_from_local_optimization=True,
                    created_at=1,
                    updated_at=1,
                )
            )
            shots.upsert(
                ShotRecord(
                    shot_id="espresso_1",
                    timestamp=2,
                    install_id="install_1",
                    machine_id="machine_1",
                    machine_adapter="gaggimate",
                    profile=profile,
                    grinder_step_size_um=12.5,
                    dose_in_g=18.0,
                    target_yield_g=36.0,
                    shot_type=ShotType.ESPRESSO,
                    exclude_from_local_optimization=False,
                    optimization_weight=1.0,
                    created_at=2,
                    updated_at=2,
                )
            )
            queue.enqueue(queue_item("flush_upload", payload(), local_record_id="flush_1"))
            queue.enqueue(queue_item("espresso_upload", payload(), local_record_id="espresso_1"))
            service = UploadQueueMaintenanceService(queue, clock=lambda: 10)

            result = service.purge_rejected(limit=10)

            self.assertEqual(result.inspected, 2)
            self.assertEqual(result.purged_uploads, 2)
            self.assertEqual(result.purged_shots, 1)
            self.assertEqual(result.kept_linked_records, 1)
            self.assertIsNone(shots.get("flush_1"))
            self.assertIsNotNone(shots.get("espresso_1"))
            self.assertEqual(store.conn.execute("SELECT COUNT(*) AS count FROM upload_queue").fetchone()["count"], 0)


if __name__ == "__main__":
    unittest.main()

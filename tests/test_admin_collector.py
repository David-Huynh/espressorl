from __future__ import annotations

import json
import threading
import unittest
from urllib import request

from espresso_rl.adapters.supabase_community_queue import (
    HttpResponse,
    SupabaseCommunityQueueClient,
    SupabaseCommunityQueueConfig,
)
from espresso_rl.application.community_mirror import CommunityMirrorService
from espresso_rl.config import Config
from espresso_rl.domain.community import CommunityRawUpload
from espresso_rl.main import maybe_start_admin_collector_worker


HASH = "a" * 64


class AdminCollectorTests(unittest.TestCase):
    def test_mirror_service_marks_mirrored_and_failed_rows(self) -> None:
        first = CommunityRawUpload(
            install_id="install_1",
            upload_id="upload_1",
            payload_hash=HASH,
            event_type="shot_record",
            payload_json={"event_type": "shot_record", "shot_id": "shot_1"},
        )
        second = CommunityRawUpload(
            install_id="install_2",
            upload_id="upload_2",
            payload_hash="b" * 64,
            event_type="recommendation_record",
            payload_json={"event_type": "recommendation_record", "recommendation_id": "rec_1"},
        )
        source = FakeSource([first, second])
        warehouse = FakeWarehouse(fail_upload_ids={"upload_2"})

        result = CommunityMirrorService(source, warehouse).mirror_once(limit=10)

        self.assertEqual(result.claimed, 2)
        self.assertEqual(result.mirrored, 1)
        self.assertEqual(result.failed, 1)
        self.assertEqual(warehouse.rows, [first])
        self.assertEqual(source.mirrored, [first])
        self.assertEqual(source.failed[0][0], second)
        self.assertIn("warehouse failure", source.failed[0][1])

    def test_supabase_queue_claims_with_rpc_lease_and_marks_rows_by_owner(self) -> None:
        calls: list[request.Request] = []

        def transport(req: request.Request, timeout_s: float) -> HttpResponse:
            calls.append(req)
            if req.full_url.endswith("/rpc/espressorl_claim_raw_uploads"):
                return HttpResponse(
                    200,
                    json.dumps(
                        [
                            {
                                "install_id": "install_1",
                                "upload_id": "upload_1",
                                "payload_hash": HASH,
                                "event_type": "shot_record",
                                "payload_json": {"event_type": "shot_record", "shot_id": "shot_1"},
                                "received_at": "2026-05-29T00:00:00Z",
                            }
                        ]
                    ),
                )
            return HttpResponse(200, json.dumps([{"install_id": "install_1", "upload_id": "upload_1"}]))

        client = SupabaseCommunityQueueClient(
            SupabaseCommunityQueueConfig(
                rest_url="https://example.supabase.co/rest/v1",
                service_role_key="service-role",
                admin_id="admin_a",
                claim_lease_seconds=120,
            ),
            transport=transport,
        )

        claimed = client.claim_batch(limit=5)
        client.mark_mirrored(claimed[0])

        self.assertEqual(len(claimed), 1)
        self.assertEqual(claimed[0].upload_id, "upload_1")
        self.assertEqual(calls[0].get_header("Authorization"), "Bearer service-role")
        self.assertEqual(calls[0].get_header("Apikey"), "service-role")
        self.assertEqual(calls[0].get_method(), "POST")
        self.assertTrue(calls[0].full_url.endswith("/rpc/espressorl_claim_raw_uploads"))
        self.assertIn('"p_claimed_by":"admin_a"', calls[0].data.decode("utf-8"))  # type: ignore[union-attr]
        self.assertIn('"p_lease_seconds":120', calls[0].data.decode("utf-8"))  # type: ignore[union-attr]
        self.assertEqual(calls[1].get_method(), "PATCH")
        self.assertIn("status=eq.mirroring", calls[1].full_url)
        self.assertIn("mirror_claimed_by=eq.admin_a", calls[1].full_url)
        self.assertIn('"status":"mirrored"', calls[1].data.decode("utf-8"))  # type: ignore[union-attr]
        self.assertIn('"mirror_completed_at"', calls[1].data.decode("utf-8"))  # type: ignore[union-attr]

    def test_admin_collector_worker_requires_admin_role_and_supabase_credentials(self) -> None:
        public_config = Config(
            mqtt_host="localhost",
            deployment_role="public",
            admin_collector_enabled=True,
        )
        self.assertIsNone(maybe_start_admin_collector_worker(public_config, threading.Event()))

        admin_config = Config(
            mqtt_host="localhost",
            deployment_role="admin",
            admin_collector_enabled=True,
            storage_backend="postgres",
            postgres_dsn="postgresql://example",
        )
        self.assertIsNone(maybe_start_admin_collector_worker(admin_config, threading.Event()))


class FakeSource:
    def __init__(self, rows: list[CommunityRawUpload]) -> None:
        self.rows = rows
        self.mirrored: list[CommunityRawUpload] = []
        self.failed: list[tuple[CommunityRawUpload, str]] = []

    def claim_batch(self, limit: int = 100) -> list[CommunityRawUpload]:
        return self.rows[:limit]

    def mark_mirrored(self, upload: CommunityRawUpload) -> None:
        self.mirrored.append(upload)

    def mark_failed(self, upload: CommunityRawUpload, error_message: str) -> None:
        self.failed.append((upload, error_message))


class FakeWarehouse:
    def __init__(self, fail_upload_ids: set[str] | None = None) -> None:
        self.fail_upload_ids = fail_upload_ids or set()
        self.rows: list[CommunityRawUpload] = []

    def upsert_raw_upload(self, upload: CommunityRawUpload) -> None:
        if upload.upload_id in self.fail_upload_ids:
            raise RuntimeError("warehouse failure")
        self.rows.append(upload)


if __name__ == "__main__":
    unittest.main()

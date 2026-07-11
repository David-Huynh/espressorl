from __future__ import annotations

import threading
import unittest

from espresso_rl.adapters.postgres_repositories import _row_to_rejection_summary
from espresso_rl.application.admin_pipeline import AdminPipelineService
from espresso_rl.application.community_mirror import CommunityMirrorResult, CommunityQueuePurgeResult
from espresso_rl.application.community_validation import CommunityValidationResult
from espresso_rl.domain.community import AdminActionLogEntry, CommunityRejectionSummary


class AdminPipelineTests(unittest.TestCase):
    def test_manual_action_returns_status_when_same_job_is_already_running(self) -> None:
        warehouse = FakeWarehouse()
        mirror = BlockingMirror()
        service = AdminPipelineService(
            warehouse=warehouse,
            mirror=mirror,
            validator=FakeValidator(),
            clock=FakeClock(),
        )

        worker = threading.Thread(
            target=lambda: service.mirror_once(limit=10, requested_by="dashboard"),
            daemon=True,
        )
        worker.start()
        self.assertTrue(mirror.entered.wait(timeout=2))

        blocked = service.mirror_once(limit=10, requested_by="dashboard")

        mirror.release.set()
        worker.join(timeout=2)
        self.assertTrue(blocked.already_running)
        self.assertIsNotNone(blocked.status_snapshot)
        self.assertEqual([entry.status for entry in warehouse.admin_actions], ["already_running", "completed"])

    def test_purge_dry_run_is_locked_and_audited_without_changes(self) -> None:
        warehouse = FakeWarehouse(purge_eligible={"validated": 4, "rejected": 2})
        service = AdminPipelineService(
            warehouse=warehouse,
            mirror=FakeMirror(),
            validator=FakeValidator(),
            clock=FakeClock(),
        )

        result = service.purge_queue_once(dry_run=True, requested_by="dashboard")

        self.assertTrue(result.dry_run)
        self.assertIsNotNone(result.status_snapshot)
        self.assertEqual(result.purge.local_eligible, 6)  # type: ignore[union-attr]
        self.assertFalse(warehouse.purged)
        self.assertEqual(warehouse.admin_actions[-1].action_type, "purge_queue_once")
        self.assertEqual(warehouse.admin_actions[-1].rows_changed, 0)

    def test_purge_deletes_local_terminal_rows_and_source_queue_when_enabled(self) -> None:
        warehouse = FakeWarehouse(purge_eligible={"validated": 3, "rejected": 1})
        service = AdminPipelineService(
            warehouse=warehouse,
            mirror=PurgingMirror(source_purged=5),
            validator=FakeValidator(),
            clock=FakeClock(),
        )

        result = service.purge_queue_once(requested_by="dashboard")

        self.assertEqual(result.purge.local_eligible, 4)  # type: ignore[union-attr]
        self.assertEqual(result.purge.local_purged, 4)  # type: ignore[union-attr]
        self.assertEqual(result.purge.source_purged, 5)  # type: ignore[union-attr]
        self.assertEqual(result.purge.purged, 9)  # type: ignore[union-attr]
        self.assertEqual(warehouse.admin_actions[-1].rows_changed, 9)

    def test_purge_can_clean_local_terminal_rows_without_supabase_credentials(self) -> None:
        warehouse = FakeWarehouse(purge_eligible={"rejected": 2})
        service = AdminPipelineService(
            warehouse=warehouse,
            mirror=None,
            validator=FakeValidator(),
            clock=FakeClock(),
        )

        result = service.purge_queue_once(requested_by="dashboard")

        self.assertEqual(result.purge.local_purged, 2)  # type: ignore[union-attr]
        self.assertEqual(result.purge.source_purged, 0)  # type: ignore[union-attr]
        self.assertFalse(result.purge.source_enabled)  # type: ignore[union-attr]
        self.assertIn("Supabase source purge is disabled", result.warnings[0])

    def test_status_uses_safe_rejection_categories_only(self) -> None:
        warehouse = FakeWarehouse(
            rejections=[
                CommunityRejectionSummary(
                    install_id="install_1",
                    upload_id="upload_1",
                    event_type="shot_record",
                    validation_errors=["invalid_schema"],
                    rejected_at="2026-06-01T00:00:00Z",
                )
            ]
        )
        service = AdminPipelineService(
            warehouse=warehouse,
            mirror=None,
            validator=FakeValidator(),
            clock=FakeClock(),
        )

        status = service.status().to_dict()

        self.assertEqual(
            status["latest_rejections"][0]["validation_error_categories"],
            ["invalid_schema"],
        )
        self.assertNotIn("validation_errors", status["latest_rejections"][0])

    def test_postgres_rejection_row_collapses_raw_errors_to_categories(self) -> None:
        summary = _row_to_rejection_summary(
            {
                "install_id": "install_1",
                "upload_id": "upload_1",
                "event_type": "shot_record",
                "validation_errors": [
                    "pressure out of range for private bean name",
                    "flow out of range 99",
                    "payload size exceeded",
                ],
                "rejected_at": None,
            }
        )

        self.assertEqual(summary.validation_errors, ["invalid_schema", "impossible_flow", "payload_too_large"])

class FakeWarehouse:
    def __init__(
        self,
        rejections: list[CommunityRejectionSummary] | None = None,
        purge_eligible: dict[str, int] | None = None,
    ) -> None:
        self.admin_actions: list[AdminActionLogEntry] = []
        self.rejections = rejections or []
        self.purge_eligible = purge_eligible or {}
        self.purged = False

    def raw_upload_counts_by_status(self) -> dict[str, int]:
        return {"mirrored": 2}

    def raw_upload_purge_eligible_counts(
        self,
        *,
        validated_retention_days: int = 14,
        rejected_retention_days: int = 30,
    ) -> dict[str, int]:
        return dict(self.purge_eligible)

    def purge_raw_uploads(
        self,
        *,
        validated_retention_days: int = 14,
        rejected_retention_days: int = 30,
    ) -> int:
        self.purged = True
        purged = sum(self.purge_eligible.values())
        self.purge_eligible = {}
        return purged

    def validated_shot_count(self) -> int:
        return 3

    def comparison_count(self) -> int:
        return 5

    def abuse_event_count(self) -> int:
        return 6

    def latest_rejections(self, limit: int = 10) -> list[CommunityRejectionSummary]:
        return self.rejections[:limit]

    def record_admin_action(self, entry: AdminActionLogEntry) -> None:
        self.admin_actions.append(entry)

    def latest_admin_actions(self, limit: int = 10) -> list[AdminActionLogEntry]:
        return list(reversed(self.admin_actions[-limit:]))


class BlockingMirror:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def mirror_once(self, limit: int = 100) -> CommunityMirrorResult:
        self.entered.set()
        self.release.wait(timeout=2)
        return CommunityMirrorResult(claimed=1, mirrored=1, failed=0)

    def purge_retained_queue(self):
        raise AssertionError("dry-run purge must not call the purge RPC")


class FakeMirror:
    def mirror_once(self, limit: int = 100) -> CommunityMirrorResult:
        return CommunityMirrorResult(claimed=0, mirrored=0, failed=0)

    def purge_retained_queue(self):
        raise AssertionError("dry-run purge must not call the purge RPC")


class PurgingMirror:
    def __init__(self, source_purged: int) -> None:
        self.source_purged = source_purged

    def mirror_once(self, limit: int = 100) -> CommunityMirrorResult:
        return CommunityMirrorResult(claimed=0, mirrored=0, failed=0)

    def purge_retained_queue(self) -> CommunityQueuePurgeResult:
        return CommunityQueuePurgeResult(purged=self.source_purged, source_purged=self.source_purged)


class FakeValidator:
    def validate_once(self, limit: int = 100, *, dry_run: bool = False) -> CommunityValidationResult:
        return CommunityValidationResult(
            processed=7,
            validated_shots=3,
            stored_recommendations=1,
            stored_comparisons=1,
            rejected=1,
        )


class FakeClock:
    def __init__(self) -> None:
        self.value = 1_780_000_000

    def __call__(self) -> int:
        self.value += 1
        return self.value


if __name__ == "__main__":
    unittest.main()

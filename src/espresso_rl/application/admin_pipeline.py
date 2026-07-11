from __future__ import annotations

import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from espresso_rl.application.community_mirror import (
    CommunityMirrorResult,
    CommunityMirrorService,
    CommunityQueuePurgeResult,
)
from espresso_rl.application.community_validation import CommunityValidationResult, CommunityValidationService
from espresso_rl.domain.community import AdminActionLogEntry
from espresso_rl.ports.community import CommunityWarehouseRepository


@dataclass(frozen=True)
class AdminPipelineStatus:
    raw_upload_counts: dict[str, int]
    local_raw_upload_purge_eligible_counts: dict[str, int]
    validated_shot_count: int
    comparison_count: int
    abuse_event_count: int
    latest_rejections: list[dict]
    latest_admin_actions: list[dict]
    mirror_enabled: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AdminPipelineActionResult:
    action: str
    mirror: CommunityMirrorResult | None = None
    purge: CommunityQueuePurgeResult | None = None
    validation: CommunityValidationResult | None = None
    dry_run: bool = False
    warnings: list[str] = field(default_factory=list)
    already_running: bool = False
    status_snapshot: AdminPipelineStatus | None = None
    error_summary: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class AdminPipelineService:
    def __init__(
        self,
        *,
        warehouse: CommunityWarehouseRepository,
        mirror: CommunityMirrorService | None,
        validator: CommunityValidationService,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self._warehouse = warehouse
        self._mirror = mirror
        self._validator = validator
        self._clock = clock or (lambda: int(time.time()))
        self._locks = {
            "mirror_once": threading.Lock(),
            "purge_queue_once": threading.Lock(),
            "validate_once": threading.Lock(),
        }

    def status(self) -> AdminPipelineStatus:
        return AdminPipelineStatus(
            raw_upload_counts=self._warehouse.raw_upload_counts_by_status(),
            local_raw_upload_purge_eligible_counts=self._warehouse.raw_upload_purge_eligible_counts(),
            validated_shot_count=self._warehouse.validated_shot_count(),
            comparison_count=self._warehouse.comparison_count(),
            abuse_event_count=self._warehouse.abuse_event_count(),
            latest_rejections=[
                {
                    "install_id": item.install_id,
                    "upload_id": item.upload_id,
                    "event_type": item.event_type,
                    "validation_error_categories": list(item.validation_errors),
                    "rejected_at": item.rejected_at,
                }
                for item in self._warehouse.latest_rejections(limit=10)
            ],
            latest_admin_actions=[
                {
                    "action_type": item.action_type,
                    "requested_at": item.requested_at,
                    "requested_by": item.requested_by,
                    "dry_run": item.dry_run,
                    "status": item.status,
                    "rows_seen": item.rows_seen,
                    "rows_changed": item.rows_changed,
                    "warnings_count": item.warnings_count,
                    "error_summary": item.error_summary,
                }
                for item in self._warehouse.latest_admin_actions(limit=10)
            ],
            mirror_enabled=self._mirror is not None,
        )

    def mirror_once(
        self,
        limit: int = 100,
        *,
        dry_run: bool = False,
        requested_by: str = "system",
    ) -> AdminPipelineActionResult:
        return self._run_locked_action(
            "mirror_once",
            dry_run=dry_run,
            requested_by=requested_by,
            run=lambda: self._mirror_once_unlocked(limit=limit, dry_run=dry_run),
        )

    def validate_once(
        self,
        limit: int = 100,
        *,
        dry_run: bool = False,
        requested_by: str = "system",
    ) -> AdminPipelineActionResult:
        return self._run_locked_action(
            "validate_once",
            dry_run=dry_run,
            requested_by=requested_by,
            run=lambda: AdminPipelineActionResult(
                action="validate_once",
                validation=self._validator.validate_once(limit=limit, dry_run=dry_run),
                dry_run=dry_run,
            ),
        )

    def purge_queue_once(
        self,
        *,
        dry_run: bool = False,
        requested_by: str = "system",
    ) -> AdminPipelineActionResult:
        return self._run_locked_action(
            "purge_queue_once",
            dry_run=dry_run,
            requested_by=requested_by,
            run=lambda: self._purge_queue_once_unlocked(dry_run=dry_run),
        )

    def _mirror_once_unlocked(self, *, limit: int, dry_run: bool) -> AdminPipelineActionResult:
        if self._mirror is None:
            return AdminPipelineActionResult(
                action="mirror_once",
                dry_run=dry_run,
                warnings=["mirror is disabled because Supabase admin credentials are not configured"],
            )
        if dry_run:
            return AdminPipelineActionResult(
                action="mirror_once",
                dry_run=True,
                status_snapshot=self.status(),
                warnings=[
                    "mirror dry_run does not claim Supabase rows because claiming takes a remote lease"
                ],
            )
        return AdminPipelineActionResult(
            action="mirror_once",
            mirror=self._mirror.mirror_once(limit=limit),
        )

    def _purge_queue_once_unlocked(self, *, dry_run: bool) -> AdminPipelineActionResult:
        source_enabled = self._mirror is not None
        local_eligible_counts = self._warehouse.raw_upload_purge_eligible_counts()
        local_eligible = sum(local_eligible_counts.values())
        if dry_run:
            warnings: list[str] = []
            if not source_enabled:
                warnings.append("Supabase source purge is disabled because admin credentials are not configured")
            warnings.append("purge dry_run reports local terminal rows eligible for deletion but does not delete them")
            return AdminPipelineActionResult(
                action="purge_queue_once",
                purge=CommunityQueuePurgeResult(
                    purged=local_eligible,
                    local_eligible=local_eligible,
                    source_enabled=source_enabled,
                ),
                dry_run=True,
                status_snapshot=self.status(),
                warnings=warnings,
            )
        local_purged = self._warehouse.purge_raw_uploads()
        source_purged = 0
        warnings = []
        if self._mirror is not None:
            source_result = self._mirror.purge_retained_queue()
            source_purged = source_result.source_purged or source_result.purged
        else:
            warnings.append("Supabase source purge is disabled because admin credentials are not configured")
        return AdminPipelineActionResult(
            action="purge_queue_once",
            purge=CommunityQueuePurgeResult(
                purged=local_purged + source_purged,
                source_purged=source_purged,
                local_purged=local_purged,
                local_eligible=local_eligible,
                source_enabled=source_enabled,
            ),
            warnings=warnings,
        )

    def _run_locked_action(
        self,
        action_type: str,
        *,
        dry_run: bool,
        requested_by: str,
        run: Callable[[], AdminPipelineActionResult],
    ) -> AdminPipelineActionResult:
        lock = self._locks[action_type]
        if not lock.acquire(blocking=False):
            result = AdminPipelineActionResult(
                action=action_type,
                dry_run=dry_run,
                warnings=[f"{action_type} is already running"],
                already_running=True,
                status_snapshot=self.status(),
            )
            self._record_admin_action(result, requested_by=requested_by, status="already_running")
            return result

        try:
            result = run()
            self._record_admin_action(result, requested_by=requested_by, status="completed")
            return result
        except Exception as exc:
            result = AdminPipelineActionResult(
                action=action_type,
                dry_run=dry_run,
                error_summary=_safe_error_summary(exc),
            )
            self._record_admin_action(result, requested_by=requested_by, status="failed")
            return result
        finally:
            lock.release()

    def _record_admin_action(
        self,
        result: AdminPipelineActionResult,
        *,
        requested_by: str,
        status: str,
    ) -> None:
        self._warehouse.record_admin_action(
            AdminActionLogEntry(
                action_type=result.action,
                requested_at=self._clock(),
                requested_by=_safe_requested_by(requested_by),
                dry_run=result.dry_run,
                status=status,
                rows_seen=_rows_seen(result),
                rows_changed=0 if result.dry_run else _rows_changed(result),
                warnings_count=len(result.warnings),
                error_summary=result.error_summary,
            )
        )


def _rows_seen(result: AdminPipelineActionResult) -> int:
    if result.mirror is not None:
        return result.mirror.claimed
    if result.purge is not None:
        return result.purge.purged
    if result.validation is not None:
        return result.validation.processed
    return 0


def _rows_changed(result: AdminPipelineActionResult) -> int:
    if result.mirror is not None:
        return result.mirror.mirrored
    if result.purge is not None:
        return result.purge.purged
    if result.validation is not None:
        return (
            result.validation.validated_shots
            + result.validation.stored_recommendations
            + result.validation.stored_comparisons
            + result.validation.rejected
        )
    return 0


def _safe_requested_by(requested_by: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.:-]", "_", requested_by.strip())[:80]
    return text or "unknown"


def _safe_error_summary(exc: Exception) -> str:
    text = re.sub(r"\s+", " ", str(exc)).strip()
    if not text:
        text = exc.__class__.__name__
    return text[:300]

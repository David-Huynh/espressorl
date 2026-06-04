from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from espresso_rl.application.upload_validation import validate_upload_payload_json
from espresso_rl.domain.models import UploadQueueItem, UploadQueueStatus, now_ts
from espresso_rl.ports.repositories import UploadQueueRepository


@dataclass(frozen=True)
class RejectedUploadSummary:
    upload_id: str
    local_record_type: str
    local_record_id: str
    attempt_count: int
    error_message: str | None
    updated_at: int


@dataclass(frozen=True)
class RequeueRejectedResult:
    inspected: int
    requeued: int
    skipped: int
    skipped_uploads: list[RejectedUploadSummary] = field(default_factory=list)


@dataclass(frozen=True)
class PurgeRejectedResult:
    inspected: int
    purged_uploads: int
    purged_shots: int
    purged_recommendations: int
    kept_linked_records: int


class UploadQueueMaintenanceService:
    def __init__(
        self,
        queue: UploadQueueRepository,
        clock: Callable[[], int] = now_ts,
    ) -> None:
        self._queue = queue
        self._clock = clock

    def latest_rejected(self) -> RejectedUploadSummary | None:
        rejected = self._queue.list_by_status(UploadQueueStatus.REJECTED, limit=1)
        return _summary(rejected[0]) if rejected else None

    def list_rejected(self, limit: int = 10) -> list[RejectedUploadSummary]:
        return [_summary(item) for item in self._queue.list_by_status(UploadQueueStatus.REJECTED, limit=_safe_limit(limit))]

    def requeue_valid_rejected(self, limit: int = 25) -> RequeueRejectedResult:
        now = self._clock()
        inspected = 0
        requeued = 0
        skipped_uploads: list[RejectedUploadSummary] = []

        for item in self._queue.list_by_status(UploadQueueStatus.REJECTED, limit=_safe_limit(limit)):
            inspected += 1
            validation = validate_upload_payload_json(item.payload_json)
            if validation.ok:
                self._queue.requeue(item.upload_id, now=now, error_message="requeued after local preflight")
                requeued += 1
                continue

            error_message = "preflight failed: " + "; ".join(validation.errors[:3])
            self._queue.mark_rejected_preflight_failed(
                item.upload_id,
                now=now,
                error_message=error_message,
            )
            skipped_uploads.append(
                RejectedUploadSummary(
                    upload_id=item.upload_id,
                    local_record_type=item.local_record_type,
                    local_record_id=item.local_record_id,
                    attempt_count=item.attempt_count,
                    error_message=error_message,
                    updated_at=item.updated_at,
                )
            )

        return RequeueRejectedResult(
            inspected=inspected,
            requeued=requeued,
            skipped=inspected - requeued,
            skipped_uploads=skipped_uploads[:5],
        )

    def purge_rejected(self, limit: int = 100) -> PurgeRejectedResult:
        counts = self._queue.purge_rejected_artifacts(now=self._clock(), limit=_safe_limit(limit))
        return PurgeRejectedResult(
            inspected=counts.get("inspected", 0),
            purged_uploads=counts.get("purged_uploads", 0),
            purged_shots=counts.get("purged_shots", 0),
            purged_recommendations=counts.get("purged_recommendations", 0),
            kept_linked_records=counts.get("kept_linked_records", 0),
        )


def _summary(item: UploadQueueItem) -> RejectedUploadSummary:
    return RejectedUploadSummary(
        upload_id=item.upload_id,
        local_record_type=item.local_record_type,
        local_record_id=item.local_record_id,
        attempt_count=item.attempt_count,
        error_message=item.error_message,
        updated_at=item.updated_at,
    )


def _safe_limit(limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = 25
    return max(1, min(value, 500))

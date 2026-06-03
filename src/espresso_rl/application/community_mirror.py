from __future__ import annotations

from dataclasses import dataclass

from espresso_rl.ports.community import CommunityUploadSource, CommunityWarehouseRepository


@dataclass(frozen=True)
class CommunityMirrorResult:
    claimed: int
    mirrored: int
    failed: int


@dataclass(frozen=True)
class CommunityQueuePurgeResult:
    purged: int
    source_purged: int = 0
    local_purged: int = 0
    local_eligible: int = 0
    source_enabled: bool = True


class CommunityMirrorService:
    def __init__(
        self,
        source: CommunityUploadSource,
        warehouse: CommunityWarehouseRepository,
    ) -> None:
        self._source = source
        self._warehouse = warehouse

    def mirror_once(self, limit: int = 100) -> CommunityMirrorResult:
        uploads = self._source.claim_batch(limit=limit)
        mirrored = 0
        failed = 0
        for upload in uploads:
            try:
                self._warehouse.upsert_raw_upload(upload)
                self._source.mark_mirrored(upload)
                mirrored += 1
            except Exception as exc:
                self._source.mark_failed(upload, str(exc))
                failed += 1
        return CommunityMirrorResult(
            claimed=len(uploads),
            mirrored=mirrored,
            failed=failed,
        )

    def purge_retained_queue(
        self,
        *,
        mirrored_retention_days: int = 14,
        rejected_retention_days: int = 30,
        failed_retention_days: int = 90,
    ) -> CommunityQueuePurgeResult:
        source_purged = self._source.purge_retained_queue(
            mirrored_retention_days=mirrored_retention_days,
            rejected_retention_days=rejected_retention_days,
            failed_retention_days=failed_retention_days,
        )
        return CommunityQueuePurgeResult(purged=source_purged, source_purged=source_purged)

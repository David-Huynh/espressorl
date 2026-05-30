from __future__ import annotations

from dataclasses import dataclass

from espresso_rl.ports.community import CommunityUploadSource, CommunityWarehouseRepository


@dataclass(frozen=True)
class CommunityMirrorResult:
    claimed: int
    mirrored: int
    failed: int


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

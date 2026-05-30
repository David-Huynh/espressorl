from __future__ import annotations

from typing import Protocol

from espresso_rl.domain.community import CommunityRawUpload


class CommunityUploadSource(Protocol):
    def claim_batch(self, limit: int = 100) -> list[CommunityRawUpload]:
        ...

    def mark_mirrored(self, upload: CommunityRawUpload) -> None:
        ...

    def mark_failed(self, upload: CommunityRawUpload, error_message: str) -> None:
        ...


class CommunityWarehouseRepository(Protocol):
    def upsert_raw_upload(self, upload: CommunityRawUpload) -> None:
        ...

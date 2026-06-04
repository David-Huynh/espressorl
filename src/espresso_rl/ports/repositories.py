from __future__ import annotations

from typing import Protocol

from espresso_rl.domain.models import Recommendation, ShotRecord, UploadQueueItem, UploadQueueStatus


class ShotRepository(Protocol):
    def upsert(self, shot: ShotRecord) -> None:
        ...

    def get(self, shot_id: str) -> ShotRecord | None:
        ...

    def list_recent(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None = None,
        limit: int = 200,
    ) -> list[ShotRecord]:
        ...


class RecommendationRepository(Protocol):
    def upsert(self, recommendation: Recommendation) -> None:
        ...

    def get(self, recommendation_id: str) -> Recommendation | None:
        ...

    def get_current(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        now: int,
    ) -> Recommendation | None:
        ...

    def get_latest(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
    ) -> Recommendation | None:
        ...

    def supersede_active(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        now: int,
        except_recommendation_id: str | None = None,
    ) -> None:
        ...


class UploadQueueRepository(Protocol):
    def enqueue(self, item: UploadQueueItem) -> None:
        ...

    def list_ready(self, now: int, limit: int = 100) -> list[UploadQueueItem]:
        ...

    def update_status(
        self,
        upload_id: str,
        status: UploadQueueStatus,
        now: int,
        error_message: str | None = None,
        next_retry_at: int | None = None,
    ) -> None:
        ...

    def count_by_status(self) -> dict[UploadQueueStatus, int]:
        ...

    def list_by_status(self, status: UploadQueueStatus, limit: int = 100) -> list[UploadQueueItem]:
        ...

    def requeue(
        self,
        upload_id: str,
        now: int,
        error_message: str | None = None,
    ) -> None:
        ...

    def mark_rejected_preflight_failed(
        self,
        upload_id: str,
        now: int,
        error_message: str,
    ) -> None:
        ...

    def purge_rejected_artifacts(self, now: int, limit: int = 100) -> dict[str, int]:
        ...

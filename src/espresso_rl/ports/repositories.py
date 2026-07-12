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
        grinder_context_id: str | None = None,
    ) -> list[ShotRecord]:
        ...

    def list_machine_shots(
        self,
        install_id: str,
        machine_id: str,
        limit: int = 500,
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
        grinder_context_id: str | None = None,
        profile_id: str | None = None,
        raw_profile_hash: str | None = None,
        taste_goal_fingerprint: str | None = None,
    ) -> Recommendation | None:
        ...

    def get_latest(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        grinder_context_id: str | None = None,
        profile_id: str | None = None,
        raw_profile_hash: str | None = None,
        taste_goal_fingerprint: str | None = None,
    ) -> Recommendation | None:
        ...

    def supersede_active(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        now: int,
        except_recommendation_id: str | None = None,
        grinder_context_id: str | None = None,
        profile_id: str | None = None,
        raw_profile_hash: str | None = None,
        taste_goal_fingerprint: str | None = None,
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

    def purge_rejected_artifacts(
        self,
        now: int,
        limit: int = 100,
        local_record_id: str | None = None,
        delete_linked_records: bool = True,
    ) -> dict[str, int]:
        ...


class LocalDataRepository(Protocol):
    def list_machine_shots(
        self,
        install_id: str,
        machine_id: str,
        limit: int = 500,
    ) -> list[ShotRecord]:
        ...

    def delete_shot(
        self,
        install_id: str,
        machine_id: str,
        shot_id: str,
        *,
        dry_run: bool = False,
    ) -> dict[str, int]:
        ...

    def exclude_shot_from_optimization(
        self,
        install_id: str,
        machine_id: str,
        shot_id: str,
        *,
        now: int,
        dry_run: bool = False,
    ) -> dict[str, int]:
        ...

    def purge_useless_shots(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None = None,
        *,
        limit: int = 100,
        dry_run: bool = False,
        grinder_context_id: str | None = None,
    ) -> dict[str, int]:
        ...

    def reset_optimizer_context(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str,
        *,
        now: int,
        dry_run: bool = False,
        grinder_context_id: str | None = None,
    ) -> dict[str, int]:
        ...

    def reset_all(
        self,
        install_id: str,
        machine_id: str,
        *,
        dry_run: bool = False,
    ) -> dict[str, int]:
        ...

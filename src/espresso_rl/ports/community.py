from __future__ import annotations

from typing import Any, Protocol

from espresso_rl.domain.community import (
    AdminActionLogEntry,
    CommunityAbuseEvent,
    CommunityInstallStats,
    CommunityPrior,
    CommunityRawUpload,
    CommunityRecommendationRecord,
    CommunityRejectionSummary,
    CommunityTrainingRow,
    CommunityUploadCredentials,
    CommunityValidatedShot,
    InstallTrustScore,
)


class CommunityUploadSource(Protocol):
    def claim_batch(self, limit: int = 100) -> list[CommunityRawUpload]:
        ...

    def mark_mirrored(self, upload: CommunityRawUpload) -> None:
        ...

    def mark_failed(self, upload: CommunityRawUpload, error_message: str) -> None:
        ...

    def purge_retained_queue(
        self,
        *,
        mirrored_retention_days: int = 14,
        rejected_retention_days: int = 30,
        failed_retention_days: int = 90,
    ) -> int:
        ...


class CommunityWarehouseRepository(Protocol):
    def upsert_raw_upload(self, upload: CommunityRawUpload) -> None:
        ...

    def list_raw_uploads(self, status: str = "mirrored", limit: int = 100) -> list[CommunityRawUpload]:
        ...

    def mark_raw_upload_validated(
        self,
        upload: CommunityRawUpload,
        validation_summary: dict[str, Any],
    ) -> None:
        ...

    def mark_raw_upload_rejected(
        self,
        upload: CommunityRawUpload,
        validation_errors: list[str],
    ) -> None:
        ...

    def upsert_validated_shot(self, shot: CommunityValidatedShot) -> int:
        ...

    def upsert_community_recommendation(self, recommendation: CommunityRecommendationRecord) -> None:
        ...

    def upsert_install_trust_score(self, score: InstallTrustScore) -> None:
        ...

    def install_stats(self, install_id: str) -> CommunityInstallStats:
        ...

    def record_abuse_event(self, event: CommunityAbuseEvent) -> None:
        ...

    def upsert_training_row(
        self,
        source_validation_id: int,
        payload_json: dict[str, Any],
        trust_weight: float,
    ) -> None:
        ...

    def list_training_rows(self, limit: int = 5000) -> list[CommunityTrainingRow]:
        ...

    def upsert_community_prior(self, prior: CommunityPrior) -> None:
        ...

    def list_community_priors(self, context_key: str, limit: int = 10) -> list[CommunityPrior]:
        ...

    def raw_upload_counts_by_status(self) -> dict[str, int]:
        ...

    def raw_upload_purge_eligible_counts(
        self,
        *,
        validated_retention_days: int = 14,
        rejected_retention_days: int = 30,
    ) -> dict[str, int]:
        ...

    def purge_raw_uploads(
        self,
        *,
        validated_retention_days: int = 14,
        rejected_retention_days: int = 30,
    ) -> int:
        ...

    def validated_shot_count(self) -> int:
        ...

    def training_row_count(self) -> int:
        ...

    def community_prior_count(self) -> int:
        ...

    def abuse_event_count(self) -> int:
        ...

    def latest_rejections(self, limit: int = 10) -> list[CommunityRejectionSummary]:
        ...

    def record_admin_action(self, entry: AdminActionLogEntry) -> None:
        ...

    def latest_admin_actions(self, limit: int = 10) -> list[AdminActionLogEntry]:
        ...


class CommunityCredentialRegistrar(Protocol):
    def register_install(self) -> CommunityUploadCredentials:
        ...

    def rotate_credentials(self, current: CommunityUploadCredentials) -> CommunityUploadCredentials:
        ...

    def revoke_credentials(self, current: CommunityUploadCredentials) -> None:
        ...


class CommunityCredentialStore(Protocol):
    def load(self) -> CommunityUploadCredentials | None:
        ...

    def save(self, credentials: CommunityUploadCredentials) -> None:
        ...

    def clear(self) -> None:
        ...

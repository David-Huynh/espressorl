from __future__ import annotations

from typing import Protocol

from typing import Any

from espresso_rl.domain.community import (
    CommunityAbuseEvent,
    CommunityInstallStats,
    CommunityPrior,
    CommunityRawUpload,
    CommunityRecommendationRecord,
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

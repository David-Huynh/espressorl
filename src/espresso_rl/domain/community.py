from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CommunityUploadCredentials:
    install_id: str
    upload_token_id: str
    upload_secret: str

    def __post_init__(self) -> None:
        if not self.install_id:
            raise ValueError("install_id is required")
        if self.upload_token_id is None:
            raise ValueError("upload_token_id is required")
        if len(self.upload_secret) < 32:
            raise ValueError("upload_secret must be at least 32 characters")


@dataclass(frozen=True)
class CommunityRawUpload:
    install_id: str
    upload_id: str
    payload_hash: str
    event_type: str
    payload_json: dict[str, Any]
    received_at: str | None = None

    def __post_init__(self) -> None:
        if not self.install_id:
            raise ValueError("install_id is required")
        if not self.upload_id:
            raise ValueError("upload_id is required")
        if len(self.payload_hash) != 64:
            raise ValueError("payload_hash must be a sha256 hex digest")
        if self.event_type not in {"shot_record", "recommendation_record"}:
            raise ValueError("unsupported community upload event_type")


@dataclass(frozen=True)
class CommunityValidatedShot:
    install_id: str
    upload_id: str
    shot_id: str
    payload_json: dict[str, Any]
    trust_weight: float
    validation_summary: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.install_id:
            raise ValueError("install_id is required")
        if not self.upload_id:
            raise ValueError("upload_id is required")
        if not self.shot_id:
            raise ValueError("shot_id is required")
        if self.payload_json.get("event_type") != "shot_record":
            raise ValueError("validated shot payload must be a shot_record")
        if not 0.0 <= float(self.trust_weight) <= 1.0:
            raise ValueError("trust_weight must be between 0 and 1")


@dataclass(frozen=True)
class CommunityRecommendationRecord:
    install_id: str
    upload_id: str
    recommendation_id: str
    payload_json: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.install_id:
            raise ValueError("install_id is required")
        if not self.upload_id:
            raise ValueError("upload_id is required")
        if not self.recommendation_id:
            raise ValueError("recommendation_id is required")
        if self.payload_json.get("event_type") != "recommendation_record":
            raise ValueError("recommendation payload must be a recommendation_record")


@dataclass(frozen=True)
class CommunityInstallStats:
    validated_shots: int = 0
    rejected_uploads: int = 0
    abuse_events: int = 0


@dataclass(frozen=True)
class InstallTrustScore:
    install_id: str
    trust_score: float
    reason: str

    def __post_init__(self) -> None:
        if not self.install_id:
            raise ValueError("install_id is required")
        if not 0.0 <= float(self.trust_score) <= 1.0:
            raise ValueError("trust_score must be between 0 and 1")


@dataclass(frozen=True)
class CommunityAbuseEvent:
    install_id: str
    upload_id: str
    payload_hash: str
    reason: str
    detail: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.install_id:
            raise ValueError("install_id is required")
        if not self.upload_id:
            raise ValueError("upload_id is required")
        if len(self.payload_hash) != 64:
            raise ValueError("payload_hash must be a sha256 hex digest")
        if not self.reason:
            raise ValueError("reason is required")

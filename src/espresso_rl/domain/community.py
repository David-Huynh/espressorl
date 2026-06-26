from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SAFE_COMMUNITY_REJECTION_CATEGORIES = frozenset(
    {
        "invalid_schema",
        "invalid_signature",
        "rate_limited",
        "impossible_flow",
        "duplicate_shot_id",
        "payload_too_large",
    }
)


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
        if len(self.payload_hash) != 64 or any(char not in "0123456789abcdef" for char in self.payload_hash.lower()):
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
class CommunityTrainingRow:
    training_row_id: int
    source_validation_id: int
    install_id: str
    payload_json: dict[str, Any]
    trust_weight: float
    payload_hash: str | None = None

    def __post_init__(self) -> None:
        if self.training_row_id <= 0:
            raise ValueError("training_row_id is required")
        if self.source_validation_id <= 0:
            raise ValueError("source_validation_id is required")
        if not self.install_id:
            raise ValueError("install_id is required")
        if self.payload_json.get("event_type") != "shot_record":
            raise ValueError("training row payload must be a shot_record")
        if not 0.0 <= float(self.trust_weight) <= 1.0:
            raise ValueError("trust_weight must be between 0 and 1")
        if self.payload_hash is not None and len(self.payload_hash) != 64:
            raise ValueError("payload_hash must be a sha256 hex digest")


@dataclass(frozen=True)
class CommunityPrior:
    context_key: str
    prior_json: dict[str, Any]
    confidence: float

    def __post_init__(self) -> None:
        if not self.context_key:
            raise ValueError("context_key is required")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


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


@dataclass(frozen=True)
class CommunityRejectionSummary:
    install_id: str
    upload_id: str
    event_type: str
    validation_errors: list[str]
    rejected_at: str | None = None

    def __post_init__(self) -> None:
        if not self.install_id:
            raise ValueError("install_id is required")
        if not self.upload_id:
            raise ValueError("upload_id is required")
        if not self.event_type:
            raise ValueError("event_type is required")
        for category in self.validation_errors:
            if category not in SAFE_COMMUNITY_REJECTION_CATEGORIES:
                raise ValueError("community rejection summaries must use safe categories")


@dataclass(frozen=True)
class AdminActionLogEntry:
    action_type: str
    requested_at: int
    requested_by: str
    dry_run: bool
    status: str
    rows_seen: int
    rows_changed: int
    warnings_count: int
    error_summary: str | None = None

    def __post_init__(self) -> None:
        if not self.action_type:
            raise ValueError("action_type is required")
        if self.requested_at <= 0:
            raise ValueError("requested_at is required")
        if not self.requested_by:
            raise ValueError("requested_by is required")
        if self.status not in {"completed", "already_running", "failed"}:
            raise ValueError("unsupported admin action status")
        if self.rows_seen < 0:
            raise ValueError("rows_seen cannot be negative")
        if self.rows_changed < 0:
            raise ValueError("rows_changed cannot be negative")
        if self.warnings_count < 0:
            raise ValueError("warnings_count cannot be negative")


def community_rejection_category(error: object) -> str:
    text = str(error).strip().lower()
    if text in SAFE_COMMUNITY_REJECTION_CATEGORIES:
        return text
    if "signature" in text or "hmac" in text:
        return "invalid_signature"
    if "rate" in text and "limit" in text:
        return "rate_limited"
    if "payload" in text and ("too large" in text or "size" in text):
        return "payload_too_large"
    if "duplicate" in text and "shot" in text:
        return "duplicate_shot_id"
    if "flow" in text and ("range" in text or "impossible" in text):
        return "impossible_flow"
    return "invalid_schema"


def community_rejection_categories(errors: list[object]) -> list[str]:
    categories: list[str] = []
    for error in errors:
        category = community_rejection_category(error)
        if category not in categories:
            categories.append(category)
    return categories or ["invalid_schema"]

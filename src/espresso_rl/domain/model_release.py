from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DREAMER_RELEASE_AUTHORIZATION_FORMAT = "espresso_rl_dreamer_release_authorization_v1"
DREAMER_RELEASE_AUTHORIZATION_SCHEMA_VERSION = 1
DREAMER_RELEASE_APPROVAL = "approved_for_runtime_inference"
DREAMER_RELEASE_RECORD_FORMAT = "espresso_rl_dreamer_release_record_v1"
DREAMER_RELEASE_RECORD_SCHEMA_VERSION = 1

_HEX_CHARS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class DreamerReleaseAuthorization:
    """Explicit, non-training authorization for one immutable checkpoint candidate."""

    candidate_artifact_sha256: str
    candidate_manifest_sha256: str
    released_by: str
    release_version: str
    released_at: int
    approval: str = DREAMER_RELEASE_APPROVAL
    format: str = DREAMER_RELEASE_AUTHORIZATION_FORMAT
    schema_version: int = DREAMER_RELEASE_AUTHORIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.format != DREAMER_RELEASE_AUTHORIZATION_FORMAT:
            raise ValueError("Dreamer release authorization format is unsupported")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != DREAMER_RELEASE_AUTHORIZATION_SCHEMA_VERSION
        ):
            raise ValueError("Dreamer release authorization schema version is unsupported")
        _sha256(self.candidate_artifact_sha256, "candidate_artifact_sha256")
        _sha256(self.candidate_manifest_sha256, "candidate_manifest_sha256")
        _safe_text(self.released_by, "released_by", max_len=120)
        _safe_text(self.release_version, "release_version", max_len=120)
        if isinstance(self.released_at, bool) or not isinstance(self.released_at, int) or self.released_at <= 0:
            raise ValueError("Dreamer release released_at must be a positive integer")
        if self.approval != DREAMER_RELEASE_APPROVAL:
            raise ValueError("Dreamer release approval statement is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "released_by": self.released_by,
            "release_version": self.release_version,
            "released_at": self.released_at,
            "approval": self.approval,
        }

    @classmethod
    def from_dict(cls, value: Any) -> DreamerReleaseAuthorization:
        if not isinstance(value, dict):
            raise ValueError("Dreamer release authorization must be an object")
        expected = {
            "format",
            "schema_version",
            "candidate_artifact_sha256",
            "candidate_manifest_sha256",
            "released_by",
            "release_version",
            "released_at",
            "approval",
        }
        if set(value) != expected:
            raise ValueError("Dreamer release authorization fields are invalid")
        return cls(
            format=value["format"],
            schema_version=value["schema_version"],
            candidate_artifact_sha256=value["candidate_artifact_sha256"],
            candidate_manifest_sha256=value["candidate_manifest_sha256"],
            released_by=value["released_by"],
            release_version=value["release_version"],
            released_at=value["released_at"],
            approval=value["approval"],
        )


def _sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(character not in _HEX_CHARS for character in value):
        raise ValueError(f"Dreamer release {label} must be a lowercase SHA-256 digest")


def _safe_text(value: object, label: str, *, max_len: int) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or len(value) > max_len:
        raise ValueError(f"Dreamer release {label} is invalid")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"Dreamer release {label} is invalid")

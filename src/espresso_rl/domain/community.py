from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

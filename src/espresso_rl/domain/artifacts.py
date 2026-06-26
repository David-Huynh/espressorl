from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactInfo:
    relative_path: str
    absolute_path: str
    content_type: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.relative_path:
            raise ValueError("artifact relative_path is required")
        if not self.absolute_path:
            raise ValueError("artifact absolute_path is required")
        if not self.content_type:
            raise ValueError("artifact content_type is required")
        if self.size_bytes < 0:
            raise ValueError("artifact size_bytes cannot be negative")
        if len(self.sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.sha256):
            raise ValueError("artifact sha256 must be a lowercase sha256 hex digest")

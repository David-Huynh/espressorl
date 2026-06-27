from __future__ import annotations

from typing import Protocol


class ModelArtifactStore(Protocol):
    """Reads opaque model bundle members from an external store."""

    def read_bytes(self, reference: str, *, max_bytes: int) -> bytes:
        ...

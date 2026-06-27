from __future__ import annotations

from pathlib import Path


class LocalModelArtifactStore:
    """Local filesystem adapter for explicitly configured model artifact paths."""

    def read_bytes(self, reference: str, *, max_bytes: int) -> bytes:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("model artifact reference must be a non-empty path")

        path = Path(reference.strip())
        try:
            stat_before = path.stat()
        except OSError as exc:
            raise ValueError("model artifact does not exist or is unreadable") from exc
        if not path.is_file():
            raise ValueError("model artifact reference is not a file")
        if stat_before.st_size <= 0:
            raise ValueError("model artifact is empty")
        if stat_before.st_size > max_bytes:
            raise ValueError("model artifact exceeds the configured size limit")

        try:
            with path.open("rb") as handle:
                payload = handle.read(max_bytes + 1)
            stat_after = path.stat()
        except OSError as exc:
            raise ValueError("model artifact could not be read") from exc
        if len(payload) > max_bytes:
            raise ValueError("model artifact exceeds the configured size limit")
        if (
            stat_before.st_size != stat_after.st_size
            or stat_before.st_mtime_ns != stat_after.st_mtime_ns
            or len(payload) != stat_after.st_size
        ):
            raise ValueError("model artifact changed while it was being read")
        return payload

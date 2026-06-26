from __future__ import annotations

import hashlib
from pathlib import Path

from espresso_rl.domain.artifacts import ArtifactInfo


class LocalTextArtifactWriter:
    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    def write_text(
        self,
        relative_path: str,
        content: str,
        *,
        content_type: str,
    ) -> ArtifactInfo:
        target = self._safe_target(relative_path)
        payload = content.encode("utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        return ArtifactInfo(
            relative_path=relative_path.replace("\\", "/"),
            absolute_path=str(target),
            content_type=content_type,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    def _safe_target(self, relative_path: str) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("artifact path must be a safe relative path")
        root = self._root.resolve()
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("artifact path must stay inside export root") from exc
        return target

from __future__ import annotations

from typing import Protocol

from espresso_rl.domain.artifacts import ArtifactInfo


class TextArtifactWriter(Protocol):
    def write_text(
        self,
        relative_path: str,
        content: str,
        *,
        content_type: str,
    ) -> ArtifactInfo:
        ...

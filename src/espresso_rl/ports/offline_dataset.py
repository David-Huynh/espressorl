from __future__ import annotations

from typing import Protocol

from espresso_rl.domain.offline_dataset import OfflinePreferenceExample


class OfflineDatasetSource(Protocol):
    """Read-only access to validated physical shots and pairwise comparisons."""

    def list_offline_preference_examples(
        self,
        *,
        limit: int | None = None,
    ) -> list[OfflinePreferenceExample]:
        ...

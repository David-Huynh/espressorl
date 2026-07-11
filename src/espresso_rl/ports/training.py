from __future__ import annotations

from typing import Protocol

from espresso_rl.domain.community import CommunityTrainingRow


class TrainingRowSource(Protocol):
    """Legacy model-export source, kept outside the community warehouse port."""

    def list_training_rows(self, limit: int = 5000) -> list[CommunityTrainingRow]:
        ...

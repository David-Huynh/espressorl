from __future__ import annotations

from collections.abc import Sequence

from espresso_rl.application.training_export import local_training_transition_from_shot
from espresso_rl.domain.models import ShotRecord
from espresso_rl.dreamer.dataset import (
    DreamerEpisodeDatasetError,
    build_dreamer_episodes_from_training_rows,
)


class CanonicalDreamerHistoryEncoder:
    """Adapter from persisted canonical shots to validated Dreamer episodes."""

    def encode(self, shots: Sequence[ShotRecord]) -> tuple[dict[str, object], ...]:
        transitions = [
            transition
            for shot in shots
            if (transition := local_training_transition_from_shot(shot)) is not None
        ]
        if not transitions:
            return ()
        try:
            episodes = build_dreamer_episodes_from_training_rows(transitions)
        except DreamerEpisodeDatasetError as exc:
            raise ValueError(f"Dreamer local context history is invalid: {exc}") from exc
        return tuple(episodes)

from __future__ import annotations

from typing import Protocol

from espresso_rl.domain.dreamer_control import DreamerLiveControlPublication
from espresso_rl.domain.models import Recommendation


class AutoTuningRuntimePublisher(Protocol):
    def publish_recommendation(self, recommendation: Recommendation) -> None:
        ...

    def publish_dreamer_live_control(self, publication: DreamerLiveControlPublication) -> None:
        ...

    def publish_status(
        self,
        machine_id: str,
        bean_context_id: str | None,
        grinder_context_id: str | None,
        *,
        profile_id: str | None = None,
        profile_label: str | None = None,
        last_shot_id: str | None = None,
        last_shot_at: int | None = None,
        last_recommendation_id: str | None = None,
        last_recommendation_at: int | None = None,
        mode: str | None = None,
    ) -> None:
        ...

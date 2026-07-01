from __future__ import annotations

from typing import Protocol

from espresso_rl.domain.dreamer_control import DreamerControlSpec
from espresso_rl.domain.dreamer_telemetry import DreamerLiveTelemetry


class DreamerLiveInference(Protocol):
    """Model boundary used by the live-control application service."""

    @property
    def control_spec(self) -> DreamerControlSpec:
        ...

    def start_episode(self, telemetry: DreamerLiveTelemetry) -> None:
        ...

    def infer_action(self, telemetry: DreamerLiveTelemetry) -> dict[str, object] | None:
        ...

    def end_episode(self, *, machine_id: str, shot_id: str) -> None:
        ...

from __future__ import annotations

from typing import Protocol

from espresso_rl.domain.live_telemetry import (
    LiveShotEndedEvent,
    LiveShotSampleEvent,
    LiveShotSession,
    LiveShotStartedEvent,
)


class LiveShotSessionRepository(Protocol):
    def get_session(self, shot_id: str) -> LiveShotSession | None: ...

    def upsert_session(self, session: LiveShotSession) -> None: ...

    def get_sample(self, shot_id: str, sequence: int) -> LiveShotSampleEvent | None: ...

    def append_sample(
        self,
        sample: LiveShotSampleEvent,
        session: LiveShotSession,
    ) -> None: ...

    def reconcile_session(self, session: LiveShotSession) -> None: ...

    def expire_before(self, cutoff_ms: int, updated_at_ms: int) -> int: ...


class LiveShotConsumer(Protocol):
    def shot_started(self, event: LiveShotStartedEvent) -> None: ...

    def shot_sample(self, event: LiveShotSampleEvent) -> None: ...

    def shot_ended(self, event: LiveShotEndedEvent) -> None: ...

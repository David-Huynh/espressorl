from __future__ import annotations

from dataclasses import replace
from threading import RLock
from typing import Callable

from espresso_rl.domain.live_telemetry import (
    LiveShotEndedEvent,
    LiveShotEvent,
    LiveShotSampleEvent,
    LiveShotSession,
    LiveShotSessionStatus,
    LiveShotStartedEvent,
)
from espresso_rl.ports.live_telemetry import LiveShotConsumer, LiveShotSessionRepository

LIVE_SESSION_RETENTION_MS = 24 * 60 * 60 * 1000


class LiveShotTelemetryService:
    def __init__(
        self,
        repository: LiveShotSessionRepository,
        *,
        clock_ms: Callable[[], int],
        consumer: LiveShotConsumer | None = None,
    ) -> None:
        self._repository = repository
        self._clock_ms = clock_ms
        self._consumer = consumer
        self._lock = RLock()

    def handle(self, event: LiveShotEvent) -> None:
        if isinstance(event, LiveShotStartedEvent):
            self.start(event)
        elif isinstance(event, LiveShotSampleEvent):
            self.append(event)
        elif isinstance(event, LiveShotEndedEvent):
            self.end(event)
        else:
            raise TypeError("unsupported live-shot event")

    def start(self, event: LiveShotStartedEvent) -> None:
        with self._lock:
            now = self._clock_ms()
            self._repository.expire_before(now - LIVE_SESSION_RETENTION_MS, now)
            existing = self._repository.get_session(event.shot_id)
            session = LiveShotSession(
                shot_id=event.shot_id,
                install_id=event.install_id,
                machine_id=event.machine_id,
                started_at_ms=event.timestamp_ms,
                sample_interval_ms=event.sample_interval_ms,
                weight_source=event.weight_source,
                flow_source=event.flow_source,
                updated_at_ms=now,
            )
            if existing is not None:
                identity = (
                    existing.install_id,
                    existing.machine_id,
                    existing.started_at_ms,
                    existing.sample_interval_ms,
                    existing.weight_source,
                    existing.flow_source,
                )
                expected = (
                    session.install_id,
                    session.machine_id,
                    session.started_at_ms,
                    session.sample_interval_ms,
                    session.weight_source,
                    session.flow_source,
                )
                if identity != expected:
                    raise ValueError(f"live shot {event.shot_id} conflicts with an existing session")
                return
            self._repository.upsert_session(session)
            if self._consumer is not None:
                self._consumer.shot_started(event)

    def append(self, event: LiveShotSampleEvent) -> None:
        with self._lock:
            session = self._require_owner(event.shot_id, event.install_id, event.machine_id)
            existing = self._repository.get_sample(event.shot_id, event.sequence)
            if existing is not None:
                if existing != event:
                    raise ValueError("conflicting duplicate live-shot sample")
                return
            if session.status != LiveShotSessionStatus.ACTIVE:
                raise ValueError("live-shot session is not active")
            if session.last_sequence is not None and event.sequence <= session.last_sequence:
                raise ValueError("live-shot sequence regressed")
            if event.timestamp_ms != session.started_at_ms + event.elapsed_ms:
                raise ValueError("live-shot timestamp does not match start plus elapsed time")

            missing = (
                event.sequence
                if session.last_sequence is None
                else event.sequence - session.last_sequence - 1
            )
            updated_session = replace(
                session,
                last_sequence=event.sequence,
                sample_count=session.sample_count + 1,
                gap_count=session.gap_count + max(0, missing),
                updated_at_ms=self._clock_ms(),
            )
            self._repository.append_sample(event, updated_session)
            if self._consumer is not None:
                self._consumer.shot_sample(event)

    def end(self, event: LiveShotEndedEvent) -> None:
        with self._lock:
            session = self._require_owner(event.shot_id, event.install_id, event.machine_id)
            if session.status != LiveShotSessionStatus.ACTIVE:
                if session.ended_at_ms == event.timestamp_ms and session.end_state == event.end_state:
                    return
                raise ValueError("conflicting duplicate live-shot end")
            if event.timestamp_ms != session.started_at_ms + event.elapsed_ms:
                raise ValueError("live-shot end timestamp does not match start plus elapsed time")
            if session.last_sequence is not None and event.final_sequence < session.last_sequence:
                raise ValueError("live-shot final sequence regressed")
            missing_tail = (
                event.final_sequence + 1
                if session.last_sequence is None
                else event.final_sequence - session.last_sequence
            )
            self._repository.upsert_session(
                replace(
                    session,
                    status=LiveShotSessionStatus.ENDED,
                    gap_count=session.gap_count + max(0, missing_tail),
                    ended_at_ms=event.timestamp_ms,
                    end_state=event.end_state,
                    updated_at_ms=self._clock_ms(),
                )
            )
            if self._consumer is not None:
                self._consumer.shot_ended(event)

    def reconcile_completed_shot(self, shot_id: str, install_id: str, machine_id: str) -> bool:
        with self._lock:
            session = self._repository.get_session(shot_id)
            if session is None:
                return True
            now = self._clock_ms()
            if session.install_id != install_id or session.machine_id != machine_id:
                self._repository.reconcile_session(
                    replace(
                        session,
                        status=LiveShotSessionStatus.EXPIRED,
                        updated_at_ms=now,
                    )
                )
                return False
            self._repository.reconcile_session(
                replace(
                    session,
                    status=LiveShotSessionStatus.RECONCILED,
                    reconciled_at_ms=now,
                    updated_at_ms=now,
                )
            )
            return True

    def _require_owner(self, shot_id: str, install_id: str, machine_id: str) -> LiveShotSession:
        session = self._repository.get_session(shot_id)
        if session is None:
            raise ValueError("live-shot session has not started")
        if session.install_id != install_id or session.machine_id != machine_id:
            raise ValueError("live-shot event does not own the session")
        return session

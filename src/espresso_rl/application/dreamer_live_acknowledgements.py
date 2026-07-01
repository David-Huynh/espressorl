from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

from espresso_rl.domain.dreamer_control import (
    DREAMER_COMMAND_REPLAY_GRACE_MS,
    DreamerLiveControlAcknowledgement,
    DreamerLiveControlPublication,
)

DREAMER_LIVE_ACK_HISTORY_LIMIT = 256

ACK_OUTCOME_ACCEPTED = "accepted"
ACK_OUTCOME_REJECTED = "rejected"
ACK_OUTCOME_DUPLICATE = "duplicate"
ACK_OUTCOME_LATE = "late"
ACK_OUTCOME_MISMATCH = "mismatch"
ACK_OUTCOME_UNKNOWN = "unknown"
ACK_OUTCOME_TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class DreamerLiveControlAckResult:
    outcome: str
    reason_category: str
    accepted: bool
    sequence: int
    step_index: int


@dataclass(frozen=True)
class _PendingPublication:
    machine_id: str
    sequence: int
    step_index: int
    published_at_ms: int


@dataclass(frozen=True)
class _CompletedPublication:
    machine_id: str
    outcome: str


@dataclass
class _MachineAckStats:
    published_count: int = 0
    duplicate_publication_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    duplicate_ack_count: int = 0
    late_ack_count: int = 0
    mismatched_ack_count: int = 0
    unknown_ack_count: int = 0
    timed_out_count: int = 0
    last_result: str | None = None
    last_reason_category: str | None = None
    last_event_at_ms: int | None = None


class DreamerLiveControlAcknowledgementService:
    """Correlates bounded ESP32-received acknowledgements with live commands."""

    def __init__(
        self,
        *,
        clock_ms: Callable[[], int] | None = None,
        ack_timeout_ms: int = DREAMER_COMMAND_REPLAY_GRACE_MS,
        history_limit: int = DREAMER_LIVE_ACK_HISTORY_LIMIT,
    ) -> None:
        if isinstance(ack_timeout_ms, bool) or not isinstance(ack_timeout_ms, int) or ack_timeout_ms <= 0:
            raise ValueError("Dreamer live acknowledgement timeout must be a positive integer")
        if isinstance(history_limit, bool) or not isinstance(history_limit, int) or history_limit <= 0:
            raise ValueError("Dreamer live acknowledgement history limit must be a positive integer")
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._ack_timeout_ms = ack_timeout_ms
        self._history_limit = history_limit
        self._pending: OrderedDict[str, _PendingPublication] = OrderedDict()
        self._completed: OrderedDict[str, _CompletedPublication] = OrderedDict()
        self._stats: dict[str, _MachineAckStats] = {}
        self._lock = Lock()

    def record_publication(self, publication: DreamerLiveControlPublication) -> bool:
        if not isinstance(publication, DreamerLiveControlPublication):
            raise ValueError("Dreamer live control publication is invalid")
        now_ms = self._now_ms()
        with self._lock:
            self._expire_pending_locked(now_ms)
            stats = self._stats_for(publication.machine_id)
            stats.published_count += 1
            if publication.publication_id in self._pending or publication.publication_id in self._completed:
                stats.duplicate_publication_count += 1
                self._set_last_event(stats, "duplicate_publication", "none", now_ms)
                return False
            self._pending[publication.publication_id] = _PendingPublication(
                machine_id=publication.machine_id,
                sequence=publication.sequence,
                step_index=publication.step_index,
                published_at_ms=now_ms,
            )
            while len(self._pending) > self._history_limit:
                publication_id, pending = self._pending.popitem(last=False)
                self._mark_timed_out_locked(publication_id, pending, now_ms)
            return True

    def record_acknowledgement(
        self,
        acknowledgement: DreamerLiveControlAcknowledgement,
    ) -> DreamerLiveControlAckResult:
        if not isinstance(acknowledgement, DreamerLiveControlAcknowledgement):
            raise ValueError("Dreamer live control acknowledgement is invalid")
        now_ms = self._now_ms()
        with self._lock:
            self._expire_pending_locked(now_ms)
            completed = self._completed.get(acknowledgement.publication_id)
            if completed is not None:
                stats = self._stats_for(completed.machine_id)
                if completed.outcome == ACK_OUTCOME_TIMED_OUT:
                    stats.late_ack_count += 1
                    outcome = ACK_OUTCOME_LATE
                    self._set_last_event(stats, outcome, acknowledgement.reason_category, now_ms)
                else:
                    stats.duplicate_ack_count += 1
                    outcome = ACK_OUTCOME_DUPLICATE
                return _result(acknowledgement, outcome)

            pending = self._pending.get(acknowledgement.publication_id)
            if pending is None:
                stats = self._stats_for(acknowledgement.machine_id)
                stats.unknown_ack_count += 1
                self._set_last_event(stats, ACK_OUTCOME_UNKNOWN, acknowledgement.reason_category, now_ms)
                return _result(acknowledgement, ACK_OUTCOME_UNKNOWN)

            stats = self._stats_for(pending.machine_id)
            if (
                pending.machine_id != acknowledgement.machine_id
                or pending.sequence != acknowledgement.sequence
                or pending.step_index != acknowledgement.step_index
            ):
                stats.mismatched_ack_count += 1
                self._set_last_event(stats, ACK_OUTCOME_MISMATCH, acknowledgement.reason_category, now_ms)
                return _result(acknowledgement, ACK_OUTCOME_MISMATCH)

            self._pending.pop(acknowledgement.publication_id)
            outcome = ACK_OUTCOME_ACCEPTED if acknowledgement.accepted else ACK_OUTCOME_REJECTED
            if acknowledgement.accepted:
                stats.accepted_count += 1
            else:
                stats.rejected_count += 1
            self._remember_completed_locked(acknowledgement.publication_id, pending.machine_id, outcome)
            self._set_last_event(stats, outcome, acknowledgement.reason_category, now_ms)
            return _result(acknowledgement, outcome)

    def status_summary(self, machine_id: str) -> dict[str, object]:
        if not isinstance(machine_id, str) or not machine_id.strip():
            raise ValueError("Dreamer live acknowledgement machine_id is required")
        now_ms = self._now_ms()
        with self._lock:
            self._expire_pending_locked(now_ms)
            stats = self._stats.get(machine_id, _MachineAckStats())
            pending_count = sum(1 for pending in self._pending.values() if pending.machine_id == machine_id)
            if stats.published_count == 0:
                health = "idle"
            elif pending_count > 0:
                health = "waiting"
            elif stats.last_result in {ACK_OUTCOME_ACCEPTED, ACK_OUTCOME_DUPLICATE, "duplicate_publication"}:
                health = "healthy"
            else:
                health = "attention"
            return {
                "health": health,
                "published_count": stats.published_count,
                "pending_count": pending_count,
                "accepted_count": stats.accepted_count,
                "rejected_count": stats.rejected_count,
                "duplicate_ack_count": stats.duplicate_ack_count,
                "late_ack_count": stats.late_ack_count,
                "mismatched_ack_count": stats.mismatched_ack_count,
                "unknown_ack_count": stats.unknown_ack_count,
                "timed_out_count": stats.timed_out_count,
                "last_result": stats.last_result,
                "last_reason_category": stats.last_reason_category,
                "last_event_at_ms": stats.last_event_at_ms,
            }

    def reset(self, machine_id: str) -> None:
        if not isinstance(machine_id, str) or not machine_id.strip():
            raise ValueError("Dreamer live acknowledgement machine_id is required")
        with self._lock:
            self._pending = OrderedDict(
                (publication_id, pending)
                for publication_id, pending in self._pending.items()
                if pending.machine_id != machine_id
            )
            self._completed = OrderedDict(
                (publication_id, completed)
                for publication_id, completed in self._completed.items()
                if completed.machine_id != machine_id
            )
            self._stats.pop(machine_id, None)

    def _now_ms(self) -> int:
        value = self._clock_ms()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Dreamer live acknowledgement clock must return non-negative integer milliseconds")
        return value

    def _stats_for(self, machine_id: str) -> _MachineAckStats:
        stats = self._stats.get(machine_id)
        if stats is None:
            stats = _MachineAckStats()
            self._stats[machine_id] = stats
        return stats

    def _expire_pending_locked(self, now_ms: int) -> None:
        expired = [
            (publication_id, pending)
            for publication_id, pending in self._pending.items()
            if now_ms - pending.published_at_ms > self._ack_timeout_ms
        ]
        for publication_id, pending in expired:
            self._pending.pop(publication_id, None)
            self._mark_timed_out_locked(publication_id, pending, now_ms)

    def _mark_timed_out_locked(
        self,
        publication_id: str,
        pending: _PendingPublication,
        now_ms: int,
    ) -> None:
        stats = self._stats_for(pending.machine_id)
        stats.timed_out_count += 1
        self._remember_completed_locked(publication_id, pending.machine_id, ACK_OUTCOME_TIMED_OUT)
        self._set_last_event(stats, ACK_OUTCOME_TIMED_OUT, "timeout", now_ms)

    def _remember_completed_locked(self, publication_id: str, machine_id: str, outcome: str) -> None:
        self._completed[publication_id] = _CompletedPublication(machine_id=machine_id, outcome=outcome)
        self._completed.move_to_end(publication_id)
        while len(self._completed) > self._history_limit:
            self._completed.popitem(last=False)

    @staticmethod
    def _set_last_event(
        stats: _MachineAckStats,
        outcome: str,
        reason_category: str,
        now_ms: int,
    ) -> None:
        stats.last_result = outcome
        stats.last_reason_category = reason_category
        stats.last_event_at_ms = now_ms


def _result(
    acknowledgement: DreamerLiveControlAcknowledgement,
    outcome: str,
) -> DreamerLiveControlAckResult:
    return DreamerLiveControlAckResult(
        outcome=outcome,
        reason_category=acknowledgement.reason_category,
        accepted=acknowledgement.accepted,
        sequence=acknowledgement.sequence,
        step_index=acknowledgement.step_index,
    )

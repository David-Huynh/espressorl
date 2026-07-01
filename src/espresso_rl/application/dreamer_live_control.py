from __future__ import annotations

from dataclasses import dataclass

from espresso_rl.domain.dreamer_control import (
    DREAMER_DYNAMIC_CONTROL_ACCEPT,
    DREAMER_DYNAMIC_CONTROL_REPLAY_LAST,
    DREAMER_DYNAMIC_CONTROL_WAIT_FOR_FIRST_COMMAND,
    DreamerControlSpec,
    DreamerLiveControlPublication,
    resolve_live_dynamic_control_action,
)
from espresso_rl.ports.runtime import AutoTuningRuntimePublisher


@dataclass(frozen=True)
class DreamerLiveControlResult:
    publication: DreamerLiveControlPublication | None
    state_key: str
    last_sanitized_action: dict[str, object] | None
    last_command_at_ms: int | None
    sequence: int


@dataclass
class _DreamerLiveControlState:
    started_at_ms: int
    sequence: int = 0
    last_sanitized_action: dict[str, object] | None = None
    last_command_at_ms: int | None = None


class DreamerLiveControlApplication:
    """Application use case for canonical Dreamer live target commands."""

    def __init__(self, publisher: AutoTuningRuntimePublisher) -> None:
        self._publisher = publisher
        self._states: dict[str, _DreamerLiveControlState] = {}

    def handle_live_action(
        self,
        *,
        machine_id: str,
        profile_id: str | None,
        action: dict[str, object] | None,
        control_spec: DreamerControlSpec,
        step_index: int,
        now_ms: int,
    ) -> DreamerLiveControlResult:
        if not isinstance(machine_id, str) or not machine_id.strip():
            raise ValueError("machine_id is required")
        if isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms < 0:
            raise ValueError("now_ms must be a non-negative integer")
        state_key = _state_key(machine_id, profile_id)
        state = self._states.get(state_key)
        if state is None:
            state = _DreamerLiveControlState(started_at_ms=now_ms)
            self._states[state_key] = state

        milliseconds_since_last_command = _milliseconds_since_last_command(
            state,
            now_ms=now_ms,
            has_new_action=action is not None,
        )
        decision = resolve_live_dynamic_control_action(
            action,
            last_sanitized_action=state.last_sanitized_action,
            control_spec=control_spec,
            step_index=step_index,
            milliseconds_since_last_command=milliseconds_since_last_command,
        )
        if decision.status == DREAMER_DYNAMIC_CONTROL_ACCEPT:
            state.last_sanitized_action = dict(decision.action or {})
            state.last_command_at_ms = now_ms

        if decision.status == DREAMER_DYNAMIC_CONTROL_WAIT_FOR_FIRST_COMMAND:
            return DreamerLiveControlResult(
                publication=None,
                state_key=state_key,
                last_sanitized_action=state.last_sanitized_action,
                last_command_at_ms=state.last_command_at_ms,
                sequence=state.sequence,
            )

        if decision.status == DREAMER_DYNAMIC_CONTROL_REPLAY_LAST and decision.action is None:
            raise ValueError("Dreamer replay decision must carry the last sanitized action")

        state.sequence += 1
        publication = DreamerLiveControlPublication(
            machine_id=machine_id,
            profile_id=profile_id,
            sequence=state.sequence,
            step_index=step_index,
            issued_at_ms=now_ms,
            decision=decision,
        )
        self._publisher.publish_dreamer_live_control(publication)
        return DreamerLiveControlResult(
            publication=publication,
            state_key=state_key,
            last_sanitized_action=state.last_sanitized_action,
            last_command_at_ms=state.last_command_at_ms,
            sequence=state.sequence,
        )

    def reset(self, *, machine_id: str, profile_id: str | None = None) -> None:
        self._states.pop(_state_key(machine_id, profile_id), None)


def _milliseconds_since_last_command(
    state: _DreamerLiveControlState,
    *,
    now_ms: int,
    has_new_action: bool,
) -> int:
    if has_new_action:
        return 0
    anchor = state.last_command_at_ms if state.last_command_at_ms is not None else state.started_at_ms
    return max(0, now_ms - anchor)


def _state_key(machine_id: str, profile_id: str | None) -> str:
    return f"{machine_id}|{profile_id or ''}"

from __future__ import annotations

import re
from collections.abc import Callable

from espresso_rl.domain.dreamer_telemetry import (
    DREAMER_AUTO_PROFILE_ID,
    DREAMER_LIVE_CONTEXT_WINDOW_SIZE,
    DreamerLiveEpisodeContext,
    DreamerLiveTelemetry,
)
from espresso_rl.domain.events import MachineStateEvent, OptimizerSettingsEvent
from espresso_rl.ports.dreamer_live_inference import DreamerHistoryEncoder
from espresso_rl.ports.repositories import ShotRepository


class DreamerLiveContextService:
    """Builds trusted live inference context from canonical local state."""

    def __init__(
        self,
        *,
        shots: ShotRepository,
        history_encoder: DreamerHistoryEncoder,
        install_id: str,
        machine_id: str,
        fallback_microns_per_step: float,
        fallback_dose_g: float,
        clock: Callable[[], int],
        history_limit: int = 500,
    ) -> None:
        self._shots = shots
        self._history_encoder = history_encoder
        self._install_id = install_id
        self._machine_id = machine_id
        self._fallback_microns_per_step = float(fallback_microns_per_step)
        self._fallback_dose_g = float(fallback_dose_g)
        self._clock = clock
        self._history_limit = max(DREAMER_LIVE_CONTEXT_WINDOW_SIZE, min(int(history_limit), 2_000))
        self._machine_states: dict[str, MachineStateEvent] = {}
        self._taste_objectives: dict[str, dict[str, str]] = {}

    def update_machine_state(self, event: MachineStateEvent) -> None:
        if event.install_id != self._install_id or not _same_machine(event.machine_id, self._machine_id):
            return
        self._machine_states[event.machine_id] = event

    def update_optimizer_settings(self, event: OptimizerSettingsEvent) -> None:
        if event.install_id != self._install_id or not _same_machine(event.machine_id, self._machine_id):
            return
        self._taste_objectives[event.machine_id] = dict(event.taste_objective)

    def context_for(self, telemetry: DreamerLiveTelemetry) -> DreamerLiveEpisodeContext:
        state = self._machine_states.get(telemetry.machine_id)
        if state is None:
            state = next(
                (
                    candidate
                    for machine_id, candidate in self._machine_states.items()
                    if _same_machine(machine_id, telemetry.machine_id)
                ),
                None,
            )
        if state is None:
            raise ValueError("Dreamer live context requires current canonical machine state")
        if state.profile_id != DREAMER_AUTO_PROFILE_ID or telemetry.profile_id != DREAMER_AUTO_PROFILE_ID:
            raise ValueError("Dreamer live context requires the automatic profile")

        microns_per_step = state.microns_per_step or self._fallback_microns_per_step
        dose_g = state.dose_in_g or self._fallback_dose_g
        relative_steps = state.relative_grind_steps_from_reference
        direction = state.grinder_step_direction.value
        relative_um = (
            relative_steps * microns_per_step * (1.0 if direction == "higher_is_finer" else -1.0)
            if relative_steps is not None
            else None
        )
        now = int(self._clock())
        episodes = self._historical_episodes(state, before_timestamp=now)
        taste_objective = self._taste_objectives.get(telemetry.machine_id)
        if taste_objective is None:
            taste_objective = next(
                (
                    candidate
                    for machine_id, candidate in self._taste_objectives.items()
                    if _same_machine(machine_id, telemetry.machine_id)
                ),
                {"mode": "auto"},
            )
        return DreamerLiveEpisodeContext(
            install_id=self._install_id,
            machine_id=telemetry.machine_id,
            timestamp=now,
            bean_context_id=state.bean_context_id,
            bean_context_name=state.bean_context_name,
            grinder_context_id=state.grinder_context_id,
            relative_grind_steps_from_reference=relative_steps,
            relative_grind_um_from_reference=relative_um,
            grind_observed=relative_steps is not None,
            dose_g=dose_g,
            dose_observed=False,
            initial_target_yield_g=telemetry.target_yield_g,
            initial_target_yield_observed=True,
            microns_per_step=microns_per_step,
            step_direction=direction,
            profile_id=telemetry.profile_id,
            profile_type="pro",
            profile_phase_count=1,
            taste_objective=taste_objective,
            historical_episodes=episodes,
        )

    def _historical_episodes(
        self,
        state: MachineStateEvent,
        *,
        before_timestamp: int,
    ) -> tuple[dict, ...]:
        current_bean_key = _bean_context_key(state.bean_context_name, state.bean_context_id)
        history = []
        for shot in self._shots.list_machine_shots(
            self._install_id,
            state.machine_id,
            limit=self._history_limit,
        ):
            if shot.timestamp >= before_timestamp:
                continue
            if shot.grinder_context_id != state.grinder_context_id:
                continue
            if _bean_context_key(shot.bean_context_name, shot.bean_context_id) != current_bean_key:
                continue
            history.append(shot)
        if not history:
            return ()
        episodes = self._history_encoder.encode(history)
        return tuple(episodes[-DREAMER_LIVE_CONTEXT_WINDOW_SIZE:])


def _bean_context_key(name: str | None, context_id: str | None) -> str:
    normalized_name = _normalized_key(name)
    if normalized_name:
        return normalized_name
    if not context_id:
        return ""
    value = context_id.strip()
    if value.casefold().startswith("bean_"):
        parts = [part for part in value[5:].split("_") if part]
        while parts and parts[-1].isdigit():
            parts.pop()
        value = " ".join(parts) or value
    return _normalized_key(value)


def _normalized_key(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _same_machine(left: str, right: str) -> bool:
    return left.removeprefix("gaggimate:").casefold() == right.removeprefix("gaggimate:").casefold()

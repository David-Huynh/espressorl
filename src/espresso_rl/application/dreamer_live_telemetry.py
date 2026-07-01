from __future__ import annotations

from dataclasses import dataclass

from espresso_rl.application.dreamer_live_control import (
    DreamerLiveControlApplication,
    DreamerLiveControlResult,
)
from espresso_rl.domain.dreamer_control import DreamerControlSpec
from espresso_rl.domain.dreamer_telemetry import DREAMER_AUTO_PROFILE_ID, DreamerLiveTelemetry
from espresso_rl.ports.dreamer_live_inference import DreamerLiveInference


@dataclass(frozen=True)
class DreamerLiveTelemetryResult:
    outcome: str
    inference_called: bool = False
    control_result: DreamerLiveControlResult | None = None


@dataclass
class _TelemetryState:
    shot_id: str
    profile_id: str
    last_step_index: int = -1


class DreamerLiveTelemetryApplication:
    """Owns live telemetry ordering and the inference-to-control use case."""

    def __init__(
        self,
        *,
        inference: DreamerLiveInference | None,
        live_control: DreamerLiveControlApplication | None,
    ) -> None:
        if (inference is None) != (live_control is None):
            raise ValueError("Dreamer live inference and control must be configured together")
        self._inference = inference
        self._live_control = live_control
        self._states: dict[str, _TelemetryState] = {}

    def handle_telemetry(self, telemetry: DreamerLiveTelemetry) -> DreamerLiveTelemetryResult:
        if not isinstance(telemetry, DreamerLiveTelemetry):
            raise ValueError("telemetry must be a canonical DreamerLiveTelemetry event")
        if telemetry.profile_id != DREAMER_AUTO_PROFILE_ID:
            return DreamerLiveTelemetryResult("wrong_profile")
        if self._inference is None or self._live_control is None:
            return DreamerLiveTelemetryResult("inactive_model")
        if not _telemetry_supports_control_spec(telemetry, self._inference.control_spec):
            return DreamerLiveTelemetryResult("incompatible_capabilities")

        state = self._states.get(telemetry.machine_id)
        if state is None or state.shot_id != telemetry.shot_id:
            if telemetry.step_index != 0:
                return DreamerLiveTelemetryResult("late_episode_start")
            if state is not None:
                self._inference.end_episode(machine_id=telemetry.machine_id, shot_id=state.shot_id)
                self._live_control.reset(machine_id=telemetry.machine_id, profile_id=state.profile_id)
            state = _TelemetryState(shot_id=telemetry.shot_id, profile_id=telemetry.profile_id)
            self._states[telemetry.machine_id] = state
            self._inference.start_episode(telemetry)

        if telemetry.profile_id != state.profile_id:
            return DreamerLiveTelemetryResult("profile_changed")
        if telemetry.step_index <= state.last_step_index:
            return DreamerLiveTelemetryResult("duplicate_or_out_of_order")

        state.last_step_index = telemetry.step_index
        if not self._inference.control_spec.is_decision_step(telemetry.step_index):
            self._inference.infer_action(telemetry)
            return DreamerLiveTelemetryResult("observation_accepted", inference_called=True)

        action = self._inference.infer_action(telemetry)
        control_result = self._live_control.handle_live_action(
            machine_id=telemetry.machine_id,
            profile_id=telemetry.profile_id,
            action=action,
            control_spec=self._inference.control_spec,
            step_index=telemetry.step_index,
            now_ms=telemetry.elapsed_ms,
        )
        return DreamerLiveTelemetryResult(
            "decision_published" if control_result.publication is not None else "decision_waiting",
            inference_called=True,
            control_result=control_result,
        )

    def end_episode(self, *, machine_id: str, shot_id: str | None = None) -> bool:
        state = self._states.get(machine_id)
        if state is None or (shot_id is not None and state.shot_id != shot_id):
            return False
        self._states.pop(machine_id, None)
        if self._inference is not None and self._live_control is not None:
            self._inference.end_episode(machine_id=machine_id, shot_id=state.shot_id)
            self._live_control.reset(machine_id=machine_id, profile_id=state.profile_id)
        return True


def _telemetry_supports_control_spec(
    telemetry: DreamerLiveTelemetry,
    control_spec: DreamerControlSpec,
) -> bool:
    if not control_spec.dynamic_control_enabled:
        return False
    if control_spec.observation_interval_ms != telemetry.sample_interval_ms:
        return False
    capabilities = telemetry.capabilities
    required = {
        "pressure_control_allowed": control_spec.pressure_control_allowed,
        "flow_control_allowed": control_spec.flow_control_allowed,
        "pump_control_allowed": control_spec.pump_control_allowed,
        "valve_control_allowed": control_spec.valve_control_allowed,
        "temperature_control_allowed": control_spec.temperature_control_allowed,
        "stop_control_allowed": control_spec.stop_control_allowed,
    }
    return all(not enabled or getattr(capabilities, field_name) for field_name, enabled in required.items())

from __future__ import annotations

import logging
from dataclasses import dataclass

from espresso_rl.application.dreamer_live_control import (
    DreamerLiveControlApplication,
    DreamerLiveControlResult,
)
from espresso_rl.domain.dreamer_control import DreamerControlSpec
from espresso_rl.domain.dreamer_telemetry import DREAMER_AUTO_PROFILE_ID, DreamerLiveTelemetry
from espresso_rl.ports.dreamer_live_inference import DreamerLiveContextProvider, DreamerLiveInference


logger = logging.getLogger(__name__)


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
        context_provider: DreamerLiveContextProvider | None = None,
        enabled: bool = False,
    ) -> None:
        configured = (inference is not None, live_control is not None, context_provider is not None)
        if any(configured) and not all(configured):
            raise ValueError("Dreamer live inference, control, and context provider must be configured together")
        self._inference = inference
        self._live_control = live_control
        self._context_provider = context_provider
        self._enabled = bool(enabled)
        self._states: dict[str, _TelemetryState] = {}

    def handle_telemetry(self, telemetry: DreamerLiveTelemetry) -> DreamerLiveTelemetryResult:
        if not isinstance(telemetry, DreamerLiveTelemetry):
            raise ValueError("telemetry must be a canonical DreamerLiveTelemetry event")
        if telemetry.profile_id != DREAMER_AUTO_PROFILE_ID:
            return DreamerLiveTelemetryResult("wrong_profile")
        if self._inference is None or self._live_control is None or self._context_provider is None:
            return DreamerLiveTelemetryResult("inactive_model")
        if not self._enabled:
            return DreamerLiveTelemetryResult("inactive_optimizer")
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
            try:
                context = self._context_provider.context_for(telemetry)
            except ValueError:
                return DreamerLiveTelemetryResult("context_unavailable")
            try:
                self._inference.start_episode(telemetry, context)
            except (RuntimeError, ValueError) as exc:
                logger.warning("Dreamer live inference failed to initialize: %s", exc)
                self._inference.end_episode(machine_id=telemetry.machine_id, shot_id=telemetry.shot_id)
                return DreamerLiveTelemetryResult("inference_failed")
            self._states[telemetry.machine_id] = state

        if telemetry.profile_id != state.profile_id:
            return DreamerLiveTelemetryResult("profile_changed")
        if telemetry.step_index <= state.last_step_index:
            return DreamerLiveTelemetryResult("duplicate_or_out_of_order")

        state.last_step_index = telemetry.step_index
        if not self._inference.control_spec.is_decision_step(telemetry.step_index):
            try:
                self._inference.infer_action(telemetry)
            except (RuntimeError, ValueError) as exc:
                logger.warning("Dreamer recurrent observation failed: %s", exc)
                return DreamerLiveTelemetryResult("inference_failed", inference_called=True)
            return DreamerLiveTelemetryResult("observation_accepted", inference_called=True)

        inference_failed = False
        try:
            action = self._inference.infer_action(telemetry)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Dreamer live actor inference failed: %s", exc)
            action = None
            inference_failed = True
        control_result = self._live_control.handle_live_action(
            machine_id=telemetry.machine_id,
            profile_id=telemetry.profile_id,
            action=action,
            control_spec=self._inference.control_spec,
            step_index=telemetry.step_index,
            now_ms=telemetry.elapsed_ms,
        )
        return DreamerLiveTelemetryResult(
            (
                "inference_failed"
                if inference_failed
                else "decision_published" if control_result.publication is not None else "decision_waiting"
            ),
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

    def set_enabled(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("Dreamer live telemetry enabled must be boolean")
        if not enabled:
            for machine_id in tuple(self._states):
                self.end_episode(machine_id=machine_id)
        self._enabled = enabled


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
        "pump_mode_control_allowed": control_spec.control_allowed_for_field("pump_target_mode"),
        "valve_control_allowed": control_spec.valve_control_allowed,
        "temperature_control_allowed": control_spec.temperature_control_allowed,
        "stop_control_allowed": control_spec.stop_control_allowed,
    }
    return all(not enabled or getattr(capabilities, field_name) for field_name, enabled in required.items())

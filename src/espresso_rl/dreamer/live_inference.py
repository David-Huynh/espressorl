from __future__ import annotations

from dataclasses import dataclass

import torch

from espresso_rl.domain.dreamer_control import DREAMER_DYNAMIC_ACTION_FIELDS, DreamerControlSpec
from espresso_rl.domain.dreamer_pre_shot import (
    DREAMER_PRE_SHOT_ACTION_FIELDS,
    build_dreamer_pre_shot_action,
    encode_dreamer_pre_shot_action,
)
from espresso_rl.domain.dreamer_telemetry import (
    DREAMER_PUMP_TARGET_MODE_FLOW,
    DREAMER_PUMP_TARGET_MODE_PRESSURE,
    DreamerLiveEpisodeContext,
    DreamerLiveTelemetry,
)
from espresso_rl.domain.model_checkpoint import DreamerCheckpointArchitecture
from espresso_rl.dreamer.checkpoint_inference import DreamerShadowModels
from espresso_rl.dreamer.dataset import (
    DREAMER_CONSTRAINT_FEATURES,
    DREAMER_DYNAMIC_ACTION_FEATURES,
    DREAMER_OBSERVED_TARGET_FEATURES,
    build_dreamer_context_encoder_batch,
    encode_dreamer_static_context,
    encode_dreamer_taste_objective,
)
from espresso_rl.dreamer.reference_world_model import behavior_tensor_from_parts


@dataclass
class _LiveInferenceState:
    shot_id: str
    profile_id: str
    deter: torch.Tensor
    stoch: torch.Tensor
    context_state: torch.Tensor
    static_context: torch.Tensor
    taste_objective: torch.Tensor
    pre_shot_actions: torch.Tensor
    pre_shot_action_mask: torch.Tensor
    pre_shot_capability_mask: torch.Tensor
    target_state: torch.Tensor
    held_action: torch.Tensor
    held_action_mask: torch.Tensor
    last_step_index: int = -1


class CheckpointDreamerLiveInference:
    """Deterministic recurrent inference over an authenticated Dreamer checkpoint."""

    def __init__(
        self,
        *,
        models: DreamerShadowModels,
        architecture: DreamerCheckpointArchitecture,
    ) -> None:
        if not architecture.control_spec.dynamic_control_enabled:
            raise ValueError("Dreamer checkpoint does not enable dynamic control")
        if architecture.observation_dim != 5:
            raise ValueError("Dreamer checkpoint live observation layout is incompatible")
        if architecture.static_dim != len(encode_dreamer_static_context(_neutral_static_context())):
            raise ValueError("Dreamer checkpoint live static context layout is incompatible")
        if architecture.live_action_dim != len(DREAMER_DYNAMIC_ACTION_FEATURES):
            raise ValueError("Dreamer checkpoint live action layout is incompatible")
        if models.actor.control_spec != architecture.control_spec:
            raise ValueError("Dreamer checkpoint actor control spec is incompatible")
        self._models = models
        self._architecture = architecture
        self._states: dict[str, _LiveInferenceState] = {}

    @property
    def control_spec(self) -> DreamerControlSpec:
        return self._architecture.control_spec

    @torch.no_grad()
    def start_episode(
        self,
        telemetry: DreamerLiveTelemetry,
        context: DreamerLiveEpisodeContext,
    ) -> None:
        if telemetry.machine_id != context.machine_id or telemetry.profile_id != context.profile_id:
            raise ValueError("Dreamer live inference context identity does not match telemetry")
        if telemetry.step_index != 0:
            raise ValueError("Dreamer live inference episode must start at step zero")
        context_batch = build_dreamer_context_encoder_batch(
            context.historical_episodes,
            target_timestamp=context.timestamp,
            device="cpu",
        )
        context_state = self._models.context_encoder(context_batch)
        static_context = torch.tensor(
            [encode_dreamer_static_context(context.static_context())],
            dtype=torch.float32,
        )
        taste_objective = torch.tensor(
            [encode_dreamer_taste_objective(context.taste_objective)],
            dtype=torch.float32,
        )
        pre_shot = _pre_shot_tensors(telemetry, context, self._architecture)
        deter, stoch = self._models.world_model.initial_state(1, torch.device("cpu"))
        action_dim = len(DREAMER_DYNAMIC_ACTION_FEATURES)
        self._states[telemetry.machine_id] = _LiveInferenceState(
            shot_id=telemetry.shot_id,
            profile_id=telemetry.profile_id,
            deter=deter,
            stoch=stoch,
            context_state=context_state,
            static_context=static_context,
            taste_objective=taste_objective,
            pre_shot_actions=pre_shot[0],
            pre_shot_action_mask=pre_shot[1],
            pre_shot_capability_mask=pre_shot[2],
            target_state=_target_state_from_telemetry(telemetry),
            held_action=torch.zeros((1, action_dim), dtype=torch.float32),
            held_action_mask=torch.zeros((1, action_dim), dtype=torch.float32),
        )

    @torch.no_grad()
    def infer_action(self, telemetry: DreamerLiveTelemetry) -> dict[str, object] | None:
        state = self._states.get(telemetry.machine_id)
        if state is None or state.shot_id != telemetry.shot_id or state.profile_id != telemetry.profile_id:
            raise ValueError("Dreamer live inference episode is not initialized")
        if telemetry.step_index <= state.last_step_index:
            raise ValueError("Dreamer live inference telemetry is duplicate or out of order")
        state.target_state = _target_state_from_telemetry(telemetry)
        decision_step = self.control_spec.is_decision_step(telemetry.step_index)
        control_mask = _control_mask(self.control_spec, telemetry) if decision_step else _zero_action_tensor()
        observed_targets, observed_target_mask = _observed_target_tensors(telemetry)
        behavior = behavior_tensor_from_parts(
            observed_profile_targets=observed_targets,
            observed_profile_target_mask=observed_target_mask,
            dynamic_actions=state.held_action,
            dynamic_action_mask=state.held_action_mask,
            control_action_mask=control_mask,
            constraints=torch.tensor(
                [[1.0 if self.control_spec.constraints()[name] else 0.0 for name in DREAMER_CONSTRAINT_FEATURES]],
                dtype=torch.float32,
            ),
            decision_step_mask=torch.tensor([1.0 if decision_step else 0.0], dtype=torch.float32),
            pre_shot_actions=state.pre_shot_actions,
            pre_shot_action_mask=state.pre_shot_action_mask,
            pre_shot_capability_mask=state.pre_shot_capability_mask,
        )
        observed = self._models.world_model.observe_step(
            observation=torch.tensor([telemetry.observation()], dtype=torch.float32),
            behavior=behavior,
            static_context=state.static_context,
            deter=state.deter,
            stoch=state.stoch,
            context_state=state.context_state,
            is_first=torch.tensor([telemetry.step_index == 0], dtype=torch.bool),
            valid=torch.ones(1, dtype=torch.float32),
            sample=False,
        )
        state.deter = observed["deter"]
        state.stoch = observed["stoch"]
        state.last_step_index = telemetry.step_index
        if not decision_step:
            return None

        actor_output = self._models.actor.select_dynamic(
            observed["features"],
            state.taste_objective,
            control_mask,
            state.target_state,
            torch.tensor(
                [[telemetry.pressure_bar, telemetry.pump_flow_ml_s]],
                dtype=torch.float32,
            ),
        )
        state.held_action = actor_output["dynamic_actions"]
        state.held_action_mask = actor_output["dynamic_action_mask"]
        return _action_from_actor_output(state.held_action[0], state.held_action_mask[0])

    def end_episode(self, *, machine_id: str, shot_id: str) -> None:
        state = self._states.get(machine_id)
        if state is not None and state.shot_id == shot_id:
            self._states.pop(machine_id, None)


def _pre_shot_tensors(
    telemetry: DreamerLiveTelemetry,
    context: DreamerLiveEpisodeContext,
    architecture: DreamerCheckpointArchitecture,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values: dict[str, object] = {
        "dose_target_g": context.dose_g,
        "yield_target_g": telemetry.target_yield_g,
    }
    capabilities = {
        "grind_delta_steps_from_current",
        "dose_target_g",
        "yield_target_g",
        "initial_stage_duration_s",
    }
    telemetry_capabilities = telemetry.capabilities
    if telemetry_capabilities.temperature_control_allowed:
        values["temperature_target_c"] = telemetry.temperature_target_c
        capabilities.add("temperature_target_c")
    if telemetry_capabilities.valve_control_allowed:
        values["valve_open"] = telemetry.valve_open
        capabilities.add("valve_open")
    if (
        telemetry.pump_target_mode == DREAMER_PUMP_TARGET_MODE_PRESSURE
        and telemetry_capabilities.pressure_control_allowed
    ):
        values["pump_target_mode"] = telemetry.pump_target_mode
        values["pressure_target_bar"] = telemetry.pressure_target_bar
        capabilities.update({"pump_target_mode", "pressure_target_bar"})
    elif (
        telemetry.pump_target_mode == DREAMER_PUMP_TARGET_MODE_FLOW
        and telemetry_capabilities.flow_control_allowed
    ):
        values["pump_target_mode"] = telemetry.pump_target_mode
        values["flow_target_ml_s"] = telemetry.pump_flow_target_ml_s
        capabilities.update({"pump_target_mode", "flow_target_ml_s"})
    observed = set(values)
    action = build_dreamer_pre_shot_action(
        values=values,
        observed_fields=observed,
        capability_fields=capabilities,
    )
    encoded, _, observed_mask, capability_mask = encode_dreamer_pre_shot_action(
        action,
        spec=architecture.pre_shot_action_spec,
    )
    return (
        torch.tensor([encoded], dtype=torch.float32),
        torch.tensor([observed_mask], dtype=torch.float32),
        torch.tensor([capability_mask], dtype=torch.float32),
    )


def _observed_target_tensors(telemetry: DreamerLiveTelemetry) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.tensor([telemetry.observed_profile_targets()], dtype=torch.float32)
    mask = torch.tensor(
        [[
            1.0 if telemetry.pump_target_mode == DREAMER_PUMP_TARGET_MODE_PRESSURE else 0.0,
            1.0 if telemetry.pump_target_mode == DREAMER_PUMP_TARGET_MODE_FLOW else 0.0,
            1.0,
            1.0,
            1.0,
        ]],
        dtype=torch.float32,
    )
    return values, mask


def _target_state_from_telemetry(telemetry: DreamerLiveTelemetry) -> torch.Tensor:
    values = {
        "pump_target_mode": float(telemetry.pump_target_mode),
        "pressure_target_bar": telemetry.pressure_target_bar,
        "flow_target_ml_s": telemetry.pump_flow_target_ml_s,
        "valve_position": 1.0 if telemetry.valve_open else 0.0,
        "temperature_target_c": telemetry.temperature_target_c,
        "yield_stop_target_g": telemetry.target_yield_g,
        "stop": 0.0,
    }
    return torch.tensor(
        [[values[name] for name in DREAMER_DYNAMIC_ACTION_FEATURES]],
        dtype=torch.float32,
    )


def _control_mask(spec: DreamerControlSpec, telemetry: DreamerLiveTelemetry) -> torch.Tensor:
    capabilities = telemetry.capabilities
    pressure_allowed = spec.pressure_control_allowed and capabilities.pressure_control_allowed
    flow_allowed = spec.flow_control_allowed and capabilities.flow_control_allowed
    values = {
        "pump_target_mode": (
            spec.control_allowed_for_field("pump_target_mode")
            and capabilities.pump_mode_control_allowed
            and (pressure_allowed or flow_allowed)
        ),
        "pressure_target_bar": pressure_allowed,
        "flow_target_ml_s": flow_allowed,
        "valve_position": spec.valve_control_allowed and capabilities.valve_control_allowed,
        "temperature_target_c": spec.temperature_control_allowed and capabilities.temperature_control_allowed,
        "yield_stop_target_g": spec.stop_control_allowed and capabilities.stop_control_allowed,
        "stop": spec.stop_control_allowed and capabilities.stop_control_allowed,
    }
    return torch.tensor(
        [[1.0 if values[name] else 0.0 for name in DREAMER_DYNAMIC_ACTION_FEATURES]],
        dtype=torch.float32,
    )


def _action_from_actor_output(values: torch.Tensor, mask: torch.Tensor) -> dict[str, object]:
    action: dict[str, object] = {}
    for index, field_name in enumerate(DREAMER_DYNAMIC_ACTION_FIELDS):
        if float(mask[index].item()) <= 0.5:
            continue
        value = float(values[index].item())
        if field_name == "pump_target_mode":
            action[field_name] = int(round(value))
        elif field_name == "stop":
            action[field_name] = value >= 0.5
        else:
            action[field_name] = value
    return action


def _zero_action_tensor() -> torch.Tensor:
    return torch.zeros((1, len(DREAMER_DYNAMIC_ACTION_FEATURES)), dtype=torch.float32)


def _neutral_static_context() -> dict[str, object]:
    return {
        "relative_grind_steps_from_reference": 0.0,
        "relative_grind_um_from_reference": 0.0,
        "dose_g": 18.0,
        "initial_target_yield_g": 36.0,
        "target_ratio": 2.0,
        "grind_observed": False,
        "dose_observed": False,
        "initial_target_yield_observed": True,
        "microns_per_step": 10.0,
        "step_direction": "higher_is_finer",
        "profile_id": "dreamer_auto",
        "profile_type": "pro",
        "profile_phase_count": 1,
        "taste_objective": {"mode": "auto"},
    }

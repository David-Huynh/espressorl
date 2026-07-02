from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path

import torch

from espresso_rl.adapters.dreamer_history import CanonicalDreamerHistoryEncoder
from espresso_rl.application.dreamer_live_context import DreamerLiveContextService
from espresso_rl.domain.dreamer_control import DreamerControlSpec
from espresso_rl.domain.dreamer_live_action import DREAMER_LIVE_ACTION_FIELDS, DreamerLiveActionSpec
from espresso_rl.domain.dreamer_telemetry import (
    DREAMER_AUTO_PROFILE_ID,
    DreamerLiveEpisodeContext,
    DreamerLiveTelemetry,
    DreamerLiveTelemetryCapabilities,
)
from espresso_rl.domain.events import MachineStateEvent, OptimizerSettingsEvent
from espresso_rl.domain.models import MachineState
from espresso_rl.dreamer.checkpoint_inference import (
    DreamerShadowModels,
    checkpoint_architecture_from_models,
)
from espresso_rl.dreamer.context_encoder import DreamerContextEncoder, DreamerContextEncoderConfig
from espresso_rl.dreamer.dataset import (
    DREAMER_CONTEXT_TIME_FEATURES,
    DREAMER_CONTEXT_TRAJECTORY_EMBEDDING_FEATURES,
    DREAMER_RESOLVED_CONTROL_FEATURES,
    DREAMER_STATIC_CONTEXT_FEATURES,
    DREAMER_TERMINAL_FEATURES,
)
from espresso_rl.dreamer.imagination import (
    DreamerV3ImaginationActor,
    DreamerV3ImaginationConfig,
    DreamerV3ImaginationCritic,
)
from espresso_rl.dreamer.live_inference import CheckpointDreamerLiveInference
from espresso_rl.dreamer.reference_world_model import (
    DreamerV3VectorWorldModel,
    default_world_model_config,
)
from tests.test_application_service import MemoryShotRepository
from tests.test_dreamer_recommendations import local_shot


class DreamerLiveInferenceTests(unittest.TestCase):
    def test_context_application_depends_on_ports_and_domain_not_dreamer_adapter(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "espresso_rl"
            / "application"
            / "dreamer_live_context.py"
        )
        imports = {
            node.module
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.ImportFrom) and node.module
        }

        self.assertFalse(any(name.startswith("espresso_rl.adapters") for name in imports))
        self.assertFalse(any(name.startswith("espresso_rl.dreamer") for name in imports))

    def test_checkpoint_inference_updates_rssm_and_emits_one_pump_mode(self) -> None:
        inference, actor = _inference()
        _select_live_actions(actor, pump_mode=1, pressure_delta=0.5, flow_delta=0.5)
        context = _context()
        start = _telemetry(step_index=0, pump_mode=2, pressure_bar=4.0, pump_flow_ml_s=2.5)

        inference.start_episode(start, context)
        action = inference.infer_action(start)

        self.assertEqual(action["pump_target_mode"], 1)  # type: ignore[index]
        self.assertAlmostEqual(action["pressure_target_bar"], 4.5)  # type: ignore[index]
        self.assertNotIn("flow_target_ml_s", action)  # type: ignore[operator]
        self.assertIsNone(inference.infer_action(_telemetry(step_index=1, pump_mode=1)))
        with self.assertRaisesRegex(ValueError, "duplicate or out of order"):
            inference.infer_action(_telemetry(step_index=1, pump_mode=1))

    def test_context_includes_old_bags_of_same_bean_and_excludes_other_grinders(self) -> None:
        shots = MemoryShotRepository()
        old_bag = local_shot("old_bag", timestamp=100, relative_steps=1.0)
        old_bag.bean_context_id = "bean_cafe_1"
        old_bag.bean_context_name = "Cafe Barbone"
        current_bag = copy.deepcopy(old_bag)
        current_bag.shot_id = "current_bag"
        current_bag.timestamp = 200
        current_bag.bean_context_id = "bean_cafe_2"
        other_grinder = copy.deepcopy(old_bag)
        other_grinder.shot_id = "other_grinder"
        other_grinder.timestamp = 300
        other_grinder.grinder_context_id = "grinder_other"
        other_bean = copy.deepcopy(old_bag)
        other_bean.shot_id = "other_bean"
        other_bean.timestamp = 400
        other_bean.bean_context_name = "Different bean"
        for shot in (old_bag, current_bag, other_grinder, other_bean):
            shots.upsert(shot)

        service = DreamerLiveContextService(
            shots=shots,
            history_encoder=CanonicalDreamerHistoryEncoder(),
            install_id="install_local",
            machine_id="machine_local",
            fallback_microns_per_step=10.0,
            fallback_dose_g=18.0,
            clock=lambda: 500,
        )
        service.update_machine_state(_machine_state())
        service.update_optimizer_settings(
            OptimizerSettingsEvent(
                install_id="install_local",
                machine_id="MACHINE_LOCAL",
                timestamp=450,
                taste_objective={"mode": "custom", "sweetness": "high"},
            )
        )

        context = service.context_for(_telemetry(step_index=0, machine_id="machine_local"))

        self.assertEqual(len(context.historical_episodes), 2)
        self.assertEqual(
            {episode["group_key"]["bean_context_id"] for episode in context.historical_episodes},
            {"bean_cafe_1", "bean_cafe_2"},
        )
        self.assertEqual(context.taste_objective, {"mode": "custom", "sweetness": "high"})


def _inference() -> tuple[CheckpointDreamerLiveInference, DreamerV3ImaginationActor]:
    world_config = default_world_model_config("espresso_debug")
    world_model = DreamerV3VectorWorldModel(
        observation_dim=5,
        behavior_dim=56,
        static_dim=len(DREAMER_STATIC_CONTEXT_FEATURES),
        config=world_config,
    )
    context_encoder = DreamerContextEncoder(
        static_dim=len(DREAMER_STATIC_CONTEXT_FEATURES),
        terminal_dim=len(DREAMER_TERMINAL_FEATURES),
        time_dim=len(DREAMER_CONTEXT_TIME_FEATURES),
        trajectory_dim=len(DREAMER_CONTEXT_TRAJECTORY_EMBEDDING_FEATURES),
        config=DreamerContextEncoderConfig(hidden_dim=16, context_dim=world_config.deter_dim),
    )
    control_spec = DreamerControlSpec(
        dynamic_control_enabled=True,
        pressure_control_allowed=True,
        flow_control_allowed=True,
        pump_mode_control_allowed=True,
    )
    imagination_config = DreamerV3ImaginationConfig(actor_hidden_dim=16, critic_hidden_dim=16)
    live_action_spec = DreamerLiveActionSpec()
    actor = DreamerV3ImaginationActor(
        feature_dim=world_model.feature_dim,
        taste_objective_dim=9,
        config=imagination_config,
        control_spec=control_spec,
        live_action_spec=live_action_spec,
    )
    critic = DreamerV3ImaginationCritic(
        feature_dim=world_model.feature_dim,
        taste_objective_dim=9,
        config=imagination_config,
    )
    architecture = checkpoint_architecture_from_models(
        world_model=world_model,
        context_encoder=context_encoder,
        actor=actor,
        critic=critic,
        observation_dim=5,
        behavior_dim=56,
        static_dim=len(DREAMER_STATIC_CONTEXT_FEATURES),
        live_action_dim=len(DREAMER_RESOLVED_CONTROL_FEATURES),
        control_spec=control_spec,
        live_action_spec=live_action_spec,
    )
    models = DreamerShadowModels(
        world_model=world_model,
        context_encoder=context_encoder,
        actor=actor,
        critic=critic,
        imagination_config=imagination_config,
        inference_probe_sha256="0" * 64,
    )
    return CheckpointDreamerLiveInference(models=models, architecture=architecture), actor


def _select_live_actions(
    actor: DreamerV3ImaginationActor,
    *,
    pump_mode: int,
    pressure_delta: float,
    flow_delta: float,
) -> None:
    selected = {
        "pump_target_mode": float(pump_mode),
        "pressure_delta_bar": pressure_delta,
        "flow_delta_ml_s": flow_delta,
        "valve_position_delta": 0.0,
        "temperature_delta_c": 0.0,
        "yield_stop_delta_g": 0.0,
        "stop": 0.0,
    }
    with torch.no_grad():
        for index, field_name in enumerate(DREAMER_LIVE_ACTION_FIELDS):
            head = actor.live_heads[index]
            bins = actor.live_action_bins[index, : actor.live_bin_counts_tuple[index]]
            selected_index = int(torch.argmin(torch.abs(bins - selected[field_name])).item())
            head.weight.zero_()
            head.bias.fill_(-100.0)
            head.bias[selected_index] = 100.0


def _context() -> DreamerLiveEpisodeContext:
    return DreamerLiveEpisodeContext(
        install_id="install_local",
        machine_id="machine_local",
        timestamp=500,
        bean_context_id="bean_cafe_2",
        bean_context_name="Cafe Barbone",
        grinder_context_id="grinder_local",
        relative_grind_steps_from_reference=0.0,
        relative_grind_um_from_reference=0.0,
        grind_observed=True,
        dose_g=18.0,
        dose_observed=True,
        initial_target_yield_g=36.0,
        microns_per_step=12.5,
        step_direction="higher_is_finer",
        profile_id=DREAMER_AUTO_PROFILE_ID,
        profile_type="pro",
        profile_phase_count=1,
        taste_objective={"mode": "auto"},
    )


def _machine_state() -> MachineStateEvent:
    return MachineStateEvent(
        install_id="install_local",
        machine_id="machine_local",
        machine_adapter="gaggimate",
        timestamp=450,
        state=MachineState.BREWING,
        bean_context_id="bean_cafe_3",
        bean_context_name="Cafe Barbone",
        grinder_context_id="grinder_local",
        relative_grind_steps_from_reference=2.0,
        microns_per_step=12.5,
        dose_in_g=18.0,
        target_yield_g=36.0,
        profile_id=DREAMER_AUTO_PROFILE_ID,
    )


def _telemetry(
    *,
    step_index: int,
    machine_id: str = "machine_local",
    pump_mode: int = 2,
    pressure_bar: float = 4.0,
    pump_flow_ml_s: float = 2.5,
) -> DreamerLiveTelemetry:
    return DreamerLiveTelemetry(
        machine_id=machine_id,
        shot_id="shot_live",
        profile_id=DREAMER_AUTO_PROFILE_ID,
        step_index=step_index,
        elapsed_ms=step_index * 250,
        pressure_bar=pressure_bar,
        pressure_target_bar=9.0,
        pump_flow_ml_s=pump_flow_ml_s,
        pump_flow_target_ml_s=3.0,
        beverage_flow_g_s=1.5,
        weight_g=5.0,
        temperature_c=92.0,
        temperature_target_c=93.0,
        pump_target_mode=pump_mode,
        valve_open=True,
        target_yield_g=36.0,
        capabilities=DreamerLiveTelemetryCapabilities(
            pressure_control_allowed=True,
            flow_control_allowed=True,
            pump_mode_control_allowed=True,
            valve_control_allowed=False,
            temperature_control_allowed=False,
            stop_control_allowed=False,
        ),
    )


if __name__ == "__main__":
    unittest.main()

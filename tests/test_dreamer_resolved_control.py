from __future__ import annotations

import unittest

import torch

from espresso_rl.domain.dreamer_live_action import DREAMER_LIVE_ACTION_FIELDS
from espresso_rl.domain.dreamer_resolved_control import (
    DREAMER_PUMP_TARGET_MODE_FLOW,
    DREAMER_PUMP_TARGET_MODE_PRESSURE,
    DREAMER_PUMP_TARGET_MODE_SIMPLE,
    DREAMER_RESOLVED_CONTROL_FIELDS,
    DreamerResolvedControl,
    resolve_applied_dreamer_control,
)
from espresso_rl.domain.dreamer_telemetry import (
    DREAMER_AUTO_PROFILE_ID,
    DreamerLiveTelemetry,
    DreamerLiveTelemetryCapabilities,
)
from espresso_rl.dreamer.dataset import build_dreamer_episode_batch, build_dreamer_episodes_from_training_rows
from espresso_rl.dreamer.imagination import DreamerV3ImaginationActor, DreamerV3ImaginationConfig
from espresso_rl.dreamer.live_inference import resolved_control_tensors_from_telemetry
from test_trainer_artifacts import training_row


class DreamerResolvedControlTests(unittest.TestCase):
    def test_pressure_flow_and_simple_modes_have_unambiguous_masks(self) -> None:
        pressure = resolve_applied_dreamer_control(
            pump_target_mode=DREAMER_PUMP_TARGET_MODE_PRESSURE,
            pressure_target_bar=8.0,
            valve_position=1.0,
            temperature_target_c=93.0,
            yield_stop_target_g=36.0,
            stop=False,
        )
        flow = resolve_applied_dreamer_control(
            pump_target_mode=DREAMER_PUMP_TARGET_MODE_FLOW,
            flow_target_ml_s=2.5,
            valve_position=1.0,
            temperature_target_c=92.0,
            yield_stop_target_g=38.0,
            stop=False,
        )
        simple = resolve_applied_dreamer_control(
            pump_target_mode=DREAMER_PUMP_TARGET_MODE_SIMPLE,
            valve_position=0.0,
            yield_stop_target_g=36.0,
            stop=False,
        )

        self.assertEqual(pressure.observed_mask, (1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0))
        self.assertEqual(flow.observed_mask, (1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        self.assertEqual(simple.observed_mask, (1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0))

    def test_rejects_missing_active_target_and_values_beyond_hard_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "pressure_target_bar"):
            resolve_applied_dreamer_control(pump_target_mode=DREAMER_PUMP_TARGET_MODE_PRESSURE)
        with self.assertRaisesRegex(ValueError, "hard bounds"):
            resolve_applied_dreamer_control(
                pump_target_mode=DREAMER_PUMP_TARGET_MODE_PRESSURE,
                pressure_target_bar=12.1,
            )
        with self.assertRaisesRegex(ValueError, "stop must be boolean"):
            resolve_applied_dreamer_control(
                pump_target_mode=DREAMER_PUMP_TARGET_MODE_FLOW,
                flow_target_ml_s=2.5,
                stop=1,
            )

    def test_rejects_conflicting_targets_and_noncanonical_direct_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "pressure target conflicts"):
            resolve_applied_dreamer_control(
                pump_target_mode=DREAMER_PUMP_TARGET_MODE_FLOW,
                pressure_target_bar=8.0,
                flow_target_ml_s=2.5,
            )
        with self.assertRaisesRegex(ValueError, "values must be finite"):
            DreamerResolvedControl(
                values=("invalid", 8.0, 0.0, 1.0, 93.0, 36.0, 0.0),
                observed_mask=(1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0),
            )
        with self.assertRaisesRegex(ValueError, "masked flow_target_ml_s must be zero"):
            DreamerResolvedControl(
                values=(1.0, 8.0, 2.5, 1.0, 93.0, 36.0, 0.0),
                observed_mask=(1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0),
            )

    def test_dataset_live_and_imagination_share_the_same_resolved_control(self) -> None:
        row = training_row(1)
        episodes = build_dreamer_episodes_from_training_rows([row])
        batch = build_dreamer_episode_batch(episodes)
        dataset_values = batch["resolved_controls"][:, 0]
        dataset_mask = batch["resolved_control_mask"][:, 0]
        telemetry = _telemetry()
        live_values, live_mask = resolved_control_tensors_from_telemetry(telemetry)

        self.assertTrue(torch.equal(dataset_values, live_values))
        self.assertTrue(torch.equal(dataset_mask, live_mask))

        actor = DreamerV3ImaginationActor(
            feature_dim=6,
            taste_objective_dim=9,
            config=DreamerV3ImaginationConfig(actor_hidden_dim=8, critic_hidden_dim=8),
        )
        _select_noop_pressure_action(actor)
        imagined = actor.select_dynamic(
            torch.ones((1, 6), dtype=torch.float32),
            torch.tensor([[1.0] + [0.0] * 8], dtype=torch.float32),
            torch.ones((1, len(DREAMER_RESOLVED_CONTROL_FIELDS)), dtype=torch.float32),
            live_values,
            live_mask,
            torch.tensor([[telemetry.pressure_bar, telemetry.pump_flow_ml_s]], dtype=torch.float32),
        )

        self.assertTrue(torch.equal(imagined["resolved_controls"], live_values))
        self.assertTrue(torch.equal(imagined["resolved_control_mask"], live_mask))


def _telemetry() -> DreamerLiveTelemetry:
    return DreamerLiveTelemetry(
        machine_id="machine_1",
        shot_id="shot_1",
        profile_id=DREAMER_AUTO_PROFILE_ID,
        step_index=0,
        elapsed_ms=0,
        pressure_bar=1.0,
        pressure_target_bar=8.0,
        pump_flow_ml_s=1.0,
        pump_flow_target_ml_s=2.4,
        beverage_flow_g_s=0.1,
        weight_g=0.0,
        temperature_c=93.0,
        temperature_target_c=92.5,
        pump_target_mode=DREAMER_PUMP_TARGET_MODE_PRESSURE,
        valve_open=True,
        target_yield_g=36.0,
        capabilities=DreamerLiveTelemetryCapabilities(
            pressure_control_allowed=True,
            flow_control_allowed=True,
            pump_mode_control_allowed=True,
            valve_control_allowed=True,
            temperature_control_allowed=True,
            stop_control_allowed=True,
        ),
    )


def _select_noop_pressure_action(actor: DreamerV3ImaginationActor) -> None:
    selected = {
        "pump_target_mode": 1.0,
        "pressure_delta_bar": 0.0,
        "flow_delta_ml_s": 0.0,
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


if __name__ == "__main__":
    unittest.main()

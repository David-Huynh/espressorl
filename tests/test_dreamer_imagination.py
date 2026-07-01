from __future__ import annotations

import unittest

import torch

from espresso_rl.dreamer.dataset import DREAMER_DYNAMIC_ACTION_FEATURES
from espresso_rl.dreamer.imagination import (
    DreamerV3ImaginationActor,
    DreamerV3ImaginationConfig,
    DreamerV3ImaginationCritic,
    dreamer_v3_imagination_rollout,
    lambda_returns,
    masked_pre_shot_behavior_loss,
    run_dreamer_v3_imagination_preview,
)
from espresso_rl.domain.dreamer_live_action import DREAMER_LIVE_ACTION_FIELDS
from espresso_rl.domain.dreamer_pre_shot import DREAMER_PRE_SHOT_ACTION_FIELDS
from espresso_rl.dreamer.reference_world_model import DreamerV3VectorWorldModel, default_world_model_config
from tests.test_dreamer_world_model_training import smoke_batch


class DreamerImaginationTests(unittest.TestCase):
    def test_actor_masks_unsupported_dynamic_controls(self) -> None:
        torch.manual_seed(3)
        actor = DreamerV3ImaginationActor(
            feature_dim=6,
            taste_objective_dim=9,
            config=DreamerV3ImaginationConfig(actor_hidden_dim=8, critic_hidden_dim=8),
        )
        features = torch.ones((2, 6), dtype=torch.float32)
        target_state = torch.zeros((2, len(DREAMER_DYNAMIC_ACTION_FEATURES)), dtype=torch.float32)
        control_mask = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        output = actor(features, _auto_taste(2), torch.ones((2, 9)), control_mask, target_state)

        self.assertEqual(output["pre_shot_logits"].shape[:2], (2, len(DREAMER_PRE_SHOT_ACTION_FIELDS)))
        self.assertEqual(output["pre_shot_actions"].shape, (2, len(DREAMER_PRE_SHOT_ACTION_FIELDS)))
        self.assertEqual(output["live_action_logits"].shape[:2], (2, len(DREAMER_LIVE_ACTION_FIELDS)))
        self.assertTrue(torch.equal(output["dynamic_action_mask"], control_mask))
        self.assertTrue(torch.equal(output["live_action_mask"], control_mask))
        self.assertEqual(float((output["dynamic_actions"] * (1.0 - control_mask)).abs().max().item()), 0.0)
        self.assertEqual(float((output["live_action_choices"] * (1.0 - output["live_action_mask"])).abs().max().item()), 0.0)

    def test_actor_forward_applies_categorical_live_deltas_and_clamps_target_state(self) -> None:
        actor = DreamerV3ImaginationActor(
            feature_dim=6,
            taste_objective_dim=9,
            config=DreamerV3ImaginationConfig(actor_hidden_dim=8, critic_hidden_dim=8),
        )
        with torch.no_grad():
            for field_name, head, bin_count in zip(
                DREAMER_LIVE_ACTION_FIELDS,
                actor.live_heads,
                actor.live_bin_counts_tuple,
                strict=True,
            ):
                head.weight.zero_()
                head.bias.fill_(-100.0)
                head.bias[0 if field_name == "pump_target_mode" else bin_count - 1] = 100.0

        features = torch.ones((1, 6), dtype=torch.float32)
        control_mask = torch.ones((1, len(DREAMER_DYNAMIC_ACTION_FEATURES)), dtype=torch.float32)
        target_state = torch.tensor([[1.0, 10.0, 18.0, 1.0, 98.0, 88.0, 0.0]], dtype=torch.float32)

        output = actor(features, _auto_taste(1), torch.ones((1, 9)), control_mask, target_state)
        actions = output["dynamic_actions"][0]

        self.assertEqual(actions[DREAMER_DYNAMIC_ACTION_FEATURES.index("pressure_target_bar")].item(), 12.0)
        self.assertEqual(actions[DREAMER_DYNAMIC_ACTION_FEATURES.index("temperature_target_c")].item(), 100.0)
        self.assertEqual(actions[DREAMER_DYNAMIC_ACTION_FEATURES.index("yield_stop_target_g")].item(), 90.0)
        self.assertEqual(actions[DREAMER_DYNAMIC_ACTION_FEATURES.index("stop")].item(), 1.0)
        self.assertGreater(
            output["live_action_choices"][0, DREAMER_LIVE_ACTION_FIELDS.index("pressure_delta_bar")].item(),
            0.0,
        )

    def test_actor_forward_allows_low_temperature_targets_down_to_twenty_c(self) -> None:
        actor = DreamerV3ImaginationActor(
            feature_dim=6,
            taste_objective_dim=9,
            config=DreamerV3ImaginationConfig(actor_hidden_dim=8, critic_hidden_dim=8),
        )
        with torch.no_grad():
            for head in actor.live_heads:
                head.weight.zero_()
                head.bias.fill_(-100.0)
                head.bias[0] = 100.0

        features = torch.ones((1, 6), dtype=torch.float32)
        control_mask = torch.ones((1, len(DREAMER_DYNAMIC_ACTION_FEATURES)), dtype=torch.float32)
        target_state = torch.tensor([[1.0, 1.0, 0.2, 1.0, 22.0, 12.0, 0.0]], dtype=torch.float32)

        output = actor(features, _auto_taste(1), torch.ones((1, 9)), control_mask, target_state)
        actions = output["dynamic_actions"][0]

        self.assertEqual(actions[DREAMER_DYNAMIC_ACTION_FEATURES.index("temperature_target_c")].item(), 20.0)

    def test_actor_mode_switch_anchors_delta_to_current_pump_measurement(self) -> None:
        actor = DreamerV3ImaginationActor(
            feature_dim=6,
            taste_objective_dim=9,
            config=DreamerV3ImaginationConfig(actor_hidden_dim=8, critic_hidden_dim=8),
        )
        with torch.no_grad():
            for field_name, head in zip(DREAMER_LIVE_ACTION_FIELDS, actor.live_heads, strict=True):
                head.weight.zero_()
                head.bias.fill_(-100.0)
                selected = {
                    "pump_target_mode": 0,
                    "pressure_delta_bar": 8,
                    "flow_delta_ml_s": 8,
                    "valve_position_delta": 1,
                    "temperature_delta_c": 5,
                    "yield_stop_delta_g": 5,
                    "stop": 0,
                }[field_name]
                head.bias[selected] = 100.0

        control_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        flow_target_state = torch.tensor([[2.0, 9.0, 3.0, 1.0, 92.0, 36.0, 0.0]])
        pressure = actor.select_dynamic(
            torch.ones((1, 6)),
            _auto_taste(1),
            control_mask,
            flow_target_state,
            torch.tensor([[4.0, 2.5]]),
        )

        self.assertEqual(pressure["dynamic_actions"][0, 0].item(), 1.0)
        self.assertEqual(pressure["dynamic_actions"][0, 1].item(), 4.5)
        self.assertEqual(pressure["dynamic_action_mask"][0, 2].item(), 0.0)

        with torch.no_grad():
            mode_head = actor.live_heads[DREAMER_LIVE_ACTION_FIELDS.index("pump_target_mode")]
            mode_head.bias.fill_(-100.0)
            mode_head.bias[1] = 100.0
        pressure_target_state = torch.tensor([[1.0, 8.0, 7.0, 1.0, 92.0, 36.0, 0.0]])
        flow = actor.select_dynamic(
            torch.ones((1, 6)),
            _auto_taste(1),
            control_mask,
            pressure_target_state,
            torch.tensor([[6.0, 1.5]]),
        )

        self.assertEqual(flow["dynamic_actions"][0, 0].item(), 2.0)
        self.assertEqual(flow["dynamic_actions"][0, 2].item(), 2.0)
        self.assertEqual(flow["dynamic_action_mask"][0, 1].item(), 0.0)

    def test_lambda_returns_are_deterministic(self) -> None:
        rewards = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        values = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32)
        continuations = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        returns = lambda_returns(rewards, values, continuations, discount=0.5, lambda_return=0.5)

        self.assertTrue(torch.allclose(returns, torch.tensor([[6.5, 2.0]], dtype=torch.float32)))

    def test_imagination_preview_shapes_are_deterministic_and_masked(self) -> None:
        config = DreamerV3ImaginationConfig(
            horizon=4,
            actor_hidden_dim=8,
            critic_hidden_dim=8,
            value_bins=41,
        )
        batch = _dynamic_control_batch(batch_size=2)

        torch.manual_seed(9)
        first = run_dreamer_v3_imagination_preview(
            world_model=_world_model_for_batch(batch),
            batch=batch,
            config=config,
        )
        torch.manual_seed(9)
        second = run_dreamer_v3_imagination_preview(
            world_model=_world_model_for_batch(batch),
            batch=batch,
            config=config,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["format"], "espresso_rl_dreamer_v3_imagination_preview_v1")
        self.assertFalse(first["inference_ready"])
        self.assertTrue(first["contract_only"])
        self.assertEqual(first["pre_shot_logits_shape"][:2], [2, 9])
        self.assertEqual(first["pre_shot_action_shape"], [2, 9])
        self.assertEqual(first["pre_shot_held_action_shape"], [2, 4, 9])
        self.assertEqual(first["dynamic_action_shape"], [2, 4, 7])
        self.assertEqual(first["critic_value_logits_shape"], [2, 5, 41])
        self.assertEqual(first["lambda_return_shape"], [2, 4])
        self.assertEqual(first["supported_dynamic_action_count"], 4)
        self.assertEqual(first["unsupported_dynamic_action_abs_max"], 0.0)
        self.assertGreater(first["actor_entropy_mean"], 0.0)

    def test_masked_behavior_loss_only_trains_observed_supported_heads(self) -> None:
        actor = DreamerV3ImaginationActor(
            feature_dim=6,
            taste_objective_dim=9,
            config=DreamerV3ImaginationConfig(actor_hidden_dim=8, critic_hidden_dim=8),
        )
        output = actor.select_pre_shot(
            torch.ones((1, 6), dtype=torch.float32),
            _auto_taste(1),
            torch.ones((1, 9), dtype=torch.float32),
        )
        target_mask = torch.zeros((1, 9), dtype=torch.float32)
        target_mask[:, 0] = 1.0
        loss = masked_pre_shot_behavior_loss(
            output,
            torch.zeros((1, 9), dtype=torch.long),
            target_mask,
            actor.pre_shot_bin_counts_tuple,
        )
        loss.backward()

        self.assertGreater(float(actor.pre_shot_heads[0].weight.grad.abs().sum()), 0.0)
        self.assertEqual(float(actor.pre_shot_heads[1].weight.grad.abs().sum()), 0.0)

        unsupported_output = actor.select_pre_shot(
            torch.ones((1, 6), dtype=torch.float32),
            _auto_taste(1),
            torch.zeros((1, 9), dtype=torch.float32),
        )
        with self.assertRaisesRegex(ValueError, "exceeds capability"):
            masked_pre_shot_behavior_loss(
                unsupported_output,
                torch.zeros((1, 9), dtype=torch.long),
                target_mask,
                actor.pre_shot_bin_counts_tuple,
            )

    def test_taste_objective_conditions_pre_shot_policy_without_changing_context(self) -> None:
        actor = DreamerV3ImaginationActor(
            feature_dim=6,
            taste_objective_dim=9,
            config=DreamerV3ImaginationConfig(actor_hidden_dim=8, critic_hidden_dim=8),
        )
        features = torch.ones((1, 6), dtype=torch.float32)
        capability = torch.ones((1, 9), dtype=torch.float32)
        auto = actor.select_pre_shot(features, _auto_taste(1), capability)["pre_shot_logits"]
        custom_taste = torch.tensor([[0.0, 1.0] + [0.0] * 7], dtype=torch.float32)
        custom = actor.select_pre_shot(features, custom_taste, capability)["pre_shot_logits"]

        self.assertFalse(torch.equal(auto, custom))

    def test_selected_pre_shot_plan_is_held_and_changes_imagined_trajectory(self) -> None:
        batch = _dynamic_control_batch(batch_size=1)
        model = _world_model_for_batch(batch)
        config = DreamerV3ImaginationConfig(horizon=3, actor_hidden_dim=8, critic_hidden_dim=8)
        actor = DreamerV3ImaginationActor(
            feature_dim=model.feature_dim,
            taste_objective_dim=9,
            config=config,
        )
        critic = DreamerV3ImaginationCritic(
            feature_dim=model.feature_dim,
            taste_objective_dim=9,
            config=config,
        )

        def select_bin(index_selector) -> None:
            with torch.no_grad():
                for head, bin_count in zip(
                    actor.pre_shot_heads,
                    actor.pre_shot_bin_counts_tuple,
                    strict=True,
                ):
                    head.weight.zero_()
                    head.bias.fill_(-100.0)
                    head.bias[index_selector(bin_count)] = 100.0

        select_bin(lambda _: 0)
        low = dreamer_v3_imagination_rollout(
            world_model=model,
            batch=batch,
            config=config,
            actor=actor,
            critic=critic,
        )
        select_bin(lambda count: count - 1)
        high = dreamer_v3_imagination_rollout(
            world_model=model,
            batch=batch,
            config=config,
            actor=actor,
            critic=critic,
        )

        self.assertTrue(
            torch.equal(low["pre_shot_actions_held"][:, 0], low["pre_shot_actions_held"][:, -1])
        )
        self.assertFalse(torch.equal(low["pre_shot_actions"], high["pre_shot_actions"]))
        self.assertFalse(torch.equal(low["imagined_features"], high["imagined_features"]))


def _dynamic_control_batch(batch_size: int) -> dict[str, torch.Tensor]:
    batch = smoke_batch(batch_size=batch_size)
    control_mask = torch.zeros_like(batch["control_action_mask"])
    control_mask[:, :, 0] = batch["decision_step_mask"]
    control_mask[:, :, 5] = batch["decision_step_mask"]
    batch["control_action_mask"] = control_mask
    constraints = torch.zeros_like(batch["constraints"])
    constraints[:, :, 0] = 1.0
    constraints[:, :, 5] = 1.0
    batch["constraints"] = constraints
    return batch


def _world_model_for_batch(batch: dict[str, torch.Tensor]) -> DreamerV3VectorWorldModel:
    return DreamerV3VectorWorldModel(
        observation_dim=batch["observations"].shape[-1],
        behavior_dim=66,
        static_dim=batch["static_context"].shape[-1],
        config=default_world_model_config("espresso_debug"),
    )


def _auto_taste(batch_size: int) -> torch.Tensor:
    return torch.tensor([[1.0] + [0.0] * 8] * batch_size, dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()

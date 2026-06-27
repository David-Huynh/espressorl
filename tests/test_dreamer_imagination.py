from __future__ import annotations

import unittest

import torch

from espresso_rl.dreamer.imagination import (
    DREAMER_STATIC_RECIPE_ACTION_HEADS,
    DreamerV3ImaginationActor,
    DreamerV3ImaginationConfig,
    lambda_returns,
    run_dreamer_v3_imagination_preview,
)
from espresso_rl.dreamer.reference_world_model import DreamerV3VectorWorldModel, default_world_model_config
from tests.test_dreamer_world_model_training import smoke_batch


class DreamerImaginationTests(unittest.TestCase):
    def test_actor_masks_unsupported_dynamic_controls(self) -> None:
        torch.manual_seed(3)
        actor = DreamerV3ImaginationActor(
            feature_dim=6,
            dynamic_action_dim=7,
            config=DreamerV3ImaginationConfig(actor_hidden_dim=8, critic_hidden_dim=8),
        )
        features = torch.ones((2, 6), dtype=torch.float32)
        control_mask = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        output = actor(features, control_mask)

        self.assertEqual(output["static_logits"].shape, (2, len(DREAMER_STATIC_RECIPE_ACTION_HEADS), 5))
        self.assertEqual(output["static_actions"].shape, (2, len(DREAMER_STATIC_RECIPE_ACTION_HEADS)))
        self.assertTrue(torch.equal(output["dynamic_action_mask"], control_mask))
        self.assertEqual(float((output["dynamic_actions"] * (1.0 - control_mask)).abs().max().item()), 0.0)

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
        self.assertEqual(first["static_logits_shape"], [2, 4, 3, 5])
        self.assertEqual(first["static_action_shape"], [2, 4, 3])
        self.assertEqual(first["dynamic_action_shape"], [2, 4, 7])
        self.assertEqual(first["critic_value_logits_shape"], [2, 5, 41])
        self.assertEqual(first["lambda_return_shape"], [2, 4])
        self.assertEqual(first["supported_dynamic_action_count"], 4)
        self.assertEqual(first["unsupported_dynamic_action_abs_max"], 0.0)
        self.assertGreater(first["actor_entropy_mean"], 0.0)


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
        behavior_dim=39,
        static_dim=batch["static_context"].shape[-1],
        config=default_world_model_config("espresso_debug"),
    )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import torch

from espresso_rl.dreamer.world_model_smoke import (
    FixedCadenceWorldModelSmokeError,
    run_fixed_cadence_world_model_smoke_train,
)


class DreamerWorldModelSmokeTests(unittest.TestCase):
    def test_smoke_training_is_deterministic_for_same_seed_and_batch(self) -> None:
        first = run_fixed_cadence_world_model_smoke_train(smoke_batch(), seed=7, train_steps=2).to_dict()
        second = run_fixed_cadence_world_model_smoke_train(smoke_batch(), seed=7, train_steps=2).to_dict()

        self.assertEqual(first, second)
        self.assertEqual(first["format"], "espresso_rl_world_model_smoke_v1")
        self.assertEqual(first["device"], "cpu")
        self.assertEqual(first["dtype"], "float32")
        self.assertLess(first["final"]["loss_total"], first["initial"]["loss_total"])

    def test_smoke_training_rejects_invalid_tensor_shapes(self) -> None:
        batch = smoke_batch()
        batch["observations"] = torch.zeros((1, 5), dtype=torch.float32)

        with self.assertRaisesRegex(FixedCadenceWorldModelSmokeError, "observations"):
            run_fixed_cadence_world_model_smoke_train(batch, seed=7, train_steps=2)


def smoke_batch() -> dict[str, torch.Tensor]:
    observations = torch.tensor(
        [
            [
                [1.0, 1.0, 0.8, 0.0, 0.0],
                [2.0, 1.0, 0.9, 0.3, 93.0],
                [3.0, 1.0, 1.0, 0.6, 93.0],
                [4.0, 1.0, 1.1, 1.0, 92.5],
            ]
        ],
        dtype=torch.float32,
    )
    observed_targets = torch.tensor(
        [
            [
                [8.0, 2.0, 93.0, 1.0, 1.0],
                [8.0, 2.0, 93.0, 1.0, 1.0],
                [8.0, 2.0, 93.0, 1.0, 1.0],
                [8.0, 2.0, 93.0, 1.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    return {
        "observations": observations,
        "observed_profile_targets": observed_targets,
        "observed_profile_target_mask": torch.ones((1, 4, 5), dtype=torch.float32),
        "dynamic_actions": torch.zeros((1, 4, 7), dtype=torch.float32),
        "dynamic_action_mask": torch.zeros((1, 4, 7), dtype=torch.float32),
        "control_action_mask": torch.zeros((1, 4, 7), dtype=torch.float32),
        "constraints": torch.zeros((1, 4, 7), dtype=torch.float32),
        "decision_step_mask": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "rewards": torch.tensor([[0.0, 0.0, 0.0, 0.8]], dtype=torch.float32),
        "continuations": torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
        "step_mask": torch.ones((1, 4), dtype=torch.float32),
        "static_context": torch.zeros((1, 18), dtype=torch.float32),
    }


if __name__ == "__main__":
    unittest.main()

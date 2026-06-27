from __future__ import annotations

import unittest

import torch

from espresso_rl.dreamer.world_model_training import (
    FixedCadenceWorldModelTrainingError,
    WorldModelTrainPreviewConfig,
    run_fixed_cadence_world_model_train_preview,
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

        with self.assertRaisesRegex(FixedCadenceWorldModelTrainingError, "observations"):
            run_fixed_cadence_world_model_smoke_train(batch, seed=7, train_steps=2)

    def test_train_preview_is_deterministic_and_reports_loss_curves(self) -> None:
        config = WorldModelTrainPreviewConfig(
            seed=11,
            epochs=2,
            batch_size=1,
            learning_rate=0.001,
            hidden_dim=8,
            latent_dim=4,
            validation_split=0.25,
            early_stop_patience=2,
        )
        split = {
            "strategy": "test",
            "train_source_training_row_ids": [1, 2],
            "validation_source_training_row_ids": [3],
            "train_episode_count": 2,
            "validation_episode_count": 1,
            "validation_split": 0.25,
        }

        first = run_fixed_cadence_world_model_train_preview(
            train_batch=smoke_batch(batch_size=2),
            validation_batch=smoke_batch(batch_size=1),
            config=config,
            dataset_split=split,
        ).to_dict()
        second = run_fixed_cadence_world_model_train_preview(
            train_batch=smoke_batch(batch_size=2),
            validation_batch=smoke_batch(batch_size=1),
            config=config,
            dataset_split=split,
        ).to_dict()

        self.assertEqual(first, second)
        self.assertEqual(first["format"], "espresso_rl_world_model_train_preview_v1")
        self.assertEqual(first["epochs_completed"], 2)
        self.assertEqual(len(first["train_loss_curve"]), 2)
        self.assertEqual(len(first["validation_loss_curve"]), 2)
        self.assertEqual(first["dataset_split"]["validation_source_training_row_ids"], [3])
        self.assertEqual(len(first["dataset_split_sha256"]), 64)


def smoke_batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
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
    batch = {
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
    if batch_size == 1:
        return batch
    return {key: value.repeat(batch_size, *([1] * (value.ndim - 1))) for key, value in batch.items()}


if __name__ == "__main__":
    unittest.main()

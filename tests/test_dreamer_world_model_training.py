from __future__ import annotations

import unittest

import torch

from espresso_rl.domain.dreamer_control import DEFAULT_DREAMER_CONTROL_SPEC
from espresso_rl.dreamer.reference_world_model import default_world_model_config
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
        self.assertEqual(first["model_config"]["model_preset"], "espresso_debug")
        self.assertIn("loss_dyn", first["final"])
        self.assertIn("loss_rep", first["final"])
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
            gradient_steps_per_epoch=1,
            model=default_world_model_config("espresso_debug"),
            validation_split=0.25,
            early_stop_patience=2,
            control_spec=DEFAULT_DREAMER_CONTROL_SPEC,
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
        self.assertEqual(first["model_config"]["stoch_size"], 4)
        self.assertEqual(first["model_config"]["class_size"], 4)
        self.assertEqual(first["epochs_completed"], 2)
        self.assertEqual(len(first["train_loss_curve"]), 2)
        self.assertEqual(len(first["validation_loss_curve"]), 2)
        self.assertEqual(first["actor_critic_train_steps"], 3)
        self.assertEqual(first["actor_learning_rate"], 0.0003)
        self.assertEqual(first["critic_learning_rate"], 0.0003)
        self.assertEqual(len(first["actor_critic_train_curve"]), 3)
        self.assertIn("actor_loss", first["actor_critic_train_curve"][0])
        self.assertIn("critic_loss", first["actor_critic_train_curve"][0])
        self.assertIn("imagined_return_mean", first["actor_critic_train_curve"][0])
        evaluation = first["evaluation_report"]
        self.assertEqual(evaluation["format"], "espresso_rl_dreamer_v3_offline_evaluation_report_v1")
        self.assertFalse(evaluation["inference_ready"])
        self.assertIn("loss_total", evaluation["world_model_validation"])
        self.assertIn("rmse", evaluation["reward_prediction"])
        self.assertIn("rmse", evaluation["continuation_prediction"])
        self.assertIn("rmse", evaluation["critic_value"])
        self.assertIn("imagined_return_mean", evaluation["actor"])
        self.assertIn("evaluation_passed", evaluation["gates"])
        self.assertIn("loss_dyn", first["train_loss_curve"][0])
        self.assertIn("loss_rep", first["validation_loss_curve"][0])
        self.assertEqual(first["dataset_split"]["validation_source_training_row_ids"], [3])
        self.assertEqual(len(first["dataset_split_sha256"]), 64)
        preview = first["imagination_preview"]
        self.assertFalse(preview["inference_ready"])
        self.assertTrue(preview["contract_only"])
        self.assertEqual(preview["dynamic_action_shape"], [1, 3, 7])
        self.assertEqual(preview["lambda_return_shape"], [1, 3])

    def test_actor_critic_training_respects_dynamic_control_masks(self) -> None:
        config = WorldModelTrainPreviewConfig(
            seed=13,
            epochs=1,
            batch_size=1,
            learning_rate=0.001,
            gradient_steps_per_epoch=1,
            model=default_world_model_config("espresso_debug"),
            validation_split=0.25,
            early_stop_patience=2,
            control_spec=DEFAULT_DREAMER_CONTROL_SPEC,
            actor_critic_train_steps=2,
            imagination_batch_size=2,
        )
        split = {
            "strategy": "test",
            "train_source_training_row_ids": [1, 2],
            "validation_source_training_row_ids": [3],
            "train_episode_count": 2,
            "validation_episode_count": 1,
            "validation_split": 0.25,
        }

        result = run_fixed_cadence_world_model_train_preview(
            train_batch=dynamic_control_batch(batch_size=2),
            validation_batch=dynamic_control_batch(batch_size=1),
            config=config,
            dataset_split=split,
        ).to_dict()

        curve = result["actor_critic_train_curve"]
        self.assertEqual(len(curve), 2)
        self.assertEqual(curve[0]["supported_dynamic_action_count"], 4.0)
        self.assertEqual(curve[0]["unsupported_dynamic_action_abs_max"], 0.0)
        self.assertGreater(curve[0]["actor_entropy_mean"], 0.0)
        evaluation = result["evaluation_report"]
        self.assertTrue(evaluation["gates"]["action_mask_ok"])
        self.assertEqual(evaluation["actor"]["unsupported_dynamic_action_abs_max"], 0.0)


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
        "pre_shot_actions": torch.zeros((1, 9), dtype=torch.float32),
        "pre_shot_action_indexes": torch.zeros((1, 9), dtype=torch.long),
        "pre_shot_action_mask": torch.ones((1, 9), dtype=torch.float32),
        "pre_shot_capability_mask": torch.ones((1, 9), dtype=torch.float32),
        "taste_objective": torch.tensor([[1.0] + [0.0] * 8], dtype=torch.float32),
        "decision_step_mask": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "rewards": torch.tensor([[0.0, 0.0, 0.0, 0.8]], dtype=torch.float32),
        "continuations": torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
        "step_mask": torch.ones((1, 4), dtype=torch.float32),
        "static_context": torch.zeros((1, 18), dtype=torch.float32),
        "context_static": torch.zeros((1, 16, 18), dtype=torch.float32),
        "context_terminal": torch.zeros((1, 16, 18), dtype=torch.float32),
        "context_time": torch.zeros((1, 16, 1), dtype=torch.float32),
        "context_trajectory_embedding": torch.zeros((1, 16, 77), dtype=torch.float32),
        "context_mask": torch.tensor([[1.0] + [0.0] * 15], dtype=torch.float32),
    }
    if batch_size == 1:
        return batch
    return {key: value.repeat(batch_size, *([1] * (value.ndim - 1))) for key, value in batch.items()}


def dynamic_control_batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
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


if __name__ == "__main__":
    unittest.main()

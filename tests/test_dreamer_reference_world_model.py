from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from espresso_rl.dreamer.reference_world_model import (
    DreamerV3VectorWorldModel,
    categorical_kl_logits,
    default_world_model_config,
    straight_through_onehot,
    unimix_logits,
)


class DreamerReferenceWorldModelTests(unittest.TestCase):
    def test_straight_through_onehot_is_onehot_and_keeps_gradients(self) -> None:
        logits = torch.tensor(
            [[[[3.0, 1.0, -2.0], [0.0, 2.0, -1.0]]]],
            dtype=torch.float32,
            requires_grad=True,
        )

        sample = straight_through_onehot(logits, sample=False)
        loss = (sample * torch.tensor([[[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]]])).sum()
        loss.backward()

        self.assertEqual(tuple(sample.shape), (1, 1, 2, 3))
        self.assertEqual(sample.sum(dim=-1).tolist(), [[[1.0, 1.0]]])
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum().item()), 0.0)

    def test_unimix_prevents_zero_probability_collapse(self) -> None:
        logits = torch.tensor([[[1000.0, -1000.0, -1000.0, -1000.0]]], dtype=torch.float32)

        probs = F.softmax(unimix_logits(logits, 0.04), dim=-1)

        self.assertGreaterEqual(float(probs.min().item()), 0.009)
        self.assertAlmostEqual(float(probs.sum(dim=-1).item()), 1.0, places=6)

    def test_dyn_and_rep_kl_have_reference_stop_gradient_directions(self) -> None:
        posterior = torch.randn((2, 3, 4), dtype=torch.float32, requires_grad=True)
        prior = torch.randn((2, 3, 4), dtype=torch.float32, requires_grad=True)

        dyn = categorical_kl_logits(posterior.detach(), prior, unimix=0.01).sum()
        dyn.backward()
        self.assertIsNone(posterior.grad)
        self.assertIsNotNone(prior.grad)
        self.assertGreater(float(prior.grad.abs().sum().item()), 0.0)

        posterior = torch.randn((2, 3, 4), dtype=torch.float32, requires_grad=True)
        prior = torch.randn((2, 3, 4), dtype=torch.float32, requires_grad=True)
        rep = categorical_kl_logits(posterior, prior.detach(), unimix=0.01).sum()
        rep.backward()
        self.assertIsNotNone(posterior.grad)
        self.assertGreater(float(posterior.grad.abs().sum().item()), 0.0)
        self.assertIsNone(prior.grad)

    def test_reset_mask_clears_recurrent_state(self) -> None:
        torch.manual_seed(7)
        config = default_world_model_config("espresso_debug")
        model = DreamerV3VectorWorldModel(
            observation_dim=5,
            behavior_dim=39,
            static_dim=18,
            config=config,
        )
        batch = reference_batch(step_count=2)
        single = {key: value[:, 1:2].clone() for key, value in batch.items() if isinstance(value, torch.Tensor)}
        single["static_context"] = batch["static_context"].clone()

        two_step = model.observe(
            batch,
            is_first=torch.tensor([[True, True]], dtype=torch.bool),
            sample=False,
        )
        one_step = model.observe(
            single,
            is_first=torch.tensor([[True]], dtype=torch.bool),
            sample=False,
        )

        self.assertTrue(torch.allclose(two_step["features"][:, 1], one_step["features"][:, 0], atol=1e-6))

    def test_reference_world_model_losses_include_dyn_and_rep_terms(self) -> None:
        torch.manual_seed(7)
        model = DreamerV3VectorWorldModel(
            observation_dim=5,
            behavior_dim=39,
            static_dim=18,
            config=default_world_model_config("espresso_debug"),
        )

        losses = model.losses(reference_batch(step_count=4), sample=False)

        self.assertEqual(
            set(losses),
            {"loss_total", "loss_observation", "loss_reward", "loss_continuation", "loss_dyn", "loss_rep"},
        )
        self.assertTrue(torch.isfinite(losses["loss_total"]))


def reference_batch(step_count: int) -> dict[str, torch.Tensor]:
    observations = torch.tensor(
        [[[float(index + 1), 1.0, 0.5 + index * 0.1, index * 0.3, 93.0] for index in range(step_count)]],
        dtype=torch.float32,
    )
    return {
        "observations": observations,
        "observed_profile_targets": torch.ones((1, step_count, 5), dtype=torch.float32),
        "observed_profile_target_mask": torch.ones((1, step_count, 5), dtype=torch.float32),
        "dynamic_actions": torch.zeros((1, step_count, 7), dtype=torch.float32),
        "dynamic_action_mask": torch.zeros((1, step_count, 7), dtype=torch.float32),
        "control_action_mask": torch.zeros((1, step_count, 7), dtype=torch.float32),
        "constraints": torch.zeros((1, step_count, 7), dtype=torch.float32),
        "decision_step_mask": torch.tensor([[1.0] + [0.0] * (step_count - 1)], dtype=torch.float32),
        "rewards": torch.tensor([[0.0] * (step_count - 1) + [0.8]], dtype=torch.float32),
        "continuations": torch.tensor([[1.0] * (step_count - 1) + [0.0]], dtype=torch.float32),
        "step_mask": torch.ones((1, step_count), dtype=torch.float32),
        "static_context": torch.zeros((1, 18), dtype=torch.float32),
    }


if __name__ == "__main__":
    unittest.main()

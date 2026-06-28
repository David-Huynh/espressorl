from __future__ import annotations

import unittest

import torch

from espresso_rl.dreamer.context_encoder import DreamerContextEncoder, DreamerContextEncoderConfig


class DreamerContextEncoderTests(unittest.TestCase):
    def test_empty_context_encodes_to_zero(self) -> None:
        torch.manual_seed(5)
        encoder = context_encoder()
        batch = context_batch(mask=torch.zeros((2, 4), dtype=torch.float32))

        encoded = encoder(batch)

        self.assertEqual(tuple(encoded.shape), (2, 8))
        self.assertTrue(torch.equal(encoded, torch.zeros_like(encoded)))

    def test_masked_context_rows_do_not_affect_output(self) -> None:
        torch.manual_seed(7)
        encoder = context_encoder()
        base = context_batch(mask=torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32))
        changed = {key: value.clone() for key, value in base.items()}
        changed["context_static"][:, 2:] = 999.0
        changed["context_terminal"][:, 2:] = -999.0
        changed["context_time"][:, 2:] = 100.0
        changed["context_trajectory_embedding"][:, 2:] = 500.0

        first = encoder(base)
        second = encoder(changed)

        self.assertTrue(torch.allclose(first, second, atol=1e-6))
        self.assertGreater(float(first.abs().sum().item()), 0.0)


def context_encoder() -> DreamerContextEncoder:
    return DreamerContextEncoder(
        static_dim=3,
        terminal_dim=2,
        time_dim=1,
        trajectory_dim=5,
        config=DreamerContextEncoderConfig(hidden_dim=8, context_dim=8),
    )


def context_batch(*, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size, window = mask.shape
    return {
        "context_static": sequence(batch_size, window, 3, scale=0.1),
        "context_terminal": sequence(batch_size, window, 2, scale=0.2),
        "context_time": sequence(batch_size, window, 1, scale=0.3),
        "context_trajectory_embedding": sequence(batch_size, window, 5, scale=0.05),
        "context_mask": mask,
    }


def sequence(batch_size: int, window: int, features: int, *, scale: float) -> torch.Tensor:
    values = torch.arange(1, batch_size * window * features + 1, dtype=torch.float32) * scale
    return values.reshape(batch_size, window, features)


if __name__ == "__main__":
    unittest.main()

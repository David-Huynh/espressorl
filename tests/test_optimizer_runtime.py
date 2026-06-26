from __future__ import annotations

import unittest

from espresso_rl.domain.optimization import DEFAULT_OPTIMIZER_MODE, OPTIMIZER_MODE_DREAMER_V3_SHADOW
from espresso_rl.optimizers.runtime import RuntimeOptimizer


class RuntimeOptimizerTests(unittest.TestCase):
    def test_dreamer_mode_without_model_files_is_not_configured(self) -> None:
        optimizer = RuntimeOptimizer(optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_SHADOW)

        status = optimizer.status()

        self.assertEqual(status.configured_mode, DEFAULT_OPTIMIZER_MODE)
        self.assertEqual(status.effective_mode, DEFAULT_OPTIMIZER_MODE)
        self.assertFalse(status.dreamer_v3_available)

    def test_dreamer_mode_with_model_files_falls_back_to_bo_until_inference_exists(self) -> None:
        optimizer = RuntimeOptimizer(
            optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_SHADOW,
            model_artifact_path="models/dreamer.pt",
            model_artifact_sha256="a" * 64,
        )

        status = optimizer.status()

        self.assertEqual(status.configured_mode, OPTIMIZER_MODE_DREAMER_V3_SHADOW)
        self.assertEqual(status.effective_mode, DEFAULT_OPTIMIZER_MODE)
        self.assertTrue(status.dreamer_v3_available)
        self.assertIn("Bayesian Optimization", status.fallback_reason or "")

    def test_optimizer_settings_preserve_configured_model_metadata(self) -> None:
        optimizer = RuntimeOptimizer(
            optimizer_mode=DEFAULT_OPTIMIZER_MODE,
            model_artifact_path="models/dreamer.pt",
            model_artifact_sha256="a" * 64,
        )

        status = optimizer.configure(optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_SHADOW)

        self.assertEqual(status.model_artifact_path, "models/dreamer.pt")
        self.assertEqual(status.model_artifact_sha256, "a" * 64)
        self.assertEqual(status.configured_mode, OPTIMIZER_MODE_DREAMER_V3_SHADOW)

    def test_aliases_normalize_to_bayesian_optimization(self) -> None:
        optimizer = RuntimeOptimizer(optimizer_mode="bo")

        self.assertEqual(optimizer.status().configured_mode, DEFAULT_OPTIMIZER_MODE)

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "optimizer_mode"):
            RuntimeOptimizer(optimizer_mode="shell")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import espresso_rl.config as config_module
from espresso_rl.domain.optimization import DEFAULT_OPTIMIZER_MODE, OPTIMIZER_MODE_DREAMER_V3_SHADOW
from espresso_rl.optimizers.runtime import RuntimeOptimizer, verify_model_artifact


class RuntimeOptimizerTests(unittest.TestCase):
    def test_dreamer_mode_without_model_files_is_not_configured(self) -> None:
        optimizer = RuntimeOptimizer(optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_SHADOW)

        status = optimizer.status()

        self.assertEqual(status.configured_mode, DEFAULT_OPTIMIZER_MODE)
        self.assertEqual(status.effective_mode, DEFAULT_OPTIMIZER_MODE)
        self.assertFalse(status.dreamer_v3_available)
        self.assertEqual(status.available_modes, (DEFAULT_OPTIMIZER_MODE,))
        self.assertIn(OPTIMIZER_MODE_DREAMER_V3_SHADOW, status.unavailable_modes or {})

    def test_dreamer_mode_with_model_files_falls_back_to_bo_until_inference_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "dreamer.pt"
            model_path.write_bytes(b"verified model")
            digest = hashlib.sha256(b"verified model").hexdigest()

            optimizer = RuntimeOptimizer(
                optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_SHADOW,
                model_artifact_path=str(model_path),
                model_artifact_sha256=digest,
            )

            status = optimizer.status()

        self.assertEqual(status.configured_mode, OPTIMIZER_MODE_DREAMER_V3_SHADOW)
        self.assertEqual(status.effective_mode, DEFAULT_OPTIMIZER_MODE)
        self.assertTrue(status.dreamer_v3_available)
        self.assertTrue(status.model_artifact_verified)
        self.assertEqual(status.model_artifact_actual_sha256, digest)
        self.assertIn(OPTIMIZER_MODE_DREAMER_V3_SHADOW, status.available_modes)
        self.assertIn("Bayesian Optimization", status.fallback_reason or "")

    def test_optimizer_settings_preserve_configured_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "dreamer.pt"
            model_path.write_bytes(b"verified model")
            digest = hashlib.sha256(b"verified model").hexdigest()

            optimizer = RuntimeOptimizer(
                optimizer_mode=DEFAULT_OPTIMIZER_MODE,
                model_artifact_path=str(model_path),
                model_artifact_sha256=digest,
            )

            status = optimizer.configure(optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_SHADOW)

        self.assertEqual(status.model_artifact_path, str(model_path))
        self.assertEqual(status.model_artifact_sha256, digest)
        self.assertEqual(status.configured_mode, OPTIMIZER_MODE_DREAMER_V3_SHADOW)

    def test_model_artifact_hash_mismatch_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "dreamer.pt"
            model_path.write_bytes(b"tampered model")

            status = verify_model_artifact(str(model_path), "a" * 64, max_bytes=1024)

        self.assertFalse(status.verified)
        self.assertIn("does not match", status.unavailable_reason or "")

    def test_model_artifact_size_limit_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "dreamer.pt"
            model_path.write_bytes(b"too large")
            digest = hashlib.sha256(b"too large").hexdigest()

            status = verify_model_artifact(str(model_path), digest, max_bytes=2)

        self.assertFalse(status.verified)
        self.assertIn("larger than", status.unavailable_reason or "")

    def test_model_artifact_max_bytes_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            RuntimeOptimizer(model_artifact_max_bytes=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            verify_model_artifact("models/dreamer.pt", "a" * 64, max_bytes=0)

    def test_config_uses_release_default_model_sha_and_path(self) -> None:
        release_digest = hashlib.sha256(b"release model").hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "espresso_rl"
            default_path = data_dir / "models" / "dreamer_v3.pt"
            with (
                patch.object(config_module, "_OPTIONS_PATH", root / "options.json"),
                patch.object(config_module, "_DATA_DIR", data_dir),
                patch.object(config_module, "_DEFAULT_DREAMER_V3_MODEL_ARTIFACT_PATH", default_path),
                patch.object(config_module, "_RELEASE_DEFAULT_MODEL_ARTIFACT_SHA256", release_digest),
            ):
                config = config_module.Config.load()

        self.assertEqual(config.optimizer_model_artifact_sha256, release_digest)
        self.assertEqual(config.default_optimizer_model_artifact_sha256, release_digest)
        self.assertEqual(config.optimizer_model_artifact_path, str(default_path))

    def test_config_model_artifact_sha_override_wins_over_release_default(self) -> None:
        release_digest = hashlib.sha256(b"release model").hexdigest()
        override_digest = hashlib.sha256(b"trainer model").hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "espresso_rl"
            options_path = root / "options.json"
            options_path.write_text(
                json.dumps(
                    {
                        "optimizer_model_artifact_path": "/models/trainer.pt",
                        "optimizer_model_artifact_sha256": override_digest,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(config_module, "_OPTIONS_PATH", options_path),
                patch.object(config_module, "_DATA_DIR", data_dir),
                patch.object(config_module, "_RELEASE_DEFAULT_MODEL_ARTIFACT_SHA256", release_digest),
            ):
                config = config_module.Config.load()

        self.assertEqual(config.optimizer_model_artifact_path, "/models/trainer.pt")
        self.assertEqual(config.optimizer_model_artifact_sha256, override_digest)
        self.assertEqual(config.default_optimizer_model_artifact_sha256, release_digest)

    def test_aliases_normalize_to_bayesian_optimization(self) -> None:
        optimizer = RuntimeOptimizer(optimizer_mode="bo")

        self.assertEqual(optimizer.status().configured_mode, DEFAULT_OPTIMIZER_MODE)

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "optimizer_mode"):
            RuntimeOptimizer(optimizer_mode="shell")


if __name__ == "__main__":
    unittest.main()

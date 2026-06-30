from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import espresso_rl.config as config_module
from espresso_rl.domain.optimization import (
    DEFAULT_OPTIMIZER_MODE,
    OPTIMIZER_MODE_DREAMER_V3_ACTIVE,
    OPTIMIZER_MODE_DREAMER_V3_SHADOW,
)
from espresso_rl.optimizers.runtime import RuntimeOptimizer, verify_model_artifact, verify_model_manifest_file


class RecordingOptimizer:
    def __init__(self, result) -> None:
        self.result = result
        self.contexts = []

    def recommend(self, context):
        self.contexts.append(context)
        return self.result


class FailingOptimizer:
    def recommend(self, context):
        raise ValueError("unsafe dreamer proposal")


class FakeCheckpoint:
    def __init__(
        self,
        *,
        artifact_reference: str,
        manifest_reference: str,
        artifact_sha256: str,
        manifest_sha256: str,
        inference_ready: bool,
    ) -> None:
        self.artifact_reference = artifact_reference
        self.manifest_reference = manifest_reference
        self.artifact_sha256 = artifact_sha256
        self.manifest_sha256 = manifest_sha256
        self.inference_ready = inference_ready
        self.tensors = (object(),)
        self.component_names = ("actor", "context_encoder", "critic", "world_model")
        self.architecture_sha256 = "f" * 64
        self.inference_probe_sha256 = "9" * 64
        self.heldout_inference_sha256 = "8" * 64


class RuntimeOptimizerTests(unittest.TestCase):
    def test_dreamer_mode_without_model_files_is_not_configured(self) -> None:
        optimizer = RuntimeOptimizer(optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_SHADOW)

        status = optimizer.status()

        self.assertEqual(status.configured_mode, OPTIMIZER_MODE_DREAMER_V3_SHADOW)
        self.assertEqual(status.effective_mode, DEFAULT_OPTIMIZER_MODE)
        self.assertFalse(status.dreamer_v3_available)
        self.assertEqual(status.available_modes, (DEFAULT_OPTIMIZER_MODE,))
        self.assertIn(OPTIMIZER_MODE_DREAMER_V3_SHADOW, status.unavailable_modes or {})
        self.assertIn(OPTIMIZER_MODE_DREAMER_V3_ACTIVE, status.unavailable_modes or {})

    def test_dreamer_mode_with_model_file_but_no_manifest_is_not_configured(self) -> None:
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
        self.assertTrue(status.model_artifact_verified)
        self.assertFalse(status.model_manifest_verified)
        self.assertFalse(status.dreamer_v3_available)

    def test_dreamer_mode_with_manifest_only_falls_back_to_bo_until_checkpoint_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "dreamer.pt"
            model_path.write_bytes(b"verified model")
            digest = hashlib.sha256(b"verified model").hexdigest()
            manifest_path = write_manifest(Path(tmp), digest)

            optimizer = RuntimeOptimizer(
                optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_SHADOW,
                model_artifact_path=str(model_path),
                model_artifact_sha256=digest,
                model_manifest_path=str(manifest_path),
            )

            status = optimizer.status()

        self.assertEqual(status.configured_mode, OPTIMIZER_MODE_DREAMER_V3_SHADOW)
        self.assertEqual(status.effective_mode, DEFAULT_OPTIMIZER_MODE)
        self.assertFalse(status.dreamer_v3_available)
        self.assertTrue(status.model_artifact_verified)
        self.assertTrue(status.model_manifest_verified)
        self.assertFalse(status.checkpoint_verified)
        self.assertEqual(status.model_artifact_actual_sha256, digest)
        self.assertEqual(status.model_manifest_dataset_sha256, "b" * 64)
        self.assertEqual(status.model_manifest_trainer_git_sha, "trainerabc")
        self.assertEqual(status.model_manifest_artifact_format, "safetensors")
        self.assertNotIn(OPTIMIZER_MODE_DREAMER_V3_SHADOW, status.available_modes)
        self.assertIn("tensor verification", status.checkpoint_unavailable_reason or "")

    def test_optimizer_settings_preserve_configured_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "dreamer.pt"
            model_path.write_bytes(b"verified model")
            digest = hashlib.sha256(b"verified model").hexdigest()
            manifest_path = write_manifest(Path(tmp), digest)

            optimizer = RuntimeOptimizer(
                optimizer_mode=DEFAULT_OPTIMIZER_MODE,
                model_artifact_path=str(model_path),
                model_artifact_sha256=digest,
                model_manifest_path=str(manifest_path),
            )

            status = optimizer.configure(optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_SHADOW)

        self.assertEqual(status.model_artifact_path, str(model_path))
        self.assertEqual(status.model_artifact_sha256, digest)
        self.assertEqual(status.model_manifest_path, str(manifest_path))
        self.assertEqual(status.configured_mode, OPTIMIZER_MODE_DREAMER_V3_SHADOW)
        self.assertTrue(status.model_manifest_verified)
        self.assertFalse(status.checkpoint_verified)

    def test_verified_shadow_checkpoint_keeps_bo_effective_and_lists_shadow_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path, digest, manifest_path, manifest_sha = write_model_bundle(Path(tmp), inference_ready=False)
            checkpoint = FakeCheckpoint(
                artifact_reference=str(model_path),
                manifest_reference=str(manifest_path),
                artifact_sha256=digest,
                manifest_sha256=manifest_sha,
                inference_ready=False,
            )
            bo = RecordingOptimizer("bo")
            optimizer = RuntimeOptimizer(
                optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_SHADOW,
                model_artifact_path=str(model_path),
                model_artifact_sha256=digest,
                model_manifest_path=str(manifest_path),
                verified_checkpoint=checkpoint,
                checkpoint_inference_parity_verified=True,
                bo_optimizer=bo,
            )

            status = optimizer.status()

        self.assertEqual(status.configured_mode, OPTIMIZER_MODE_DREAMER_V3_SHADOW)
        self.assertEqual(status.effective_mode, DEFAULT_OPTIMIZER_MODE)
        self.assertTrue(status.dreamer_v3_shadow_available)
        self.assertFalse(status.dreamer_v3_active_available)
        self.assertIn(OPTIMIZER_MODE_DREAMER_V3_SHADOW, status.available_modes)
        self.assertNotIn(OPTIMIZER_MODE_DREAMER_V3_ACTIVE, status.available_modes)
        context = object()
        self.assertEqual(optimizer.recommend(context), "bo")
        self.assertEqual(bo.contexts, [context])

    def test_active_mode_uses_dreamer_only_for_inference_ready_release_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path, digest, manifest_path, manifest_sha = write_model_bundle(
                Path(tmp),
                inference_ready=True,
                optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_ACTIVE,
            )
            checkpoint = FakeCheckpoint(
                artifact_reference=str(model_path),
                manifest_reference=str(manifest_path),
                artifact_sha256=digest,
                manifest_sha256=manifest_sha,
                inference_ready=True,
            )
            bo = RecordingOptimizer("bo")
            dreamer = RecordingOptimizer("dreamer")
            optimizer = RuntimeOptimizer(
                optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_ACTIVE,
                model_artifact_path=str(model_path),
                model_artifact_sha256=digest,
                model_manifest_path=str(manifest_path),
                verified_checkpoint=checkpoint,
                checkpoint_inference_parity_verified=True,
                bo_optimizer=bo,
                dreamer_optimizer=dreamer,
            )

            status = optimizer.status()

        self.assertEqual(status.configured_mode, OPTIMIZER_MODE_DREAMER_V3_ACTIVE)
        self.assertEqual(status.effective_mode, OPTIMIZER_MODE_DREAMER_V3_ACTIVE)
        self.assertTrue(status.dreamer_v3_available)
        self.assertTrue(status.dreamer_v3_shadow_available)
        self.assertTrue(status.dreamer_v3_active_available)
        self.assertEqual(
            status.available_modes,
            (
                DEFAULT_OPTIMIZER_MODE,
                OPTIMIZER_MODE_DREAMER_V3_SHADOW,
                OPTIMIZER_MODE_DREAMER_V3_ACTIVE,
            ),
        )
        context = object()
        self.assertEqual(optimizer.recommend(context), "dreamer")
        self.assertEqual(dreamer.contexts, [context])
        self.assertEqual(bo.contexts, [])

    def test_active_mode_falls_back_to_bo_without_dreamer_optimizer_or_on_safety_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path, digest, manifest_path, manifest_sha = write_model_bundle(
                Path(tmp),
                inference_ready=True,
                optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_ACTIVE,
            )
            checkpoint = FakeCheckpoint(
                artifact_reference=str(model_path),
                manifest_reference=str(manifest_path),
                artifact_sha256=digest,
                manifest_sha256=manifest_sha,
                inference_ready=True,
            )
            bo = RecordingOptimizer("bo")
            optimizer = RuntimeOptimizer(
                optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_ACTIVE,
                model_artifact_path=str(model_path),
                model_artifact_sha256=digest,
                model_manifest_path=str(manifest_path),
                verified_checkpoint=checkpoint,
                checkpoint_inference_parity_verified=True,
                bo_optimizer=bo,
            )

            unavailable = optimizer.status()

            bo_on_error = RecordingOptimizer("bo_after_error")
            fallback = RuntimeOptimizer(
                optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_ACTIVE,
                model_artifact_path=str(model_path),
                model_artifact_sha256=digest,
                model_manifest_path=str(manifest_path),
                verified_checkpoint=checkpoint,
                checkpoint_inference_parity_verified=True,
                bo_optimizer=bo_on_error,
                dreamer_optimizer=FailingOptimizer(),
            )

        self.assertEqual(unavailable.configured_mode, OPTIMIZER_MODE_DREAMER_V3_ACTIVE)
        self.assertEqual(unavailable.effective_mode, DEFAULT_OPTIMIZER_MODE)
        self.assertFalse(unavailable.dreamer_v3_active_available)
        self.assertIn(OPTIMIZER_MODE_DREAMER_V3_ACTIVE, unavailable.unavailable_modes or {})
        context = object()
        self.assertEqual(optimizer.recommend(context), "bo")
        self.assertEqual(fallback.recommend(context), "bo_after_error")

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

    def test_model_manifest_hash_mismatch_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = write_manifest(Path(tmp), "a" * 64)

            status = verify_model_manifest_file(
                str(manifest_path),
                expected_model_sha256="b" * 64,
            )

        self.assertFalse(status.verified)
        self.assertIn("does not match", status.unavailable_reason or "")

    def test_model_manifest_unsupported_family_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = write_manifest(
                Path(tmp),
                "a" * 64,
                overrides={"model_family": "shell_model"},
            )

            status = verify_model_manifest_file(str(manifest_path), expected_model_sha256="a" * 64)

        self.assertFalse(status.verified)
        self.assertIn("model_family", status.unavailable_reason or "")

    def test_model_manifest_rejects_pickle_style_artifact_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = write_manifest(
                Path(tmp),
                "a" * 64,
                overrides={"model_artifact": {"format": "torch_pickle", "sha256": "a" * 64}},
            )

            status = verify_model_manifest_file(str(manifest_path), expected_model_sha256="a" * 64)

        self.assertFalse(status.verified)
        self.assertIn("safetensors", status.unavailable_reason or "")

    def test_model_manifest_invalid_json_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "dreamer_v3_manifest.json"
            manifest_path.write_text("{not json", encoding="utf-8")

            status = verify_model_manifest_file(str(manifest_path))

        self.assertFalse(status.verified)
        self.assertIn("valid UTF-8 JSON", status.unavailable_reason or "")

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
            default_path = data_dir / "models" / "dreamer_v3.safetensors"
            default_manifest_path = data_dir / "models" / "dreamer_v3_manifest.json"
            with (
                patch.object(config_module, "_OPTIONS_PATH", root / "options.json"),
                patch.object(config_module, "_DATA_DIR", data_dir),
                patch.object(config_module, "_DEFAULT_DREAMER_V3_MODEL_ARTIFACT_PATH", default_path),
                patch.object(config_module, "_DEFAULT_DREAMER_V3_MODEL_MANIFEST_PATH", default_manifest_path),
                patch.object(config_module, "_RELEASE_DEFAULT_MODEL_ARTIFACT_SHA256", release_digest),
            ):
                config = config_module.Config.load()

        self.assertEqual(config.optimizer_model_artifact_sha256, release_digest)
        self.assertEqual(config.default_optimizer_model_artifact_sha256, release_digest)
        self.assertEqual(config.optimizer_model_artifact_path, str(default_path))
        self.assertEqual(config.optimizer_model_manifest_path, str(default_manifest_path))

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
                        "optimizer_model_manifest_path": "/models/trainer_manifest.json",
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
        self.assertEqual(config.optimizer_model_manifest_path, "/models/trainer_manifest.json")
        self.assertEqual(config.default_optimizer_model_artifact_sha256, release_digest)

    def test_aliases_normalize_to_bayesian_optimization(self) -> None:
        optimizer = RuntimeOptimizer(optimizer_mode="bo")

        self.assertEqual(optimizer.status().configured_mode, DEFAULT_OPTIMIZER_MODE)

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "optimizer_mode"):
            RuntimeOptimizer(optimizer_mode="shell")


def write_manifest(
    root: Path,
    model_sha256: str,
    *,
    overrides: dict | None = None,
) -> Path:
    manifest = {
        "format": "espresso_rl_model_manifest_v1",
        "schema_version": 1,
        "model_family": "dreamer_v3",
        "model_artifact": {"format": "safetensors", "sha256": model_sha256},
        "dataset": {
            "format": "espresso_rl_training_dataset_v1",
            "sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
        },
        "trainer": {
            "git_sha": "trainerabc",
            "training_config_sha256": "d" * 64,
        },
        "schemas": {
            "state_schema_version": 1,
            "action_schema_version": 1,
            "reward_schema_version": 1,
        },
        "runtime_compatibility": {
            "optimizer_mode": "dreamer_v3_shadow",
            "espresso_rl_runtime_schema_version": 1,
            "inference_ready": True,
        },
    }
    if overrides:
        manifest.update(overrides)
    path = root / "dreamer_v3_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


def write_model_bundle(
    root: Path,
    *,
    inference_ready: bool,
    optimizer_mode: str = OPTIMIZER_MODE_DREAMER_V3_SHADOW,
) -> tuple[Path, str, Path, str]:
    model_path = root / "dreamer_v3.safetensors"
    model_path.write_bytes(b"verified model")
    digest = hashlib.sha256(b"verified model").hexdigest()
    manifest_path = write_manifest(
        root,
        digest,
        overrides={
            "runtime_compatibility": {
                "optimizer_mode": optimizer_mode,
                "espresso_rl_runtime_schema_version": 1,
                "inference_ready": inference_ready,
            }
        },
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return model_path, digest, manifest_path, manifest_sha


if __name__ == "__main__":
    unittest.main()

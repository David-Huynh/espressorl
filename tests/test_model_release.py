from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from espresso_rl.application.checkpoint_loading import load_verified_dreamer_checkpoint
from espresso_rl.application.model_release import (
    CHECKSUMS_FILENAME,
    MODEL_FILENAME,
    MODEL_MANIFEST_FILENAME,
    RELEASE_RECORD_FILENAME,
    DreamerModelReleaseError,
    release_dreamer_checkpoint,
)
from espresso_rl.application.trainer_artifacts import build_dreamer_trainer_artifacts
from espresso_rl.domain.model_release import DreamerReleaseAuthorization
from espresso_rl.domain.optimization import OPTIMIZER_MODE_DREAMER_V3_ACTIVE
from espresso_rl.domain.trainer_artifacts import (
    TRAINER_ARTIFACT_STAGE_WORLD_MODEL_RELEASE_CANDIDATE,
    default_training_config,
)
from espresso_rl.release_cli import main as release_cli_main
from test_trainer_artifacts import canonical_json, dataset_export_text, training_row


class MemoryModelStore:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    def read_bytes(self, reference: str, *, max_bytes: int) -> bytes:
        payload = self.payloads[reference]
        if len(payload) > max_bytes:
            raise ValueError("artifact exceeds limit")
        return payload


class DreamerModelReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset_text, dataset_manifest = dataset_export_text([training_row(index) for index in range(1, 5)])
        config = default_training_config(
            seed=17,
            artifact_stage=TRAINER_ARTIFACT_STAGE_WORLD_MODEL_RELEASE_CANDIDATE,
        )
        config.update(
            {
                "world_model_release_epochs": 1,
                "world_model_release_batch_size": 2,
                "world_model_release_model_preset": "espresso_debug",
                "world_model_release_deter_dim": 16,
                "world_model_release_hidden_dim": 16,
                "world_model_release_stoch_size": 2,
                "world_model_release_class_size": 4,
                "world_model_release_action_embed_dim": 8,
                "world_model_release_gradient_steps_per_epoch": 1,
                "world_model_release_early_stop_patience": 1,
                "world_model_release_imagination_horizon": 3,
                "world_model_release_imagination_actor_hidden_dim": 16,
                "world_model_release_imagination_critic_hidden_dim": 16,
                "world_model_release_actor_critic_train_steps": 1,
                "world_model_release_imagination_batch_size": 2,
                "world_model_release_min_train_episodes": 3,
                "world_model_release_min_validation_episodes": 1,
            }
        )
        result = build_dreamer_trainer_artifacts(
            training_rows_jsonl=dataset_text,
            training_dataset_manifest_json=dataset_manifest,
            training_config_json=canonical_json(config) + "\n",
            trainer_git_sha="release-trainer-test",
            created_at=1_800_000_000,
        )
        files = {item.relative_path: item.content for item in result.files}
        cls.candidate_artifact = files[MODEL_FILENAME]
        cls.candidate_manifest = files[MODEL_MANIFEST_FILENAME]
        cls.candidate_artifact_sha256 = hashlib.sha256(cls.candidate_artifact).hexdigest()
        cls.candidate_manifest_sha256 = hashlib.sha256(cls.candidate_manifest).hexdigest()

    def test_release_is_deterministic_inference_ready_and_preserves_tensor_bytes(self) -> None:
        authorization = self.authorization()

        first = self.release(authorization)
        second = self.release(authorization)

        self.assertEqual(first.released_artifact_sha256, second.released_artifact_sha256)
        self.assertEqual(first.released_manifest_sha256, second.released_manifest_sha256)
        first_files = {item.relative_path: item for item in first.files}
        second_files = {item.relative_path: item for item in second.files}
        self.assertEqual(set(first_files), {MODEL_FILENAME, MODEL_MANIFEST_FILENAME, RELEASE_RECORD_FILENAME, CHECKSUMS_FILENAME})
        self.assertEqual(first_files[MODEL_FILENAME].content, second_files[MODEL_FILENAME].content)
        self.assertEqual(_tensor_data(self.candidate_artifact), _tensor_data(first_files[MODEL_FILENAME].content))

        manifest = json.loads(first_files[MODEL_MANIFEST_FILENAME].content)
        self.assertTrue(manifest["runtime_compatibility"]["inference_ready"])
        self.assertEqual(manifest["runtime_compatibility"]["optimizer_mode"], OPTIMIZER_MODE_DREAMER_V3_ACTIVE)
        self.assertEqual(manifest["release_authorization"], authorization.to_dict())
        record = json.loads(first_files[RELEASE_RECORD_FILENAME].content)
        self.assertTrue(record["verification"]["tensor_payloads_preserved"])
        self.assertFalse(record["verification"]["pickle_content_allowed"])

        loaded = load_verified_dreamer_checkpoint(
            MemoryModelStore(
                {
                    MODEL_FILENAME: first_files[MODEL_FILENAME].content,
                    MODEL_MANIFEST_FILENAME: first_files[MODEL_MANIFEST_FILENAME].content,
                }
            ),
            artifact_reference=MODEL_FILENAME,
            manifest_reference=MODEL_MANIFEST_FILENAME,
            expected_artifact_sha256=first.released_artifact_sha256,
        )
        self.assertTrue(loaded.inference_ready)
        self.assertEqual(loaded.release_authorization, authorization)

    def test_rejects_wrong_manifest_identity_before_release(self) -> None:
        authorization = self.authorization(candidate_manifest_sha256="0" * 64)

        with self.assertRaisesRegex(DreamerModelReleaseError, "manifest SHA-256"):
            self.release(authorization)

    def test_rejects_non_release_candidate_stage(self) -> None:
        header, tensor_data = _split_safetensors(self.candidate_artifact)
        header["__metadata__"]["artifact_stage"] = "world_model_train_preview"
        artifact = _encode_safetensors(header, tensor_data)
        manifest = json.loads(self.candidate_manifest)
        manifest["trainer"]["artifact_stage"] = "world_model_train_preview"
        manifest["model_artifact"]["sha256"] = hashlib.sha256(artifact).hexdigest()
        manifest_payload = (canonical_json(manifest) + "\n").encode("utf-8")
        authorization = self.authorization(
            candidate_artifact_sha256=hashlib.sha256(artifact).hexdigest(),
            candidate_manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        )

        with self.assertRaisesRegex(DreamerModelReleaseError, "only a world-model release candidate"):
            release_dreamer_checkpoint(
                MemoryModelStore({"candidate": artifact, "manifest": manifest_payload}),
                candidate_artifact_reference="candidate",
                candidate_manifest_reference="manifest",
                authorization=authorization,
            )

    def test_rejects_candidate_tensor_tampering_even_with_updated_outer_hash(self) -> None:
        artifact = bytearray(self.candidate_artifact)
        artifact[-1] ^= 1
        artifact_payload = bytes(artifact)
        manifest = json.loads(self.candidate_manifest)
        manifest["model_artifact"]["sha256"] = hashlib.sha256(artifact_payload).hexdigest()
        manifest_payload = (canonical_json(manifest) + "\n").encode("utf-8")
        authorization = self.authorization(
            candidate_artifact_sha256=hashlib.sha256(artifact_payload).hexdigest(),
            candidate_manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        )

        with self.assertRaisesRegex(DreamerModelReleaseError, "tensor .* SHA-256"):
            release_dreamer_checkpoint(
                MemoryModelStore({"candidate": artifact_payload, "manifest": manifest_payload}),
                candidate_artifact_reference="candidate",
                candidate_manifest_reference="manifest",
                authorization=authorization,
            )

    def test_rejects_already_released_checkpoint(self) -> None:
        released = self.release(self.authorization())
        files = {item.relative_path: item.content for item in released.files}
        authorization = self.authorization(
            candidate_artifact_sha256=released.released_artifact_sha256,
            candidate_manifest_sha256=released.released_manifest_sha256,
            release_version="v1.0.1-test",
        )

        with self.assertRaisesRegex(DreamerModelReleaseError, "already inference-ready"):
            release_dreamer_checkpoint(
                MemoryModelStore(
                    {
                        "candidate": files[MODEL_FILENAME],
                        "manifest": files[MODEL_MANIFEST_FILENAME],
                    }
                ),
                candidate_artifact_reference="candidate",
                candidate_manifest_reference="manifest",
                authorization=authorization,
            )

    def test_cli_writes_release_bundle_without_overwriting_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate_dir = root / "candidate"
            output_dir = root / "release"
            candidate_dir.mkdir()
            artifact_path = candidate_dir / MODEL_FILENAME
            manifest_path = candidate_dir / MODEL_MANIFEST_FILENAME
            artifact_path.write_bytes(self.candidate_artifact)
            manifest_path.write_bytes(self.candidate_manifest)
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = release_cli_main(
                    [
                        "--candidate-artifact",
                        str(artifact_path),
                        "--candidate-manifest",
                        str(manifest_path),
                        "--candidate-artifact-sha256",
                        self.candidate_artifact_sha256,
                        "--candidate-manifest-sha256",
                        self.candidate_manifest_sha256,
                        "--released-by",
                        "release-test",
                        "--release-version",
                        "v1.0.0-test",
                        "--released-at",
                        "1800000100",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["candidate_artifact_sha256"], self.candidate_artifact_sha256)
            self.assertTrue((output_dir / MODEL_FILENAME).is_file())
            self.assertTrue((output_dir / MODEL_MANIFEST_FILENAME).is_file())
            self.assertTrue((output_dir / RELEASE_RECORD_FILENAME).is_file())
            self.assertTrue((output_dir / CHECKSUMS_FILENAME).is_file())
            self.assertEqual(artifact_path.read_bytes(), self.candidate_artifact)

    def test_authorization_rejects_ambiguous_identity_and_approval(self) -> None:
        with self.assertRaisesRegex(ValueError, "released_by"):
            self.authorization(released_by=" release-test")
        with self.assertRaisesRegex(ValueError, "approval"):
            DreamerReleaseAuthorization(
                candidate_artifact_sha256=self.candidate_artifact_sha256,
                candidate_manifest_sha256=self.candidate_manifest_sha256,
                released_by="release-test",
                release_version="v1.0.0-test",
                released_at=1_800_000_100,
                approval="maybe",
            )
        with self.assertRaisesRegex(ValueError, "schema version"):
            DreamerReleaseAuthorization(
                candidate_artifact_sha256=self.candidate_artifact_sha256,
                candidate_manifest_sha256=self.candidate_manifest_sha256,
                released_by="release-test",
                release_version="v1.0.0-test",
                released_at=1_800_000_100,
                schema_version=True,
            )

    def authorization(self, **overrides) -> DreamerReleaseAuthorization:
        values = {
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "released_by": "release-test",
            "release_version": "v1.0.0-test",
            "released_at": 1_800_000_100,
        }
        values.update(overrides)
        return DreamerReleaseAuthorization(**values)

    def release(self, authorization: DreamerReleaseAuthorization):
        return release_dreamer_checkpoint(
            MemoryModelStore(
                {
                    "candidate": self.candidate_artifact,
                    "manifest": self.candidate_manifest,
                }
            ),
            candidate_artifact_reference="candidate",
            candidate_manifest_reference="manifest",
            authorization=authorization,
        )


def _split_safetensors(payload: bytes) -> tuple[dict, bytes]:
    header_length = int.from_bytes(payload[:8], "little")
    data_start = 8 + header_length
    return json.loads(payload[8:data_start]), payload[data_start:]


def _tensor_data(payload: bytes) -> bytes:
    return _split_safetensors(payload)[1]


def _encode_safetensors(header: dict, data: bytes) -> bytes:
    header_payload = canonical_json(header).encode("utf-8")
    return len(header_payload).to_bytes(8, "little") + header_payload + data


if __name__ == "__main__":
    unittest.main()

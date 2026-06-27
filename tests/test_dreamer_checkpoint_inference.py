from __future__ import annotations

from dataclasses import replace
import json
import unittest

from espresso_rl.application.checkpoint_loading import load_verified_dreamer_checkpoint
from espresso_rl.application.dreamer_shadow_inference import (
    DreamerShadowInferenceError,
    build_dreamer_shadow_inference_session,
)
from espresso_rl.application.trainer_artifacts import (
    MODEL_FILENAME,
    MODEL_MANIFEST_FILENAME,
    build_dreamer_trainer_artifacts,
)
from espresso_rl.domain.trainer_artifacts import (
    TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW,
    default_training_config,
)
from tests.test_trainer_artifacts import canonical_json, dataset_export_text, training_row


class MemoryArtifactStore:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self._payloads = payloads

    def read_bytes(self, reference: str, *, max_bytes: int) -> bytes:
        payload = self._payloads[reference]
        if len(payload) > max_bytes:
            raise ValueError("artifact exceeds limit")
        return payload


class DreamerCheckpointInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset_text, dataset_manifest = dataset_export_text(
            [training_row(1), training_row(2), training_row(3), training_row(4)]
        )
        config = default_training_config(
            seed=23,
            artifact_stage=TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW,
        )
        config["world_model_preview_epochs"] = 1
        config["world_model_preview_batch_size"] = 2
        config["world_model_preview_deter_dim"] = 16
        config["world_model_preview_hidden_dim"] = 16
        config["world_model_preview_stoch_size"] = 2
        config["world_model_preview_class_size"] = 4
        config["world_model_preview_action_embed_dim"] = 8
        result = build_dreamer_trainer_artifacts(
            training_rows_jsonl=dataset_text,
            training_dataset_manifest_json=dataset_manifest,
            training_config_json=canonical_json(config) + "\n",
            trainer_git_sha="trainerabc",
            created_at=1_800_000_000,
        )
        files = {file.relative_path: file for file in result.files}
        cls.checkpoint = load_verified_dreamer_checkpoint(
            MemoryArtifactStore(
                {
                    MODEL_FILENAME: files[MODEL_FILENAME].content,
                    MODEL_MANIFEST_FILENAME: files[MODEL_MANIFEST_FILENAME].content,
                }
            ),
            artifact_reference=MODEL_FILENAME,
            manifest_reference=MODEL_MANIFEST_FILENAME,
            expected_artifact_sha256=files[MODEL_FILENAME].sha256,
        )
        cls.manifest = json.loads(files[MODEL_MANIFEST_FILENAME].content)

    def test_materialized_outputs_match_authenticated_training_probe(self) -> None:
        session = build_dreamer_shadow_inference_session(self.checkpoint)

        self.assertTrue(session.status.parity_verified)
        self.assertEqual(
            session.status.inference_probe_sha256,
            self.manifest["model_artifact"]["inference_probe_sha256"],
        )
        self.assertEqual(
            session.status.heldout_inference_sha256,
            self.manifest["model_artifact"]["heldout_inference_sha256"],
        )
        self.assertFalse(session.status.inference_ready)
        self.assertFalse(session.status.recommendation_enabled)
        self.assertFalse(session.status.machine_control_enabled)

    def test_missing_parameter_is_rejected(self) -> None:
        checkpoint = replace(self.checkpoint, tensors=self.checkpoint.tensors[1:])

        with self.assertRaisesRegex(DreamerShadowInferenceError, "missing"):
            build_dreamer_shadow_inference_session(checkpoint)

    def test_extra_parameter_is_rejected(self) -> None:
        extra = replace(self.checkpoint.tensors[0], name="actor.unexpected_parameter")
        checkpoint = replace(self.checkpoint, tensors=(*self.checkpoint.tensors, extra))

        with self.assertRaisesRegex(DreamerShadowInferenceError, "extra"):
            build_dreamer_shadow_inference_session(checkpoint)

    def test_wrong_parameter_shape_is_rejected(self) -> None:
        target_index = next(
            index
            for index, tensor in enumerate(self.checkpoint.tensors)
            if len(tensor.shape) == 2 and tensor.element_count > 1
        )
        target = self.checkpoint.tensors[target_index]
        tensors = list(self.checkpoint.tensors)
        tensors[target_index] = replace(target, shape=(target.element_count,))
        checkpoint = replace(self.checkpoint, tensors=tuple(tensors))

        with self.assertRaisesRegex(DreamerShadowInferenceError, "shape is incompatible"):
            build_dreamer_shadow_inference_session(checkpoint)

    def test_probe_mismatch_is_rejected(self) -> None:
        checkpoint = replace(self.checkpoint, inference_probe_sha256="0" * 64)

        with self.assertRaisesRegex(DreamerShadowInferenceError, "inference probe"):
            build_dreamer_shadow_inference_session(checkpoint)


if __name__ == "__main__":
    unittest.main()

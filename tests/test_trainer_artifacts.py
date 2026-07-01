from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from espresso_rl.application.checkpoint_loading import load_verified_dreamer_checkpoint
from espresso_rl.application.dreamer_shadow_inference import build_dreamer_shadow_inference_session
from espresso_rl.application.trainer_artifacts import (
    AUDIT_REPORT_FILENAME,
    CHECKPOINT_ARTIFACT_FORMAT,
    CHECKSUMS_FILENAME,
    MODEL_FILENAME,
    MODEL_MANIFEST_FILENAME,
    TRAINING_CONFIG_FILENAME,
    TrainerArtifactError,
    build_dreamer_trainer_artifacts,
    DEFAULT_MAX_DATASET_BYTES,
    validate_dreamer_checkpoint_safetensors,
)
from espresso_rl.domain.model_manifest import validate_model_manifest
from espresso_rl.domain.trainer_artifacts import (
    TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW,
    TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE,
    default_training_config,
)
from espresso_rl.trainer_cli import main as trainer_cli_main


class MemoryArtifactStore:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self._payloads = payloads

    def read_bytes(self, reference: str, *, max_bytes: int) -> bytes:
        payload = self._payloads[reference]
        if len(payload) > max_bytes:
            raise ValueError("artifact exceeds limit")
        return payload


class TrainerArtifactTests(unittest.TestCase):
    def test_builds_expected_contract_artifacts_from_valid_dataset(self) -> None:
        dataset_text, manifest_text = dataset_export_text([training_row(1)])
        config_text = canonical_json(default_training_config(seed=7)) + "\n"

        result = build_dreamer_trainer_artifacts(
            training_rows_jsonl=dataset_text,
            training_dataset_manifest_json=manifest_text,
            training_config_json=config_text,
            trainer_git_sha="trainerabc",
            created_at=1_800_000_000,
        )

        files = {file.relative_path: file for file in result.files}
        self.assertEqual(
            set(files),
            {
                MODEL_FILENAME,
                MODEL_MANIFEST_FILENAME,
                TRAINING_CONFIG_FILENAME,
                AUDIT_REPORT_FILENAME,
                CHECKSUMS_FILENAME,
            },
        )
        self.assertEqual(result.row_count, 1)
        model_header = safetensors_header(files[MODEL_FILENAME].content)
        model_metadata = model_header["__metadata__"]
        manifest = json.loads(files[MODEL_MANIFEST_FILENAME].content.decode("utf-8"))
        config = json.loads(files[TRAINING_CONFIG_FILENAME].content.decode("utf-8"))
        self.assertEqual(config["dreamer_control_spec"]["observation_interval_ms"], 250)
        self.assertEqual(config["dreamer_control_spec"]["decision_interval_ms"], 1000)
        self.assertIn("dreamer_live_action_spec", config)
        self.assertEqual(manifest["model_artifact"]["format"], "safetensors")
        self.assertEqual(manifest["model_artifact"]["checkpoint_format"], CHECKPOINT_ARTIFACT_FORMAT)
        self.assertEqual(manifest["model_artifact"]["tensor_count"], 0)
        self.assertEqual(manifest["model_artifact"]["sha256"], files[MODEL_FILENAME].sha256)
        self.assertEqual(model_metadata["format"], CHECKPOINT_ARTIFACT_FORMAT)
        self.assertEqual(model_metadata["inference_ready"], "false")
        self.assertEqual(manifest["dataset"]["sha256"], hashlib.sha256(dataset_text.encode("utf-8")).hexdigest())
        self.assertEqual(manifest["trainer"]["training_config_sha256"], files[TRAINING_CONFIG_FILENAME].sha256)
        self.assertFalse(manifest["runtime_compatibility"]["inference_ready"])
        validate_dreamer_checkpoint_safetensors(files[MODEL_FILENAME].content, manifest)
        manifest_validation = validate_model_manifest(
            manifest,
            expected_model_sha256=files[MODEL_FILENAME].sha256,
        )
        self.assertFalse(manifest_validation.verified)
        self.assertIn("not inference-ready", manifest_validation.unavailable_reason or "")
        audit = json.loads(files[AUDIT_REPORT_FILENAME].content.decode("utf-8"))
        self.assertFalse(audit["inference_ready"])
        self.assertTrue(audit["zero_trust"]["dreamer_tensors_revalidated"])
        tensor_contract = audit["dreamer_tensor_contract"]
        self.assertEqual(tensor_contract["episode_format"], "espresso_rl_dreamer_episode_v4")
        self.assertEqual(tensor_contract["episode_schema_version"], 4)
        self.assertEqual(tensor_contract["episode_count"], 1)
        self.assertEqual(tensor_contract["episode_length_steps"], {"avg": 4.0, "max": 4, "min": 4})
        self.assertEqual(tensor_contract["observation_interval_ms"], 250)
        self.assertEqual(tensor_contract["decision_interval_ms"], 1000)
        self.assertEqual(tensor_contract["decision_step_count"], 4)
        self.assertEqual(tensor_contract["context_window_size"], 16)
        self.assertEqual(tensor_contract["tensor_shapes"]["observations"], [1, 4, 5])
        self.assertEqual(tensor_contract["tensor_shapes"]["dynamic_actions"], [1, 4, 7])
        self.assertEqual(tensor_contract["tensor_shapes"]["pre_shot_actions"], [1, 9])
        self.assertEqual(tensor_contract["tensor_shapes"]["pre_shot_action_indexes"], [1, 9])
        self.assertEqual(tensor_contract["tensor_shapes"]["pre_shot_action_mask"], [1, 9])
        self.assertEqual(tensor_contract["tensor_shapes"]["pre_shot_capability_mask"], [1, 9])
        self.assertEqual(tensor_contract["tensor_shapes"]["context_static"], [1, 16, 20])
        self.assertEqual(tensor_contract["tensor_shapes"]["context_terminal"], [1, 16, 18])
        self.assertEqual(tensor_contract["tensor_shapes"]["context_trajectory_embedding"], [1, 16, 77])
        self.assertEqual(tensor_contract["tensor_shapes"]["context_mask"], [1, 16])
        self.assertIn("temperature_c", tensor_contract["feature_names"]["observations"])
        self.assertIn("yield_stop_target_g", tensor_contract["feature_names"]["dynamic_actions"])
        self.assertIn("pump_target_mode", tensor_contract["feature_names"]["pre_shot_actions"])
        self.assertIn("reward", tensor_contract["feature_names"]["context_terminal"])
        self.assertIn(
            "observation_pressure_bar_mean",
            tensor_contract["feature_names"]["context_trajectory_embedding"],
        )
        self.assertEqual(len(tensor_contract["feature_layout_sha256"]), 64)
        self.assertEqual(len(tensor_contract["control_spec_sha256"]), 64)
        self.assertEqual(len(tensor_contract["pre_shot_action_spec_sha256"]), 64)
        self.assertEqual(len(tensor_contract["live_action_spec_sha256"]), 64)
        self.assertEqual(len(tensor_contract["taste_objective_spec_sha256"]), 64)
        self.assertEqual(len(tensor_contract["tensor_contract_sha256"]), 64)
        self.assertEqual(
            manifest["model_artifact"]["pre_shot_action_spec_sha256"],
            tensor_contract["pre_shot_action_spec_sha256"],
        )
        self.assertEqual(
            model_metadata["pre_shot_action_spec_sha256"],
            tensor_contract["pre_shot_action_spec_sha256"],
        )
        self.assertEqual(
            manifest["model_artifact"]["live_action_spec_sha256"],
            tensor_contract["live_action_spec_sha256"],
        )
        self.assertEqual(
            model_metadata["live_action_spec_sha256"],
            tensor_contract["live_action_spec_sha256"],
        )
        self.assertEqual(
            manifest["model_artifact"]["taste_objective_spec_sha256"],
            tensor_contract["taste_objective_spec_sha256"],
        )
        self.assertEqual(
            model_metadata["taste_objective_spec_sha256"],
            tensor_contract["taste_objective_spec_sha256"],
        )
        self.assertTrue(audit["zero_trust"]["checkpoint_safetensors_validated"])
        self.assertIn(f"{files[MODEL_FILENAME].sha256}  {MODEL_FILENAME}", files[CHECKSUMS_FILENAME].content.decode("utf-8"))
        loaded = load_verified_dreamer_checkpoint(
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
        self.assertFalse(loaded.inference_ready)
        self.assertEqual(loaded.tensors, ())

    def test_rejects_dataset_manifest_hash_mismatch(self) -> None:
        dataset_text, manifest_text = dataset_export_text([training_row(1)])
        manifest = json.loads(manifest_text)
        manifest["dataset_sha256"] = "0" * 64

        with self.assertRaisesRegex(TrainerArtifactError, "dataset_sha256"):
            build_dreamer_trainer_artifacts(
                training_rows_jsonl=dataset_text,
                training_dataset_manifest_json=canonical_json(manifest) + "\n",
                training_config_json=canonical_json(default_training_config()) + "\n",
                trainer_git_sha="trainerabc",
            )

    def test_rejects_absolute_grinder_fields_in_dataset_rows(self) -> None:
        row = training_row(1)
        row["action"]["current_absolute_step"] = 42
        dataset_text, manifest_text = dataset_export_text([row])

        with self.assertRaisesRegex(TrainerArtifactError, "current_absolute_step"):
            build_dreamer_trainer_artifacts(
                training_rows_jsonl=dataset_text,
                training_dataset_manifest_json=manifest_text,
                training_config_json=canonical_json(default_training_config()) + "\n",
                trainer_git_sha="trainerabc",
            )

    def test_rejects_non_safetensors_output_filename(self) -> None:
        dataset_text, manifest_text = dataset_export_text([training_row(1)])

        with self.assertRaisesRegex(TrainerArtifactError, "unsupported output filename"):
            build_dreamer_trainer_artifacts(
                training_rows_jsonl=dataset_text,
                training_dataset_manifest_json=manifest_text,
                training_config_json=canonical_json(default_training_config()) + "\n",
                trainer_git_sha="trainerabc",
                model_filename="dreamer_v3.pt",
            )

    def test_rejects_dataset_that_cannot_build_dreamer_tensors(self) -> None:
        row = training_row(1)
        row["observation"].pop("fixed_cadence_sequence")
        dataset_text, manifest_text = dataset_export_text([row])

        with self.assertRaisesRegex(TrainerArtifactError, "Dreamer tensor contract"):
            build_dreamer_trainer_artifacts(
                training_rows_jsonl=dataset_text,
                training_dataset_manifest_json=manifest_text,
                training_config_json=canonical_json(default_training_config()) + "\n",
                trainer_git_sha="trainerabc",
            )

    def test_training_config_control_spec_changes_tensor_audit(self) -> None:
        dataset_text, manifest_text = dataset_export_text([training_row(1)])
        config = default_training_config()
        config["dreamer_control_spec"]["decision_interval_ms"] = 500
        config["dreamer_control_spec"]["dynamic_control_enabled"] = True
        config["dreamer_control_spec"]["pressure_control_allowed"] = True

        result = build_dreamer_trainer_artifacts(
            training_rows_jsonl=dataset_text,
            training_dataset_manifest_json=manifest_text,
            training_config_json=canonical_json(config) + "\n",
            trainer_git_sha="trainerabc",
        )

        files = {file.relative_path: file for file in result.files}
        audit = json.loads(files[AUDIT_REPORT_FILENAME].content.decode("utf-8"))
        tensor_contract = audit["dreamer_tensor_contract"]
        self.assertEqual(tensor_contract["decision_interval_ms"], 500)
        self.assertEqual(tensor_contract["decision_step_count"], 2)
        self.assertTrue(tensor_contract["control_spec"]["pressure_control_allowed"])

    def test_training_config_pre_shot_spec_changes_authenticated_contract(self) -> None:
        dataset_text, manifest_text = dataset_export_text([training_row(1)])
        config = default_training_config()
        config["dreamer_pre_shot_action_spec"]["bins"]["pressure_target_bar"] = [
            float(value) for value in range(13)
        ]

        result = build_dreamer_trainer_artifacts(
            training_rows_jsonl=dataset_text,
            training_dataset_manifest_json=manifest_text,
            training_config_json=canonical_json(config) + "\n",
            trainer_git_sha="trainerabc",
        )

        files = {file.relative_path: file for file in result.files}
        audit = json.loads(files[AUDIT_REPORT_FILENAME].content.decode("utf-8"))
        manifest = json.loads(files[MODEL_MANIFEST_FILENAME].content.decode("utf-8"))
        metadata = safetensors_metadata(files[MODEL_FILENAME].content)
        expected_hash = audit["dreamer_tensor_contract"]["pre_shot_action_spec_sha256"]
        self.assertEqual(manifest["model_artifact"]["pre_shot_action_spec_sha256"], expected_hash)
        self.assertEqual(metadata["pre_shot_action_spec_sha256"], expected_hash)

    def test_rejects_training_config_without_pre_shot_contract(self) -> None:
        dataset_text, manifest_text = dataset_export_text([training_row(1)])
        config = default_training_config()
        config.pop("dreamer_pre_shot_action_spec")

        with self.assertRaisesRegex(TrainerArtifactError, "dreamer_pre_shot_action_spec is required"):
            build_dreamer_trainer_artifacts(
                training_rows_jsonl=dataset_text,
                training_dataset_manifest_json=manifest_text,
                training_config_json=canonical_json(config) + "\n",
                trainer_git_sha="trainerabc",
            )

    def test_world_model_smoke_stage_writes_deterministic_metrics_without_inference_ready(self) -> None:
        dataset_text, manifest_text = dataset_export_text([training_row(1)])
        config_text = canonical_json(
            default_training_config(seed=11, artifact_stage=TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE)
        ) + "\n"

        first = build_dreamer_trainer_artifacts(
            training_rows_jsonl=dataset_text,
            training_dataset_manifest_json=manifest_text,
            training_config_json=config_text,
            trainer_git_sha="trainerabc",
            created_at=1_800_000_000,
        )
        second = build_dreamer_trainer_artifacts(
            training_rows_jsonl=dataset_text,
            training_dataset_manifest_json=manifest_text,
            training_config_json=config_text,
            trainer_git_sha="trainerabc",
            created_at=1_800_000_000,
        )

        first_files = {file.relative_path: file for file in first.files}
        second_files = {file.relative_path: file for file in second.files}
        self.assertEqual(first_files[AUDIT_REPORT_FILENAME].content, second_files[AUDIT_REPORT_FILENAME].content)
        self.assertEqual(first_files[MODEL_FILENAME].content, second_files[MODEL_FILENAME].content)
        audit = json.loads(first_files[AUDIT_REPORT_FILENAME].content.decode("utf-8"))
        manifest = json.loads(first_files[MODEL_MANIFEST_FILENAME].content.decode("utf-8"))
        model_metadata = safetensors_metadata(first_files[MODEL_FILENAME].content)

        self.assertEqual(audit["artifact_stage"], TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE)
        self.assertTrue(audit["zero_trust"]["world_model_smoke_trained"])
        self.assertEqual(audit["world_model_smoke"]["seed"], 11)
        self.assertEqual(audit["world_model_smoke"]["train_steps"], 2)
        self.assertLess(
            audit["world_model_smoke"]["final"]["loss_total"],
            audit["world_model_smoke"]["initial"]["loss_total"],
        )
        self.assertFalse(manifest["runtime_compatibility"]["inference_ready"])
        self.assertEqual(manifest["trainer"]["artifact_stage"], TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE)
        self.assertEqual(model_metadata["artifact_stage"], TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE)
        self.assertEqual(len(model_metadata["world_model_smoke_sha256"]), 64)
        self.assertEqual(manifest["model_artifact"]["tensor_count"], 0)

    def test_world_model_train_preview_stage_writes_split_and_loss_curves_without_inference_ready(self) -> None:
        dataset_text, manifest_text = dataset_export_text(
            [training_row(1), training_row(2), training_row(3), training_row(4)]
        )
        config = default_training_config(seed=17, artifact_stage=TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW)
        config["world_model_preview_epochs"] = 2
        config["world_model_preview_batch_size"] = 2
        config["world_model_preview_deter_dim"] = 16
        config["world_model_preview_hidden_dim"] = 16
        config["world_model_preview_stoch_size"] = 2
        config["world_model_preview_class_size"] = 4
        config["world_model_preview_action_embed_dim"] = 8
        config_text = canonical_json(config) + "\n"

        first = build_dreamer_trainer_artifacts(
            training_rows_jsonl=dataset_text,
            training_dataset_manifest_json=manifest_text,
            training_config_json=config_text,
            trainer_git_sha="trainerabc",
            created_at=1_800_000_000,
        )
        second = build_dreamer_trainer_artifacts(
            training_rows_jsonl=dataset_text,
            training_dataset_manifest_json=manifest_text,
            training_config_json=config_text,
            trainer_git_sha="trainerabc",
            created_at=1_800_000_000,
        )

        first_files = {file.relative_path: file for file in first.files}
        second_files = {file.relative_path: file for file in second.files}
        self.assertEqual(first_files[AUDIT_REPORT_FILENAME].content, second_files[AUDIT_REPORT_FILENAME].content)
        self.assertEqual(first_files[MODEL_FILENAME].content, second_files[MODEL_FILENAME].content)
        audit = json.loads(first_files[AUDIT_REPORT_FILENAME].content.decode("utf-8"))
        manifest = json.loads(first_files[MODEL_MANIFEST_FILENAME].content.decode("utf-8"))
        model_metadata = safetensors_metadata(first_files[MODEL_FILENAME].content)
        model_header = safetensors_header(first_files[MODEL_FILENAME].content)
        preview = audit["world_model_train_preview"]

        self.assertEqual(audit["artifact_stage"], TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW)
        self.assertTrue(audit["zero_trust"]["world_model_train_preview_trained"])
        self.assertFalse(audit["inference_ready"])
        self.assertFalse(manifest["runtime_compatibility"]["inference_ready"])
        self.assertEqual(manifest["trainer"]["artifact_stage"], TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW)
        self.assertEqual(model_metadata["artifact_stage"], TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW)
        self.assertEqual(len(model_metadata["world_model_train_preview_sha256"]), 64)
        self.assertEqual(model_metadata["format"], CHECKPOINT_ARTIFACT_FORMAT)
        self.assertGreater(manifest["model_artifact"]["tensor_count"], 0)
        self.assertEqual(
            manifest["model_artifact"]["component_names"],
            ["actor", "context_encoder", "critic", "world_model"],
        )
        self.assertEqual(
            manifest["model_artifact"]["architecture"]["control_spec"],
            config["dreamer_control_spec"],
        )
        self.assertEqual(
            manifest["model_artifact"]["architecture"]["context_encoder"]["context_dim"],
            16,
        )
        self.assertIn("world_model.reward_bins", model_header)
        self.assertIn("context_encoder.recurrent.weight_ih_l0", model_header)
        self.assertIn("actor.pre_shot_action_bins", model_header)
        self.assertIn("actor.pre_shot_action_bin_counts", model_header)
        self.assertIn("actor.live_action_bins", model_header)
        self.assertIn("actor.live_action_bin_counts", model_header)
        self.assertIn("critic.value_bins", model_header)
        self.assertEqual(
            manifest["model_artifact"]["tensor_manifest_sha256"],
            model_metadata["tensor_manifest_sha256"],
        )
        self.assertEqual(
            manifest["model_artifact"]["evaluation_report_sha256"],
            model_metadata["evaluation_report_sha256"],
        )
        self.assertEqual(
            manifest["model_artifact"]["inference_probe_sha256"],
            model_metadata["inference_probe_sha256"],
        )
        self.assertEqual(
            manifest["model_artifact"]["heldout_inference_sha256"],
            model_metadata["heldout_inference_sha256"],
        )
        validate_dreamer_checkpoint_safetensors(first_files[MODEL_FILENAME].content, manifest)
        self.assertEqual(preview["seed"], 17)
        self.assertEqual(preview["epochs_requested"], 2)
        self.assertEqual(preview["epochs_completed"], 2)
        self.assertEqual(preview["batch_size"], 2)
        self.assertEqual(preview["gradient_steps_per_epoch"], 1)
        self.assertEqual(preview["actor_critic_train_steps"], 3)
        self.assertEqual(preview["actor_learning_rate"], 0.0003)
        self.assertEqual(preview["critic_learning_rate"], 0.0003)
        self.assertEqual(preview["imagination_batch_size"], 4)
        self.assertEqual(preview["actor_critic_gradient_clip_norm"], 10.0)
        self.assertEqual(preview["model_config"]["deter_dim"], 16)
        self.assertEqual(preview["model_config"]["hidden_dim"], 16)
        self.assertEqual(preview["model_config"]["stoch_size"], 2)
        self.assertEqual(preview["model_config"]["class_size"], 4)
        self.assertEqual(len(preview["train_loss_curve"]), 2)
        self.assertEqual(len(preview["validation_loss_curve"]), 2)
        self.assertEqual(len(preview["actor_critic_train_curve"]), 3)
        self.assertIn("actor_loss", preview["actor_critic_train_curve"][0])
        self.assertIn("critic_loss", preview["actor_critic_train_curve"][0])
        self.assertIn("imagined_return_mean", preview["actor_critic_train_curve"][0])
        evaluation = preview["evaluation_report"]
        evaluation_sha = hashlib.sha256(canonical_json(evaluation).encode("utf-8")).hexdigest()
        self.assertEqual(audit["evaluation_report_sha256"], evaluation_sha)
        self.assertEqual(
            audit["heldout_inference_sha256"],
            manifest["model_artifact"]["heldout_inference_sha256"],
        )
        self.assertTrue(audit["zero_trust"]["heldout_inference_parity_verified"])
        self.assertEqual(manifest["model_artifact"]["evaluation_report_sha256"], evaluation_sha)
        self.assertFalse(evaluation["inference_ready"])
        self.assertIn("evaluation_passed", evaluation["gates"])
        self.assertTrue(evaluation["gates"]["action_mask_ok"])
        self.assertEqual(evaluation["actor"]["unsupported_dynamic_action_abs_max"], 0.0)
        self.assertIn("loss_dyn", preview["train_loss_curve"][0])
        self.assertIn("loss_rep", preview["validation_loss_curve"][0])
        self.assertFalse(preview["imagination_preview"]["inference_ready"])
        self.assertTrue(preview["imagination_preview"]["contract_only"])
        self.assertEqual(preview["imagination_preview"]["pre_shot_logits_shape"][:2], [1, 9])
        self.assertEqual(preview["imagination_preview"]["live_action_logits_shape"][:2], [1, 3])
        self.assertEqual(preview["imagination_preview"]["live_action_choice_shape"], [1, 3, 7])
        self.assertEqual(preview["imagination_preview"]["pre_shot_held_action_shape"], [1, 3, 9])
        self.assertEqual(preview["imagination_preview"]["lambda_return_shape"], [1, 3])
        self.assertEqual(preview["dataset_split"]["train_source_training_row_ids"], [1, 2, 3])
        self.assertEqual(preview["dataset_split"]["validation_source_training_row_ids"], [4])
        self.assertEqual(preview["dataset_split_sha256"], preview["dataset_split"]["dataset_split_sha256"])
        loaded = load_verified_dreamer_checkpoint(
            MemoryArtifactStore(
                {
                    MODEL_FILENAME: first_files[MODEL_FILENAME].content,
                    MODEL_MANIFEST_FILENAME: first_files[MODEL_MANIFEST_FILENAME].content,
                }
            ),
            artifact_reference=MODEL_FILENAME,
            manifest_reference=MODEL_MANIFEST_FILENAME,
            expected_artifact_sha256=first_files[MODEL_FILENAME].sha256,
        )
        self.assertEqual(loaded.component_names, ("actor", "context_encoder", "critic", "world_model"))
        self.assertGreater(len(loaded.tensors), 0)
        shadow_session = build_dreamer_shadow_inference_session(loaded)
        self.assertTrue(shadow_session.status.parity_verified)
        self.assertFalse(shadow_session.status.inference_ready)
        self.assertFalse(shadow_session.status.recommendation_enabled)
        self.assertFalse(shadow_session.status.machine_control_enabled)
        self.assertEqual(
            shadow_session.status.inference_probe_sha256,
            manifest["model_artifact"]["inference_probe_sha256"],
        )
        self.assertEqual(
            shadow_session.status.heldout_inference_sha256,
            manifest["model_artifact"]["heldout_inference_sha256"],
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in shadow_session.models.world_model.parameters())
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in shadow_session.models.context_encoder.parameters())
        )

    def test_checkpoint_safetensors_validation_rejects_manifest_tampering(self) -> None:
        dataset_text, manifest_text = dataset_export_text(
            [training_row(1), training_row(2), training_row(3), training_row(4)]
        )
        config = default_training_config(seed=17, artifact_stage=TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW)
        config["world_model_preview_epochs"] = 1
        result = build_dreamer_trainer_artifacts(
            training_rows_jsonl=dataset_text,
            training_dataset_manifest_json=manifest_text,
            training_config_json=canonical_json(config) + "\n",
            trainer_git_sha="trainerabc",
            created_at=1_800_000_000,
        )
        files = {file.relative_path: file for file in result.files}
        manifest = json.loads(files[MODEL_MANIFEST_FILENAME].content.decode("utf-8"))
        manifest["model_artifact"]["tensor_manifest"]["tensors"]["actor.pre_shot_action_bins"]["shape"] = [99]

        with self.assertRaisesRegex(TrainerArtifactError, "hash"):
            validate_dreamer_checkpoint_safetensors(files[MODEL_FILENAME].content, manifest)

        payload = bytearray(files[MODEL_FILENAME].content)
        payload[-1] ^= 1
        tampered_manifest = json.loads(files[MODEL_MANIFEST_FILENAME].content.decode("utf-8"))
        tampered_manifest["model_artifact"]["sha256"] = hashlib.sha256(payload).hexdigest()

        with self.assertRaisesRegex(TrainerArtifactError, "sha256"):
            validate_dreamer_checkpoint_safetensors(bytes(payload), tampered_manifest)

    def test_world_model_train_preview_requires_enough_episodes_for_validation(self) -> None:
        dataset_text, manifest_text = dataset_export_text([training_row(1)])
        config_text = canonical_json(
            default_training_config(seed=17, artifact_stage=TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW)
        ) + "\n"

        with self.assertRaisesRegex(TrainerArtifactError, "requires at least two episodes"):
            build_dreamer_trainer_artifacts(
                training_rows_jsonl=dataset_text,
                training_dataset_manifest_json=manifest_text,
                training_config_json=config_text,
                trainer_git_sha="trainerabc",
            )

    def test_rejects_training_config_with_non_dreamer_adaptive_control(self) -> None:
        dataset_text, manifest_text = dataset_export_text([training_row(1)])
        config = default_training_config()
        config["dreamer_control_spec"]["optimizer_family"] = "bayesian_optimization"
        config["dreamer_control_spec"]["dynamic_control_enabled"] = True
        config["dreamer_control_spec"]["pressure_control_allowed"] = True

        with self.assertRaisesRegex(TrainerArtifactError, "only available for Dreamer"):
            build_dreamer_trainer_artifacts(
                training_rows_jsonl=dataset_text,
                training_dataset_manifest_json=manifest_text,
                training_config_json=canonical_json(config) + "\n",
                trainer_git_sha="trainerabc",
            )

    def test_dataset_size_guard_is_configurable_resource_protection(self) -> None:
        dataset_text, manifest_text = dataset_export_text([training_row(1)])
        self.assertGreater(DEFAULT_MAX_DATASET_BYTES, len(dataset_text.encode("utf-8")))

        with self.assertRaisesRegex(TrainerArtifactError, "too large"):
            build_dreamer_trainer_artifacts(
                training_rows_jsonl=dataset_text,
                training_dataset_manifest_json=manifest_text,
                training_config_json=canonical_json(default_training_config()) + "\n",
                trainer_git_sha="trainerabc",
                max_dataset_bytes=1,
            )

    def test_cli_writes_artifact_files(self) -> None:
        dataset_text, manifest_text = dataset_export_text([training_row(1)])
        config_text = canonical_json(default_training_config()) + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "training_rows.jsonl"
            manifest_path = root / "manifest.json"
            config_path = root / "training_config.json"
            output_dir = root / "out"
            dataset_path.write_text(dataset_text, encoding="utf-8")
            manifest_path.write_text(manifest_text, encoding="utf-8")
            config_path.write_text(config_text, encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                exit_code = trainer_cli_main(
                    [
                        "--dataset-jsonl",
                        str(dataset_path),
                        "--dataset-manifest",
                        str(manifest_path),
                        "--training-config",
                        str(config_path),
                        "--output-dir",
                        str(output_dir),
                        "--trainer-git-sha",
                        "trainerabc",
                        "--created-at",
                        "1800000000",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / MODEL_FILENAME).is_file())
            self.assertTrue((output_dir / MODEL_MANIFEST_FILENAME).is_file())
            self.assertTrue((output_dir / CHECKSUMS_FILENAME).is_file())

    def test_cli_writes_world_model_smoke_default_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "training_config.json"

            with redirect_stdout(io.StringIO()):
                exit_code = trainer_cli_main(
                    [
                        "--write-default-config",
                        str(config_path),
                        "--artifact-stage",
                        "world_model_smoke",
                        "--seed",
                        "13",
                    ]
                )

            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(config["artifact_stage"], TRAINER_ARTIFACT_STAGE_WORLD_MODEL_SMOKE)
            self.assertEqual(config["seed"], 13)
            self.assertEqual(config["world_model_smoke_steps"], 2)

    def test_cli_writes_world_model_train_preview_default_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "training_config.json"

            with redirect_stdout(io.StringIO()):
                exit_code = trainer_cli_main(
                    [
                        "--write-default-config",
                        str(config_path),
                        "--artifact-stage",
                        "world_model_train_preview",
                        "--seed",
                        "19",
                    ]
                )

            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(config["artifact_stage"], TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW)
            self.assertEqual(config["seed"], 19)
            self.assertEqual(config["world_model_preview_epochs"], 3)
            self.assertEqual(config["world_model_preview_batch_size"], 4)
            self.assertEqual(config["world_model_preview_model_preset"], "espresso_debug")
            self.assertEqual(config["world_model_preview_stoch_size"], 4)
            self.assertEqual(config["world_model_preview_imagination_horizon"], 3)
            self.assertEqual(config["world_model_preview_imagination_actor_hidden_dim"], 32)
            self.assertEqual(config["world_model_preview_imagination_critic_hidden_dim"], 32)
            self.assertEqual(config["world_model_preview_actor_critic_train_steps"], 3)
            self.assertEqual(config["world_model_preview_actor_learning_rate"], 0.0003)
            self.assertEqual(config["world_model_preview_critic_learning_rate"], 0.0003)
            self.assertEqual(config["world_model_preview_imagination_batch_size"], 4)
            self.assertEqual(config["world_model_preview_actor_critic_gradient_clip_norm"], 10.0)


def dataset_export_text(rows: list[dict]) -> tuple[str, str]:
    dataset_text = "".join(canonical_json(row) + "\n" for row in rows)
    dataset_sha256 = hashlib.sha256(dataset_text.encode("utf-8")).hexdigest()
    manifest = {
        "format": "espresso_rl_training_dataset_v1",
        "schema_version": 1,
        "created_at": 1_800_000_000,
        "export_id": "training_dataset_v1_1800000000_test",
        "source": "validated_training_dataset",
        "source_git_sha": "sourceabc",
        "row_count": len(rows),
        "skipped_row_count": 0,
        "limit": 50_000,
        "dataset_sha256": dataset_sha256,
        "canonical_dataset_file": "training_rows.jsonl",
        "canonical_row_format": "espresso_rl_training_transition_v1",
        "files": [],
        "zero_trust": {
            "canonical_transitions_only": True,
            "absolute_grinder_fields_included": False,
            "raw_uploads_included": False,
            "adapter_payloads_included": False,
            "executable_content_included": False,
        },
    }
    return dataset_text, canonical_json(manifest) + "\n"


def training_row(row_id: int) -> dict:
    return {
        "format": "espresso_rl_training_transition_v1",
        "schema_version": 1,
        "training_row_id": row_id,
        "source": {
            "source_kind": "community_validated_shot",
            "source_validation_id": row_id,
            "install_id": "install_1",
            "payload_hash": "a" * 64,
            "trust_weight": 0.2,
        },
        "context": {
            "machine_id": "machine_1",
            "machine_adapter": "gaggimate",
            "bean_context_id": "bean_1",
            "grinder_context_id": "grinder_1",
            "microns_per_step": 12.5,
            "step_direction": "higher_is_finer",
        },
        "action": {
            "relative_grind_steps_from_reference": 0.0,
            "relative_grind_um_from_reference": 0.0,
            "dose_g": 18.0,
            "target_yield_g": 36.0,
            "target_ratio": 2.0,
        },
        "observation": {
            "shot_id": f"shot_{row_id}",
            "timestamp": 1_800_000_000 + row_id,
            "beverage_out_g": 36.0,
            "brew_ratio": 2.0,
            "shot_time_s": 30.0,
            "profile_resampled": profile(),
            "raw_profile_available": True,
            "profile_flow_valid": True,
            "profile_flow_masked": False,
            "profile_id": "classic_9_bar",
            "profile_type": "static",
            "profile_phase_count": 2,
            "profile_temperature_c": 93.0,
            "final_phase_temperature_c": 92.5,
            "beverage_flow_profile": [round(0.1 + index * 0.1, 4) for index in range(100)],
            "temperature_profile": [93.0 for _ in range(100)],
            "target_temperature_profile": [92.5 for _ in range(100)],
            "pump_target_mode_profile": [1 for _ in range(100)],
            "fixed_cadence_sequence": fixed_cadence_sequence(),
        },
        "reward": {
            "human_rating": 4,
            "taste_tags": ["balanced"],
            "reward": 0.8,
            "confidence": 1.0,
            "feedback_recorded": True,
            "optimization_weight": 1.0,
        },
    }


def profile() -> list[list[float]]:
    return [
        [round(1.0 + index * 0.1, 4) for index in range(100)],
        [8.0 for _ in range(100)],
        [round(1.0 + index * 0.02, 4) for index in range(100)],
        [2.4 for _ in range(100)],
        [round(index * 0.36, 4) for index in range(100)],
    ]


def fixed_cadence_sequence(step_count: int = 4) -> dict:
    return {
        "sample_interval_ms": 250,
        "pressure_bar": [float(index + 1) for index in range(step_count)],
        "pressure_target_bar": [8.0 for _ in range(step_count)],
        "pump_flow_ml_s": [round(1.0 + index * 0.02, 4) for index in range(step_count)],
        "pump_flow_target_ml_s": [2.4 for _ in range(step_count)],
        "beverage_flow_g_s": [round(0.1 + index * 0.1, 4) for index in range(step_count)],
        "weight_g": [round(index * 0.36, 4) for index in range(step_count)],
        "temperature_c": [93.0 for _ in range(step_count)],
        "temperature_target_c": [92.5 for _ in range(step_count)],
        "pump_target_mode": [1 for _ in range(step_count)],
        "valve_open": [True for _ in range(step_count)],
    }


def safetensors_metadata(payload: bytes) -> dict:
    return safetensors_header(payload)["__metadata__"]


def safetensors_header(payload: bytes) -> dict:
    header_len = int.from_bytes(payload[:8], "little")
    return json.loads(payload[8 : 8 + header_len].decode("utf-8"))


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

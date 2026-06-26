from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from espresso_rl.application.trainer_artifacts import (
    AUDIT_REPORT_FILENAME,
    CHECKSUMS_FILENAME,
    MODEL_FILENAME,
    MODEL_MANIFEST_FILENAME,
    TRAINING_CONFIG_FILENAME,
    TrainerArtifactError,
    build_dreamer_trainer_artifacts,
    DEFAULT_MAX_DATASET_BYTES,
)
from espresso_rl.domain.model_manifest import validate_model_manifest
from espresso_rl.domain.trainer_artifacts import default_training_config
from espresso_rl.trainer_cli import main as trainer_cli_main


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
        self.assertTrue(files[MODEL_FILENAME].content.startswith(len(files[MODEL_FILENAME].content[8:]).to_bytes(8, "little")))
        manifest = json.loads(files[MODEL_MANIFEST_FILENAME].content.decode("utf-8"))
        self.assertEqual(manifest["model_artifact"]["format"], "safetensors")
        self.assertEqual(manifest["model_artifact"]["sha256"], files[MODEL_FILENAME].sha256)
        self.assertEqual(manifest["dataset"]["sha256"], hashlib.sha256(dataset_text.encode("utf-8")).hexdigest())
        self.assertEqual(manifest["trainer"]["training_config_sha256"], files[TRAINING_CONFIG_FILENAME].sha256)
        self.assertFalse(manifest["runtime_compatibility"]["inference_ready"])
        manifest_validation = validate_model_manifest(
            manifest,
            expected_model_sha256=files[MODEL_FILENAME].sha256,
        )
        self.assertFalse(manifest_validation.verified)
        self.assertIn("not inference-ready", manifest_validation.unavailable_reason or "")
        audit = json.loads(files[AUDIT_REPORT_FILENAME].content.decode("utf-8"))
        self.assertFalse(audit["inference_ready"])
        self.assertIn(f"{files[MODEL_FILENAME].sha256}  {MODEL_FILENAME}", files[CHECKSUMS_FILENAME].content.decode("utf-8"))

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
            "profile_flow_valid": True,
            "profile_flow_masked": False,
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


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from espresso_rl.adapters.file_artifacts import LocalTextArtifactWriter
from espresso_rl.application.training_export import TrainingDatasetExportService
from espresso_rl.domain.community import CommunityTrainingRow
from espresso_rl.domain.training import FORBIDDEN_TRAINING_FIELD_NAMES, validate_training_transition


class TrainingDatasetExportTests(unittest.TestCase):
    def test_export_writes_plain_text_jsonl_csv_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = TrainingDatasetExportService(
                warehouse=FakeWarehouse(
                    [
                        training_row(
                            1,
                            payload_overrides={
                                "grinder_context_id": "=formula_like_context",
                                "human_rating": 4,
                                "reward": 0.8,
                                "reward_confidence": 0.7,
                                "beverage_flow_profile": [1.5 for _ in range(100)],
                                "temperature_profile": [93.0 for _ in range(100)],
                                "target_temperature_profile": [92.5 for _ in range(100)],
                                "pump_target_mode_profile": [1 for _ in range(100)],
                                "fixed_cadence_sequence": fixed_cadence_sequence(),
                            },
                        )
                    ]
                ),
                writer=LocalTextArtifactWriter(tmp),
                source_git_sha="abc123",
                clock=lambda: 1_800_000_000,
            )

            result = service.export_once(limit=10)

            self.assertEqual(result.row_count, 1)
            self.assertEqual(result.skipped_row_count, 0)
            names = {Path(file.relative_path).name for file in result.files}
            self.assertEqual(names, {"training_rows.jsonl", "training_rows.csv", "manifest.json", "README.txt"})
            self.assertFalse(any(Path(file.relative_path).suffix in {".pkl", ".pt", ".sqlite", ".parquet"} for file in result.files))

            files = {Path(file.relative_path).name: Path(file.absolute_path) for file in result.files}
            jsonl_text = files["training_rows.jsonl"].read_text(encoding="utf-8")
            csv_text = files["training_rows.csv"].read_text(encoding="utf-8")
            manifest = json.loads(files["manifest.json"].read_text(encoding="utf-8"))

        exported_row = json.loads(jsonl_text)
        self.assertEqual(exported_row["format"], "espresso_rl_training_transition_v1")
        self.assertNotIn("payload", exported_row)
        self.assertEqual(exported_row["observation"]["shot_id"], "shot_1")
        self.assertEqual(exported_row["context"]["grinder_context_id"], "=formula_like_context")
        self.assertEqual(exported_row["action"]["relative_grind_steps_from_reference"], 0.0)
        self.assertEqual(exported_row["observation"]["temperature_profile"][0], 93.0)
        self.assertEqual(exported_row["observation"]["target_temperature_profile"][0], 92.5)
        self.assertEqual(exported_row["observation"]["pump_target_mode_profile"][0], 1)
        self.assertEqual(exported_row["observation"]["beverage_flow_profile"][0], 1.5)
        self.assertEqual(exported_row["observation"]["fixed_cadence_sequence"]["sample_interval_ms"], 250)
        self.assertEqual(len(exported_row["observation"]["fixed_cadence_sequence"]["pressure_bar"]), 4)
        self.assertEqual(validate_training_transition(exported_row), [])
        self.assertIn("'=formula_like_context", csv_text)
        self.assertIn("temperature_profile_sha256", csv_text)
        self.assertIn("beverage_flow_profile_sha256", csv_text)
        self.assertIn("fixed_cadence_sequence_sha256", csv_text)
        self.assertEqual(manifest["format"], "espresso_rl_training_dataset_v1")
        self.assertEqual(manifest["canonical_row_format"], "espresso_rl_training_transition_v1")
        self.assertEqual(manifest["source_git_sha"], "abc123")
        self.assertFalse(manifest["zero_trust"]["executable_content_included"])
        self.assertFalse(manifest["zero_trust"]["adapter_payloads_included"])
        self.assertFalse(manifest["zero_trust"]["absolute_grinder_fields_included"])
        self.assertTrue(manifest["zero_trust"]["canonical_rows_revalidated"])
        self.assertTrue(manifest["zero_trust"]["canonical_transitions_only"])
        self.assertEqual(manifest["zero_trust"]["canonical_grind"], "relative_normalized_only")
        self.assertTrue(manifest["zero_trust"]["csv_formula_strings_escaped"])
        self.assertEqual(manifest["dataset_sha256"], hashlib.sha256(jsonl_text.encode("utf-8")).hexdigest())

    def test_export_skips_rows_that_fail_export_time_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = TrainingDatasetExportService(
                warehouse=FakeWarehouse(
                    [
                        training_row(1),
                        training_row(2, payload_overrides={"install_id": "spoofed_install"}),
                    ]
                ),
                writer=LocalTextArtifactWriter(tmp),
                clock=lambda: 1_800_000_000,
            )

            result = service.export_once(limit=10)

            self.assertEqual(result.row_count, 1)
            self.assertEqual(result.skipped_row_count, 1)
            self.assertIn("Skipped 1", result.warnings[0])

    def test_export_canonicalizes_absolute_display_grinder_state_without_emitting_absolute_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = TrainingDatasetExportService(
                warehouse=FakeWarehouse(
                    [
                        training_row(
                            1,
                            payload_overrides={
                                "relative_grind_steps_from_reference": None,
                                "relative_grind_um_from_reference": None,
                                "current_absolute_step": 42,
                                "absolute_reference_step": 40,
                                "step_direction": "higher_is_coarser",
                            },
                        )
                    ]
                ),
                writer=LocalTextArtifactWriter(tmp),
                clock=lambda: 1_800_000_000,
            )

            result = service.export_once(limit=10)
            files = {Path(file.relative_path).name: Path(file.absolute_path) for file in result.files}
            jsonl_text = files["training_rows.jsonl"].read_text(encoding="utf-8")

        exported_row = json.loads(jsonl_text)
        self.assertEqual(exported_row["action"]["relative_grind_steps_from_reference"], 2.0)
        self.assertEqual(exported_row["action"]["relative_grind_um_from_reference"], -25.0)
        for forbidden in FORBIDDEN_TRAINING_FIELD_NAMES:
            self.assertNotIn(forbidden, jsonl_text)

    def test_not_followed_recommendations_are_not_successful_training_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = TrainingDatasetExportService(
                warehouse=FakeWarehouse(
                    [
                        training_row(
                            1,
                            payload_overrides={
                                "recommendation_id": "rec_1",
                                "recommended_grind_delta_steps_from_current": -2,
                                "recommended_projected_relative_step_from_reference": -2,
                                "recommendation_decision": "accepted",
                                "recommendation_followed": "not_followed",
                                "recommendation_attribution_weight": 1.0,
                            },
                        )
                    ]
                ),
                writer=LocalTextArtifactWriter(tmp),
                clock=lambda: 1_800_000_000,
            )

            result = service.export_once(limit=10)
            files = {Path(file.relative_path).name: Path(file.absolute_path) for file in result.files}
            exported_row = json.loads(files["training_rows.jsonl"].read_text(encoding="utf-8"))

        self.assertEqual(result.row_count, 1)
        self.assertEqual(exported_row["recommendation"]["follow_through"], "not_followed")
        self.assertEqual(exported_row["recommendation"]["attribution_weight"], 0.0)

    def test_export_skips_noncanonical_recommendation_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = TrainingDatasetExportService(
                warehouse=FakeWarehouse(
                    [
                        training_row(
                            1,
                            payload_overrides={
                                "recommendation_id": "rec_1",
                                "recommended_grind_delta_steps_from_current": 1.5,
                            },
                        )
                    ]
                ),
                writer=LocalTextArtifactWriter(tmp),
                clock=lambda: 1_800_000_000,
            )

            result = service.export_once(limit=10)

        self.assertEqual(result.row_count, 0)
        self.assertEqual(result.skipped_row_count, 1)

    def test_export_order_is_deterministic_by_training_row_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = TrainingDatasetExportService(
                warehouse=FakeWarehouse([training_row(2), training_row(1)]),
                writer=LocalTextArtifactWriter(tmp),
                clock=lambda: 1_800_000_000,
            )

            result = service.export_once(limit=10)
            files = {Path(file.relative_path).name: Path(file.absolute_path) for file in result.files}
            lines = [
                json.loads(line)
                for line in files["training_rows.jsonl"].read_text(encoding="utf-8").splitlines()
                if line
            ]

        self.assertEqual(result.row_count, 2)
        self.assertEqual([line["observation"]["shot_id"] for line in lines], ["shot_1", "shot_2"])

    def test_file_writer_rejects_paths_outside_export_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = LocalTextArtifactWriter(tmp)

            with self.assertRaises(ValueError):
                writer.write_text("../escape.txt", "x", content_type="text/plain")


class FakeWarehouse:
    def __init__(self, rows: list[CommunityTrainingRow]) -> None:
        self.rows = rows

    def list_training_rows(self, limit: int = 5000) -> list[CommunityTrainingRow]:
        return self.rows[:limit]


def training_row(
    row_id: int,
    *,
    payload_overrides: dict[str, Any] | None = None,
    trust_weight: float = 0.2,
) -> CommunityTrainingRow:
    payload: dict[str, Any] = {
        "event_type": "shot_record",
        "schema_version": 1,
        "shot_id": f"shot_{row_id}",
        "install_id": "install_1",
        "machine_id": "machine_1",
        "timestamp": 1_800_000_000 + row_id,
        "dose_in_g": 18.0,
        "target_yield_g": 36.0,
        "beverage_out_g": 36.0,
        "target_ratio": 2.0,
        "shot_time_s": 30.0,
        "profile_temperature_c": 93.0,
        "final_phase_temperature_c": 92.5,
        "microns_per_step": 12.5,
        "relative_grind_steps_from_reference": 0,
        "relative_grind_um_from_reference": 0,
        "optimization_weight": 1.0,
        "recommendation_followed": "followed",
    }
    if payload_overrides:
        payload.update(payload_overrides)
    return CommunityTrainingRow(
        training_row_id=row_id,
        source_validation_id=row_id,
        install_id="install_1",
        payload_json=payload,
        trust_weight=trust_weight,
        payload_hash="a" * 64,
    )


def fixed_cadence_sequence() -> dict[str, Any]:
    return {
        "sample_interval_ms": 250,
        "pressure_bar": [0.0, 2.0, 5.0, 8.0],
        "pressure_target_bar": [2.0, 4.0, 8.0, 9.0],
        "pump_flow_ml_s": [0.0, 1.0, 2.0, 2.2],
        "pump_flow_target_ml_s": [0.0, 0.0, 0.0, 0.0],
        "beverage_flow_g_s": [0.0, 0.5, 1.5, 2.0],
        "weight_g": [0.0, 0.1, 0.5, 1.0],
        "temperature_c": [92.0, 92.1, 92.2, 92.3],
        "temperature_target_c": [93.0, 93.0, 93.0, 93.0],
        "pump_target_mode": [1, 1, 1, 1],
        "valve_open": [True, True, True, True],
    }


if __name__ == "__main__":
    unittest.main()

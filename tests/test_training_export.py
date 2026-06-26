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
        self.assertEqual(exported_row["format"], "espresso_rl_training_row_v1")
        self.assertEqual(exported_row["payload"]["shot_id"], "shot_1")
        self.assertIn("'=formula_like_context", csv_text)
        self.assertEqual(manifest["format"], "espresso_rl_training_dataset_v1")
        self.assertEqual(manifest["source_git_sha"], "abc123")
        self.assertFalse(manifest["zero_trust"]["executable_content_included"])
        self.assertTrue(manifest["zero_trust"]["canonical_rows_revalidated"])
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


if __name__ == "__main__":
    unittest.main()

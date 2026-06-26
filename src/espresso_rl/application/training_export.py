from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable

from espresso_rl.application.upload_validation import validate_upload_payload
from espresso_rl.domain.artifacts import ArtifactInfo
from espresso_rl.domain.community import CommunityTrainingRow
from espresso_rl.ports.artifacts import TextArtifactWriter
from espresso_rl.ports.community import CommunityWarehouseRepository

EXPORT_SCHEMA_VERSION = 1
TRAINING_ROW_FORMAT = "espresso_rl_training_row_v1"
DATASET_FORMAT = "espresso_rl_training_dataset_v1"
JSONL_FILENAME = "training_rows.jsonl"
CSV_FILENAME = "training_rows.csv"
README_FILENAME = "README.txt"
MANIFEST_FILENAME = "manifest.json"

CSV_COLUMNS = [
    "training_row_id",
    "source_validation_id",
    "install_id",
    "payload_hash",
    "trust_weight",
    "shot_id",
    "timestamp",
    "machine_id",
    "machine_adapter",
    "bean_context_id",
    "grinder_context_id",
    "profile_resampled_sha256",
    "raw_profile_hash",
    "dose_in_g",
    "target_yield_g",
    "beverage_out_g",
    "target_ratio",
    "shot_time_s",
    "microns_per_step",
    "relative_grind_steps_from_reference",
    "relative_grind_um_from_reference",
    "human_rating",
    "taste_tags",
    "reward",
    "reward_confidence",
    "recommendation_followed",
    "optimization_weight",
    "profile_flow_valid",
    "profile_flow_masked",
]


@dataclass(frozen=True)
class TrainingDatasetExportResult:
    export_id: str
    export_dir: str
    row_count: int
    skipped_row_count: int
    dataset_sha256: str
    manifest_sha256: str
    files: list[ArtifactInfo]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["files"] = [asdict(item) for item in self.files]
        return data


class TrainingDatasetExportService:
    def __init__(
        self,
        *,
        warehouse: CommunityWarehouseRepository,
        writer: TextArtifactWriter,
        source_git_sha: str = "",
        max_rows: int = 50_000,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self._warehouse = warehouse
        self._writer = writer
        self._source_git_sha = _safe_git_sha(source_git_sha)
        self._max_rows = _positive_limit(max_rows)
        self._clock = clock or (lambda: 0)

    def export_once(self, limit: int = 50_000) -> TrainingDatasetExportResult:
        limit = min(_positive_limit(limit), self._max_rows)
        rows = sorted(
            self._warehouse.list_training_rows(limit=limit),
            key=lambda row: row.training_row_id,
        )
        exported: list[dict[str, Any]] = []
        skipped = 0
        for row in rows:
            export_row = _export_training_row(row)
            if export_row is None:
                skipped += 1
                continue
            exported.append(export_row)

        jsonl_text = "".join(_canonical_json(row) + "\n" for row in exported)
        dataset_sha256 = _sha256_text(jsonl_text)
        created_at = int(self._clock())
        export_id = f"training_dataset_v{EXPORT_SCHEMA_VERSION}_{created_at}_{dataset_sha256[:12]}"
        export_dir = export_id

        csv_text = _csv_text(exported)
        readme_text = _readme_text()

        data_files = [
            self._writer.write_text(
                f"{export_dir}/{JSONL_FILENAME}",
                jsonl_text,
                content_type="application/x-ndjson; charset=utf-8",
            ),
            self._writer.write_text(
                f"{export_dir}/{CSV_FILENAME}",
                csv_text,
                content_type="text/csv; charset=utf-8",
            ),
            self._writer.write_text(
                f"{export_dir}/{README_FILENAME}",
                readme_text,
                content_type="text/plain; charset=utf-8",
            ),
        ]
        manifest_text = _canonical_json(
            _manifest(
                export_id=export_id,
                created_at=created_at,
                limit=limit,
                row_count=len(exported),
                skipped_row_count=skipped,
                dataset_sha256=dataset_sha256,
                files=data_files,
                source_git_sha=self._source_git_sha,
            )
        ) + "\n"
        manifest_file = self._writer.write_text(
            f"{export_dir}/{MANIFEST_FILENAME}",
            manifest_text,
            content_type="application/json; charset=utf-8",
        )
        return TrainingDatasetExportResult(
            export_id=export_id,
            export_dir=export_dir,
            row_count=len(exported),
            skipped_row_count=skipped,
            dataset_sha256=dataset_sha256,
            manifest_sha256=manifest_file.sha256,
            files=[*data_files, manifest_file],
            warnings=_warnings(skipped),
        )


def _export_training_row(row: CommunityTrainingRow) -> dict[str, Any] | None:
    payload = dict(row.payload_json)
    if payload.get("install_id") != row.install_id:
        return None
    if row.trust_weight <= 0:
        return None
    validation = validate_upload_payload(payload)
    if not validation.ok:
        return None
    return {
        "format": TRAINING_ROW_FORMAT,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "training_row_id": row.training_row_id,
        "source_validation_id": row.source_validation_id,
        "install_id": row.install_id,
        "payload_hash": row.payload_hash,
        "trust_weight": round(float(row.trust_weight), 6),
        "payload": payload,
    }


def _csv_text(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(_csv_row(row))
    return output.getvalue()


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = row["payload"]
    return {
        "training_row_id": row["training_row_id"],
        "source_validation_id": row["source_validation_id"],
        "install_id": _safe_csv_cell(row["install_id"]),
        "payload_hash": row.get("payload_hash") or "",
        "trust_weight": row["trust_weight"],
        "shot_id": _safe_csv_cell(payload.get("shot_id")),
        "timestamp": payload.get("timestamp"),
        "machine_id": _safe_csv_cell(payload.get("machine_id")),
        "machine_adapter": _safe_csv_cell(payload.get("machine_adapter")),
        "bean_context_id": _safe_csv_cell(payload.get("bean_context_id")),
        "grinder_context_id": _safe_csv_cell(payload.get("grinder_context_id")),
        "profile_resampled_sha256": _value_sha256(payload.get("profile_resampled")),
        "raw_profile_hash": payload.get("raw_profile_hash") or "",
        "dose_in_g": payload.get("dose_in_g"),
        "target_yield_g": payload.get("target_yield_g"),
        "beverage_out_g": payload.get("beverage_out_g"),
        "target_ratio": payload.get("target_ratio"),
        "shot_time_s": payload.get("shot_time_s"),
        "microns_per_step": payload.get("microns_per_step"),
        "relative_grind_steps_from_reference": payload.get("relative_grind_steps_from_reference"),
        "relative_grind_um_from_reference": payload.get("relative_grind_um_from_reference"),
        "human_rating": payload.get("human_rating"),
        "taste_tags": "|".join(str(tag) for tag in payload.get("taste_tags", [])),
        "reward": payload.get("reward"),
        "reward_confidence": payload.get("reward_confidence"),
        "recommendation_followed": payload.get("recommendation_followed"),
        "optimization_weight": payload.get("optimization_weight"),
        "profile_flow_valid": payload.get("profile_flow_valid"),
        "profile_flow_masked": payload.get("profile_flow_masked"),
    }


def _manifest(
    *,
    export_id: str,
    created_at: int,
    limit: int,
    row_count: int,
    skipped_row_count: int,
    dataset_sha256: str,
    files: list[ArtifactInfo],
    source_git_sha: str,
) -> dict[str, Any]:
    return {
        "format": DATASET_FORMAT,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "created_at": created_at,
        "export_id": export_id,
        "source": "validated_training_dataset",
        "source_git_sha": source_git_sha,
        "row_count": row_count,
        "skipped_row_count": skipped_row_count,
        "limit": limit,
        "dataset_sha256": dataset_sha256,
        "canonical_dataset_file": JSONL_FILENAME,
        "files": [_manifest_file_info(file) for file in files],
        "zero_trust": {
            "raw_uploads_included": False,
            "secrets_included": False,
            "executable_content_included": False,
            "canonical_rows_revalidated": True,
            "formats": ["jsonl", "csv", "json", "txt"],
            "csv_formula_strings_escaped": True,
        },
    }


def _manifest_file_info(file: ArtifactInfo) -> dict[str, Any]:
    return {
        "relative_path": file.relative_path,
        "content_type": file.content_type,
        "size_bytes": file.size_bytes,
        "sha256": file.sha256,
    }


def _readme_text() -> str:
    return (
        "EspressoRL training dataset export\n"
        "\n"
        "This export intentionally uses plain UTF-8 text files only.\n"
        "training_rows.jsonl is the canonical dataset: one JSON object per line.\n"
        "training_rows.csv is a human-inspection summary and omits profile arrays.\n"
        "manifest.json records file hashes, row counts, and provenance metadata.\n"
        "\n"
        "There are no pickles, model binaries, SQLite dumps, parquet files, macros, or executable files.\n"
        "CSV string cells that could be interpreted as spreadsheet formulas are escaped.\n"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _value_sha256(value: Any) -> str:
    if value is None:
        return ""
    return _sha256_text(_canonical_json(value))


def _safe_csv_cell(value: object) -> object:
    if value is None:
        return ""
    if not isinstance(value, str):
        return value
    if value and value[0] in {"=", "+", "-", "@", "\t", "\r"}:
        return "'" + value
    return value


def _positive_limit(limit: int) -> int:
    parsed = int(limit)
    if parsed <= 0:
        raise ValueError("training export limit must be positive")
    return parsed


def _safe_git_sha(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) > 64:
        return text[:64]
    return text


def _warnings(skipped: int) -> list[str]:
    if skipped <= 0:
        return []
    return [f"Skipped {skipped} training rows that failed export-time zero-trust validation."]

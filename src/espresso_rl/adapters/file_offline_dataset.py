from __future__ import annotations

import os
from pathlib import Path

from espresso_rl.application.offline_dataset_export import (
    OfflineDatasetExport,
    manifest_json,
)


def write_offline_dataset_export(
    export: OfflineDatasetExport,
    output_directory: Path,
    *,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    """Writes only the three documented plain-text artifact files."""

    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    records_path = output_directory / "preference_examples.jsonl"
    manifest_path = output_directory / "manifest.json"
    readme_path = output_directory / "README.txt"
    for path in (records_path, manifest_path, readme_path):
        if path.exists() and not force:
            raise FileExistsError(f"{path} already exists; pass force=True to replace it")

    _write_file(records_path, export.records_jsonl)
    _write_file(manifest_path, manifest_json(export))
    _write_file(readme_path, export.readme.encode("ascii"))
    return records_path, manifest_path, readme_path


def _write_file(path: Path, value: bytes) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)

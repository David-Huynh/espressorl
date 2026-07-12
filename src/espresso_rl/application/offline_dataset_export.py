from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from espresso_rl.domain.offline_dataset import (
    OFFLINE_DATASET_FORMAT,
    OfflinePreferenceExample,
)
from espresso_rl.ports.offline_dataset import OfflineDatasetSource


@dataclass(frozen=True)
class OfflineDatasetExport:
    records_jsonl: bytes
    manifest: dict[str, Any]
    readme: str


class OfflineDatasetExportService:
    """Builds deterministic, non-executable community preference artifacts."""

    def __init__(
        self,
        source: OfflineDatasetSource,
        *,
        clock: Callable[[], int] | None = None,
        exporter_version: str = "1",
    ) -> None:
        self._source = source
        self._clock = clock or (lambda: int(time.time()))
        self._exporter_version = str(exporter_version).strip() or "1"

    def build(self, *, limit: int | None = None) -> OfflineDatasetExport:
        if limit is not None and (isinstance(limit, bool) or not 1 <= int(limit) <= 10_000_000):
            raise ValueError("offline dataset limit must be between 1 and 10000000")
        examples = self._source.list_offline_preference_examples(
            limit=(int(limit) if limit is not None else None)
        )
        examples = sorted(
            examples,
            key=lambda item: (item.created_at, item.comparison_id),
        )
        comparison_ids = [item.comparison_id for item in examples]
        if len(comparison_ids) != len(set(comparison_ids)):
            raise ValueError("offline dataset contains duplicate comparison identities")

        records_jsonl = b"".join(
            _canonical_json(example.to_dict()).encode("ascii") + b"\n"
            for example in examples
        )
        records_sha256 = hashlib.sha256(records_jsonl).hexdigest()
        labels = {label: 0 for label in ("new_better", "anchor_better", "tie")}
        for example in examples:
            labels[example.label] += 1
        generated_at = int(self._clock())
        manifest = {
            "format": OFFLINE_DATASET_FORMAT,
            "schema_version": 1,
            "exporter_version": self._exporter_version,
            "generated_at": generated_at,
            "record_count": len(examples),
            "records_file": "preference_examples.jsonl",
            "records_content_type": "application/x-ndjson",
            "records_sha256": records_sha256,
            "label_counts": labels,
            "label_orientation": {
                "new_better": "new_shot is preferred to anchor_shot",
                "anchor_better": "anchor_shot is preferred to new_shot",
                "tie": "no noticeable difference",
            },
            "contains_scalar_ratings": False,
            "contains_executable_artifacts": False,
        }
        readme = _readme(records_sha256, len(examples))
        return OfflineDatasetExport(
            records_jsonl=records_jsonl,
            manifest=manifest,
            readme=readme,
        )


def manifest_json(export: OfflineDatasetExport) -> bytes:
    return (_canonical_json(export.manifest) + "\n").encode("ascii")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _readme(records_sha256: str, record_count: int) -> str:
    return (
        "EspressoRL offline preference dataset\n"
        "\n"
        "This directory contains plain UTF-8 JSON and no executable model or pickle files.\n"
        "Each JSONL record joins one oriented pairwise preference to the two immutable\n"
        "physical shot records used in that comparison. The new shot is always the first\n"
        "operand. A tie means no noticeable difference and is not a numeric score.\n"
        "\n"
        f"Records: {record_count}\n"
        f"preference_examples.jsonl SHA-256: {records_sha256}\n"
        "Verify with any standard SHA-256 utility before training.\n"
    )

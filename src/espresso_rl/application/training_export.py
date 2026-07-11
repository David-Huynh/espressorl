from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable

from espresso_rl.application.upload_validation import validate_upload_payload
from espresso_rl.application.upload_payloads import canonical_payload_json, shot_upload_payload
from espresso_rl.domain.artifacts import ArtifactInfo
from espresso_rl.domain.community import CommunityTrainingRow
from espresso_rl.domain.models import ShotRecord
from espresso_rl.domain.training import (
    TRAINING_DATASET_FORMAT,
    TRAINING_SCHEMA_VERSION,
    TRAINING_TRANSITION_FORMAT,
    validate_training_transition,
)
from espresso_rl.ports.artifacts import TextArtifactWriter
from espresso_rl.ports.training import TrainingRowSource

EXPORT_SCHEMA_VERSION = TRAINING_SCHEMA_VERSION
TRAINING_ROW_FORMAT = TRAINING_TRANSITION_FORMAT
DATASET_FORMAT = TRAINING_DATASET_FORMAT
JSONL_FILENAME = "training_rows.jsonl"
CSV_FILENAME = "training_rows.csv"
README_FILENAME = "README.txt"
MANIFEST_FILENAME = "manifest.json"
_INVALID_RECOMMENDATION = object()


def local_training_transition_from_shot(shot: ShotRecord) -> dict[str, Any] | None:
    if (
        shot.fixed_cadence_sequence is None
        or not shot.bean_context_id
        or not shot.grinder_context_id
        or shot.exclude_from_local_optimization
        or shot.optimization_weight <= 0
    ):
        return None
    payload = shot_upload_payload(shot)
    payload_json = canonical_payload_json(payload)
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    stable_id = int.from_bytes(
        hashlib.sha256(f"{shot.install_id}:{shot.machine_id}:{shot.shot_id}".encode("utf-8")).digest()[:8],
        "big",
    ) & ((1 << 63) - 1)
    row = CommunityTrainingRow(
        training_row_id=stable_id or 1,
        source_validation_id=stable_id or 1,
        install_id=shot.install_id,
        payload_json=payload,
        trust_weight=min(1.0, max(0.0, float(shot.optimization_weight))),
        payload_hash=digest,
    )
    transition = _training_transition_from_payload(row, payload)
    if transition is None:
        return None
    transition["source"]["source_kind"] = "local_validated_shot"
    return transition if not validate_training_transition(transition) else None

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
    "temperature_profile_sha256",
    "target_temperature_profile_sha256",
    "pump_target_mode_profile_sha256",
    "beverage_flow_profile_sha256",
    "fixed_cadence_sequence_sha256",
    "raw_profile_hash",
    "dose_g",
    "target_yield_g",
    "target_ratio",
    "relative_grind_steps_from_reference",
    "relative_grind_um_from_reference",
    "grind_observed",
    "dose_observed",
    "target_yield_observed",
    "microns_per_step",
    "step_direction",
    "beverage_out_g",
    "brew_ratio",
    "shot_time_s",
    "human_rating",
    "taste_tags",
    "reward",
    "reward_confidence",
    "optimization_weight",
    "recommendation_id",
    "recommendation_followed",
    "recommendation_attribution_weight",
    "recommended_grind_delta_steps_from_current",
    "recommended_grind_delta_um_from_current",
    "recommended_projected_relative_step_from_reference",
    "recommended_projected_relative_grind_um_from_reference",
    "recommended_dose_g",
    "recommended_target_yield_g",
    "recommended_target_ratio",
    "profile_flow_valid",
    "profile_flow_masked",
    "final_pump_target",
    "final_target_pressure",
    "final_target_flow",
    "profile_temperature_c",
    "final_phase_temperature_c",
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
        warehouse: TrainingRowSource,
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
    transition = _training_transition_from_payload(row, payload)
    if transition is None:
        return None
    if validate_training_transition(transition):
        return None
    return transition


def _training_transition_from_payload(
    row: CommunityTrainingRow,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    microns_per_step = _number(payload.get("microns_per_step"))
    if microns_per_step is None or microns_per_step <= 0:
        return None
    step_direction = str(payload.get("step_direction") or "higher_is_finer")
    if step_direction not in {"higher_is_finer", "higher_is_coarser"}:
        return None

    action_observed = _action_observed(payload)
    relative_steps = _relative_grind_steps_from_reference(payload)
    if relative_steps is None:
        action_observed["grind"] = False
        relative_steps = 0.0
    relative_um = _number(payload.get("relative_grind_um_from_reference"))
    if relative_um is None:
        relative_um = relative_steps * microns_per_step * _direction_sign(step_direction)

    dose_g = _number(payload.get("dose_in_g"))
    target_yield_g = _number(payload.get("target_yield_g"))
    if dose_g is None or target_yield_g is None:
        return None
    target_ratio = _number(payload.get("target_ratio"))
    if target_ratio is None:
        target_ratio = target_yield_g / dose_g

    recommendation = _recommendation_from_payload(payload, microns_per_step, step_direction)
    if recommendation is _INVALID_RECOMMENDATION:
        return None

    transition = {
        "format": TRAINING_ROW_FORMAT,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "training_row_id": int(row.training_row_id),
        "source": {
            "source_kind": "community_validated_shot",
            "source_validation_id": int(row.source_validation_id),
            "install_id": row.install_id,
            "payload_hash": row.payload_hash,
            "trust_weight": round(float(row.trust_weight), 6),
        },
        "context": {
            "machine_id": payload.get("machine_id"),
            "machine_adapter": payload.get("machine_adapter"),
            "bean_context_id": payload.get("bean_context_id"),
            "grinder_context_id": payload.get("grinder_context_id"),
            "microns_per_step": round(microns_per_step, 6),
            "step_direction": step_direction,
        },
        "action": {
            "relative_grind_steps_from_reference": round(relative_steps, 6),
            "relative_grind_um_from_reference": round(relative_um, 6),
            "dose_g": round(dose_g, 4),
            "target_yield_g": round(target_yield_g, 4),
            "target_ratio": round(target_ratio, 6),
            "observed": action_observed,
        },
        "recommendation": recommendation,
        "observation": {
            "shot_id": payload.get("shot_id"),
            "timestamp": int(payload.get("timestamp")),
            "beverage_out_g": _rounded(payload.get("beverage_out_g"), 4),
            "brew_ratio": _rounded(payload.get("brew_ratio"), 6),
            "shot_time_s": _rounded(payload.get("shot_time_s"), 4),
            "profile_resampled": payload.get("profile_resampled"),
            "raw_profile_available": bool(payload.get("raw_profile_available", payload.get("profile_resampled") is not None)),
            "raw_profile_hash": payload.get("raw_profile_hash"),
            "profile_score": _rounded(payload.get("profile_score"), 6),
            "profile_mse": _rounded(payload.get("profile_mse"), 6),
            "profile_flow_valid": bool(payload.get("profile_flow_valid", True)),
            "profile_flow_masked": bool(payload.get("profile_flow_masked", False)),
            "profile_id": payload.get("profile_id"),
            "profile_type": payload.get("profile_type"),
            "profile_phase_count": payload.get("profile_phase_count"),
            "final_pump_target": payload.get("final_pump_target"),
            "final_target_pressure": _rounded(payload.get("final_target_pressure"), 4),
            "final_target_flow": _rounded(payload.get("final_target_flow"), 4),
            "profile_temperature_c": _rounded(payload.get("profile_temperature_c"), 4),
            "final_phase_temperature_c": _rounded(payload.get("final_phase_temperature_c"), 4),
            "beverage_flow_profile": payload.get("beverage_flow_profile"),
            "temperature_profile": payload.get("temperature_profile"),
            "target_temperature_profile": payload.get("target_temperature_profile"),
            "pump_target_mode_profile": payload.get("pump_target_mode_profile"),
            "fixed_cadence_sequence": payload.get("fixed_cadence_sequence"),
            "shot_end_state": payload.get("shot_end_state"),
        },
        "reward": {
            "human_rating": payload.get("human_rating"),
            "taste_tags": list(payload.get("taste_tags") or []),
            "reward": _rounded(payload.get("reward"), 6),
            "confidence": round(_clamp(_number(payload.get("reward_confidence")) or 0.0, 0.0, 1.0), 6),
            "feedback_recorded": bool(payload.get("feedback_recorded", payload.get("human_rating") is not None)),
            "optimization_weight": round(_clamp(_number(payload.get("optimization_weight")) or 0.0, 0.0, 1.0), 6),
        },
    }
    return _drop_none_values(transition)


def _recommendation_from_payload(
    payload: dict[str, Any],
    microns_per_step: float,
    step_direction: str,
) -> dict[str, Any] | None | object:
    has_recommendation = any(
        payload.get(key) is not None
        for key in (
            "recommendation_id",
            "recommended_grind_delta_steps_from_current",
            "recommended_grind_delta_um_from_current",
            "recommended_projected_relative_step_from_reference",
            "recommended_dose_g",
            "recommended_target_yield_g",
            "recommended_target_ratio",
        )
    )
    if not has_recommendation:
        return None

    grind_delta_steps = _number(payload.get("recommended_grind_delta_steps_from_current"))
    if payload.get("recommended_grind_delta_steps_from_current") is not None and grind_delta_steps is None:
        return _INVALID_RECOMMENDATION

    projected_relative_step = _number(payload.get("recommended_projected_relative_step_from_reference"))
    projected_relative_um = None
    if projected_relative_step is not None:
        projected_relative_um = projected_relative_step * microns_per_step * _direction_sign(step_direction)

    return _drop_none_values(
        {
            "recommendation_id": payload.get("recommendation_id"),
            "grind_delta_steps_from_current": _rounded(grind_delta_steps, 6),
            "grind_delta_um_from_current": _rounded(payload.get("recommended_grind_delta_um_from_current"), 6),
            "projected_relative_step_from_reference": _rounded(projected_relative_step, 6),
            "projected_relative_grind_um_from_reference": _rounded(projected_relative_um, 6),
            "next_dose_g": _rounded(payload.get("recommended_dose_g"), 4),
            "target_yield_g": _rounded(payload.get("recommended_target_yield_g"), 4),
            "target_ratio": _rounded(payload.get("recommended_target_ratio"), 6),
            "decision": payload.get("recommendation_decision", "unknown"),
            "follow_through": payload.get("recommendation_followed", "unknown"),
            "attribution_weight": _recommendation_attribution_weight(payload),
            "field_trust": _drop_none_values(
                {
                    "grind": _rounded(_clamp_optional(payload.get("grind_recommendation_trust")), 6),
                    "dose": _rounded(_clamp_optional(payload.get("dose_recommendation_trust")), 6),
                    "yield": _rounded(_clamp_optional(payload.get("yield_recommendation_trust")), 6),
                }
            ),
        }
    )


def _relative_grind_steps_from_reference(payload: dict[str, Any]) -> float | None:
    relative_steps = _number(payload.get("relative_grind_steps_from_reference"))
    if relative_steps is not None:
        return relative_steps
    current_absolute_step = _number(payload.get("current_absolute_step"))
    absolute_reference_step = _number(payload.get("absolute_reference_step"))
    if current_absolute_step is None or absolute_reference_step is None:
        return None
    return current_absolute_step - absolute_reference_step


def _action_observed(payload: dict[str, Any]) -> dict[str, bool]:
    declared = payload.get("action_observed")
    if isinstance(declared, dict):
        return {
            "grind": declared.get("grind") is True,
            "dose": declared.get("dose") is True,
            "target_yield": declared.get("target_yield") is True,
        }
    return {
        "grind": _relative_grind_steps_from_reference(payload) is not None,
        "dose": _number(payload.get("dose_in_g")) is not None,
        "target_yield": _number(payload.get("target_yield_g")) is not None,
    }


def _recommendation_attribution_weight(payload: dict[str, Any]) -> float:
    decision = payload.get("recommendation_decision", "unknown")
    follow_through = payload.get("recommendation_followed", "unknown")
    if decision in {"ignored", "dismissed"} or follow_through in {"not_followed", "unknown"}:
        return 0.0
    raw_weight = _number(payload.get("recommendation_attribution_weight")) or 0.0
    weight = _clamp(raw_weight, 0.0, 1.0)
    if follow_through == "partially_followed":
        return round(min(weight, 0.5), 6)
    return round(weight, 6)


def _csv_text(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(_csv_row(row))
    return output.getvalue()


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    source = row["source"]
    context = row["context"]
    action = row["action"]
    recommendation = row.get("recommendation") or {}
    observation = row["observation"]
    reward = row["reward"]
    return {
        "training_row_id": row["training_row_id"],
        "source_validation_id": source["source_validation_id"],
        "install_id": _safe_csv_cell(source["install_id"]),
        "payload_hash": source.get("payload_hash") or "",
        "trust_weight": source["trust_weight"],
        "shot_id": _safe_csv_cell(observation.get("shot_id")),
        "timestamp": observation.get("timestamp"),
        "machine_id": _safe_csv_cell(context.get("machine_id")),
        "machine_adapter": _safe_csv_cell(context.get("machine_adapter")),
        "bean_context_id": _safe_csv_cell(context.get("bean_context_id")),
        "grinder_context_id": _safe_csv_cell(context.get("grinder_context_id")),
        "profile_resampled_sha256": _value_sha256(observation.get("profile_resampled")),
        "temperature_profile_sha256": _value_sha256(observation.get("temperature_profile")),
        "target_temperature_profile_sha256": _value_sha256(observation.get("target_temperature_profile")),
        "pump_target_mode_profile_sha256": _value_sha256(observation.get("pump_target_mode_profile")),
        "beverage_flow_profile_sha256": _value_sha256(observation.get("beverage_flow_profile")),
        "fixed_cadence_sequence_sha256": _value_sha256(observation.get("fixed_cadence_sequence")),
        "raw_profile_hash": observation.get("raw_profile_hash") or "",
        "dose_g": action.get("dose_g"),
        "target_yield_g": action.get("target_yield_g"),
        "target_ratio": action.get("target_ratio"),
        "relative_grind_steps_from_reference": action.get("relative_grind_steps_from_reference"),
        "relative_grind_um_from_reference": action.get("relative_grind_um_from_reference"),
        "grind_observed": (action.get("observed") or {}).get("grind", True),
        "dose_observed": (action.get("observed") or {}).get("dose", True),
        "target_yield_observed": (action.get("observed") or {}).get("target_yield", True),
        "microns_per_step": context.get("microns_per_step"),
        "step_direction": context.get("step_direction"),
        "beverage_out_g": observation.get("beverage_out_g"),
        "brew_ratio": observation.get("brew_ratio"),
        "shot_time_s": observation.get("shot_time_s"),
        "human_rating": reward.get("human_rating"),
        "taste_tags": "|".join(str(tag) for tag in reward.get("taste_tags", [])),
        "reward": reward.get("reward"),
        "reward_confidence": reward.get("confidence"),
        "optimization_weight": reward.get("optimization_weight"),
        "recommendation_id": _safe_csv_cell(recommendation.get("recommendation_id")),
        "recommendation_followed": recommendation.get("follow_through") or "",
        "recommendation_attribution_weight": recommendation.get("attribution_weight", ""),
        "recommended_grind_delta_steps_from_current": recommendation.get("grind_delta_steps_from_current", ""),
        "recommended_grind_delta_um_from_current": recommendation.get("grind_delta_um_from_current", ""),
        "recommended_projected_relative_step_from_reference": recommendation.get("projected_relative_step_from_reference", ""),
        "recommended_projected_relative_grind_um_from_reference": recommendation.get(
            "projected_relative_grind_um_from_reference",
            "",
        ),
        "recommended_dose_g": recommendation.get("next_dose_g", ""),
        "recommended_target_yield_g": recommendation.get("target_yield_g", ""),
        "recommended_target_ratio": recommendation.get("target_ratio", ""),
        "profile_flow_valid": observation.get("profile_flow_valid"),
        "profile_flow_masked": observation.get("profile_flow_masked"),
        "final_pump_target": observation.get("final_pump_target") or "",
        "final_target_pressure": observation.get("final_target_pressure", ""),
        "final_target_flow": observation.get("final_target_flow", ""),
        "profile_temperature_c": observation.get("profile_temperature_c", ""),
        "final_phase_temperature_c": observation.get("final_phase_temperature_c", ""),
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
        "canonical_row_format": TRAINING_ROW_FORMAT,
        "files": [_manifest_file_info(file) for file in files],
        "zero_trust": {
            "raw_uploads_included": False,
            "adapter_payloads_included": False,
            "secrets_included": False,
            "executable_content_included": False,
            "canonical_rows_revalidated": True,
            "canonical_transitions_only": True,
            "absolute_grinder_fields_included": False,
            "canonical_grind": "relative_normalized_only",
            "formats": ["jsonl", "csv", "json", "txt"],
            "csv_formula_strings_escaped": True,
            "sequence_group_keys": ["install_id", "machine_id", "bean_context_id", "grinder_context_id"],
            "dreamer_fixed_cadence_interval_ms": 250,
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
        "Each line is an espresso_rl_training_transition_v1 object with context, action,\n"
        "recommendation, observation, reward, and source sections.\n"
        "training_rows.csv is a human-inspection summary and omits profile arrays.\n"
        "manifest.json records file hashes, row counts, and provenance metadata.\n"
        "\n"
        "The JSONL file is not a raw upload dump. It excludes adapter payloads,\n"
        "absolute grinder display fields, secrets, and executable content.\n"
        "Grind is canonicalized as relative steps and relative microns from the\n"
        "grinder context reference. CSV string cells that could be interpreted as\n"
        "spreadsheet formulas are escaped.\n"
        "Dreamer rows may include fixed_cadence_sequence as named JSON arrays at\n"
        "exact 250 ms intervals; 5x100 profile_resampled remains the BO summary.\n"
        "\n"
        "There are no pickles, model binaries, SQLite dumps, parquet files, macros, or executable files.\n"
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


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return parsed


def _rounded(value: Any, digits: int) -> float | int | None:
    parsed = _number(value)
    if parsed is None:
        return None
    rounded = round(parsed, digits)
    if isinstance(value, int) and float(rounded).is_integer():
        return int(rounded)
    return rounded


def _direction_sign(step_direction: str) -> int:
    return 1 if step_direction == "higher_is_finer" else -1


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _clamp_optional(value: Any) -> float | None:
    parsed = _number(value)
    if parsed is None:
        return None
    return _clamp(parsed, 0.0, 1.0)


def _drop_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_none_values(item)
            for key, item in value.items()
            if item is not None and _drop_none_values(item) != {}
        }
    if isinstance(value, list):
        return [_drop_none_values(item) for item in value]
    return value

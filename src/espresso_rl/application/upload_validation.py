from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from espresso_rl.domain.models import VALID_TASTE_TAGS

SHA256_HEX_LENGTH = 64
SUPPORTED_SCHEMA_VERSION = 1
SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,160}$")
SAFE_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
CONTROL_OR_HTML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f<>]")

SHOT_RECORD_FIELDS = frozenset(
    {
        "event_type",
        "schema_version",
        "shot_id",
        "timestamp",
        "install_id",
        "machine_id",
        "machine_adapter",
        "bean_context_id",
        "grinder_context_id",
        "profile_resampled",
        "raw_profile_available",
        "raw_profile_hash",
        "grinder_calibration_mode",
        "microns_per_step",
        "step_direction",
        "reference_label",
        "relative_grind_steps_from_reference",
        "relative_grind_um_from_reference",
        "current_absolute_step",
        "absolute_reference_step",
        "dose_in_g",
        "beverage_out_g",
        "brew_ratio",
        "target_yield_g",
        "target_ratio",
        "shot_time_s",
        "recommendation_id",
        "recommended_grind_delta_steps_from_current",
        "recommended_grind_delta_um_from_current",
        "recommended_projected_relative_step_from_reference",
        "recommended_dose_g",
        "recommended_target_yield_g",
        "recommended_target_ratio",
        "recommendation_decision",
        "recommendation_followed",
        "recommendation_attribution_weight",
        "human_rating",
        "taste_tags",
        "feedback_recorded",
        "profile_score",
        "profile_mse",
        "reward",
        "reward_confidence",
        "shot_type",
        "exclude_from_local_optimization",
        "optimization_weight",
        "rating_prompt_allowed",
        "grind_followed",
        "dose_followed",
        "yield_followed",
        "grind_recommendation_trust",
        "dose_recommendation_trust",
        "yield_recommendation_trust",
        "weight_source",
        "flow_source",
        "flow_units",
        "pump_flow_source",
        "pump_flow_units",
        "pump_flow_calibration_required",
        "profile_flow_valid",
        "profile_flow_masked",
        "profile_id",
        "profile_label",
        "profile_type",
        "profile_phase_count",
        "final_phase_index",
        "final_phase_name",
        "final_phase_type",
        "final_phase_elapsed_s",
        "final_pump_target",
        "final_target_pressure",
        "final_target_flow",
        "final_valve_open",
        "profile_temperature_c",
        "final_phase_temperature_c",
        "shot_end_state",
        "created_at",
        "updated_at",
    }
)
RECOMMENDATION_RECORD_FIELDS = frozenset(
    {
        "event_type",
        "schema_version",
        "recommendation_id",
        "created_at",
        "updated_at",
        "expires_at",
        "install_id",
        "machine_id",
        "bean_context_id",
        "grinder_context_id",
        "grind_delta_steps_from_current",
        "grind_delta_um_from_current",
        "projected_relative_step_from_reference",
        "projected_relative_grind_um_from_reference",
        "grinder_calibration_mode",
        "microns_per_step",
        "step_direction",
        "reference_label",
        "current_absolute_step",
        "absolute_reference_step",
        "projected_absolute_step",
        "next_dose_g",
        "target_yield_g",
        "target_ratio",
        "mode",
        "confidence",
        "reason",
        "status",
        "shown_count",
        "accepted_at",
        "ignored_at",
        "edited_at",
        "used_at",
        "superseded_at",
        "source_shot_id",
        "apply_status",
        "apply_acknowledged_at",
        "applied_fields",
        "manual_fields",
        "apply_error",
    }
)


@dataclass(frozen=True)
class UploadPayloadValidation:
    ok: bool
    errors: list[str] = field(default_factory=list)


def validate_upload_payload_json(payload_json: str) -> UploadPayloadValidation:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        return UploadPayloadValidation(False, [f"invalid JSON: {exc.msg}"])
    if not isinstance(payload, dict):
        return UploadPayloadValidation(False, ["payload must be an object"])
    return validate_upload_payload(payload)


def validate_upload_envelope(
    payload_json: str,
    payload_hash: str,
    *,
    expected_install_id: str | None = None,
    max_payload_bytes: int | None = None,
) -> UploadPayloadValidation:
    errors: list[str] = []
    if max_payload_bytes is not None and len(payload_json.encode("utf-8")) > max_payload_bytes:
        errors.append("payload too large")

    hash_validation = validate_payload_hash(payload_json, payload_hash)
    errors.extend(hash_validation.errors)

    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON: {exc.msg}")
        return UploadPayloadValidation(False, errors)
    if not isinstance(payload, dict):
        errors.append("payload must be an object")
        return UploadPayloadValidation(False, errors)

    if expected_install_id is not None and payload.get("install_id") != expected_install_id:
        errors.append("payload install_id does not match upload credential")

    validation = validate_upload_payload(payload)
    errors.extend(validation.errors)
    return UploadPayloadValidation(ok=not errors, errors=errors)


def validate_payload_hash(payload_json: str, payload_hash: str) -> UploadPayloadValidation:
    errors: list[str] = []
    if not _is_sha256_hex(payload_hash):
        errors.append("payload_hash must be a sha256 hex digest")
        return UploadPayloadValidation(False, errors)
    actual = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    if actual != payload_hash.lower():
        errors.append("payload_hash does not match payload_json")
    return UploadPayloadValidation(ok=not errors, errors=errors)


def validate_canonical_payload_hash(payload: dict[str, Any], payload_hash: str) -> UploadPayloadValidation:
    if not _is_sha256_hex(payload_hash):
        return UploadPayloadValidation(False, ["payload_hash must be a sha256 hex digest"])
    try:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return UploadPayloadValidation(False, ["payload_json is not canonicalizable"])
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual != payload_hash.lower():
        return UploadPayloadValidation(False, ["payload_hash does not match canonical payload_json"])
    return UploadPayloadValidation(True, [])


def validate_upload_payload(payload: dict[str, Any]) -> UploadPayloadValidation:
    errors: list[str] = []
    event_type = payload.get("event_type")
    if event_type == "shot_record":
        _validate_shot_record(payload, errors)
    elif event_type == "recommendation_record":
        _validate_recommendation_record(payload, errors)
    else:
        errors.append("unsupported event_type")
    return UploadPayloadValidation(ok=not errors, errors=errors)


def sanitize_upload_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return an allowlisted copy for trusted storage.

    The raw upload queue keeps the original client JSON for audit/quarantine.
    Validated/training tables should only receive the canonical fields in the
    upload schema, and callers should run validation before trusting the result.
    """
    allowed = _allowed_fields(payload.get("event_type"))
    if not allowed:
        return {}
    return {key: deepcopy(payload[key]) for key in allowed if key in payload}


def mask_untrusted_profile_channels(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a trusted-storage copy with unusable flow channels masked.

    Raw uploads are intentionally preserved in the raw queue. This function is
    only for the validated/training copy after the upload credential and schema
    checks have already passed. It avoids letting corrupt or differently-scaled
    flow telemetry poison community priors or model input.
    """
    copied = dict(payload)
    profile = copied.get("profile_resampled")
    if not isinstance(profile, list) or len(profile) != 5:
        return copied
    channels = [list(channel) if isinstance(channel, list) else channel for channel in profile]
    target_flow = channels[3]
    flow = channels[2]
    flow_valid = _channel_in_range(flow, 0, 20)
    target_flow_valid = _channel_in_range(target_flow, 0, 20)
    copied["profile_flow_valid"] = flow_valid
    copied["profile_flow_masked"] = False
    if not flow_valid or not target_flow_valid:
        channels[2] = [0.0 for _ in range(100)]
        channels[3] = [0.0 for _ in range(100)]
        copied["profile_resampled"] = channels
        copied["profile_flow_masked"] = True
    return copied


def _validate_shot_record(payload: dict[str, Any], errors: list[str]) -> None:
    _reject_unknown_fields(payload, SHOT_RECORD_FIELDS, errors)
    _require_number_range(payload, "schema_version", SUPPORTED_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSION, errors)
    _require_identifier(payload, "shot_id", errors)
    _require_identifier(payload, "install_id", errors)
    _require_identifier(payload, "machine_id", errors)
    _optional_identifier(payload, "machine_adapter", errors)
    _optional_identifier(payload, "bean_context_id", errors)
    _optional_string(payload, "grinder_context_id", 120, errors)
    _optional_bool(payload, "raw_profile_available", errors)
    _optional_hash(payload, "raw_profile_hash", errors)
    _require_number_range(payload, "timestamp", 0, 9_007_199_254_740_991, errors)
    _require_number_range(payload, "dose_in_g", 5, 30, errors)
    _optional_number_range(payload, "beverage_out_g", 0, 120, errors)
    _optional_number_range(payload, "brew_ratio", 0.1, 10, errors)
    _require_number_range(payload, "target_yield_g", 5, 100, errors)
    _optional_number_range(payload, "target_ratio", 1.2, 3.5, errors)
    _optional_number_range(payload, "shot_time_s", 0, 180, errors)
    _optional_enum(
        payload,
        "grinder_calibration_mode",
        {"uncalibrated", "relative_calibrated", "absolute_display_calibrated"},
        errors,
    )
    _optional_enum(payload, "step_direction", {"higher_is_finer", "higher_is_coarser"}, errors)
    _optional_string(payload, "reference_label", 80, errors)
    _optional_number_range(payload, "microns_per_step", 0.1, 100, errors)
    _optional_number_range(payload, "relative_grind_steps_from_reference", -10_000, 10_000, errors)
    _optional_number_range(payload, "relative_grind_um_from_reference", -1_000_000, 1_000_000, errors)
    _optional_number_range(payload, "current_absolute_step", -10_000, 10_000, errors)
    _optional_number_range(payload, "absolute_reference_step", -10_000, 10_000, errors)
    _optional_identifier(payload, "recommendation_id", errors)
    _optional_number_range(payload, "recommended_grind_delta_steps_from_current", -1000, 1000, errors)
    _optional_number_range(payload, "recommended_grind_delta_um_from_current", -100_000, 100_000, errors)
    _optional_number_range(payload, "recommended_projected_relative_step_from_reference", -10_000, 10_000, errors)
    _optional_number_range(payload, "recommended_dose_g", 5, 30, errors)
    _optional_number_range(payload, "recommended_target_yield_g", 5, 100, errors)
    _optional_number_range(payload, "recommended_target_ratio", 1.2, 3.5, errors)
    _optional_enum(payload, "recommendation_decision", {"accepted", "edited", "ignored", "dismissed", "unknown"}, errors)
    _optional_enum(payload, "recommendation_followed", {"followed", "partially_followed", "not_followed", "unknown"}, errors)
    _optional_number_range(payload, "human_rating", 1, 5, errors)
    _optional_bool(payload, "feedback_recorded", errors)
    _optional_number_range(payload, "profile_score", 0, 1, errors)
    _optional_number_range(payload, "profile_mse", 0, 1_000_000, errors)
    _optional_number_range(payload, "reward", 0, 1, errors)
    _optional_number_range(payload, "optimization_weight", 0, 1, errors)
    _optional_number_range(payload, "recommendation_attribution_weight", 0, 1, errors)
    _optional_number_range(payload, "grind_recommendation_trust", 0, 1, errors)
    _optional_number_range(payload, "dose_recommendation_trust", 0, 1, errors)
    _optional_number_range(payload, "yield_recommendation_trust", 0, 1, errors)
    _optional_number_range(payload, "reward_confidence", 0, 1, errors)
    _optional_bool(payload, "exclude_from_local_optimization", errors)
    _optional_bool(payload, "rating_prompt_allowed", errors)
    _optional_bool(payload, "grind_followed", errors)
    _optional_bool(payload, "dose_followed", errors)
    _optional_bool(payload, "yield_followed", errors)
    _optional_bool(payload, "pump_flow_calibration_required", errors)
    _optional_bool(payload, "profile_flow_valid", errors)
    _optional_bool(payload, "profile_flow_masked", errors)
    _optional_string(payload, "weight_source", 80, errors)
    _optional_string(payload, "flow_source", 80, errors)
    _optional_string(payload, "flow_units", 40, errors)
    _optional_string(payload, "pump_flow_source", 80, errors)
    _optional_string(payload, "pump_flow_units", 40, errors)
    _optional_string(payload, "profile_id", 120, errors)
    _optional_string(payload, "profile_label", 120, errors)
    _optional_string(payload, "profile_type", 80, errors)
    _optional_int_range(payload, "profile_phase_count", 0, 100, errors)
    _optional_int_range(payload, "final_phase_index", 0, 100, errors)
    _optional_string(payload, "final_phase_name", 120, errors)
    _optional_enum(payload, "final_phase_type", {"preinfusion", "brew"}, errors)
    _optional_number_range(payload, "final_phase_elapsed_s", 0, 600, errors)
    _optional_enum(payload, "final_pump_target", {"simple", "pressure", "flow"}, errors)
    _optional_number_range(payload, "final_target_pressure", 0, 15, errors)
    _optional_number_range(payload, "final_target_flow", 0, 25, errors)
    _optional_bool(payload, "final_valve_open", errors)
    _optional_number_range(payload, "profile_temperature_c", 0, 160, errors)
    _optional_number_range(payload, "final_phase_temperature_c", 0, 160, errors)
    _optional_enum(payload, "shot_end_state", {"finished", "manual_or_interrupted", "unknown"}, errors)
    _optional_string_list_enum(payload, "taste_tags", VALID_TASTE_TAGS, errors)
    _optional_number_range(payload, "created_at", 0, 9_007_199_254_740_991, errors)
    _optional_number_range(payload, "updated_at", 0, 9_007_199_254_740_991, errors)
    _optional_enum(
        payload,
        "shot_type",
        {"espresso", "utility_flush", "cleaning", "calibration", "unknown"},
        errors,
    )
    if payload.get("shot_type") is not None and payload.get("shot_type") != "espresso":
        errors.append("non-espresso shot uploads are not trusted training shots")
        errors.append("shot_type must be espresso")
    profile = payload.get("profile_resampled")
    if profile is not None:
        _validate_profile_resampled(profile, payload.get("beverage_out_g"), errors)


def _validate_recommendation_record(payload: dict[str, Any], errors: list[str]) -> None:
    _reject_unknown_fields(payload, RECOMMENDATION_RECORD_FIELDS, errors)
    _require_number_range(payload, "schema_version", SUPPORTED_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSION, errors)
    _require_identifier(payload, "recommendation_id", errors)
    _require_identifier(payload, "install_id", errors)
    _require_identifier(payload, "machine_id", errors)
    _optional_identifier(payload, "bean_context_id", errors)
    _optional_string(payload, "grinder_context_id", 120, errors)
    _optional_enum(
        payload,
        "grinder_calibration_mode",
        {"uncalibrated", "relative_calibrated", "absolute_display_calibrated"},
        errors,
    )
    _optional_enum(payload, "step_direction", {"higher_is_finer", "higher_is_coarser"}, errors)
    _optional_string(payload, "reference_label", 80, errors)
    _optional_number_range(payload, "microns_per_step", 0.1, 100, errors)
    _optional_number_range(payload, "grind_delta_steps_from_current", -1000, 1000, errors)
    _optional_number_range(payload, "projected_relative_step_from_reference", -10_000, 10_000, errors)
    _optional_number_range(payload, "projected_relative_grind_um_from_reference", -1_000_000, 1_000_000, errors)
    _optional_number_range(payload, "current_absolute_step", -10_000, 10_000, errors)
    _optional_number_range(payload, "absolute_reference_step", -10_000, 10_000, errors)
    _optional_number_range(payload, "projected_absolute_step", -10_000, 10_000, errors)
    _require_number_range(payload, "next_dose_g", 5, 30, errors)
    _require_number_range(payload, "target_yield_g", 5, 100, errors)
    _require_number_range(payload, "target_ratio", 1.2, 3.5, errors)
    _optional_number_range(payload, "grind_delta_um_from_current", -100_000, 100_000, errors)
    _optional_enum(
        payload,
        "mode",
        {"zero_observe", "zero_immediate_bo", "warm_started_bo", "local_bo", "dreamer_candidate", "dreamer_active", "bo_fallback"},
        errors,
    )
    _optional_number_range(payload, "confidence", 0, 1, errors)
    _optional_string(payload, "reason", 500, errors)
    _optional_enum(payload, "status", {"pending", "shown", "accepted", "edited", "ignored", "expired", "used", "superseded"}, errors)
    _optional_int_range(payload, "shown_count", 0, 1_000_000, errors)
    _optional_number_range(payload, "created_at", 0, 9_007_199_254_740_991, errors)
    _optional_number_range(payload, "updated_at", 0, 9_007_199_254_740_991, errors)
    _optional_number_range(payload, "expires_at", 0, 9_007_199_254_740_991, errors)
    _optional_number_range(payload, "accepted_at", 0, 9_007_199_254_740_991, errors)
    _optional_number_range(payload, "ignored_at", 0, 9_007_199_254_740_991, errors)
    _optional_number_range(payload, "edited_at", 0, 9_007_199_254_740_991, errors)
    _optional_number_range(payload, "used_at", 0, 9_007_199_254_740_991, errors)
    _optional_number_range(payload, "superseded_at", 0, 9_007_199_254_740_991, errors)
    _optional_identifier(payload, "source_shot_id", errors)
    _optional_enum(payload, "apply_status", {"unknown", "applied", "partially_applied", "manual_required", "failed"}, errors)
    _optional_number_range(payload, "apply_acknowledged_at", 0, 9_007_199_254_740_991, errors)
    _optional_object(payload, "applied_fields", errors)
    _optional_string_list(payload, "manual_fields", errors)
    _optional_string(payload, "apply_error", 500, errors)


def _validate_profile_resampled(profile: Any, beverage_out_g: Any, errors: list[str]) -> None:
    ranges = [
        (0, 15, "pressure"),
        (0, 15, "target_pressure"),
        (0, 20, "flow"),
        (0, 20, "target_flow"),
        (-1, 120, "weight"),
    ]
    if not isinstance(profile, list) or len(profile) != 5:
        errors.append("profile_resampled must have 5 channels")
        return
    for channel_index, (minimum, maximum, label) in enumerate(ranges):
        channel = profile[channel_index]
        if not isinstance(channel, list) or len(channel) != 100:
            errors.append(f"profile_resampled {label} channel must have exactly 100 samples")
            continue
        if not _channel_numeric_finite(channel):
            errors.append(f"profile_resampled {label} contains non-finite or nonnumeric values")
            continue
        if not _channel_in_range(channel, minimum, maximum):
            if label in {"flow", "target_flow"}:
                continue
            errors.append(f"profile_resampled {label} out of range")
    weight = profile[4]
    if isinstance(beverage_out_g, (int, float)) and isinstance(weight, list) and len(weight) == 100:
        final_weight = weight[-1]
        if isinstance(final_weight, (int, float)) and abs(float(final_weight) - float(beverage_out_g)) > 5:
            errors.append("final profile weight does not match beverage_out_g")


def _require_string(payload: dict[str, Any], key: str, errors: list[str], max_len: int = 160) -> None:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} is required")
        return
    if len(value) > max_len or CONTROL_OR_HTML_RE.search(value):
        errors.append(f"{key} contains unsafe characters")


def _require_identifier(payload: dict[str, Any], key: str, errors: list[str]) -> None:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} is required")
        return
    if not SAFE_IDENTIFIER_RE.fullmatch(value):
        errors.append(f"{key} contains unsafe characters")


def _require_number_range(
    payload: dict[str, Any],
    key: str,
    minimum: float,
    maximum: float,
    errors: list[str],
) -> None:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(f"{key} out of range")
        return
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        errors.append(f"{key} out of range")


def _optional_number_range(
    payload: dict[str, Any],
    key: str,
    minimum: float,
    maximum: float,
    errors: list[str],
) -> None:
    if payload.get(key) is None:
        return
    _require_number_range(payload, key, minimum, maximum, errors)


def _optional_int_range(
    payload: dict[str, Any],
    key: str,
    minimum: int,
    maximum: int,
    errors: list[str],
) -> None:
    value = payload.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{key} out of range")
        return
    if not minimum <= value <= maximum:
        errors.append(f"{key} out of range")


def _optional_bool(payload: dict[str, Any], key: str, errors: list[str]) -> None:
    if payload.get(key) is None:
        return
    if not isinstance(payload.get(key), bool):
        errors.append(f"{key} must be boolean")


def _optional_string(payload: dict[str, Any], key: str, max_len: int, errors: list[str]) -> None:
    value = payload.get(key)
    if value is None:
        return
    if not isinstance(value, str) or len(value) > max_len or CONTROL_OR_HTML_RE.search(value):
        errors.append(f"{key} must be a short string")


def _optional_identifier(payload: dict[str, Any], key: str, errors: list[str]) -> None:
    value = payload.get(key)
    if value is None:
        return
    if not isinstance(value, str) or not SAFE_IDENTIFIER_RE.fullmatch(value):
        errors.append(f"{key} contains unsafe characters")


def _optional_hash(payload: dict[str, Any], key: str, errors: list[str]) -> None:
    value = payload.get(key)
    if value is None:
        return
    if not isinstance(value, str) or len(value) != SHA256_HEX_LENGTH or not SAFE_HEX_RE.fullmatch(value):
        errors.append(f"{key} must be a sha256 hex digest")


def _optional_enum(
    payload: dict[str, Any],
    key: str,
    allowed: set[str],
    errors: list[str],
) -> None:
    if payload.get(key) is None:
        return
    if not isinstance(payload.get(key), str) or payload.get(key) not in allowed:
        errors.append(f"{key} is invalid")


def _optional_string_list_enum(
    payload: dict[str, Any],
    key: str,
    allowed: set[str],
    errors: list[str],
) -> None:
    value = payload.get(key)
    if value is None:
        return
    if not isinstance(value, list):
        errors.append(f"{key} must be a list")
        return
    invalid = [item for item in value if not isinstance(item, str) or item not in allowed]
    if invalid:
        errors.append(f"{key} contains invalid values")


def _optional_string_list(payload: dict[str, Any], key: str, errors: list[str], max_len: int = 80) -> None:
    value = payload.get(key)
    if value is None:
        return
    if not isinstance(value, list):
        errors.append(f"{key} must be a list")
        return
    for item in value:
        if not isinstance(item, str) or len(item) > max_len or CONTROL_OR_HTML_RE.search(item):
            errors.append(f"{key} contains invalid values")
            return


def _optional_object(payload: dict[str, Any], key: str, errors: list[str]) -> None:
    value = payload.get(key)
    if value is None:
        return
    if not isinstance(value, dict) or len(value) > 25:
        errors.append(f"{key} must be an object")
        return
    for item_key, item_value in value.items():
        if not isinstance(item_key, str) or len(item_key) > 80 or CONTROL_OR_HTML_RE.search(item_key):
            errors.append(f"{key} contains unsafe keys")
            return
        if item_value is None or isinstance(item_value, (str, int, float, bool)):
            if isinstance(item_value, str) and (len(item_value) > 160 or CONTROL_OR_HTML_RE.search(item_value)):
                errors.append(f"{key} contains unsafe values")
                return
            if isinstance(item_value, (int, float)) and not isinstance(item_value, bool) and not math.isfinite(float(item_value)):
                errors.append(f"{key} contains unsafe values")
                return
            continue
        errors.append(f"{key} contains unsafe values")
        return


def _reject_unknown_fields(payload: dict[str, Any], allowed: frozenset[str], errors: list[str]) -> None:
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        errors.append(f"unknown fields: {', '.join(unknown[:10])}")


def _allowed_fields(event_type: object) -> frozenset[str]:
    if event_type == "shot_record":
        return SHOT_RECORD_FIELDS
    if event_type == "recommendation_record":
        return RECOMMENDATION_RECORD_FIELDS
    return frozenset()


def _is_sha256_hex(value: str) -> bool:
    if not isinstance(value, str) or len(value) != SHA256_HEX_LENGTH:
        return False
    return all(char in "0123456789abcdef" for char in value.lower())


def _channel_in_range(channel: Any, minimum: float, maximum: float) -> bool:
    if not _channel_numeric_finite(channel):
        return False
    for value in channel:
        parsed = float(value)
        if not minimum <= parsed <= maximum:
            return False
    return True


def _channel_numeric_finite(channel: Any) -> bool:
    if not isinstance(channel, list) or len(channel) != 100:
        return False
    for value in channel:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        parsed = float(value)
        if not math.isfinite(parsed):
            return False
    return True

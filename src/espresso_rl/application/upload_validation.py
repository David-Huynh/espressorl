from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from espresso_rl.domain.models import VALID_TASTE_TAGS


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


def _validate_shot_record(payload: dict[str, Any], errors: list[str]) -> None:
    _require_string(payload, "shot_id", errors)
    _require_string(payload, "install_id", errors)
    _require_string(payload, "machine_id", errors)
    _require_number_range(payload, "timestamp", 0, 9_007_199_254_740_991, errors)
    _require_number_range(payload, "dose_in_g", 5, 30, errors)
    _optional_number_range(payload, "beverage_out_g", 5, 100, errors)
    _require_number_range(payload, "target_yield_g", 5, 100, errors)
    _optional_number_range(payload, "target_ratio", 1.2, 3.5, errors)
    _optional_number_range(payload, "shot_time_s", 5, 90, errors)
    _optional_number_range(payload, "human_rating", 1, 5, errors)
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
    _optional_string_list_enum(payload, "taste_tags", VALID_TASTE_TAGS, errors)
    _optional_enum(
        payload,
        "shot_type",
        {"espresso", "utility_flush", "cleaning", "calibration", "unknown"},
        errors,
    )
    profile = payload.get("profile_resampled")
    if profile is not None:
        _validate_profile_resampled(profile, payload.get("beverage_out_g"), errors)


def _validate_recommendation_record(payload: dict[str, Any], errors: list[str]) -> None:
    _require_string(payload, "recommendation_id", errors)
    _require_string(payload, "install_id", errors)
    _require_string(payload, "machine_id", errors)
    _require_number_range(payload, "next_dose_g", 5, 30, errors)
    _require_number_range(payload, "target_yield_g", 5, 100, errors)
    _require_number_range(payload, "target_ratio", 1.2, 3.5, errors)


def _validate_profile_resampled(profile: Any, beverage_out_g: Any, errors: list[str]) -> None:
    ranges = [
        (0, 15, "pressure"),
        (0, 15, "target_pressure"),
        (0, 20, "flow"),
        (0, 20, "target_flow"),
        (0, 100, "weight"),
    ]
    if not isinstance(profile, list) or len(profile) != 5:
        errors.append("profile_resampled must have 5 channels")
        return
    for channel_index, (minimum, maximum, label) in enumerate(ranges):
        channel = profile[channel_index]
        if not isinstance(channel, list) or len(channel) != 100:
            errors.append(f"profile_resampled {label} channel must have exactly 100 samples")
            continue
        for value in channel:
            if not isinstance(value, (int, float)) or not minimum <= float(value) <= maximum:
                errors.append(f"profile_resampled {label} out of range")
                break
    weight = profile[4]
    if isinstance(beverage_out_g, (int, float)) and isinstance(weight, list) and len(weight) == 100:
        final_weight = weight[-1]
        if isinstance(final_weight, (int, float)) and abs(float(final_weight) - float(beverage_out_g)) > 5:
            errors.append("final profile weight does not match beverage_out_g")


def _require_string(payload: dict[str, Any], key: str, errors: list[str]) -> None:
    if not isinstance(payload.get(key), str) or not str(payload.get(key)).strip():
        errors.append(f"{key} is required")


def _require_number_range(
    payload: dict[str, Any],
    key: str,
    minimum: float,
    maximum: float,
    errors: list[str],
) -> None:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or not minimum <= float(value) <= maximum:
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


def _optional_bool(payload: dict[str, Any], key: str, errors: list[str]) -> None:
    if payload.get(key) is None:
        return
    if not isinstance(payload.get(key), bool):
        errors.append(f"{key} must be boolean")


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

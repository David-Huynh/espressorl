from __future__ import annotations

import hashlib
import json
from typing import Any

from espresso_rl.domain.community import PairwiseShotComparison
from espresso_rl.domain.models import (
    Recommendation,
    ShotRecord,
    UploadQueueItem,
    UploadQueueStatus,
)


def make_shot_upload_item(shot: ShotRecord, now: int) -> UploadQueueItem:
    payload = shot_upload_payload(shot)
    return _make_item("shot", shot.shot_id, payload, now)


def make_recommendation_upload_item(
    recommendation: Recommendation,
    now: int,
) -> UploadQueueItem:
    payload = recommendation_upload_payload(recommendation)
    upload_id = f"recommendation_{recommendation.recommendation_id}"
    return _make_item("recommendation", recommendation.recommendation_id, payload, now, upload_id=upload_id)


def make_comparison_upload_item(
    comparison: PairwiseShotComparison,
    now: int,
) -> UploadQueueItem:
    payload = comparison_upload_payload(comparison)
    return _make_item("comparison", comparison.comparison_id, payload, now)


def shot_upload_payload(shot: ShotRecord) -> dict[str, Any]:
    return {
        "event_type": "shot_record",
        "schema_version": 1,
        "shot_id": shot.shot_id,
        "timestamp": shot.timestamp,
        "install_id": shot.install_id,
        "machine_id": shot.machine_id,
        "machine_adapter": shot.machine_adapter,
        "bean_context_id": shot.bean_context_id,
        "grinder_context_id": shot.grinder_context_id,
        "profile_resampled": shot.profile.round(4).tolist(),
        "raw_profile_available": shot.raw_profile_available,
        "raw_profile_hash": shot.raw_profile_hash,
        "grinder_calibration_mode": shot.grinder_calibration_mode.value,
        "grinder_adjustment_mode": shot.grinder_adjustment_mode.value,
        "microns_per_step": shot.microns_per_step,
        "step_direction": shot.grinder_step_direction.value,
        "reference_label": shot.grinder_reference_label,
        "relative_grind_steps_from_reference": shot.relative_grind_steps_from_reference,
        "relative_grind_um_from_reference": shot.relative_grind_um_from_reference,
        "current_absolute_step": shot.current_absolute_step,
        "absolute_reference_step": shot.absolute_reference_step,
        "action_observed": shot.action_observed,
        "dose_in_g": shot.dose_in_g,
        "beverage_out_g": shot.beverage_out_g,
        "brew_ratio": shot.brew_ratio,
        "target_yield_g": shot.target_yield_g,
        "target_ratio": shot.target_ratio,
        "shot_time_s": shot.shot_time_s,
        "recommendation_id": shot.recommendation_id,
        "recommended_grind_delta_steps_from_current": shot.recommended_grind_delta_steps_from_current,
        "recommended_grind_delta_um_from_current": shot.recommended_grind_delta_um_from_current,
        "recommended_projected_relative_step_from_reference": shot.recommended_projected_relative_step_from_reference,
        "recommended_dose_g": shot.recommended_dose_g,
        "recommended_target_yield_g": shot.recommended_target_yield_g,
        "recommended_target_ratio": shot.recommended_target_ratio,
        "recommendation_decision": shot.recommendation_decision.value,
        "recommendation_followed": shot.recommendation_followed.value,
        "shot_type": shot.shot_type.value,
        "exclude_from_local_optimization": shot.exclude_from_local_optimization,
        "grind_followed": shot.grind_followed,
        "dose_followed": shot.dose_followed,
        "yield_followed": shot.yield_followed,
        "weight_source": shot.weight_source,
        "flow_source": shot.flow_source,
        "flow_units": shot.flow_units,
        "pump_flow_source": shot.pump_flow_source,
        "pump_flow_units": shot.pump_flow_units,
        "pump_flow_calibration_required": shot.pump_flow_calibration_required,
        "profile_flow_valid": shot.profile_flow_valid,
        "profile_flow_masked": shot.profile_flow_masked,
        "profile_id": shot.profile_id,
        "profile_label": shot.profile_label,
        "profile_type": shot.profile_type,
        "profile_phase_count": shot.profile_phase_count,
        "final_phase_index": shot.final_phase_index,
        "final_phase_name": shot.final_phase_name,
        "final_phase_type": shot.final_phase_type,
        "final_phase_elapsed_s": shot.final_phase_elapsed_s,
        "final_pump_target": shot.final_pump_target,
        "final_target_pressure": shot.final_target_pressure,
        "final_target_flow": shot.final_target_flow,
        "final_valve_open": shot.final_valve_open,
        "profile_temperature_c": shot.profile_temperature_c,
        "final_phase_temperature_c": shot.final_phase_temperature_c,
        "beverage_flow_profile": _optional_array(shot.beverage_flow_profile, ndigits=3),
        "temperature_profile": _optional_array(shot.temperature_profile, ndigits=3),
        "target_temperature_profile": _optional_array(shot.target_temperature_profile, ndigits=3),
        "pump_target_mode_profile": _optional_int_array(shot.pump_target_mode_profile),
        "fixed_cadence_sequence": (
            shot.fixed_cadence_sequence.to_dict(ndigits=4)
            if shot.fixed_cadence_sequence is not None
            else None
        ),
        "shot_end_state": shot.shot_end_state,
        "created_at": shot.created_at,
        "updated_at": shot.updated_at,
    }


def recommendation_upload_payload(recommendation: Recommendation) -> dict[str, Any]:
    return {
        "event_type": "recommendation_record",
        "schema_version": 1,
        "recommendation_id": recommendation.recommendation_id,
        "created_at": recommendation.created_at,
        "updated_at": recommendation.updated_at,
        "expires_at": recommendation.expires_at,
        "install_id": recommendation.install_id,
        "machine_id": recommendation.machine_id,
        "bean_context_id": recommendation.bean_context_id,
        "grinder_context_id": recommendation.grinder_context_id,
        "profile_id": recommendation.profile_id,
        "raw_profile_hash": recommendation.raw_profile_hash,
        "grind_delta_steps_from_current": recommendation.grind_delta_steps_from_current,
        "grind_delta_um_from_current": recommendation.grind_delta_um_from_current,
        "projected_relative_step_from_reference": recommendation.projected_relative_step_from_reference,
        "projected_relative_grind_um_from_reference": recommendation.projected_relative_grind_um_from_reference,
        "grinder_calibration_mode": recommendation.grinder_calibration_mode.value,
        "grinder_adjustment_mode": recommendation.grinder_adjustment_mode.value,
        "microns_per_step": (
            abs(recommendation.grind_delta_um_from_current / recommendation.grind_delta_steps_from_current)
            if recommendation.grind_delta_steps_from_current
            else (
                abs(recommendation.projected_relative_grind_um_from_reference / recommendation.projected_relative_step_from_reference)
                if recommendation.projected_relative_step_from_reference
                else None
            )
        ),
        "step_direction": recommendation.grinder_step_direction.value,
        "reference_label": recommendation.grinder_reference_label,
        "current_absolute_step": recommendation.current_absolute_step,
        "absolute_reference_step": recommendation.absolute_reference_step,
        "projected_absolute_step": recommendation.projected_absolute_step,
        "next_dose_g": recommendation.next_dose_g,
        "target_yield_g": recommendation.target_yield_g,
        "target_ratio": recommendation.target_ratio,
        "mode": recommendation.mode.value,
        "confidence": recommendation.confidence,
        "reason": recommendation.reason,
        "status": recommendation.status.value,
        "shown_count": recommendation.shown_count,
        "accepted_at": recommendation.accepted_at,
        "ignored_at": recommendation.ignored_at,
        "edited_at": recommendation.edited_at,
        "used_at": recommendation.used_at,
        "superseded_at": recommendation.superseded_at,
        "source_shot_id": recommendation.source_shot_id,
        "optimization_run_id": recommendation.optimization_run_id,
        "comparison_anchor_shot_id": recommendation.comparison_anchor_shot_id,
        "comparison_mode": recommendation.comparison_mode,
        "preference_feedback_required": recommendation.preference_feedback_required,
        "apply_status": recommendation.apply_status.value,
        "apply_acknowledged_at": recommendation.apply_acknowledged_at,
        "applied_fields": dict(recommendation.applied_fields),
        "manual_fields": list(recommendation.manual_fields),
        "apply_error": recommendation.apply_error,
    }


def comparison_upload_payload(comparison: PairwiseShotComparison) -> dict[str, Any]:
    return {
        "event_type": "comparison_record",
        "schema_version": 1,
        "comparison_id": comparison.comparison_id,
        "optimization_run_id": comparison.optimization_run_id,
        "new_shot_id": comparison.new_shot_id,
        "anchor_shot_id": comparison.anchor_shot_id,
        "label": comparison.label,
        "comparison_mode": comparison.comparison_mode,
        "created_at": comparison.created_at,
        "install_id": comparison.install_id,
        "machine_id": comparison.machine_id,
        "machine_adapter": comparison.machine_adapter,
        "recommendation_id": comparison.recommendation_id,
        "bean_context_id": comparison.bean_context_id,
        "grinder_context_id": comparison.grinder_context_id,
        "profile_id": comparison.profile_id,
        "raw_profile_hash": comparison.raw_profile_hash,
    }


def canonical_payload_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_hash(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _optional_array(value: Any, *, ndigits: int) -> list[float] | None:
    if value is None:
        return None
    return [round(float(item), ndigits) for item in value]


def _optional_int_array(value: Any) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in value]


def _make_item(
    record_type: str,
    record_id: str,
    payload: dict[str, Any],
    now: int,
    *,
    upload_id: str | None = None,
) -> UploadQueueItem:
    payload_json = canonical_payload_json(payload)
    digest = payload_hash(payload_json)
    return UploadQueueItem(
        upload_id=upload_id or f"{record_type}_{record_id}_{digest[:16]}",
        local_record_type=record_type,
        local_record_id=record_id,
        payload_hash=digest,
        payload_json=payload_json,
        status=UploadQueueStatus.PENDING,
        created_at=now,
        updated_at=now,
    )

from __future__ import annotations

import hashlib
import json
from typing import Any

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
    return _make_item("recommendation", recommendation.recommendation_id, payload, now)


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
        "profile_resampled": shot.profile.round(4).tolist(),
        "raw_profile_available": shot.raw_profile_available,
        "raw_profile_hash": shot.raw_profile_hash,
        "grind_steps": shot.grind_steps,
        "grind_um": shot.grind_um,
        "grinder_step_size_um": shot.grinder_step_size_um,
        "dose_in_g": shot.dose_in_g,
        "beverage_out_g": shot.beverage_out_g,
        "brew_ratio": shot.brew_ratio,
        "target_yield_g": shot.target_yield_g,
        "target_ratio": shot.target_ratio,
        "shot_time_s": shot.shot_time_s,
        "recommendation_id": shot.recommendation_id,
        "recommended_grind_delta_steps": shot.recommended_grind_delta_steps,
        "recommended_grind_delta_um": shot.recommended_grind_delta_um,
        "recommended_next_grind_steps": shot.recommended_next_grind_steps,
        "recommended_dose_g": shot.recommended_dose_g,
        "recommended_target_yield_g": shot.recommended_target_yield_g,
        "recommended_target_ratio": shot.recommended_target_ratio,
        "recommendation_decision": shot.recommendation_decision.value,
        "recommendation_followed": shot.recommendation_followed.value,
        "recommendation_attribution_weight": shot.recommendation_attribution_weight,
        "human_rating": shot.human_rating,
        "taste_tags": list(shot.taste_tags),
        "profile_score": shot.profile_score,
        "profile_mse": shot.profile_mse,
        "reward": shot.reward,
        "reward_confidence": shot.reward_confidence,
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
        "grind_delta_steps": recommendation.grind_delta_steps,
        "grind_delta_um": recommendation.grind_delta_um,
        "next_grind_steps": recommendation.next_grind_steps,
        "next_grind_um": recommendation.next_grind_um,
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
        "apply_status": recommendation.apply_status.value,
        "apply_acknowledged_at": recommendation.apply_acknowledged_at,
        "applied_fields": dict(recommendation.applied_fields),
        "manual_fields": list(recommendation.manual_fields),
        "apply_error": recommendation.apply_error,
    }


def canonical_payload_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_hash(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _make_item(
    record_type: str,
    record_id: str,
    payload: dict[str, Any],
    now: int,
) -> UploadQueueItem:
    payload_json = canonical_payload_json(payload)
    digest = payload_hash(payload_json)
    return UploadQueueItem(
        upload_id=f"{record_type}_{record_id}_{digest[:16]}",
        local_record_type=record_type,
        local_record_id=record_id,
        payload_hash=digest,
        payload_json=payload_json,
        status=UploadQueueStatus.PENDING,
        created_at=now,
        updated_at=now,
    )

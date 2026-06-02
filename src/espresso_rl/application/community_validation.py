from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from espresso_rl.application.upload_validation import (
    mask_untrusted_profile_channels,
    validate_upload_payload,
)
from espresso_rl.domain.community import (
    CommunityAbuseEvent,
    CommunityInstallStats,
    CommunityRawUpload,
    CommunityRecommendationRecord,
    CommunityValidatedShot,
    InstallTrustScore,
)
from espresso_rl.ports.community import CommunityWarehouseRepository


MAX_COMMUNITY_TRUST_SCORE = 0.35
MAX_COMMUNITY_TRAINING_WEIGHT = 0.25


@dataclass(frozen=True)
class CommunityValidationResult:
    processed: int
    validated_shots: int
    stored_recommendations: int
    rejected: int
    training_rows: int


class CommunityValidationService:
    def __init__(self, warehouse: CommunityWarehouseRepository) -> None:
        self._warehouse = warehouse

    def validate_once(self, limit: int = 100, *, dry_run: bool = False) -> CommunityValidationResult:
        uploads = self._warehouse.list_raw_uploads(status="mirrored", limit=limit)
        validated_shots = 0
        stored_recommendations = 0
        rejected = 0
        training_rows = 0

        for upload in uploads:
            outcome = self._validate_upload(upload, mutate=not dry_run)
            if outcome.errors:
                if not dry_run:
                    self._reject_upload(upload, outcome.errors)
                rejected += 1
                continue

            if upload.event_type == "shot_record":
                validation_id = None
                if not dry_run:
                    validation_id = self._store_validated_shot(upload, outcome.payload)
                if outcome.trust_weight > 0:
                    if validation_id is not None:
                        self._warehouse.upsert_training_row(
                            validation_id,
                            outcome.payload,
                            outcome.trust_weight,
                        )
                    training_rows += 1
                validated_shots += 1
            elif upload.event_type == "recommendation_record":
                if not dry_run:
                    self._store_recommendation(upload, outcome.payload)
                stored_recommendations += 1

        return CommunityValidationResult(
            processed=len(uploads),
            validated_shots=validated_shots,
            stored_recommendations=stored_recommendations,
            rejected=rejected,
            training_rows=training_rows,
        )

    def _validate_upload(self, upload: CommunityRawUpload, *, mutate: bool) -> "_ValidationOutcome":
        errors: list[str] = []
        payload = dict(upload.payload_json)

        if payload.get("event_type") != upload.event_type:
            errors.append("payload event_type does not match raw upload event_type")

        payload_install_id = payload.get("install_id")
        if payload_install_id != upload.install_id:
            errors.append("payload install_id does not match verified upload credential")

        # Ownership for trusted storage is always taken from the verified raw
        # queue row, never from client-controlled JSON.
        payload["install_id"] = upload.install_id

        validation = validate_upload_payload(payload)
        errors.extend(validation.errors)
        if not errors and upload.event_type == "shot_record":
            payload = mask_untrusted_profile_channels(payload)

        if upload.event_type == "shot_record":
            shot_type = payload.get("shot_type", "espresso")
            if shot_type != "espresso":
                errors.append("non-espresso shot uploads are not trusted training shots")

        if errors:
            return _ValidationOutcome(payload=payload, errors=errors, trust_weight=0.0)

        stats = self._warehouse.install_stats(upload.install_id)
        install_trust = install_trust_score(
            CommunityInstallStats(
                validated_shots=stats.validated_shots + (1 if upload.event_type == "shot_record" else 0),
                rejected_uploads=stats.rejected_uploads,
                abuse_events=stats.abuse_events,
            )
        )
        if mutate:
            self._warehouse.upsert_install_trust_score(
                InstallTrustScore(
                    install_id=upload.install_id,
                    trust_score=install_trust,
                    reason="validated_upload_history",
                )
            )
        trust_weight = (
            payload_trust_weight(payload, install_trust)
            if upload.event_type == "shot_record"
            else 0.0
        )
        return _ValidationOutcome(payload=payload, errors=[], trust_weight=trust_weight)

    def _store_validated_shot(self, upload: CommunityRawUpload, payload: dict[str, Any]) -> int:
        stats = self._warehouse.install_stats(upload.install_id)
        install_trust = install_trust_score(
            CommunityInstallStats(
                validated_shots=stats.validated_shots + 1,
                rejected_uploads=stats.rejected_uploads,
                abuse_events=stats.abuse_events,
            )
        )
        trust_weight = payload_trust_weight(payload, install_trust)
        summary = validation_summary(payload, install_trust, trust_weight)
        shot = CommunityValidatedShot(
            install_id=upload.install_id,
            upload_id=upload.upload_id,
            shot_id=str(payload["shot_id"]),
            payload_json=payload,
            trust_weight=trust_weight,
            validation_summary=summary,
        )
        validation_id = self._warehouse.upsert_validated_shot(shot)
        self._warehouse.mark_raw_upload_validated(upload, summary)
        return validation_id

    def _store_recommendation(self, upload: CommunityRawUpload, payload: dict[str, Any]) -> None:
        recommendation = CommunityRecommendationRecord(
            install_id=upload.install_id,
            upload_id=upload.upload_id,
            recommendation_id=str(payload["recommendation_id"]),
            payload_json=payload,
        )
        self._warehouse.upsert_community_recommendation(recommendation)
        self._warehouse.mark_raw_upload_validated(
            upload,
            {
                "event_type": "recommendation_record",
                "training_eligible": False,
            },
        )

    def _reject_upload(self, upload: CommunityRawUpload, errors: list[str]) -> None:
        self._warehouse.mark_raw_upload_rejected(upload, errors)
        self._warehouse.record_abuse_event(
            CommunityAbuseEvent(
                install_id=upload.install_id,
                upload_id=upload.upload_id,
                payload_hash=upload.payload_hash,
                reason="community_upload_validation_failed",
                detail={"errors": errors[:20], "event_type": upload.event_type},
            )
        )
        stats = self._warehouse.install_stats(upload.install_id)
        self._warehouse.upsert_install_trust_score(
            InstallTrustScore(
                install_id=upload.install_id,
                trust_score=install_trust_score(
                    CommunityInstallStats(
                        validated_shots=stats.validated_shots,
                        rejected_uploads=stats.rejected_uploads + 1,
                        abuse_events=stats.abuse_events + 1,
                    )
                ),
                reason="community_upload_validation_failed",
            )
        )


@dataclass(frozen=True)
class _ValidationOutcome:
    payload: dict[str, Any]
    errors: list[str]
    trust_weight: float


def install_trust_score(stats: CommunityInstallStats) -> float:
    score = 0.05 + min(stats.validated_shots, 25) * 0.012
    score -= min(0.30, stats.rejected_uploads * 0.035 + stats.abuse_events * 0.05)
    return _clamp(score, 0.0, MAX_COMMUNITY_TRUST_SCORE)


def payload_trust_weight(payload: dict[str, Any], install_trust: float) -> float:
    if payload.get("exclude_from_local_optimization") is True:
        return 0.0
    optimization_weight = _optional_float(payload.get("optimization_weight"), default=1.0)
    if optimization_weight <= 0:
        return 0.0

    quality = optimization_weight
    if payload.get("human_rating") is None:
        quality *= 0.4
    quality *= _optional_float(payload.get("reward_confidence"), default=0.5)

    followed = payload.get("recommendation_followed")
    if followed == "not_followed":
        quality *= 0.2
    elif followed == "partially_followed":
        quality *= 0.6
    elif followed == "unknown":
        quality *= 0.5

    taste_tags = payload.get("taste_tags")
    if isinstance(taste_tags, list) and "channeling_suspected" in taste_tags:
        quality *= 0.7
    if payload.get("profile_flow_masked") is True or payload.get("profile_flow_valid") is False:
        quality *= 0.7

    return round(_clamp(install_trust * quality, 0.0, MAX_COMMUNITY_TRAINING_WEIGHT), 6)


def validation_summary(
    payload: dict[str, Any],
    install_trust: float,
    trust_weight: float,
) -> dict[str, Any]:
    return {
        "event_type": payload.get("event_type"),
        "training_eligible": trust_weight > 0,
        "trust_weight": trust_weight,
        "install_trust_score": round(install_trust, 6),
        "rating_present": payload.get("human_rating") is not None,
        "recommendation_followed": payload.get("recommendation_followed"),
        "optimization_weight": _optional_float(payload.get("optimization_weight"), default=1.0),
        "profile_flow_valid": payload.get("profile_flow_valid", True),
        "profile_flow_masked": payload.get("profile_flow_masked", False),
    }


def _optional_float(value: Any, *, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))

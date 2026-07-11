from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from espresso_rl.application.upload_validation import (
    mask_untrusted_profile_channels,
    sanitize_upload_payload,
    validate_canonical_payload_hash,
    validate_upload_payload,
)
from espresso_rl.domain.community import (
    CommunityAbuseEvent,
    CommunityComparisonRecord,
    CommunityInstallStats,
    CommunityRawUpload,
    CommunityRecommendationRecord,
    CommunityValidatedShot,
    InstallTrustScore,
)
from espresso_rl.ports.community import CommunityWarehouseRepository


MAX_COMMUNITY_TRUST_SCORE = 0.35
MAX_COMMUNITY_MODELING_WEIGHT = 0.25


@dataclass(frozen=True)
class CommunityValidationResult:
    processed: int
    validated_shots: int
    stored_recommendations: int
    stored_comparisons: int
    rejected: int


class CommunityValidationService:
    def __init__(self, warehouse: CommunityWarehouseRepository) -> None:
        self._warehouse = warehouse

    def validate_once(self, limit: int = 100, *, dry_run: bool = False) -> CommunityValidationResult:
        uploads = self._warehouse.list_raw_uploads(status="mirrored", limit=limit)
        validated_shots = 0
        stored_recommendations = 0
        stored_comparisons = 0
        rejected = 0

        for upload in uploads:
            outcome = self._validate_upload(upload, mutate=not dry_run)
            if outcome.errors:
                if not dry_run:
                    self._reject_upload(upload, outcome.errors)
                rejected += 1
                continue

            try:
                if upload.event_type == "shot_record":
                    if not dry_run:
                        self._store_validated_shot(upload, outcome.payload)
                    validated_shots += 1
                elif upload.event_type == "recommendation_record":
                    if not dry_run:
                        self._store_recommendation(upload, outcome.payload)
                    stored_recommendations += 1
                elif upload.event_type == "comparison_record":
                    if not dry_run:
                        self._store_comparison(upload, outcome.payload, outcome.trust_weight)
                    stored_comparisons += 1
            except ValueError as exc:
                if not dry_run:
                    self._reject_upload(upload, [str(exc)])
                rejected += 1

        return CommunityValidationResult(
            processed=len(uploads),
            validated_shots=validated_shots,
            stored_recommendations=stored_recommendations,
            stored_comparisons=stored_comparisons,
            rejected=rejected,
        )

    def _validate_upload(self, upload: CommunityRawUpload, *, mutate: bool) -> "_ValidationOutcome":
        errors: list[str] = []
        payload = dict(upload.payload_json)

        if payload.get("event_type") != upload.event_type:
            errors.append("payload event_type does not match raw upload event_type")

        payload_install_id = payload.get("install_id")
        if payload_install_id != upload.install_id:
            errors.append("payload install_id does not match verified upload credential")

        hash_validation = validate_canonical_payload_hash(payload, upload.payload_hash)
        errors.extend(hash_validation.errors)

        # Ownership for trusted storage is always taken from the verified raw
        # queue row, never from client-controlled JSON.
        payload["install_id"] = upload.install_id

        validation = validate_upload_payload(payload)
        errors.extend(validation.errors)
        if not errors:
            payload = sanitize_upload_payload(payload)
            if upload.event_type == "shot_record":
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
        if upload.event_type == "shot_record":
            trust_weight = payload_trust_weight(payload, install_trust)
        elif upload.event_type == "comparison_record":
            trust_weight = round(min(install_trust, MAX_COMMUNITY_MODELING_WEIGHT), 6)
        else:
            trust_weight = 0.0
        return _ValidationOutcome(payload=payload, errors=[], trust_weight=trust_weight)

    def _store_validated_shot(self, upload: CommunityRawUpload, payload: dict[str, Any]) -> None:
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
        self._warehouse.upsert_validated_shot(shot)
        self._warehouse.mark_raw_upload_validated(upload, summary)

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
                "modeling_eligible": False,
            },
        )

    def _store_comparison(
        self,
        upload: CommunityRawUpload,
        payload: dict[str, Any],
        trust_weight: float,
    ) -> None:
        summary = {
            "event_type": "comparison_record",
            "modeling_eligible": trust_weight > 0,
            "label_type": "pairwise_preference",
            "trust_weight": trust_weight,
        }
        comparison = CommunityComparisonRecord(
            install_id=upload.install_id,
            upload_id=upload.upload_id,
            comparison_id=str(payload["comparison_id"]),
            payload_json=payload,
            trust_weight=trust_weight,
            validation_summary=summary,
        )
        self._warehouse.upsert_community_comparison(comparison)
        self._warehouse.mark_raw_upload_validated(
            upload,
            summary,
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
    quality = 1.0
    if payload.get("profile_flow_masked") is True or payload.get("profile_flow_valid") is False:
        quality *= 0.7

    return round(_clamp(install_trust * quality, 0.0, MAX_COMMUNITY_MODELING_WEIGHT), 6)


def validation_summary(
    payload: dict[str, Any],
    install_trust: float,
    trust_weight: float,
) -> dict[str, Any]:
    return {
        "event_type": payload.get("event_type"),
        "modeling_eligible": trust_weight > 0,
        "trust_weight": trust_weight,
        "install_trust_score": round(install_trust, 6),
        "recommendation_followed": payload.get("recommendation_followed"),
        "action_observed": payload.get("action_observed"),
        "profile_flow_valid": payload.get("profile_flow_valid", True),
        "profile_flow_masked": payload.get("profile_flow_masked", False),
    }


def _optional_float(value: Any, *, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))

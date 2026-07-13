from __future__ import annotations

import unittest
from typing import Any

from espresso_rl.application.community_validation import (
    CommunityValidationService,
    install_trust_score,
    payload_trust_weight,
)
from espresso_rl.application.upload_payloads import canonical_payload_json, payload_hash
from espresso_rl.domain.community import (
    CommunityAbuseEvent,
    CommunityComparisonRecord,
    CommunityInstallStats,
    CommunityRawUpload,
    CommunityRecommendationRecord,
    CommunityValidatedShot,
    InstallTrustScore,
)


class CommunityValidationTests(unittest.TestCase):
    def test_valid_shot_upload_moves_to_validated_physical_shots(self) -> None:
        upload = raw_upload(payload=shot_payload())
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once(limit=10)

        self.assertEqual(result.processed, 1)
        self.assertEqual(result.validated_shots, 1)
        self.assertEqual(result.rejected, 0)
        self.assertEqual(warehouse.statuses[upload.upload_id], "validated")
        self.assertEqual(warehouse.validated_shots[0].install_id, "verified_install")
        self.assertGreater(warehouse.validated_shots[0].trust_weight, 0)
        self.assertLessEqual(warehouse.validated_shots[0].trust_weight, 0.25)

    def test_dry_run_validation_reports_proposed_changes_without_mutating_state(self) -> None:
        upload = raw_upload(payload=shot_payload())
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once(limit=10, dry_run=True)

        self.assertEqual(result.processed, 1)
        self.assertEqual(result.validated_shots, 1)
        self.assertEqual(result.rejected, 0)
        self.assertEqual(warehouse.statuses[upload.upload_id], "mirrored")
        self.assertEqual(warehouse.validated_shots, [])
        self.assertEqual(warehouse.trust_scores, [])

    def test_dry_run_rejection_does_not_mark_raw_row_or_log_abuse(self) -> None:
        payload = shot_payload()
        payload["install_id"] = "spoofed_install"
        upload = raw_upload(payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once(limit=10, dry_run=True)

        self.assertEqual(result.rejected, 1)
        self.assertEqual(warehouse.statuses[upload.upload_id], "mirrored")
        self.assertEqual(warehouse.rejections, {})
        self.assertEqual(warehouse.abuse_events, [])

    def test_payload_install_id_mismatch_is_rejected(self) -> None:
        payload = shot_payload()
        payload["install_id"] = "spoofed_install"
        upload = raw_upload(payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.rejected, 1)
        self.assertEqual(warehouse.validated_shots, [])
        self.assertEqual(warehouse.statuses[upload.upload_id], "rejected")
        self.assertIn("install_id", warehouse.rejections[upload.upload_id][0])
        self.assertEqual(warehouse.abuse_events[0].reason, "community_upload_validation_failed")

    def test_payload_hash_mismatch_is_rejected(self) -> None:
        upload = raw_upload(payload=shot_payload(), payload_hash_override="0" * 64)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.rejected, 1)
        self.assertEqual(warehouse.validated_shots, [])
        self.assertEqual(warehouse.statuses[upload.upload_id], "rejected")
        self.assertIn("payload_hash", " ".join(warehouse.rejections[upload.upload_id]))

    def test_unknown_payload_fields_are_rejected_before_trusted_storage(self) -> None:
        payload = shot_payload()
        payload["debug_html"] = "<script>alert(1)</script>"
        upload = raw_upload(payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.rejected, 1)
        self.assertEqual(warehouse.validated_shots, [])
        self.assertIn("unknown fields", " ".join(warehouse.rejections[upload.upload_id]))

    def test_malformed_action_observation_mask_is_rejected(self) -> None:
        payload = shot_payload()
        payload["action_observed"] = {
            "grind": "yes",
            "dose": True,
            "target_yield": True,
            "unexpected": False,
        }
        upload = raw_upload(payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.rejected, 1)
        self.assertEqual(warehouse.validated_shots, [])
        rejection = " ".join(warehouse.rejections[upload.upload_id])
        self.assertIn("action_observed", rejection)

    def test_xss_like_metadata_string_is_rejected_before_trusted_storage(self) -> None:
        payload = shot_payload()
        payload["profile_label"] = "<img src=x onerror=alert(1)>"
        upload = raw_upload(payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.rejected, 1)
        self.assertEqual(warehouse.validated_shots, [])
        self.assertIn("profile_label", " ".join(warehouse.rejections[upload.upload_id]))

    def test_wrong_schema_version_is_rejected_before_trusted_storage(self) -> None:
        payload = shot_payload()
        payload["schema_version"] = 2
        upload = raw_upload(payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.rejected, 1)
        self.assertEqual(warehouse.validated_shots, [])
        self.assertIn("schema_version", " ".join(warehouse.rejections[upload.upload_id]))

    def test_utility_upload_is_rejected_before_trusted_storage(self) -> None:
        payload = shot_payload()
        payload["shot_type"] = "utility_flush"
        upload = raw_upload(payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.rejected, 1)
        self.assertEqual(warehouse.validated_shots, [])
        self.assertIn("non-espresso", warehouse.rejections[upload.upload_id][0])

    def test_excluded_espresso_shot_is_validated_with_zero_modeling_weight(self) -> None:
        payload = shot_payload()
        payload["exclude_from_local_optimization"] = True
        upload = raw_upload(payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.validated_shots, 1)
        self.assertEqual(warehouse.validated_shots[0].trust_weight, 0.0)

    def test_recommendation_upload_is_stored_separately_from_physical_shots(self) -> None:
        upload = raw_upload(
            event_type="recommendation_record",
            payload={
                "event_type": "recommendation_record",
                "schema_version": 1,
                "recommendation_id": "rec_1",
                "install_id": "verified_install",
                "machine_id": "machine_1",
                "taste_goal": balanced_taste_goal(),
                "created_at": 1_779_999_000,
                "updated_at": 1_779_999_000,
                "next_dose_g": 18.0,
                "target_yield_g": 38.0,
                "target_ratio": 2.111,
            },
        )
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.stored_recommendations, 1)
        self.assertEqual(warehouse.recommendations[0].recommendation_id, "rec_1")
        self.assertEqual(warehouse.statuses[upload.upload_id], "validated")

    def test_pairwise_comparison_is_stored_without_scalar_taste_score(self) -> None:
        payload = comparison_payload()
        upload = raw_upload(event_type="comparison_record", payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.stored_comparisons, 1)
        self.assertEqual(warehouse.comparisons[0].comparison_id, "comparison_1")
        self.assertEqual(warehouse.comparisons[0].payload_json["label"], "new_better")
        self.assertEqual(warehouse.statuses[upload.upload_id], "validated")

    def test_comparison_rejects_reversed_identity_and_unknown_scalar_rating(self) -> None:
        payload = comparison_payload(anchor_shot_id="candidate", human_rating=5)
        upload = raw_upload(event_type="comparison_record", payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.rejected, 1)
        errors = " ".join(warehouse.rejections[upload.upload_id])
        self.assertIn("unknown fields", errors)
        self.assertIn("distinct physical shots", errors)

    def test_scalar_rating_is_not_part_of_community_shot_contract(self) -> None:
        payload = shot_payload()
        payload["human_rating"] = 4
        upload = raw_upload(payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.rejected, 1)
        self.assertIn("unknown fields", " ".join(warehouse.rejections[upload.upload_id]))

    def test_physical_shot_rejects_optimizer_comparison_metadata(self) -> None:
        payload = shot_payload()
        payload.update(
            {
                "optimization_run_id": "run_1",
                "comparison_anchor_shot_id": "anchor_1",
                "comparison_mode": "best_incumbent",
                "preference_feedback_required": True,
            }
        )
        upload = raw_upload(payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.rejected, 1)
        self.assertIn("unknown fields", " ".join(warehouse.rejections[upload.upload_id]))

    def test_immutable_shot_identity_conflict_is_rejected(self) -> None:
        upload = raw_upload(payload=shot_payload())
        warehouse = FakeWarehouse([upload])
        warehouse.store_error = "duplicate shot_id conflicts with an immutable physical shot"

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.validated_shots, 0)
        self.assertEqual(result.rejected, 1)
        self.assertEqual(warehouse.statuses[upload.upload_id], "rejected")

    def test_immutable_comparison_identity_conflict_is_rejected(self) -> None:
        upload = raw_upload(event_type="comparison_record", payload=comparison_payload())
        warehouse = FakeWarehouse([upload])
        warehouse.store_error = "comparison_id conflicts with an immutable oriented comparison"

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.stored_comparisons, 0)
        self.assertEqual(result.rejected, 1)
        self.assertEqual(warehouse.statuses[upload.upload_id], "rejected")

    def test_bad_profile_values_are_rejected(self) -> None:
        payload = shot_payload()
        payload["profile_resampled"][0][10] = 99.0
        upload = raw_upload(payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.rejected, 1)
        self.assertIn("pressure out of range", " ".join(warehouse.rejections[upload.upload_id]))

    def test_invalid_inactive_flow_is_masked_before_trusted_storage(self) -> None:
        payload = shot_payload()
        payload["profile_resampled"][2] = [100_000.0 for _ in range(100)]
        payload["profile_resampled"][3] = [0.0 for _ in range(100)]
        upload = raw_upload(payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.validated_shots, 1)
        stored = warehouse.validated_shots[0].payload_json
        self.assertEqual(stored["profile_resampled"][2], [0.0 for _ in range(100)])
        self.assertFalse(stored["profile_flow_valid"])
        self.assertTrue(stored["profile_flow_masked"])
        self.assertTrue(warehouse.validated_summaries[upload.upload_id]["profile_flow_masked"])

    def test_invalid_active_flow_pair_is_masked_before_trusted_storage(self) -> None:
        payload = shot_payload()
        payload["profile_resampled"][2] = [100_000.0 for _ in range(100)]
        payload["profile_resampled"][3] = [2.0 for _ in range(100)]
        upload = raw_upload(payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.validated_shots, 1)
        stored = warehouse.validated_shots[0].payload_json
        self.assertEqual(stored["profile_resampled"][2], [0.0 for _ in range(100)])
        self.assertEqual(stored["profile_resampled"][3], [0.0 for _ in range(100)])
        self.assertFalse(stored["profile_flow_valid"])
        self.assertTrue(stored["profile_flow_masked"])
        self.assertTrue(warehouse.validated_summaries[upload.upload_id]["profile_flow_masked"])

    def test_nonfinite_flow_is_rejected_not_masked(self) -> None:
        payload = shot_payload()
        payload["profile_resampled"][2] = [float("nan") for _ in range(100)]
        upload = raw_upload(payload=payload, payload_hash_override="0" * 64)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.rejected, 1)
        self.assertIn("non-finite", " ".join(warehouse.rejections[upload.upload_id]))

    def test_community_trust_remains_capped_and_penalized_by_rejections(self) -> None:
        self.assertEqual(
            install_trust_score(
                CommunityInstallStats(validated_shots=100, rejected_uploads=0, abuse_events=0)
            ),
            0.35,
        )
        self.assertEqual(
            install_trust_score(
                CommunityInstallStats(validated_shots=10, rejected_uploads=20, abuse_events=20)
            ),
            0.0,
        )

    def test_not_followed_payload_keeps_actual_shot_modeling_weight(self) -> None:
        payload = shot_payload()
        followed_weight = payload_trust_weight(payload, install_trust=0.35)
        payload["recommendation_followed"] = "not_followed"
        weight = payload_trust_weight(payload, install_trust=0.35)
        self.assertEqual(weight, followed_weight)


def raw_upload(
    payload: dict[str, Any],
    *,
    event_type: str = "shot_record",
    upload_id: str = "upload_1",
    payload_hash_override: str | None = None,
) -> CommunityRawUpload:
    digest = payload_hash_override
    if digest is None:
        digest = payload_hash(canonical_payload_json(payload))
    return CommunityRawUpload(
        install_id="verified_install",
        upload_id=upload_id,
        payload_hash=digest,
        event_type=event_type,
        payload_json=payload,
    )


def shot_payload() -> dict[str, Any]:
    profile = [[0.0 for _ in range(100)] for _ in range(5)]
    profile[0] = [9.0 for _ in range(100)]
    profile[1] = [9.0 for _ in range(100)]
    profile[2] = [2.0 for _ in range(100)]
    profile[3] = [2.0 for _ in range(100)]
    profile[4] = [i * 0.38 for i in range(100)]
    profile[4][-1] = 38.0
    return {
        "event_type": "shot_record",
        "schema_version": 1,
        "shot_id": "shot_1",
        "timestamp": 1_779_999_000,
        "install_id": "verified_install",
        "machine_id": "machine_1",
        "machine_adapter": "gaggimate",
        "bean_context_id": "bean_1",
        "taste_goal": balanced_taste_goal(),
        "profile_resampled": profile,
        "dose_in_g": 18.0,
        "dose_target_g": 18.0,
        "beverage_out_g": 38.0,
        "target_yield_g": 38.0,
        "target_ratio": 2.111,
        "shot_time_s": 30.0,
        "profile_temperature_c": 93.0,
        "final_phase_temperature_c": 92.5,
        "recommendation_followed": "followed",
        "shot_type": "espresso",
        "exclude_from_local_optimization": False,
    }


def comparison_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "event_type": "comparison_record",
        "schema_version": 1,
        "comparison_id": "comparison_1",
        "optimization_run_id": "run_1",
        "new_shot_id": "candidate",
        "anchor_shot_id": "anchor",
        "label": "new_better",
        "comparison_mode": "best_incumbent",
        "created_at": 1_779_999_100,
        "install_id": "verified_install",
        "machine_id": "machine_1",
        "machine_adapter": "gaggimate",
        "bean_context_id": "bean_1",
        "grinder_context_id": "grinder_1",
        "profile_id": "profile_1",
        "taste_goal": balanced_taste_goal(),
    }
    payload.update(overrides)
    return payload


def balanced_taste_goal() -> dict[str, Any]:
    return {"schema_version": 1, "mode": "balanced", "targets": {}}


class FakeWarehouse:
    def __init__(self, uploads: list[CommunityRawUpload]) -> None:
        self.uploads = uploads
        self.statuses: dict[str, str] = {upload.upload_id: "mirrored" for upload in uploads}
        self.rejections: dict[str, list[str]] = {}
        self.validated_shots: list[CommunityValidatedShot] = []
        self.recommendations: list[CommunityRecommendationRecord] = []
        self.comparisons: list[CommunityComparisonRecord] = []
        self.trust_scores: list[InstallTrustScore] = []
        self.abuse_events: list[CommunityAbuseEvent] = []
        self.validated_summaries: dict[str, dict[str, Any]] = {}
        self.store_error: str | None = None

    def upsert_raw_upload(self, upload: CommunityRawUpload) -> None:
        self.uploads.append(upload)
        self.statuses[upload.upload_id] = "mirrored"

    def list_raw_uploads(self, status: str = "mirrored", limit: int = 100) -> list[CommunityRawUpload]:
        return [upload for upload in self.uploads if self.statuses[upload.upload_id] == status][:limit]

    def mark_raw_upload_validated(
        self,
        upload: CommunityRawUpload,
        validation_summary: dict[str, Any],
    ) -> None:
        self.statuses[upload.upload_id] = "validated"
        self.validated_summaries[upload.upload_id] = validation_summary

    def mark_raw_upload_rejected(
        self,
        upload: CommunityRawUpload,
        validation_errors: list[str],
    ) -> None:
        self.statuses[upload.upload_id] = "rejected"
        self.rejections[upload.upload_id] = validation_errors

    def upsert_validated_shot(self, shot: CommunityValidatedShot) -> int:
        if self.store_error:
            raise ValueError(self.store_error)
        self.validated_shots.append(shot)
        return len(self.validated_shots)

    def upsert_community_recommendation(self, recommendation: CommunityRecommendationRecord) -> None:
        self.recommendations.append(recommendation)

    def upsert_community_comparison(self, comparison: CommunityComparisonRecord) -> None:
        if self.store_error:
            raise ValueError(self.store_error)
        self.comparisons.append(comparison)

    def upsert_install_trust_score(self, score: InstallTrustScore) -> None:
        self.trust_scores.append(score)

    def install_stats(self, install_id: str) -> CommunityInstallStats:
        return CommunityInstallStats(
            validated_shots=len([shot for shot in self.validated_shots if shot.install_id == install_id]),
            rejected_uploads=len(
                [
                    upload
                    for upload in self.uploads
                    if upload.install_id == install_id and self.statuses[upload.upload_id] == "rejected"
                ]
            ),
            abuse_events=len([event for event in self.abuse_events if event.install_id == install_id]),
        )

    def record_abuse_event(self, event: CommunityAbuseEvent) -> None:
        self.abuse_events.append(event)

if __name__ == "__main__":
    unittest.main()

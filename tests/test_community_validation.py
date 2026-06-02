from __future__ import annotations

import unittest
from typing import Any

from espresso_rl.application.community_validation import (
    CommunityValidationService,
    install_trust_score,
    payload_trust_weight,
)
from espresso_rl.domain.community import (
    CommunityAbuseEvent,
    CommunityInstallStats,
    CommunityRawUpload,
    CommunityRecommendationRecord,
    CommunityValidatedShot,
    InstallTrustScore,
)


HASH = "f" * 64


class CommunityValidationTests(unittest.TestCase):
    def test_valid_shot_upload_moves_to_validated_and_training_tables(self) -> None:
        upload = raw_upload(payload=shot_payload())
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once(limit=10)

        self.assertEqual(result.processed, 1)
        self.assertEqual(result.validated_shots, 1)
        self.assertEqual(result.training_rows, 1)
        self.assertEqual(result.rejected, 0)
        self.assertEqual(warehouse.statuses[upload.upload_id], "validated")
        self.assertEqual(warehouse.validated_shots[0].install_id, "verified_install")
        self.assertGreater(warehouse.validated_shots[0].trust_weight, 0)
        self.assertLessEqual(warehouse.validated_shots[0].trust_weight, 0.25)
        self.assertEqual(warehouse.training_rows[0][0], 1)

    def test_dry_run_validation_reports_proposed_changes_without_mutating_state(self) -> None:
        upload = raw_upload(payload=shot_payload())
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once(limit=10, dry_run=True)

        self.assertEqual(result.processed, 1)
        self.assertEqual(result.validated_shots, 1)
        self.assertEqual(result.training_rows, 1)
        self.assertEqual(result.rejected, 0)
        self.assertEqual(warehouse.statuses[upload.upload_id], "mirrored")
        self.assertEqual(warehouse.validated_shots, [])
        self.assertEqual(warehouse.training_rows, [])
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
        self.assertEqual(warehouse.training_rows, [])
        self.assertEqual(warehouse.statuses[upload.upload_id], "rejected")
        self.assertIn("install_id", warehouse.rejections[upload.upload_id][0])
        self.assertEqual(warehouse.abuse_events[0].reason, "community_upload_validation_failed")

    def test_utility_upload_is_rejected_before_trusted_storage(self) -> None:
        payload = shot_payload()
        payload["shot_type"] = "utility_flush"
        upload = raw_upload(payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.rejected, 1)
        self.assertEqual(warehouse.validated_shots, [])
        self.assertEqual(warehouse.training_rows, [])
        self.assertIn("non-espresso", warehouse.rejections[upload.upload_id][0])

    def test_excluded_espresso_shot_is_validated_but_not_training_weighted(self) -> None:
        payload = shot_payload()
        payload["exclude_from_local_optimization"] = True
        payload["optimization_weight"] = 0.0
        upload = raw_upload(payload=payload)
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.validated_shots, 1)
        self.assertEqual(result.training_rows, 0)
        self.assertEqual(warehouse.validated_shots[0].trust_weight, 0.0)
        self.assertEqual(warehouse.training_rows, [])

    def test_recommendation_upload_is_stored_but_not_added_to_training_dataset(self) -> None:
        upload = raw_upload(
            event_type="recommendation_record",
            payload={
                "event_type": "recommendation_record",
                "schema_version": 1,
                "recommendation_id": "rec_1",
                "install_id": "verified_install",
                "machine_id": "machine_1",
                "next_dose_g": 18.0,
                "target_yield_g": 38.0,
                "target_ratio": 2.111,
            },
        )
        warehouse = FakeWarehouse([upload])

        result = CommunityValidationService(warehouse).validate_once()

        self.assertEqual(result.stored_recommendations, 1)
        self.assertEqual(result.training_rows, 0)
        self.assertEqual(warehouse.recommendations[0].recommendation_id, "rec_1")
        self.assertEqual(warehouse.statuses[upload.upload_id], "validated")

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

    def test_not_followed_payload_gets_low_training_weight(self) -> None:
        payload = shot_payload()
        payload["recommendation_followed"] = "not_followed"
        weight = payload_trust_weight(payload, install_trust=0.35)
        self.assertLess(weight, 0.08)


def raw_upload(
    payload: dict[str, Any],
    *,
    event_type: str = "shot_record",
    upload_id: str = "upload_1",
) -> CommunityRawUpload:
    return CommunityRawUpload(
        install_id="verified_install",
        upload_id=upload_id,
        payload_hash=HASH,
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
        "profile_resampled": profile,
        "dose_in_g": 18.0,
        "beverage_out_g": 38.0,
        "target_yield_g": 38.0,
        "target_ratio": 2.111,
        "shot_time_s": 30.0,
        "human_rating": 4,
        "taste_tags": ["balanced"],
        "reward_confidence": 1.0,
        "optimization_weight": 1.0,
        "recommendation_followed": "followed",
        "recommendation_attribution_weight": 1.0,
        "shot_type": "espresso",
        "exclude_from_local_optimization": False,
    }


class FakeWarehouse:
    def __init__(self, uploads: list[CommunityRawUpload]) -> None:
        self.uploads = uploads
        self.statuses: dict[str, str] = {upload.upload_id: "mirrored" for upload in uploads}
        self.rejections: dict[str, list[str]] = {}
        self.validated_shots: list[CommunityValidatedShot] = []
        self.recommendations: list[CommunityRecommendationRecord] = []
        self.trust_scores: list[InstallTrustScore] = []
        self.abuse_events: list[CommunityAbuseEvent] = []
        self.training_rows: list[tuple[int, dict[str, Any], float]] = []
        self.validated_summaries: dict[str, dict[str, Any]] = {}

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
        self.validated_shots.append(shot)
        return len(self.validated_shots)

    def upsert_community_recommendation(self, recommendation: CommunityRecommendationRecord) -> None:
        self.recommendations.append(recommendation)

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

    def upsert_training_row(
        self,
        source_validation_id: int,
        payload_json: dict[str, Any],
        trust_weight: float,
    ) -> None:
        self.training_rows.append((source_validation_id, payload_json, trust_weight))


if __name__ == "__main__":
    unittest.main()

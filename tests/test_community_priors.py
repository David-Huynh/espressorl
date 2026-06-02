from __future__ import annotations

import unittest
from typing import Any

from espresso_rl.application.community_priors import (
    MAX_COMMUNITY_PRIOR_CONFIDENCE,
    CommunityPriorGenerationService,
    community_prior_contribution_bucket,
    community_prior_context_key,
)
from espresso_rl.domain.community import (
    CommunityAbuseEvent,
    CommunityInstallStats,
    CommunityPrior,
    CommunityRawUpload,
    CommunityRecommendationRecord,
    CommunityTrainingRow,
    CommunityUploadCredentials,
    CommunityValidatedShot,
    InstallTrustScore,
)


HASH = "d" * 64


class CommunityPriorTests(unittest.TestCase):
    def test_requires_multiple_independent_installs_before_writing_prior(self) -> None:
        warehouse = FakeWarehouse(
            [
                training_row(1, install_id="install_a", rating=5),
                training_row(2, install_id="install_a", rating=4),
                training_row(3, install_id="install_a", rating=5),
                training_row(4, install_id="install_a", rating=4),
            ]
        )

        result = CommunityPriorGenerationService(
            warehouse,
            min_independent_installs=3,
            min_context_points=3,
        ).generate_once()

        self.assertEqual(result.eligible, 4)
        self.assertEqual(result.priors_written, 0)
        self.assertEqual(warehouse.priors, [])

    def test_writes_low_confidence_prior_from_valid_capped_independent_data(self) -> None:
        rows = [
            training_row(1, install_id="install_a", rating=5, reward=1.0),
            training_row(2, install_id="install_a", rating=5, reward=0.9),
            training_row(3, install_id="install_a", rating=1, reward=0.0),
            training_row(4, install_id="install_b", rating=4, reward=0.75),
            training_row(5, install_id="install_b", rating=4, reward=0.7),
            training_row(6, install_id="install_c", rating=4, reward=0.8),
            training_row(7, install_id="install_c", rating=3, reward=0.5),
        ]
        warehouse = FakeWarehouse(rows)

        result = CommunityPriorGenerationService(
            warehouse,
            min_independent_installs=3,
            min_context_points=6,
            max_points_per_install_per_context=2,
        ).generate_once()

        self.assertEqual(result.priors_written, 1)
        prior = warehouse.priors[0]
        self.assertEqual(prior.context_key, "adapter:gaggimate|dose:18.0|ratio:2.1")
        self.assertLessEqual(prior.confidence, MAX_COMMUNITY_PRIOR_CONFIDENCE)
        point = prior.prior_json["points"][0]
        self.assertEqual(point["observation_noise"], 0.5)
        self.assertEqual(point["independent_install_count"], 3)
        self.assertEqual(point["support"], 6)
        self.assertGreater(point["predicted_reward"], 0.0)
        self.assertEqual(prior.prior_json["zero_trust"]["per_install_contribution_bucket_cap"], 2)
        self.assertIn("diminishing-install-weight", prior.prior_json["aggregation"]["method"])

    def test_dry_run_reports_proposed_priors_without_writing(self) -> None:
        rows = [
            training_row(1, install_id="install_a", rating=5, reward=1.0),
            training_row(2, install_id="install_a", rating=5, reward=0.9),
            training_row(3, install_id="install_b", rating=4, reward=0.75),
            training_row(4, install_id="install_b", rating=4, reward=0.7),
            training_row(5, install_id="install_c", rating=4, reward=0.8),
            training_row(6, install_id="install_c", rating=3, reward=0.5),
        ]
        warehouse = FakeWarehouse(rows)

        result = CommunityPriorGenerationService(
            warehouse,
            min_independent_installs=3,
            min_context_points=6,
            max_points_per_install_per_context=2,
        ).generate_once(dry_run=True)

        self.assertEqual(result.priors_written, 1)
        self.assertEqual(warehouse.priors, [])

    def test_revalidates_training_rows_and_rejects_spoofed_or_impossible_payloads(self) -> None:
        bad_install = training_row(1, install_id="install_a")
        bad_install.payload_json["install_id"] = "spoofed_install"
        impossible = training_row(2, install_id="install_b")
        impossible.payload_json["target_yield_g"] = 500.0
        good = [
            training_row(3, install_id="install_a"),
            training_row(4, install_id="install_b"),
            training_row(5, install_id="install_c"),
        ]
        warehouse = FakeWarehouse([bad_install, impossible, *good])

        result = CommunityPriorGenerationService(
            warehouse,
            min_independent_installs=3,
            min_context_points=3,
        ).generate_once()

        self.assertEqual(result.rejected, 2)
        self.assertEqual(result.eligible, 3)
        self.assertEqual(result.priors_written, 1)

    def test_excluded_or_zero_weight_rows_do_not_influence_prior(self) -> None:
        excluded = training_row(1, install_id="install_a")
        excluded.payload_json["exclude_from_local_optimization"] = True
        zero_weight = training_row(2, install_id="install_b", trust_weight=0.0)
        warehouse = FakeWarehouse(
            [
                excluded,
                zero_weight,
                training_row(3, install_id="install_a"),
                training_row(4, install_id="install_b"),
                training_row(5, install_id="install_c"),
            ]
        )

        result = CommunityPriorGenerationService(
            warehouse,
            min_independent_installs=3,
            min_context_points=3,
        ).generate_once()

        self.assertEqual(result.rejected, 2)
        self.assertEqual(result.eligible, 3)
        self.assertEqual(warehouse.priors[0].prior_json["points"][0]["support"], 3)

    def test_context_key_uses_broad_machine_dose_and_ratio_buckets(self) -> None:
        payload = shot_payload(machine_adapter="Gaggimate Pro!", dose=18.24, ratio=2.07)
        self.assertEqual(
            community_prior_context_key(payload),
            "adapter:gaggimate_pro|dose:18.0|ratio:2.1",
        )

    def test_high_volume_diverse_install_can_contribute_across_narrow_buckets(self) -> None:
        rows = [
            training_row(
                index,
                install_id="cafe_install",
                bean_context_id=f"bean_{index}",
                shot_time_s=22.0 + index,
                profile_peak_pressure=7.0 + (index % 4),
                reward=0.72 + index * 0.01,
            )
            for index in range(1, 9)
        ]
        warehouse = FakeWarehouse(rows)

        result = CommunityPriorGenerationService(
            warehouse,
            min_independent_installs=3,
            min_context_points=6,
            min_diverse_buckets_for_single_install=6,
            max_points_per_install_per_bucket=2,
        ).generate_once()

        self.assertEqual(result.priors_written, 1)
        prior = warehouse.priors[0]
        aggregation = prior.prior_json["aggregation"]
        self.assertEqual(aggregation["independent_install_count"], 1)
        self.assertGreaterEqual(aggregation["contribution_bucket_count"], 6)
        self.assertLess(prior.confidence, 0.06)

    def test_high_volume_repetitive_install_does_not_release_public_prior(self) -> None:
        rows = [
            training_row(index, install_id="repetitive_install", reward=0.8)
            for index in range(1, 301)
        ]
        warehouse = FakeWarehouse(rows)

        result = CommunityPriorGenerationService(
            warehouse,
            min_independent_installs=3,
            min_context_points=6,
            min_diverse_buckets_for_single_install=6,
            max_points_per_install_per_bucket=2,
        ).generate_once()

        self.assertEqual(result.eligible, 300)
        self.assertEqual(result.priors_written, 0)
        self.assertEqual(warehouse.priors, [])

    def test_contribution_bucket_changes_for_different_beans_recipes_actions_and_profiles(self) -> None:
        base = shot_payload(bean_context_id="bean_a", dose=18.0, ratio=2.1, shot_time_s=30.0)
        different = shot_payload(
            bean_context_id="bean_b",
            dose=18.0,
            ratio=2.1,
            shot_time_s=38.0,
            profile_peak_pressure=7.0,
        )
        different["recommended_grind_delta_um"] = 25.0
        different["recommended_target_yield_g"] = different["target_yield_g"] + 2.0

        self.assertNotEqual(
            community_prior_contribution_bucket(base),
            community_prior_contribution_bucket(different),
        )


def training_row(
    row_id: int,
    *,
    install_id: str,
    rating: int = 4,
    reward: float | None = None,
    trust_weight: float = 0.2,
    bean_context_id: str = "bean_1",
    dose: float = 18.0,
    ratio: float = 2.1,
    shot_time_s: float = 30.0,
    profile_peak_pressure: float | None = None,
) -> CommunityTrainingRow:
    payload = shot_payload(
        install_id=install_id,
        rating=rating,
        reward=reward,
        bean_context_id=bean_context_id,
        dose=dose,
        ratio=ratio,
        shot_time_s=shot_time_s,
        profile_peak_pressure=profile_peak_pressure,
    )
    return CommunityTrainingRow(
        training_row_id=row_id,
        source_validation_id=row_id,
        install_id=install_id,
        payload_json=payload,
        trust_weight=trust_weight,
        payload_hash=HASH,
    )


def shot_payload(
    *,
    install_id: str = "install_a",
    machine_adapter: str = "gaggimate",
    dose: float = 18.0,
    ratio: float = 2.1,
    rating: int = 4,
    reward: float | None = None,
    bean_context_id: str = "bean_1",
    shot_time_s: float = 30.0,
    profile_peak_pressure: float | None = None,
) -> dict[str, Any]:
    target_yield = round(dose * ratio, 1)
    payload = {
        "event_type": "shot_record",
        "schema_version": 1,
        "shot_id": f"shot_{install_id}_{rating}_{bean_context_id}_{shot_time_s}",
        "timestamp": 1_779_999_000,
        "install_id": install_id,
        "machine_id": "machine_1",
        "machine_adapter": machine_adapter,
        "bean_context_id": bean_context_id,
        "dose_in_g": dose,
        "beverage_out_g": target_yield,
        "target_yield_g": target_yield,
        "target_ratio": ratio,
        "shot_time_s": shot_time_s,
        "human_rating": rating,
        "taste_tags": ["balanced"],
        "reward": reward if reward is not None else (rating - 1.0) / 4.0,
        "reward_confidence": 1.0,
        "optimization_weight": 1.0,
        "recommendation_followed": "followed",
        "recommendation_attribution_weight": 1.0,
        "shot_type": "espresso",
        "exclude_from_local_optimization": False,
    }
    if profile_peak_pressure is not None:
        profile = [[0.0 for _ in range(100)] for _ in range(5)]
        profile[0] = [profile_peak_pressure for _ in range(100)]
        profile[1] = [9.0 for _ in range(100)]
        profile[2] = [2.0 for _ in range(100)]
        profile[3] = [2.0 for _ in range(100)]
        profile[4] = [target_yield for _ in range(100)]
        payload["profile_resampled"] = profile
    return payload


class FakeWarehouse:
    def __init__(self, rows: list[CommunityTrainingRow]) -> None:
        self.rows = rows
        self.priors: list[CommunityPrior] = []

    def upsert_raw_upload(self, upload: CommunityRawUpload) -> None:
        raise NotImplementedError

    def list_raw_uploads(self, status: str = "mirrored", limit: int = 100) -> list[CommunityRawUpload]:
        return []

    def mark_raw_upload_validated(
        self,
        upload: CommunityRawUpload,
        validation_summary: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    def mark_raw_upload_rejected(
        self,
        upload: CommunityRawUpload,
        validation_errors: list[str],
    ) -> None:
        raise NotImplementedError

    def upsert_validated_shot(self, shot: CommunityValidatedShot) -> int:
        raise NotImplementedError

    def upsert_community_recommendation(self, recommendation: CommunityRecommendationRecord) -> None:
        raise NotImplementedError

    def upsert_install_trust_score(self, score: InstallTrustScore) -> None:
        raise NotImplementedError

    def install_stats(self, install_id: str) -> CommunityInstallStats:
        return CommunityInstallStats()

    def record_abuse_event(self, event: CommunityAbuseEvent) -> None:
        raise NotImplementedError

    def upsert_training_row(
        self,
        source_validation_id: int,
        payload_json: dict[str, Any],
        trust_weight: float,
    ) -> None:
        raise NotImplementedError

    def list_training_rows(self, limit: int = 5000) -> list[CommunityTrainingRow]:
        return self.rows[:limit]

    def upsert_community_prior(self, prior: CommunityPrior) -> None:
        self.priors.append(prior)


if __name__ == "__main__":
    unittest.main()

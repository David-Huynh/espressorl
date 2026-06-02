from __future__ import annotations

import unittest

import numpy as np

from espresso_rl.application.prior_providers import CommunityPriorProvider
from espresso_rl.application.services import EspressoRLService
from espresso_rl.domain.community import CommunityPrior
from espresso_rl.domain.models import (
    FollowThroughState,
    Recipe,
    RecommendationMode,
    SafetyBounds,
    ShotRecord,
)
from espresso_rl.domain.optimization import OptimizationContext, PriorPoint
from espresso_rl.optimizers.conservative_bo import ConservativeBOOptimizer
from tests.test_application_service import (
    MemoryRecommendationRepository,
    MemoryShotRepository,
    shot_event,
)


class WarmStartPriorTests(unittest.TestCase):
    def test_external_prior_enables_warm_started_mode_inside_safety_bounds(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(
            shots,
            recs,
            ConservativeBOOptimizer(),
            prior_provider=StaticPriorProvider(
                [
                    PriorPoint(
                        grind_delta_um=0.0,
                        dose_g=18.0,
                        target_yield_g=40.0,
                        target_ratio=40.0 / 18.0,
                        predicted_reward=0.95,
                        confidence=0.18,
                        observation_noise=0.5,
                        source="community",
                    )
                ]
            ),
            clock=lambda: 10,
        )

        result = service.ingest_shot_profile(shot_event("shot_1", 1))

        self.assertEqual(result.recommendation.mode, RecommendationMode.WARM_STARTED_BO)
        self.assertLessEqual(abs(result.recommendation.target_yield_g - 36.0), 4.0)
        self.assertLessEqual(abs(result.recommendation.grind_delta_steps), 2)

    def test_bad_prior_outside_safety_bounds_is_not_used_for_warm_start(self) -> None:
        shots = MemoryShotRepository()
        recs = MemoryRecommendationRepository()
        service = EspressoRLService(
            shots,
            recs,
            ConservativeBOOptimizer(),
            prior_provider=StaticPriorProvider(
                [
                    PriorPoint(
                        grind_delta_um=500.0,
                        dose_g=40.0,
                        target_yield_g=120.0,
                        target_ratio=3.0,
                        predicted_reward=1.0,
                        confidence=1.0,
                        observation_noise=0.01,
                        source="community",
                    )
                ]
            ),
            clock=lambda: 10,
        )

        result = service.ingest_shot_profile(shot_event("shot_1", 1))

        self.assertEqual(result.recommendation.mode, RecommendationMode.ZERO_IMMEDIATE_BO)
        self.assertLessEqual(abs(result.recommendation.grind_delta_steps), 2)
        self.assertLessEqual(abs(result.recommendation.target_yield_g - 36.0), 4.0)

    def test_local_data_disables_external_priors_after_sparse_startup(self) -> None:
        current = Recipe(
            grind_steps=42,
            grinder_step_size_um=12.5,
            dose_g=18.0,
            target_yield_g=36.0,
        )
        context = OptimizationContext(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_1",
            machine_adapter="gaggimate",
            current_recipe=current,
            shots=[
                shot_record(f"shot_{index}", timestamp=index, reward=0.9, rating=5)
                for index in range(1, 6)
            ],
            safety_bounds=SafetyBounds(),
            now=100,
            prior_points=[
                PriorPoint(
                    grind_delta_um=25.0,
                    dose_g=18.0,
                    target_yield_g=44.0,
                    target_ratio=44.0 / 18.0,
                    predicted_reward=1.0,
                    confidence=1.0,
                    observation_noise=0.01,
                    source="community",
                )
            ],
        )

        recommendation = ConservativeBOOptimizer().recommend(context)

        self.assertEqual(recommendation.mode, RecommendationMode.LOCAL_BO)
        self.assertNotEqual(recommendation.target_yield_g, 44.0)

    def test_community_provider_revalidates_and_caps_released_prior_json(self) -> None:
        repo = FakeCommunityPriorRepo(
            [
                released_prior(confidence=1.0, point_confidence=1.0, observation_noise=0.01),
                CommunityPrior(
                    context_key="adapter:gaggimate|dose:18.0|ratio:2.0",
                    prior_json={
                        "context_key": "adapter:gaggimate|dose:18.0|ratio:2.0",
                        "zero_trust": {
                            "validated_training_rows_only": False,
                            "revalidated_before_aggregation": True,
                        },
                        "points": [
                            {
                                "grind_delta_um": 999.0,
                                "dose_g": 18.0,
                                "target_yield_g": 36.0,
                                "target_ratio": 2.0,
                                "predicted_reward": 1.0,
                                "confidence": 1.0,
                                "observation_noise": 0.01,
                            }
                        ],
                    },
                    confidence=1.0,
                ),
            ]
        )
        context = OptimizationContext(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_1",
            machine_adapter="gaggimate",
            current_recipe=Recipe(
                grind_steps=42,
                grinder_step_size_um=12.5,
                dose_g=18.0,
                target_yield_g=36.0,
            ),
            shots=[shot_record("shot_1", timestamp=1)],
            safety_bounds=SafetyBounds(),
            now=10,
        )

        points = CommunityPriorProvider(repo).get_prior_points(context)

        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].source, "community")
        self.assertEqual(points[0].confidence, 0.18)
        self.assertEqual(points[0].observation_noise, 0.5)


class StaticPriorProvider:
    def __init__(self, points: list[PriorPoint]) -> None:
        self.points = points

    def get_prior_points(self, context: OptimizationContext) -> list[PriorPoint]:
        return list(self.points)


class FakeCommunityPriorRepo:
    def __init__(self, priors: list[CommunityPrior]) -> None:
        self.priors = priors

    def list_community_priors(self, context_key: str, limit: int = 10) -> list[CommunityPrior]:
        return [prior for prior in self.priors if prior.context_key == context_key][:limit]


def released_prior(
    *,
    confidence: float,
    point_confidence: float,
    observation_noise: float,
) -> CommunityPrior:
    return CommunityPrior(
        context_key="adapter:gaggimate|dose:18.0|ratio:2.0",
        prior_json={
            "context_key": "adapter:gaggimate|dose:18.0|ratio:2.0",
            "zero_trust": {
                "validated_training_rows_only": True,
                "revalidated_before_aggregation": True,
            },
            "points": [
                {
                    "grind_delta_um": 0.0,
                    "dose_g": 18.0,
                    "target_yield_g": 36.0,
                    "target_ratio": 2.0,
                    "predicted_reward": 0.7,
                    "confidence": point_confidence,
                    "observation_noise": observation_noise,
                }
            ],
        },
        confidence=confidence,
    )


def shot_record(
    shot_id: str,
    *,
    timestamp: int,
    reward: float = 0.6,
    rating: int | None = None,
) -> ShotRecord:
    return ShotRecord(
        shot_id=shot_id,
        timestamp=timestamp,
        install_id="install_1",
        machine_id="machine_1",
        machine_adapter="gaggimate",
        profile=np.zeros((5, 100), dtype=np.float32),
        grinder_step_size_um=12.5,
        dose_in_g=18.0,
        target_yield_g=36.0,
        grind_steps=42,
        beverage_out_g=36.0,
        bean_context_id="bean_1",
        reward=reward,
        reward_confidence=1.0,
        human_rating=rating,
        recommendation_followed=FollowThroughState.FOLLOWED,
    )


if __name__ == "__main__":
    unittest.main()

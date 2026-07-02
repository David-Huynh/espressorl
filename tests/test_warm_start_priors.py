from __future__ import annotations

import unittest

import numpy as np

from espresso_rl.application.prior_providers import CommunityPriorProvider, LocalHistoryPriorProvider
from espresso_rl.application.services import EspressoRLService
from espresso_rl.config import Config
from espresso_rl.domain.community import CommunityPrior
from espresso_rl.domain.models import (
    FollowThroughState,
    GrinderAdjustmentMode,
    GrinderStepDirection,
    Recipe,
    RecommendationDecision,
    RecommendationMode,
    SafetyBounds,
    ShotRecord,
)
from espresso_rl.domain.optimization import OptimizationContext, PriorPoint, PriorSignal
from espresso_rl.optimizers.conservative_bo import ConservativeBOOptimizer
from espresso_rl.main import open_prior_provider
from tests.test_application_service import (
    MemoryRecommendationRepository,
    MemoryShotRepository,
    ingest_and_feedback,
    shot_event,
)


class WarmStartPriorTests(unittest.TestCase):
    def test_single_observation_probe_uses_whole_step_for_stepped_grinder(self) -> None:
        current = Recipe(42, 12.5, 18.0, 36.0)
        context = OptimizationContext(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_1",
            machine_adapter="gaggimate",
            current_recipe=current,
            shots=[shot_record("shot_1", timestamp=1, reward=0.5, rating=3)],
            safety_bounds=SafetyBounds(),
            now=100,
        )

        recommendation = ConservativeBOOptimizer().recommend(context)

        self.assertEqual(recommendation.grinder_adjustment_mode, GrinderAdjustmentMode.STEPPED)
        self.assertEqual(recommendation.grind_delta_steps_from_current, 1.0)
        self.assertEqual(recommendation.projected_relative_step_from_reference, 43.0)

    def test_single_observation_probe_can_use_fractional_steps_for_stepless_grinder(self) -> None:
        current = Recipe(
            42,
            12.5,
            18.0,
            36.0,
            grinder_adjustment_mode=GrinderAdjustmentMode.STEPLESS,
        )
        context = OptimizationContext(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_1",
            machine_adapter="gaggimate",
            current_recipe=current,
            shots=[shot_record("shot_1", timestamp=1, reward=0.5, rating=3)],
            safety_bounds=SafetyBounds(),
            now=100,
        )

        recommendation = ConservativeBOOptimizer().recommend(context)

        self.assertEqual(recommendation.grinder_adjustment_mode, GrinderAdjustmentMode.STEPLESS)
        self.assertEqual(recommendation.grind_delta_steps_from_current, 0.5)
        self.assertEqual(recommendation.projected_relative_step_from_reference, 42.5)

    def test_finer_rule_uses_active_grinder_step_direction(self) -> None:
        optimizer = ConservativeBOOptimizer()
        signal = PriorSignal(
            grind_direction=1,
            ratio_direction=0,
            dose_direction=0,
            confidence=0.65,
            observation_noise=0.3,
            source="user_rule",
        )

        def recommendation_for(direction: GrinderStepDirection):
            current = Recipe(
                42,
                12.5,
                18.0,
                36.0,
                grinder_step_direction=direction,
            )
            return optimizer.recommend(
                OptimizationContext(
                    install_id="install_1",
                    machine_id="machine_1",
                    bean_context_id="bean_1",
                    machine_adapter="gaggimate",
                    current_recipe=current,
                    shots=[shot_record("shot_1", timestamp=1, reward=0.5, rating=3)],
                    safety_bounds=SafetyBounds(),
                    now=100,
                    prior_signals=[signal],
                )
            )

        higher_is_finer = recommendation_for(GrinderStepDirection.HIGHER_IS_FINER)
        higher_is_coarser = recommendation_for(GrinderStepDirection.HIGHER_IS_COARSER)

        self.assertGreater(higher_is_finer.grind_delta_steps_from_current, 0)
        self.assertLess(higher_is_coarser.grind_delta_steps_from_current, 0)
        self.assertGreater(higher_is_coarser.grind_delta_um_from_current, 0)
        self.assertEqual(higher_is_finer.mode, RecommendationMode.WARM_STARTED_BO)
        self.assertIn("BO selects a bounded step", higher_is_finer.reason)

    def test_partial_action_distance_uses_achieved_yield_and_masks_unknown_dimensions(self) -> None:
        optimizer = ConservativeBOOptimizer()
        shot = shot_record("partial", timestamp=1)
        shot.grind_observed = False
        shot.dose_observed = False
        shot.beverage_out_g = 40.0
        at_achieved_yield = Recipe(
            relative_grind_steps_from_reference=5,
            microns_per_step=12.5,
            dose_g=21.0,
            target_yield_g=40.0,
        )
        at_intended_yield = Recipe(
            relative_grind_steps_from_reference=42,
            microns_per_step=12.5,
            dose_g=18.0,
            target_yield_g=36.0,
        )

        self.assertEqual(optimizer._action_coverage(shot), 1 / 3)
        self.assertEqual(optimizer._distance(at_achieved_yield, shot, 5, 4.0, 1.0), 0.0)
        self.assertEqual(optimizer._distance(at_intended_yield, shot, 5, 4.0, 1.0), 1.0)

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
                        grind_delta_um_from_current=0.0,
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

        recommendation = ingest_and_feedback(service, shot_event("shot_1", 1))

        self.assertEqual(recommendation.mode, RecommendationMode.WARM_STARTED_BO)
        self.assertLessEqual(abs(recommendation.target_yield_g - 36.0), 4.0)
        self.assertLessEqual(abs(recommendation.grind_delta_steps_from_current), 2)

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
                        grind_delta_um_from_current=500.0,
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

        recommendation = ingest_and_feedback(service, shot_event("shot_1", 1))

        self.assertEqual(recommendation.mode, RecommendationMode.ZERO_IMMEDIATE_BO)
        self.assertLessEqual(abs(recommendation.grind_delta_steps_from_current), 2)
        self.assertLessEqual(abs(recommendation.target_yield_g - 36.0), 4.0)

    def test_taste_tags_without_selected_rules_do_not_expand_first_probe(self) -> None:
        current = Recipe(42, 12.5, 18.0, 36.0)
        context = OptimizationContext(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_1",
            machine_adapter="gaggimate",
            current_recipe=current,
            shots=[
                shot_record(
                    "shot_1",
                    timestamp=1,
                    reward=0.25,
                    rating=2,
                    taste_tags=["sour", "too_fast"],
                    shot_time_s=18.0,
                )
            ],
            safety_bounds=SafetyBounds(),
            now=100,
        )

        recommendation = ConservativeBOOptimizer().recommend(context)

        self.assertEqual(recommendation.mode, RecommendationMode.ZERO_IMMEDIATE_BO)
        self.assertLessEqual(abs(recommendation.grind_delta_steps_from_current), 2)
        self.assertLessEqual(abs(recommendation.target_yield_g - current.target_yield_g), 4.0)

    def test_first_short_shot_without_directional_feedback_stays_small(self) -> None:
        current = Recipe(42, 12.5, 18.0, 36.0)
        context = OptimizationContext(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_1",
            machine_adapter="gaggimate",
            current_recipe=current,
            shots=[
                shot_record(
                    "shot_1",
                    timestamp=1,
                    reward=0.25,
                    rating=2,
                    taste_tags=[],
                    shot_time_s=18.0,
                )
            ],
            safety_bounds=SafetyBounds(),
            now=100,
        )

        recommendation = ConservativeBOOptimizer().recommend(context)

        self.assertEqual(recommendation.mode, RecommendationMode.ZERO_IMMEDIATE_BO)
        self.assertLessEqual(abs(recommendation.grind_delta_steps_from_current), 2)
        self.assertLessEqual(abs(recommendation.target_yield_g - current.target_yield_g), 4.0)

    def test_first_near_good_shot_stays_in_small_refinement_region(self) -> None:
        current = Recipe(42, 12.5, 18.0, 36.0)
        context = OptimizationContext(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_1",
            machine_adapter="gaggimate",
            current_recipe=current,
            shots=[
                shot_record(
                    "shot_1",
                    timestamp=1,
                    reward=0.8,
                    rating=4,
                    taste_tags=["balanced"],
                    shot_time_s=30.0,
                )
            ],
            safety_bounds=SafetyBounds(),
            now=100,
        )

        recommendation = ConservativeBOOptimizer().recommend(context)

        self.assertLessEqual(abs(recommendation.grind_delta_steps_from_current), 2)
        self.assertLessEqual(abs(recommendation.target_yield_g - current.target_yield_g), 4.0)

    def test_strong_same_bean_prior_can_use_full_safe_early_envelope(self) -> None:
        current = Recipe(42, 12.5, 18.0, 36.0)
        context = OptimizationContext(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_lavazza_2",
            machine_adapter="gaggimate",
            current_recipe=current,
            shots=[shot_record("shot_1", timestamp=1, reward=0.55, rating=3)],
            safety_bounds=SafetyBounds(),
            now=100,
            prior_points=[
                PriorPoint(
                    grind_delta_um_from_current=-62.5,
                    dose_g=18.0,
                    target_yield_g=36.0,
                    target_ratio=2.0,
                    predicted_reward=1.0,
                    confidence=0.85,
                    observation_noise=0.25,
                    source="local_bean_history",
                )
            ],
        )

        recommendation = ConservativeBOOptimizer().recommend(context)

        self.assertEqual(recommendation.mode, RecommendationMode.WARM_STARTED_BO)
        self.assertLessEqual(recommendation.grind_delta_steps_from_current, -3)
        self.assertGreaterEqual(recommendation.grind_delta_steps_from_current, -5)

    def test_empirical_prior_um_maps_through_grinder_step_direction(self) -> None:
        current = Recipe(
            42,
            12.5,
            18.0,
            36.0,
            grinder_step_direction=GrinderStepDirection.HIGHER_IS_COARSER,
        )
        context = OptimizationContext(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_1",
            machine_adapter="gaggimate",
            current_recipe=current,
            shots=[shot_record("shot_1", timestamp=1, reward=0.5, rating=3)],
            safety_bounds=SafetyBounds(),
            now=100,
            prior_points=[
                PriorPoint(
                    grind_delta_um_from_current=50.0,
                    dose_g=18.0,
                    target_yield_g=36.0,
                    target_ratio=2.0,
                    predicted_reward=1.0,
                    confidence=0.85,
                    observation_noise=0.25,
                    source="local_bean_history",
                )
            ],
        )

        recommendation = ConservativeBOOptimizer().recommend(context)

        self.assertLess(recommendation.grind_delta_steps_from_current, 0)
        self.assertGreater(recommendation.grind_delta_um_from_current, 0)

    def test_same_bean_history_prior_uses_generic_warm_started_mode(self) -> None:
        current = Recipe(
            relative_grind_steps_from_reference=42,
            microns_per_step=12.5,
            dose_g=18.0,
            target_yield_g=36.0,
        )
        context = OptimizationContext(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_lavazza_2",
            machine_adapter="gaggimate",
            current_recipe=current,
            shots=[shot_record("shot_1", timestamp=1, reward=0.55, rating=3)],
            safety_bounds=SafetyBounds(),
            now=100,
            prior_points=[
                PriorPoint(
                    grind_delta_um_from_current=-25.0,
                    dose_g=18.0,
                    target_yield_g=38.0,
                    target_ratio=38.0 / 18.0,
                    predicted_reward=0.95,
                    confidence=0.85,
                    observation_noise=0.25,
                    source="local_bean_history",
                )
            ],
        )

        recommendation = ConservativeBOOptimizer().recommend(context)

        self.assertEqual(recommendation.mode, RecommendationMode.WARM_STARTED_BO)
        self.assertIn("Same-bean previous bag history", recommendation.reason)

    def test_local_data_disables_external_priors_after_sparse_startup(self) -> None:
        current = Recipe(
            relative_grind_steps_from_reference=42,
            microns_per_step=12.5,
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
                    grind_delta_um_from_current=25.0,
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

    def test_taste_tags_do_not_create_hidden_directional_rules(self) -> None:
        current = Recipe(42, 12.5, 18.0, 36.0)
        context = OptimizationContext(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_1",
            machine_adapter="gaggimate",
            current_recipe=current,
            shots=[
                shot_record("shot_1", timestamp=1, reward=0.5, rating=3, taste_tags=["sour"]),
                shot_record("shot_2", timestamp=2, reward=0.5, rating=3, taste_tags=["thin"]),
            ],
            safety_bounds=SafetyBounds(),
            now=100,
        )

        optimizer = ConservativeBOOptimizer()
        recommendation = optimizer.recommend(context)
        without_tags = optimizer.recommend(
            OptimizationContext(
                install_id=context.install_id,
                machine_id=context.machine_id,
                bean_context_id=context.bean_context_id,
                machine_adapter=context.machine_adapter,
                current_recipe=current,
                shots=[
                    shot_record("shot_1", timestamp=1, reward=0.5, rating=3),
                    shot_record("shot_2", timestamp=2, reward=0.5, rating=3),
                ],
                safety_bounds=context.safety_bounds,
                now=context.now,
            )
        )

        self.assertEqual(recommendation.source_shot_id, "shot_2")
        self.assertEqual(
            recommendation.grind_delta_steps_from_current,
            without_tags.grind_delta_steps_from_current,
        )
        self.assertEqual(
            recommendation.target_yield_g,
            without_tags.target_yield_g,
        )

    def test_flat_local_evidence_still_probes_a_new_bounded_candidate(self) -> None:
        current = Recipe(42, 12.5, 18.0, 36.0)
        context = OptimizationContext(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_1",
            machine_adapter="gaggimate",
            current_recipe=current,
            shots=[
                shot_record("shot_1", timestamp=1, reward=0.5, rating=3),
                shot_record("shot_2", timestamp=2, reward=0.5, rating=3),
            ],
            safety_bounds=SafetyBounds(),
            now=100,
        )

        recommendation = ConservativeBOOptimizer().recommend(context)

        self.assertTrue(
            recommendation.projected_relative_step_from_reference != current.relative_grind_steps_from_reference
            or recommendation.target_yield_g != current.target_yield_g
        )
        self.assertLessEqual(abs(recommendation.grind_delta_steps_from_current), 2)
        self.assertLessEqual(abs(recommendation.target_yield_g - current.target_yield_g), 4.0)

    def test_ignored_and_not_followed_shots_still_inform_optimizer(self) -> None:
        current = Recipe(42, 12.5, 18.0, 36.0)
        ignored = shot_record("ignored", timestamp=1, reward=1.0, rating=5)
        ignored.recommendation_decision = RecommendationDecision.IGNORED
        not_followed = shot_record("not_followed", timestamp=2, reward=1.0, rating=5)
        not_followed.recommendation_followed = FollowThroughState.NOT_FOLLOWED
        context = OptimizationContext(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_1",
            machine_adapter="gaggimate",
            current_recipe=current,
            shots=[ignored, not_followed],
            safety_bounds=SafetyBounds(),
            now=100,
        )

        recommendation = ConservativeBOOptimizer().recommend(context)

        self.assertEqual(recommendation.mode, RecommendationMode.ZERO_IMMEDIATE_BO)
        self.assertEqual(recommendation.source_shot_id, "not_followed")

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
                                "grind_delta_um_from_current": 999.0,
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
                relative_grind_steps_from_reference=42,
                microns_per_step=12.5,
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

    def test_community_provider_consumes_more_than_five_low_weight_prior_points(self) -> None:
        prior = released_prior(confidence=0.12, point_confidence=0.02, observation_noise=0.5)
        prior.prior_json["points"] = [
            {
                "grind_delta_um_from_current": float(index),
                "dose_g": 18.0,
                "target_yield_g": 36.0,
                "target_ratio": 2.0,
                "predicted_reward": 0.4 + index * 0.01,
                "confidence": 0.02,
                "observation_noise": 0.5,
            }
            for index in range(12)
        ]
        repo = FakeCommunityPriorRepo([prior])
        context = OptimizationContext(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_1",
            machine_adapter="gaggimate",
            current_recipe=Recipe(
                relative_grind_steps_from_reference=42,
                microns_per_step=12.5,
                dose_g=18.0,
                target_yield_g=36.0,
            ),
            shots=[shot_record("shot_1", timestamp=1)],
            safety_bounds=SafetyBounds(),
            now=10,
        )

        points = CommunityPriorProvider(repo).get_prior_points(context)

        self.assertEqual(len(points), 12)
        self.assertEqual(points[-1].grind_delta_um_from_current, 11.0)
        self.assertLessEqual(max(point.confidence for point in points), 0.02)

    def test_local_history_provider_can_emit_more_than_five_rank_weighted_points(self) -> None:
        current = Recipe(42, 12.5, 18.0, 36.0)
        context = OptimizationContext(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_1",
            machine_adapter="gaggimate",
            current_recipe=current,
            shots=[
                shot_record(f"shot_{index}", timestamp=index, reward=0.7, rating=4)
                for index in range(12)
            ],
            safety_bounds=SafetyBounds(),
            now=100,
        )

        points = LocalHistoryPriorProvider().get_prior_points(context)

        self.assertEqual(len(points), 12)
        self.assertGreater(points[0].confidence, points[-1].confidence)
        self.assertLess(points[0].observation_noise, points[-1].observation_noise)

    def test_default_prior_provider_does_not_emit_handwritten_rule_priors(self) -> None:
        current = Recipe(42, 12.5, 18.0, 36.0)
        context = OptimizationContext(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_1",
            machine_adapter="gaggimate",
            current_recipe=current,
            shots=[
                shot_record(
                    "shot_1",
                    timestamp=1,
                    reward=None,
                    rating=None,
                    taste_tags=["too_fast"],
                    shot_time_s=18.0,
                )
            ],
            safety_bounds=SafetyBounds(),
            now=100,
        )

        points = open_prior_provider(Config(mqtt_host="unused")).get_prior_points(context)

        self.assertEqual(points, [])


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
                    "grind_delta_um_from_current": 0.0,
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
    taste_tags: list[str] | None = None,
    shot_time_s: float = 30.0,
) -> ShotRecord:
    return ShotRecord(
        shot_id=shot_id,
        timestamp=timestamp,
        install_id="install_1",
        machine_id="machine_1",
        machine_adapter="gaggimate",
        profile=np.zeros((5, 100), dtype=np.float32),
        microns_per_step=12.5,
        dose_in_g=18.0,
        target_yield_g=36.0,
        relative_grind_steps_from_reference=42,
        beverage_out_g=36.0,
        shot_time_s=shot_time_s,
        bean_context_id="bean_1",
        reward=reward,
        reward_confidence=1.0,
        human_rating=rating,
        taste_tags=taste_tags or [],
        feedback_recorded=True,
        recommendation_followed=FollowThroughState.FOLLOWED,
    )


if __name__ == "__main__":
    unittest.main()

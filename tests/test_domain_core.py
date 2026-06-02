from __future__ import annotations

import unittest

import numpy as np

from espresso_rl.domain.events import ShotProfileEvent
from espresso_rl.domain.follow_through import infer_follow_through
from espresso_rl.domain.models import (
    FollowThroughState,
    Recipe,
    Recommendation,
    RecommendationDecision,
    RecommendationMode,
    ShotRecord,
)
from espresso_rl.domain.profile import profile_mse, profile_score, resample_profile
from espresso_rl.domain.reward import compute_reward
from espresso_rl.domain.staleness import check_recommendation_staleness


def event(**overrides) -> ShotProfileEvent:
    base = {
        "shot_id": "shot_1",
        "install_id": "install_1",
        "machine_id": "machine_1",
        "machine_adapter": "gaggimate",
        "timestamp": 1000,
        "time_ms": [0, 500, 1000],
        "pressure": [0.0, 8.0, 9.0],
        "target_pressure": [0.0, 8.0, 9.0],
        "flow": [0.0, 2.0, 2.0],
        "target_flow": [0.0, 2.0, 2.0],
        "weight": [0.0, 10.0, 36.0],
        "grinder_step_size_um": 12.5,
        "grind_steps": 42,
        "dose_in_g": 18.0,
        "target_yield_g": 36.0,
        "beverage_out_g": 36.0,
        "shot_time_s": 30.0,
    }
    base.update(overrides)
    return ShotProfileEvent(**base)


class DomainCoreTests(unittest.TestCase):
    def test_canonical_profile_requires_aligned_arrays(self) -> None:
        with self.assertRaises(ValueError):
            event(flow=[0.0, 1.0])

    def test_profile_resamples_to_fixed_shape(self) -> None:
        profile = resample_profile(event())
        self.assertEqual(profile.shape, (5, 100))
        self.assertEqual(profile.dtype, np.float32)
        self.assertAlmostEqual(float(profile[4, -1]), 36.0)

    def test_profile_mse_ignores_inactive_zero_flow_target(self) -> None:
        profile = resample_profile(event(flow=[100_000.0, 100_000.0, 100_000.0], target_flow=[0.0, 0.0, 0.0]))

        self.assertEqual(profile_mse(profile), 0.0)
        self.assertEqual(profile_score(profile), 1.0)

    def test_profile_mse_uses_only_active_valid_target_channels(self) -> None:
        profile = resample_profile(
            event(
                pressure=[3.0, 3.0, 3.0],
                target_pressure=[0.0, 0.0, 0.0],
                flow=[1.0, 2.0, 3.0],
                target_flow=[1.0, 2.0, 3.0],
            )
        )

        self.assertEqual(profile_mse(profile), 0.0)

    def test_ignored_recommendation_is_not_followed(self) -> None:
        rec = Recommendation(
            recommendation_id="rec_1",
            created_at=1,
            updated_at=1,
            expires_at=None,
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id=None,
            grind_delta_steps=1,
            grind_delta_um=12.5,
            next_grind_steps=43,
            next_grind_um=537.5,
            next_dose_g=18.0,
            target_yield_g=36.0,
            target_ratio=2.0,
            mode=RecommendationMode.ZERO_IMMEDIATE_BO,
            confidence=0.3,
            reason="test",
        )
        shot = ShotRecord(
            shot_id="shot_1",
            timestamp=2,
            install_id="install_1",
            machine_id="machine_1",
            machine_adapter="gaggimate",
            profile=np.zeros((5, 100), dtype=np.float32),
            grinder_step_size_um=12.5,
            grind_steps=43,
            dose_in_g=18.0,
            target_yield_g=36.0,
            beverage_out_g=36.0,
        )
        result = infer_follow_through(shot, rec, RecommendationDecision.IGNORED)
        self.assertEqual(result.state, FollowThroughState.NOT_FOLLOWED)
        self.assertEqual(result.attribution_weight, 0.0)

    def test_human_rating_dominates_reward_and_not_followed_lowers_confidence(self) -> None:
        result = compute_reward(
            human_rating=5,
            profile_score=0.1,
            follow_through=FollowThroughState.NOT_FOLLOWED,
            taste_tags=["balanced"],
        )
        self.assertGreater(result.reward, 0.8)
        self.assertLess(result.confidence, 0.25)

    def test_reward_confidence_uses_tags_and_channeling_signal(self) -> None:
        untagged = compute_reward(
            human_rating=None,
            profile_score=0.8,
            follow_through=FollowThroughState.UNKNOWN,
        )
        tagged = compute_reward(
            human_rating=None,
            profile_score=0.8,
            follow_through=FollowThroughState.UNKNOWN,
            taste_tags=["balanced"],
        )
        channeling = compute_reward(
            human_rating=None,
            profile_score=0.8,
            follow_through=FollowThroughState.UNKNOWN,
            taste_tags=["channeling_suspected"],
        )

        self.assertGreater(tagged.confidence, untagged.confidence)
        self.assertLess(channeling.confidence, tagged.confidence)

    def test_stale_rules_detect_expiry_bean_change_and_manual_recipe_change(self) -> None:
        rec = Recommendation(
            recommendation_id="rec_1",
            created_at=1,
            updated_at=1,
            expires_at=10,
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_a",
            grind_delta_steps=1,
            grind_delta_um=12.5,
            next_grind_steps=43,
            next_grind_um=537.5,
            next_dose_g=18.0,
            target_yield_g=36.0,
            target_ratio=2.0,
            mode=RecommendationMode.ZERO_IMMEDIATE_BO,
            confidence=0.3,
            reason="test",
        )
        self.assertTrue(check_recommendation_staleness(rec, now=10, bean_context_id="bean_a").stale)
        self.assertTrue(check_recommendation_staleness(rec, now=5, bean_context_id="bean_b").stale)
        self.assertTrue(
            check_recommendation_staleness(
                rec,
                now=5,
                bean_context_id="bean_a",
                current_recipe=Recipe(
                    grind_steps=48,
                    grinder_step_size_um=12.5,
                    dose_g=18.0,
                    target_yield_g=36.0,
                ),
            ).stale
        )


if __name__ == "__main__":
    unittest.main()

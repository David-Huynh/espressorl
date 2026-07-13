from __future__ import annotations

import math
import unittest

import torch

from espresso_rl.domain.cpbo import (
    OptimizationRunContext,
    PhysicalShotStatus,
    PreferenceComparison,
    PreferenceLabel,
    PreferenceShot,
    RecipeDomain,
    RecipeParameter,
    RecipePoint,
    RecipeSpace,
)
from espresso_rl.domain.models import GrinderStepDirection
from espresso_rl.domain.taste_goal import TasteGoal
from espresso_rl.optimizers.cpbo_jnd import jnd_probabilities
from espresso_rl.optimizers.cpbo_config import (
    cpbo_config_from_dict,
    paper_fidelity_cpbo_config,
)


class RecipeSpaceTests(unittest.TestCase):
    def test_paper_and_application_profiles_keep_mes_but_expose_scale_assumptions(self) -> None:
        paper = paper_fidelity_cpbo_config()
        application = cpbo_config_from_dict({"model": {"sigma_pref": 0.35}})
        self.assertEqual(paper.acquisition.maximum_strategy, "paper_gumbel")
        self.assertEqual(paper.acquisition.gumbel_maximum_samples, 25_000)
        self.assertEqual(paper.acquisition.maximum_value_bins, 20)
        self.assertEqual(paper.acquisition.truncated_samples_per_bin, 1_000)
        self.assertEqual(application.model.sigma_pref, 0.35)
        with self.assertRaisesRegex(ValueError, "unknown CPBO"):
            cpbo_config_from_dict({"ordinary_acquisition": "EI"})
        with self.assertRaisesRegex(ValueError, "profile_name"):
            cpbo_config_from_dict({"profile_name": "typo"})

    def test_effective_configuration_version_tracks_every_runtime_setting(self) -> None:
        first = cpbo_config_from_dict({"model": {"sigma_pref": 0.20}})
        equivalent = cpbo_config_from_dict({"model": {"sigma_pref": 0.20}})
        changed = cpbo_config_from_dict({"model": {"sigma_pref": 0.21}})
        self.assertEqual(
            first.effective_configuration_version,
            equivalent.effective_configuration_version,
        )
        self.assertNotEqual(
            first.effective_configuration_version,
            changed.effective_configuration_version,
        )

    def test_fineness_always_increases_toward_finer_setting(self) -> None:
        finer_scale = recipe_space(GrinderStepDirection.HIGHER_IS_FINER)
        coarser_scale = recipe_space(GrinderStepDirection.HIGHER_IS_COARSER)
        self.assertLess(
            finer_scale.normalize_recipe(2.0, 18.0, 36.0)[0],
            finer_scale.normalize_recipe(8.0, 18.0, 36.0)[0],
        )
        self.assertLess(
            coarser_scale.normalize_recipe(8.0, 18.0, 36.0)[0],
            coarser_scale.normalize_recipe(2.0, 18.0, 36.0)[0],
        )

    def test_quantization_precedes_recipe_identity(self) -> None:
        space = recipe_space(GrinderStepDirection.HIGHER_IS_FINER)
        first = RecipePoint.create("run", space, 4.49, 18.04, 35.96, created_at=1)
        second = RecipePoint.create("run", space, 4.1, 18.0, 36.0, created_at=2)
        self.assertEqual(first.recipe_id, second.recipe_id)
        self.assertEqual(first.grind_size, 4.0)
        self.assertEqual(first.dose_g, 18.0)
        self.assertEqual(first.target_output_g, 36.0)
        self.assertAlmostEqual(first.brew_ratio, 2.0)

    def test_brew_ratio_is_derived_without_an_independent_ratio_bound(self) -> None:
        space = recipe_space(GrinderStepDirection.HIGHER_IS_FINER)
        point = RecipePoint.create("run", space, 5.0, 18.0, 38.0, created_at=1)
        self.assertAlmostEqual(point.brew_ratio, 38.0 / 18.0)
        one_to_one = RecipePoint.create("run", space, 5.0, 20.0, 20.0, created_at=1)
        self.assertEqual(one_to_one.brew_ratio, 1.0)

    def test_recipe_domain_accepts_wide_defaults_but_rejects_abusive_extremes(self) -> None:
        domain = RecipeDomain()
        self.assertEqual(domain.dose_min_g, 6.0)
        self.assertEqual(domain.dose_max_g, 30.0)
        self.assertEqual(domain.target_output_max_g, 250.0)
        with self.assertRaisesRegex(ValueError, "integrity envelope"):
            RecipeDomain(dose_max_g=101.0)
        with self.assertRaisesRegex(ValueError, "integrity envelope"):
            RecipeDomain(target_output_max_g=1_001.0)

    def test_context_fingerprint_partitions_material_context(self) -> None:
        base = OptimizationRunContext("install", "machine", "bean", "grinder", "profile")
        other_profile = OptimizationRunContext("install", "machine", "bean", "grinder", "other")
        other_water = OptimizationRunContext(
            "install", "machine", "bean", "grinder", "profile", water_id="water_b"
        )
        self.assertNotEqual(base.fingerprint, other_profile.fingerprint)
        self.assertNotEqual(base.fingerprint, other_water.fingerprint)

    def test_context_fingerprint_partitions_taste_goal_without_changing_recipe_space(self) -> None:
        balanced = OptimizationRunContext("install", "machine", "bean", "grinder", "profile")
        sweet = OptimizationRunContext(
            "install",
            "machine",
            "bean",
            "grinder",
            "profile",
            taste_goal=TasteGoal.custom({"sweet": "high", "bitter": "low"}),
        )
        self.assertNotEqual(balanced.fingerprint, sweet.fingerprint)
        self.assertEqual(
            sweet.taste_goal.to_dict(),
            {
                "schema_version": 1,
                "mode": "custom",
                "targets": {"sweet": "high", "bitter": "low"},
            },
        )

    def test_taste_goal_schema_is_strict_and_canonical(self) -> None:
        first = TasteGoal.custom({"bitter": "low", "sweet": "high"})
        second = TasteGoal.custom({"sweet": "high", "bitter": "low"})
        self.assertEqual(first, second)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.summary, "sweet high, bitter low")
        for invalid in (
            {"schema_version": 1, "mode": "balanced", "targets": {"sweet": "high"}},
            {"schema_version": 1, "mode": "custom", "targets": {}},
            {"schema_version": 1, "mode": "custom", "targets": {"unknown": "high"}},
            {"schema_version": 1, "mode": "custom", "targets": {"sweet": "maximum"}},
            {"schema_version": 1, "mode": "balanced", "targets": {}, "extra": True},
        ):
            with self.assertRaises(ValueError):
                TasteGoal.from_dict(invalid)

    def test_repeated_recipe_shots_are_distinct_physical_records(self) -> None:
        first = PreferenceShot(
            "shot_1",
            "recipe_1",
            "run_1",
            1,
            1,
            2,
            PhysicalShotStatus.VALID,
            False,
        )
        second = PreferenceShot(
            "shot_2",
            "recipe_1",
            "run_1",
            2,
            3,
            4,
            PhysicalShotStatus.VALID,
            False,
        )
        comparison = PreferenceComparison(
            "cmp",
            "run_1",
            second.shot_id,
            first.shot_id,
            PreferenceLabel.TIE,
            "global_previous",
            5,
        )
        self.assertNotEqual(first.shot_id, second.shot_id)
        self.assertEqual(first.recipe_id, second.recipe_id)
        self.assertEqual(comparison.label, PreferenceLabel.TIE)


class JNDLikelihoodTests(unittest.TestCase):
    def test_probabilities_sum_to_one_and_stay_in_unit_interval(self) -> None:
        difference = torch.linspace(-5.0, 5.0, 101, dtype=torch.float64)
        probabilities = jnd_probabilities(difference, gamma=0.3, sigma_pref=0.2)
        self.assertTrue(torch.all(probabilities >= 0.0))
        self.assertTrue(torch.all(probabilities <= 1.0))
        self.assertTrue(
            torch.allclose(
                probabilities.sum(-1),
                torch.ones(101, dtype=torch.float64),
                atol=1e-12,
            )
        )

    def test_swapping_new_and_anchor_swaps_preference_outcomes(self) -> None:
        difference = torch.tensor([-1.3, -0.2, 0.0, 0.7], dtype=torch.float64)
        forward = jnd_probabilities(difference, gamma=0.25, sigma_pref=0.3)
        reversed_pair = jnd_probabilities(-difference, gamma=0.25, sigma_pref=0.3)
        self.assertTrue(torch.allclose(forward[:, 0], reversed_pair[:, 2], atol=1e-12))
        self.assertTrue(torch.allclose(forward[:, 2], reversed_pair[:, 0], atol=1e-12))
        self.assertTrue(torch.allclose(forward[:, 1], reversed_pair[:, 1], atol=1e-12))

    def test_tie_is_symmetric_and_gamma_zero_is_binary_probit(self) -> None:
        difference = torch.tensor([-1.0, -0.3, 0.0, 0.3, 1.0], dtype=torch.float64)
        with_tie = jnd_probabilities(difference, gamma=0.4, sigma_pref=0.2)
        self.assertTrue(torch.allclose(with_tie[:, 1], torch.flip(with_tie[:, 1], (0,)), atol=1e-12))
        binary = jnd_probabilities(difference, gamma=0.0, sigma_pref=0.2)
        expected_new = torch.special.ndtr(difference / (math.sqrt(2.0) * 0.2))
        self.assertTrue(
            torch.allclose(binary[:, 1], torch.zeros(5, dtype=torch.float64), atol=1e-15)
        )
        self.assertTrue(torch.allclose(binary[:, 0], expected_new, atol=1e-12))
        self.assertTrue(torch.allclose(binary[:, 2], 1.0 - expected_new, atol=1e-12))

    def test_positive_difference_increases_new_better_probability(self) -> None:
        difference = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        probabilities = jnd_probabilities(difference, gamma=0.2, sigma_pref=0.2)
        self.assertTrue(torch.all(torch.diff(probabilities[:, 0]) > 0))

    def test_increasing_gamma_increases_tie_probability_near_zero(self) -> None:
        difference = torch.tensor([0.0], dtype=torch.float64)
        low = jnd_probabilities(difference, gamma=0.1, sigma_pref=0.2)
        high = jnd_probabilities(difference, gamma=0.5, sigma_pref=0.2)
        self.assertGreater(float(high[0, 1]), float(low[0, 1]))


def recipe_space(direction: GrinderStepDirection) -> RecipeSpace:
    return RecipeSpace(
        grind=RecipeParameter("grind_size", 0.0, 10.0, 1.0, "step"),
        dose=RecipeParameter("dose_g", 14.0, 22.0, 0.1, "g"),
        target_output=RecipeParameter("target_output_g", 20.0, 60.0, 0.1, "g"),
        grinder_step_direction=direction,
    )


if __name__ == "__main__":
    unittest.main()

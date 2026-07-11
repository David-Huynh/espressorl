from __future__ import annotations

import inspect
import unittest
from dataclasses import replace

import torch

from espresso_rl.domain.cpbo import (
    PreferenceLabel,
    RecipeParameter,
    RecipePoint,
    RecipeSpace,
    TrustRegionState,
)
from espresso_rl.domain.models import GrinderStepDirection
from espresso_rl.optimizers.cpbo_candidates import build_candidate_domain
from espresso_rl.optimizers.cpbo_config import MESConfig, TrustRegionConfig
from espresso_rl.optimizers.cpbo_mes import (
    approximate_maximum_distribution,
    evaluate_cpbo_mes,
    predictive_outcome_probabilities,
)
from espresso_rl.optimizers.cpbo_truncation import sample_upper_truncated_bivariate_gaussian
from espresso_rl.optimizers.cpbo_trust_region import trust_region_bounds, update_trust_region
import espresso_rl.optimizers.cpbo_mes as cpbo_mes_module


class MESAcquisitionTests(unittest.TestCase):
    def test_acquisition_is_finite_three_outcome_mes(self) -> None:
        mean, covariance = posterior_fixture()
        result = evaluate_cpbo_mes(
            posterior_mean=mean,
            posterior_covariance=covariance,
            candidate_indices=(1, 2, 3),
            anchor_index=0,
            gamma=0.2,
            sigma_pref=0.2,
            config=fast_mes_config(),
            seed=1,
            covariance_jitter=1e-6,
        )
        self.assertGreaterEqual(result.acquisition_value, 0.0)
        self.assertEqual(
            set(result.outcome_probabilities),
            {"new_better", "tie", "anchor_better"},
        )
        self.assertAlmostEqual(sum(result.outcome_probabilities.values()), 1.0, places=8)

    def test_candidate_anchor_covariance_changes_predictive_distribution(self) -> None:
        mean = torch.tensor([0.0, 0.4], dtype=torch.float64)
        independent = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        correlated = torch.tensor([[1.0, 0.9], [0.9, 1.0]], dtype=torch.float64)
        first = predictive_outcome_probabilities(
            mean,
            independent,
            candidate_indices=(1,),
            anchor_index=0,
            gamma=0.2,
            sigma_pref=0.2,
            variance_roundoff_floor=1e-12,
        )
        second = predictive_outcome_probabilities(
            mean,
            correlated,
            candidate_indices=(1,),
            anchor_index=0,
            gamma=0.2,
            sigma_pref=0.2,
            variance_roundoff_floor=1e-12,
        )
        self.assertFalse(torch.allclose(first, second))

    def test_local_dual_truncation_enforces_candidate_and_anchor_bounds(self) -> None:
        samples, diagnostics = sample_upper_truncated_bivariate_gaussian(
            mean=torch.tensor([0.8, 0.7], dtype=torch.float64),
            covariance=torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
            upper=torch.tensor([0.1, -0.1], dtype=torch.float64),
            sample_count=200,
            seed=3,
            rejection_batch_size=64,
            rejection_max_batches=2,
            rejection_min_acceptance=0.2,
            gibbs_burn_in=30,
            gibbs_thinning=2,
            jitter=1e-8,
        )
        self.assertTrue(torch.all(samples[:, 0] <= 0.1 + 1e-10))
        self.assertTrue(torch.all(samples[:, 1] <= -0.1 + 1e-10))
        self.assertIn(diagnostics.method, {"rejection", "gibbs", "rejection_then_gibbs"})

    def test_direct_and_gumbel_maximum_strategies_agree_qualitatively(self) -> None:
        mean, covariance = posterior_fixture()
        direct_config = replace(
            fast_mes_config(),
            maximum_strategy="direct_max_samples",
            posterior_max_function_samples=1_000,
        )
        gumbel_config = replace(
            fast_mes_config(),
            maximum_strategy="paper_gumbel",
            posterior_max_function_samples=1_000,
            gumbel_maximum_samples=2_000,
        )
        direct = approximate_maximum_distribution(
            mean - mean.mean(), covariance, config=direct_config, seed=19, jitter=1e-6
        )
        gumbel = approximate_maximum_distribution(
            mean - mean.mean(), covariance, config=gumbel_config, seed=19, jitter=1e-6
        )
        direct_mean = float(torch.sum(direct.representative_values * direct.weights))
        gumbel_mean = float(torch.sum(gumbel.representative_values * gumbel.weights))
        self.assertGreater(direct_mean, 0.0)
        self.assertGreater(gumbel_mean, 0.0)
        self.assertLess(abs(direct_mean - gumbel_mean), 0.75)

    def test_cpbo_path_does_not_import_standard_bo_acquisitions(self) -> None:
        source = inspect.getsource(cpbo_mes_module)
        for forbidden in (
            "ExpectedImprovement",
            "UpperConfidenceBound",
            "ThompsonSampling",
            "qExpectedImprovement",
            "AnalyticExpectedUtilityOfBestOption",
            "qExpectedUtilityOfBestOption",
        ):
            self.assertNotIn(forbidden, source)

    def test_candidate_is_quantized_and_never_anchor_or_accidental_duplicate(self) -> None:
        space = recipe_space()
        anchor = RecipePoint.create("run", space, 5.0, 18.0, 36.0, created_at=1)
        observed = RecipePoint.create("run", space, 6.0, 18.0, 36.0, created_at=2)
        domain = build_candidate_domain(
            run_id="run",
            recipe_space=space,
            evaluated_recipes=(anchor, observed),
            anchor_recipe=anchor,
            config=fast_mes_config(),
            seed=4,
            created_at=3,
        )
        ids = [recipe.recipe_id for recipe in domain.proposal_recipes]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn(anchor.recipe_id, ids)
        self.assertNotIn(observed.recipe_id, ids)
        self.assertTrue(all(recipe.grind_size.is_integer() for recipe in domain.proposal_recipes))


class TrustRegionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TrustRegionConfig()
        self.state = TrustRegionState(center=(0.5, 0.5, 0.5))

    def test_success_and_failure_counters_reset_each_other(self) -> None:
        success = update_trust_region(
            self.state,
            PreferenceLabel.NEW_BETTER,
            candidate_center=(0.6, 0.5, 0.5),
            config=self.config,
        )
        self.assertEqual(success.success_count, 1)
        self.assertEqual(success.failure_count, 0)
        loss = update_trust_region(
            success,
            PreferenceLabel.ANCHOR_BETTER,
            candidate_center=(0.7, 0.5, 0.5),
            config=self.config,
        )
        self.assertEqual(loss.success_count, 0)
        self.assertEqual(loss.failure_count, 1)
        tie = update_trust_region(
            success,
            PreferenceLabel.TIE,
            candidate_center=(0.7, 0.5, 0.5),
            config=self.config,
        )
        self.assertEqual(tie.center, success.center)
        self.assertEqual(tie.failure_count, 1)

    def test_expansion_and_contraction_thresholds(self) -> None:
        state = self.state
        for index in range(3):
            state = update_trust_region(
                state,
                PreferenceLabel.NEW_BETTER,
                candidate_center=(0.5 + index * 0.01, 0.5, 0.5),
                config=self.config,
            )
        self.assertEqual(state.length, 1.6)
        for _ in range(4):
            state = update_trust_region(
                state,
                PreferenceLabel.ANCHOR_BETTER,
                candidate_center=(0.2, 0.2, 0.2),
                config=self.config,
            )
        self.assertEqual(state.length, 0.8)

    def test_bounds_are_lengthscale_shaped_and_feasible(self) -> None:
        lower, upper = trust_region_bounds(
            self.state,
            torch.tensor([0.2, 1.0, 2.0], dtype=torch.float64),
            self.config,
        )
        self.assertTrue(all(0.0 <= value <= 1.0 for value in (*lower, *upper)))
        self.assertTrue(all(low <= high for low, high in zip(lower, upper)))
        self.assertNotAlmostEqual(upper[0] - lower[0], upper[2] - lower[2])

    def test_restart_requests_one_full_domain_iteration_without_moving_incumbent(self) -> None:
        tiny = TrustRegionState(
            center=(0.4, 0.5, 0.6),
            length=self.config.minimum_length * 1.1,
            failure_count=3,
        )
        restarted = update_trust_region(
            tiny,
            PreferenceLabel.TIE,
            candidate_center=(0.9, 0.9, 0.9),
            config=self.config,
        )
        self.assertTrue(restarted.restart_pending)
        self.assertEqual(restarted.center, tiny.center)
        self.assertEqual(restarted.length, self.config.initial_length)
        after_restart_loss = update_trust_region(
            restarted,
            PreferenceLabel.ANCHOR_BETTER,
            candidate_center=(0.9, 0.9, 0.9),
            config=self.config,
        )
        self.assertFalse(after_restart_loss.restart_pending)
        self.assertEqual(after_restart_loss.center, tiny.center)


def fast_mes_config() -> MESConfig:
    return MESConfig(
        sobol_candidate_count=24,
        local_candidate_count=8,
        posterior_max_function_samples=128,
        gumbel_maximum_samples=300,
        maximum_value_bins=5,
        truncated_samples_per_bin=96,
        candidate_chunk_size=4,
        rejection_batch_size=512,
        rejection_max_batches=4,
        gibbs_burn_in=30,
    )


def posterior_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    locations = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.float64)
    covariance = torch.exp(-((locations - locations.T) ** 2) / 0.3)
    covariance += torch.eye(4, dtype=torch.float64) * 1e-6
    return torch.zeros(4, dtype=torch.float64), covariance


def recipe_space() -> RecipeSpace:
    return RecipeSpace(
        RecipeParameter("grind_size", 0.0, 10.0, 1.0, "step"),
        RecipeParameter("dose_g", 14.0, 22.0, 0.1, "g"),
        RecipeParameter("target_output_g", 20.0, 60.0, 0.1, "g"),
        GrinderStepDirection.HIGHER_IS_FINER,
        1.2,
        3.5,
    )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import torch

from espresso_rl.domain.cpbo import RecipeParameter, RecipePoint, RecipeSpace
from espresso_rl.domain.models import FixedCadenceShotSequence, GrinderStepDirection
from espresso_rl.optimizers.cpbo_config import (
    PhysicsProxyConfig,
    PreferenceGPConfig,
    TraceSurrogateConfig,
)
from espresso_rl.optimizers.cpbo_kernel import (
    PhysicsInformedAdditiveKernel,
    assert_fixed_output_scale,
)
from espresso_rl.optimizers.cpbo_model import fit_preference_gp
from espresso_rl.optimizers.cpbo_physics import PHYSICS_FEATURE_NAMES, phi0
from espresso_rl.optimizers.cpbo_trace import (
    TRACE_FEATURE_NAMES,
    IndependentTraceSurrogate,
    expected_uncertain_rbf,
    extract_trace_features,
)


class PhysicsKernelTests(unittest.TestCase):
    def test_physics_features_are_finite_at_valid_bounds_and_fallback_is_explicit(self) -> None:
        space = recipe_space()
        config = PhysicsProxyConfig(basket_diameter_mm=None)
        for normalized in ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5)):
            try:
                physical = space.inverse_recipe(normalized)
            except ValueError:
                continue
            result = phi0(RecipePoint.create("run", space, *physical, created_at=1), config)
            self.assertEqual(len(result.values), len(PHYSICS_FEATURE_NAMES))
            self.assertTrue(all(np.isfinite(result.values)))
            self.assertTrue(result.fallback_bed_depth)
            self.assertIn("bed_depth_proxy_fallback_standardized_dose", result.diagnostics)

    def test_additive_kernel_is_symmetric_psd_and_weights_form_fixed_scale_simplex(self) -> None:
        config = PreferenceGPConfig()
        kernel = PhysicsInformedAdditiveKernel(
            physics_dimensions=len(PHYSICS_FEATURE_NAMES),
            trace_dimensions=0,
            model_config=config,
        ).to(dtype=torch.float64)
        x = torch.randn((8, 3 + len(PHYSICS_FEATURE_NAMES)), dtype=torch.float64)
        covariance = kernel(x, x).to_dense()
        eigenvalues = torch.linalg.eigvalsh((covariance + covariance.T) / 2.0)
        self.assertTrue(torch.allclose(covariance, covariance.T, atol=1e-10))
        self.assertGreaterEqual(float(torch.min(eigenvalues).detach()), -1e-8)
        weights = kernel.weights_dict()
        self.assertTrue(all(value >= 0.0 for value in weights.values()))
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=10)
        self.assertEqual(weights["trace"], 0.0)
        assert_fixed_output_scale(kernel)
        self.assertTrue(torch.allclose(torch.diagonal(covariance), torch.ones(8, dtype=torch.float64)))

    def test_variational_preference_gp_warm_starts_from_safe_checkpoint(self) -> None:
        config = PreferenceGPConfig(
            fit_steps=3,
            likelihood_samples=4,
            early_stopping_patience=2,
        )
        inputs = torch.randn((4, 3 + len(PHYSICS_FEATURE_NAMES)), dtype=torch.float64)
        comparisons = torch.tensor([[1, 0], [2, 1], [3, 2]], dtype=torch.long)
        labels = torch.tensor([0, 1, 2], dtype=torch.long)
        first = fit_preference_gp(
            train_inputs=inputs,
            comparison_indices=comparisons,
            labels=labels,
            physics_dimensions=len(PHYSICS_FEATURE_NAMES),
            trace_dimensions=0,
            config=config,
            random_seed=2,
        )
        second = fit_preference_gp(
            train_inputs=inputs,
            comparison_indices=comparisons,
            labels=labels,
            physics_dimensions=len(PHYSICS_FEATURE_NAMES),
            trace_dimensions=0,
            config=config,
            warm_start_checkpoint=first.checkpoint_json,
            random_seed=3,
        )
        self.assertGreater(float(second.likelihood.gamma.detach()), 0.0)
        self.assertEqual(float(second.likelihood.sigma_pref), config.sigma_pref)
        self.assertNotIn("pickle", second.checkpoint_json.lower())
        assert_fixed_output_scale(second.model.covar_module)


class TraceSurrogateTests(unittest.TestCase):
    def test_trace_summary_is_fixed_length_and_finite(self) -> None:
        sequence = fixed_sequence()
        features = extract_trace_features(sequence, TraceSurrogateConfig())
        self.assertEqual(len(features.values), len(TRACE_FEATURE_NAMES))
        self.assertTrue(all(np.isfinite(features.values)))

    def test_missing_telemetry_keeps_trace_surrogate_disabled(self) -> None:
        config = TraceSurrogateConfig(minimum_valid_telemetry_shots=4)
        surrogate = IndependentTraceSurrogate(config)
        prediction = surrogate.predict(torch.zeros((2, 3), dtype=torch.float64))
        self.assertFalse(prediction.enabled)
        self.assertEqual(prediction.mean.shape, (2, 0))

    def test_activation_threshold_and_candidate_features_come_from_predictions(self) -> None:
        config = TraceSurrogateConfig(
            minimum_valid_telemetry_shots=3,
            fit_steps=3,
            early_stopping_patience=2,
            validation_max_standardized_rmse=100.0,
        )
        surrogate = IndependentTraceSurrogate(config)
        x = torch.tensor(
            [[0.0, 0.2, 0.3], [0.4, 0.5, 0.6], [1.0, 0.8, 0.7]],
            dtype=torch.float64,
        )
        rows = torch.stack(
            [torch.linspace(index, index + 1.0, len(TRACE_FEATURE_NAMES)) for index in range(3)]
        ).to(torch.float64)
        self.assertTrue(surrogate.fit(x, rows))
        candidate = surrogate.predict(torch.tensor([[0.25, 0.4, 0.5]], dtype=torch.float64))
        self.assertTrue(candidate.enabled)
        self.assertEqual(candidate.mean.shape, (1, len(TRACE_FEATURE_NAMES)))
        self.assertEqual(candidate.variance.shape, candidate.mean.shape)
        self.assertTrue(torch.all(candidate.variance > 0.0))

    def test_expected_uncertain_rbf_is_symmetric_psd(self) -> None:
        mean = torch.tensor([[0.0, 0.0], [0.5, -0.2], [1.0, 0.4]], dtype=torch.float64)
        variance = torch.full_like(mean, 0.05)
        lengthscales = torch.tensor([0.7, 1.2], dtype=torch.float64)
        covariance = expected_uncertain_rbf(mean, variance, mean, variance, lengthscales)
        eigenvalues = torch.linalg.eigvalsh((covariance + covariance.T) / 2.0)
        self.assertTrue(torch.allclose(covariance, covariance.T, atol=1e-12))
        self.assertGreaterEqual(float(torch.min(eigenvalues)), -1e-8)

    def test_trace_uncertainty_prevents_unit_confidence_for_distinct_distributions(self) -> None:
        mean_x = torch.tensor([[0.0]], dtype=torch.float64)
        mean_y = torch.tensor([[0.6]], dtype=torch.float64)
        certain = torch.zeros_like(mean_x)
        uncertain = torch.full_like(mean_x, 0.5)
        lengthscale = torch.tensor([1.0], dtype=torch.float64)
        uncertain_similarity = expected_uncertain_rbf(
            mean_x,
            uncertain,
            mean_y,
            uncertain,
            lengthscale,
        )
        self.assertLess(float(uncertain_similarity), 1.0)
        self.assertTrue(torch.isfinite(uncertain_similarity).all())


def recipe_space() -> RecipeSpace:
    return RecipeSpace(
        RecipeParameter("grind_size", 0.0, 10.0, 1.0, "step"),
        RecipeParameter("dose_g", 14.0, 22.0, 0.1, "g"),
        RecipeParameter("target_output_g", 20.0, 60.0, 0.1, "g"),
        GrinderStepDirection.HIGHER_IS_FINER,
        1.2,
        3.5,
    )


def fixed_sequence() -> FixedCadenceShotSequence:
    steps = 12
    return FixedCadenceShotSequence(
        sample_interval_ms=250,
        pressure_bar=np.linspace(0.0, 9.0, steps),
        pressure_target_bar=np.linspace(1.0, 9.0, steps),
        pump_flow_ml_s=np.linspace(0.0, 3.0, steps),
        pump_flow_target_ml_s=np.linspace(1.0, 3.0, steps),
        beverage_flow_g_s=np.linspace(0.0, 2.0, steps),
        weight_g=np.linspace(0.0, 6.0, steps),
        temperature_c=np.linspace(92.0, 93.0, steps),
        temperature_target_c=np.full(steps, 93.0),
        pump_target_mode=np.ones(steps, dtype=np.uint8),
        valve_open=np.ones(steps, dtype=np.uint8),
    )


if __name__ == "__main__":
    unittest.main()

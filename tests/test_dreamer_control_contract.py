from __future__ import annotations

import unittest

from espresso_rl.domain.dreamer_control import (
    DreamerControlSafetyLimits,
    DreamerControlSpec,
    expand_decision_actions_to_observation_steps,
    validate_dynamic_action_for_control_spec,
)
from espresso_rl.domain.optimization import (
    OPTIMIZER_FAMILY_BAYESIAN_OPTIMIZATION,
    OPTIMIZER_FAMILY_DREAMER_V3,
    OPTIMIZER_FAMILY_TRUST_REGION_BO,
    OPTIMIZER_FAMILY_TRUST_REGION_PPO,
    optimizer_family_allows_adaptive_profile_control,
    require_adaptive_profile_control_optimizer,
)


class DreamerControlContractTests(unittest.TestCase):
    def test_default_control_spec_separates_observation_and_decision_cadence(self) -> None:
        spec = DreamerControlSpec()

        self.assertEqual(spec.observation_interval_ms, 250)
        self.assertEqual(spec.decision_interval_ms, 1000)
        self.assertEqual(spec.decision_step_count, 4)
        self.assertTrue(spec.is_decision_step(0))
        self.assertFalse(spec.is_decision_step(1))
        self.assertTrue(spec.is_decision_step(4))
        self.assertFalse(spec.dynamic_control_enabled)
        self.assertEqual(spec.action_capability_mask(), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_control_spec_rejects_implausibly_fast_cadence(self) -> None:
        with self.assertRaisesRegex(ValueError, "sensor cadence"):
            DreamerControlSpec(observation_interval_ms=100)
        with self.assertRaisesRegex(ValueError, "control cadence"):
            DreamerControlSpec(decision_interval_ms=250)
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            DreamerControlSpec(observation_interval_ms=250, decision_interval_ms=700)

    def test_adaptive_profile_control_is_dreamer_only(self) -> None:
        self.assertTrue(optimizer_family_allows_adaptive_profile_control(OPTIMIZER_FAMILY_DREAMER_V3))
        self.assertFalse(
            optimizer_family_allows_adaptive_profile_control(OPTIMIZER_FAMILY_BAYESIAN_OPTIMIZATION)
        )
        self.assertFalse(optimizer_family_allows_adaptive_profile_control(OPTIMIZER_FAMILY_TRUST_REGION_BO))
        self.assertFalse(optimizer_family_allows_adaptive_profile_control(OPTIMIZER_FAMILY_TRUST_REGION_PPO))

        require_adaptive_profile_control_optimizer(OPTIMIZER_FAMILY_DREAMER_V3)
        with self.assertRaisesRegex(ValueError, "only available for DreamerV3"):
            require_adaptive_profile_control_optimizer(OPTIMIZER_FAMILY_BAYESIAN_OPTIMIZATION)

        with self.assertRaisesRegex(ValueError, "only available for Dreamer"):
            DreamerControlSpec(
                optimizer_family=OPTIMIZER_FAMILY_BAYESIAN_OPTIMIZATION,
                dynamic_control_enabled=True,
                pressure_control_allowed=True,
            )

    def test_dynamic_action_validation_enforces_capability_cadence_and_safety(self) -> None:
        spec = DreamerControlSpec(
            dynamic_control_enabled=True,
            pressure_control_allowed=True,
            stop_control_allowed=True,
        )

        self.assertEqual(
            validate_dynamic_action_for_control_spec(
                {"pressure_target_bar": 8.5, "stop": False},
                control_spec=spec,
                step_index=0,
            ),
            [],
        )
        self.assertTrue(
            any(
                "decision steps" in error
                for error in validate_dynamic_action_for_control_spec(
                    {"pressure_target_bar": 8.5},
                    control_spec=spec,
                    step_index=1,
                )
            )
        )
        self.assertTrue(
            any(
                "flow_target_ml_s" in error
                for error in validate_dynamic_action_for_control_spec(
                    {"flow_target_ml_s": 2.0},
                    control_spec=spec,
                    step_index=0,
                )
            )
        )
        self.assertTrue(
            any(
                "outside safety limits" in error
                for error in validate_dynamic_action_for_control_spec(
                    {"pressure_target_bar": 20.0},
                    control_spec=spec,
                    step_index=0,
                )
            )
        )

    def test_dynamic_action_rejects_simultaneous_pressure_and_flow_targets(self) -> None:
        spec = DreamerControlSpec(
            dynamic_control_enabled=True,
            pressure_control_allowed=True,
            flow_control_allowed=True,
        )

        errors = validate_dynamic_action_for_control_spec(
            {"pressure_target_bar": 8.5, "flow_target_ml_s": 2.0},
            control_spec=spec,
            step_index=0,
        )

        self.assertTrue(any("pressure and flow" in error for error in errors))

    def test_safety_limits_reject_crazy_yield_and_temperature_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "100g"):
            DreamerControlSafetyLimits(max_yield_stop_target_g=120.0)
        with self.assertRaisesRegex(ValueError, "105C"):
            DreamerControlSafetyLimits(max_temperature_c=120.0)

    def test_decision_actions_are_held_across_observation_steps(self) -> None:
        spec = DreamerControlSpec(
            decision_interval_ms=500,
            dynamic_control_enabled=True,
            pressure_control_allowed=True,
            stop_control_allowed=True,
        )

        expanded = expand_decision_actions_to_observation_steps(
            [{"pressure_target_bar": 2.0}, {"pressure_target_bar": 8.0}, {"stop": True}],
            step_count=5,
            control_spec=spec,
        )

        self.assertEqual(
            expanded,
            [
                {"pressure_target_bar": 2.0},
                {"pressure_target_bar": 2.0},
                {"pressure_target_bar": 8.0},
                {"pressure_target_bar": 8.0},
                {"stop": True},
            ],
        )


if __name__ == "__main__":
    unittest.main()

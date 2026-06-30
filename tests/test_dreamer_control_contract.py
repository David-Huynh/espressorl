from __future__ import annotations

import unittest

from espresso_rl.domain.dreamer_control import (
    DREAMER_DYNAMIC_CONTROL_ACCEPT,
    DREAMER_DYNAMIC_CONTROL_FAIL_SAFE,
    DREAMER_DYNAMIC_CONTROL_REPLAY_LAST,
    DREAMER_DYNAMIC_CONTROL_WAIT_FOR_FIRST_COMMAND,
    DreamerControlSafetyLimits,
    DreamerControlSpec,
    expand_decision_actions_to_observation_steps,
    resolve_live_dynamic_control_action,
    sanitize_dynamic_action_for_control_spec,
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
        self.assertEqual(spec.safety_limits.max_pressure_bar, 12.0)
        self.assertEqual(spec.safety_limits.min_temperature_c, 20.0)
        self.assertEqual(spec.safety_limits.max_temperature_c, 100.0)
        self.assertEqual(spec.safety_limits.max_yield_stop_target_g, 90.0)
        self.assertEqual(spec.safety_limits.max_shot_duration_s, 90.0)

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
        self.assertTrue(
            any(
                "outside safety limits" in error
                for error in validate_dynamic_action_for_control_spec(
                    {"yield_stop_target_g": 90.5},
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

    def test_safety_limits_reject_crazy_pressure_temperature_yield_and_duration_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "12 bar"):
            DreamerControlSafetyLimits(max_pressure_bar=12.5)
        with self.assertRaisesRegex(ValueError, "20C"):
            DreamerControlSafetyLimits(min_temperature_c=19.5)
        with self.assertRaisesRegex(ValueError, "100C"):
            DreamerControlSafetyLimits(max_temperature_c=100.5)
        with self.assertRaisesRegex(ValueError, "90g"):
            DreamerControlSafetyLimits(max_yield_stop_target_g=91.0)
        with self.assertRaisesRegex(ValueError, "90s"):
            DreamerControlSafetyLimits(max_shot_duration_s=91.0)

    def test_dynamic_action_sanitizer_clamps_finite_values_without_fallback_errors(self) -> None:
        spec = DreamerControlSpec(
            dynamic_control_enabled=True,
            pressure_control_allowed=True,
            temperature_control_allowed=True,
            stop_control_allowed=True,
        )

        sanitized = sanitize_dynamic_action_for_control_spec(
            {
                "pressure_target_bar": 13.5,
                "temperature_target_c": 10.0,
                "yield_stop_target_g": 120.0,
                "stop": False,
            },
            control_spec=spec,
            step_index=0,
        )

        self.assertTrue(sanitized.ok)
        self.assertEqual(sanitized.errors, ())
        self.assertEqual(set(sanitized.clamped_fields), {"pressure_target_bar", "temperature_target_c", "yield_stop_target_g"})
        self.assertEqual(
            sanitized.sanitized_action,
            {
                "pressure_target_bar": 12.0,
                "temperature_target_c": 20.0,
                "yield_stop_target_g": 90.0,
                "stop": False,
            },
        )

    def test_dynamic_action_sanitizer_rejects_malformed_or_conflicting_values(self) -> None:
        spec = DreamerControlSpec(
            dynamic_control_enabled=True,
            pressure_control_allowed=True,
            flow_control_allowed=True,
            stop_control_allowed=True,
        )

        conflict = sanitize_dynamic_action_for_control_spec(
            {"pressure_target_bar": 8.0, "flow_target_ml_s": 2.0},
            control_spec=spec,
            step_index=0,
        )
        bad_stop = sanitize_dynamic_action_for_control_spec(
            {"stop": 1},
            control_spec=spec,
            step_index=0,
        )

        self.assertFalse(conflict.ok)
        self.assertIsNone(conflict.sanitized_action)
        self.assertTrue(any("pressure and flow" in error for error in conflict.errors))
        self.assertFalse(bad_stop.ok)
        self.assertIsNone(bad_stop.sanitized_action)
        self.assertTrue(any("stop must be boolean" in error for error in bad_stop.errors))

    def test_live_dynamic_control_accepts_sanitized_commands_and_reports_clamps(self) -> None:
        spec = DreamerControlSpec(
            dynamic_control_enabled=True,
            pressure_control_allowed=True,
            temperature_control_allowed=True,
            stop_control_allowed=True,
        )

        decision = resolve_live_dynamic_control_action(
            {
                "pressure_target_bar": 13.5,
                "temperature_target_c": 10.0,
                "yield_stop_target_g": 95.0,
            },
            last_sanitized_action=None,
            control_spec=spec,
            step_index=0,
            milliseconds_since_last_command=0,
        )

        self.assertEqual(decision.status, DREAMER_DYNAMIC_CONTROL_ACCEPT)
        self.assertFalse(decision.fail_safe_required)
        self.assertEqual(
            decision.action,
            {
                "pressure_target_bar": 12.0,
                "temperature_target_c": 20.0,
                "yield_stop_target_g": 90.0,
            },
        )
        self.assertEqual(
            set(decision.clamped_fields),
            {"pressure_target_bar", "temperature_target_c", "yield_stop_target_g"},
        )

    def test_live_dynamic_control_replays_missed_commands_until_grace_expires(self) -> None:
        spec = DreamerControlSpec(
            dynamic_control_enabled=True,
            pressure_control_allowed=True,
        )
        last_action = {"pressure_target_bar": 8.0}

        replay = resolve_live_dynamic_control_action(
            None,
            last_sanitized_action=last_action,
            control_spec=spec,
            step_index=4,
            milliseconds_since_last_command=4_999,
        )
        stale = resolve_live_dynamic_control_action(
            None,
            last_sanitized_action=last_action,
            control_spec=spec,
            step_index=8,
            milliseconds_since_last_command=5_001,
        )

        self.assertEqual(replay.status, DREAMER_DYNAMIC_CONTROL_REPLAY_LAST)
        self.assertEqual(replay.action, last_action)
        self.assertFalse(replay.fail_safe_required)
        self.assertEqual(stale.status, DREAMER_DYNAMIC_CONTROL_FAIL_SAFE)
        self.assertEqual(stale.reason, "command_timeout")
        self.assertTrue(stale.fail_safe_required)

    def test_live_dynamic_control_waits_for_first_command_then_fails_safe(self) -> None:
        spec = DreamerControlSpec(
            dynamic_control_enabled=True,
            pressure_control_allowed=True,
        )

        waiting = resolve_live_dynamic_control_action(
            None,
            last_sanitized_action=None,
            control_spec=spec,
            step_index=0,
            milliseconds_since_last_command=5_000,
        )
        timeout = resolve_live_dynamic_control_action(
            None,
            last_sanitized_action=None,
            control_spec=spec,
            step_index=0,
            milliseconds_since_last_command=5_001,
        )

        self.assertEqual(waiting.status, DREAMER_DYNAMIC_CONTROL_WAIT_FOR_FIRST_COMMAND)
        self.assertFalse(waiting.fail_safe_required)
        self.assertEqual(timeout.status, DREAMER_DYNAMIC_CONTROL_FAIL_SAFE)
        self.assertEqual(timeout.reason, "initial_command_timeout")

    def test_live_dynamic_control_fails_safe_on_invalid_new_command(self) -> None:
        spec = DreamerControlSpec(
            dynamic_control_enabled=True,
            pressure_control_allowed=True,
            flow_control_allowed=True,
        )

        decision = resolve_live_dynamic_control_action(
            {"pressure_target_bar": 8.0, "flow_target_ml_s": 2.0},
            last_sanitized_action={"pressure_target_bar": 7.0},
            control_spec=spec,
            step_index=0,
            milliseconds_since_last_command=250,
        )

        self.assertEqual(decision.status, DREAMER_DYNAMIC_CONTROL_FAIL_SAFE)
        self.assertEqual(decision.reason, "invalid_command")
        self.assertTrue(decision.fail_safe_required)
        self.assertTrue(any("pressure and flow" in error for error in decision.errors))

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

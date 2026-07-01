from __future__ import annotations

import ast
import inspect
import math
import unittest

import espresso_rl.domain.dreamer_pre_shot as pre_shot_module
import espresso_rl.domain.dreamer_taste as taste_module
from espresso_rl.domain.dreamer_pre_shot import (
    DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC,
    DREAMER_PRE_SHOT_ACTION_FIELDS,
    DreamerPreShotActionSpec,
    build_dreamer_pre_shot_action,
    encode_dreamer_pre_shot_action,
    validate_dreamer_pre_shot_action,
)
from espresso_rl.domain.dreamer_taste import (
    DEFAULT_DREAMER_TASTE_OBJECTIVE_SPEC,
    DreamerTasteObjectiveSpec,
    validate_dreamer_taste_objective,
)


class DreamerPreShotActionTests(unittest.TestCase):
    def test_default_spec_round_trips_with_stable_field_order(self) -> None:
        payload = DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC.to_dict()

        parsed = DreamerPreShotActionSpec.from_dict(payload)

        self.assertEqual(parsed.to_dict(), payload)
        self.assertEqual(payload["action_fields"], list(DREAMER_PRE_SHOT_ACTION_FIELDS))
        self.assertEqual(payload["capability_fields"], list(DREAMER_PRE_SHOT_ACTION_FIELDS))

    def test_encoding_uses_deterministic_lower_bin_for_exact_tie(self) -> None:
        action = build_dreamer_pre_shot_action(
            values={"pressure_target_bar": 0.125},
            observed_fields={"pressure_target_bar"},
            capability_fields={"pressure_target_bar"},
        )

        values, indexes, observed, capabilities = encode_dreamer_pre_shot_action(action)
        field_index = DREAMER_PRE_SHOT_ACTION_FIELDS.index("pressure_target_bar")

        self.assertEqual(values[field_index], 0.0)
        self.assertEqual(indexes[field_index], 0)
        self.assertEqual(observed[field_index], 1.0)
        self.assertEqual(capabilities[field_index], 1.0)

    def test_unknown_fields_are_absent_and_masked_without_fabricated_values(self) -> None:
        action = build_dreamer_pre_shot_action(
            values={"dose_target_g": 18.0},
            observed_fields={"dose_target_g"},
            capability_fields={"grind_delta_steps_from_current", "dose_target_g"},
        )

        values, indexes, observed, capabilities = encode_dreamer_pre_shot_action(action)
        grind_index = DREAMER_PRE_SHOT_ACTION_FIELDS.index("grind_delta_steps_from_current")
        dose_index = DREAMER_PRE_SHOT_ACTION_FIELDS.index("dose_target_g")

        self.assertNotIn("grind_delta_steps_from_current", action["values"])
        self.assertEqual(values[grind_index], 0.0)
        self.assertEqual(indexes[grind_index], 0)
        self.assertEqual(observed[grind_index], 0.0)
        self.assertEqual(capabilities[grind_index], 1.0)
        self.assertEqual(observed[dose_index], 1.0)

    def test_rejects_nonfinite_out_of_bounds_and_conflicting_control_actions(self) -> None:
        nonfinite = build_dreamer_pre_shot_action(
            values={},
            observed_fields=set(),
            capability_fields={"temperature_target_c"},
        )
        nonfinite["values"]["temperature_target_c"] = math.nan
        nonfinite["observed"]["temperature_target_c"] = True
        self.assertTrue(any("must be finite" in error for error in validate_dreamer_pre_shot_action(nonfinite)))

        unsafe = build_dreamer_pre_shot_action(
            values={},
            observed_fields=set(),
            capability_fields={"pressure_target_bar"},
        )
        unsafe["values"]["pressure_target_bar"] = 12.5
        unsafe["observed"]["pressure_target_bar"] = True
        self.assertTrue(any("outside hard bounds" in error for error in validate_dreamer_pre_shot_action(unsafe)))

        conflict = build_dreamer_pre_shot_action(
            values={},
            observed_fields=set(),
            capability_fields={"pump_target_mode", "pressure_target_bar", "flow_target_ml_s"},
        )
        conflict["values"].update({"pump_target_mode": 1, "pressure_target_bar": 8.0, "flow_target_ml_s": 2.0})
        conflict["observed"].update(
            {"pump_target_mode": True, "pressure_target_bar": True, "flow_target_ml_s": True}
        )
        self.assertTrue(any("pressure mode" in error for error in validate_dreamer_pre_shot_action(conflict)))

        numeric_string = build_dreamer_pre_shot_action(
            values={},
            observed_fields=set(),
            capability_fields={"pump_target_mode"},
        )
        numeric_string["values"]["pump_target_mode"] = "1"
        numeric_string["observed"]["pump_target_mode"] = True
        self.assertTrue(any("must be finite" in error for error in validate_dreamer_pre_shot_action(numeric_string)))

    def test_custom_bin_contract_rejects_values_outside_its_declared_range(self) -> None:
        payload = DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC.to_dict()
        payload["bins"]["dose_target_g"] = [19.0, 20.0]
        spec = DreamerPreShotActionSpec.from_dict(payload)
        action = build_dreamer_pre_shot_action(
            values={"dose_target_g": 18.0},
            observed_fields={"dose_target_g"},
            capability_fields={"dose_target_g"},
        )

        errors = validate_dreamer_pre_shot_action(action, spec=spec)

        self.assertTrue(any("outside configured bins" in error for error in errors))

    def test_domain_contract_has_no_application_or_adapter_imports(self) -> None:
        imports: set[str] = set()
        for module in (pre_shot_module, taste_module):
            tree = ast.parse(inspect.getsource(module))
            imports.update(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            imports.update(
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            )

        self.assertFalse(any(name.startswith("espresso_rl.adapters") for name in imports))
        self.assertFalse(any(name.startswith("espresso_rl.application") for name in imports))

    def test_taste_objective_spec_is_versioned_and_validates_canonical_goals(self) -> None:
        payload = DEFAULT_DREAMER_TASTE_OBJECTIVE_SPEC.to_dict()
        self.assertEqual(DreamerTasteObjectiveSpec.from_dict(payload).to_dict(), payload)
        self.assertEqual(validate_dreamer_taste_objective({"mode": "auto"}), [])
        self.assertEqual(
            validate_dreamer_taste_objective({"mode": "explicit", "sweetness": "high"}),
            [],
        )
        errors = validate_dreamer_taste_objective({"mode": "explicit", "sweetness": "maximum"})
        self.assertTrue(any("sweetness" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

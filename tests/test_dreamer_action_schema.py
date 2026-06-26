from __future__ import annotations

import unittest

from espresso_rl.domain.dreamer_actions import (
    DreamerActionCandidate,
    dreamer_action_from_payload,
    dreamer_action_to_recipe,
)
from espresso_rl.domain.models import Recipe, SafetyBounds


class DreamerActionSchemaTests(unittest.TestCase):
    def test_action_can_use_full_safe_bo_envelope(self) -> None:
        current = Recipe(42, 12.5, 18.0, 36.0)
        action = DreamerActionCandidate(
            grind_delta_steps_from_current=5,
            next_dose_g=19.0,
            target_yield_g=44.0,
            confidence=0.7,
        )

        recipe = dreamer_action_to_recipe(action, current=current, bounds=SafetyBounds())

        self.assertEqual(recipe.relative_grind_steps_from_reference, 47)
        self.assertEqual(recipe.dose_g, 19.0)
        self.assertEqual(recipe.target_yield_g, 44.0)

    def test_action_rejects_moves_outside_safety_envelope(self) -> None:
        current = Recipe(42, 12.5, 18.0, 36.0)

        with self.assertRaisesRegex(ValueError, "grind delta"):
            dreamer_action_to_recipe(
                DreamerActionCandidate(
                    grind_delta_steps_from_current=6,
                    next_dose_g=18.0,
                    target_yield_g=36.0,
                    confidence=0.7,
                ),
                current=current,
                bounds=SafetyBounds(),
            )
        with self.assertRaisesRegex(ValueError, "yield delta"):
            dreamer_action_to_recipe(
                DreamerActionCandidate(
                    grind_delta_steps_from_current=0,
                    next_dose_g=18.0,
                    target_yield_g=45.0,
                    confidence=0.7,
                ),
                current=current,
                bounds=SafetyBounds(),
            )

    def test_action_payload_rejects_absolute_grinder_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            dreamer_action_from_payload(
                {
                    "format": "espresso_rl_dreamer_action_v1",
                    "schema_version": 1,
                    "grind_delta_steps_from_current": 1,
                    "next_dose_g": 18.0,
                    "target_yield_g": 36.0,
                    "confidence": 0.5,
                    "current_absolute_step": 42,
                }
            )

    def test_action_payload_requires_integer_relative_grind_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            dreamer_action_from_payload(
                {
                    "format": "espresso_rl_dreamer_action_v1",
                    "schema_version": 1,
                    "grind_delta_steps_from_current": 1.5,
                    "next_dose_g": 18.0,
                    "target_yield_g": 36.0,
                    "confidence": 0.5,
                }
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from espresso_rl.domain.prior_rules import MAX_PRIOR_RULES, PriorRule, parse_prior_rules


def valid_rule(index: int = 1) -> dict:
    return {
        "rule_id": f"rule_{index}",
        "name": f"Rule {index}",
        "metric": "taste_tag",
        "operator": "contains",
        "condition_value": "sour",
        "ratio_direction": "increase",
    }


class PriorRuleValidationTests(unittest.TestCase):
    def test_rule_is_plain_bounded_declarative_data(self) -> None:
        rule = PriorRule.from_dict(valid_rule())

        self.assertEqual(rule.condition_value, "sour")
        self.assertEqual(rule.ratio_direction.value, "increase")
        self.assertEqual(rule.to_dict()["source"], "user")

    def test_rule_rejects_unknown_or_executable_fields(self) -> None:
        payload = valid_rule()
        payload["expression"] = "__import__('os').system('bad')"

        with self.assertRaisesRegex(ValueError, "unknown fields"):
            PriorRule.from_dict(payload)

    def test_rule_rejects_invalid_operator_and_exact_adjustment(self) -> None:
        invalid_operator = valid_rule()
        invalid_operator["operator"] = "eval"
        with self.assertRaises(ValueError):
            PriorRule.from_dict(invalid_operator)

        exact_adjustment = valid_rule()
        exact_adjustment["target_ratio_delta"] = 0.2
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            PriorRule.from_dict(exact_adjustment)

        invalid_direction = valid_rule()
        invalid_direction["ratio_direction"] = "a_little_more"
        with self.assertRaises(ValueError):
            PriorRule.from_dict(invalid_direction)

    def test_rule_collection_is_bounded_and_ids_are_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "more than"):
            parse_prior_rules([valid_rule(index) for index in range(MAX_PRIOR_RULES + 1)])

        with self.assertRaisesRegex(ValueError, "unique"):
            parse_prior_rules([valid_rule(1), valid_rule(1)])


if __name__ == "__main__":
    unittest.main()

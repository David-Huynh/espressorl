from __future__ import annotations

import unittest

from espresso_rl.application.upload_payloads import (
    comparison_upload_payload,
    make_comparison_upload_item,
)
from espresso_rl.application.upload_validation import validate_upload_payload
from espresso_rl.domain.community import PairwiseShotComparison


def comparison(**overrides) -> PairwiseShotComparison:
    values = {
        "comparison_id": "comparison_1",
        "optimization_run_id": "run_1",
        "new_shot_id": "shot_new",
        "anchor_shot_id": "shot_anchor",
        "label": "new_better",
        "comparison_mode": "best_incumbent",
        "created_at": 100,
        "install_id": "install_1",
        "machine_id": "machine_1",
        "machine_adapter": "generic",
        "recommendation_id": "rec_1",
        "bean_context_id": "bean_1",
        "grinder_context_id": "grinder_1",
        "profile_id": "profile_1",
    }
    values.update(overrides)
    return PairwiseShotComparison(**values)


class CommunityComparisonContractTests(unittest.TestCase):
    def test_comparison_payload_is_optimizer_neutral_and_valid(self) -> None:
        payload = comparison_upload_payload(comparison(label="tie"))

        self.assertEqual(payload["event_type"], "comparison_record")
        self.assertEqual(payload["label"], "tie")
        self.assertNotIn("cpbo", payload["event_type"])
        self.assertTrue(validate_upload_payload(payload).ok)

    def test_upload_item_uses_comparison_identity_and_canonical_payload(self) -> None:
        item = make_comparison_upload_item(comparison(), now=101)

        self.assertEqual(item.local_record_type, "comparison")
        self.assertEqual(item.local_record_id, "comparison_1")
        self.assertTrue(item.upload_id.startswith("comparison_comparison_1_"))

    def test_comparison_rejects_same_physical_shot_on_both_sides(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct physical shots"):
            comparison(anchor_shot_id="shot_new")

    def test_comparison_rejects_numeric_or_unknown_outcomes(self) -> None:
        with self.assertRaisesRegex(ValueError, "label is invalid"):
            comparison(label="5")

        payload = comparison_upload_payload(comparison())
        payload["human_rating"] = 5
        validation = validate_upload_payload(payload)
        self.assertFalse(validation.ok)
        self.assertIn("unknown fields: human_rating", validation.errors)


if __name__ == "__main__":
    unittest.main()

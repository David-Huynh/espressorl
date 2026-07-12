from __future__ import annotations

import hashlib
import json
import unittest

from espresso_rl.adapters.postgres_repositories import PostgresCommunityWarehouse
from espresso_rl.application.offline_dataset_export import (
    OfflineDatasetExportService,
    manifest_json,
)
from espresso_rl.domain.offline_dataset import OfflinePreferenceExample


def shot_payload(shot_id: str, timestamp: int, **overrides) -> dict:
    payload = {
        "event_type": "shot_record",
        "schema_version": 1,
        "shot_id": shot_id,
        "timestamp": timestamp,
        "install_id": "install_1",
        "machine_id": "machine_1",
        "machine_adapter": "generic",
        "bean_context_id": "bean_1",
        "grinder_context_id": "grinder_1",
        "profile_id": "profile_1",
        "profile_label": "Profile One",
        "profile_type": "pro",
        "taste_goal": {"schema_version": 1, "mode": "balanced", "targets": {}},
        "action_observed": {"grind": True, "dose": True, "target_yield": True},
        "grinder_calibration_mode": "relative_calibrated",
        "grinder_adjustment_mode": "stepped",
        "microns_per_step": 10.0,
        "step_direction": "higher_is_finer",
        "reference_label": "initial",
        "relative_grind_steps_from_reference": 1.0,
        "relative_grind_um_from_reference": 10.0,
        "current_absolute_step": None,
        "absolute_reference_step": None,
        "dose_in_g": 18.0,
        "dose_target_g": 18.0,
        "target_yield_g": 36.0,
        "target_ratio": 2.0,
        "beverage_out_g": 35.8,
        "brew_ratio": 1.9889,
        "shot_time_s": 31.0,
        "shot_end_state": "finished",
        "profile_resampled": [[0.0, 1.0], [0.0, 1.0]],
        "fixed_cadence_sequence": None,
        "raw_profile_available": True,
        "raw_profile_hash": "a" * 64,
        "profile_flow_valid": True,
        "profile_flow_masked": False,
        "exclude_from_local_optimization": False,
    }
    payload.update(overrides)
    return payload


def comparison_payload(**overrides) -> dict:
    payload = {
        "event_type": "comparison_record",
        "schema_version": 1,
        "comparison_id": "comparison_1",
        "optimization_run_id": "run_1",
        "new_shot_id": "shot_new",
        "anchor_shot_id": "shot_anchor",
        "label": "new_better",
        "comparison_mode": "best_incumbent",
        "created_at": 30,
        "install_id": "install_1",
        "machine_id": "machine_1",
        "machine_adapter": "generic",
        "recommendation_id": "recommendation_1",
        "bean_context_id": "bean_1",
        "grinder_context_id": "grinder_1",
        "profile_id": "profile_1",
        "raw_profile_hash": None,
        "taste_goal": {"schema_version": 1, "mode": "balanced", "targets": {}},
    }
    payload.update(overrides)
    return payload


def example(**comparison_overrides) -> OfflinePreferenceExample:
    comparison = comparison_payload(**comparison_overrides)
    return OfflinePreferenceExample.from_joined_payloads(
        comparison_payload=comparison,
        new_shot_payload=shot_payload(comparison["new_shot_id"], 20),
        anchor_shot_payload=shot_payload(comparison["anchor_shot_id"], 10),
        comparison_trust_weight=0.2,
        new_shot_trust_weight=0.15,
        anchor_shot_trust_weight=0.1,
    )


class OfflineDatasetDomainTests(unittest.TestCase):
    def test_example_keeps_physical_data_and_oriented_preference_without_scalar_feedback(self) -> None:
        record = example(label="tie").to_dict()

        self.assertEqual(record["comparison"]["label"], "tie")
        self.assertEqual(record["new_shot"]["recipe"]["dose_in_g"], 18.0)
        self.assertEqual(record["new_shot"]["recipe"]["dose_target_g"], 18.0)
        self.assertIn("profile_resampled", record["new_shot"]["trajectory"])
        self.assertEqual(record["trust"]["example"], 0.1)
        encoded = json.dumps(record, sort_keys=True)
        for field_name in ("human_rating", "taste_tags", "reward", "reward_confidence"):
            self.assertNotIn(field_name, encoded)

    def test_mixed_context_and_reversed_join_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "bean_context_id"):
            OfflinePreferenceExample.from_joined_payloads(
                comparison_payload=comparison_payload(),
                new_shot_payload=shot_payload("shot_new", 20, bean_context_id="bean_other"),
                anchor_shot_payload=shot_payload("shot_anchor", 10),
                comparison_trust_weight=0.2,
                new_shot_trust_weight=0.2,
                anchor_shot_trust_weight=0.2,
            )

        with self.assertRaisesRegex(ValueError, "taste_goal"):
            OfflinePreferenceExample.from_joined_payloads(
                comparison_payload=comparison_payload(),
                new_shot_payload=shot_payload(
                    "shot_new",
                    20,
                    taste_goal={
                        "schema_version": 1,
                        "mode": "custom",
                        "targets": {"sweet": "high"},
                    },
                ),
                anchor_shot_payload=shot_payload("shot_anchor", 10),
                comparison_trust_weight=0.2,
                new_shot_trust_weight=0.2,
                anchor_shot_trust_weight=0.2,
            )

        with self.assertRaisesRegex(ValueError, "new_shot_id"):
            OfflinePreferenceExample.from_joined_payloads(
                comparison_payload=comparison_payload(),
                new_shot_payload=shot_payload("shot_anchor", 20),
                anchor_shot_payload=shot_payload("shot_new", 10),
                comparison_trust_weight=0.2,
                new_shot_trust_weight=0.2,
                anchor_shot_trust_weight=0.2,
            )

    def test_future_and_scalar_feedback_shots_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            OfflinePreferenceExample.from_joined_payloads(
                comparison_payload=comparison_payload(created_at=15),
                new_shot_payload=shot_payload("shot_new", 20),
                anchor_shot_payload=shot_payload("shot_anchor", 10),
                comparison_trust_weight=0.2,
                new_shot_trust_weight=0.2,
                anchor_shot_trust_weight=0.2,
            )

        with self.assertRaisesRegex(ValueError, "scalar feedback"):
            OfflinePreferenceExample.from_joined_payloads(
                comparison_payload=comparison_payload(),
                new_shot_payload=shot_payload("shot_new", 20, reward=0.9),
                anchor_shot_payload=shot_payload("shot_anchor", 10),
                comparison_trust_weight=0.2,
                new_shot_trust_weight=0.2,
                anchor_shot_trust_weight=0.2,
            )


class OfflineDatasetExportTests(unittest.TestCase):
    def test_export_is_deterministic_plain_jsonl_with_verifiable_manifest(self) -> None:
        late = example(comparison_id="comparison_b", created_at=40)
        early = example(comparison_id="comparison_a", created_at=30)
        service = OfflineDatasetExportService(
            FakeSource([late, early]),
            clock=lambda: 1234,
            exporter_version="test",
        )

        first = service.build()
        second = service.build()

        self.assertEqual(first.records_jsonl, second.records_jsonl)
        rows = [json.loads(line) for line in first.records_jsonl.splitlines()]
        self.assertEqual(
            [row["comparison"]["comparison_id"] for row in rows],
            ["comparison_a", "comparison_b"],
        )
        self.assertEqual(
            first.manifest["records_sha256"],
            hashlib.sha256(first.records_jsonl).hexdigest(),
        )
        self.assertFalse(first.manifest["contains_scalar_ratings"])
        self.assertFalse(first.manifest["contains_executable_artifacts"])
        self.assertTrue(manifest_json(first).endswith(b"\n"))

    def test_duplicate_comparison_identity_is_rejected(self) -> None:
        service = OfflineDatasetExportService(FakeSource([example(), example()]), clock=lambda: 1)
        with self.assertRaisesRegex(ValueError, "duplicate comparison"):
            service.build()


class PostgresOfflineDatasetSourceTests(unittest.TestCase):
    def test_source_uses_inner_joins_and_positive_trust_filters(self) -> None:
        row = {
            "comparison_payload": comparison_payload(),
            "comparison_trust_weight": 0.2,
            "new_shot_payload": shot_payload("shot_new", 20),
            "new_shot_trust_weight": 0.2,
            "anchor_shot_payload": shot_payload("shot_anchor", 10),
            "anchor_shot_trust_weight": 0.2,
        }
        connection = FakeConnection([row])
        warehouse = PostgresCommunityWarehouse(FakeStore(connection))

        records = warehouse.list_offline_preference_examples(limit=5)

        self.assertEqual(len(records), 1)
        normalized_query = " ".join(connection.query.split())
        self.assertIn("INNER JOIN community_validated_shots AS new_shot", normalized_query)
        self.assertIn("INNER JOIN community_validated_shots AS anchor_shot", normalized_query)
        self.assertIn("comparison.trust_weight > 0.0", normalized_query)
        self.assertEqual(connection.parameters, (5,))


class FakeSource:
    def __init__(self, examples: list[OfflinePreferenceExample]) -> None:
        self.examples = examples

    def list_offline_preference_examples(self, *, limit: int | None = None):
        return list(self.examples if limit is None else self.examples[:limit])


class FakeStore:
    def __init__(self, connection) -> None:
        self.conn = connection


class FakeConnection:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.query = ""
        self.parameters = ()

    def execute(self, query: str, parameters=()):
        self.query = query
        self.parameters = parameters
        return self

    def fetchall(self):
        return self.rows


if __name__ == "__main__":
    unittest.main()

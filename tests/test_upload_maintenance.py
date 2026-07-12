from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from espresso_rl.adapters.sqlite_repositories import SQLiteShotRepository, SQLiteStore, SQLiteUploadQueueRepository
from espresso_rl.application.upload_payloads import payload_hash as hash_payload_json
from espresso_rl.application.upload_maintenance import UploadQueueMaintenanceService
from espresso_rl.application.upload_validation import (
    mask_untrusted_profile_channels,
    validate_upload_payload_json,
)
from espresso_rl.domain.models import ShotRecord, ShotType, UploadQueueItem, UploadQueueStatus


def payload(**overrides) -> str:
    data = {
        "event_type": "shot_record",
        "schema_version": 1,
        "shot_id": "shot_1",
        "install_id": "install_1",
        "machine_id": "machine_1",
        "taste_goal": {"schema_version": 1, "mode": "balanced", "targets": {}},
        "timestamp": 1,
        "dose_in_g": 18.0,
        "dose_target_g": 18.0,
        "target_yield_g": 36.0,
        "target_ratio": 2.0,
        "beverage_out_g": 36.0,
        "shot_time_s": 30.0,
        "profile_temperature_c": 93.0,
        "final_phase_temperature_c": 92.5,
    }
    data.update(overrides)
    return json.dumps(data, sort_keys=True)


def payload_dict(**overrides) -> dict:
    data = json.loads(payload())
    data.update(overrides)
    return data


def valid_profile() -> list[list[float]]:
    profile = [[0.0 for _ in range(100)] for _ in range(5)]
    profile[0] = [9.0 for _ in range(100)]
    profile[1] = [9.0 for _ in range(100)]
    profile[2] = [2.0 for _ in range(100)]
    profile[3] = [0.0 for _ in range(100)]
    profile[4] = [i * 0.36 for i in range(100)]
    profile[4][-1] = 36.0
    return profile


def valid_fixed_cadence_sequence() -> dict:
    return {
        "sample_interval_ms": 250,
        "pressure_bar": [0.0, 2.0, 5.0, 8.0],
        "pressure_target_bar": [2.0, 4.0, 8.0, 9.0],
        "pump_flow_ml_s": [0.0, 1.0, 2.0, 2.2],
        "pump_flow_target_ml_s": [0.0, 0.0, 0.0, 0.0],
        "beverage_flow_g_s": [0.0, 0.5, 1.5, 2.0],
        "weight_g": [0.0, 0.1, 0.5, 1.0],
        "temperature_c": [92.0, 92.1, 92.2, 92.3],
        "temperature_target_c": [93.0, 93.0, 93.0, 93.0],
        "pump_target_mode": [1, 1, 1, 1],
        "valve_open": [True, True, True, True],
    }


def queue_item(
    upload_id: str,
    payload_json: str,
    *,
    local_record_id: str = "shot_1",
    status: UploadQueueStatus = UploadQueueStatus.REJECTED,
    payload_hash_override: str | None = None,
) -> UploadQueueItem:
    return UploadQueueItem(
        upload_id=upload_id,
        local_record_type="shot",
        local_record_id=local_record_id,
        payload_hash=payload_hash_override or hash_payload_json(payload_json),
        payload_json=payload_json,
        status=status,
        attempt_count=3,
        error_message="HTTP 400: old validation error",
        created_at=1,
        updated_at=2,
    )


class UploadMaintenanceTests(unittest.TestCase):
    def test_preflight_accepts_valid_shot_payload(self) -> None:
        result = validate_upload_payload_json(payload())

        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])

    def test_preflight_requires_commanded_dose_separately_from_measured_dose(self) -> None:
        result = validate_upload_payload_json(payload(dose_target_g=None))

        self.assertFalse(result.ok)
        self.assertIn("dose_target_g out of range", result.errors)

    def test_preflight_accepts_safe_execution_metadata(self) -> None:
        result = validate_upload_payload_json(
            payload(
                profile_id="profile_1",
                profile_label="Cremina lever machine",
                profile_type="pro",
                profile_phase_count=5,
                final_phase_index=3,
                final_phase_name="ramp",
                final_phase_type="brew",
                final_phase_elapsed_s=8.5,
                final_pump_target="pressure",
                final_target_pressure=9.0,
                final_target_flow=0.0,
                final_valve_open=True,
                profile_temperature_c=86.5,
                final_phase_temperature_c=86.5,
                shot_end_state="manual_or_interrupted",
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])

    def test_preflight_accepts_small_negative_tare_noise_and_close_final_weight(self) -> None:
        profile = valid_profile()
        profile[4][0] = -0.1
        profile[4][-1] = 37.5

        result = validate_upload_payload_json(
            payload(
                target_yield_g=38.0,
                target_ratio=38.0 / 18.0,
                beverage_out_g=37.5,
                profile_resampled=profile,
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])

    def test_preflight_requires_temperature_metadata(self) -> None:
        result = validate_upload_payload_json(payload(profile_temperature_c=None))

        self.assertFalse(result.ok)
        self.assertIn("profile_temperature_c out of range", result.errors)

    def test_preflight_accepts_resampled_temperature_and_pump_mode_profiles(self) -> None:
        result = validate_upload_payload_json(
            payload(
                beverage_flow_profile=[1.5 for _ in range(100)],
                temperature_profile=[93.0 for _ in range(100)],
                target_temperature_profile=[92.5 for _ in range(100)],
                pump_target_mode_profile=[2 for _ in range(100)],
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])

    def test_preflight_rejects_invalid_pump_mode_profile(self) -> None:
        result = validate_upload_payload_json(payload(pump_target_mode_profile=[3 for _ in range(100)]))

        self.assertFalse(result.ok)
        self.assertIn("pump_target_mode_profile contains invalid pump target mode values", result.errors)

    def test_preflight_rejects_invalid_beverage_flow_profile(self) -> None:
        result = validate_upload_payload_json(payload(beverage_flow_profile=[21.0 for _ in range(100)]))

        self.assertFalse(result.ok)
        self.assertIn("beverage_flow_profile out of range", result.errors)

    def test_preflight_accepts_fixed_cadence_sequence(self) -> None:
        result = validate_upload_payload_json(
            payload(fixed_cadence_sequence=valid_fixed_cadence_sequence())
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])

    def test_preflight_rejects_nonfixed_or_misaligned_sequence(self) -> None:
        sequence = valid_fixed_cadence_sequence()
        sequence["sample_interval_ms"] = 200
        sequence["temperature_c"].pop()

        result = validate_upload_payload_json(payload(fixed_cadence_sequence=sequence))

        self.assertFalse(result.ok)
        self.assertIn("fixed_cadence_sequence.sample_interval_ms must be 250", result.errors)
        self.assertIn("fixed_cadence_sequence channels must have matching lengths", result.errors)

    def test_preflight_rejects_weight_below_tare_noise_floor(self) -> None:
        profile = valid_profile()
        profile[4][0] = -1.01

        result = validate_upload_payload_json(payload(profile_resampled=profile))

        self.assertFalse(result.ok)
        self.assertIn("profile_resampled weight out of range", result.errors)

    def test_preflight_allows_profile_weight_to_differ_from_beverage_output(self) -> None:
        profile = valid_profile()
        profile[4][-1] = 95.0

        result = validate_upload_payload_json(payload(profile_resampled=profile, beverage_out_g=36.0))

        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])

    def test_preflight_accepts_hardware_scale_cutoff_and_predicted_final_weight(self) -> None:
        profile = valid_profile()
        profile[4][-1] = 92.0

        result = validate_upload_payload_json(
            payload(
                weight_source="hardware_scale",
                beverage_out_g=33.2,
                beverage_out_observation="control_cutoff",
                predicted_final_beverage_out_g=36.0,
                predictive_stop_applied=True,
                predictive_stop_delay_ms=800.0,
                predictive_stop_rate_g_per_s=3.5,
                predictive_stop_lead_g=2.8,
                target_yield_g=36.0,
                profile_resampled=profile,
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])

    def test_preflight_keeps_unknown_manual_dose_as_masked_partial_data(self) -> None:
        result = validate_upload_payload_json(
            payload(
                dose_in_g=None,
                dose_observed=False,
                dose_target_confirmed=False,
                action_observed={"grind": False, "dose": False, "target_yield": True},
            )
        )

        self.assertTrue(result.ok)

    def test_preflight_rejects_observed_dose_without_measurement_or_confirmation(self) -> None:
        result = validate_upload_payload_json(
            payload(
                dose_in_g=None,
                dose_observed=False,
                dose_target_confirmed=False,
                action_observed={"grind": False, "dose": True, "target_yield": True},
            )
        )

        self.assertFalse(result.ok)
        self.assertIn("action_observed.dose cannot be true without a measured or confirmed dose", result.errors)

    def test_preflight_accepts_observed_pressure_overshoot_without_relaxing_target_pressure(self) -> None:
        profile = valid_profile()
        profile[0] = [15.4 for _ in range(100)]
        profile[1] = [10.0 for _ in range(100)]

        result = validate_upload_payload_json(payload(profile_resampled=profile))

        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])

        profile[1] = [15.1 for _ in range(100)]
        result = validate_upload_payload_json(payload(profile_resampled=profile))

        self.assertFalse(result.ok)
        self.assertIn("profile_resampled target_pressure out of range", result.errors)

    def test_preflight_rejects_impossible_observed_pressure(self) -> None:
        profile = valid_profile()
        profile[0] = [20.1 for _ in range(100)]

        result = validate_upload_payload_json(payload(profile_resampled=profile))

        self.assertFalse(result.ok)
        self.assertIn("profile_resampled pressure out of range", result.errors)

    def test_preflight_rejects_invalid_execution_metadata(self) -> None:
        result = validate_upload_payload_json(
            payload(
                profile_phase_count=1.5,
                final_phase_type="steam",
                final_target_pressure=99.0,
                final_valve_open="true",
            )
        )

        self.assertFalse(result.ok)
        self.assertIn("profile_phase_count out of range", result.errors)
        self.assertIn("final_phase_type is invalid", result.errors)
        self.assertIn("final_target_pressure out of range", result.errors)
        self.assertIn("final_valve_open must be boolean", result.errors)

    def test_preflight_rejects_utility_flush_payload(self) -> None:
        result = validate_upload_payload_json(
            payload(
                shot_id="flush_1",
                shot_type="utility_flush",
                beverage_out_g=1.0,
                shot_time_s=3.0,
            )
        )

        self.assertFalse(result.ok)
        self.assertIn("shot_type must be espresso", result.errors)

    def test_preflight_accepts_bad_espresso_as_useful_negative_signal(self) -> None:
        result = validate_upload_payload_json(
            payload(
                shot_type="espresso",
                beverage_out_g=1.0,
                shot_time_s=3.0,
                target_yield_g=38.0,
                target_ratio=38.0 / 18.0,
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])

    def test_preflight_allows_invalid_flow_when_flow_target_is_inactive(self) -> None:
        profile = valid_profile()
        profile[2] = [100_000.0 for _ in range(100)]

        result = validate_upload_payload_json(payload(profile_resampled=profile))

        self.assertTrue(result.ok)

    def test_preflight_allows_invalid_flow_when_flow_target_is_active(self) -> None:
        profile = valid_profile()
        profile[2] = [100_000.0 for _ in range(100)]
        profile[3] = [2.0 for _ in range(100)]

        result = validate_upload_payload_json(payload(profile_resampled=profile))

        self.assertTrue(result.ok)

    def test_preflight_rejects_nonfinite_flow_even_when_maskable(self) -> None:
        profile = valid_profile()
        profile[2] = [float("inf") for _ in range(100)]

        result = validate_upload_payload_json(payload(profile_resampled=profile))

        self.assertFalse(result.ok)
        self.assertIn("profile_resampled pump_flow contains non-finite or nonnumeric values", result.errors)

    def test_trusted_payload_copy_masks_invalid_inactive_flow(self) -> None:
        profile = valid_profile()
        profile[2] = [100_000.0 for _ in range(100)]
        raw = payload_dict(profile_resampled=profile)

        trusted = mask_untrusted_profile_channels(raw)

        self.assertEqual(trusted["profile_resampled"][2], [0.0 for _ in range(100)])
        self.assertEqual(trusted["profile_resampled"][3], [0.0 for _ in range(100)])
        self.assertFalse(trusted["profile_flow_valid"])
        self.assertTrue(trusted["profile_flow_masked"])

    def test_trusted_payload_copy_masks_invalid_active_flow_pair(self) -> None:
        profile = valid_profile()
        profile[2] = [100_000.0 for _ in range(100)]
        profile[3] = [2.0 for _ in range(100)]
        raw = payload_dict(profile_resampled=profile)

        trusted = mask_untrusted_profile_channels(raw)

        self.assertEqual(trusted["profile_resampled"][2], [0.0 for _ in range(100)])
        self.assertEqual(trusted["profile_resampled"][3], [0.0 for _ in range(100)])
        self.assertFalse(trusted["profile_flow_valid"])
        self.assertTrue(trusted["profile_flow_masked"])

    def test_trusted_payload_copy_masks_uncalibrated_fixed_cadence_pump_flow(self) -> None:
        sequence = valid_fixed_cadence_sequence()
        raw = payload_dict(
            fixed_cadence_sequence=sequence,
            pump_flow_calibration_required=True,
        )

        trusted = mask_untrusted_profile_channels(raw)

        self.assertEqual(trusted["fixed_cadence_sequence"]["pump_flow_ml_s"], [0.0] * 4)
        self.assertEqual(trusted["fixed_cadence_sequence"]["pump_flow_target_ml_s"], [0.0] * 4)
        self.assertTrue(trusted["profile_flow_masked"])

    def test_requeue_valid_rejected_uploads_leaves_invalid_rows_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                queue.enqueue(queue_item("valid", payload(), local_record_id="shot_valid"))
                queue.enqueue(
                    queue_item(
                        "invalid",
                        payload(
                            shot_id="flush_1",
                            shot_type="utility_flush",
                            beverage_out_g=1.0,
                            shot_time_s=3.0,
                        ),
                        local_record_id="shot_invalid",
                    )
                )
                service = UploadQueueMaintenanceService(queue, clock=lambda: 10)

                result = service.requeue_valid_rejected(limit=10)

                self.assertEqual(result.inspected, 2)
                self.assertEqual(result.requeued, 1)
                self.assertEqual(result.skipped, 1)
                statuses = {
                    row["upload_id"]: row["status"]
                    for row in store.conn.execute("SELECT upload_id, status FROM upload_queue").fetchall()
                }
                invalid = store.conn.execute(
                    "SELECT attempt_count, error_message, updated_at FROM upload_queue WHERE upload_id=?",
                    ("invalid",),
                ).fetchone()
                self.assertEqual(statuses["valid"], "pending")
                self.assertEqual(statuses["invalid"], "rejected")
                self.assertEqual(invalid["attempt_count"], 3)
                self.assertEqual(invalid["updated_at"], 10)
                self.assertIn("preflight failed", invalid["error_message"])
                self.assertIn("shot_type must be espresso", invalid["error_message"])

    def test_requeue_rejects_payload_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                queue.enqueue(
                    queue_item(
                        "tampered",
                        payload(),
                        payload_hash_override="0" * 64,
                    )
                )
                service = UploadQueueMaintenanceService(queue, clock=lambda: 10)

                result = service.requeue_valid_rejected(limit=10)

                self.assertEqual(result.requeued, 0)
                self.assertEqual(result.skipped, 1)
                row = store.conn.execute(
                    "SELECT status, error_message FROM upload_queue WHERE upload_id='tampered'"
                ).fetchone()
                self.assertEqual(row["status"], "rejected")
                self.assertIn("payload_hash does not match payload_json", row["error_message"])

    def test_latest_rejected_summary_does_not_expose_payload_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                queue.enqueue(queue_item("rejected", payload(), local_record_id="shot_1"))
                service = UploadQueueMaintenanceService(queue, clock=lambda: 10)

                summary = service.latest_rejected()

                self.assertEqual(summary.upload_id, "rejected")  # type: ignore[union-attr]
                self.assertEqual(summary.local_record_id, "shot_1")  # type: ignore[union-attr]
                self.assertFalse(hasattr(summary, "payload_json"))

    def test_purge_rejected_deletes_linked_local_shots_after_permanent_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                queue = SQLiteUploadQueueRepository(store)
                profile = np.zeros((5, 100), dtype=np.float32)
                shots.upsert(
                    ShotRecord(
                        shot_id="flush_1",
                        timestamp=1,
                        install_id="install_1",
                        machine_id="machine_1",
                        machine_adapter="gaggimate",
                        profile=profile,
                        microns_per_step=12.5,
                        dose_in_g=18.0,
                        target_yield_g=36.0,
                        shot_type=ShotType.UTILITY_FLUSH,
                        exclude_from_local_optimization=True,
                        created_at=1,
                        updated_at=1,
                    )
                )
                shots.upsert(
                    ShotRecord(
                        shot_id="espresso_1",
                        timestamp=2,
                        install_id="install_1",
                        machine_id="machine_1",
                        machine_adapter="gaggimate",
                        profile=profile,
                        microns_per_step=12.5,
                        dose_in_g=18.0,
                        target_yield_g=36.0,
                        shot_type=ShotType.ESPRESSO,
                        exclude_from_local_optimization=False,
                        optimization_weight=1.0,
                        created_at=2,
                        updated_at=2,
                    )
                )
                queue.enqueue(queue_item("flush_upload", payload(), local_record_id="flush_1"))
                queue.enqueue(queue_item("espresso_upload", payload(), local_record_id="espresso_1"))
                service = UploadQueueMaintenanceService(queue, clock=lambda: 10)

                result = service.purge_rejected(limit=10)

                self.assertEqual(result.inspected, 2)
                self.assertEqual(result.purged_uploads, 2)
                self.assertEqual(result.purged_shots, 2)
                self.assertEqual(result.kept_linked_records, 0)
                self.assertIsNone(shots.get("flush_1"))
                self.assertIsNone(shots.get("espresso_1"))
                self.assertEqual(store.conn.execute("SELECT COUNT(*) AS count FROM upload_queue").fetchone()["count"], 0)

    def test_purge_rejected_can_target_one_selected_local_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                queue = SQLiteUploadQueueRepository(store)
                profile = np.zeros((5, 100), dtype=np.float32)
                for shot_id in ("flush_1", "flush_2"):
                    shots.upsert(
                        ShotRecord(
                            shot_id=shot_id,
                            timestamp=1,
                            install_id="install_1",
                            machine_id="machine_1",
                            machine_adapter="gaggimate",
                            profile=profile,
                            microns_per_step=12.5,
                            dose_in_g=18.0,
                            target_yield_g=36.0,
                            shot_type=ShotType.UTILITY_FLUSH,
                            exclude_from_local_optimization=True,
                            created_at=1,
                            updated_at=1,
                        )
                    )
                queue.enqueue(queue_item("flush_1_upload", payload(), local_record_id="flush_1"))
                queue.enqueue(queue_item("flush_2_upload", payload(), local_record_id="flush_2"))
                service = UploadQueueMaintenanceService(queue, clock=lambda: 10)

                result = service.purge_rejected(limit=10, local_record_id="flush_1")

                self.assertEqual(result.inspected, 1)
                self.assertEqual(result.purged_uploads, 1)
                self.assertEqual(result.purged_shots, 1)
                self.assertIsNone(shots.get("flush_1"))
                self.assertIsNotNone(shots.get("flush_2"))
                self.assertEqual(
                    store.conn.execute("SELECT local_record_id FROM upload_queue").fetchone()["local_record_id"],
                    "flush_2",
                )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import ast
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from urllib import error

from espresso_rl.config import Config
from espresso_rl.adapters.sqlite_repositories import (
    SQLiteRecommendationRepository,
    SQLiteShotRepository,
    SQLiteStore,
    SQLiteUploadQueueRepository,
    _shot_to_row,
)
from espresso_rl.adapters.postgres_repositories import PostgresStore, PostgresUploadQueueRepository, _upsert
from espresso_rl.application.upload_payloads import payload_hash as hash_payload_json
from espresso_rl.application.services import EspressoRLService
from espresso_rl.domain.events import RecommendationApplyEvent, ShotFeedbackEvent, ShotProfileEvent
from espresso_rl.domain.models import RecommendationApplyStatus, UploadQueueItem, UploadQueueStatus
from espresso_rl.optimizers.conservative_bo import ConservativeBOOptimizer
from espresso_rl.adapters.supabase_upload import (
    SignedSupabaseUploadClient,
    SignedUploadConfig,
    UploadCredentialRejected,
    UploadRejected,
    UploadQueueWorker,
    UploadRateLimited,
)
from espresso_rl.main import build_status_payload, maybe_start_upload_worker, upload_queue_for_service


def shot_event(**overrides) -> ShotProfileEvent:
    base = {
        "shot_id": "shot_1",
        "install_id": "install_1",
        "machine_id": "machine_1",
        "machine_adapter": "gaggimate",
        "timestamp": 1,
        "time_ms": [0, 500, 1000],
        "pressure": [0.0, 8.0, 9.0],
        "target_pressure": [0.0, 8.0, 9.0],
        "pump_flow": [0.0, 2.0, 2.0],
        "target_flow": [0.0, 2.0, 2.0],
        "beverage_flow": [0.0, 1.8, 2.0],
        "weight": [0.0, 10.0, 36.0],
        "microns_per_step": 12.5,
        "relative_grind_steps_from_reference": 42,
        "dose_in_g": 18.0,
        "target_yield_g": 36.0,
        "beverage_out_g": 36.0,
        "shot_time_s": 30.0,
        "weight_source": "hardware_scale",
        "flow_source": "beverage_weight_derivative",
        "flow_units": "g_per_s",
        "pump_flow_source": "gaggimate_pump_model",
        "pump_flow_units": "ml_per_s",
        "pump_flow_calibration_required": False,
        "profile_id": "profile_1",
        "profile_label": "Cremina lever machine",
        "profile_type": "pro",
        "profile_phase_count": 5,
        "final_phase_index": 3,
        "final_phase_name": "ramp",
        "final_phase_type": "brew",
        "final_phase_elapsed_s": 8.5,
        "final_pump_target": "pressure",
        "final_target_pressure": 9.0,
        "final_target_flow": 0.0,
        "final_valve_open": True,
        "profile_temperature_c": 86.5,
        "final_phase_temperature_c": 86.5,
        "shot_end_state": "manual_or_interrupted",
    }
    base.update(overrides)
    return ShotProfileEvent(**base)


def queue_item(
    upload_id: str,
    payload_hash: str,
    *,
    status: UploadQueueStatus = UploadQueueStatus.PENDING,
    local_record_type: str = "shot",
    local_record_id: str = "shot_1",
    attempt_count: int = 0,
) -> UploadQueueItem:
    return UploadQueueItem(
        upload_id=upload_id,
        local_record_type=local_record_type,
        local_record_id=local_record_id,
        payload_hash=payload_hash,
        payload_json=valid_upload_payload(local_record_id),
        status=status,
        attempt_count=attempt_count,
        created_at=1,
        updated_at=1,
    )


def uploadable_queue_item(
    upload_id: str,
    *,
    status: UploadQueueStatus = UploadQueueStatus.PENDING,
    local_record_type: str = "shot",
    local_record_id: str = "shot_1",
    attempt_count: int = 0,
) -> UploadQueueItem:
    payload_json = valid_upload_payload(local_record_id)
    return UploadQueueItem(
        upload_id=upload_id,
        local_record_type=local_record_type,
        local_record_id=local_record_id,
        payload_hash=hash_payload_json(payload_json),
        payload_json=payload_json,
        status=status,
        attempt_count=attempt_count,
        created_at=1,
        updated_at=1,
    )


def valid_upload_payload(shot_id: str = "shot_1") -> str:
    return json.dumps(
        {
            "event_type": "shot_record",
            "schema_version": 1,
            "shot_id": shot_id,
            "install_id": "install_1",
            "machine_id": "machine_1",
            "timestamp": 1,
            "dose_in_g": 18.0,
            "target_yield_g": 36.0,
            "target_ratio": 2.0,
            "beverage_out_g": 36.0,
            "shot_time_s": 30.0,
            "profile_temperature_c": 93.0,
            "final_phase_temperature_c": 92.5,
        },
        sort_keys=True,
    )


class SQLiteAndBoundaryTests(unittest.TestCase):
    def test_sqlite_repositories_round_trip_core_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                recs = SQLiteRecommendationRepository(store)
                service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

                result = service.ingest_shot_profile(
                    shot_event(
                        bean_context_id="bean_lavazza_100_001",
                        bean_context_name="Lavazza",
                        temperature=[86.0, 86.5, 87.0],
                        target_temperature=[86.5, 86.5, 87.0],
                        pump_target_mode=[1, 1, 2],
                        valve_open=[False, True, True],
                        grind_observed=False,
                        dose_observed=False,
                        target_yield_observed=False,
                    )
                )
                self.assertIsNotNone(result.shot.temperature_profile)
                self.assertIsNotNone(result.shot.target_temperature_profile)
                self.assertIsNotNone(result.shot.pump_target_mode_profile)
                self.assertIsNotNone(result.shot.beverage_flow_profile)
                self.assertIsNotNone(result.shot.fixed_cadence_sequence)
                feedback = service.record_feedback(
                    ShotFeedbackEvent(
                        shot_id="shot_1",
                        install_id="install_1",
                        machine_id="machine_1",
                        timestamp=2,
                        rating=4,
                    )
                )
                service.record_recommendation_apply(
                    RecommendationApplyEvent(
                        recommendation_id=feedback.recommendation.recommendation_id,
                        status=RecommendationApplyStatus.MANUAL_REQUIRED,
                        timestamp=2,
                        manual_fields=["projected_relative_step_from_reference", "next_dose_g"],
                    )
                )
                stored_shot = shots.get("shot_1")
                stored_rec = recs.get(feedback.recommendation.recommendation_id)

                self.assertIsNotNone(stored_shot)
                self.assertIsNotNone(stored_rec)
                self.assertEqual(stored_shot.profile.shape, (5, 100))  # type: ignore[union-attr]
                self.assertEqual(stored_shot.bean_context_name, "Lavazza")  # type: ignore[union-attr]
                self.assertEqual(stored_shot.weight_source, "hardware_scale")  # type: ignore[union-attr]
                self.assertEqual(stored_shot.profile_label, "Cremina lever machine")  # type: ignore[union-attr]
                self.assertEqual(stored_shot.final_phase_name, "ramp")  # type: ignore[union-attr]
                self.assertTrue(stored_shot.final_valve_open)  # type: ignore[union-attr]
                self.assertEqual(stored_shot.shot_end_state, "manual_or_interrupted")  # type: ignore[union-attr]
                self.assertIsNotNone(stored_shot.temperature_profile)  # type: ignore[union-attr]
                self.assertIsNotNone(stored_shot.target_temperature_profile)  # type: ignore[union-attr]
                self.assertIsNotNone(stored_shot.pump_target_mode_profile)  # type: ignore[union-attr]
                self.assertIsNotNone(stored_shot.beverage_flow_profile)  # type: ignore[union-attr]
                self.assertEqual(stored_shot.pump_target_mode_profile[-1].item(), 2)  # type: ignore[union-attr]
                self.assertEqual(stored_shot.fixed_cadence_sequence.step_count, 5)  # type: ignore[union-attr]
                self.assertEqual(stored_shot.fixed_cadence_sequence.valve_open.tolist(), [0, 0, 1, 1, 1])  # type: ignore[union-attr]
                self.assertTrue(stored_shot.feedback_recorded)  # type: ignore[union-attr]
                self.assertFalse(stored_shot.grind_observed)  # type: ignore[union-attr]
                self.assertFalse(stored_shot.dose_observed)  # type: ignore[union-attr]
                self.assertFalse(stored_shot.target_yield_observed)  # type: ignore[union-attr]
                self.assertEqual(stored_rec.reason, feedback.recommendation.reason)  # type: ignore[union-attr]
                self.assertEqual(stored_rec.apply_status, RecommendationApplyStatus.MANUAL_REQUIRED)  # type: ignore[union-attr]
                self.assertEqual(stored_rec.manual_fields, ["projected_relative_step_from_reference", "next_dose_g"])  # type: ignore[union-attr]

    def test_sqlite_scopes_recent_shots_and_current_recommendations_by_grinder_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                recs = SQLiteRecommendationRepository(store)
                service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

                service.ingest_shot_profile(
                    shot_event(shot_id="shot_a", bean_context_id="bean_1", grinder_context_id="grinder_a")
                )
                rec_a = service.record_feedback(
                    ShotFeedbackEvent(
                        shot_id="shot_a",
                        install_id="install_1",
                        machine_id="machine_1",
                        timestamp=2,
                        rating=4,
                    )
                ).recommendation
                service.ingest_shot_profile(
                    shot_event(
                        shot_id="shot_b",
                        timestamp=3,
                        bean_context_id="bean_1",
                        grinder_context_id="grinder_b",
                        relative_grind_steps_from_reference=52,
                    )
                )
                rec_b = service.record_feedback(
                    ShotFeedbackEvent(
                        shot_id="shot_b",
                        install_id="install_1",
                        machine_id="machine_1",
                        timestamp=4,
                        rating=2,
                    )
                ).recommendation

                self.assertEqual(
                    [shot.shot_id for shot in shots.list_recent("install_1", "machine_1", "bean_1", grinder_context_id="grinder_a")],
                    ["shot_a"],
                )
                self.assertEqual(
                    [shot.shot_id for shot in shots.list_recent("install_1", "machine_1", "bean_1", grinder_context_id="grinder_b")],
                    ["shot_b"],
                )
                self.assertEqual(
                    recs.get_current("install_1", "machine_1", "bean_1", 20, grinder_context_id="grinder_a").recommendation_id,
                    rec_a.recommendation_id,
                )
                self.assertEqual(
                    recs.get_current("install_1", "machine_1", "bean_1", 20, grinder_context_id="grinder_b").recommendation_id,
                    rec_b.recommendation_id,
                )

    def test_shared_shot_row_uses_boolean_for_postgres_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                service = EspressoRLService(
                    SQLiteShotRepository(store),
                    SQLiteRecommendationRepository(store),
                    ConservativeBOOptimizer(),
                    clock=lambda: 10,
                )

                result = service.ingest_shot_profile(shot_event())
                row = _shot_to_row(result.shot)

                self.assertIs(row["raw_profile_available"], True)

    def test_status_payload_includes_sanitized_recent_shot_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(mqtt_host="localhost", data_dir=Path(tmp), install_id="install_1")
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                recs = SQLiteRecommendationRepository(store)
                service = EspressoRLService(shots, recs, ConservativeBOOptimizer(), clock=lambda: 10)

                service.ingest_shot_profile(shot_event())
                status = build_status_payload(
                    config=config,
                    service=service,
                    shot_repo=shots,
                    upload_maintenance=None,
                    upload_queue_repo=None,
                    machine_id="machine_1",
                    bean_context_id=None,
                )

                recent = status["recent_shots"]
                self.assertEqual(len(recent), 1)
                self.assertEqual(recent[0]["shot_id"], "shot_1")
                self.assertEqual(status["grinder_catalog_search_url"], "")
                self.assertEqual(status["prior_rule_catalog_search_url"], "")
                self.assertEqual(recent[0]["profile_label"], "Cremina lever machine")
                self.assertEqual(recent[0]["final_phase_name"], "ramp")
                self.assertEqual(recent[0]["shot_end_state"], "manual_or_interrupted")
                self.assertNotIn("profile_resampled", recent[0])

    def test_status_payload_scopes_progress_and_best_recipe_to_active_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(mqtt_host="localhost", data_dir=Path(tmp), install_id="install_1")
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                recommendations = SQLiteRecommendationRepository(store)
                service = EspressoRLService(
                    shots,
                    recommendations,
                    ConservativeBOOptimizer(),
                    clock=lambda: 10,
                )
                service.ingest_shot_profile(shot_event())
                profile_1 = service.record_feedback(
                    ShotFeedbackEvent(
                        shot_id="shot_1",
                        install_id="install_1",
                        machine_id="machine_1",
                        timestamp=2,
                        rating=5,
                    )
                ).recommendation
                service.ingest_shot_profile(
                    shot_event(
                        shot_id="shot_2",
                        timestamp=3,
                        profile_id="profile_2",
                        profile_label="Turbo Profile",
                        target_yield_g=42.0,
                        beverage_out_g=42.0,
                    )
                )
                profile_2 = service.record_feedback(
                    ShotFeedbackEvent(
                        shot_id="shot_2",
                        install_id="install_1",
                        machine_id="machine_1",
                        timestamp=4,
                        rating=2,
                    )
                ).recommendation

                self.assertEqual(
                    recommendations.get_current(
                        "install_1",
                        "machine_1",
                        None,
                        now=10,
                        profile_id="profile_1",
                    ).recommendation_id,
                    profile_1.recommendation_id,
                )
                self.assertEqual(
                    recommendations.get_current(
                        "install_1",
                        "machine_1",
                        None,
                        now=10,
                        profile_id="profile_2",
                    ).recommendation_id,
                    profile_2.recommendation_id,
                )

                status = build_status_payload(
                    config=config,
                    service=service,
                    shot_repo=shots,
                    upload_maintenance=None,
                    upload_queue_repo=None,
                    machine_id="machine_1",
                    bean_context_id=None,
                    profile_id="profile_1",
                    profile_label="Cremina lever machine",
                )

        self.assertEqual(status["optimizer_profile_id"], "profile_1")
        self.assertEqual(status["optimizer_profile_label"], "Cremina lever machine")
        self.assertEqual(status["local_shot_count"], 1)
        self.assertEqual(status["rated_shot_count"], 1)
        self.assertEqual([shot["shot_id"] for shot in status["recent_shots"]], ["shot_1"])
        self.assertEqual(status["best_known_recipe"]["shot_id"], "shot_1")

    def test_status_payload_includes_redacted_runtime_health(self) -> None:
        secret = "s" * 32
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(
                mqtt_host="localhost",
                data_dir=Path(tmp),
                install_id="install_1",
                community_upload_enabled=True,
                supabase_ingest_url="https://project.supabase.co/functions/v1/espresso-rl-ingest",
                upload_secret=secret,
            )
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                service = EspressoRLService(
                    SQLiteShotRepository(store),
                    SQLiteRecommendationRepository(store),
                    ConservativeBOOptimizer(),
                    clock=lambda: 10,
                )
                status = build_status_payload(
                    config=config,
                    service=service,
                    shot_repo=None,
                    upload_maintenance=None,
                    upload_queue_repo=SQLiteUploadQueueRepository(store),
                    machine_id="machine_1",
                    bean_context_id="bean_1",
                    grinder_context_id="grinder_1",
                )

        runtime_health = {
            key: value for key, value in status.items() if key.startswith("runtime_health")
        }
        encoded = json.dumps(runtime_health)
        self.assertEqual(status["runtime_health_status"], "waiting")
        self.assertEqual(status["runtime_health_summary"], "Ready - waiting for the first shot")
        self.assertTrue(status["runtime_health_upload_configured"])
        self.assertEqual(status["runtime_health_storage_backend"], "sqlite")
        self.assertEqual(status["runtime_health_warnings"], [])
        self.assertEqual(
            [step["key"] for step in status["auto_tuning_diagnostic_steps"]],
            [
                "context",
                "shot_observed",
                "shot_stored",
                "shot_usable",
                "rating",
                "recommendation",
                "community_upload",
                "status_published",
            ],
        )
        self.assertNotIn(secret, encoded)
        self.assertNotIn("supabase.co", encoded)
        self.assertNotIn("espresso-rl-ingest", encoded)

    def test_status_payload_diagnostic_flags_observed_shot_missing_from_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(mqtt_host="localhost", data_dir=Path(tmp), install_id="install_1")
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                service = EspressoRLService(
                    shots,
                    SQLiteRecommendationRepository(store),
                    ConservativeBOOptimizer(),
                    clock=lambda: 10,
                )

                status = build_status_payload(
                    config=config,
                    service=service,
                    shot_repo=shots,
                    upload_maintenance=None,
                    upload_queue_repo=None,
                    machine_id="machine_1",
                    bean_context_id="bean_1",
                    grinder_context_id="grinder_1",
                    last_shot_id="shot_observed_but_missing",
                    last_shot_at=9,
                )

        steps = {step["key"]: step for step in status["auto_tuning_diagnostic_steps"]}
        self.assertEqual(steps["shot_observed"]["state"], "ok")
        self.assertEqual(steps["shot_stored"]["state"], "attention")
        self.assertIn("not found", steps["shot_stored"]["detail"])
        self.assertEqual(steps["shot_usable"]["state"], "waiting")
        self.assertEqual(steps["status_published"]["state"], "ok")

    def test_status_payload_reports_optimizer_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(
                mqtt_host="localhost",
                data_dir=Path(tmp),
                install_id="install_1",
                optimizer_mode="dreamer_v3_shadow",
            )
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                service = EspressoRLService(
                    SQLiteShotRepository(store),
                    SQLiteRecommendationRepository(store),
                    ConservativeBOOptimizer(),
                    clock=lambda: 10,
                )

                status = build_status_payload(
                    config=config,
                    service=service,
                    shot_repo=None,
                    upload_maintenance=None,
                    upload_queue_repo=None,
                    machine_id="machine_1",
                    bean_context_id=None,
                    optimizer_status={
                        "configured_mode": "bayesian_optimization",
                        "effective_mode": "bayesian_optimization",
                        "model_artifact_path": "models/dreamer.pt",
                        "model_artifact_sha256": "a" * 64,
                        "model_artifact_actual_sha256": "a" * 64,
                        "model_artifact_size_bytes": 123,
                        "model_artifact_verified": True,
                        "model_artifact_unavailable_reason": None,
                        "model_manifest_path": "models/dreamer_manifest.json",
                        "model_manifest_sha256": "b" * 64,
                        "model_manifest_size_bytes": 321,
                        "model_manifest_verified": True,
                        "model_manifest_unavailable_reason": None,
                        "model_manifest_model_family": "dreamer_v3",
                        "model_manifest_artifact_format": "safetensors",
                        "model_manifest_dataset_sha256": "c" * 64,
                        "model_manifest_dataset_manifest_sha256": "d" * 64,
                        "model_manifest_trainer_git_sha": "trainerabc",
                        "model_manifest_training_config_sha256": "e" * 64,
                        "model_manifest_state_schema_version": 1,
                        "model_manifest_action_schema_version": 1,
                        "model_manifest_reward_schema_version": 1,
                        "checkpoint_verified": True,
                        "checkpoint_inference_ready": False,
                        "checkpoint_tensor_count": 3,
                        "checkpoint_component_names": ["actor", "critic", "world_model"],
                        "checkpoint_unavailable_reason": "Runtime inference is not enabled.",
                        "checkpoint_inference_parity_verified": True,
                        "checkpoint_inference_parity_reason": None,
                        "dreamer_v3_available": False,
                        "dreamer_v3_active_recommendation_count": 2,
                        "dreamer_v3_bo_fallback_count": 1,
                        "dreamer_v3_bo_fallback_reason_counts": {"dreamer_candidate_rejected": 1},
                        "dreamer_v3_last_runtime_event": "bo_fallback",
                        "dreamer_v3_last_bo_fallback_reason": "dreamer_candidate_rejected",
                        "available_modes": ["bayesian_optimization"],
                        "unavailable_modes": {"dreamer_v3_shadow": "Runtime inference is not enabled."},
                        "fallback_reason": None,
                    },
                )

            self.assertEqual(status["optimizer_configured_mode"], "bayesian_optimization")
            self.assertEqual(status["optimizer_effective_mode"], "bayesian_optimization")
            self.assertFalse(status["optimizer_dreamer_v3_available"])
            self.assertTrue(status["optimizer_checkpoint_verified"])
            self.assertFalse(status["optimizer_checkpoint_inference_ready"])
            self.assertEqual(status["optimizer_checkpoint_tensor_count"], 3)
            self.assertEqual(status["optimizer_checkpoint_component_names"], ["actor", "critic", "world_model"])
            self.assertTrue(status["optimizer_checkpoint_inference_parity_verified"])
            self.assertEqual(status["optimizer_model_artifact_path"], "models/dreamer.pt")
            self.assertEqual(status["optimizer_model_artifact_actual_sha256"], "a" * 64)
            self.assertTrue(status["optimizer_model_artifact_verified"])
            self.assertEqual(status["optimizer_model_artifact_size_bytes"], 123)
            self.assertEqual(status["optimizer_model_manifest_path"], "models/dreamer_manifest.json")
            self.assertTrue(status["optimizer_model_manifest_verified"])
            self.assertEqual(status["optimizer_model_manifest_artifact_format"], "safetensors")
            self.assertEqual(status["optimizer_model_manifest_dataset_sha256"], "c" * 64)
            self.assertEqual(status["optimizer_model_manifest_trainer_git_sha"], "trainerabc")
            self.assertNotIn("dreamer_v3_shadow", status["optimizer_available_modes"])
            self.assertIn("not enabled", status["optimizer_checkpoint_unavailable_reason"])
            self.assertEqual(status["optimizer_dreamer_v3_active_recommendation_count"], 2)
            self.assertEqual(status["optimizer_dreamer_v3_bo_fallback_count"], 1)
            self.assertEqual(
                status["optimizer_dreamer_v3_bo_fallback_reason_counts"],
                {"dreamer_candidate_rejected": 1},
            )
            self.assertEqual(status["optimizer_dreamer_v3_last_runtime_event"], "bo_fallback")
            self.assertEqual(
                status["optimizer_dreamer_v3_last_bo_fallback_reason"],
                "dreamer_candidate_rejected",
            )

    def test_status_payload_exposes_only_sanitized_live_ack_health(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(mqtt_host="localhost", data_dir=Path(tmp), install_id="install_1")
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                service = EspressoRLService(
                    SQLiteShotRepository(store),
                    SQLiteRecommendationRepository(store),
                    ConservativeBOOptimizer(),
                    clock=lambda: 10,
                )

                status = build_status_payload(
                    config=config,
                    service=service,
                    shot_repo=None,
                    upload_maintenance=None,
                    upload_queue_repo=None,
                    machine_id="machine_1",
                    bean_context_id=None,
                    dreamer_live_ack_summary={
                        "health": "attention",
                        "published_count": 4,
                        "pending_count": 1,
                        "accepted_count": 2,
                        "rejected_count": 1,
                        "duplicate_ack_count": 1,
                        "late_ack_count": 0,
                        "mismatched_ack_count": 1,
                        "unknown_ack_count": 0,
                        "timed_out_count": 1,
                        "last_result": "rejected",
                        "last_reason_category": "out_of_bounds",
                        "last_event_at_ms": 12_000,
                        "publication_id": "must-not-leak",
                        "reason": "pressure_target_bar_out_of_bounds",
                        "target_update": {"pressure_target_bar": 99},
                    },
                )

        summary = status["dreamer_live_control_ack"]
        self.assertEqual(summary["health"], "attention")
        self.assertEqual(summary["published_count"], 4)
        self.assertEqual(summary["last_reason_category"], "out_of_bounds")
        self.assertNotIn("publication_id", summary)
        self.assertNotIn("reason", summary)
        self.assertNotIn("target_update", summary)
        self.assertNotIn("pressure_target_bar_out_of_bounds", str(summary))

    def test_status_payload_exposes_only_aggregate_shadow_quality_results(self) -> None:
        class AggregateReport:
            def status_summary(self):
                return {
                    "report_id": "shadow_quality_report_1",
                    "generated_at": 20,
                    "overall_status": "insufficient_data",
                    "evaluated_record_count": 3,
                    "gates": [{"name": "minimum_evidence", "status": "insufficient_data"}],
                    "observational_only": True,
                    "shadow_only": True,
                    "recommendation_enabled": False,
                    "machine_control_enabled": False,
                }

        class AggregateQualityService:
            def build_context_report(self, **scope):
                self.scope = scope
                return AggregateReport()

        with tempfile.TemporaryDirectory() as tmp:
            config = Config(mqtt_host="localhost", data_dir=Path(tmp), install_id="install_1")
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                service = EspressoRLService(
                    SQLiteShotRepository(store),
                    SQLiteRecommendationRepository(store),
                    ConservativeBOOptimizer(),
                    clock=lambda: 10,
                )
                quality = AggregateQualityService()

                status = build_status_payload(
                    config=config,
                    service=service,
                    shot_repo=None,
                    upload_maintenance=None,
                    upload_queue_repo=None,
                    machine_id="machine_1",
                    bean_context_id="bean_1",
                    grinder_context_id="grinder_1",
                    shadow_quality_service=quality,
                )

        summary = status["dreamer_shadow_quality_report"]
        self.assertEqual(summary["report_id"], "shadow_quality_report_1")
        self.assertEqual(summary["overall_status"], "insufficient_data")
        self.assertFalse(summary["recommendation_enabled"])
        self.assertFalse(summary["machine_control_enabled"])
        self.assertNotIn("dreamer_proposal", summary)
        self.assertEqual(quality.scope["bean_context_id"], "bean_1")

    def test_status_payload_derives_grinder_catalog_search_url_from_supabase_function_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(
                mqtt_host="localhost",
                data_dir=Path(tmp),
                install_id="install_1",
                supabase_ingest_url="https://project.supabase.co/functions/v1/espresso-rl-ingest",
            )
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                service = EspressoRLService(
                    SQLiteShotRepository(store),
                    SQLiteRecommendationRepository(store),
                    ConservativeBOOptimizer(),
                    clock=lambda: 10,
                )

                status = build_status_payload(
                    config=config,
                    service=service,
                    shot_repo=None,
                    upload_maintenance=None,
                    upload_queue_repo=None,
                    machine_id="machine_1",
                    bean_context_id=None,
                )

            self.assertEqual(
                status["grinder_catalog_search_url"],
                "https://project.supabase.co/functions/v1/espresso-rl-grinder-search",
            )
            self.assertEqual(
                status["prior_rule_catalog_search_url"],
                "https://project.supabase.co/functions/v1/espresso-rl-prior-rule-search",
            )

    def test_status_payload_derives_grinder_catalog_search_url_from_registration_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(
                mqtt_host="localhost",
                data_dir=Path(tmp),
                install_id="install_1",
                supabase_registration_url="https://project.supabase.co/functions/v1/espresso-rl-register",
            )
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                service = EspressoRLService(
                    SQLiteShotRepository(store),
                    SQLiteRecommendationRepository(store),
                    ConservativeBOOptimizer(),
                    clock=lambda: 10,
                )

                status = build_status_payload(
                    config=config,
                    service=service,
                    shot_repo=None,
                    upload_maintenance=None,
                    upload_queue_repo=None,
                    machine_id="machine_1",
                    bean_context_id=None,
                )

            self.assertEqual(
                status["grinder_catalog_search_url"],
                "https://project.supabase.co/functions/v1/espresso-rl-grinder-search",
            )
            self.assertEqual(
                status["prior_rule_catalog_search_url"],
                "https://project.supabase.co/functions/v1/espresso-rl-prior-rule-search",
            )

    def test_postgres_upsert_rolls_back_failed_transaction(self) -> None:
        class FailingConnection:
            def __init__(self) -> None:
                self.rolled_back = False
                self.committed = False

            def execute(self, *_args: object, **_kwargs: object) -> None:
                raise RuntimeError("database rejected row")

            def commit(self) -> None:
                self.committed = True

            def rollback(self) -> None:
                self.rolled_back = True

        conn = FailingConnection()

        with self.assertRaises(RuntimeError):
            _upsert(conn, "shots", "shot_id", {"shot_id": "shot_1"})

        self.assertTrue(conn.rolled_back)
        self.assertFalse(conn.committed)

    def test_sqlite_upload_queue_tracks_retry_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                queue.enqueue(
                    UploadQueueItem(
                        upload_id="upload_1",
                        local_record_type="shot",
                        local_record_id="shot_1",
                        payload_hash=hash_payload_json(valid_upload_payload()),
                        payload_json=valid_upload_payload(),
                        status=UploadQueueStatus.PENDING,
                        created_at=1,
                        updated_at=1,
                    )
                )

                self.assertEqual([item.upload_id for item in queue.list_ready(now=2)], ["upload_1"])
                queue.update_status(
                    upload_id="upload_1",
                    status=UploadQueueStatus.FAILED,
                    now=3,
                    error_message="network",
                    next_retry_at=10,
                )
                self.assertEqual(queue.list_ready(now=4), [])
                ready = queue.list_ready(now=10)
                self.assertEqual(ready[0].attempt_count, 1)
                self.assertEqual(ready[0].error_message, "network")

    def test_upload_worker_marks_successful_records_uploaded(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.uploaded: list[str] = []

            def upload(self, item: UploadQueueItem) -> None:
                self.uploaded.append(item.upload_id)

        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                payload_json = valid_upload_payload()
                queue.enqueue(
                    UploadQueueItem(
                        upload_id="upload_1",
                        local_record_type="shot",
                        local_record_id="shot_1",
                        payload_hash=hash_payload_json(payload_json),
                        payload_json=payload_json,
                        status=UploadQueueStatus.PENDING,
                        created_at=1,
                        updated_at=1,
                    )
                )
                client = FakeClient()
                worker = UploadQueueWorker(queue, client, clock=lambda: 5)

                self.assertEqual(worker.run_once(), 1)
                self.assertEqual(client.uploaded, ["upload_1"])
                self.assertEqual(queue.list_ready(now=6), [])

    def test_upload_worker_rejects_payload_hash_mismatch_before_client_upload(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.uploaded: list[str] = []

            def upload(self, item: UploadQueueItem) -> None:
                self.uploaded.append(item.upload_id)

        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                queue.enqueue(
                    UploadQueueItem(
                        upload_id="upload_1",
                        local_record_type="shot",
                        local_record_id="shot_1",
                        payload_hash="0" * 64,
                        payload_json=valid_upload_payload(),
                        status=UploadQueueStatus.PENDING,
                        created_at=1,
                        updated_at=1,
                    )
                )
                client = FakeClient()
                worker = UploadQueueWorker(queue, client, clock=lambda: 5)

                self.assertEqual(worker.run_once(), 0)
                self.assertEqual(client.uploaded, [])
                self.assertEqual(
                    store.conn.execute("SELECT COUNT(*) AS c FROM upload_queue").fetchone()["c"],
                    0,
                )

    def test_signed_client_rejects_install_id_mismatch_before_network(self) -> None:
        payload_json = valid_upload_payload().replace('"install_id": "install_1"', '"install_id": "other"')
        item = UploadQueueItem(
            upload_id="upload_1",
            local_record_type="shot",
            local_record_id="shot_1",
            payload_hash=hash_payload_json(payload_json),
            payload_json=payload_json,
            status=UploadQueueStatus.PENDING,
            created_at=1,
            updated_at=1,
        )
        client = SignedSupabaseUploadClient(
            SignedUploadConfig(
                ingest_url="https://example.invalid/ingest",
                install_id="install_1",
                upload_secret="x" * 32,
            )
        )

        with self.assertRaises(UploadRejected) as raised:
            client.upload(item)

        self.assertEqual(raised.exception.status, 422)
        self.assertIn("install_id", str(raised.exception))

    def test_signed_client_classifies_credential_rejection_separately_from_payload_rejection(self) -> None:
        client = SignedSupabaseUploadClient(
            SignedUploadConfig(
                ingest_url="https://example.invalid/ingest",
                install_id="install_1",
                upload_secret="x" * 32,
            )
        )
        response = error.HTTPError(
            "https://example.invalid/ingest",
            403,
            "Forbidden",
            {},
            io.BytesIO(b'{"error":"unknown or revoked upload credential"}'),
        )

        with mock.patch("espresso_rl.adapters.supabase_upload.request.urlopen", side_effect=response):
            with self.assertRaises(UploadCredentialRejected) as raised:
                client.upload(uploadable_queue_item("upload_1"))

        self.assertEqual(raised.exception.status, 403)
        self.assertIn("unknown or revoked", str(raised.exception))

    def test_enqueue_coalesces_pending_versions_of_same_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                for payload_hash in ("h1", "h2", "h3"):
                    queue.enqueue(queue_item(f"shot_shot_1_{payload_hash}", payload_hash))
                ready = queue.list_ready(now=10)
                self.assertEqual(len(ready), 1)
                self.assertEqual(ready[0].payload_hash, "h3")  # newest queued state wins

    def test_enqueue_skips_content_already_uploaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                queue.enqueue(queue_item("u_abc", "abc"))
                queue.update_status("u_abc", UploadQueueStatus.UPLOADED, now=2)
                queue.enqueue(queue_item("u_abc", "abc"))  # identical content already sent
                self.assertEqual(queue.list_ready(now=10), [])
                count = store.conn.execute(
                    "SELECT COUNT(*) AS c FROM upload_queue WHERE local_record_id='shot_1'"
                ).fetchone()["c"]
                self.assertEqual(count, 1)

    def test_enqueue_rearms_when_content_changes_after_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                queue.enqueue(queue_item("u_a", "a"))
                queue.update_status("u_a", UploadQueueStatus.UPLOADED, now=2)
                queue.enqueue(queue_item("u_b", "b"))  # e.g. a rating added later
                self.assertEqual([item.upload_id for item in queue.list_ready(now=10)], ["u_b"])
                count = store.conn.execute(
                    "SELECT COUNT(*) AS c FROM upload_queue WHERE local_record_id='shot_1'"
                ).fetchone()["c"]
                self.assertEqual(count, 2)  # uploaded 'a' kept as memory + pending 'b'

    def test_enqueue_rearms_same_upload_id_when_content_changes_after_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                queue.enqueue(
                    queue_item(
                        "recommendation_rec_1",
                        "a",
                        local_record_type="recommendation",
                        local_record_id="rec_1",
                    )
                )
                queue.update_status("recommendation_rec_1", UploadQueueStatus.UPLOADED, now=2)
                queue.enqueue(
                    queue_item(
                        "recommendation_rec_1",
                        "b",
                        local_record_type="recommendation",
                        local_record_id="rec_1",
                    )
                )

                ready = queue.list_ready(now=10)
                self.assertEqual([item.upload_id for item in ready], ["recommendation_rec_1"])
                self.assertEqual(ready[0].payload_hash, "b")
                count = store.conn.execute(
                    "SELECT COUNT(*) AS c FROM upload_queue WHERE local_record_id='rec_1'"
                ).fetchone()["c"]
                self.assertEqual(count, 1)

    def test_enqueue_never_deletes_in_flight_uploading_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                queue.enqueue(queue_item("u_a", "a"))
                queue.update_status("u_a", UploadQueueStatus.UPLOADING, now=2)
                queue.enqueue(queue_item("u_b", "b"))  # coalesce must leave u_a alone
                statuses = {
                    row["upload_id"]: row["status"]
                    for row in store.conn.execute(
                        "SELECT upload_id, status FROM upload_queue WHERE local_record_id='shot_1'"
                    ).fetchall()
                }
                self.assertEqual(statuses, {"u_a": "uploading", "u_b": "pending"})

    def test_enqueue_rearms_rejected_record_when_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                queue.enqueue(queue_item("u_a", "a"))
                queue.update_status("u_a", UploadQueueStatus.REJECTED, now=2, error_message="schema")
                queue.enqueue(queue_item("u_b", "b"))

                ready = queue.list_ready(now=10)
                self.assertEqual([item.upload_id for item in ready], ["u_b"])
                statuses = {
                    row["upload_id"]: row["status"]
                    for row in store.conn.execute(
                        "SELECT upload_id, status FROM upload_queue WHERE local_record_id='shot_1'"
                    ).fetchall()
                }
                self.assertEqual(statuses, {"u_b": "pending"})

    def test_upload_queue_counts_by_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                queue.enqueue(queue_item("u_a", "a"))
                queue.enqueue(queue_item("u_b", "b", local_record_id="shot_2"))
                queue.update_status("u_b", UploadQueueStatus.REJECTED, now=2, error_message="schema")

                counts = queue.count_by_status()

                self.assertEqual(counts[UploadQueueStatus.PENDING], 1)
                self.assertEqual(counts[UploadQueueStatus.REJECTED], 1)

    def test_uploading_transition_is_not_counted_as_an_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                queue.enqueue(queue_item("u_a", "a"))
                queue.update_status("u_a", UploadQueueStatus.UPLOADING, now=2)
                queue.update_status("u_a", UploadQueueStatus.FAILED, now=3, next_retry_at=10)
                ready = queue.list_ready(now=10)
                self.assertEqual(ready[0].attempt_count, 1)  # one failed try, not two

    def test_worker_keeps_transient_failures_retryable(self) -> None:
        class BoomClient:
            def upload(self, item: UploadQueueItem) -> None:
                raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                queue.enqueue(uploadable_queue_item("u_a", attempt_count=99))
                worker = UploadQueueWorker(queue, BoomClient(), clock=lambda: 100)
                worker.run_once()
                row = store.conn.execute(
                    "SELECT status, attempt_count, next_retry_at FROM upload_queue WHERE upload_id='u_a'"
                ).fetchone()
                self.assertEqual(row["status"], "failed")
                self.assertEqual(row["attempt_count"], 100)
                self.assertGreater(row["next_retry_at"], 100)
                self.assertEqual([item.upload_id for item in queue.list_ready(now=10_000)], ["u_a"])

    def test_worker_notifies_once_after_each_queue_changing_cycle(self) -> None:
        class SuccessfulClient:
            def upload(self, item: UploadQueueItem) -> None:
                return None

        class RejectingClient:
            def upload(self, item: UploadQueueItem) -> None:
                raise UploadRejected(422, "schema")

        class FailingClient:
            def upload(self, item: UploadQueueItem) -> None:
                raise RuntimeError("network")

        for label, client in (
            ("uploaded", SuccessfulClient()),
            ("rejected", RejectingClient()),
            ("failed", FailingClient()),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                with SQLiteStore(Path(tmp) / "espresso.db") as store:
                    queue = SQLiteUploadQueueRepository(store)
                    queue.enqueue(uploadable_queue_item("u_a"))
                    notifications: list[str] = []
                    worker = UploadQueueWorker(
                        queue,
                        client,
                        clock=lambda: 100,
                        on_queue_changed=lambda: notifications.append("changed"),
                    )

                    worker.run_once()

                    self.assertEqual(notifications, ["changed"])

    def test_worker_does_not_notify_when_queue_is_idle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                notifications: list[str] = []
                worker = UploadQueueWorker(
                    SQLiteUploadQueueRepository(store),
                    object(),
                    clock=lambda: 100,
                    on_queue_changed=lambda: notifications.append("changed"),
                )

                worker.run_once()

                self.assertEqual(notifications, [])

    def test_worker_discards_rejected_upload_snapshot_without_deleting_local_shot(self) -> None:
        class RejectingClient:
            def upload(self, item: UploadQueueItem) -> None:
                raise UploadRejected(422, "schema")

        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                queue = SQLiteUploadQueueRepository(store)
                EspressoRLService(
                    shots,
                    SQLiteRecommendationRepository(store),
                    ConservativeBOOptimizer(),
                ).ingest_shot_profile(shot_event())
                queue.enqueue(uploadable_queue_item("u_a"))
                worker = UploadQueueWorker(queue, RejectingClient(), clock=lambda: 100)

                worker.run_once()

                self.assertIsNotNone(shots.get("shot_1"))
                self.assertEqual(
                    store.conn.execute("SELECT COUNT(*) AS c FROM upload_queue").fetchone()["c"],
                    0,
                )

    def test_worker_discards_credential_rejected_upload_snapshot_without_deleting_local_shot(self) -> None:
        class CredentialRejectingClient:
            def upload(self, item: UploadQueueItem) -> None:
                raise UploadCredentialRejected(403, "unknown or revoked upload credential")

        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                shots = SQLiteShotRepository(store)
                queue = SQLiteUploadQueueRepository(store)
                EspressoRLService(shots, SQLiteRecommendationRepository(store), ConservativeBOOptimizer()).ingest_shot_profile(
                    shot_event()
                )
                queue.enqueue(uploadable_queue_item("u_a"))
                worker = UploadQueueWorker(queue, CredentialRejectingClient(), clock=lambda: 100)

                worker.run_once()

                self.assertIsNotNone(shots.get("shot_1"))
                self.assertEqual(
                    store.conn.execute("SELECT COUNT(*) AS c FROM upload_queue").fetchone()["c"],
                    0,
                )

    def test_worker_defers_rate_limited_upload_without_charging_attempt(self) -> None:
        class LimitedClient:
            def upload(self, item: UploadQueueItem) -> None:
                raise UploadRateLimited(retry_after=120)

        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                queue.enqueue(uploadable_queue_item("u_a"))
                worker = UploadQueueWorker(queue, LimitedClient(), clock=lambda: 1000)
                worker.run_once()
                self.assertEqual(queue.list_ready(now=1001), [])  # deferred past now
                ready = queue.list_ready(now=2000)
                self.assertEqual(len(ready), 1)
                self.assertEqual(ready[0].attempt_count, 0)  # rate limiting never charges an attempt
                self.assertEqual(ready[0].status, UploadQueueStatus.PENDING)

    def test_worker_rate_limited_without_header_defers_to_utc_day_reset(self) -> None:
        class LimitedClient:
            def upload(self, item: UploadQueueItem) -> None:
                raise UploadRateLimited(retry_after=None)

        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                queue.enqueue(uploadable_queue_item("u_a"))
                worker = UploadQueueWorker(queue, LimitedClient(), clock=lambda: 1000)
                worker.run_once()
                next_retry_at = store.conn.execute(
                    "SELECT next_retry_at FROM upload_queue WHERE upload_id='u_a'"
                ).fetchone()["next_retry_at"]
                self.assertEqual(next_retry_at, 1000 + (86_400 - (1000 % 86_400)))

    def test_admin_role_never_pushes_to_community_upload_queue(self) -> None:
        config = Config(
            mqtt_host="localhost",
            community_upload_enabled=True,
            supabase_ingest_url="https://example.invalid/ingest",
            upload_secret="x" * 32,
            deployment_role="admin",
        )

        self.assertFalse(config.should_enqueue_community_uploads())
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "espresso.db") as store:
                queue = SQLiteUploadQueueRepository(store)
                self.assertIsNone(upload_queue_for_service(config, queue))
                worker = maybe_start_upload_worker(
                    config,
                    queue,
                    threading.Event(),
                )
        self.assertIsNone(worker)

    def test_postgres_schema_defines_public_and_admin_storage_tables(self) -> None:
        schema = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "espresso_rl"
            / "adapters"
            / "postgres_schema.sql"
        ).read_text()

        for table_name in (
            "shots",
            "recommendations",
            "upload_queue",
            "dreamer_shadow_evaluations",
            "dreamer_shadow_quality_reports",
            "community_raw_uploads",
            "community_validated_shots",
            "community_recommendations",
            "install_trust_scores",
            "abuse_events",
            "training_dataset",
            "community_priors",
            "community_grinder_catalog",
            "community_grinder_aliases",
        ):
            self.assertIn(f"CREATE TABLE IF NOT EXISTS {table_name}", schema)

        self.assertIn("PRIMARY KEY (install_id, upload_id)", schema)
        self.assertIn("UNIQUE (install_id, payload_hash)", schema)
        self.assertIn("validation_summary JSONB", schema)
        self.assertIn("validation_errors JSONB", schema)
        self.assertIn("UNIQUE (source_validation_id)", schema)
        self.assertIn("idx_community_priors_context_key", schema)
        self.assertIn("microns_per_step DOUBLE PRECISION", schema)
        self.assertIn("grind_observed BOOLEAN NOT NULL DEFAULT TRUE", schema)
        self.assertIn("dose_observed BOOLEAN NOT NULL DEFAULT TRUE", schema)
        self.assertIn("target_yield_observed BOOLEAN NOT NULL DEFAULT TRUE", schema)
        self.assertIn("min_steps INTEGER", schema)
        self.assertIn("max_steps INTEGER", schema)
        self.assertIn("normalized_alias TEXT NOT NULL", schema)
        self.assertIn("feedback_recorded BOOLEAN NOT NULL DEFAULT FALSE", schema)
        self.assertIn("ADD COLUMN IF NOT EXISTS feedback_recorded", schema)
        self.assertIn("ADD COLUMN IF NOT EXISTS relative_grind_steps_from_reference", schema)
        self.assertIn("ADD COLUMN IF NOT EXISTS relative_grind_um_from_reference", schema)
        self.assertIn("ADD COLUMN IF NOT EXISTS microns_per_step", schema)
        self.assertIn(
            "ADD COLUMN IF NOT EXISTS recommended_grind_delta_steps_from_current",
            schema,
        )
        self.assertIn(
            "ADD COLUMN IF NOT EXISTS recommended_grind_delta_um_from_current",
            schema,
        )
        self.assertIn(
            "ADD COLUMN IF NOT EXISTS recommended_projected_relative_step_from_reference",
            schema,
        )
        self.assertIn(
            "ADD COLUMN IF NOT EXISTS grind_delta_steps_from_current",
            schema,
        )
        self.assertIn(
            "ADD COLUMN IF NOT EXISTS grind_delta_um_from_current",
            schema,
        )
        self.assertIn(
            "ADD COLUMN IF NOT EXISTS projected_relative_step_from_reference",
            schema,
        )
        self.assertIn(
            "ADD COLUMN IF NOT EXISTS projected_relative_grind_um_from_reference",
            schema,
        )

    def test_postgres_store_runs_legacy_migrations_before_indexes(self) -> None:
        class RecordingConnection:
            def __init__(self) -> None:
                self.statements: list[str] = []
                self.committed = False

            def execute(self, statement: str, params: tuple[object, ...] | None = None) -> RecordingConnection:
                self.statements.append(statement)
                self.params = params
                return self

            def fetchone(self) -> dict[str, bool]:
                return {"exists": True}

            def commit(self) -> None:
                self.committed = True

        store = object.__new__(PostgresStore)
        store.conn = RecordingConnection()

        PostgresStore._create_tables(store)

        statements = [" ".join(statement.split()) for statement in store.conn.statements]

        def statement_index(*fragments: str) -> int:
            for index, statement in enumerate(statements):
                if all(fragment in statement for fragment in fragments):
                    return index
            self.fail(f"statement not found: {fragments}")

        shadow_eval_migration = statement_index(
            "ALTER TABLE dreamer_shadow_evaluations",
            "ADD COLUMN IF NOT EXISTS inference_contract_id",
        )
        shadow_eval_index = statement_index("CREATE INDEX IF NOT EXISTS idx_dreamer_shadow_context_contract")
        shadow_quality_migration = statement_index(
            "ALTER TABLE dreamer_shadow_quality_reports",
            "ADD COLUMN IF NOT EXISTS inference_contract_id",
        )
        shadow_quality_index = statement_index(
            "CREATE INDEX IF NOT EXISTS idx_dreamer_shadow_quality_context_contract"
        )
        shot_grind_delta_add = statement_index(
            "ALTER TABLE shots",
            "ADD COLUMN IF NOT EXISTS recommended_grind_delta_steps_from_current",
        )
        shot_relative_grind_add = statement_index(
            "ALTER TABLE shots",
            "ADD COLUMN IF NOT EXISTS relative_grind_steps_from_reference",
        )
        shot_microns_add = statement_index(
            "ALTER TABLE shots",
            "ADD COLUMN IF NOT EXISTS microns_per_step",
        )
        shot_grind_delta_type = statement_index(
            "ALTER TABLE shots",
            "ALTER COLUMN recommended_grind_delta_steps_from_current TYPE DOUBLE PRECISION",
        )
        legacy_grinder_step_size_nullable = statement_index(
            "ALTER TABLE shots",
            "ALTER COLUMN grinder_step_size_um DROP NOT NULL",
        )
        legacy_grinder_step_size_default = statement_index(
            "ALTER TABLE shots",
            "ALTER COLUMN grinder_step_size_um SET DEFAULT 12.5",
        )
        recommendation_grind_delta_add = statement_index(
            "ALTER TABLE recommendations",
            "ADD COLUMN IF NOT EXISTS grind_delta_steps_from_current",
        )
        recommendation_grind_delta_um_add = statement_index(
            "ALTER TABLE recommendations",
            "ADD COLUMN IF NOT EXISTS grind_delta_um_from_current",
        )
        recommendation_projected_relative_add = statement_index(
            "ALTER TABLE recommendations",
            "ADD COLUMN IF NOT EXISTS projected_relative_step_from_reference",
        )
        legacy_recommendation_next_grind_nullable = statement_index(
            "ALTER TABLE recommendations",
            "ALTER COLUMN next_grind_steps DROP NOT NULL",
        )
        recommendation_grind_delta_type = statement_index(
            "ALTER TABLE recommendations",
            "ALTER COLUMN grind_delta_steps_from_current TYPE DOUBLE PRECISION",
        )

        self.assertLess(shadow_eval_migration, shadow_eval_index)
        self.assertLess(shadow_quality_migration, shadow_quality_index)
        self.assertLess(shot_relative_grind_add, shot_grind_delta_type)
        self.assertLess(shot_microns_add, shot_grind_delta_type)
        self.assertLess(shot_grind_delta_add, shot_grind_delta_type)
        self.assertLess(shot_microns_add, legacy_grinder_step_size_nullable)
        self.assertLess(legacy_grinder_step_size_nullable, legacy_grinder_step_size_default)
        self.assertLess(recommendation_grind_delta_add, recommendation_grind_delta_type)
        self.assertLess(recommendation_grind_delta_um_add, recommendation_grind_delta_type)
        self.assertLess(recommendation_projected_relative_add, recommendation_grind_delta_type)
        self.assertLess(recommendation_projected_relative_add, legacy_recommendation_next_grind_nullable)
        self.assertTrue(store.conn.committed)

    def test_postgres_upload_queue_ready_read_commits_transaction(self) -> None:
        class EmptyResult:
            def fetchall(self) -> list[dict[str, object]]:
                return []

        class RecordingConnection:
            def __init__(self) -> None:
                self.statements: list[str] = []
                self.commits = 0
                self.rollbacks = 0

            def execute(self, statement: str, params: tuple[object, ...]) -> EmptyResult:
                self.statements.append(statement)
                self.params = params
                return EmptyResult()

            def commit(self) -> None:
                self.commits += 1

            def rollback(self) -> None:
                self.rollbacks += 1

        class RecordingStore:
            def __init__(self) -> None:
                self.conn = RecordingConnection()

        store = RecordingStore()
        repo = PostgresUploadQueueRepository(store)  # type: ignore[arg-type]

        self.assertEqual([], repo.list_ready(now=123, limit=10))
        self.assertEqual(1, store.conn.commits)
        self.assertEqual(0, store.conn.rollbacks)
        self.assertIn("SELECT * FROM upload_queue", store.conn.statements[0])

    def test_postgres_upload_queue_ready_reopens_closed_connection(self) -> None:
        class ClosedConnection:
            closed = True

        class EmptyResult:
            def fetchall(self) -> list[dict[str, object]]:
                return []

        class OpenConnection:
            closed = False

            def __init__(self) -> None:
                self.statements: list[str] = []
                self.commits = 0

            def execute(self, statement: str, params: tuple[object, ...]) -> EmptyResult:
                self.statements.append(statement)
                self.params = params
                return EmptyResult()

            def commit(self) -> None:
                self.commits += 1

            def rollback(self) -> None:
                raise AssertionError("rollback should not be called")

        store = object.__new__(PostgresStore)
        open_connection = OpenConnection()
        reconnects = 0

        def reconnect() -> None:
            nonlocal reconnects
            reconnects += 1
            store.conn = open_connection

        store.conn = ClosedConnection()
        store._connect = reconnect  # type: ignore[attr-defined]
        repo = PostgresUploadQueueRepository(store)

        self.assertEqual([], repo.list_ready(now=123, limit=10))
        self.assertEqual(1, reconnects)
        self.assertEqual(1, open_connection.commits)
        self.assertIn("SELECT * FROM upload_queue", open_connection.statements[0])

    def test_core_layers_do_not_import_adapters_or_infrastructure(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "espresso_rl"
        core_dirs = ["domain", "application", "optimizers", "ports"]
        forbidden = {
            "espresso_rl.adapters",
            "paho",
            "sqlite3",
            "supabase",
        }
        violations: list[str] = []
        for dirname in core_dirs:
            for path in (root / dirname).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    module = None
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module = alias.name
                            if any(module == item or module.startswith(f"{item}.") for item in forbidden):
                                violations.append(f"{path}: import {module}")
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        module = node.module
                        if any(module == item or module.startswith(f"{item}.") for item in forbidden):
                            violations.append(f"{path}: from {module}")
        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()

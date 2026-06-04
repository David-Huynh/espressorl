from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from espresso_rl.adapters.gaggimate_mqtt import GaggimateMQTTClient
from espresso_rl.adapters.sqlite_repositories import (
    SQLiteRecommendationRepository,
    SQLiteShotRepository,
    SQLiteStore,
)
from espresso_rl.application.services import EspressoRLService
from espresso_rl.config import Config
from espresso_rl.domain.models import (
    FollowThroughState,
    Recommendation,
    RecommendationApplyStatus,
    RecommendationMode,
    RecommendationStatus,
)
from espresso_rl.optimizers.conservative_bo import ConservativeBOOptimizer


class FakeMQTT:
    def __init__(self) -> None:
        self.published: list[tuple[str, str, int, bool]] = []

    def publish(self, topic: str, payload: str, qos: int, retain: bool) -> None:
        self.published.append((topic, payload, qos, retain))


class GaggimateAdapterTests(unittest.TestCase):
    def test_shot_profile_payload_accepts_gaggimate_recipe_metadata(self) -> None:
        client = GaggimateMQTTClient(
            config=Config(mqtt_host="localhost", data_dir=Path("/tmp")),
            on_shot=lambda event: None,
            on_feedback=lambda event: None,
            on_correction=lambda event: None,
            on_upload_maintenance=lambda event: None,
            on_decision=lambda event: None,
            on_apply=lambda event: None,
            on_machine_state=lambda event: None,
        )

        event = client.translate_shot_payload(
            {
                "shot_id": "shot_1",
                "machine_id": "gaggimate:AA_BB",
                "timestamp": 100,
                "time_ms": [0, 250, 500],
                "pressure": [0, 4, 9],
                "target_pressure": [0, 4, 9],
                "flow": [0, 1, 2],
                "target_flow": [0, 1, 2],
                "weight": [0, 8, 36],
                "dose_in_g": 18.5,
                "target_yield_g": 40.0,
                "target_ratio": 2.16,
                "beverage_out_g": 39.8,
                "shot_time_s": 31.2,
                "bean_context_id": "profile_1",
                "shot_type": "espresso",
                "utility": False,
                "local_optimization_enabled": False,
                "exclude_from_local_optimization": True,
                "optimization_weight": 0.0,
                "rating_prompt_allowed": True,
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
                "recommendation_id": "rec_1",
            },
            mac="AA_BB",
        )

        self.assertEqual(event.shot_id, "shot_1")
        self.assertEqual(event.machine_id, "gaggimate:AA_BB")
        self.assertEqual(event.dose_in_g, 18.5)
        self.assertEqual(event.target_yield_g, 40.0)
        self.assertEqual(event.beverage_out_g, 39.8)
        self.assertEqual(event.recommendation_id, "rec_1")
        self.assertEqual(event.shot_type.value, "espresso")
        self.assertFalse(event.utility)
        self.assertFalse(event.local_optimization_enabled)
        self.assertTrue(event.exclude_from_local_optimization)
        self.assertEqual(event.optimization_weight, 0.0)
        self.assertTrue(event.rating_prompt_allowed)
        self.assertEqual(event.weight_source, "hardware_scale")
        self.assertEqual(event.flow_source, "beverage_weight_derivative")
        self.assertEqual(event.flow_units, "g_per_s")
        self.assertEqual(event.pump_flow_source, "gaggimate_pump_model")
        self.assertEqual(event.pump_flow_units, "ml_per_s")
        self.assertFalse(event.pump_flow_calibration_required)
        self.assertEqual(event.profile_id, "profile_1")
        self.assertEqual(event.profile_label, "Cremina lever machine")
        self.assertEqual(event.profile_type, "pro")
        self.assertEqual(event.profile_phase_count, 5)
        self.assertEqual(event.final_phase_index, 3)
        self.assertEqual(event.final_phase_name, "ramp")
        self.assertEqual(event.final_phase_type, "brew")
        self.assertEqual(event.final_phase_elapsed_s, 8.5)
        self.assertEqual(event.final_pump_target, "pressure")
        self.assertEqual(event.final_target_pressure, 9.0)
        self.assertEqual(event.final_target_flow, 0.0)
        self.assertTrue(event.final_valve_open)
        self.assertEqual(event.profile_temperature_c, 86.5)
        self.assertEqual(event.final_phase_temperature_c, 86.5)
        self.assertEqual(event.shot_end_state, "manual_or_interrupted")

    def test_shot_profile_payload_treats_zero_final_weight_as_missing(self) -> None:
        client = GaggimateMQTTClient(
            config=Config(mqtt_host="localhost", data_dir=Path("/tmp")),
            on_shot=lambda event: None,
            on_feedback=lambda event: None,
            on_correction=lambda event: None,
            on_upload_maintenance=lambda event: None,
            on_decision=lambda event: None,
            on_apply=lambda event: None,
            on_machine_state=lambda event: None,
        )

        event = client.translate_shot_payload(
            {
                "shot_id": "shot_1",
                "timestamp": 100,
                "time_ms": [0, 250, 500],
                "pressure": [0, 4, 9],
                "target_pressure": [0, 4, 9],
                "flow": [0, 1, 2],
                "target_flow": [0, 1, 2],
                "weight": [0, 0, 0],
                "dose_in_g": 18.0,
                "target_yield_g": 36.0,
                "beverage_out_g": 0.0,
                "shot_time_s": 30.0,
            },
            mac="AA_BB",
        )

        self.assertIsNone(event.beverage_out_g)

    def test_shot_profile_payload_rejects_malformed_execution_metadata(self) -> None:
        client = GaggimateMQTTClient(
            config=Config(mqtt_host="localhost", data_dir=Path("/tmp")),
            on_shot=lambda event: None,
            on_feedback=lambda event: None,
            on_correction=lambda event: None,
            on_upload_maintenance=lambda event: None,
            on_decision=lambda event: None,
            on_apply=lambda event: None,
            on_machine_state=lambda event: None,
        )

        with self.assertRaises(ValueError):
            client.translate_shot_payload(
                {
                    "shot_id": "shot_1",
                    "timestamp": 100,
                    "time_ms": [0, 250, 500],
                    "pressure": [0, 4, 9],
                    "target_pressure": [0, 4, 9],
                    "flow": [0, 1, 2],
                    "target_flow": [0, 1, 2],
                    "weight": [0, 8, 36],
                    "dose_in_g": 18.0,
                    "target_yield_g": 36.0,
                    "beverage_out_g": 36.0,
                    "shot_time_s": 30.0,
                    "final_phase_index": 1.5,
                },
                mac="AA_BB",
            )

    def test_recommendation_payload_keeps_completed_shot_id_for_rating_popup(self) -> None:
        client = GaggimateMQTTClient(
            config=Config(mqtt_host="localhost", data_dir=Path("/tmp")),
            on_shot=lambda event: None,
            on_feedback=lambda event: None,
            on_correction=lambda event: None,
            on_upload_maintenance=lambda event: None,
            on_decision=lambda event: None,
            on_apply=lambda event: None,
            on_machine_state=lambda event: None,
        )
        fake = FakeMQTT()
        client._client = fake  # type: ignore[assignment]
        rec = Recommendation(
            recommendation_id="rec_1",
            created_at=1,
            updated_at=1,
            expires_at=None,
            install_id="install_1",
            machine_id="gaggimate:AA_BB",
            bean_context_id=None,
            grind_delta_steps=1,
            grind_delta_um=12.5,
            next_grind_steps=43,
            next_grind_um=537.5,
            next_dose_g=18.0,
            target_yield_g=36.0,
            target_ratio=2.0,
            mode=RecommendationMode.ZERO_IMMEDIATE_BO,
            confidence=0.3,
            reason="test",
            source_shot_id="shot_1",
        )

        client.publish_recommendation(rec)

        topic, payload, qos, retain = fake.published[0]
        self.assertEqual(topic, "gaggimate/AA_BB/rl/recommendation")
        self.assertEqual(qos, 1)
        self.assertTrue(retain)
        decoded = json.loads(payload)
        self.assertEqual(decoded["shot_id"], "shot_1")
        self.assertEqual(decoded["recommendation_id"], "rec_1")

    def test_status_payload_is_retained_for_gaggimate_settings(self) -> None:
        client = GaggimateMQTTClient(
            config=Config(mqtt_host="localhost", data_dir=Path("/tmp")),
            on_shot=lambda event: None,
            on_feedback=lambda event: None,
            on_correction=lambda event: None,
            on_upload_maintenance=lambda event: None,
            on_decision=lambda event: None,
            on_apply=lambda event: None,
            on_machine_state=lambda event: None,
        )
        fake = FakeMQTT()
        client._client = fake  # type: ignore[assignment]

        client.publish_status(
            "gaggimate:AA_BB",
            {
                "addon_online": True,
                "install_id": "install_1",
                "timestamp": 10,
                "last_shot_id": "shot_1",
                "last_shot_at": 9,
                "last_recommendation_id": "rec_1",
                "last_recommendation_at": 10,
                "recommendation_apply_status": "partially_applied",
                "mode": "zero_immediate_bo",
                "local_shot_count": 1,
                "upload_queue_count": 0,
                "community_upload_enabled": False,
            },
        )

        topic, payload, qos, retain = fake.published[0]
        self.assertEqual(topic, "gaggimate/AA_BB/rl/status")
        self.assertEqual(qos, 1)
        self.assertTrue(retain)
        decoded = json.loads(payload)
        self.assertEqual(decoded["event_type"], "espresso_rl_status")
        self.assertTrue(decoded["addon_online"])
        self.assertEqual(decoded["machine_id"], "gaggimate:AA_BB")
        self.assertEqual(decoded["last_recommendation_id"], "rec_1")
        self.assertEqual(decoded["recommendation_apply_status"], "partially_applied")

    def test_apply_payload_records_machine_apply_separately_from_follow_through(self) -> None:
        client = GaggimateMQTTClient(
            config=Config(mqtt_host="localhost", data_dir=Path("/tmp")),
            on_shot=lambda event: None,
            on_feedback=lambda event: None,
            on_correction=lambda event: None,
            on_upload_maintenance=lambda event: None,
            on_decision=lambda event: None,
            on_apply=lambda event: None,
            on_machine_state=lambda event: None,
        )

        event = client.translate_apply_payload(
            {
                "event_type": "recommendation_apply",
                "schema_version": 1,
                "recommendation_id": "rec_1",
                "status": "partially_applied",
                "timestamp": 12,
                "applied_fields": {"target_yield_g": 40.0},
                "manual_fields": ["next_grind_steps", "next_dose_g"],
                "message": "Target yield applied; grind and dose are manual.",
                "source": "gaggimate_lvgl",
            },
            mac="AA_BB",
        )

        self.assertEqual(event.recommendation_id, "rec_1")
        self.assertEqual(event.status, RecommendationApplyStatus.PARTIALLY_APPLIED)
        self.assertEqual(event.machine_id, "gaggimate:AA_BB")
        self.assertEqual(event.applied_fields["target_yield_g"], 40.0)
        self.assertEqual(event.manual_fields, ["next_grind_steps", "next_dose_g"])

    def test_correction_payload_records_manual_exclusion_and_follow_through(self) -> None:
        client = GaggimateMQTTClient(
            config=Config(mqtt_host="localhost", data_dir=Path("/tmp"), install_id="install_1"),
            on_shot=lambda event: None,
            on_feedback=lambda event: None,
            on_correction=lambda event: None,
            on_upload_maintenance=lambda event: None,
            on_decision=lambda event: None,
            on_apply=lambda event: None,
            on_machine_state=lambda event: None,
        )

        event = client.translate_correction_payload(
            {
                "event_type": "shot_correction",
                "schema_version": 1,
                "shot_id": "shot_1",
                "machine_id": "gaggimate:AA_BB",
                "timestamp": 12,
                "exclude_from_local_optimization": True,
                "grind_followed": False,
                "dose_followed": True,
                "yield_followed": True,
                "correction_tags": ["did_not_follow_grind", "changed_manually"],
                "source": "gaggimate_webui",
            },
            mac="AA_BB",
        )

        self.assertEqual(event.shot_id, "shot_1")
        self.assertEqual(event.install_id, "install_1")
        self.assertEqual(event.machine_id, "gaggimate:AA_BB")
        self.assertTrue(event.exclude_from_local_optimization)
        self.assertFalse(event.grind_followed)
        self.assertTrue(event.dose_followed)
        self.assertTrue(event.yield_followed)
        self.assertEqual(event.correction_tags, ["did_not_follow_grind", "changed_manually"])

    def test_upload_maintenance_payload_requests_safe_requeue(self) -> None:
        client = GaggimateMQTTClient(
            config=Config(mqtt_host="localhost", data_dir=Path("/tmp"), install_id="install_1"),
            on_shot=lambda event: None,
            on_feedback=lambda event: None,
            on_correction=lambda event: None,
            on_upload_maintenance=lambda event: None,
            on_decision=lambda event: None,
            on_apply=lambda event: None,
            on_machine_state=lambda event: None,
        )

        event = client.translate_upload_maintenance_payload(
            {
                "event_type": "upload_queue_maintenance",
                "schema_version": 1,
                "machine_id": "gaggimate:AA_BB",
                "bean_context_id": "bean_1",
                "timestamp": 12,
                "action": "requeue_valid_rejected",
                "limit": 50,
                "source": "gaggimate_webui",
            },
            mac="AA_BB",
        )

        self.assertEqual(event.install_id, "install_1")
        self.assertEqual(event.machine_id, "gaggimate:AA_BB")
        self.assertEqual(event.bean_context_id, "bean_1")
        self.assertEqual(event.action, "requeue_valid_rejected")
        self.assertEqual(event.limit, 50)

    def test_upload_maintenance_payload_can_request_safe_purge(self) -> None:
        client = GaggimateMQTTClient(
            config=Config(mqtt_host="localhost", data_dir=Path("/tmp"), install_id="install_1"),
            on_shot=lambda event: None,
            on_feedback=lambda event: None,
            on_correction=lambda event: None,
            on_upload_maintenance=lambda event: None,
            on_decision=lambda event: None,
            on_apply=lambda event: None,
            on_machine_state=lambda event: None,
        )

        event = client.translate_upload_maintenance_payload(
            {
                "event_type": "upload_queue_maintenance",
                "schema_version": 1,
                "machine_id": "gaggimate:AA_BB",
                "timestamp": 12,
                "action": "purge_rejected",
                "limit": 50,
                "local_record_id": "shot_1",
                "source": "gaggimate_webui",
            },
            mac="AA_BB",
        )

        self.assertEqual(event.action, "purge_rejected")
        self.assertEqual(event.limit, 50)
        self.assertEqual(event.local_record_id, "shot_1")

    def test_current_gaggimate_payload_drives_local_bo_data_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(
                mqtt_host="localhost",
                data_dir=Path(tmp),
                install_id="install_1",
                grinder_step_size_um=12.5,
                initial_grind_steps=42,
                initial_dose_g=18.0,
                initial_target_yield_g=36.0,
            )
            store = SQLiteStore(Path(tmp) / "espresso.db")
            service = EspressoRLService(
                SQLiteShotRepository(store),
                SQLiteRecommendationRepository(store),
                ConservativeBOOptimizer(),
                clock=lambda: 100,
            )
            client = GaggimateMQTTClient(
                config=config,
                on_shot=lambda event: None,
                on_feedback=lambda event: None,
                on_correction=lambda event: None,
                on_upload_maintenance=lambda event: None,
                on_decision=lambda event: None,
                on_apply=lambda event: None,
                on_machine_state=lambda event: None,
            )
            fake = FakeMQTT()
            client._client = fake  # type: ignore[assignment]

            first_payload = {
                "event_type": "shot_profile",
                "schema_version": 1,
                "shot_id": "shot_1",
                "machine_id": "gaggimate:AA_BB",
                "machine_adapter": "gaggimate",
                "timestamp": 1,
                "n_samples": 4,
                "time_ms": [0, 250, 500, 750],
                "pressure": [0.0, 4.0, 8.5, 9.0],
                "target_pressure": [0.0, 4.0, 8.5, 9.0],
                "flow": [0.0, 1.0, 2.0, 2.1],
                "target_flow": [0.0, 1.0, 2.0, 2.0],
                "weight": [0.0, 8.0, 22.0, 36.0],
                "grinder_step_size_um": 12.5,
                "grind_steps": 42,
                "dose_in_g": 18.0,
                "target_yield_g": 36.0,
                "target_ratio": 2.0,
                "beverage_out_g": 36.0,
                "shot_time_s": 29.0,
                "bean_context_id": "profile_1",
            }

            first = service.ingest_shot_profile(client.translate_shot_payload(first_payload, mac="AA_BB"))
            client.publish_recommendation(first.recommendation)

            topic, payload, qos, retain = fake.published[-1]
            recommendation_payload = json.loads(payload)
            self.assertEqual(topic, "gaggimate/AA_BB/rl/recommendation")
            self.assertEqual(qos, 1)
            self.assertTrue(retain)
            self.assertEqual(recommendation_payload["recommendation_id"], first.recommendation.recommendation_id)
            self.assertIn("next_dose_g", recommendation_payload)
            self.assertIn("target_yield_g", recommendation_payload)

            accepted = service.record_recommendation_decision(
                client.translate_decision_payload(
                    {
                        "event_type": "recommendation_decision",
                        "schema_version": 1,
                        "recommendation_id": first.recommendation.recommendation_id,
                        "decision": "accepted",
                        "edited_fields": {
                            "next_dose_g": recommendation_payload["next_dose_g"],
                            "target_yield_g": recommendation_payload["target_yield_g"],
                        },
                        "timestamp": 2,
                    },
                    mac="AA_BB",
                )
            )
            self.assertEqual(accepted.status, RecommendationStatus.ACCEPTED)

            second_payload = {
                **first_payload,
                "shot_id": "shot_2",
                "timestamp": 3,
                "recommendation_id": first.recommendation.recommendation_id,
                "grind_steps": first.recommendation.next_grind_steps,
                "dose_in_g": first.recommendation.next_dose_g,
                "target_yield_g": first.recommendation.target_yield_g,
                "target_ratio": first.recommendation.target_ratio,
                "beverage_out_g": first.recommendation.target_yield_g,
                "weight": [0.0, 9.0, 24.0, first.recommendation.target_yield_g],
            }
            second = service.ingest_shot_profile(client.translate_shot_payload(second_payload, mac="AA_BB"))
            self.assertEqual(second.shot.recommendation_followed, FollowThroughState.FOLLOWED)

            updated = service.record_feedback(
                client.translate_feedback_payload(
                    {
                        "shot_id": "shot_2",
                        "rating": 4,
                        "timestamp": 4,
                        "source": "gaggimate_webui",
                    },
                    mac="AA_BB",
                )
            )
            self.assertEqual(updated.human_rating, 4)
            self.assertGreater(updated.reward_confidence, 0.5)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest
from pathlib import Path

from espresso_rl.adapters.gaggimate_mqtt import (
    APPLY_TOPIC,
    CORRECTION_TOPIC,
    DECISION_TOPIC,
    LOCAL_RESET_TOPIC,
    MACHINE_STATE_TOPIC,
    OPTIMIZER_SETTINGS_TOPIC,
    PREFERENCE_TOPIC,
    SHOT_TOPIC,
    UPLOAD_REQUEUE_TOPIC,
    GaggimateMQTTClient,
)
from espresso_rl.config import Config
from espresso_rl.domain.cpbo import ComparisonMode, PreferenceLabel
from espresso_rl.domain.models import (
    Recommendation,
    RecommendationMode,
    RecommendationStatus,
)


FIXTURE = Path(__file__).parent / "fixtures" / "gaggimate_shot_profile.json"


class FakeMQTT:
    def __init__(self) -> None:
        self.published: list[tuple[str, str, int, bool]] = []
        self.subscriptions: list[str] = []

    def publish(self, topic: str, payload: str, qos: int, retain: bool) -> None:
        self.published.append((topic, payload, qos, retain))

    def subscribe(self, topic: str) -> None:
        self.subscriptions.append(topic)


class GaggimateAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.preferences = []
        self.client = GaggimateMQTTClient(
            config=Config(mqtt_host="localhost", data_dir=Path(".")),
            on_shot=lambda event: None,
            on_correction=lambda event: None,
            on_upload_maintenance=lambda event: None,
            on_decision=lambda event: None,
            on_apply=lambda event: None,
            on_machine_state=lambda event: None,
            on_preference=self.preferences.append,
        )

    def test_preference_payload_is_strict_and_oriented(self) -> None:
        event = self.client.translate_preference_payload(
            {
                "event_type": "preference_feedback",
                "schema_version": 1,
                "optimization_run_id": "run_1",
                "new_shot_id": "shot_new",
                "anchor_shot_id": "shot_anchor",
                "label": "new_better",
                "comparison_mode": "best_incumbent",
                "taste_goal": {"schema_version": 1, "mode": "balanced", "targets": {}},
                "install_id": "install_1",
                "machine_id": "gaggimate:AA_BB",
                "timestamp": 100,
                "source": "webui",
            },
            "AA_BB",
        )
        self.assertEqual(event.label, PreferenceLabel.NEW_BETTER)
        self.assertEqual(event.comparison_mode, ComparisonMode.BEST_INCUMBENT)
        with self.assertRaises(ValueError):
            self.client.translate_preference_payload(
                {
                    "event_type": "preference_feedback",
                    "schema_version": 1,
                    "optimization_run_id": "run_1",
                    "new_shot_id": "shot_new",
                    "anchor_shot_id": "shot_anchor",
                    "label": 5,
                    "install_id": "install_1",
                    "machine_id": "gaggimate:AA_BB",
                    "timestamp": 100,
                    "source": "webui",
                },
                "AA_BB",
            )

    def test_shot_payload_preserves_observed_recipe_and_telemetry(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        event = self.client.translate_shot_payload(payload, "AA_BB")
        self.assertEqual(event.relative_grind_steps_from_reference, 2.0)
        self.assertEqual(event.dose_in_g, 18.0)
        self.assertEqual(event.beverage_out_g, 38.0)
        self.assertEqual(event.profile_id, payload["profile_id"])
        self.assertEqual(len(event.temperature or []), len(event.time_ms))

    def test_shot_payload_keeps_commanded_dose_when_physical_dose_is_unmeasured(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload.pop("dose_in_g")
        payload["dose_observed"] = False
        payload["dose_target_g"] = 18.2

        event = self.client.translate_shot_payload(payload, "AA_BB")

        self.assertFalse(event.dose_observed)
        self.assertEqual(event.dose_target_g, 18.2)
        self.assertEqual(event.dose_in_g, 18.2)

    def test_shot_payload_rejects_an_invalid_declared_dose_target(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload.pop("dose_in_g")
        payload["dose_observed"] = False
        payload["dose_target_g"] = -1

        with self.assertRaisesRegex(ValueError, "dose_target_g"):
            self.client.translate_shot_payload(payload, "AA_BB")

    def test_optimizer_settings_accepts_cpbo_and_rejects_removed_dreamer(self) -> None:
        event = self.client.translate_optimizer_settings_payload(
            {
                "event_type": "optimizer_settings",
                "schema_version": 1,
                "optimizer_mode": "cpbo",
                "install_id": "install_1",
                "machine_id": "gaggimate:AA_BB",
                "timestamp": 100,
                "profile_id": "profile_1",
                "profile_label": "Profile One",
                "source": "webui",
            },
            "AA_BB",
        )
        self.assertEqual(event.optimizer_mode, "cpbo")
        self.assertEqual(event.profile_id, "profile_1")
        self.assertEqual(event.profile_label, "Profile One")
        with self.assertRaises(ValueError):
            self.client.translate_optimizer_settings_payload(
                {
                    "event_type": "optimizer_settings",
                    "schema_version": 1,
                    "optimizer_mode": "dreamer_v3",
                    "install_id": "install_1",
                    "machine_id": "gaggimate:AA_BB",
                    "timestamp": 100,
                    "source": "webui",
                },
                "AA_BB",
            )

    def test_connect_subscribes_to_preference_contract_without_rating_or_model_topics(self) -> None:
        mqtt = FakeMQTT()
        self.client._on_connect(mqtt, None, None, 0, None)  # type: ignore[arg-type]
        self.assertEqual(
            set(mqtt.subscriptions),
            {
                SHOT_TOPIC,
                PREFERENCE_TOPIC,
                CORRECTION_TOPIC,
                UPLOAD_REQUEUE_TOPIC,
                DECISION_TOPIC,
                APPLY_TOPIC,
                MACHINE_STATE_TOPIC,
                OPTIMIZER_SETTINGS_TOPIC,
                LOCAL_RESET_TOPIC,
            },
        )
        self.assertFalse(any("rating" in topic or "dreamer" in topic for topic in mqtt.subscriptions))

    def test_recommendation_publication_contains_comparison_identity(self) -> None:
        mqtt = FakeMQTT()
        self.client._client = mqtt  # type: ignore[assignment]
        recommendation = _recommendation()
        self.client.publish_recommendation(recommendation)
        topic, raw, qos, retain = mqtt.published[-1]
        payload = json.loads(raw)
        self.assertEqual(topic, "gaggimate/AA_BB/rl/recommendation")
        self.assertEqual(payload["optimization_run_id"], "run_1")
        self.assertEqual(payload["comparison_anchor_shot_id"], "shot_anchor")
        self.assertTrue(payload["preference_feedback_required"])
        self.assertNotIn("human_rating", payload)
        self.assertEqual((qos, retain), (1, True))

    def test_status_and_clear_are_retained(self) -> None:
        mqtt = FakeMQTT()
        self.client._client = mqtt  # type: ignore[assignment]
        self.client.publish_status("gaggimate:AA_BB", {"optimizer_mode": "cpbo_best_incumbent"})
        self.client.clear_recommendation("gaggimate:AA_BB")
        self.assertTrue(all(item[3] for item in mqtt.published))


def _recommendation() -> Recommendation:
    return Recommendation(
        recommendation_id="rec_1",
        created_at=100,
        updated_at=100,
        expires_at=200,
        install_id="install_1",
        machine_id="gaggimate:AA_BB",
        bean_context_id="bean_1",
        grinder_context_id="grinder_1",
        grind_delta_steps_from_current=1.0,
        grind_delta_um_from_current=12.5,
        projected_relative_step_from_reference=3.0,
        projected_relative_grind_um_from_reference=37.5,
        next_dose_g=18.0,
        target_yield_g=36.0,
        target_ratio=2.0,
        mode=RecommendationMode.CPBO_BEST_INCUMBENT,
        confidence=0.5,
        reason="CPBO-MES suggestion",
        status=RecommendationStatus.PENDING,
        source_shot_id="shot_new",
        optimization_run_id="run_1",
        comparison_anchor_shot_id="shot_anchor",
        comparison_mode="best_incumbent",
        preference_feedback_required=True,
    )


if __name__ == "__main__":
    unittest.main()

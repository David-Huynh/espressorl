from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from espresso_rl.adapters.gaggimate_mqtt import (
    APPLY_TOPIC,
    CORRECTION_TOPIC,
    DECISION_TOPIC,
    LOCAL_RESET_TOPIC,
    LIVE_SHOT_TOPIC,
    MACHINE_STATE_TOPIC,
    OPTIMIZER_CONTROL_TOPIC,
    OPTIMIZER_SETTINGS_TOPIC,
    PREFERENCE_TOPIC,
    SHOT_TOPIC,
    SHOT_ACK_EVENT_TYPE,
    SHOT_ACK_TOPIC_SUFFIX,
    STATUS_PAYLOAD_MAX_BYTES,
    STATUS_RECENT_SHOTS_MAX_BYTES,
    UPLOAD_REQUEUE_TOPIC,
    GaggimateMQTTClient,
)
from espresso_rl.config import Config
from espresso_rl.domain.cpbo import (
    CPBOProfile,
    ComparisonMode,
    OptimizerControlAction,
    PendingPreferenceRequest,
    PreferenceLabel,
)
from espresso_rl.domain.models import (
    Recommendation,
    RecommendationMode,
    RecommendationStatus,
)
from espresso_rl.domain.taste_goal import TasteGoal
from espresso_rl.main import maybe_publish_startup_recommendation


FIXTURE = Path(__file__).parent / "fixtures" / "gaggimate_shot_profile.json"


def _live_sample_payload(sequence: int, elapsed_ms: int) -> dict[str, object]:
    return {
        "event_type": "live_shot_sample",
        "schema_version": 1,
        "shot_id": "shot_live_1",
        "machine_id": "gaggimate:AA_BB",
        "timestamp_ms": 1_800_000_000_000 + elapsed_ms,
        "sequence": sequence,
        "elapsed_ms": elapsed_ms,
        "sample": {
            "pressure_bar": 9.0,
            "pressure_target_bar": 9.0,
            "pump_flow_ml_s": 4.0,
            "pump_flow_target_ml_s": 4.0,
            "beverage_flow_g_s": 2.0,
            "weight_g": 5.0,
            "temperature_c": 93.0,
            "temperature_target_c": 93.0,
            "pump_target_mode": 1,
            "valve_open": True,
        },
    }


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

    def test_correction_payload_validates_numeric_fields_and_topic_owner(self) -> None:
        payload = {
            "event_type": "shot_correction",
            "schema_version": 1,
            "shot_id": "shot_1",
            "machine_id": "gaggimate:AA_BB",
            "timestamp": 100,
            "source": "gaggimate_shot_history",
            "correction_tags": [],
            "dose_in_g": 17.5,
            "target_yield_g": 42.0,
            "beverage_out_g": 41.5,
            "relative_grind_steps_from_reference": 3.0,
        }

        event = self.client.translate_correction_payload(payload, "AA_BB")

        self.assertEqual(event.dose_in_g, 17.5)
        self.assertEqual(event.relative_grind_steps_from_reference, 3.0)
        with self.assertRaises(ValueError):
            self.client.translate_correction_payload({**payload, "dose_in_g": True}, "AA_BB")
        with self.assertRaises(ValueError):
            self.client.translate_correction_payload({**payload, "unexpected": 1}, "AA_BB")
        with self.assertRaises(ValueError):
            self.client.translate_correction_payload(
                {**payload, "machine_id": "gaggimate:CC_DD"},
                "AA_BB",
            )

    def test_machine_state_preserves_taste_goal(self) -> None:
        event = self.client.translate_machine_state_payload(
            {
                "event_type": "machine_state",
                "schema_version": 1,
                "machine_id": "gaggimate:AA_BB",
                "timestamp": 100,
                "state": "idle",
                "taste_goal": {
                    "schema_version": 1,
                    "mode": "custom",
                    "targets": {"sweet": "high", "nutty_cocoa": "high", "roasted": "medium"},
                },
            },
            "AA_BB",
        )

        self.assertEqual(event.taste_goal.mode.value, "custom")
        self.assertEqual(dict(event.taste_goal.targets)["sweet"].value, "high")

    def test_startup_without_scope_does_not_clear_retained_recommendation(self) -> None:
        mqtt = SimpleNamespace(
            published=[],
            cleared=[],
            publish_recommendation=lambda recommendation: mqtt.published.append(recommendation),
            clear_recommendation=lambda machine_id: mqtt.cleared.append(machine_id),
            publish_status=lambda machine_id, status: mqtt.published.append((machine_id, status)),
        )
        config = SimpleNamespace(
            machine_id="gaggimate:AA_BB",
            install_id="install_1",
            bean_context_id=None,
            grinder_context_id=None,
        )

        with patch("espresso_rl.main.build_status_payload", return_value={}):
            maybe_publish_startup_recommendation(config, SimpleNamespace(), mqtt)

        self.assertEqual(mqtt.cleared, [])

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

    def test_shot_payload_rejects_machine_identity_mismatch(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["machine_id"] = "gaggimate:DIFFERENT"

        with self.assertRaisesRegex(ValueError, "machine_id does not match topic"):
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
                "cpbo_profile_name": "paper_fidelity",
                "cpbo_comparison_mode": "global_previous",
                "profile_id": "profile_1",
                "profile_label": "Profile One",
                "source": "webui",
            },
            "AA_BB",
        )
        self.assertEqual(event.optimizer_mode, "cpbo")
        self.assertEqual(event.cpbo_profile_name, CPBOProfile.PAPER_FIDELITY)
        self.assertEqual(event.cpbo_comparison_mode, ComparisonMode.GLOBAL_PREVIOUS)
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
        for field, invalid in (
            ("cpbo_profile_name", "fastest"),
            ("cpbo_comparison_mode", "random_anchor"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.client.translate_optimizer_settings_payload(
                    {
                        "event_type": "optimizer_settings",
                        "schema_version": 1,
                        "optimizer_mode": "cpbo",
                        field: invalid,
                        "install_id": "install_1",
                        "machine_id": "gaggimate:AA_BB",
                        "timestamp": 100,
                    },
                    "AA_BB",
                )

    def test_optimizer_control_is_strict_and_canonical(self) -> None:
        payload = {
            "event_type": "optimizer_control",
            "schema_version": 1,
            "request_id": "resume_123",
            "optimization_run_id": "run_1",
            "action": "resume_local_exploration",
            "machine_id": "gaggimate:AA_BB",
            "timestamp": 123,
            "source": "gaggimate_mqtt",
        }
        event = self.client.translate_optimizer_control_payload(payload, "AA_BB")
        self.assertEqual(
            event.action,
            OptimizerControlAction.RESUME_LOCAL_EXPLORATION,
        )
        self.assertEqual(event.optimization_run_id, "run_1")
        with self.assertRaises(ValueError):
            self.client.translate_optimizer_control_payload(
                {**payload, "unexpected": True},
                "AA_BB",
            )
        with self.assertRaises(ValueError):
            self.client.translate_optimizer_control_payload(
                {**payload, "action": "restart_global"},
                "AA_BB",
            )
        with self.assertRaisesRegex(ValueError, "does not match topic"):
            self.client.translate_optimizer_control_payload(
                {**payload, "machine_id": "gaggimate:CC_DD"},
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
                OPTIMIZER_CONTROL_TOPIC,
                LOCAL_RESET_TOPIC,
                LIVE_SHOT_TOPIC,
            },
        )
        self.assertFalse(any("rating" in topic or "dreamer" in topic for topic in mqtt.subscriptions))

    def test_live_shot_payload_translates_to_canonical_events(self) -> None:
        started = self.client.translate_live_shot_payload(
            {
                "event_type": "live_shot_started",
                "schema_version": 1,
                "shot_id": "shot_live_1",
                "machine_id": "gaggimate:AA_BB",
                "timestamp_ms": 1_800_000_000_000,
                "sample_interval_ms": 250,
                "weight_source": "hardware_scale",
                "flow_source": "hardware_scale",
            },
            "AA_BB",
        )
        self.assertEqual(started.shot_id, "shot_live_1")
        sample = self.client.translate_live_shot_payload(
            _live_sample_payload(sequence=0, elapsed_ms=250),
            "AA_BB",
        )
        self.assertEqual(sample.temperature_c, 93.0)
        self.assertEqual(sample.pump_target_mode, 1)

    def test_live_shot_payload_rejects_untrusted_identity_and_channels(self) -> None:
        wrong_machine = _live_sample_payload(sequence=0, elapsed_ms=250)
        wrong_machine["machine_id"] = "gaggimate:OTHER"
        with self.assertRaisesRegex(ValueError, "machine_id"):
            self.client.translate_live_shot_payload(wrong_machine, "AA_BB")

        invalid_pressure = _live_sample_payload(sequence=0, elapsed_ms=250)
        invalid_pressure["sample"]["pressure_bar"] = 100.0
        with self.assertRaisesRegex(ValueError, "pressure_bar"):
            self.client.translate_live_shot_payload(invalid_pressure, "AA_BB")

        numeric_shot_id = _live_sample_payload(sequence=0, elapsed_ms=250)
        numeric_shot_id["shot_id"] = 123
        with self.assertRaisesRegex(ValueError, "shot_id"):
            self.client.translate_live_shot_payload(numeric_shot_id, "AA_BB")

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

    def test_status_bounds_recent_history_without_losing_newest_shots(self) -> None:
        mqtt = FakeMQTT()
        self.client._client = mqtt  # type: ignore[assignment]
        recent_shots = [
            {"shot_id": f"shot_{index}", "timestamp": 100 - index, "detail": "x" * 900}
            for index in range(20)
        ]

        self.client.publish_status(
            "gaggimate:AA_BB",
            {
                "event_type": "untrusted_override",
                "machine_id": "untrusted",
                "timestamp": 100,
                "recent_shots": recent_shots,
            },
        )

        raw = mqtt.published[-1][1]
        payload = json.loads(raw)
        self.assertLessEqual(len(raw.encode("utf-8")), STATUS_PAYLOAD_MAX_BYTES)
        self.assertLessEqual(
            len(json.dumps(payload["recent_shots"], separators=(",", ":")).encode("utf-8")),
            STATUS_RECENT_SHOTS_MAX_BYTES,
        )
        self.assertEqual(payload["event_type"], "espresso_rl_status")
        self.assertEqual(payload["machine_id"], "gaggimate:AA_BB")
        self.assertEqual(payload["recent_shots"][0]["shot_id"], "shot_0")
        self.assertLess(len(payload["recent_shots"]), len(recent_shots))

    def test_status_uses_bounded_fallback_for_invalid_or_unbounded_details(self) -> None:
        mqtt = FakeMQTT()
        self.client._client = mqtt  # type: ignore[assignment]

        self.client.publish_status(
            "gaggimate:AA_BB",
            {
                "timestamp": 100,
                "runtime_health_summary": "x" * (STATUS_PAYLOAD_MAX_BYTES * 2),
                "invalid_number": float("nan"),
                "recent_shots": [{"shot_id": "shot_1"}],
            },
        )

        raw = mqtt.published[-1][1]
        payload = json.loads(raw)
        self.assertLessEqual(len(raw.encode("utf-8")), STATUS_PAYLOAD_MAX_BYTES)
        self.assertEqual(payload["runtime_health_status"], "attention")
        self.assertEqual(payload["recent_shots"], [])

    def test_shot_delivery_acknowledges_acceptance_and_duplicate_replay(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        mqtt = FakeMQTT()
        outcomes = iter(
            (
                SimpleNamespace(shot=object(), replayed=False, dropped_reason=None),
                SimpleNamespace(shot=object(), replayed=True, dropped_reason=None),
            )
        )
        self.client._client = mqtt  # type: ignore[assignment]
        self.client._on_shot = lambda event: next(outcomes)

        self.client._handle_shot_message(payload, "AA_BB")
        self.client._handle_shot_message(payload, "AA_BB")

        first = json.loads(mqtt.published[-2][1])
        second = json.loads(mqtt.published[-1][1])
        self.assertEqual(mqtt.published[-1][0], f"gaggimate/AA_BB/{SHOT_ACK_TOPIC_SUFFIX}")
        self.assertEqual((mqtt.published[-1][2], mqtt.published[-1][3]), (1, False))
        self.assertEqual(first["event_type"], SHOT_ACK_EVENT_TYPE)
        self.assertEqual(first["schema_version"], 3)
        self.assertEqual(first["record_revision"], 1)
        self.assertEqual(first["outcome"], "accepted")
        self.assertFalse(first["retryable"])
        self.assertEqual(second["outcome"], "already_processed")
        self.assertFalse(second["retryable"])

    def test_shot_delivery_rejects_invalid_delivery_context_without_ingesting(self) -> None:
        invalid_deliveries = (
            {"record_revision": 0, "reprocess": False},
            {"record_revision": True, "reprocess": False},
            {"record_revision": 1, "reprocess": 1},
            {"record_revision": 1},
            {"record_revision": 1, "reprocess": False, "extra": "field"},
        )
        for delivery in invalid_deliveries:
            with self.subTest(delivery=delivery):
                payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
                payload["delivery"] = delivery
                mqtt = FakeMQTT()
                ingested: list[object] = []
                self.client._client = mqtt  # type: ignore[assignment]
                self.client._on_shot = ingested.append

                self.client._handle_shot_message(payload, "AA_BB")

                self.assertEqual(ingested, [])
                self.assertEqual(mqtt.published, [])

    def test_shot_delivery_ack_includes_canonical_pending_preference(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        mqtt = FakeMQTT()
        install_id = self.client._config.install_id
        request = PendingPreferenceRequest(
            install_id=install_id,
            machine_id="gaggimate:AA_BB",
            optimization_run_id="run_1",
            new_shot_id=str(payload["shot_id"]),
            anchor_shot_id="shot_anchor",
            comparison_mode=ComparisonMode.BEST_INCUMBENT,
            taste_goal=TasteGoal.custom({"sweet": "high"}),
        )
        self.client._client = mqtt  # type: ignore[assignment]
        self.client._on_shot = lambda event: SimpleNamespace(
            shot=object(),
            replayed=True,
            dropped_reason=None,
            preference_request=request,
        )

        self.client._handle_shot_message(payload, "AA_BB")

        ack = json.loads(mqtt.published[-1][1])
        self.assertEqual(ack["outcome"], "already_processed")
        self.assertEqual(
            ack["preference_request"],
            {
                "install_id": install_id,
                "optimization_run_id": "run_1",
                "new_shot_id": payload["shot_id"],
                "anchor_shot_id": "shot_anchor",
                "comparison_mode": "best_incumbent",
                "taste_goal": {
                    "schema_version": 1,
                    "mode": "custom",
                    "targets": {"sweet": "high"},
                },
                "recommendation_id": None,
            },
        )

    def test_shot_delivery_classifies_permanent_and_transient_failures(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        mqtt = FakeMQTT()
        self.client._client = mqtt  # type: ignore[assignment]

        self.client._on_shot = lambda event: (_ for _ in ()).throw(ValueError("bad shot"))
        self.client._handle_shot_message(payload, "AA_BB")
        permanent = json.loads(mqtt.published[-1][1])
        self.assertEqual(permanent["outcome"], "permanent_rejection")
        self.assertEqual(permanent["reason"], "invalid_shot")
        self.assertFalse(permanent["retryable"])

        self.client._on_shot = lambda event: (_ for _ in ()).throw(RuntimeError("database offline"))
        self.client._handle_shot_message(payload, "AA_BB")
        transient = json.loads(mqtt.published[-1][1])
        self.assertEqual(transient["outcome"], "transient_failure")
        self.assertEqual(transient["reason"], "ingest_unavailable")
        self.assertTrue(transient["retryable"])
        self.assertNotIn("database offline", mqtt.published[-1][1])

        self.client._on_shot = lambda event: (_ for _ in ()).throw(TypeError("runtime bug"))
        self.client._handle_shot_message(payload, "AA_BB")
        programming_failure = json.loads(mqtt.published[-1][1])
        self.assertEqual(programming_failure["outcome"], "transient_failure")

    def test_shot_delivery_dropped_by_application_is_terminal(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        mqtt = FakeMQTT()
        self.client._client = mqtt  # type: ignore[assignment]
        self.client._on_shot = lambda event: SimpleNamespace(
            shot=None,
            replayed=False,
            dropped_reason="local_optimization_disabled",
        )

        self.client._handle_shot_message(payload, "AA_BB")

        ack = json.loads(mqtt.published[-1][1])
        self.assertEqual(ack["outcome"], "permanent_rejection")
        self.assertEqual(ack["reason"], "local_optimization_disabled")


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

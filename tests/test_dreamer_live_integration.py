from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from espresso_rl.adapters.gaggimate_mqtt import GaggimateMQTTClient
from espresso_rl.adapters.sqlite_repositories import SQLiteRecommendationRepository, SQLiteShotRepository, SQLiteStore
from espresso_rl.application.dreamer_live_acknowledgements import DreamerLiveControlAcknowledgementService
from espresso_rl.application.dreamer_live_control import DreamerLiveControlApplication
from espresso_rl.application.dreamer_live_telemetry import DreamerLiveTelemetryApplication
from espresso_rl.application.services import EspressoRLService
from espresso_rl.config import Config
from espresso_rl.domain.dreamer_control import DreamerControlSpec
from espresso_rl.domain.dreamer_telemetry import DreamerLiveEpisodeContext
from espresso_rl.domain.optimization import DEFAULT_OPTIMIZER_MODE, OPTIMIZER_MODE_DREAMER_V3_ACTIVE
from espresso_rl.main import build_status_payload
from espresso_rl.optimizers.conservative_bo import ConservativeBOOptimizer
from espresso_rl.optimizers.runtime import RuntimeOptimizer
from tests.test_dreamer_live_inference import _inference, _select_live_actions


class RecordingPublisher:
    def __init__(self) -> None:
        self.live_control = []

    def publish_recommendation(self, recommendation) -> None:
        raise AssertionError("recommendation publishing is outside this test")

    def publish_dreamer_live_control(self, publication) -> None:
        self.live_control.append(publication)

    def publish_status(self, *args, **kwargs) -> None:
        raise AssertionError("status publishing is outside this test")


class MQTTBackedRuntimePublisher:
    def __init__(
        self,
        *,
        client: GaggimateMQTTClient,
        acknowledgements: DreamerLiveControlAcknowledgementService,
    ) -> None:
        self.client = client
        self.acknowledgements = acknowledgements
        self.publications = []

    def publish_recommendation(self, recommendation) -> None:
        raise AssertionError("recommendation publishing is outside this test")

    def publish_dreamer_live_control(self, publication) -> None:
        self.publications.append(publication)
        self.acknowledgements.record_publication(publication)
        self.client.publish_dreamer_live_control(publication)

    def publish_status(self, *args, **kwargs) -> None:
        raise AssertionError("status publishing is outside this test")


class FakeMQTT:
    def __init__(self) -> None:
        self.published = []

    def publish(self, topic, payload=None, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))


class RecordingInference:
    def __init__(self) -> None:
        self.control_spec = DreamerControlSpec(
            dynamic_control_enabled=True,
            pressure_control_allowed=True,
        )
        self.started = []
        self.observed = []
        self.ended = []

    def start_episode(self, telemetry, context) -> None:
        self.started.append((telemetry.episode_key, context.profile_id))

    def infer_action(self, telemetry):
        self.observed.append(telemetry.step_index)
        if self.control_spec.is_decision_step(telemetry.step_index):
            return {"pump_target_mode": 1, "pressure_target_bar": 8.0}
        return None

    def end_episode(self, *, machine_id: str, shot_id: str) -> None:
        self.ended.append((machine_id, shot_id))


class StaticContextProvider:
    def context_for(self, telemetry) -> DreamerLiveEpisodeContext:
        return DreamerLiveEpisodeContext(
            install_id="install_1",
            machine_id=telemetry.machine_id,
            timestamp=10,
            bean_context_id="bean_1",
            bean_context_name="Test bean",
            grinder_context_id="grinder_1",
            relative_grind_steps_from_reference=0.0,
            relative_grind_um_from_reference=0.0,
            grind_observed=True,
            dose_g=18.0,
            dose_observed=True,
            initial_target_yield_g=36.0,
            initial_target_yield_observed=True,
            microns_per_step=12.5,
            step_direction="higher_is_finer",
            profile_id=telemetry.profile_id,
            profile_type="pro",
            profile_phase_count=1,
            taste_objective={"mode": "auto"},
        )


class DreamerLiveIntegrationTests(unittest.TestCase):
    def test_mqtt_closed_loop_publishes_live_targets_and_accepts_firmware_acks(self) -> None:
        fake_mqtt = FakeMQTT()
        acknowledgements = DreamerLiveControlAcknowledgementService(clock_ms=lambda: 10_000)
        inference = RecordingInference()
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            client = GaggimateMQTTClient(
                config=Config(mqtt_host="localhost", data_dir=Path(tmp), install_id="install_1"),
                on_shot=lambda event: None,
                on_feedback=lambda event: None,
                on_correction=lambda event: None,
                on_upload_maintenance=lambda event: None,
                on_decision=lambda event: None,
                on_apply=lambda event: None,
                on_machine_state=lambda event: None,
                on_dreamer_live_ack=acknowledgements.record_acknowledgement,
            )
            client._client = fake_mqtt  # type: ignore[assignment]
            runtime_publisher = MQTTBackedRuntimePublisher(
                client=client,
                acknowledgements=acknowledgements,
            )
            app = DreamerLiveTelemetryApplication(
                inference=inference,
                live_control=DreamerLiveControlApplication(runtime_publisher),
                context_provider=StaticContextProvider(),
                enabled=True,
            )
            client._on_dreamer_live_telemetry = lambda event: results.append(app.handle_telemetry(event))

            _dispatch_telemetry(client, _telemetry_payload(step_index=0, profile_id="dreamer_auto"))
            _dispatch_firmware_ack(client, json.loads(fake_mqtt.published[-1][1]))
            _dispatch_telemetry(client, _telemetry_payload(step_index=1, profile_id="dreamer_auto"))
            _dispatch_telemetry(client, _telemetry_payload(step_index=4, profile_id="dreamer_auto"))
            _dispatch_firmware_ack(client, json.loads(fake_mqtt.published[-1][1]))

        self.assertEqual(
            [result.outcome for result in results],
            ["decision_published", "observation_accepted", "decision_published"],
        )
        self.assertEqual(inference.started, [("gaggimate:AA_BB|shot_1|dreamer_auto", "dreamer_auto")])
        self.assertEqual(inference.observed, [0, 1, 4])
        self.assertEqual([publication.sequence for publication in runtime_publisher.publications], [1, 2])
        self.assertEqual([publication.step_index for publication in runtime_publisher.publications], [0, 4])
        published_topics = [topic for topic, _payload, _qos, retain in fake_mqtt.published if not retain]
        self.assertEqual(
            published_topics,
            [
                "gaggimate/AA_BB/rl/dreamer/live_target",
                "gaggimate/AA_BB/rl/dreamer/live_target",
            ],
        )
        first_payload = json.loads(fake_mqtt.published[0][1])
        self.assertEqual(first_payload["event_type"], "dreamer_live_target_update")
        self.assertEqual(first_payload["publication_id"], "gaggimate:AA_BB:1")
        self.assertEqual(first_payload["ack_scope"], "esp32_received")
        self.assertEqual(first_payload["target_update"], {"pressure_target_bar": 8.0})
        self.assertFalse(first_payload["fail_safe_required"])
        summary = acknowledgements.status_summary("gaggimate:AA_BB")
        self.assertEqual(summary["health"], "healthy")
        self.assertEqual(summary["accepted_count"], 2)
        self.assertEqual(summary["pending_count"], 0)
        self.assertNotIn("publication_id", summary)

    def test_gaggimate_live_telemetry_reaches_control_only_for_automatic_profile(self) -> None:
        publisher = RecordingPublisher()
        inference = RecordingInference()
        app = DreamerLiveTelemetryApplication(
            inference=inference,
            live_control=DreamerLiveControlApplication(publisher),
            context_provider=StaticContextProvider(),
            enabled=True,
        )
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            client = GaggimateMQTTClient(
                config=Config(mqtt_host="localhost", data_dir=Path(tmp), install_id="install_1"),
                on_shot=lambda event: None,
                on_feedback=lambda event: None,
                on_correction=lambda event: None,
                on_upload_maintenance=lambda event: None,
                on_decision=lambda event: None,
                on_apply=lambda event: None,
                on_machine_state=lambda event: None,
                on_dreamer_live_telemetry=lambda event: results.append(app.handle_telemetry(event)),
            )

            _dispatch_telemetry(client, _telemetry_payload(step_index=0, profile_id="dreamer_auto"))
            _dispatch_telemetry(
                client,
                _telemetry_payload(step_index=0, shot_id="shot_static", profile_id="classic_9_bar"),
            )

        self.assertEqual([result.outcome for result in results], ["decision_published", "wrong_profile"])
        self.assertEqual(inference.started, [("gaggimate:AA_BB|shot_1|dreamer_auto", "dreamer_auto")])
        self.assertEqual(inference.observed, [0])
        self.assertEqual([publication.step_index for publication in publisher.live_control], [0])
        self.assertEqual(publisher.live_control[0].profile_id, "dreamer_auto")
        self.assertEqual(publisher.live_control[0].action, {"pump_target_mode": 1, "pressure_target_bar": 8.0})

    def test_checkpoint_backed_live_inference_publishes_through_gaggimate_adapter_path(self) -> None:
        publisher = RecordingPublisher()
        inference, actor = _inference()
        _select_live_actions(actor, pump_mode=1, pressure_delta=0.5, flow_delta=0.5)
        app = DreamerLiveTelemetryApplication(
            inference=inference,
            live_control=DreamerLiveControlApplication(publisher),
            context_provider=StaticContextProvider(),
            enabled=True,
        )
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            client = GaggimateMQTTClient(
                config=Config(mqtt_host="localhost", data_dir=Path(tmp), install_id="install_1"),
                on_shot=lambda event: None,
                on_feedback=lambda event: None,
                on_correction=lambda event: None,
                on_upload_maintenance=lambda event: None,
                on_decision=lambda event: None,
                on_apply=lambda event: None,
                on_machine_state=lambda event: None,
                on_dreamer_live_telemetry=lambda event: results.append(app.handle_telemetry(event)),
            )

            _dispatch_telemetry(client, _telemetry_payload(step_index=0, profile_id="dreamer_auto"))

        self.assertEqual([result.outcome for result in results], ["decision_published"])
        self.assertEqual(len(publisher.live_control), 1)
        action = publisher.live_control[0].action
        self.assertEqual(action["pump_target_mode"], 1)
        self.assertAlmostEqual(action["pressure_target_bar"], 3.5)
        self.assertNotIn("flow_target_ml_s", action)

    def test_active_dreamer_request_without_release_ready_model_reports_bo_fallback_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            runtime_optimizer = RuntimeOptimizer(optimizer_mode=OPTIMIZER_MODE_DREAMER_V3_ACTIVE)
            with SQLiteStore(data_dir / "espresso.db") as store:
                service = EspressoRLService(
                    SQLiteShotRepository(store),
                    SQLiteRecommendationRepository(store),
                    runtime_optimizer,
                    clock=lambda: 10,
                )
                status = build_status_payload(
                    config=Config(
                        mqtt_host="localhost",
                        data_dir=data_dir,
                        install_id="install_1",
                        optimizer_mode="dreamer_v3_active",
                    ),
                    service=service,
                    shot_repo=None,
                    upload_maintenance=None,
                    upload_queue_repo=None,
                    machine_id="gaggimate:AA_BB",
                    bean_context_id="bean_1",
                    grinder_context_id="grinder_1",
                    profile_id="dreamer_auto",
                    profile_label="DreamerV3 Auto Control",
                    optimizer_status=runtime_optimizer.status().to_dict(),
                )

        self.assertEqual(status["optimizer_profile_id"], "dreamer_auto")
        self.assertEqual(status["optimizer_configured_mode"], OPTIMIZER_MODE_DREAMER_V3_ACTIVE)
        self.assertEqual(status["optimizer_effective_mode"], DEFAULT_OPTIMIZER_MODE)
        self.assertFalse(status["optimizer_dreamer_v3_available"])
        self.assertFalse(status["optimizer_dreamer_v3_active_available"])
        self.assertIn(DEFAULT_OPTIMIZER_MODE, status["optimizer_available_modes"])
        self.assertIn(OPTIMIZER_MODE_DREAMER_V3_ACTIVE, status["optimizer_unavailable_modes"])
        self.assertIn("unavailable", status["optimizer_fallback_reason"])


def _dispatch_telemetry(client: GaggimateMQTTClient, payload: dict) -> None:
    message = type(
        "Message",
        (),
        {
            "topic": "gaggimate/AA_BB/rl/dreamer/telemetry",
            "payload": json.dumps(payload).encode(),
        },
    )()
    client._on_message(client._client, None, message)


def _dispatch_firmware_ack(client: GaggimateMQTTClient, command_payload: dict) -> None:
    payload = {
        "event_type": "dreamer_live_ack",
        "schema_version": 1,
        "machine_id": command_payload["machine_id"],
        "publication_id": command_payload["publication_id"],
        "sequence": command_payload["sequence"],
        "step_index": command_payload["step_index"],
        "ack_scope": "esp32_received",
        "accepted": True,
        "status": "accepted",
        "reason": "accepted",
        "source": "gaggimate_mqtt",
        "timestamp": 20,
    }
    message = type(
        "Message",
        (),
        {
            "topic": "gaggimate/AA_BB/rl/dreamer/ack",
            "payload": json.dumps(payload).encode(),
        },
    )()
    client._on_message(client._client, None, message)


def _telemetry_payload(
    *,
    step_index: int,
    shot_id: str = "shot_1",
    profile_id: str,
) -> dict:
    return {
        "event_type": "dreamer_live_telemetry",
        "schema_version": 2,
        "machine_id": "gaggimate:AA_BB",
        "shot_id": shot_id,
        "profile_id": profile_id,
        "step_index": step_index,
        "elapsed_ms": step_index * 250,
        "sample_interval_ms": 250,
        "observation": {
            "pressure_bar": 2.0,
            "pump_flow_ml_s": 2.5,
            "beverage_flow_g_s": 1.5,
            "weight_g": 4.0,
            "temperature_c": 92.0,
        },
        "target": {
            "pressure_bar": 3.0,
            "pump_flow_ml_s": 2.0,
            "temperature_c": 93.0,
            "pump_target_mode": 1,
            "valve_open": True,
            "yield_g": 36.0,
        },
        "capabilities": {
            "pressure_control_allowed": True,
            "flow_control_allowed": True,
            "pump_mode_control_allowed": True,
            "valve_control_allowed": True,
            "temperature_control_allowed": True,
            "stop_control_allowed": True,
        },
        "source": "gaggimate_mqtt",
    }


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from espresso_rl.application.dreamer_live_acknowledgements import (
    ACK_OUTCOME_DUPLICATE,
    ACK_OUTCOME_LATE,
    ACK_OUTCOME_MISMATCH,
    ACK_OUTCOME_UNKNOWN,
    DreamerLiveControlAcknowledgementService,
)
from espresso_rl.application.dreamer_live_control import DreamerLiveControlApplication
from espresso_rl.application.dreamer_live_telemetry import DreamerLiveTelemetryApplication
from espresso_rl.domain.dreamer_control import (
    DREAMER_DYNAMIC_CONTROL_ACCEPT,
    DREAMER_DYNAMIC_CONTROL_FAIL_SAFE,
    DREAMER_DYNAMIC_CONTROL_REPLAY_LAST,
    DreamerControlSpec,
    DreamerLiveControlAcknowledgement,
    DreamerLiveControlDecision,
    DreamerLiveControlPublication,
)
from espresso_rl.domain.dreamer_telemetry import (
    DreamerLiveEpisodeContext,
    DreamerLiveTelemetry,
    DreamerLiveTelemetryCapabilities,
)


class RecordingPublisher:
    def __init__(self) -> None:
        self.live_control = []

    def publish_recommendation(self, recommendation) -> None:
        raise AssertionError("not used")

    def publish_dreamer_live_control(self, publication) -> None:
        self.live_control.append(publication)

    def publish_status(self, *args, **kwargs) -> None:
        raise AssertionError("not used")


class ManualClock:
    def __init__(self, now_ms: int = 1_000) -> None:
        self.now_ms = now_ms

    def __call__(self) -> int:
        return self.now_ms


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
        self.started.append(telemetry.episode_key)

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
            dose_observed=False,
            initial_target_yield_g=36.0,
            microns_per_step=12.5,
            step_direction="higher_is_finer",
            profile_id="dreamer_auto",
            profile_type="pro",
            profile_phase_count=1,
            taste_objective={"mode": "auto"},
        )


class FailingInference(RecordingInference):
    def infer_action(self, telemetry):
        raise RuntimeError("synthetic inference failure")


class DreamerLiveControlApplicationTests(unittest.TestCase):
    def test_live_telemetry_stays_inactive_without_release_ready_model(self) -> None:
        app = DreamerLiveTelemetryApplication(inference=None, live_control=None)

        result = app.handle_telemetry(_telemetry(step_index=0))

        self.assertEqual(result.outcome, "inactive_model")
        self.assertFalse(result.inference_called)

    def test_live_telemetry_updates_inference_at_four_hz_and_controls_at_one_hz(self) -> None:
        publisher = RecordingPublisher()
        inference = RecordingInference()
        app = DreamerLiveTelemetryApplication(
            inference=inference,
            live_control=DreamerLiveControlApplication(publisher),
            context_provider=StaticContextProvider(),
            enabled=True,
        )

        results = [app.handle_telemetry(_telemetry(step_index=index)) for index in range(5)]

        self.assertEqual(inference.started, ["gaggimate:AA_BB|shot_1|dreamer_auto"])
        self.assertEqual(inference.observed, [0, 1, 2, 3, 4])
        self.assertEqual([result.outcome for result in results], [
            "decision_published",
            "observation_accepted",
            "observation_accepted",
            "observation_accepted",
            "decision_published",
        ])
        self.assertEqual([item.step_index for item in publisher.live_control], [0, 4])

    def test_live_telemetry_rejects_late_duplicate_and_out_of_order_episode_data(self) -> None:
        publisher = RecordingPublisher()
        inference = RecordingInference()
        app = DreamerLiveTelemetryApplication(
            inference=inference,
            live_control=DreamerLiveControlApplication(publisher),
            context_provider=StaticContextProvider(),
            enabled=True,
        )

        late = app.handle_telemetry(_telemetry(step_index=3))
        accepted = app.handle_telemetry(_telemetry(step_index=0))
        duplicate = app.handle_telemetry(_telemetry(step_index=0))

        self.assertEqual(late.outcome, "late_episode_start")
        self.assertEqual(accepted.outcome, "decision_published")
        self.assertEqual(duplicate.outcome, "duplicate_or_out_of_order")
        self.assertEqual(inference.observed, [0])

    def test_live_telemetry_rejects_model_capabilities_the_machine_does_not_support(self) -> None:
        publisher = RecordingPublisher()
        inference = RecordingInference()
        inference.control_spec = DreamerControlSpec(
            dynamic_control_enabled=True,
            pressure_control_allowed=True,
        )
        app = DreamerLiveTelemetryApplication(
            inference=inference,
            live_control=DreamerLiveControlApplication(publisher),
            context_provider=StaticContextProvider(),
            enabled=True,
        )

        result = app.handle_telemetry(_telemetry(step_index=0, pump_mode_control_allowed=False))

        self.assertEqual(result.outcome, "incompatible_capabilities")
        self.assertEqual(inference.started, [])
        self.assertEqual(publisher.live_control, [])

    def test_episode_end_clears_recurrent_state_without_reusing_publication_sequence(self) -> None:
        publisher = RecordingPublisher()
        inference = RecordingInference()
        app = DreamerLiveTelemetryApplication(
            inference=inference,
            live_control=DreamerLiveControlApplication(publisher),
            context_provider=StaticContextProvider(),
            enabled=True,
        )

        app.handle_telemetry(_telemetry(step_index=0, shot_id="shot_1"))
        self.assertTrue(app.end_episode(machine_id="gaggimate:AA_BB", shot_id="shot_1"))
        app.handle_telemetry(_telemetry(step_index=0, shot_id="shot_2"))

        self.assertEqual(inference.ended, [("gaggimate:AA_BB", "shot_1")])
        self.assertEqual([item.sequence for item in publisher.live_control], [1, 2])

    def test_inference_failure_replays_then_fails_safe_without_crashing_runtime(self) -> None:
        publisher = RecordingPublisher()
        app = DreamerLiveTelemetryApplication(
            inference=FailingInference(),
            live_control=DreamerLiveControlApplication(publisher),
            context_provider=StaticContextProvider(),
            enabled=True,
        )

        first = app.handle_telemetry(_telemetry(step_index=0))
        timed_out = app.handle_telemetry(_telemetry(step_index=24))

        self.assertEqual(first.outcome, "inference_failed")
        self.assertIsNone(first.control_result.publication)  # type: ignore[union-attr]
        self.assertEqual(timed_out.outcome, "inference_failed")
        self.assertTrue(timed_out.control_result.publication.fail_safe_required)  # type: ignore[union-attr]

    def test_accepts_bounded_live_target_update_and_publishes_canonical_command(self) -> None:
        publisher = RecordingPublisher()
        app = DreamerLiveControlApplication(publisher)
        spec = DreamerControlSpec(
            dynamic_control_enabled=True,
            pressure_control_allowed=True,
            temperature_control_allowed=True,
            stop_control_allowed=True,
        )

        result = app.handle_live_action(
            machine_id="gaggimate:AA_BB",
            profile_id="dreamer_auto",
            action={
                "pump_target_mode": 1,
                "pressure_target_bar": 13.0,
                "temperature_target_c": 10.0,
                "yield_stop_target_g": 95.0,
            },
            control_spec=spec,
            step_index=0,
            now_ms=1_000,
        )

        self.assertIsNotNone(result.publication)
        self.assertEqual(len(publisher.live_control), 1)
        publication = publisher.live_control[0]
        self.assertEqual(publication.decision.status, DREAMER_DYNAMIC_CONTROL_ACCEPT)
        self.assertFalse(publication.fail_safe_required)
        self.assertEqual(
            publication.action,
            {
                "pump_target_mode": 1,
                "pressure_target_bar": 12.0,
                "temperature_target_c": 20.0,
                "yield_stop_target_g": 90.0,
            },
        )
        self.assertEqual(publication.publication_id, "gaggimate:AA_BB:1")
        self.assertEqual(result.last_command_at_ms, 1_000)

    def test_waits_for_first_command_without_publishing_until_grace_expires(self) -> None:
        publisher = RecordingPublisher()
        app = DreamerLiveControlApplication(publisher)
        spec = DreamerControlSpec(dynamic_control_enabled=True, pressure_control_allowed=True)

        waiting = app.handle_live_action(
            machine_id="gaggimate:AA_BB",
            profile_id="dreamer_auto",
            action=None,
            control_spec=spec,
            step_index=0,
            now_ms=1_000,
        )
        timeout = app.handle_live_action(
            machine_id="gaggimate:AA_BB",
            profile_id="dreamer_auto",
            action=None,
            control_spec=spec,
            step_index=4,
            now_ms=6_001,
        )

        self.assertIsNone(waiting.publication)
        self.assertEqual(len(publisher.live_control), 1)
        self.assertIsNotNone(timeout.publication)
        self.assertEqual(timeout.publication.decision.status, DREAMER_DYNAMIC_CONTROL_FAIL_SAFE)
        self.assertEqual(timeout.publication.decision.reason, "initial_command_timeout")
        self.assertTrue(timeout.publication.fail_safe_required)

    def test_replays_last_command_without_refreshing_deadline_then_fails_safe(self) -> None:
        publisher = RecordingPublisher()
        app = DreamerLiveControlApplication(publisher)
        spec = DreamerControlSpec(dynamic_control_enabled=True, pressure_control_allowed=True)

        accepted = app.handle_live_action(
            machine_id="gaggimate:AA_BB",
            profile_id="dreamer_auto",
            action={"pump_target_mode": 1, "pressure_target_bar": 8.0},
            control_spec=spec,
            step_index=0,
            now_ms=1_000,
        )
        replay = app.handle_live_action(
            machine_id="gaggimate:AA_BB",
            profile_id="dreamer_auto",
            action=None,
            control_spec=spec,
            step_index=4,
            now_ms=5_999,
        )
        stale = app.handle_live_action(
            machine_id="gaggimate:AA_BB",
            profile_id="dreamer_auto",
            action=None,
            control_spec=spec,
            step_index=8,
            now_ms=6_001,
        )

        self.assertEqual(accepted.last_command_at_ms, 1_000)
        self.assertEqual(replay.publication.decision.status, DREAMER_DYNAMIC_CONTROL_REPLAY_LAST)  # type: ignore[union-attr]
        self.assertEqual(replay.publication.action, {"pump_target_mode": 1, "pressure_target_bar": 8.0})  # type: ignore[union-attr]
        self.assertEqual(replay.last_command_at_ms, 1_000)
        self.assertEqual(stale.publication.decision.status, DREAMER_DYNAMIC_CONTROL_FAIL_SAFE)  # type: ignore[union-attr]
        self.assertEqual(stale.publication.decision.reason, "command_timeout")  # type: ignore[union-attr]
        self.assertEqual([item.sequence for item in publisher.live_control], [1, 2, 3])

    def test_invalid_new_command_publishes_fail_safe_without_raw_action(self) -> None:
        publisher = RecordingPublisher()
        app = DreamerLiveControlApplication(publisher)
        spec = DreamerControlSpec(
            dynamic_control_enabled=True,
            pressure_control_allowed=True,
            flow_control_allowed=True,
            pump_mode_control_allowed=True,
        )

        result = app.handle_live_action(
            machine_id="gaggimate:AA_BB",
            profile_id="dreamer_auto",
            action={"pump_target_mode": 1, "pressure_target_bar": 8.0, "flow_target_ml_s": 2.0},
            control_spec=spec,
            step_index=0,
            now_ms=1_000,
        )

        self.assertIsNotNone(result.publication)
        self.assertEqual(result.publication.decision.status, DREAMER_DYNAMIC_CONTROL_FAIL_SAFE)
        self.assertIsNone(result.publication.action)
        self.assertTrue(result.publication.fail_safe_required)
        self.assertTrue(any("pressure and flow" in error for error in result.publication.decision.errors))

    def test_actor_selected_flow_mode_is_accepted_without_pressure_target(self) -> None:
        publisher = RecordingPublisher()
        app = DreamerLiveControlApplication(publisher)
        spec = DreamerControlSpec(
            dynamic_control_enabled=True,
            pressure_control_allowed=True,
            flow_control_allowed=True,
            pump_mode_control_allowed=True,
        )

        result = app.handle_live_action(
            machine_id="gaggimate:AA_BB",
            profile_id="dreamer_auto",
            action={"pump_target_mode": 2, "flow_target_ml_s": 2.25},
            control_spec=spec,
            step_index=0,
            now_ms=1_000,
        )

        self.assertEqual(result.publication.decision.status, DREAMER_DYNAMIC_CONTROL_ACCEPT)  # type: ignore[union-attr]
        self.assertEqual(
            result.publication.action,  # type: ignore[union-attr]
            {"pump_target_mode": 2, "flow_target_ml_s": 2.25},
        )
        self.assertFalse(result.publication.fail_safe_required)  # type: ignore[union-attr]

    def test_correlates_acknowledgement_and_treats_qos_duplicate_idempotently(self) -> None:
        clock = ManualClock()
        acknowledgements = DreamerLiveControlAcknowledgementService(clock_ms=clock)
        publication = _publication(sequence=1, step_index=4)
        ack = _ack(sequence=1, step_index=4, accepted=True, status="accepted", reason="accepted")

        self.assertTrue(acknowledgements.record_publication(publication))
        self.assertEqual(acknowledgements.status_summary(publication.machine_id)["health"], "waiting")

        result = acknowledgements.record_acknowledgement(ack)
        duplicate = acknowledgements.record_acknowledgement(ack)
        summary = acknowledgements.status_summary(publication.machine_id)

        self.assertEqual(result.outcome, "accepted")
        self.assertEqual(duplicate.outcome, ACK_OUTCOME_DUPLICATE)
        self.assertEqual(summary["health"], "healthy")
        self.assertEqual(summary["accepted_count"], 1)
        self.assertEqual(summary["duplicate_ack_count"], 1)
        self.assertEqual(summary["pending_count"], 0)
        self.assertNotIn("publication_id", summary)
        self.assertNotIn("reason", summary)

    def test_rejected_acknowledgement_exposes_only_reason_category(self) -> None:
        clock = ManualClock()
        acknowledgements = DreamerLiveControlAcknowledgementService(clock_ms=clock)
        publication = _publication(sequence=2, step_index=8)
        acknowledgements.record_publication(publication)

        result = acknowledgements.record_acknowledgement(
            _ack(
                sequence=2,
                step_index=8,
                accepted=False,
                status="rejected",
                reason="pressure_target_bar_out_of_bounds",
            )
        )
        summary = acknowledgements.status_summary(publication.machine_id)

        self.assertEqual(result.outcome, "rejected")
        self.assertEqual(result.reason_category, "out_of_bounds")
        self.assertEqual(summary["health"], "attention")
        self.assertEqual(summary["rejected_count"], 1)
        self.assertEqual(summary["last_reason_category"], "out_of_bounds")
        self.assertNotIn("pressure_target_bar_out_of_bounds", str(summary))

    def test_mismatched_unknown_timed_out_and_late_acknowledgements_are_distinct(self) -> None:
        clock = ManualClock()
        acknowledgements = DreamerLiveControlAcknowledgementService(clock_ms=clock, ack_timeout_ms=5_000)
        publication = _publication(sequence=3, step_index=12)
        acknowledgements.record_publication(publication)

        mismatch = acknowledgements.record_acknowledgement(
            _ack(sequence=3, step_index=13, accepted=True, status="accepted", reason="accepted")
        )
        unknown = acknowledgements.record_acknowledgement(
            _ack(sequence=99, step_index=0, accepted=True, status="accepted", reason="accepted")
        )
        clock.now_ms = 6_001
        timed_out = acknowledgements.status_summary(publication.machine_id)
        late = acknowledgements.record_acknowledgement(
            _ack(sequence=3, step_index=12, accepted=True, status="accepted", reason="accepted")
        )
        summary = acknowledgements.status_summary(publication.machine_id)

        self.assertEqual(mismatch.outcome, ACK_OUTCOME_MISMATCH)
        self.assertEqual(unknown.outcome, ACK_OUTCOME_UNKNOWN)
        self.assertEqual(timed_out["timed_out_count"], 1)
        self.assertEqual(late.outcome, ACK_OUTCOME_LATE)
        self.assertEqual(summary["mismatched_ack_count"], 1)
        self.assertEqual(summary["late_ack_count"], 1)
        self.assertEqual(summary["health"], "attention")

def _publication(*, sequence: int, step_index: int) -> DreamerLiveControlPublication:
    return DreamerLiveControlPublication(
        machine_id="gaggimate:AA_BB",
        profile_id="dreamer_auto",
        sequence=sequence,
        step_index=step_index,
        issued_at_ms=1_000,
        decision=DreamerLiveControlDecision(
            status=DREAMER_DYNAMIC_CONTROL_ACCEPT,
            action={"pump_target_mode": 1, "pressure_target_bar": 8.0},
        ),
    )


def _ack(
    *,
    sequence: int,
    step_index: int,
    accepted: bool,
    status: str,
    reason: str,
) -> DreamerLiveControlAcknowledgement:
    return DreamerLiveControlAcknowledgement(
        machine_id="gaggimate:AA_BB",
        publication_id=f"gaggimate:AA_BB:{sequence}",
        sequence=sequence,
        step_index=step_index,
        accepted=accepted,
        status=status,
        reason=reason,
        reported_at=10,
    )


def _telemetry(
    *,
    step_index: int,
    shot_id: str = "shot_1",
    pump_mode_control_allowed: bool = True,
) -> DreamerLiveTelemetry:
    return DreamerLiveTelemetry(
        machine_id="gaggimate:AA_BB",
        shot_id=shot_id,
        profile_id="dreamer_auto",
        step_index=step_index,
        elapsed_ms=step_index * 250,
        pressure_bar=2.0,
        pressure_target_bar=3.0,
        pump_flow_ml_s=2.5,
        pump_flow_target_ml_s=2.0,
        beverage_flow_g_s=1.5,
        weight_g=4.0,
        temperature_c=92.0,
        temperature_target_c=93.0,
        pump_target_mode=1,
        valve_open=True,
        target_yield_g=36.0,
        capabilities=DreamerLiveTelemetryCapabilities(
            pressure_control_allowed=True,
            flow_control_allowed=True,
            pump_mode_control_allowed=pump_mode_control_allowed,
            valve_control_allowed=True,
            temperature_control_allowed=True,
            stop_control_allowed=True,
        ),
    )


if __name__ == "__main__":
    unittest.main()

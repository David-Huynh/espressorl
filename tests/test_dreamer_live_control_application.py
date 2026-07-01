from __future__ import annotations

import unittest

from espresso_rl.application.dreamer_live_control import DreamerLiveControlApplication
from espresso_rl.domain.dreamer_control import (
    DREAMER_DYNAMIC_CONTROL_ACCEPT,
    DREAMER_DYNAMIC_CONTROL_FAIL_SAFE,
    DREAMER_DYNAMIC_CONTROL_REPLAY_LAST,
    DreamerControlSpec,
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


class DreamerLiveControlApplicationTests(unittest.TestCase):
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
            action={"pressure_target_bar": 8.0},
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
        self.assertEqual(replay.publication.action, {"pressure_target_bar": 8.0})  # type: ignore[union-attr]
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
        )

        result = app.handle_live_action(
            machine_id="gaggimate:AA_BB",
            profile_id="dreamer_auto",
            action={"pressure_target_bar": 8.0, "flow_target_ml_s": 2.0},
            control_spec=spec,
            step_index=0,
            now_ms=1_000,
        )

        self.assertIsNotNone(result.publication)
        self.assertEqual(result.publication.decision.status, DREAMER_DYNAMIC_CONTROL_FAIL_SAFE)
        self.assertIsNone(result.publication.action)
        self.assertTrue(result.publication.fail_safe_required)
        self.assertTrue(any("pressure and flow" in error for error in result.publication.decision.errors))


if __name__ == "__main__":
    unittest.main()

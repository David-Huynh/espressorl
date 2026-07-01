from __future__ import annotations

import json
import unittest

from espresso_rl.domain.dreamer_control import DreamerControlSpec
from espresso_rl.domain.dreamer_episodes import DREAMER_EPISODE_FORMAT, validate_dreamer_episode
from espresso_rl.domain.dreamer_pre_shot import (
    DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC,
    DREAMER_PRE_SHOT_ACTION_FIELDS,
)
from espresso_rl.dreamer.dataset import (
    DREAMER_CONTEXT_TRAJECTORY_EMBEDDING_FEATURES,
    DREAMER_CONTEXT_WINDOW_SIZE,
    DREAMER_DYNAMIC_ACTION_FEATURES,
    DREAMER_OBSERVATION_FEATURES,
    DREAMER_OBSERVED_TARGET_FEATURES,
    DREAMER_STATIC_CONTEXT_FEATURES,
    DREAMER_TERMINAL_FEATURES,
    DreamerEpisodeDatasetError,
    build_dreamer_episode_batch,
    build_dreamer_episodes_from_training_rows,
    load_dreamer_episodes_from_jsonl,
)


class DreamerEpisodeLoaderTests(unittest.TestCase):
    def test_loads_validated_rows_as_context_isolated_sorted_episodes(self) -> None:
        episodes = build_dreamer_episodes_from_training_rows(
            [
                training_row(3, bean_context_id="bean_b", timestamp=1_800_000_003),
                training_row(2, bean_context_id="bean_a", timestamp=1_800_000_002),
                training_row(1, bean_context_id="bean_a", timestamp=1_800_000_001),
            ]
        )

        self.assertEqual(
            [(episode["group_key"]["bean_context_id"], episode["source_training_row_id"]) for episode in episodes],
            [("bean_a", 1), ("bean_a", 2), ("bean_b", 3)],
        )
        self.assertTrue(all(episode["format"] == DREAMER_EPISODE_FORMAT for episode in episodes))
        self.assertTrue(all(validate_dreamer_episode(episode) == [] for episode in episodes))

    def test_fixed_cadence_channels_become_recurrent_step_observations(self) -> None:
        episode = build_dreamer_episodes_from_training_rows([training_row(1)])[0]

        self.assertEqual(episode["sample_interval_ms"], 250)
        self.assertEqual(len(episode["steps"]), 4)
        self.assertEqual(episode["steps"][0]["elapsed_ms"], 0)
        self.assertEqual(episode["steps"][3]["elapsed_ms"], 750)
        self.assertEqual(episode["steps"][0]["observation"]["pressure_bar"], 1.0)
        self.assertEqual(episode["steps"][3]["observation"]["pressure_bar"], 4.0)
        self.assertEqual(episode["steps"][2]["observation"]["weight_g"], 0.72)
        self.assertEqual(episode["steps"][2]["observed_profile_target"]["pressure_target_bar"], 8.0)
        self.assertTrue(episode["steps"][2]["observed_profile_target"]["pressure_target_active"])
        self.assertEqual(episode["steps"][2]["observed_profile_target"]["flow_target_ml_s"], 2.4)
        self.assertFalse(episode["steps"][2]["observed_profile_target"]["flow_target_active"])
        self.assertEqual(episode["steps"][2]["observation"]["pump_flow_ml_s"], 1.04)
        self.assertEqual(episode["steps"][2]["observation"]["beverage_flow_g_s"], 0.3)
        self.assertEqual(episode["steps"][2]["observation"]["temperature_c"], 93.0)
        self.assertEqual(episode["steps"][2]["observed_profile_target"]["temperature_target_c"], 92.5)
        self.assertTrue(episode["steps"][2]["observed_profile_target"]["temperature_target_active"])
        self.assertEqual(episode["steps"][2]["observed_profile_target"]["pump_target_mode"], 1)
        self.assertTrue(episode["steps"][2]["observed_profile_target"]["valve_open"])

    def test_sampled_temperature_becomes_step_telemetry(self) -> None:
        episode = build_dreamer_episodes_from_training_rows(
            [
                training_row(
                    1,
                    observation_overrides={
                        "final_pump_target": "pressure",
                        "final_target_pressure": 9.0,
                        "final_target_flow": 0.0,
                        "profile_temperature_c": 93.0,
                        "final_phase_temperature_c": 92.5,
                        "fixed_cadence_sequence": fixed_cadence_sequence(
                            temperature_c=[92.0, 92.1, 92.2, 92.3],
                            temperature_target_c=[93.0, 93.0, 93.0, 93.0],
                        ),
                    },
                )
            ]
        )[0]

        self.assertEqual(episode["steps"][0]["observation"]["temperature_c"], 92.0)
        self.assertEqual(episode["steps"][3]["observation"]["temperature_c"], 92.3)
        self.assertEqual(episode["steps"][0]["observed_profile_target"]["temperature_target_c"], 93.0)
        self.assertTrue(episode["steps"][0]["observed_profile_target"]["temperature_target_active"])
        self.assertEqual(episode["terminal"]["final_pump_target"], "pressure")
        self.assertEqual(validate_dreamer_episode(episode), [])

    def test_static_recipe_fields_are_not_dynamic_actions(self) -> None:
        episode = build_dreamer_episodes_from_training_rows(
            [
                training_row(
                    1,
                    action_overrides={
                        "relative_grind_steps_from_reference": -2.0,
                        "relative_grind_um_from_reference": -25.0,
                        "dose_g": 18.5,
                        "target_yield_g": 39.0,
                        "target_ratio": 2.108108,
                    },
                )
            ]
        )[0]

        self.assertEqual(episode["static_context"]["relative_grind_steps_from_reference"], -2.0)
        self.assertEqual(episode["static_context"]["dose_g"], 18.5)
        self.assertEqual(episode["static_context"]["initial_target_yield_g"], 39.0)
        self.assertNotIn("target_yield_g", episode["static_context"])
        self.assertEqual(episode["static_context"]["taste_objective"], {"mode": "auto"})
        self.assertIsNone(episode["steps"][0]["dynamic_action"])
        self.assertNotIn("dose_g", episode["steps"][0]["observed_profile_target"])
        self.assertNotIn("relative_grind_steps_from_reference", json.dumps(episode["steps"]))

    def test_partial_static_action_is_masked_without_discarding_episode(self) -> None:
        episode = build_dreamer_episodes_from_training_rows(
            [
                training_row(
                    1,
                    action_overrides={
                        "observed": {
                            "grind": False,
                            "dose": False,
                            "target_yield": True,
                        }
                    },
                    observation_overrides={"beverage_out_g": 41.5},
                )
            ]
        )[0]

        self.assertFalse(episode["static_context"]["grind_observed"])
        self.assertFalse(episode["static_context"]["dose_observed"])
        self.assertTrue(episode["static_context"]["initial_target_yield_observed"])
        self.assertEqual(episode["terminal"]["beverage_out_g"], 41.5)
        self.assertEqual(len(episode["steps"]), 4)

        batch = build_dreamer_episode_batch([episode])
        grind_mask_index = DREAMER_STATIC_CONTEXT_FEATURES.index("grind_observed")
        dose_mask_index = DREAMER_STATIC_CONTEXT_FEATURES.index("dose_observed")
        yield_mask_index = DREAMER_STATIC_CONTEXT_FEATURES.index("initial_target_yield_observed")
        self.assertEqual(batch["static_context"][0, grind_mask_index].item(), 0.0)
        self.assertEqual(batch["static_context"][0, dose_mask_index].item(), 0.0)
        self.assertEqual(batch["static_context"][0, yield_mask_index].item(), 1.0)

    def test_pre_shot_actions_are_deterministic_and_grind_delta_uses_exact_context_history(self) -> None:
        episodes = build_dreamer_episodes_from_training_rows(
            [
                training_row(1, action_overrides={"relative_grind_steps_from_reference": 0.0}),
                training_row(
                    2,
                    action_overrides={
                        "relative_grind_steps_from_reference": 3.0,
                        "relative_grind_um_from_reference": 37.5,
                    },
                ),
            ]
        )

        first_action = episodes[0]["pre_shot_action"]
        second_action = episodes[1]["pre_shot_action"]
        self.assertFalse(first_action["observed"]["grind_delta_steps_from_current"])
        self.assertNotIn("grind_delta_steps_from_current", first_action["values"])
        self.assertEqual(second_action["values"]["grind_delta_steps_from_current"], 3.0)
        self.assertEqual(second_action["values"]["dose_target_g"], 18.0)
        self.assertEqual(second_action["values"]["yield_target_g"], 36.0)
        self.assertEqual(second_action["values"]["pump_target_mode"], 1)
        self.assertEqual(second_action["values"]["pressure_target_bar"], 8.0)
        self.assertEqual(second_action["values"]["temperature_target_c"], 92.5)
        self.assertFalse(second_action["capabilities"]["flow_target_ml_s"])

        batch = build_dreamer_episode_batch(episodes)
        grind_index = DREAMER_PRE_SHOT_ACTION_FIELDS.index("grind_delta_steps_from_current")
        pressure_index = DREAMER_PRE_SHOT_ACTION_FIELDS.index("pressure_target_bar")
        expected_grind_bin = DEFAULT_DREAMER_PRE_SHOT_ACTION_SPEC.bins[
            "grind_delta_steps_from_current"
        ].index(3.0)
        self.assertEqual(batch["pre_shot_action_mask"][:, grind_index].tolist(), [0.0, 1.0])
        self.assertEqual(batch["pre_shot_action_indexes"][1, grind_index].item(), expected_grind_bin)
        self.assertEqual(batch["pre_shot_actions"][1, pressure_index].item(), 8.0)

    def test_unknown_grind_does_not_cross_or_poison_context_boundaries(self) -> None:
        episodes = build_dreamer_episodes_from_training_rows(
            [
                training_row(1, bean_context_id="bean_a"),
                training_row(
                    2,
                    bean_context_id="bean_a",
                    action_overrides={"observed": {"grind": False, "dose": True, "target_yield": True}},
                ),
                training_row(
                    3,
                    bean_context_id="bean_a",
                    action_overrides={
                        "relative_grind_steps_from_reference": 2.0,
                        "relative_grind_um_from_reference": 25.0,
                    },
                ),
                training_row(
                    4,
                    bean_context_id="bean_b",
                    action_overrides={
                        "relative_grind_steps_from_reference": 9.0,
                        "relative_grind_um_from_reference": 112.5,
                    },
                ),
            ]
        )

        by_row = {episode["source_training_row_id"]: episode for episode in episodes}
        self.assertFalse(by_row[1]["pre_shot_action"]["observed"]["grind_delta_steps_from_current"])
        self.assertFalse(by_row[2]["pre_shot_action"]["observed"]["grind_delta_steps_from_current"])
        self.assertFalse(by_row[3]["pre_shot_action"]["observed"]["grind_delta_steps_from_current"])
        self.assertFalse(by_row[4]["pre_shot_action"]["observed"]["grind_delta_steps_from_current"])

    def test_pre_shot_flow_mode_is_extracted_without_fabricating_stage_duration(self) -> None:
        episode = build_dreamer_episodes_from_training_rows(
            [
                training_row(
                    1,
                    observation_overrides={
                        "fixed_cadence_sequence": fixed_cadence_sequence(
                            pressure_target_bar=[1.0, 1.0, 8.0, 8.0],
                            pump_flow_target_ml_s=[2.0, 2.0, 0.0, 0.0],
                            pump_target_mode=[2, 2, 1, 1],
                        )
                    },
                )
            ]
        )[0]

        action = episode["pre_shot_action"]
        self.assertEqual(action["values"]["pump_target_mode"], 2)
        self.assertEqual(action["values"]["flow_target_ml_s"], 2.0)
        self.assertNotIn("pressure_target_bar", action["values"])
        self.assertTrue(action["capabilities"]["pressure_target_bar"])
        self.assertTrue(action["capabilities"]["flow_target_ml_s"])
        self.assertNotIn("initial_stage_duration_s", action["values"])
        self.assertFalse(action["observed"]["initial_stage_duration_s"])
        self.assertFalse(action["capabilities"]["initial_stage_duration_s"])

    def test_pre_shot_extraction_rejects_unsafe_initial_profile_target(self) -> None:
        row = training_row(
            1,
            observation_overrides={
                "fixed_cadence_sequence": fixed_cadence_sequence(
                    pressure_target_bar=[13.0, 13.0, 13.0, 13.0]
                )
            },
        )

        with self.assertRaisesRegex(DreamerEpisodeDatasetError, "pressure_target_bar is outside hard bounds"):
            build_dreamer_episodes_from_training_rows([row])

    def test_future_dynamic_yield_stop_target_is_not_initial_static_yield(self) -> None:
        episode = build_dreamer_episodes_from_training_rows([training_row(1)])[0]

        episode["steps"][0]["dynamic_action"] = {"yield_stop_target_g": 38.5, "stop": False}
        self.assertEqual(validate_dreamer_episode(episode), [])

        episode["steps"][0]["dynamic_action"] = {"target_yield_g": 38.5}
        errors = validate_dreamer_episode(episode)
        self.assertTrue(any("dynamic_action contains static recipe fields" in error for error in errors))

    def test_preserves_not_followed_recommendation_attribution(self) -> None:
        episode = build_dreamer_episodes_from_training_rows(
            [
                training_row(
                    1,
                    recommendation={
                        "recommendation_id": "rec_1",
                        "grind_delta_steps_from_current": -2,
                        "grind_delta_um_from_current": -25.0,
                        "projected_relative_step_from_reference": -2.0,
                        "projected_relative_grind_um_from_reference": -25.0,
                        "next_dose_g": 18.0,
                        "target_yield_g": 36.0,
                        "target_ratio": 2.0,
                        "decision": "accepted",
                        "follow_through": "not_followed",
                        "attribution_weight": 0.0,
                        "field_trust": {"grind": 0.0, "dose": 0.0, "yield": 0.0},
                    },
                )
            ]
        )[0]

        self.assertEqual(episode["recommendation"]["follow_through"], "not_followed")
        self.assertEqual(episode["recommendation"]["attribution_weight"], 0.0)
        self.assertEqual(episode["terminal"]["reward"], 0.8)

    def test_rejects_absolute_grinder_fields_before_episode_build(self) -> None:
        row = training_row(1)
        row["action"]["current_absolute_step"] = 42

        with self.assertRaisesRegex(DreamerEpisodeDatasetError, "current_absolute_step"):
            build_dreamer_episodes_from_training_rows([row])

    def test_rejects_missing_profile_for_recurrent_episode_training(self) -> None:
        row = training_row(1)
        row["observation"].pop("profile_resampled")

        with self.assertRaisesRegex(DreamerEpisodeDatasetError, "profile_resampled"):
            build_dreamer_episodes_from_training_rows([row])

    def test_jsonl_loader_rejects_non_object_rows(self) -> None:
        with self.assertRaisesRegex(DreamerEpisodeDatasetError, "must be an object"):
            load_dreamer_episodes_from_jsonl("[]\n")

    def test_episode_batch_has_deterministic_feature_order_and_shapes(self) -> None:
        episodes = build_dreamer_episodes_from_training_rows(
            [
                training_row(2, timestamp=1_800_000_002),
                training_row(1, timestamp=1_800_000_001, action_overrides={"relative_grind_steps_from_reference": -2.0}),
            ]
        )

        batch = build_dreamer_episode_batch(list(reversed(episodes)))

        self.assertEqual(tuple(batch["source_training_row_ids"].tolist()), (1, 2))
        self.assertEqual(batch["episode_ids"], ("training_row_1", "training_row_2"))
        self.assertEqual(tuple(batch["observations"].shape), (2, 4, len(DREAMER_OBSERVATION_FEATURES)))
        self.assertEqual(tuple(batch["observed_profile_targets"].shape), (2, 4, len(DREAMER_OBSERVED_TARGET_FEATURES)))
        self.assertEqual(tuple(batch["observed_profile_target_mask"].shape), (2, 4, len(DREAMER_OBSERVED_TARGET_FEATURES)))
        self.assertEqual(tuple(batch["control_action_mask"].shape), (2, 4, len(DREAMER_DYNAMIC_ACTION_FEATURES)))
        self.assertEqual(tuple(batch["decision_step_mask"].shape), (2, 4))
        self.assertEqual(tuple(batch["static_context"].shape), (2, len(DREAMER_STATIC_CONTEXT_FEATURES)))
        self.assertEqual(tuple(batch["context_static"].shape), (2, DREAMER_CONTEXT_WINDOW_SIZE, len(DREAMER_STATIC_CONTEXT_FEATURES)))
        self.assertEqual(tuple(batch["context_terminal"].shape), (2, DREAMER_CONTEXT_WINDOW_SIZE, len(DREAMER_TERMINAL_FEATURES)))
        self.assertEqual(tuple(batch["context_time"].shape), (2, DREAMER_CONTEXT_WINDOW_SIZE, 1))
        self.assertEqual(
            tuple(batch["context_trajectory_embedding"].shape),
            (2, DREAMER_CONTEXT_WINDOW_SIZE, len(DREAMER_CONTEXT_TRAJECTORY_EMBEDDING_FEATURES)),
        )
        self.assertEqual(tuple(batch["context_mask"].shape), (2, DREAMER_CONTEXT_WINDOW_SIZE))
        self.assertEqual(tuple(batch["context_source_training_row_ids"].shape), (2, DREAMER_CONTEXT_WINDOW_SIZE))
        self.assertEqual(batch["feature_names"]["observations"], DREAMER_OBSERVATION_FEATURES)
        self.assertEqual(batch["feature_names"]["observed_profile_target_mask"], DREAMER_OBSERVED_TARGET_FEATURES)
        self.assertEqual(batch["feature_names"]["control_action_mask"], DREAMER_DYNAMIC_ACTION_FEATURES)
        self.assertEqual(batch["feature_names"]["context_static"], DREAMER_STATIC_CONTEXT_FEATURES)
        self.assertEqual(batch["feature_names"]["context_terminal"], DREAMER_TERMINAL_FEATURES)
        self.assertEqual(
            batch["feature_names"]["context_trajectory_embedding"],
            DREAMER_CONTEXT_TRAJECTORY_EMBEDDING_FEATURES,
        )

        pressure_index = DREAMER_OBSERVATION_FEATURES.index("pressure_bar")
        weight_index = DREAMER_OBSERVATION_FEATURES.index("weight_g")
        temp_index = DREAMER_OBSERVATION_FEATURES.index("temperature_c")
        target_temp_index = DREAMER_OBSERVED_TARGET_FEATURES.index("temperature_target_c")
        target_pressure_index = DREAMER_OBSERVED_TARGET_FEATURES.index("pressure_target_bar")
        target_flow_index = DREAMER_OBSERVED_TARGET_FEATURES.index("flow_target_ml_s")
        grind_index = DREAMER_STATIC_CONTEXT_FEATURES.index("relative_grind_steps_from_reference")
        initial_yield_index = DREAMER_STATIC_CONTEXT_FEATURES.index("initial_target_yield_g")
        direction_index = DREAMER_STATIC_CONTEXT_FEATURES.index("step_direction_sign")
        auto_index = DREAMER_STATIC_CONTEXT_FEATURES.index("taste_objective_auto")

        self.assertEqual(batch["observations"][0, 0, pressure_index].item(), 1.0)
        self.assertAlmostEqual(batch["observations"][0, 2, weight_index].item(), 0.72, places=6)
        self.assertEqual(batch["observations"][0, 0, temp_index].item(), 93.0)
        self.assertEqual(batch["observed_profile_targets"][0, 0, target_temp_index].item(), 92.5)
        self.assertEqual(batch["observed_profile_targets"][0, 0, target_pressure_index].item(), 8.0)
        self.assertAlmostEqual(batch["observed_profile_targets"][0, 0, target_flow_index].item(), 2.4, places=6)
        self.assertEqual(batch["observed_profile_target_mask"][0, 0].tolist(), [1.0, 0.0, 1.0, 1.0, 1.0])
        self.assertEqual(batch["static_context"][0, grind_index].item(), -2.0)
        self.assertEqual(batch["static_context"][0, initial_yield_index].item(), 36.0)
        self.assertEqual(batch["static_context"][0, direction_index].item(), 1.0)
        self.assertEqual(batch["static_context"][0, auto_index].item(), 1.0)
        self.assertAlmostEqual(batch["rewards"][0, 3].item(), 0.8, places=6)
        self.assertEqual(batch["continuations"][0, 2].item(), 1.0)
        self.assertEqual(batch["continuations"][0, 3].item(), 0.0)
        self.assertEqual(batch["step_mask"][0, 3].item(), 1.0)
        self.assertEqual(batch["elapsed_seconds"][0, :, 0].tolist(), [0.0, 0.25, 0.5, 0.75])
        self.assertEqual(batch["step_duration_seconds"][0, :, 0].tolist(), [0.25, 0.25, 0.25, 0.25])
        self.assertEqual(batch["decision_step_mask"][0].tolist(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(batch["control_action_mask"][0].sum().item(), 0.0)
        self.assertEqual(batch["control_spec"]["decision_interval_ms"], 1000)
        self.assertAlmostEqual(batch["episode_weights"][0].item(), 0.2, places=6)
        self.assertEqual(batch["context_mask"][0].sum().item(), 0.0)
        self.assertEqual(batch["context_source_training_row_ids"][1, 0].item(), 1)
        self.assertEqual(batch["context_mask"][1, 0].item(), 1.0)
        self.assertEqual(batch["context_time"][1, 0, 0].item(), 1.0)

    def test_episode_batch_context_window_is_exact_context_bounded_and_chronological(self) -> None:
        episodes = build_dreamer_episodes_from_training_rows(
            [
                training_row(
                    1,
                    timestamp=1_800_000_001,
                    action_overrides={"observed": {"grind": False, "dose": True, "target_yield": True}},
                ),
                training_row(2, bean_context_id="bean_other", timestamp=1_800_000_002),
                training_row(3, grinder_context_id="grinder_other", timestamp=1_800_000_003),
                training_row(4, timestamp=1_800_000_004),
                training_row(5, timestamp=1_800_000_005),
            ]
        )

        batch = build_dreamer_episode_batch(episodes, context_window_size=1)

        self.assertEqual(tuple(batch["source_training_row_ids"].tolist()), (1, 4, 5, 3, 2))
        self.assertEqual(batch["context_window_size"], 1)
        self.assertEqual(batch["context_mask"][:, 0].tolist(), [0.0, 1.0, 1.0, 0.0, 0.0])
        self.assertEqual(batch["context_source_training_row_ids"][:, 0].tolist(), [0, 1, 4, 0, 0])
        grind_observed_index = DREAMER_STATIC_CONTEXT_FEATURES.index("grind_observed")
        self.assertEqual(batch["context_static"][1, 0, grind_observed_index].item(), 0.0)
        self.assertEqual(batch["context_static"][2, 0, grind_observed_index].item(), 1.0)
        sample_count_index = DREAMER_CONTEXT_TRAJECTORY_EMBEDDING_FEATURES.index("trajectory_sample_count")
        duration_index = DREAMER_CONTEXT_TRAJECTORY_EMBEDDING_FEATURES.index("trajectory_duration_seconds")
        pressure_mean_index = DREAMER_CONTEXT_TRAJECTORY_EMBEDDING_FEATURES.index("observation_pressure_bar_mean")
        pressure_last_index = DREAMER_CONTEXT_TRAJECTORY_EMBEDDING_FEATURES.index("observation_pressure_bar_last")
        pressure_delta_index = DREAMER_CONTEXT_TRAJECTORY_EMBEDDING_FEATURES.index("observation_pressure_bar_delta")
        target_pressure_mask_index = DREAMER_CONTEXT_TRAJECTORY_EMBEDDING_FEATURES.index(
            "target_mask_pressure_target_bar_mean"
        )
        self.assertEqual(batch["context_trajectory_embedding"][1, 0, sample_count_index].item(), 4.0)
        self.assertEqual(batch["context_trajectory_embedding"][1, 0, duration_index].item(), 1.0)
        self.assertEqual(batch["context_trajectory_embedding"][1, 0, pressure_mean_index].item(), 2.5)
        self.assertEqual(batch["context_trajectory_embedding"][1, 0, pressure_last_index].item(), 4.0)
        self.assertEqual(batch["context_trajectory_embedding"][1, 0, pressure_delta_index].item(), 3.0)
        self.assertEqual(batch["context_trajectory_embedding"][1, 0, target_pressure_mask_index].item(), 1.0)

    def test_episode_batch_encodes_temperature_when_present(self) -> None:
        episode = build_dreamer_episodes_from_training_rows(
            [
                training_row(
                    1,
                    observation_overrides={
                        "final_pump_target": "flow",
                        "final_target_pressure": 0.0,
                        "final_target_flow": 2.5,
                        "profile_temperature_c": 93.0,
                        "final_phase_temperature_c": 92.5,
                    },
                )
            ]
        )[0]

        batch = build_dreamer_episode_batch([episode])

        temp_index = DREAMER_OBSERVATION_FEATURES.index("temperature_c")
        target_feature_index = DREAMER_OBSERVED_TARGET_FEATURES.index("temperature_target_c")
        pump_pressure_index = DREAMER_TERMINAL_FEATURES.index("final_pump_target_pressure")
        pump_flow_index = DREAMER_TERMINAL_FEATURES.index("final_pump_target_flow")
        final_flow_index = DREAMER_TERMINAL_FEATURES.index("final_target_flow")
        self.assertEqual(batch["observations"][0, 0, temp_index].item(), 93.0)
        self.assertEqual(batch["observed_profile_targets"][0, 0, target_feature_index].item(), 92.5)
        self.assertEqual(batch["observed_profile_target_mask"][0, 0, target_feature_index].item(), 1.0)
        self.assertEqual(batch["terminal"][0, pump_pressure_index].item(), 0.0)
        self.assertEqual(batch["terminal"][0, pump_flow_index].item(), 1.0)
        self.assertEqual(batch["terminal"][0, final_flow_index].item(), 2.5)

    def test_episode_batch_masks_inactive_profile_targets(self) -> None:
        episodes = build_dreamer_episodes_from_training_rows(
            [
                training_row(1),
                training_row(
                    2,
                    observation_overrides={
                        "fixed_cadence_sequence": fixed_cadence_sequence(pump_target_mode=[2, 2, 2, 2]),
                    },
                ),
                training_row(
                    3,
                    observation_overrides={
                        "fixed_cadence_sequence": fixed_cadence_sequence(
                            temperature_target_c=[0.0, 0.0, 0.0, 0.0],
                            pump_target_mode=[0, 0, 0, 0],
                        ),
                    },
                ),
            ]
        )

        batch = build_dreamer_episode_batch(episodes)

        self.assertEqual(batch["observed_profile_target_mask"][0, 0].tolist(), [1.0, 0.0, 1.0, 1.0, 1.0])
        self.assertEqual(batch["observed_profile_target_mask"][1, 0].tolist(), [0.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(batch["observed_profile_target_mask"][2, 0].tolist(), [0.0, 0.0, 0.0, 1.0, 1.0])

    def test_episode_batch_uses_explicit_pump_target_mode_masks(self) -> None:
        episode = build_dreamer_episodes_from_training_rows(
            [
                training_row(
                    1,
                    observation_overrides={
                        "fixed_cadence_sequence": fixed_cadence_sequence(
                            pressure_target_bar=[1.0, 1.0, 1.0, 1.0],
                            pump_flow_target_ml_s=[8.0, 8.0, 8.0, 8.0],
                            temperature_c=[86.0, 86.1, 86.2, 86.3],
                            temperature_target_c=[86.5, 86.5, 86.5, 86.5],
                            pump_target_mode=[2, 2, 2, 2],
                        ),
                    },
                )
            ]
        )[0]

        batch = build_dreamer_episode_batch([episode])

        temp_index = DREAMER_OBSERVATION_FEATURES.index("temperature_c")
        target_temp_index = DREAMER_OBSERVED_TARGET_FEATURES.index("temperature_target_c")
        self.assertEqual(batch["observed_profile_target_mask"][0, 0].tolist(), [0.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(batch["observations"][0, 0, temp_index].item(), 86.0)
        self.assertEqual(batch["observed_profile_targets"][0, 0, target_temp_index].item(), 86.5)

    def test_episode_batch_pads_shorter_episodes_and_masks_padding(self) -> None:
        first, second = build_dreamer_episodes_from_training_rows(
            [
                training_row(1, observation_overrides={"fixed_cadence_sequence": fixed_cadence_sequence(2)}),
                training_row(2, observation_overrides={"fixed_cadence_sequence": fixed_cadence_sequence(3)}),
            ]
        )

        batch = build_dreamer_episode_batch([second, first], pad_to_step_count=4)

        self.assertEqual(tuple(batch["observations"].shape), (2, 4, len(DREAMER_OBSERVATION_FEATURES)))
        self.assertEqual(batch["step_mask"][0].tolist(), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(batch["step_mask"][1].tolist(), [1.0, 1.0, 1.0, 0.0])
        self.assertAlmostEqual(batch["rewards"][0, 1].item(), 0.8, places=6)
        self.assertEqual(batch["rewards"][0, [0, 2, 3]].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(batch["continuations"][1].tolist(), [1.0, 1.0, 0.0, 0.0])

    def test_episode_batch_encodes_dynamic_action_values_and_presence_masks(self) -> None:
        episode = build_dreamer_episodes_from_training_rows([training_row(1)])[0]
        episode["steps"][0]["dynamic_action"] = {"yield_stop_target_g": 38.5, "stop": False}

        batch = build_dreamer_episode_batch(
            [episode],
            control_spec=DreamerControlSpec(dynamic_control_enabled=True, stop_control_allowed=True),
        )

        yield_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("yield_stop_target_g")
        stop_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("stop")
        pressure_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("pressure_target_bar")
        self.assertEqual(batch["dynamic_actions"][0, 0, yield_index].item(), 38.5)
        self.assertEqual(batch["dynamic_action_mask"][0, 0, yield_index].item(), 1.0)
        self.assertEqual(batch["dynamic_actions"][0, 0, stop_index].item(), 0.0)
        self.assertEqual(batch["dynamic_action_mask"][0, 0, stop_index].item(), 1.0)
        self.assertEqual(batch["dynamic_action_mask"][0, 0, pressure_index].item(), 0.0)
        self.assertEqual(batch["control_action_mask"][0, 0, yield_index].item(), 1.0)
        self.assertEqual(batch["control_action_mask"][0, 0, stop_index].item(), 1.0)
        self.assertEqual(batch["control_action_mask"][0, 0, pressure_index].item(), 0.0)
        self.assertEqual(batch["dynamic_actions"][0, 1, yield_index].item(), 38.5)
        self.assertEqual(batch["dynamic_action_mask"][0, 1, yield_index].item(), 1.0)
        self.assertEqual(batch["dynamic_action_mask"][0, 3, stop_index].item(), 1.0)

    def test_episode_batch_rejects_dynamic_actions_between_decision_steps(self) -> None:
        episode = build_dreamer_episodes_from_training_rows([training_row(1)])[0]
        episode["steps"][1]["dynamic_action"] = {"stop": False}

        with self.assertRaisesRegex(DreamerEpisodeDatasetError, "decision steps"):
            build_dreamer_episode_batch(
                [episode],
                control_spec=DreamerControlSpec(dynamic_control_enabled=True, stop_control_allowed=True),
            )

    def test_episode_batch_masks_controls_by_decision_cadence_and_capability(self) -> None:
        episode = build_dreamer_episodes_from_training_rows(
            [training_row(1, observation_overrides={"fixed_cadence_sequence": fixed_cadence_sequence(5)})]
        )[0]

        batch = build_dreamer_episode_batch(
            [episode],
            control_spec=DreamerControlSpec(
                decision_interval_ms=500,
                dynamic_control_enabled=True,
                pressure_control_allowed=True,
                stop_control_allowed=True,
            ),
        )

        pressure_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("pressure_target_bar")
        stop_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("stop")
        flow_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("flow_target_ml_s")
        self.assertEqual(batch["decision_step_mask"][0].tolist(), [1.0, 0.0, 1.0, 0.0, 1.0])
        self.assertEqual(batch["control_action_mask"][0, 0, pressure_index].item(), 1.0)
        self.assertEqual(batch["control_action_mask"][0, 0, stop_index].item(), 1.0)
        self.assertEqual(batch["control_action_mask"][0, 0, flow_index].item(), 0.0)
        self.assertEqual(batch["control_action_mask"][0, 1].sum().item(), 0.0)

    def test_episode_batch_holds_decision_action_until_next_fixed_decision_step(self) -> None:
        episode = build_dreamer_episodes_from_training_rows(
            [training_row(1, observation_overrides={"fixed_cadence_sequence": fixed_cadence_sequence(5)})]
        )[0]
        episode["steps"][0]["dynamic_action"] = {"pressure_target_bar": 2.0}
        episode["steps"][2]["dynamic_action"] = {"pressure_target_bar": 8.0}

        batch = build_dreamer_episode_batch(
            [episode],
            control_spec=DreamerControlSpec(
                decision_interval_ms=500,
                dynamic_control_enabled=True,
                pressure_control_allowed=True,
            ),
        )

        pressure_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("pressure_target_bar")
        self.assertEqual(batch["decision_step_mask"][0].tolist(), [1.0, 0.0, 1.0, 0.0, 1.0])
        self.assertEqual(batch["dynamic_actions"][0, :, pressure_index].tolist(), [2.0, 2.0, 8.0, 8.0, 8.0])
        self.assertEqual(batch["dynamic_action_mask"][0, :, pressure_index].tolist(), [1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(batch["control_action_mask"][0, :, pressure_index].tolist(), [1.0, 0.0, 1.0, 0.0, 1.0])

    def test_episode_batch_rejects_invalid_episode(self) -> None:
        episode = build_dreamer_episodes_from_training_rows([training_row(1)])[0]
        episode["static_context"]["current_absolute_step"] = 42

        with self.assertRaisesRegex(DreamerEpisodeDatasetError, "current_absolute_step"):
            build_dreamer_episode_batch([episode])

    def test_episode_loader_rejects_missing_temperature(self) -> None:
        row = training_row(1)
        row["observation"].pop("profile_temperature_c")

        with self.assertRaisesRegex(DreamerEpisodeDatasetError, "profile_temperature_c"):
            build_dreamer_episodes_from_training_rows([row])

    def test_episode_loader_rejects_missing_fixed_cadence_sequence(self) -> None:
        row = training_row(1)
        row["observation"].pop("fixed_cadence_sequence")

        with self.assertRaisesRegex(DreamerEpisodeDatasetError, "fixed_cadence_sequence"):
            build_dreamer_episodes_from_training_rows([row])

    def test_episode_loader_rejects_invalid_fixed_cadence_channel(self) -> None:
        row = training_row(1)
        row["observation"]["fixed_cadence_sequence"]["temperature_c"].pop()

        with self.assertRaisesRegex(DreamerEpisodeDatasetError, "matching lengths"):
            build_dreamer_episodes_from_training_rows([row])


def training_row(
    row_id: int,
    *,
    bean_context_id: str = "bean_1",
    grinder_context_id: str = "grinder_1",
    timestamp: int | None = None,
    action_overrides: dict | None = None,
    observation_overrides: dict | None = None,
    recommendation: dict | None = None,
) -> dict:
    action = {
        "relative_grind_steps_from_reference": 0.0,
        "relative_grind_um_from_reference": 0.0,
        "dose_g": 18.0,
        "target_yield_g": 36.0,
        "target_ratio": 2.0,
    }
    if action_overrides:
        action.update(action_overrides)
    observation = {
        "shot_id": f"shot_{row_id}",
        "timestamp": timestamp if timestamp is not None else 1_800_000_000 + row_id,
        "beverage_out_g": 36.0,
        "brew_ratio": 2.0,
        "shot_time_s": 30.0,
        "profile_resampled": profile(),
        "raw_profile_available": True,
        "profile_score": 0.91,
        "profile_mse": 0.04,
        "profile_flow_valid": True,
        "profile_flow_masked": False,
        "profile_id": "classic_9_bar",
        "profile_type": "static",
        "profile_phase_count": 2,
        "profile_temperature_c": 93.0,
        "final_phase_temperature_c": 92.5,
        "beverage_flow_profile": [round(0.1 + index * 0.1, 4) for index in range(100)],
        "temperature_profile": [93.0 for _ in range(100)],
        "target_temperature_profile": [92.5 for _ in range(100)],
        "pump_target_mode_profile": [1 for _ in range(100)],
        "fixed_cadence_sequence": fixed_cadence_sequence(),
        "shot_end_state": "completed",
    }
    if observation_overrides:
        observation.update(observation_overrides)
    row = {
        "format": "espresso_rl_training_transition_v1",
        "schema_version": 1,
        "training_row_id": row_id,
        "source": {
            "source_kind": "community_validated_shot",
            "source_validation_id": row_id,
            "install_id": "install_1",
            "payload_hash": "a" * 64,
            "trust_weight": 0.2,
        },
        "context": {
            "machine_id": "machine_1",
            "machine_adapter": "gaggimate",
            "bean_context_id": bean_context_id,
            "grinder_context_id": grinder_context_id,
            "microns_per_step": 12.5,
            "step_direction": "higher_is_finer",
        },
        "action": action,
        "observation": observation,
        "reward": {
            "human_rating": 4,
            "taste_tags": ["balanced"],
            "reward": 0.8,
            "confidence": 1.0,
            "feedback_recorded": True,
            "optimization_weight": 1.0,
        },
    }
    if recommendation is not None:
        row["recommendation"] = recommendation
    return row


def profile() -> list[list[float]]:
    return [
        [round(1.0 + index * 0.1, 4) for index in range(100)],
        [8.0 for _ in range(100)],
        [round(1.0 + index * 0.02, 4) for index in range(100)],
        [2.4 for _ in range(100)],
        [round(index * 0.36, 4) for index in range(100)],
    ]


def fixed_cadence_sequence(step_count: int = 4, **overrides) -> dict:
    sequence = {
        "sample_interval_ms": 250,
        "pressure_bar": [float(index + 1) for index in range(step_count)],
        "pressure_target_bar": [8.0 for _ in range(step_count)],
        "pump_flow_ml_s": [round(1.0 + index * 0.02, 4) for index in range(step_count)],
        "pump_flow_target_ml_s": [2.4 for _ in range(step_count)],
        "beverage_flow_g_s": [round(0.1 + index * 0.1, 4) for index in range(step_count)],
        "weight_g": [round(index * 0.36, 4) for index in range(step_count)],
        "temperature_c": [93.0 for _ in range(step_count)],
        "temperature_target_c": [92.5 for _ in range(step_count)],
        "pump_target_mode": [1 for _ in range(step_count)],
        "valve_open": [True for _ in range(step_count)],
    }
    sequence.update(overrides)
    return sequence


if __name__ == "__main__":
    unittest.main()

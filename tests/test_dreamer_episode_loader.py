from __future__ import annotations

import copy
import json
import unittest

from espresso_rl.domain.dreamer_episodes import DREAMER_EPISODE_FORMAT, validate_dreamer_episode
from espresso_rl.dreamer.dataset import (
    DREAMER_DYNAMIC_ACTION_FEATURES,
    DREAMER_OBSERVATION_FEATURES,
    DREAMER_STATIC_CONTEXT_FEATURES,
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

    def test_profile_channels_become_recurrent_step_observations(self) -> None:
        episode = build_dreamer_episodes_from_training_rows([training_row(1)])[0]

        self.assertEqual(len(episode["steps"]), 100)
        self.assertEqual(episode["steps"][0]["elapsed_fraction"], 0.0)
        self.assertEqual(episode["steps"][99]["elapsed_fraction"], 1.0)
        self.assertEqual(episode["steps"][0]["observation"]["pressure_bar"], 1.0)
        self.assertEqual(episode["steps"][99]["observation"]["pressure_bar"], 10.9)
        self.assertEqual(episode["steps"][10]["observation"]["weight_g"], 3.6)
        self.assertEqual(episode["steps"][10]["observed_profile_target"]["pressure_target_bar"], 8.0)
        self.assertEqual(episode["steps"][10]["observed_profile_target"]["flow_target_ml_s"], 2.4)

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
        self.assertEqual(tuple(batch["observations"].shape), (2, 100, len(DREAMER_OBSERVATION_FEATURES)))
        self.assertEqual(tuple(batch["static_context"].shape), (2, len(DREAMER_STATIC_CONTEXT_FEATURES)))
        self.assertEqual(batch["feature_names"]["observations"], DREAMER_OBSERVATION_FEATURES)

        pressure_index = DREAMER_OBSERVATION_FEATURES.index("pressure_bar")
        weight_index = DREAMER_OBSERVATION_FEATURES.index("weight_g")
        grind_index = DREAMER_STATIC_CONTEXT_FEATURES.index("relative_grind_steps_from_reference")
        initial_yield_index = DREAMER_STATIC_CONTEXT_FEATURES.index("initial_target_yield_g")
        direction_index = DREAMER_STATIC_CONTEXT_FEATURES.index("step_direction_sign")
        auto_index = DREAMER_STATIC_CONTEXT_FEATURES.index("taste_objective_auto")

        self.assertEqual(batch["observations"][0, 0, pressure_index].item(), 1.0)
        self.assertAlmostEqual(batch["observations"][0, 10, weight_index].item(), 3.6, places=6)
        self.assertEqual(batch["static_context"][0, grind_index].item(), -2.0)
        self.assertEqual(batch["static_context"][0, initial_yield_index].item(), 36.0)
        self.assertEqual(batch["static_context"][0, direction_index].item(), 1.0)
        self.assertEqual(batch["static_context"][0, auto_index].item(), 1.0)
        self.assertAlmostEqual(batch["rewards"][0, 99].item(), 0.8, places=6)
        self.assertEqual(batch["continuations"][0, 98].item(), 1.0)
        self.assertEqual(batch["continuations"][0, 99].item(), 0.0)
        self.assertEqual(batch["step_mask"][0, 99].item(), 1.0)
        self.assertAlmostEqual(batch["episode_weights"][0].item(), 0.2, places=6)

    def test_episode_batch_pads_shorter_episodes_and_masks_padding(self) -> None:
        first = build_dreamer_episodes_from_training_rows([training_row(1)])[0]
        second = copy.deepcopy(first)
        second["episode_id"] = "training_row_2"
        second["source_training_row_id"] = 2
        first["steps"] = first["steps"][:2]
        second["steps"] = second["steps"][:3]

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

        batch = build_dreamer_episode_batch([episode])

        yield_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("yield_stop_target_g")
        stop_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("stop")
        pressure_index = DREAMER_DYNAMIC_ACTION_FEATURES.index("pressure_target_bar")
        self.assertEqual(batch["dynamic_actions"][0, 0, yield_index].item(), 38.5)
        self.assertEqual(batch["dynamic_action_mask"][0, 0, yield_index].item(), 1.0)
        self.assertEqual(batch["dynamic_actions"][0, 0, stop_index].item(), 0.0)
        self.assertEqual(batch["dynamic_action_mask"][0, 0, stop_index].item(), 1.0)
        self.assertEqual(batch["dynamic_action_mask"][0, 0, pressure_index].item(), 0.0)
        self.assertEqual(batch["dynamic_action_mask"][0, 1, yield_index].item(), 0.0)

    def test_episode_batch_rejects_invalid_episode(self) -> None:
        episode = build_dreamer_episodes_from_training_rows([training_row(1)])[0]
        episode["static_context"]["current_absolute_step"] = 42

        with self.assertRaisesRegex(DreamerEpisodeDatasetError, "current_absolute_step"):
            build_dreamer_episode_batch([episode])


def training_row(
    row_id: int,
    *,
    bean_context_id: str = "bean_1",
    grinder_context_id: str = "grinder_1",
    timestamp: int | None = None,
    action_overrides: dict | None = None,
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
        "observation": {
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
            "shot_end_state": "completed",
        },
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


if __name__ == "__main__":
    unittest.main()

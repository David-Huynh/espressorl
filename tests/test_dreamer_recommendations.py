from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from espresso_rl.application.checkpoint_loading import load_verified_dreamer_checkpoint
from espresso_rl.application.dreamer_recommendations import (
    DreamerRecommendationError,
    DreamerRecommendationService,
)
from espresso_rl.application.dreamer_shadow_inference import build_dreamer_shadow_inference_session
from espresso_rl.application.trainer_artifacts import (
    MODEL_FILENAME,
    MODEL_MANIFEST_FILENAME,
    build_dreamer_trainer_artifacts,
)
from espresso_rl.domain.models import (
    FixedCadenceShotSequence,
    Recipe,
    RecommendationMode,
    SafetyBounds,
    ShotRecord,
)
from espresso_rl.domain.dreamer_pre_shot import DREAMER_PRE_SHOT_ACTION_FIELDS
from espresso_rl.domain.optimization import OptimizationContext
from espresso_rl.domain.trainer_artifacts import (
    TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW,
    default_training_config,
)
from tests.test_trainer_artifacts import (
    canonical_json,
    dataset_export_text,
    fixed_cadence_sequence,
    profile,
    training_row,
)


class MemoryArtifactStore:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self._payloads = payloads

    def read_bytes(self, reference: str, *, max_bytes: int) -> bytes:
        payload = self._payloads[reference]
        if len(payload) > max_bytes:
            raise ValueError("artifact exceeds limit")
        return payload


class DreamerRecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset_text, dataset_manifest = dataset_export_text(
            [training_row(1), training_row(2), training_row(3), training_row(4)]
        )
        config = default_training_config(
            seed=41,
            artifact_stage=TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW,
        )
        config["world_model_preview_epochs"] = 1
        config["world_model_preview_batch_size"] = 2
        config["world_model_preview_deter_dim"] = 16
        config["world_model_preview_hidden_dim"] = 16
        config["world_model_preview_stoch_size"] = 2
        config["world_model_preview_class_size"] = 4
        config["world_model_preview_action_embed_dim"] = 8
        result = build_dreamer_trainer_artifacts(
            training_rows_jsonl=dataset_text,
            training_dataset_manifest_json=dataset_manifest,
            training_config_json=canonical_json(config) + "\n",
            trainer_git_sha="trainerabc",
            created_at=1_800_000_000,
        )
        files = {file.relative_path: file for file in result.files}
        checkpoint = load_verified_dreamer_checkpoint(
            MemoryArtifactStore(
                {
                    MODEL_FILENAME: files[MODEL_FILENAME].content,
                    MODEL_MANIFEST_FILENAME: files[MODEL_MANIFEST_FILENAME].content,
                }
            ),
            artifact_reference=MODEL_FILENAME,
            manifest_reference=MODEL_MANIFEST_FILENAME,
            expected_artifact_sha256=files[MODEL_FILENAME].sha256,
        )
        cls.session = build_dreamer_shadow_inference_session(checkpoint)

    def setUp(self) -> None:
        self.service = DreamerRecommendationService(session=self.session)
        configure_recipe_heads(self.session.models.actor, grind=0.0, dose=18.0, target_yield=36.0)

    def test_builds_safety_checked_candidate_recommendation_from_context_shots(self) -> None:
        first = local_shot("shot_first", timestamp=1_800_000_010, relative_steps=1.0)
        current = local_shot("shot_current", timestamp=1_800_000_020, relative_steps=2.0)
        current.current_absolute_step = 42.0
        current.absolute_reference_step = 40.0
        context = optimization_context([first, current])

        recommendation = self.service.recommend(context)

        self.assertEqual(recommendation.mode, RecommendationMode.DREAMER_CANDIDATE)
        self.assertEqual(recommendation.install_id, "install_local")
        self.assertEqual(recommendation.machine_id, "machine_local")
        self.assertEqual(recommendation.bean_context_id, "bean_local")
        self.assertEqual(recommendation.grinder_context_id, "grinder_local")
        self.assertEqual(recommendation.profile_id, "classic")
        self.assertEqual(recommendation.source_shot_id, "shot_current")
        self.assertEqual(recommendation.expires_at, context.now + 12 * 60 * 60)
        self.assertAlmostEqual(
            recommendation.projected_relative_step_from_reference,
            context.current_recipe.relative_grind_steps_from_reference
            + recommendation.grind_delta_steps_from_current,
        )
        self.assertAlmostEqual(
            recommendation.grind_delta_um_from_current,
            recommendation.grind_delta_steps_from_current
            * context.current_recipe.microns_per_step
            * context.current_recipe.grinder_direction_sign,
        )
        self.assertEqual(
            recommendation.projected_absolute_step,
            current.current_absolute_step + recommendation.grind_delta_steps_from_current,
        )

    def test_unknown_latest_grind_still_contextualizes_when_current_recipe_is_known(self) -> None:
        known = local_shot("shot_known", timestamp=1_800_000_010, relative_steps=1.0)
        unknown = local_shot("shot_unknown", timestamp=1_800_000_020, relative_steps=None)
        context = optimization_context(
            [known, unknown],
            current_recipe=Recipe(
                relative_grind_steps_from_reference=3.0,
                microns_per_step=12.5,
                dose_g=18.0,
                target_yield_g=36.0,
                target_ratio=2.0,
            ),
        )

        recommendation = self.service.recommend(context)

        self.assertEqual(recommendation.source_shot_id, "shot_unknown")
        self.assertAlmostEqual(
            recommendation.projected_relative_step_from_reference,
            3.0 + recommendation.grind_delta_steps_from_current,
        )

    def test_mixed_context_shots_are_rejected_before_inference(self) -> None:
        current = local_shot("shot_current", timestamp=1_800_000_020, relative_steps=2.0)
        other = copy.deepcopy(current)
        other.shot_id = "shot_other"
        other.timestamp = 1_800_000_010
        other.bean_context_id = "bean_other"

        with self.assertRaisesRegex(DreamerRecommendationError, "mixes bean or grinder"):
            self.service.recommend(optimization_context([other, current]))

    def test_unsafe_actor_output_cannot_become_recommendation(self) -> None:
        actor = self.session.models.actor
        dose_index = DREAMER_PRE_SHOT_ACTION_FIELDS.index("dose_target_g")
        original_weight = actor.pre_shot_heads[dose_index].weight.detach().clone()
        original_bias = actor.pre_shot_heads[dose_index].bias.detach().clone()
        try:
            with torch.no_grad():
                actor.pre_shot_heads[dose_index].weight.zero_()
                actor.pre_shot_heads[dose_index].bias.fill_(-100.0)
                actor.pre_shot_heads[dose_index].bias[0] = 100.0
            shot = local_shot("shot_low_dose", timestamp=1_800_000_020, relative_steps=2.0)
            context = optimization_context(
                [shot],
                current_recipe=Recipe(
                    relative_grind_steps_from_reference=2.0,
                    microns_per_step=12.5,
                    dose_g=14.0,
                    target_yield_g=36.0,
                    target_ratio=36.0 / 14.0,
                ),
                safety_bounds=SafetyBounds(),
            )

            with self.assertRaisesRegex(DreamerRecommendationError, "dose"):
                self.service.recommend(context)
        finally:
            with torch.no_grad():
                actor.pre_shot_heads[dose_index].weight.copy_(original_weight)
                actor.pre_shot_heads[dose_index].bias.copy_(original_bias)


def configure_recipe_heads(actor, *, grind: float, dose: float, target_yield: float) -> None:
    requested = {
        "grind_delta_steps_from_current": grind,
        "dose_target_g": dose,
        "yield_target_g": target_yield,
    }
    with torch.no_grad():
        for field_name, value in requested.items():
            field_index = DREAMER_PRE_SHOT_ACTION_FIELDS.index(field_name)
            head = actor.pre_shot_heads[field_index]
            bins = actor.pre_shot_action_bins[field_index, : actor.pre_shot_bin_counts_tuple[field_index]]
            bin_index = int(torch.argmin(torch.abs(bins - value)).item())
            head.weight.zero_()
            head.bias.fill_(-100.0)
            head.bias[bin_index] = 100.0


def optimization_context(
    shots: list[ShotRecord],
    *,
    current_recipe: Recipe | None = None,
    safety_bounds: SafetyBounds | None = None,
) -> OptimizationContext:
    latest = shots[-1]
    recipe = current_recipe or Recipe(
        relative_grind_steps_from_reference=latest.relative_grind_steps_from_reference or 0.0,
        microns_per_step=latest.microns_per_step,
        dose_g=latest.dose_in_g,
        target_yield_g=latest.target_yield_g,
        target_ratio=latest.target_ratio,
        grinder_step_direction=latest.grinder_step_direction,
    )
    return OptimizationContext(
        install_id="install_local",
        machine_id="machine_local",
        bean_context_id="bean_local",
        grinder_context_id="grinder_local",
        machine_adapter="gaggimate",
        current_recipe=recipe,
        shots=shots,
        safety_bounds=safety_bounds or SafetyBounds(),
        now=1_900_000_000,
    )


def local_shot(
    shot_id: str,
    *,
    timestamp: int,
    relative_steps: float | None,
) -> ShotRecord:
    sequence = FixedCadenceShotSequence.from_dict(fixed_cadence_sequence())
    relative_um = None if relative_steps is None else relative_steps * 12.5
    return ShotRecord(
        shot_id=shot_id,
        timestamp=timestamp,
        install_id="install_local",
        machine_id="machine_local",
        machine_adapter="gaggimate",
        profile=np.asarray(profile(), dtype=np.float32),
        microns_per_step=12.5,
        dose_in_g=18.0,
        target_yield_g=36.0,
        target_ratio=2.0,
        relative_grind_steps_from_reference=relative_steps,
        relative_grind_um_from_reference=relative_um,
        bean_context_id="bean_local",
        grinder_context_id="grinder_local",
        beverage_out_g=36.0,
        brew_ratio=2.0,
        shot_time_s=30.0,
        human_rating=4,
        feedback_recorded=True,
        reward=0.8,
        reward_confidence=1.0,
        profile_score=0.8,
        profile_mse=0.1,
        profile_id="classic",
        profile_type="static",
        profile_phase_count=2,
        profile_temperature_c=93.0,
        final_phase_temperature_c=92.5,
        beverage_flow_profile=np.asarray([0.1 + index * 0.1 for index in range(100)], dtype=np.float32),
        temperature_profile=np.asarray([93.0] * 100, dtype=np.float32),
        target_temperature_profile=np.asarray([92.5] * 100, dtype=np.float32),
        pump_target_mode_profile=np.asarray([1] * 100, dtype=np.uint8),
        fixed_cadence_sequence=sequence,
        raw_profile_hash="b" * 64,
        created_at=timestamp,
        updated_at=timestamp,
    )


if __name__ == "__main__":
    unittest.main()

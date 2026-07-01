from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from espresso_rl.adapters.sqlite_repositories import SQLiteShadowEvaluationRepository, SQLiteStore
from espresso_rl.application.checkpoint_loading import load_verified_dreamer_checkpoint
from espresso_rl.application.dreamer_shadow_evaluation import (
    DreamerShadowEvaluationError,
    DreamerShadowEvaluationService,
)
from espresso_rl.application.dreamer_shadow_inference import build_dreamer_shadow_inference_session
from espresso_rl.application.trainer_artifacts import (
    MODEL_FILENAME,
    MODEL_MANIFEST_FILENAME,
    build_dreamer_trainer_artifacts,
)
from espresso_rl.application.training_export import local_training_transition_from_shot
from espresso_rl.domain.models import (
    FixedCadenceShotSequence,
    Recommendation,
    RecommendationMode,
    ShotRecord,
)
from espresso_rl.domain.dreamer_pre_shot import DREAMER_PRE_SHOT_ACTION_FIELDS
from espresso_rl.domain.shadow_contract import SHADOW_INFERENCE_CONTRACT_LEARNED_CONTEXT_ENCODER_V1
from espresso_rl.domain.shadow_evaluation import ShadowEvaluationStatus, ShadowProposalMatch
from espresso_rl.domain.trainer_artifacts import (
    TRAINER_ARTIFACT_STAGE_WORLD_MODEL_TRAIN_PREVIEW,
    default_training_config,
)
from espresso_rl.main import try_record_shadow_evaluation
from espresso_rl.main import local_context_transitions_for_shadow_replay
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


class MemoryShadowRepository:
    def __init__(self) -> None:
        self.records = {}

    def upsert(self, evaluation) -> None:
        self.records[evaluation.evaluation_id] = evaluation

    def get(self, evaluation_id: str):
        return self.records.get(evaluation_id)

    def get_pending(
        self,
        *,
        install_id,
        machine_id,
        bean_context_id,
        grinder_context_id,
        inference_contract_id=None,
    ):
        matches = [
            record
            for record in self.records.values()
            if record.context_key == (install_id, machine_id, bean_context_id, grinder_context_id)
            and (inference_contract_id is None or record.inference_contract_id == inference_contract_id)
            and record.status == ShadowEvaluationStatus.PENDING_OUTCOME
        ]
        return max(matches, key=lambda record: record.source_timestamp, default=None)

    def list_context(
        self,
        *,
        install_id,
        machine_id,
        bean_context_id,
        grinder_context_id,
        inference_contract_id=None,
        limit=100,
    ):
        matches = [
            record
            for record in self.records.values()
            if record.context_key == (install_id, machine_id, bean_context_id, grinder_context_id)
            and (inference_contract_id is None or record.inference_contract_id == inference_contract_id)
        ]
        return sorted(matches, key=lambda record: record.source_timestamp, reverse=True)[:limit]


class IncrementingClock:
    def __init__(self, value: int = 1_900_000_000) -> None:
        self.value = value

    def __call__(self) -> int:
        self.value += 1
        return self.value


class FailingShadowService:
    def evaluate_transition(self, transition, *, bo_recommendation=None):
        raise RuntimeError("shadow storage unavailable")


class MemoryShotRepository:
    def __init__(self, shots) -> None:
        self.shots = shots

    def list_recent(
        self,
        install_id,
        machine_id,
        bean_context_id=None,
        limit=200,
        grinder_context_id=None,
    ):
        matches = [
            shot
            for shot in self.shots
            if shot.install_id == install_id
            and shot.machine_id == machine_id
            and shot.bean_context_id == bean_context_id
            and shot.grinder_context_id == grinder_context_id
        ]
        return sorted(matches, key=lambda shot: shot.timestamp, reverse=True)[:limit]


class DreamerShadowEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset_text, dataset_manifest = dataset_export_text(
            [training_row(1), training_row(2), training_row(3), training_row(4)]
        )
        config = default_training_config(
            seed=31,
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
        configure_recipe_heads(self.session.models.actor, grind=0.0, dose=18.0, target_yield=38.0)
        self.repository = MemoryShadowRepository()
        self.service = DreamerShadowEvaluationService(
            session=self.session,
            repository=self.repository,
            clock=IncrementingClock(),
        )

    def test_creates_relative_shadow_proposal_with_context_matched_bo_comparison(self) -> None:
        row = training_row(10)
        recommendation = bo_recommendation(row)

        result = self.service.evaluate_transition(row, bo_recommendation=recommendation)

        evaluation = result.evaluation
        self.assertTrue(result.created)
        self.assertEqual(evaluation.context_key, ("install_1", "machine_1", "bean_1", "grinder_1"))
        self.assertEqual(
            evaluation.inference_contract_id,
            SHADOW_INFERENCE_CONTRACT_LEARNED_CONTEXT_ENCODER_V1,
        )
        self.assertEqual(evaluation.status, ShadowEvaluationStatus.PENDING_OUTCOME)
        self.assertEqual(evaluation.dreamer_proposal.source, "dreamer_v3")
        self.assertEqual(evaluation.bo_proposal.source, "bayesian_optimization")
        self.assertAlmostEqual(
            evaluation.dreamer_proposal.projected_relative_step_from_reference,
            row["action"]["relative_grind_steps_from_reference"]
            + evaluation.dreamer_proposal.grind_delta_steps_from_current,
        )
        self.assertNotIn("absolute", canonical_json(evaluation.to_dict()))

    def test_release_ready_session_can_still_run_shadow_evaluation(self) -> None:
        release_session = replace(
            self.session,
            status=replace(
                self.session.status,
                inference_ready=True,
                recommendation_enabled=True,
            ),
        )
        service = DreamerShadowEvaluationService(
            session=release_session,
            repository=MemoryShadowRepository(),
            clock=IncrementingClock(),
        )

        result = service.evaluate_transition(training_row(11))

        self.assertEqual(result.evaluation.dreamer_proposal.source, "dreamer_v3")
        self.assertEqual(result.evaluation.status, ShadowEvaluationStatus.PENDING_OUTCOME)

    def test_next_same_context_transition_resolves_previous_outcome(self) -> None:
        first_row = training_row(20)
        first = self.service.evaluate_transition(first_row).evaluation
        second_row = copy.deepcopy(training_row(21))
        second_row["observation"]["shot_id"] = "shot_21"
        second_row["observation"]["timestamp"] += 10
        second_row["action"]["relative_grind_steps_from_reference"] = (
            first.dreamer_proposal.projected_relative_step_from_reference
        )
        second_row["action"]["relative_grind_um_from_reference"] = (
            first.dreamer_proposal.projected_relative_grind_um_from_reference
        )
        second_row["action"]["dose_g"] = first.dreamer_proposal.next_dose_g
        second_row["action"]["target_yield_g"] = first.dreamer_proposal.target_yield_g
        second_row["action"]["target_ratio"] = first.dreamer_proposal.target_ratio

        result = self.service.evaluate_transition(second_row)

        resolved = result.resolved_previous
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.status, ShadowEvaluationStatus.OUTCOME_OBSERVED)
        self.assertEqual(resolved.dreamer_match, ShadowProposalMatch.MATCHED)
        self.assertEqual(resolved.outcome_shot_id, "shot_21")
        summary = self.service.context_summary(
            install_id="install_1",
            machine_id="machine_1",
            bean_context_id="bean_1",
            grinder_context_id="grinder_1",
        )
        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["observed_count"], 1)
        self.assertEqual(summary["dreamer_matched_count"], 1)
        self.assertTrue(summary["shadow_only"])

    def test_different_bean_or_grinder_context_does_not_resolve_pending_record(self) -> None:
        original = self.service.evaluate_transition(training_row(30)).evaluation
        other = copy.deepcopy(training_row(31))
        other["context"]["bean_context_id"] = "bean_2"
        other["context"]["grinder_context_id"] = "grinder_2"
        other["observation"]["shot_id"] = "shot_other"

        result = self.service.evaluate_transition(other)

        self.assertIsNone(result.resolved_previous)
        self.assertEqual(self.repository.get(original.evaluation_id).status, ShadowEvaluationStatus.PENDING_OUTCOME)
        self.assertEqual(len(self.repository.records), 2)

    def test_repeated_transition_is_idempotent(self) -> None:
        row = training_row(40)
        first = self.service.evaluate_transition(row)
        second = self.service.evaluate_transition(row)

        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertEqual(first.evaluation.evaluation_id, second.evaluation.evaluation_id)
        self.assertEqual(len(self.repository.records), 1)

    def test_context_history_replay_is_exact_context_and_current_episode_scoped(self) -> None:
        first = training_row(81)
        second = training_row(82)
        current = training_row(83)

        batch, current_index = self.service._episode_batch(
            current,
            context_transitions=[second, first],
        )

        self.assertEqual(current_index, 2)
        self.assertEqual(tuple(batch["source_training_row_ids"].tolist()), (81, 82, 83))
        self.assertEqual(batch["context_source_training_row_ids"][2, :2].tolist(), [81, 82])
        self.assertEqual(batch["context_mask"][2, :2].tolist(), [1.0, 1.0])

        result = self.service.evaluate_transition(current, context_transitions=[second, first])
        self.assertTrue(result.created)

    def test_context_history_replay_rejects_mixed_or_future_context(self) -> None:
        current = training_row(90)
        mixed = copy.deepcopy(training_row(89))
        mixed["context"]["bean_context_id"] = "other_bean"
        with self.assertRaisesRegex(DreamerShadowEvaluationError, "mixes bean or grinder"):
            self.service.evaluate_transition(current, context_transitions=[mixed])

        future = copy.deepcopy(training_row(91))
        future["observation"]["timestamp"] = current["observation"]["timestamp"] + 1
        with self.assertRaisesRegex(DreamerShadowEvaluationError, "future context"):
            self.service.evaluate_transition(current, context_transitions=[future])

    def test_late_older_transition_does_not_resolve_newer_pending_record(self) -> None:
        newer = copy.deepcopy(training_row(45))
        newer["observation"]["timestamp"] = 1_900_000_100
        pending = self.service.evaluate_transition(newer).evaluation
        older = copy.deepcopy(training_row(46))
        older["observation"]["shot_id"] = "shot_older"
        older["observation"]["timestamp"] = 1_900_000_000

        result = self.service.evaluate_transition(older)

        self.assertIsNone(result.resolved_previous)
        self.assertEqual(self.repository.get(pending.evaluation_id).status, ShadowEvaluationStatus.PENDING_OUTCOME)

    def test_mismatched_bo_context_is_rejected(self) -> None:
        row = training_row(50)
        recommendation = bo_recommendation(row)
        recommendation.bean_context_id = "other_bean"

        with self.assertRaisesRegex(DreamerShadowEvaluationError, "context"):
            self.service.evaluate_transition(row, bo_recommendation=recommendation)

    def test_unsafe_actor_output_is_recorded_and_not_clamped(self) -> None:
        actor = self.session.models.actor
        dose_index = DREAMER_PRE_SHOT_ACTION_FIELDS.index("dose_target_g")
        original_weight = actor.pre_shot_heads[dose_index].weight.detach().clone()
        original_bias = actor.pre_shot_heads[dose_index].bias.detach().clone()
        try:
            with torch.no_grad():
                actor.pre_shot_heads[dose_index].weight.zero_()
                actor.pre_shot_heads[dose_index].bias.fill_(-100.0)
                actor.pre_shot_heads[dose_index].bias[0] = 100.0
            row = training_row(60)
            row["action"]["dose_g"] = 14.0
            row["action"]["target_ratio"] = row["action"]["target_yield_g"] / 14.0

            evaluation = self.service.evaluate_transition(row).evaluation
        finally:
            with torch.no_grad():
                actor.pre_shot_heads[dose_index].weight.copy_(original_weight)
                actor.pre_shot_heads[dose_index].bias.copy_(original_bias)

        self.assertFalse(evaluation.dreamer_proposal.safety_valid)
        self.assertEqual(evaluation.dreamer_proposal.next_dose_g, 5.0)
        self.assertIn("dose", evaluation.dreamer_proposal.safety_errors[0])

    def test_local_shot_conversion_is_validated_and_context_preserving(self) -> None:
        shot = local_shot()

        transition = local_training_transition_from_shot(shot)

        self.assertIsNotNone(transition)
        self.assertEqual(transition["source"]["source_kind"], "local_validated_shot")
        self.assertEqual(transition["context"]["bean_context_id"], "bean_local")
        self.assertEqual(transition["context"]["grinder_context_id"], "grinder_local")
        self.assertNotIn("current_absolute_step", canonical_json(transition))

    def test_sqlite_adapter_round_trips_and_scopes_records(self) -> None:
        first = self.service.evaluate_transition(training_row(70)).evaluation
        with tempfile.TemporaryDirectory() as temporary_directory:
            with SQLiteStore(Path(temporary_directory) / "shadow.db") as store:
                repository = SQLiteShadowEvaluationRepository(store)
                repository.upsert(first)

                loaded = repository.get(first.evaluation_id)
                scoped = repository.list_context(
                    install_id="install_1",
                    machine_id="machine_1",
                    bean_context_id="bean_1",
                    grinder_context_id="grinder_1",
                )
                other = repository.list_context(
                    install_id="install_1",
                    machine_id="machine_1",
                    bean_context_id="bean_other",
                    grinder_context_id="grinder_1",
                )

        self.assertEqual(loaded, first)
        self.assertEqual(scoped, [first])
        self.assertEqual(other, [])

    def test_shadow_failure_is_contained_before_bo_publication_path_resumes(self) -> None:
        shot = local_shot()
        with patch("espresso_rl.main.logger.exception") as log_exception:
            result = try_record_shadow_evaluation(
                FailingShadowService(),
                shot=shot,
                recommendation=None,
            )

        self.assertIsNone(result)
        log_exception.assert_called_once()

    def test_runtime_context_history_uses_recent_local_exact_context_only(self) -> None:
        first = local_shot()
        first.shot_id = "shot_first"
        first.timestamp = 1_800_000_010
        second = copy.deepcopy(first)
        second.shot_id = "shot_second"
        second.timestamp = 1_800_000_020
        other = copy.deepcopy(first)
        other.shot_id = "shot_other"
        other.timestamp = 1_800_000_030
        other.bean_context_id = "other_bean"
        current = copy.deepcopy(first)
        current.shot_id = "shot_current"
        current.timestamp = 1_800_000_040

        rows = local_context_transitions_for_shadow_replay(
            current,
            shot_repo=MemoryShotRepository([current, other, second, first]),
            limit=8,
        )

        self.assertEqual([row["observation"]["shot_id"] for row in rows], ["shot_first", "shot_second"])


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


def bo_recommendation(row: dict) -> Recommendation:
    action = row["action"]
    context = row["context"]
    return Recommendation(
        recommendation_id=f"bo_{row['training_row_id']}",
        created_at=1_900_000_000,
        updated_at=1_900_000_000,
        expires_at=None,
        install_id=row["source"]["install_id"],
        machine_id=context["machine_id"],
        bean_context_id=context["bean_context_id"],
        grinder_context_id=context["grinder_context_id"],
        grind_delta_steps_from_current=0,
        grind_delta_um_from_current=0.0,
        projected_relative_step_from_reference=action["relative_grind_steps_from_reference"],
        projected_relative_grind_um_from_reference=action["relative_grind_um_from_reference"],
        next_dose_g=action["dose_g"],
        target_yield_g=action["target_yield_g"],
        target_ratio=action["target_ratio"],
        mode=RecommendationMode.LOCAL_BO,
        confidence=0.7,
        reason="BO comparator.",
        source_shot_id=row["observation"]["shot_id"],
    )


def local_shot() -> ShotRecord:
    sequence = FixedCadenceShotSequence.from_dict(fixed_cadence_sequence())
    return ShotRecord(
        shot_id="shot_local",
        timestamp=1_800_000_100,
        install_id="install_local",
        machine_id="machine_local",
        machine_adapter="gaggimate",
        profile=np.asarray(profile(), dtype=np.float32),
        microns_per_step=12.5,
        dose_in_g=18.0,
        target_yield_g=36.0,
        target_ratio=2.0,
        relative_grind_steps_from_reference=2.0,
        relative_grind_um_from_reference=25.0,
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
        created_at=1_800_000_100,
        updated_at=1_800_000_100,
    )


if __name__ == "__main__":
    unittest.main()

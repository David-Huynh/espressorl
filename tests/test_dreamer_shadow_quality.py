from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from espresso_rl.adapters.sqlite_repositories import (
    SQLiteShadowEvaluationRepository,
    SQLiteShadowQualityReportRepository,
    SQLiteStore,
)
from espresso_rl.application.dreamer_shadow_quality import (
    DreamerShadowQualityError,
    DreamerShadowQualityReportService,
    build_shadow_quality_report,
)
from espresso_rl.domain.shadow_evaluation import (
    DreamerShadowEvaluation,
    ShadowEvaluationStatus,
    ShadowProposalMatch,
    ShadowRecipeProposal,
)
from espresso_rl.domain.shadow_contract import (
    SHADOW_INFERENCE_CONTRACT_LEARNED_CONTEXT_ENCODER_V1,
    SHADOW_INFERENCE_CONTRACT_LEGACY_V1,
)
from espresso_rl.domain.shadow_quality import (
    DreamerShadowQualityReport,
    ShadowQualityGateName,
    ShadowQualityStatus,
)

CHECKPOINT_SHA = "a" * 64
PROBE_SHA = "b" * 64
STALE_CHECKPOINT_SHA = "c" * 64
CONTRACT_ID = SHADOW_INFERENCE_CONTRACT_LEARNED_CONTEXT_ENCODER_V1
CONTEXT = ("install_1", "machine_1", "bean_1", "grinder_1")


class MemoryEvaluationRepository:
    def __init__(self, records: list[DreamerShadowEvaluation]) -> None:
        self.records = records

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
        return [
            record
            for record in self.records
            if inference_contract_id is None or record.inference_contract_id == inference_contract_id
        ][:limit]


class MemoryReportRepository:
    def __init__(self) -> None:
        self.reports: dict[str, DreamerShadowQualityReport] = {}

    def upsert(self, report: DreamerShadowQualityReport) -> None:
        self.reports[report.report_id] = report

    def get(self, report_id: str) -> DreamerShadowQualityReport | None:
        return self.reports.get(report_id)

    def get_latest(self, **scope) -> DreamerShadowQualityReport | None:
        matches = [
            report
            for report in self.reports.values()
            if report.context_key
            == (
                scope["install_id"],
                scope["machine_id"],
                scope["bean_context_id"],
                scope["grinder_context_id"],
            )
            and report.checkpoint_artifact_sha256 == scope["checkpoint_artifact_sha256"]
            and report.checkpoint_inference_probe_sha256 == scope["checkpoint_inference_probe_sha256"]
            and report.inference_contract_id == scope["inference_contract_id"]
        ]
        return max(matches, key=lambda report: report.generated_at, default=None)


class DreamerShadowQualityTests(unittest.TestCase):
    def test_balanced_safe_matched_cohorts_pass_all_gates(self) -> None:
        records = _passing_records()

        report = _build(records)

        self.assertEqual(report.overall_status, ShadowQualityStatus.PASS)
        self.assertEqual(report.dreamer_only_matched_count, 10)
        self.assertEqual(report.inference_contract_id, CONTRACT_ID)
        self.assertEqual(report.bo_only_matched_count, 10)
        self.assertEqual(report.both_matched_count, 0)
        self.assertAlmostEqual(report.confidence_brier_score, 0.04)
        self.assertAlmostEqual(report.dreamer_reward_delta_advantage, 0.05)
        self.assertTrue(all(gate.status == ShadowQualityStatus.PASS for gate in report.gates))
        self.assertTrue(report.observational_only)
        self.assertFalse(report.recommendation_enabled)
        self.assertFalse(report.machine_control_enabled)

    def test_insufficient_sample_is_not_reported_as_pass_or_failure(self) -> None:
        report = _build([_evaluation(1, status=ShadowEvaluationStatus.PENDING_OUTCOME)])

        self.assertEqual(report.overall_status, ShadowQualityStatus.INSUFFICIENT_DATA)
        self.assertEqual(_gate(report, ShadowQualityGateName.MINIMUM_EVIDENCE).status, ShadowQualityStatus.INSUFFICIENT_DATA)
        self.assertEqual(_gate(report, ShadowQualityGateName.OUTCOME_COVERAGE).status, ShadowQualityStatus.INSUFFICIENT_DATA)

    def test_unsafe_proposal_fails_even_when_other_evidence_is_sufficient(self) -> None:
        records = _passing_records()
        records[0] = _evaluation(
            1,
            cohort="dreamer",
            source_reward=0.5,
            reward_delta=0.1,
            safety_valid=False,
        )

        report = _build(records)

        self.assertEqual(report.overall_status, ShadowQualityStatus.FAIL)
        self.assertEqual(_gate(report, ShadowQualityGateName.SAFETY_RATE).status, ShadowQualityStatus.FAIL)

    def test_selection_imbalance_fails_observational_comparison(self) -> None:
        records = [
            _evaluation(index, cohort="dreamer", source_reward=0.2, reward_delta=0.1)
            for index in range(1, 11)
        ]
        records.extend(
            _evaluation(index, cohort="bo", source_reward=0.8, reward_delta=0.05)
            for index in range(11, 21)
        )

        report = _build(records)

        self.assertEqual(report.overall_status, ShadowQualityStatus.FAIL)
        self.assertAlmostEqual(report.source_reward_mean_gap, 0.6)
        self.assertEqual(_gate(report, ShadowQualityGateName.SELECTION_BALANCE).status, ShadowQualityStatus.FAIL)

    def test_partial_unmatched_and_both_matched_outcomes_are_not_comparison_samples(self) -> None:
        records = [
            _evaluation(1, cohort="partial", source_reward=0.5, reward_delta=0.1),
            _evaluation(2, cohort="unmatched", source_reward=0.5, reward_delta=-0.1),
            _evaluation(3, cohort="both", source_reward=0.5, reward_delta=0.1),
        ]

        report = _build(records)

        self.assertEqual(report.partial_match_count, 1)
        self.assertEqual(report.unmatched_count, 1)
        self.assertEqual(report.both_matched_count, 1)
        self.assertEqual(report.dreamer_only_matched_count, 0)
        self.assertEqual(report.bo_only_matched_count, 0)
        self.assertIsNone(report.dreamer_reward_delta_advantage)

    def test_stale_checkpoint_records_are_counted_but_never_mixed(self) -> None:
        current = _evaluation(1, cohort="dreamer", source_reward=0.5, reward_delta=0.1)
        stale = _evaluation(
            2,
            cohort="bo",
            source_reward=0.5,
            reward_delta=0.05,
            checkpoint_sha=STALE_CHECKPOINT_SHA,
        )

        report = _build([current, stale])

        self.assertEqual(report.source_record_count, 2)
        self.assertEqual(report.evaluated_record_count, 1)
        self.assertEqual(report.stale_checkpoint_record_count, 1)
        self.assertEqual(report.dreamer_only_matched_count, 1)
        self.assertEqual(report.bo_only_matched_count, 0)

    def test_mixed_inference_contract_records_are_rejected(self) -> None:
        current = _evaluation(1, cohort="dreamer", source_reward=0.5, reward_delta=0.1)
        legacy = _evaluation(
            2,
            cohort="bo",
            source_reward=0.5,
            reward_delta=0.05,
            inference_contract_id=SHADOW_INFERENCE_CONTRACT_LEGACY_V1,
        )

        with self.assertRaisesRegex(DreamerShadowQualityError, "mixes inference contracts"):
            _build([current, legacy])

    def test_cross_context_records_and_duplicate_outcomes_are_rejected(self) -> None:
        mixed = _evaluation(2, cohort="dreamer", source_reward=0.5, reward_delta=0.1, bean_context_id="bean_2")
        with self.assertRaisesRegex(DreamerShadowQualityError, "mixes evaluation contexts"):
            _build([_evaluation(1, cohort="dreamer", source_reward=0.5, reward_delta=0.1), mixed])

        first = _evaluation(3, cohort="dreamer", source_reward=0.5, reward_delta=0.1, outcome_shot_id="same")
        second = _evaluation(4, cohort="bo", source_reward=0.5, reward_delta=0.05, outcome_shot_id="same")
        with self.assertRaisesRegex(DreamerShadowQualityError, "reuses an outcome shot"):
            _build([first, second])

    def test_strict_report_parser_rejects_unknown_fields_and_activation_flags(self) -> None:
        payload = _build(_passing_records()).to_dict()
        self.assertEqual(DreamerShadowQualityReport.from_dict(payload).to_dict(), payload)

        legacy = copy.deepcopy(payload)
        legacy.pop("inference_contract_id")
        legacy["schema_version"] = 1
        parsed_legacy = DreamerShadowQualityReport.from_dict(legacy)
        self.assertEqual(parsed_legacy.inference_contract_id, SHADOW_INFERENCE_CONTRACT_LEGACY_V1)

        unknown = copy.deepcopy(payload)
        unknown["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "fields are invalid"):
            DreamerShadowQualityReport.from_dict(unknown)

        activated = copy.deepcopy(payload)
        activated["recommendation_enabled"] = True
        with self.assertRaisesRegex(ValueError, "observational and shadow-only"):
            DreamerShadowQualityReport.from_dict(activated)

        cherry_picked = copy.deepcopy(payload)
        cherry_picked["policy"]["minimum_safety_rate"] = 0.5
        with self.assertRaisesRegex(ValueError, "thresholds are fixed"):
            DreamerShadowQualityReport.from_dict(cherry_picked)

    def test_service_persists_and_reads_latest_exact_checkpoint_report(self) -> None:
        evaluation_repository = MemoryEvaluationRepository(_passing_records())
        report_repository = MemoryReportRepository()
        service = DreamerShadowQualityReportService(
            evaluations=evaluation_repository,
            reports=report_repository,
            checkpoint_artifact_sha256=CHECKPOINT_SHA,
            checkpoint_inference_probe_sha256=PROBE_SHA,
            inference_contract_id=CONTRACT_ID,
            clock=lambda: 2_000_000_000,
        )

        built = service.build_context_report(
            install_id=CONTEXT[0], machine_id=CONTEXT[1], bean_context_id=CONTEXT[2], grinder_context_id=CONTEXT[3]
        )
        latest = service.latest_context_report(
            install_id=CONTEXT[0], machine_id=CONTEXT[1], bean_context_id=CONTEXT[2], grinder_context_id=CONTEXT[3]
        )

        self.assertEqual(latest, built)
        self.assertEqual(built.status_summary()["overall_status"], "pass")
        self.assertNotIn("dreamer_proposal", built.status_summary())

    def test_sqlite_report_repository_round_trip_and_checkpoint_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with SQLiteStore(Path(tmp) / "quality.db") as store:
                evaluations = SQLiteShadowEvaluationRepository(store)
                reports = SQLiteShadowQualityReportRepository(store)
                for record in _passing_records():
                    evaluations.upsert(record)
                service = DreamerShadowQualityReportService(
                    evaluations=evaluations,
                    reports=reports,
                    checkpoint_artifact_sha256=CHECKPOINT_SHA,
                    checkpoint_inference_probe_sha256=PROBE_SHA,
                    inference_contract_id=CONTRACT_ID,
                    clock=lambda: 2_000_000_001,
                )

                built = service.build_context_report(
                    install_id=CONTEXT[0], machine_id=CONTEXT[1], bean_context_id=CONTEXT[2], grinder_context_id=CONTEXT[3]
                )

                self.assertEqual(reports.get(built.report_id), built)
                self.assertEqual(
                    reports.get_latest(
                        install_id=CONTEXT[0],
                        machine_id=CONTEXT[1],
                        bean_context_id=CONTEXT[2],
                        grinder_context_id=CONTEXT[3],
                        checkpoint_artifact_sha256=CHECKPOINT_SHA,
                        checkpoint_inference_probe_sha256=PROBE_SHA,
                        inference_contract_id=CONTRACT_ID,
                    ),
                    built,
                )
                self.assertIsNone(
                    reports.get_latest(
                        install_id=CONTEXT[0],
                        machine_id=CONTEXT[1],
                        bean_context_id=CONTEXT[2],
                        grinder_context_id=CONTEXT[3],
                        checkpoint_artifact_sha256=STALE_CHECKPOINT_SHA,
                        checkpoint_inference_probe_sha256=PROBE_SHA,
                        inference_contract_id=CONTRACT_ID,
                    )
                )
                self.assertIsNone(
                    reports.get_latest(
                        install_id=CONTEXT[0],
                        machine_id=CONTEXT[1],
                        bean_context_id=CONTEXT[2],
                        grinder_context_id=CONTEXT[3],
                        checkpoint_artifact_sha256=CHECKPOINT_SHA,
                        checkpoint_inference_probe_sha256=PROBE_SHA,
                        inference_contract_id=SHADOW_INFERENCE_CONTRACT_LEGACY_V1,
                    )
                )


def _passing_records() -> list[DreamerShadowEvaluation]:
    records = [
        _evaluation(index, cohort="dreamer", source_reward=0.5, reward_delta=0.1)
        for index in range(1, 11)
    ]
    records.extend(
        _evaluation(index, cohort="bo", source_reward=0.5, reward_delta=0.05)
        for index in range(11, 21)
    )
    return records


def _build(records: list[DreamerShadowEvaluation]) -> DreamerShadowQualityReport:
    return build_shadow_quality_report(
        records,
        install_id=CONTEXT[0],
        machine_id=CONTEXT[1],
        bean_context_id=CONTEXT[2],
        grinder_context_id=CONTEXT[3],
        checkpoint_artifact_sha256=CHECKPOINT_SHA,
        checkpoint_inference_probe_sha256=PROBE_SHA,
        inference_contract_id=CONTRACT_ID,
        generated_at=2_000_000_000,
    )


def _gate(report: DreamerShadowQualityReport, name: ShadowQualityGateName):
    return next(gate for gate in report.gates if gate.name == name)


def _evaluation(
    index: int,
    *,
    cohort: str = "unmatched",
    status: ShadowEvaluationStatus = ShadowEvaluationStatus.OUTCOME_OBSERVED,
    source_reward: float | None = None,
    reward_delta: float | None = None,
    safety_valid: bool = True,
    checkpoint_sha: str = CHECKPOINT_SHA,
    inference_contract_id: str = CONTRACT_ID,
    bean_context_id: str = CONTEXT[2],
    outcome_shot_id: str | None = None,
) -> DreamerShadowEvaluation:
    dreamer_match, bo_match = {
        "dreamer": (ShadowProposalMatch.MATCHED, ShadowProposalMatch.NOT_MATCHED),
        "bo": (ShadowProposalMatch.NOT_MATCHED, ShadowProposalMatch.MATCHED),
        "both": (ShadowProposalMatch.MATCHED, ShadowProposalMatch.MATCHED),
        "partial": (ShadowProposalMatch.PARTIALLY_MATCHED, ShadowProposalMatch.NOT_MATCHED),
        "unmatched": (ShadowProposalMatch.NOT_MATCHED, ShadowProposalMatch.NOT_MATCHED),
    }[cohort]
    pending = status == ShadowEvaluationStatus.PENDING_OUTCOME
    if pending:
        dreamer_match = ShadowProposalMatch.UNKNOWN
        bo_match = ShadowProposalMatch.UNKNOWN
        source_reward = None
        reward_delta = None
    outcome_reward = (
        source_reward + reward_delta
        if source_reward is not None and reward_delta is not None and not pending
        else None
    )
    source_timestamp = 1_900_000_000 + index * 2
    return DreamerShadowEvaluation(
        evaluation_id=f"evaluation_{index}",
        created_at=source_timestamp,
        updated_at=source_timestamp + (1 if not pending else 0),
        checkpoint_artifact_sha256=checkpoint_sha,
        checkpoint_inference_probe_sha256=PROBE_SHA,
        inference_contract_id=inference_contract_id,
        install_id=CONTEXT[0],
        machine_id=CONTEXT[1],
        bean_context_id=bean_context_id,
        grinder_context_id=CONTEXT[3],
        source_training_row_id=index,
        source_shot_id=f"source_{index}",
        source_timestamp=source_timestamp,
        microns_per_step=12.5,
        step_direction="higher_is_finer",
        current_relative_step_from_reference=0.0,
        current_dose_g=18.0,
        current_target_yield_g=36.0,
        current_target_ratio=2.0,
        dreamer_proposal=_proposal("dreamer_v3", safety_valid=safety_valid),
        bo_proposal=_proposal("bayesian_optimization"),
        source_reward=source_reward,
        status=status,
        outcome_shot_id=None if pending else (outcome_shot_id or f"outcome_{index}"),
        outcome_timestamp=None if pending else source_timestamp + 1,
        outcome_relative_step_from_reference=None if pending else 0.0,
        outcome_dose_g=None if pending else 18.0,
        outcome_target_yield_g=None if pending else 36.0,
        outcome_reward=outcome_reward,
        reward_delta=reward_delta,
        dreamer_match=dreamer_match,
        bo_match=bo_match,
    )


def _proposal(source: str, *, safety_valid: bool = True) -> ShadowRecipeProposal:
    return ShadowRecipeProposal(
        source=source,
        grind_delta_steps_from_current=0,
        projected_relative_step_from_reference=0.0,
        projected_relative_grind_um_from_reference=0.0,
        next_dose_g=18.0,
        target_yield_g=36.0,
        target_ratio=2.0,
        confidence=0.8,
        safety_valid=safety_valid,
        safety_errors=() if safety_valid else ("unsafe test proposal",),
    )


if __name__ == "__main__":
    unittest.main()

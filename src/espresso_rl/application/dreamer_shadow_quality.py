from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Iterable

from espresso_rl.domain.shadow_evaluation import (
    DreamerShadowEvaluation,
    ShadowEvaluationStatus,
    ShadowProposalMatch,
)
from espresso_rl.domain.shadow_contract import validate_shadow_inference_contract_id
from espresso_rl.domain.shadow_quality import (
    DreamerShadowQualityReport,
    ShadowQualityGate,
    ShadowQualityGateName,
    ShadowQualityPolicy,
    ShadowQualityStatus,
    overall_shadow_quality_status,
)
from espresso_rl.ports.shadow_evaluations import ShadowEvaluationRepository
from espresso_rl.ports.shadow_quality_reports import ShadowQualityReportRepository


class DreamerShadowQualityError(ValueError):
    pass


@dataclass(frozen=True)
class _OutcomeCohorts:
    both: tuple[DreamerShadowEvaluation, ...]
    dreamer_only: tuple[DreamerShadowEvaluation, ...]
    bo_only: tuple[DreamerShadowEvaluation, ...]
    partial: tuple[DreamerShadowEvaluation, ...]
    unmatched: tuple[DreamerShadowEvaluation, ...]


class DreamerShadowQualityReportService:
    def __init__(
        self,
        *,
        evaluations: ShadowEvaluationRepository,
        reports: ShadowQualityReportRepository,
        checkpoint_artifact_sha256: str,
        checkpoint_inference_probe_sha256: str,
        inference_contract_id: str,
        clock: Callable[[], int],
        policy: ShadowQualityPolicy | None = None,
    ) -> None:
        _sha256(checkpoint_artifact_sha256, "checkpoint artifact")
        _sha256(checkpoint_inference_probe_sha256, "checkpoint inference probe")
        self._evaluations = evaluations
        self._reports = reports
        self._checkpoint_artifact_sha256 = checkpoint_artifact_sha256
        self._checkpoint_inference_probe_sha256 = checkpoint_inference_probe_sha256
        self._inference_contract_id = validate_shadow_inference_contract_id(inference_contract_id)
        self._clock = clock
        self._policy = policy or ShadowQualityPolicy()

    def build_context_report(
        self,
        *,
        install_id: str,
        machine_id: str,
        bean_context_id: str,
        grinder_context_id: str,
        limit: int = 10_000,
    ) -> DreamerShadowQualityReport:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise DreamerShadowQualityError("shadow quality report limit must be 1..10000")
        records = self._evaluations.list_context(
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            grinder_context_id=grinder_context_id,
            inference_contract_id=self._inference_contract_id,
            limit=limit,
        )
        generated_at = self._clock()
        if isinstance(generated_at, bool) or not isinstance(generated_at, int) or generated_at <= 0:
            raise DreamerShadowQualityError("shadow quality report clock must return a positive integer")
        report = build_shadow_quality_report(
            records,
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            grinder_context_id=grinder_context_id,
            checkpoint_artifact_sha256=self._checkpoint_artifact_sha256,
            checkpoint_inference_probe_sha256=self._checkpoint_inference_probe_sha256,
            inference_contract_id=self._inference_contract_id,
            generated_at=generated_at,
            policy=self._policy,
        )
        existing = self._reports.get(report.report_id)
        if existing is not None:
            return existing
        self._reports.upsert(report)
        return report

    def latest_context_report(
        self,
        *,
        install_id: str,
        machine_id: str,
        bean_context_id: str,
        grinder_context_id: str,
    ) -> DreamerShadowQualityReport | None:
        return self._reports.get_latest(
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            grinder_context_id=grinder_context_id,
            checkpoint_artifact_sha256=self._checkpoint_artifact_sha256,
            checkpoint_inference_probe_sha256=self._checkpoint_inference_probe_sha256,
            inference_contract_id=self._inference_contract_id,
        )


def build_shadow_quality_report(
    records: Iterable[DreamerShadowEvaluation],
    *,
    install_id: str,
    machine_id: str,
    bean_context_id: str,
    grinder_context_id: str,
    checkpoint_artifact_sha256: str,
    checkpoint_inference_probe_sha256: str,
    inference_contract_id: str,
    generated_at: int,
    policy: ShadowQualityPolicy | None = None,
) -> DreamerShadowQualityReport:
    policy = policy or ShadowQualityPolicy()
    inference_contract_id = validate_shadow_inference_contract_id(inference_contract_id)
    expected_context = (install_id, machine_id, bean_context_id, grinder_context_id)
    source_records = tuple(records)
    for record in source_records:
        if not isinstance(record, DreamerShadowEvaluation):
            raise DreamerShadowQualityError("shadow quality input contains an invalid record")
        if record.context_key != expected_context:
            raise DreamerShadowQualityError("shadow quality input mixes evaluation contexts")
        if record.inference_contract_id != inference_contract_id:
            raise DreamerShadowQualityError("shadow quality input mixes inference contracts")
        _validate_record_consistency(record)

    current_records = tuple(
        record
        for record in source_records
        if record.checkpoint_artifact_sha256 == checkpoint_artifact_sha256
        and record.checkpoint_inference_probe_sha256 == checkpoint_inference_probe_sha256
    )
    observed = tuple(
        record for record in current_records if record.status == ShadowEvaluationStatus.OUTCOME_OBSERVED
    )
    _reject_duplicate_outcomes(observed)
    cohorts = _partition_outcomes(observed)

    safe_count = sum(record.dreamer_proposal.safety_valid for record in current_records)
    unsafe_count = len(current_records) - safe_count
    safety_rate = _ratio(safe_count, len(current_records))
    outcome_coverage = _ratio(len(observed), len(current_records))

    calibration_records = tuple(record for record in cohorts.dreamer_only if record.reward_delta is not None)
    confidence_brier_score = _mean(
        (record.dreamer_proposal.confidence - (1.0 if record.reward_delta >= 0.0 else 0.0)) ** 2
        for record in calibration_records
    )
    dreamer_source_records = tuple(record for record in cohorts.dreamer_only if record.source_reward is not None)
    bo_source_records = tuple(record for record in cohorts.bo_only if record.source_reward is not None)
    dreamer_source_reward_mean = _mean(record.source_reward for record in dreamer_source_records)
    bo_source_reward_mean = _mean(record.source_reward for record in bo_source_records)
    source_reward_mean_gap = (
        abs(dreamer_source_reward_mean - bo_source_reward_mean)
        if dreamer_source_reward_mean is not None and bo_source_reward_mean is not None
        else None
    )
    dreamer_reward_records = tuple(record for record in cohorts.dreamer_only if record.reward_delta is not None)
    bo_reward_records = tuple(record for record in cohorts.bo_only if record.reward_delta is not None)
    dreamer_reward_delta_mean = _mean(record.reward_delta for record in dreamer_reward_records)
    bo_reward_delta_mean = _mean(record.reward_delta for record in bo_reward_records)
    dreamer_reward_delta_advantage = (
        dreamer_reward_delta_mean - bo_reward_delta_mean
        if dreamer_reward_delta_mean is not None and bo_reward_delta_mean is not None
        else None
    )

    gates = _build_gates(
        policy=policy,
        evaluated_count=len(current_records),
        observed_count=len(observed),
        safety_rate=safety_rate,
        outcome_coverage=outcome_coverage,
        calibration_count=len(calibration_records),
        confidence_brier_score=confidence_brier_score,
        dreamer_source_count=len(dreamer_source_records),
        bo_source_count=len(bo_source_records),
        source_reward_mean_gap=source_reward_mean_gap,
        dreamer_reward_count=len(dreamer_reward_records),
        bo_reward_count=len(bo_reward_records),
        dreamer_reward_delta_advantage=dreamer_reward_delta_advantage,
    )
    evaluation_set_sha256 = _evaluation_set_sha256(source_records)
    report_id = _report_id(
        expected_context=expected_context,
        checkpoint_artifact_sha256=checkpoint_artifact_sha256,
        checkpoint_inference_probe_sha256=checkpoint_inference_probe_sha256,
        inference_contract_id=inference_contract_id,
        evaluation_set_sha256=evaluation_set_sha256,
        policy_version=policy.version,
    )
    return DreamerShadowQualityReport(
        report_id=report_id,
        generated_at=generated_at,
        checkpoint_artifact_sha256=checkpoint_artifact_sha256,
        checkpoint_inference_probe_sha256=checkpoint_inference_probe_sha256,
        inference_contract_id=inference_contract_id,
        install_id=install_id,
        machine_id=machine_id,
        bean_context_id=bean_context_id,
        grinder_context_id=grinder_context_id,
        evaluation_set_sha256=evaluation_set_sha256,
        source_record_count=len(source_records),
        evaluated_record_count=len(current_records),
        stale_checkpoint_record_count=len(source_records) - len(current_records),
        pending_count=len(current_records) - len(observed),
        observed_count=len(observed),
        safe_proposal_count=safe_count,
        unsafe_proposal_count=unsafe_count,
        both_matched_count=len(cohorts.both),
        dreamer_only_matched_count=len(cohorts.dreamer_only),
        bo_only_matched_count=len(cohorts.bo_only),
        partial_match_count=len(cohorts.partial),
        unmatched_count=len(cohorts.unmatched),
        safety_rate=safety_rate,
        outcome_coverage=outcome_coverage,
        confidence_brier_score=confidence_brier_score,
        dreamer_source_reward_mean=dreamer_source_reward_mean,
        bo_source_reward_mean=bo_source_reward_mean,
        source_reward_mean_gap=source_reward_mean_gap,
        dreamer_reward_delta_mean=dreamer_reward_delta_mean,
        bo_reward_delta_mean=bo_reward_delta_mean,
        dreamer_reward_delta_advantage=dreamer_reward_delta_advantage,
        gates=gates,
        overall_status=overall_shadow_quality_status(gates),
        policy=policy,
    )


def _partition_outcomes(records: tuple[DreamerShadowEvaluation, ...]) -> _OutcomeCohorts:
    both: list[DreamerShadowEvaluation] = []
    dreamer_only: list[DreamerShadowEvaluation] = []
    bo_only: list[DreamerShadowEvaluation] = []
    partial: list[DreamerShadowEvaluation] = []
    unmatched: list[DreamerShadowEvaluation] = []
    for record in records:
        dreamer_matched = record.dreamer_match == ShadowProposalMatch.MATCHED
        bo_matched = record.bo_match == ShadowProposalMatch.MATCHED
        has_partial = ShadowProposalMatch.PARTIALLY_MATCHED in (record.dreamer_match, record.bo_match)
        if dreamer_matched and bo_matched:
            both.append(record)
        elif dreamer_matched and not has_partial:
            dreamer_only.append(record)
        elif bo_matched and not has_partial:
            bo_only.append(record)
        elif has_partial:
            partial.append(record)
        else:
            unmatched.append(record)
    return _OutcomeCohorts(
        both=tuple(both),
        dreamer_only=tuple(dreamer_only),
        bo_only=tuple(bo_only),
        partial=tuple(partial),
        unmatched=tuple(unmatched),
    )


def _build_gates(
    *,
    policy: ShadowQualityPolicy,
    evaluated_count: int,
    observed_count: int,
    safety_rate: float | None,
    outcome_coverage: float | None,
    calibration_count: int,
    confidence_brier_score: float | None,
    dreamer_source_count: int,
    bo_source_count: int,
    source_reward_mean_gap: float | None,
    dreamer_reward_count: int,
    bo_reward_count: int,
    dreamer_reward_delta_advantage: float | None,
) -> tuple[ShadowQualityGate, ...]:
    enough_evidence = (
        evaluated_count >= policy.minimum_record_count
        and observed_count >= policy.minimum_observed_count
    )
    minimum_evidence = ShadowQualityGate(
        name=ShadowQualityGateName.MINIMUM_EVIDENCE,
        status=ShadowQualityStatus.PASS if enough_evidence else ShadowQualityStatus.INSUFFICIENT_DATA,
        sample_count=evaluated_count,
        observed_value=float(observed_count),
        threshold=float(policy.minimum_observed_count),
        reason=(
            f"Requires at least {policy.minimum_record_count} records and "
            f"{policy.minimum_observed_count} observed outcomes; found {evaluated_count} and {observed_count}."
        ),
    )
    safety = _minimum_gate(
        name=ShadowQualityGateName.SAFETY_RATE,
        sample_count=evaluated_count,
        observed_value=safety_rate,
        threshold=policy.minimum_safety_rate,
        insufficient_reason="No current-checkpoint proposals are available for safety evaluation.",
        pass_reason="Dreamer proposal safety rate meets the fixed policy threshold.",
        fail_reason="Dreamer proposal safety rate is below the fixed policy threshold.",
        minimum_count=1,
    )
    coverage = _minimum_gate(
        name=ShadowQualityGateName.OUTCOME_COVERAGE,
        sample_count=evaluated_count,
        observed_value=outcome_coverage,
        threshold=policy.minimum_outcome_coverage,
        insufficient_reason="No current-checkpoint proposals are available for outcome coverage.",
        pass_reason="Observed outcome coverage meets the fixed policy threshold.",
        fail_reason="Observed outcome coverage is below the fixed policy threshold.",
        minimum_count=policy.minimum_record_count,
    )
    calibration = _maximum_gate_with_minimum_count(
        name=ShadowQualityGateName.CONFIDENCE_CALIBRATION,
        sample_count=calibration_count,
        minimum_count=policy.minimum_calibration_count,
        observed_value=confidence_brier_score,
        threshold=policy.maximum_confidence_brier_score,
        pass_reason="Dreamer confidence Brier score meets the fixed policy threshold.",
        fail_reason="Dreamer confidence Brier score exceeds the fixed policy threshold.",
    )
    selection_sample_count = min(dreamer_source_count, bo_source_count)
    selection_balance = _maximum_gate_with_minimum_count(
        name=ShadowQualityGateName.SELECTION_BALANCE,
        sample_count=selection_sample_count,
        minimum_count=policy.minimum_comparison_count_per_source,
        observed_value=source_reward_mean_gap,
        threshold=policy.maximum_source_reward_mean_gap,
        pass_reason="Exclusive Dreamer and BO cohorts have sufficiently balanced source rewards.",
        fail_reason="Exclusive Dreamer and BO cohorts have a source-reward imbalance; comparison is biased.",
    )
    comparison_sample_count = min(dreamer_reward_count, bo_reward_count)
    comparison = _minimum_gate_with_minimum_count(
        name=ShadowQualityGateName.MATCHED_OUTCOME_COMPARISON,
        sample_count=comparison_sample_count,
        minimum_count=policy.minimum_comparison_count_per_source,
        observed_value=dreamer_reward_delta_advantage,
        threshold=policy.minimum_dreamer_reward_delta_advantage,
        pass_reason="Exclusive matched Dreamer outcomes meet the reward-delta advantage threshold.",
        fail_reason="Exclusive matched Dreamer outcomes underperform the BO comparison cohort.",
    )
    return (minimum_evidence, safety, coverage, calibration, selection_balance, comparison)


def _minimum_gate(
    *,
    name: ShadowQualityGateName,
    sample_count: int,
    observed_value: float | None,
    threshold: float,
    insufficient_reason: str,
    pass_reason: str,
    fail_reason: str,
    minimum_count: int,
) -> ShadowQualityGate:
    if sample_count < minimum_count or observed_value is None:
        status = ShadowQualityStatus.INSUFFICIENT_DATA
        reason = (
            insufficient_reason
            if sample_count == 0
            else f"Requires at least {minimum_count} current-checkpoint records; found {sample_count}."
        )
    elif observed_value >= threshold:
        status = ShadowQualityStatus.PASS
        reason = pass_reason
    else:
        status = ShadowQualityStatus.FAIL
        reason = fail_reason
    return ShadowQualityGate(name, status, sample_count, observed_value, threshold, reason)


def _maximum_gate_with_minimum_count(
    *,
    name: ShadowQualityGateName,
    sample_count: int,
    minimum_count: int,
    observed_value: float | None,
    threshold: float,
    pass_reason: str,
    fail_reason: str,
) -> ShadowQualityGate:
    if sample_count < minimum_count or observed_value is None:
        return ShadowQualityGate(
            name,
            ShadowQualityStatus.INSUFFICIENT_DATA,
            sample_count,
            observed_value,
            threshold,
            f"Requires {minimum_count} exclusive matched outcomes per applicable cohort; found {sample_count}.",
        )
    return ShadowQualityGate(
        name,
        ShadowQualityStatus.PASS if observed_value <= threshold else ShadowQualityStatus.FAIL,
        sample_count,
        observed_value,
        threshold,
        pass_reason if observed_value <= threshold else fail_reason,
    )


def _minimum_gate_with_minimum_count(
    *,
    name: ShadowQualityGateName,
    sample_count: int,
    minimum_count: int,
    observed_value: float | None,
    threshold: float,
    pass_reason: str,
    fail_reason: str,
) -> ShadowQualityGate:
    if sample_count < minimum_count or observed_value is None:
        return ShadowQualityGate(
            name,
            ShadowQualityStatus.INSUFFICIENT_DATA,
            sample_count,
            observed_value,
            threshold,
            f"Requires {minimum_count} exclusive matched outcomes per source; found {sample_count}.",
        )
    return ShadowQualityGate(
        name,
        ShadowQualityStatus.PASS if observed_value >= threshold else ShadowQualityStatus.FAIL,
        sample_count,
        observed_value,
        threshold,
        pass_reason if observed_value >= threshold else fail_reason,
    )


def _reject_duplicate_outcomes(records: tuple[DreamerShadowEvaluation, ...]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.outcome_shot_id in seen:
            raise DreamerShadowQualityError("shadow quality input reuses an outcome shot")
        seen.add(record.outcome_shot_id)


def _validate_record_consistency(record: DreamerShadowEvaluation) -> None:
    if record.status == ShadowEvaluationStatus.PENDING_OUTCOME and (
        record.dreamer_match != ShadowProposalMatch.UNKNOWN
        or record.bo_match != ShadowProposalMatch.UNKNOWN
    ):
        raise DreamerShadowQualityError("pending shadow quality input contains outcome match state")
    if record.bo_proposal is None and record.bo_match != ShadowProposalMatch.UNKNOWN:
        raise DreamerShadowQualityError("shadow quality input matches a missing BO proposal")
    for label, reward in (("source", record.source_reward), ("outcome", record.outcome_reward)):
        if reward is not None and not 0.0 <= reward <= 1.0:
            raise DreamerShadowQualityError(f"shadow quality {label} reward is out of range")
    if record.source_reward is None or record.outcome_reward is None:
        if record.reward_delta is not None:
            raise DreamerShadowQualityError("shadow quality reward delta lacks source or outcome reward")
    else:
        expected_delta = record.outcome_reward - record.source_reward
        if record.reward_delta is None or abs(record.reward_delta - expected_delta) > 1e-8:
            raise DreamerShadowQualityError("shadow quality reward delta is inconsistent")


def _mean(values: Iterable[float | None]) -> float | None:
    parsed = tuple(float(value) for value in values if value is not None)
    return round(sum(parsed) / len(parsed), 8) if parsed else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 8) if denominator else None


def _evaluation_set_sha256(records: tuple[DreamerShadowEvaluation, ...]) -> str:
    payload = [record.to_dict() for record in sorted(records, key=lambda item: item.evaluation_id)]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _report_id(
    *,
    expected_context: tuple[str, str, str, str],
    checkpoint_artifact_sha256: str,
    checkpoint_inference_probe_sha256: str,
    inference_contract_id: str,
    evaluation_set_sha256: str,
    policy_version: str,
) -> str:
    canonical = "\n".join(
        (
            *expected_context,
            inference_contract_id,
            checkpoint_artifact_sha256,
            checkpoint_inference_probe_sha256,
            evaluation_set_sha256,
            policy_version,
        )
    )
    return f"shadow_quality_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:32]}"


def _sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise DreamerShadowQualityError(f"shadow quality {label} SHA-256 is invalid")

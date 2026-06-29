from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from espresso_rl.domain.shadow_contract import (
    SHADOW_INFERENCE_CONTRACT_LEGACY_V1,
    validate_shadow_inference_contract_id,
)

SHADOW_QUALITY_REPORT_FORMAT = "espresso_rl_dreamer_shadow_quality_report_v1"
SHADOW_QUALITY_REPORT_SCHEMA_VERSION = 2
SHADOW_QUALITY_POLICY_VERSION = "shadow_quality_v1"


class ShadowQualityStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


class ShadowQualityGateName(str, Enum):
    MINIMUM_EVIDENCE = "minimum_evidence"
    SAFETY_RATE = "safety_rate"
    OUTCOME_COVERAGE = "context_outcome_coverage"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    SELECTION_BALANCE = "selection_balance"
    MATCHED_OUTCOME_COMPARISON = "matched_outcome_comparison"


@dataclass(frozen=True)
class ShadowQualityPolicy:
    minimum_record_count: int = 20
    minimum_observed_count: int = 15
    minimum_safety_rate: float = 0.99
    minimum_outcome_coverage: float = 0.75
    minimum_calibration_count: int = 10
    maximum_confidence_brier_score: float = 0.25
    minimum_comparison_count_per_source: int = 10
    maximum_source_reward_mean_gap: float = 0.10
    minimum_dreamer_reward_delta_advantage: float = 0.0
    version: str = SHADOW_QUALITY_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.version != SHADOW_QUALITY_POLICY_VERSION:
            raise ValueError("shadow quality policy version is unsupported")
        for field_name in (
            "minimum_record_count",
            "minimum_observed_count",
            "minimum_calibration_count",
            "minimum_comparison_count_per_source",
        ):
            _positive_int(getattr(self, field_name), f"shadow quality policy {field_name}")
        for field_name in (
            "minimum_safety_rate",
            "minimum_outcome_coverage",
            "maximum_confidence_brier_score",
            "maximum_source_reward_mean_gap",
        ):
            value = getattr(self, field_name)
            _finite(value, f"shadow quality policy {field_name}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"shadow quality policy {field_name} must be within 0..1")
        _finite(
            self.minimum_dreamer_reward_delta_advantage,
            "shadow quality policy minimum_dreamer_reward_delta_advantage",
        )
        if (
            self.minimum_record_count,
            self.minimum_observed_count,
            self.minimum_safety_rate,
            self.minimum_outcome_coverage,
            self.minimum_calibration_count,
            self.maximum_confidence_brier_score,
            self.minimum_comparison_count_per_source,
            self.maximum_source_reward_mean_gap,
            self.minimum_dreamer_reward_delta_advantage,
        ) != (20, 15, 0.99, 0.75, 10, 0.25, 10, 0.10, 0.0):
            raise ValueError("shadow quality v1 policy thresholds are fixed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "minimum_record_count": self.minimum_record_count,
            "minimum_observed_count": self.minimum_observed_count,
            "minimum_safety_rate": self.minimum_safety_rate,
            "minimum_outcome_coverage": self.minimum_outcome_coverage,
            "minimum_calibration_count": self.minimum_calibration_count,
            "maximum_confidence_brier_score": self.maximum_confidence_brier_score,
            "minimum_comparison_count_per_source": self.minimum_comparison_count_per_source,
            "maximum_source_reward_mean_gap": self.maximum_source_reward_mean_gap,
            "minimum_dreamer_reward_delta_advantage": self.minimum_dreamer_reward_delta_advantage,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ShadowQualityPolicy":
        if not isinstance(value, dict):
            raise ValueError("shadow quality policy must be an object")
        expected = {
            "version",
            "minimum_record_count",
            "minimum_observed_count",
            "minimum_safety_rate",
            "minimum_outcome_coverage",
            "minimum_calibration_count",
            "maximum_confidence_brier_score",
            "minimum_comparison_count_per_source",
            "maximum_source_reward_mean_gap",
            "minimum_dreamer_reward_delta_advantage",
        }
        _exact_fields(value, expected, "shadow quality policy")
        return cls(**value)


@dataclass(frozen=True)
class ShadowQualityGate:
    name: ShadowQualityGateName
    status: ShadowQualityStatus
    sample_count: int
    observed_value: float | None
    threshold: float
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", ShadowQualityGateName(self.name))
        object.__setattr__(self, "status", ShadowQualityStatus(self.status))
        _nonnegative_int(self.sample_count, "shadow quality gate sample_count")
        if self.observed_value is not None:
            _finite(self.observed_value, "shadow quality gate observed_value")
        _finite(self.threshold, "shadow quality gate threshold")
        _safe_text(self.reason, "shadow quality gate reason", maximum=320)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "status": self.status.value,
            "sample_count": self.sample_count,
            "observed_value": self.observed_value,
            "threshold": self.threshold,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ShadowQualityGate":
        if not isinstance(value, dict):
            raise ValueError("shadow quality gate must be an object")
        expected = {"name", "status", "sample_count", "observed_value", "threshold", "reason"}
        _exact_fields(value, expected, "shadow quality gate")
        return cls(**value)


@dataclass(frozen=True)
class DreamerShadowQualityReport:
    report_id: str
    generated_at: int
    checkpoint_artifact_sha256: str
    checkpoint_inference_probe_sha256: str
    inference_contract_id: str
    install_id: str
    machine_id: str
    bean_context_id: str
    grinder_context_id: str
    evaluation_set_sha256: str
    source_record_count: int
    evaluated_record_count: int
    stale_checkpoint_record_count: int
    pending_count: int
    observed_count: int
    safe_proposal_count: int
    unsafe_proposal_count: int
    both_matched_count: int
    dreamer_only_matched_count: int
    bo_only_matched_count: int
    partial_match_count: int
    unmatched_count: int
    safety_rate: float | None
    outcome_coverage: float | None
    confidence_brier_score: float | None
    dreamer_source_reward_mean: float | None
    bo_source_reward_mean: float | None
    source_reward_mean_gap: float | None
    dreamer_reward_delta_mean: float | None
    bo_reward_delta_mean: float | None
    dreamer_reward_delta_advantage: float | None
    gates: tuple[ShadowQualityGate, ...]
    overall_status: ShadowQualityStatus
    policy: ShadowQualityPolicy = field(default_factory=ShadowQualityPolicy)
    observational_only: bool = True
    shadow_only: bool = True
    recommendation_enabled: bool = False
    machine_control_enabled: bool = False
    format: str = SHADOW_QUALITY_REPORT_FORMAT
    schema_version: int = SHADOW_QUALITY_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.format != SHADOW_QUALITY_REPORT_FORMAT or self.schema_version != SHADOW_QUALITY_REPORT_SCHEMA_VERSION:
            raise ValueError("shadow quality report format or schema version is unsupported")
        for field_name in (
            "report_id",
            "install_id",
            "machine_id",
            "bean_context_id",
            "grinder_context_id",
        ):
            _safe_text(getattr(self, field_name), f"shadow quality report {field_name}", maximum=180)
        _positive_int(self.generated_at, "shadow quality report generated_at")
        _sha256(self.checkpoint_artifact_sha256, "checkpoint_artifact_sha256")
        _sha256(self.checkpoint_inference_probe_sha256, "checkpoint_inference_probe_sha256")
        object.__setattr__(
            self,
            "inference_contract_id",
            validate_shadow_inference_contract_id(self.inference_contract_id),
        )
        _sha256(self.evaluation_set_sha256, "evaluation_set_sha256")
        count_fields = (
            "source_record_count",
            "evaluated_record_count",
            "stale_checkpoint_record_count",
            "pending_count",
            "observed_count",
            "safe_proposal_count",
            "unsafe_proposal_count",
            "both_matched_count",
            "dreamer_only_matched_count",
            "bo_only_matched_count",
            "partial_match_count",
            "unmatched_count",
        )
        for field_name in count_fields:
            _nonnegative_int(getattr(self, field_name), f"shadow quality report {field_name}")
        if self.source_record_count != self.evaluated_record_count + self.stale_checkpoint_record_count:
            raise ValueError("shadow quality report source records are inconsistent")
        if self.evaluated_record_count != self.pending_count + self.observed_count:
            raise ValueError("shadow quality report outcome counts are inconsistent")
        if self.evaluated_record_count != self.safe_proposal_count + self.unsafe_proposal_count:
            raise ValueError("shadow quality report safety counts are inconsistent")
        if self.observed_count != sum(
            (
                self.both_matched_count,
                self.dreamer_only_matched_count,
                self.bo_only_matched_count,
                self.partial_match_count,
                self.unmatched_count,
            )
        ):
            raise ValueError("shadow quality report match cohorts are inconsistent")
        for field_name in (
            "safety_rate",
            "outcome_coverage",
            "confidence_brier_score",
            "dreamer_source_reward_mean",
            "bo_source_reward_mean",
            "source_reward_mean_gap",
            "dreamer_reward_delta_mean",
            "bo_reward_delta_mean",
            "dreamer_reward_delta_advantage",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _finite(value, f"shadow quality report {field_name}")
        for field_name in (
            "safety_rate",
            "outcome_coverage",
            "confidence_brier_score",
            "dreamer_source_reward_mean",
            "bo_source_reward_mean",
            "source_reward_mean_gap",
        ):
            value = getattr(self, field_name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"shadow quality report {field_name} must be within 0..1")
        if not isinstance(self.policy, ShadowQualityPolicy):
            raise ValueError("shadow quality report policy is invalid")
        object.__setattr__(self, "overall_status", ShadowQualityStatus(self.overall_status))
        expected_names = tuple(ShadowQualityGateName)
        actual_names = tuple(gate.name for gate in self.gates)
        if actual_names != expected_names:
            raise ValueError("shadow quality report gates are missing, duplicated, or out of order")
        derived_status = overall_shadow_quality_status(self.gates)
        if self.overall_status != derived_status:
            raise ValueError("shadow quality report overall status is inconsistent")
        if (
            self.observational_only is not True
            or self.shadow_only is not True
            or self.recommendation_enabled is not False
            or self.machine_control_enabled is not False
        ):
            raise ValueError("shadow quality report must remain observational and shadow-only")

    @property
    def context_key(self) -> tuple[str, str, str, str]:
        return (self.install_id, self.machine_id, self.bean_context_id, self.grinder_context_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "checkpoint_artifact_sha256": self.checkpoint_artifact_sha256,
            "checkpoint_inference_probe_sha256": self.checkpoint_inference_probe_sha256,
            "inference_contract_id": self.inference_contract_id,
            "install_id": self.install_id,
            "machine_id": self.machine_id,
            "bean_context_id": self.bean_context_id,
            "grinder_context_id": self.grinder_context_id,
            "evaluation_set_sha256": self.evaluation_set_sha256,
            "source_record_count": self.source_record_count,
            "evaluated_record_count": self.evaluated_record_count,
            "stale_checkpoint_record_count": self.stale_checkpoint_record_count,
            "pending_count": self.pending_count,
            "observed_count": self.observed_count,
            "safe_proposal_count": self.safe_proposal_count,
            "unsafe_proposal_count": self.unsafe_proposal_count,
            "both_matched_count": self.both_matched_count,
            "dreamer_only_matched_count": self.dreamer_only_matched_count,
            "bo_only_matched_count": self.bo_only_matched_count,
            "partial_match_count": self.partial_match_count,
            "unmatched_count": self.unmatched_count,
            "safety_rate": self.safety_rate,
            "outcome_coverage": self.outcome_coverage,
            "confidence_brier_score": self.confidence_brier_score,
            "dreamer_source_reward_mean": self.dreamer_source_reward_mean,
            "bo_source_reward_mean": self.bo_source_reward_mean,
            "source_reward_mean_gap": self.source_reward_mean_gap,
            "dreamer_reward_delta_mean": self.dreamer_reward_delta_mean,
            "bo_reward_delta_mean": self.bo_reward_delta_mean,
            "dreamer_reward_delta_advantage": self.dreamer_reward_delta_advantage,
            "gates": [gate.to_dict() for gate in self.gates],
            "overall_status": self.overall_status.value,
            "policy": self.policy.to_dict(),
            "observational_only": self.observational_only,
            "shadow_only": self.shadow_only,
            "recommendation_enabled": self.recommendation_enabled,
            "machine_control_enabled": self.machine_control_enabled,
        }

    @classmethod
    def from_dict(cls, value: object) -> "DreamerShadowQualityReport":
        if not isinstance(value, dict):
            raise ValueError("shadow quality report must be an object")
        if "inference_contract_id" not in value and value.get("schema_version") == 1:
            value = {
                **value,
                "schema_version": SHADOW_QUALITY_REPORT_SCHEMA_VERSION,
                "inference_contract_id": SHADOW_INFERENCE_CONTRACT_LEGACY_V1,
            }
        expected = {
            "format",
            "schema_version",
            "report_id",
            "generated_at",
            "checkpoint_artifact_sha256",
            "checkpoint_inference_probe_sha256",
            "inference_contract_id",
            "install_id",
            "machine_id",
            "bean_context_id",
            "grinder_context_id",
            "evaluation_set_sha256",
            "source_record_count",
            "evaluated_record_count",
            "stale_checkpoint_record_count",
            "pending_count",
            "observed_count",
            "safe_proposal_count",
            "unsafe_proposal_count",
            "both_matched_count",
            "dreamer_only_matched_count",
            "bo_only_matched_count",
            "partial_match_count",
            "unmatched_count",
            "safety_rate",
            "outcome_coverage",
            "confidence_brier_score",
            "dreamer_source_reward_mean",
            "bo_source_reward_mean",
            "source_reward_mean_gap",
            "dreamer_reward_delta_mean",
            "bo_reward_delta_mean",
            "dreamer_reward_delta_advantage",
            "gates",
            "overall_status",
            "policy",
            "observational_only",
            "shadow_only",
            "recommendation_enabled",
            "machine_control_enabled",
        }
        _exact_fields(value, expected, "shadow quality report")
        gate_values = value["gates"]
        if not isinstance(gate_values, list):
            raise ValueError("shadow quality report gates must be a list")
        return cls(
            **{
                key: value[key]
                for key in expected
                if key not in {"gates", "policy"}
            },
            gates=tuple(ShadowQualityGate.from_dict(item) for item in gate_values),
            policy=ShadowQualityPolicy.from_dict(value["policy"]),
        )

    def status_summary(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "checkpoint_artifact_sha256": self.checkpoint_artifact_sha256,
            "checkpoint_inference_probe_sha256": self.checkpoint_inference_probe_sha256,
            "inference_contract_id": self.inference_contract_id,
            "overall_status": self.overall_status.value,
            "evaluated_record_count": self.evaluated_record_count,
            "stale_checkpoint_record_count": self.stale_checkpoint_record_count,
            "observed_count": self.observed_count,
            "safety_rate": self.safety_rate,
            "outcome_coverage": self.outcome_coverage,
            "dreamer_reward_delta_advantage": self.dreamer_reward_delta_advantage,
            "gates": [gate.to_dict() for gate in self.gates],
            "observational_only": True,
            "shadow_only": True,
            "recommendation_enabled": False,
            "machine_control_enabled": False,
        }


def overall_shadow_quality_status(gates: tuple[ShadowQualityGate, ...]) -> ShadowQualityStatus:
    if any(gate.status == ShadowQualityStatus.FAIL for gate in gates):
        return ShadowQualityStatus.FAIL
    if any(gate.status == ShadowQualityStatus.INSUFFICIENT_DATA for gate in gates):
        return ShadowQualityStatus.INSUFFICIENT_DATA
    return ShadowQualityStatus.PASS


def _finite(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")


def _positive_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _nonnegative_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")


def _safe_text(value: object, label: str, *, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} is invalid")


def _sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"shadow quality report {label} is invalid")


def _exact_fields(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields are invalid")

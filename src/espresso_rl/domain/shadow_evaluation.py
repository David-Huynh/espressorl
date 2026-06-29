from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from espresso_rl.domain.shadow_contract import (
    SHADOW_INFERENCE_CONTRACT_LEGACY_V1,
    validate_shadow_inference_contract_id,
)

SHADOW_EVALUATION_FORMAT = "espresso_rl_dreamer_shadow_evaluation_v1"
SHADOW_EVALUATION_SCHEMA_VERSION = 2


class ShadowEvaluationStatus(str, Enum):
    PENDING_OUTCOME = "pending_outcome"
    OUTCOME_OBSERVED = "outcome_observed"


class ShadowProposalMatch(str, Enum):
    UNKNOWN = "unknown"
    MATCHED = "matched"
    PARTIALLY_MATCHED = "partially_matched"
    NOT_MATCHED = "not_matched"


@dataclass(frozen=True)
class ShadowRecipeProposal:
    source: str
    grind_delta_steps_from_current: int
    projected_relative_step_from_reference: float
    projected_relative_grind_um_from_reference: float
    next_dose_g: float
    target_yield_g: float
    target_ratio: float
    confidence: float
    safety_valid: bool
    safety_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source not in {"dreamer_v3", "bayesian_optimization"}:
            raise ValueError("shadow proposal source is invalid")
        if isinstance(self.grind_delta_steps_from_current, bool) or not isinstance(
            self.grind_delta_steps_from_current, int
        ):
            raise ValueError("shadow proposal grind delta must be integer relative steps")
        for field_name in (
            "projected_relative_step_from_reference",
            "projected_relative_grind_um_from_reference",
            "next_dose_g",
            "target_yield_g",
            "target_ratio",
            "confidence",
        ):
            _finite(getattr(self, field_name), f"shadow proposal {field_name}")
        if self.next_dose_g <= 0 or self.target_yield_g <= 0 or self.target_ratio <= 0:
            raise ValueError("shadow proposal recipe values must be positive")
        if abs(self.target_ratio - self.target_yield_g / self.next_dose_g) > 0.05:
            raise ValueError("shadow proposal target ratio is inconsistent")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("shadow proposal confidence is invalid")
        if not isinstance(self.safety_valid, bool):
            raise ValueError("shadow proposal safety_valid must be boolean")
        if self.safety_valid and self.safety_errors:
            raise ValueError("safe shadow proposal must not contain safety errors")
        if not self.safety_valid and not self.safety_errors:
            raise ValueError("unsafe shadow proposal must contain safety errors")
        for error in self.safety_errors:
            _safe_text(error, "shadow proposal safety error", maximum=240)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "grind_delta_steps_from_current": self.grind_delta_steps_from_current,
            "projected_relative_step_from_reference": self.projected_relative_step_from_reference,
            "projected_relative_grind_um_from_reference": self.projected_relative_grind_um_from_reference,
            "next_dose_g": self.next_dose_g,
            "target_yield_g": self.target_yield_g,
            "target_ratio": self.target_ratio,
            "confidence": self.confidence,
            "safety_valid": self.safety_valid,
            "safety_errors": list(self.safety_errors),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ShadowRecipeProposal":
        if not isinstance(value, dict):
            raise ValueError("shadow proposal must be an object")
        _exact_fields(
            value,
            {
                "source",
                "grind_delta_steps_from_current",
                "projected_relative_step_from_reference",
                "projected_relative_grind_um_from_reference",
                "next_dose_g",
                "target_yield_g",
                "target_ratio",
                "confidence",
                "safety_valid",
                "safety_errors",
            },
            "shadow proposal",
        )
        errors = value["safety_errors"]
        if not isinstance(errors, list) or any(not isinstance(item, str) for item in errors):
            raise ValueError("shadow proposal safety_errors must be a string list")
        return cls(
            source=value["source"],
            grind_delta_steps_from_current=value["grind_delta_steps_from_current"],
            projected_relative_step_from_reference=value["projected_relative_step_from_reference"],
            projected_relative_grind_um_from_reference=value["projected_relative_grind_um_from_reference"],
            next_dose_g=value["next_dose_g"],
            target_yield_g=value["target_yield_g"],
            target_ratio=value["target_ratio"],
            confidence=value["confidence"],
            safety_valid=value["safety_valid"],
            safety_errors=tuple(errors),
        )


@dataclass(frozen=True)
class DreamerShadowEvaluation:
    evaluation_id: str
    created_at: int
    updated_at: int
    checkpoint_artifact_sha256: str
    checkpoint_inference_probe_sha256: str
    inference_contract_id: str
    install_id: str
    machine_id: str
    bean_context_id: str
    grinder_context_id: str
    source_training_row_id: int
    source_shot_id: str
    source_timestamp: int
    microns_per_step: float
    step_direction: str
    current_relative_step_from_reference: float
    current_dose_g: float
    current_target_yield_g: float
    current_target_ratio: float
    dreamer_proposal: ShadowRecipeProposal
    bo_proposal: ShadowRecipeProposal | None
    source_reward: float | None
    status: ShadowEvaluationStatus = ShadowEvaluationStatus.PENDING_OUTCOME
    outcome_shot_id: str | None = None
    outcome_timestamp: int | None = None
    outcome_relative_step_from_reference: float | None = None
    outcome_dose_g: float | None = None
    outcome_target_yield_g: float | None = None
    outcome_reward: float | None = None
    reward_delta: float | None = None
    dreamer_match: ShadowProposalMatch = ShadowProposalMatch.UNKNOWN
    bo_match: ShadowProposalMatch = ShadowProposalMatch.UNKNOWN
    format: str = SHADOW_EVALUATION_FORMAT
    schema_version: int = SHADOW_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.format != SHADOW_EVALUATION_FORMAT or self.schema_version != SHADOW_EVALUATION_SCHEMA_VERSION:
            raise ValueError("shadow evaluation format or schema version is unsupported")
        for field_name in (
            "evaluation_id",
            "install_id",
            "machine_id",
            "bean_context_id",
            "grinder_context_id",
            "source_shot_id",
        ):
            _safe_text(getattr(self, field_name), f"shadow evaluation {field_name}", maximum=180)
        _sha256(self.checkpoint_artifact_sha256, "checkpoint_artifact_sha256")
        _sha256(self.checkpoint_inference_probe_sha256, "checkpoint_inference_probe_sha256")
        object.__setattr__(
            self,
            "inference_contract_id",
            validate_shadow_inference_contract_id(self.inference_contract_id),
        )
        for field_name in ("created_at", "updated_at", "source_training_row_id", "source_timestamp"):
            _positive_int(getattr(self, field_name), f"shadow evaluation {field_name}")
        if self.updated_at < self.created_at:
            raise ValueError("shadow evaluation updated_at cannot precede created_at")
        _finite(self.microns_per_step, "shadow evaluation microns_per_step")
        if self.microns_per_step <= 0:
            raise ValueError("shadow evaluation microns_per_step must be positive")
        if self.step_direction not in {"higher_is_finer", "higher_is_coarser"}:
            raise ValueError("shadow evaluation step direction is invalid")
        for field_name in (
            "current_relative_step_from_reference",
            "current_dose_g",
            "current_target_yield_g",
            "current_target_ratio",
        ):
            _finite(getattr(self, field_name), f"shadow evaluation {field_name}")
        if not isinstance(self.dreamer_proposal, ShadowRecipeProposal) or self.dreamer_proposal.source != "dreamer_v3":
            raise ValueError("shadow evaluation Dreamer proposal is invalid")
        if self.bo_proposal is not None and self.bo_proposal.source != "bayesian_optimization":
            raise ValueError("shadow evaluation BO proposal is invalid")
        object.__setattr__(self, "status", ShadowEvaluationStatus(self.status))
        object.__setattr__(self, "dreamer_match", ShadowProposalMatch(self.dreamer_match))
        object.__setattr__(self, "bo_match", ShadowProposalMatch(self.bo_match))
        for field_name in (
            "source_reward",
            "outcome_relative_step_from_reference",
            "outcome_dose_g",
            "outcome_target_yield_g",
            "outcome_reward",
            "reward_delta",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _finite(value, f"shadow evaluation {field_name}")
        if self.status == ShadowEvaluationStatus.PENDING_OUTCOME:
            if any(
                value is not None
                for value in (
                    self.outcome_shot_id,
                    self.outcome_timestamp,
                    self.outcome_relative_step_from_reference,
                    self.outcome_dose_g,
                    self.outcome_target_yield_g,
                    self.outcome_reward,
                    self.reward_delta,
                )
            ):
                raise ValueError("pending shadow evaluation must not contain an outcome")
        else:
            _safe_text(self.outcome_shot_id, "shadow evaluation outcome_shot_id", maximum=180)
            _positive_int(self.outcome_timestamp, "shadow evaluation outcome_timestamp")
            if self.outcome_timestamp <= self.source_timestamp:
                raise ValueError("shadow evaluation outcome must be later than its source shot")
            for field_name in (
                "outcome_relative_step_from_reference",
                "outcome_dose_g",
                "outcome_target_yield_g",
            ):
                if getattr(self, field_name) is None:
                    raise ValueError(f"observed shadow evaluation requires {field_name}")

    @property
    def context_key(self) -> tuple[str, str, str, str]:
        return (self.install_id, self.machine_id, self.bean_context_id, self.grinder_context_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "checkpoint_artifact_sha256": self.checkpoint_artifact_sha256,
            "checkpoint_inference_probe_sha256": self.checkpoint_inference_probe_sha256,
            "inference_contract_id": self.inference_contract_id,
            "install_id": self.install_id,
            "machine_id": self.machine_id,
            "bean_context_id": self.bean_context_id,
            "grinder_context_id": self.grinder_context_id,
            "source_training_row_id": self.source_training_row_id,
            "source_shot_id": self.source_shot_id,
            "source_timestamp": self.source_timestamp,
            "microns_per_step": self.microns_per_step,
            "step_direction": self.step_direction,
            "current_relative_step_from_reference": self.current_relative_step_from_reference,
            "current_dose_g": self.current_dose_g,
            "current_target_yield_g": self.current_target_yield_g,
            "current_target_ratio": self.current_target_ratio,
            "dreamer_proposal": self.dreamer_proposal.to_dict(),
            "bo_proposal": self.bo_proposal.to_dict() if self.bo_proposal is not None else None,
            "source_reward": self.source_reward,
            "status": self.status.value,
            "outcome_shot_id": self.outcome_shot_id,
            "outcome_timestamp": self.outcome_timestamp,
            "outcome_relative_step_from_reference": self.outcome_relative_step_from_reference,
            "outcome_dose_g": self.outcome_dose_g,
            "outcome_target_yield_g": self.outcome_target_yield_g,
            "outcome_reward": self.outcome_reward,
            "reward_delta": self.reward_delta,
            "dreamer_match": self.dreamer_match.value,
            "bo_match": self.bo_match.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> "DreamerShadowEvaluation":
        if not isinstance(value, dict):
            raise ValueError("shadow evaluation must be an object")
        if "inference_contract_id" not in value and value.get("schema_version") == 1:
            value = {
                **value,
                "schema_version": SHADOW_EVALUATION_SCHEMA_VERSION,
                "inference_contract_id": SHADOW_INFERENCE_CONTRACT_LEGACY_V1,
            }
        expected = {
            "format",
            "schema_version",
            "evaluation_id",
            "created_at",
            "updated_at",
            "checkpoint_artifact_sha256",
            "checkpoint_inference_probe_sha256",
            "inference_contract_id",
            "install_id",
            "machine_id",
            "bean_context_id",
            "grinder_context_id",
            "source_training_row_id",
            "source_shot_id",
            "source_timestamp",
            "microns_per_step",
            "step_direction",
            "current_relative_step_from_reference",
            "current_dose_g",
            "current_target_yield_g",
            "current_target_ratio",
            "dreamer_proposal",
            "bo_proposal",
            "source_reward",
            "status",
            "outcome_shot_id",
            "outcome_timestamp",
            "outcome_relative_step_from_reference",
            "outcome_dose_g",
            "outcome_target_yield_g",
            "outcome_reward",
            "reward_delta",
            "dreamer_match",
            "bo_match",
        }
        _exact_fields(value, expected, "shadow evaluation")
        bo_value = value["bo_proposal"]
        return cls(
            **{
                key: value[key]
                for key in expected
                if key not in {"dreamer_proposal", "bo_proposal"}
            },
            dreamer_proposal=ShadowRecipeProposal.from_dict(value["dreamer_proposal"]),
            bo_proposal=ShadowRecipeProposal.from_dict(bo_value) if bo_value is not None else None,
        )


def resolve_shadow_evaluation(
    evaluation: DreamerShadowEvaluation,
    *,
    outcome_shot_id: str,
    outcome_timestamp: int,
    relative_step_from_reference: float,
    dose_g: float,
    target_yield_g: float,
    reward: float | None,
    updated_at: int,
) -> DreamerShadowEvaluation:
    if evaluation.status != ShadowEvaluationStatus.PENDING_OUTCOME:
        raise ValueError("shadow evaluation outcome is already recorded")
    dreamer_match = proposal_match(
        evaluation.dreamer_proposal,
        relative_step_from_reference=relative_step_from_reference,
        dose_g=dose_g,
        target_yield_g=target_yield_g,
    )
    bo_match = (
        proposal_match(
            evaluation.bo_proposal,
            relative_step_from_reference=relative_step_from_reference,
            dose_g=dose_g,
            target_yield_g=target_yield_g,
        )
        if evaluation.bo_proposal is not None
        else ShadowProposalMatch.UNKNOWN
    )
    reward_delta = (
        float(reward) - evaluation.source_reward
        if reward is not None and evaluation.source_reward is not None
        else None
    )
    return replace(
        evaluation,
        updated_at=updated_at,
        status=ShadowEvaluationStatus.OUTCOME_OBSERVED,
        outcome_shot_id=outcome_shot_id,
        outcome_timestamp=outcome_timestamp,
        outcome_relative_step_from_reference=relative_step_from_reference,
        outcome_dose_g=dose_g,
        outcome_target_yield_g=target_yield_g,
        outcome_reward=reward,
        reward_delta=reward_delta,
        dreamer_match=dreamer_match,
        bo_match=bo_match,
    )


def proposal_match(
    proposal: ShadowRecipeProposal,
    *,
    relative_step_from_reference: float,
    dose_g: float,
    target_yield_g: float,
) -> ShadowProposalMatch:
    matches = (
        abs(proposal.projected_relative_step_from_reference - relative_step_from_reference) <= 0.25,
        abs(proposal.next_dose_g - dose_g) <= 0.15,
        abs(proposal.target_yield_g - target_yield_g) <= 0.5,
    )
    if all(matches):
        return ShadowProposalMatch.MATCHED
    if any(matches):
        return ShadowProposalMatch.PARTIALLY_MATCHED
    return ShadowProposalMatch.NOT_MATCHED


def _finite(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")


def _positive_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


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
        raise ValueError(f"shadow evaluation {label} is invalid")


def _exact_fields(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{label} fields are invalid")

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

from espresso_rl.application.dreamer_recommendations import (
    DreamerRecommendationError,
    dreamer_episode_batch_from_training_rows,
    dreamer_recipe_proposal,
)
from espresso_rl.application.dreamer_shadow_inference import DreamerShadowInferenceSession
from espresso_rl.domain.dreamer_actions import DreamerActionCandidate, validate_dreamer_action
from espresso_rl.domain.models import (
    GrinderStepDirection,
    Recipe,
    Recommendation,
    RecommendationMode,
    SafetyBounds,
)
from espresso_rl.domain.shadow_evaluation import (
    DreamerShadowEvaluation,
    ShadowEvaluationStatus,
    ShadowProposalMatch,
    ShadowRecipeProposal,
    resolve_shadow_evaluation,
)
from espresso_rl.domain.training import validate_training_transition
from espresso_rl.ports.shadow_evaluations import ShadowEvaluationRepository


class DreamerShadowEvaluationError(ValueError):
    pass


@dataclass(frozen=True)
class DreamerShadowEvaluationResult:
    evaluation: DreamerShadowEvaluation
    resolved_previous: DreamerShadowEvaluation | None = None
    created: bool = True


class DreamerShadowEvaluationService:
    def __init__(
        self,
        *,
        session: DreamerShadowInferenceSession,
        repository: ShadowEvaluationRepository,
        safety_bounds: SafetyBounds | None = None,
        clock: Callable[[], int],
    ) -> None:
        if not session.status.parity_verified:
            raise ValueError("Dreamer shadow evaluation requires a parity-verified session")
        self._session = session
        self._repository = repository
        self._safety_bounds = safety_bounds or SafetyBounds()
        self._clock = clock

    def context_summary(
        self,
        *,
        install_id: str,
        machine_id: str,
        bean_context_id: str,
        grinder_context_id: str,
        limit: int = 1000,
    ) -> dict[str, Any]:
        records = self._repository.list_context(
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            grinder_context_id=grinder_context_id,
            inference_contract_id=self._session.status.inference_contract_id,
            limit=limit,
        )
        observed = [record for record in records if record.status == ShadowEvaluationStatus.OUTCOME_OBSERVED]
        dreamer_followed_reward_deltas = [
            record.reward_delta
            for record in observed
            if record.dreamer_match == ShadowProposalMatch.MATCHED and record.reward_delta is not None
        ]
        bo_followed_reward_deltas = [
            record.reward_delta
            for record in observed
            if record.bo_match == ShadowProposalMatch.MATCHED and record.reward_delta is not None
        ]
        return {
            "inference_contract_id": self._session.status.inference_contract_id,
            "record_count": len(records),
            "pending_count": len(records) - len(observed),
            "observed_count": len(observed),
            "safe_proposal_count": sum(record.dreamer_proposal.safety_valid for record in records),
            "unsafe_proposal_count": sum(not record.dreamer_proposal.safety_valid for record in records),
            "dreamer_matched_count": sum(
                record.dreamer_match == ShadowProposalMatch.MATCHED for record in observed
            ),
            "bo_matched_count": sum(record.bo_match == ShadowProposalMatch.MATCHED for record in observed),
            "mean_dreamer_followed_reward_delta": (
                round(sum(dreamer_followed_reward_deltas) / len(dreamer_followed_reward_deltas), 6)
                if dreamer_followed_reward_deltas
                else None
            ),
            "mean_bo_followed_reward_delta": (
                round(sum(bo_followed_reward_deltas) / len(bo_followed_reward_deltas), 6)
                if bo_followed_reward_deltas
                else None
            ),
            "shadow_only": True,
        }

    def evaluate_transition(
        self,
        transition: dict[str, Any],
        *,
        bo_recommendation: Recommendation | None = None,
        context_transitions: list[dict[str, Any]] | None = None,
    ) -> DreamerShadowEvaluationResult:
        errors = validate_training_transition(transition)
        if errors:
            raise DreamerShadowEvaluationError(f"shadow transition is invalid: {'; '.join(errors[:10])}")
        context = transition["context"]
        source = transition["source"]
        action = transition["action"]
        observation = transition["observation"]
        reward = transition["reward"]
        install_id = _required_context(source.get("install_id"), "install_id")
        machine_id = _required_context(context.get("machine_id"), "machine_id")
        bean_context_id = _required_context(context.get("bean_context_id"), "bean_context_id")
        grinder_context_id = _required_context(context.get("grinder_context_id"), "grinder_context_id")
        now = self._clock()
        if isinstance(now, bool) or not isinstance(now, int) or now <= 0:
            raise DreamerShadowEvaluationError("shadow evaluation clock must return a positive integer")

        current_recipe = Recipe(
            relative_grind_steps_from_reference=float(action["relative_grind_steps_from_reference"]),
            microns_per_step=float(context["microns_per_step"]),
            dose_g=float(action["dose_g"]),
            target_yield_g=float(action["target_yield_g"]),
            target_ratio=float(action["target_ratio"]),
            grinder_step_direction=GrinderStepDirection(context["step_direction"]),
        )
        resolved_previous = self._resolve_pending(
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            grinder_context_id=grinder_context_id,
            transition=transition,
            now=now,
        )
        evaluation_id = _evaluation_id(
            checkpoint_sha256=self._session.status.checkpoint_artifact_sha256,
            inference_contract_id=self._session.status.inference_contract_id,
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            grinder_context_id=grinder_context_id,
            shot_id=str(observation["shot_id"]),
        )
        existing = self._repository.get(evaluation_id)
        if existing is not None:
            return DreamerShadowEvaluationResult(
                evaluation=existing,
                resolved_previous=resolved_previous,
                created=False,
            )

        batch, current_episode_index = self._episode_batch(
            transition,
            context_transitions=context_transitions or [],
        )
        dreamer_proposal = self._dreamer_proposal(
            batch,
            current_episode_index=current_episode_index,
            current_recipe=current_recipe,
        )
        bo_proposal = self._cpbo_proposal(
            bo_recommendation,
            current_recipe=current_recipe,
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            grinder_context_id=grinder_context_id,
            source_shot_id=str(observation["shot_id"]),
        )
        evaluation = DreamerShadowEvaluation(
            evaluation_id=evaluation_id,
            created_at=now,
            updated_at=now,
            checkpoint_artifact_sha256=self._session.status.checkpoint_artifact_sha256,
            checkpoint_inference_probe_sha256=self._session.status.inference_probe_sha256,
            inference_contract_id=self._session.status.inference_contract_id,
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            grinder_context_id=grinder_context_id,
            source_training_row_id=int(transition["training_row_id"]),
            source_shot_id=str(observation["shot_id"]),
            source_timestamp=int(observation["timestamp"]),
            microns_per_step=current_recipe.microns_per_step,
            step_direction=current_recipe.grinder_step_direction.value,
            current_relative_step_from_reference=current_recipe.relative_grind_steps_from_reference,
            current_dose_g=current_recipe.dose_g,
            current_target_yield_g=current_recipe.target_yield_g,
            current_target_ratio=float(current_recipe.target_ratio),
            dreamer_proposal=dreamer_proposal,
            bo_proposal=bo_proposal,
            source_reward=_optional_reward(reward.get("reward")),
        )
        self._repository.upsert(evaluation)
        return DreamerShadowEvaluationResult(
            evaluation=evaluation,
            resolved_previous=resolved_previous,
        )

    def _episode_batch(
        self,
        transition: dict[str, Any],
        *,
        context_transitions: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], int]:
        try:
            return dreamer_episode_batch_from_training_rows(
                self._session,
                transition,
                context_transitions=context_transitions,
            )
        except DreamerRecommendationError as exc:
            raise DreamerShadowEvaluationError(str(exc)) from exc

    def _dreamer_proposal(
        self,
        batch: dict[str, Any],
        *,
        current_episode_index: int,
        current_recipe: Recipe,
    ) -> ShadowRecipeProposal:
        try:
            proposal = dreamer_recipe_proposal(
                self._session,
                batch,
                current_episode_index=current_episode_index,
                current_recipe=current_recipe,
                safety_bounds=self._safety_bounds,
                reason="DreamerV3 shadow proposal.",
            )
        except DreamerRecommendationError as exc:
            raise DreamerShadowEvaluationError(str(exc)) from exc
        return _proposal(
            source="dreamer_v3",
            current_recipe=current_recipe,
            grind_delta_steps=proposal.grind_delta_steps_from_current,
            next_dose_g=proposal.next_dose_g,
            target_yield_g=proposal.target_yield_g,
            confidence=proposal.confidence,
            safety_errors=proposal.safety_errors,
        )

    def _cpbo_proposal(
        self,
        recommendation: Recommendation | None,
        *,
        current_recipe: Recipe,
        install_id: str,
        machine_id: str,
        bean_context_id: str,
        grinder_context_id: str,
        source_shot_id: str,
    ) -> ShadowRecipeProposal | None:
        if recommendation is None:
            return None
        if (
            recommendation.install_id != install_id
            or recommendation.machine_id != machine_id
            or recommendation.bean_context_id != bean_context_id
            or recommendation.grinder_context_id != grinder_context_id
        ):
            raise DreamerShadowEvaluationError("CPBO comparison recommendation context does not match transition")
        if recommendation.source_shot_id not in {None, source_shot_id}:
            raise DreamerShadowEvaluationError("CPBO comparison recommendation source shot does not match transition")
        if recommendation.mode in {RecommendationMode.DREAMER_CANDIDATE, RecommendationMode.DREAMER_ACTIVE}:
            raise DreamerShadowEvaluationError("Dreamer recommendation cannot be used as the CPBO comparator")
        safety_errors: tuple[str, ...] = ()
        try:
            candidate = DreamerActionCandidate(
                grind_delta_steps_from_current=recommendation.grind_delta_steps_from_current,
                next_dose_g=recommendation.next_dose_g,
                target_yield_g=recommendation.target_yield_g,
                target_ratio=recommendation.target_ratio,
                confidence=recommendation.confidence,
                reason="CPBO shadow comparator.",
            )
            validate_dreamer_action(candidate, current=current_recipe, bounds=self._safety_bounds)
        except ValueError as exc:
            safety_errors = (str(exc),)
        return _proposal(
            source="cpbo",
            current_recipe=current_recipe,
            grind_delta_steps=recommendation.grind_delta_steps_from_current,
            next_dose_g=recommendation.next_dose_g,
            target_yield_g=recommendation.target_yield_g,
            confidence=recommendation.confidence,
            safety_errors=safety_errors,
        )

    def _resolve_pending(
        self,
        *,
        install_id: str,
        machine_id: str,
        bean_context_id: str,
        grinder_context_id: str,
        transition: dict[str, Any],
        now: int,
    ) -> DreamerShadowEvaluation | None:
        pending = self._repository.get_pending(
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            grinder_context_id=grinder_context_id,
            inference_contract_id=self._session.status.inference_contract_id,
        )
        observation = transition["observation"]
        if (
            pending is None
            or pending.source_shot_id == observation["shot_id"]
            or int(observation["timestamp"]) <= pending.source_timestamp
        ):
            return None
        action = transition["action"]
        resolved = resolve_shadow_evaluation(
            pending,
            outcome_shot_id=str(observation["shot_id"]),
            outcome_timestamp=int(observation["timestamp"]),
            relative_step_from_reference=float(action["relative_grind_steps_from_reference"]),
            dose_g=float(action["dose_g"]),
            target_yield_g=float(action["target_yield_g"]),
            reward=_optional_reward(transition["reward"].get("reward")),
            updated_at=now,
        )
        self._repository.upsert(resolved)
        return resolved


def _proposal(
    *,
    source: str,
    current_recipe: Recipe,
    grind_delta_steps: int,
    next_dose_g: float,
    target_yield_g: float,
    confidence: float,
    safety_errors: tuple[str, ...],
) -> ShadowRecipeProposal:
    projected_step = current_recipe.relative_grind_steps_from_reference + grind_delta_steps
    direction_sign = 1 if current_recipe.grinder_step_direction == GrinderStepDirection.HIGHER_IS_FINER else -1
    return ShadowRecipeProposal(
        source=source,
        grind_delta_steps_from_current=grind_delta_steps,
        projected_relative_step_from_reference=projected_step,
        projected_relative_grind_um_from_reference=(
            projected_step * current_recipe.microns_per_step * direction_sign
        ),
        next_dose_g=next_dose_g,
        target_yield_g=target_yield_g,
        target_ratio=target_yield_g / next_dose_g,
        confidence=max(0.0, min(1.0, confidence)),
        safety_valid=not safety_errors,
        safety_errors=safety_errors,
    )


def _evaluation_id(
    *,
    checkpoint_sha256: str,
    inference_contract_id: str,
    install_id: str,
    machine_id: str,
    bean_context_id: str,
    grinder_context_id: str,
    shot_id: str,
) -> str:
    canonical = "\n".join(
        (
            inference_contract_id,
            checkpoint_sha256,
            install_id,
            machine_id,
            bean_context_id,
            grinder_context_id,
            shot_id,
        )
    )
    return f"shadow_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:32]}"


def _required_context(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DreamerShadowEvaluationError(f"shadow evaluation requires {field_name}")
    return value


def _optional_reward(value: object) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise DreamerShadowEvaluationError("shadow transition reward is out of range")
    return parsed

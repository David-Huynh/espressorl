from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from espresso_rl.application.dreamer_shadow_inference import DreamerShadowInferenceSession
from espresso_rl.application.training_export import local_training_transition_from_shot
from espresso_rl.domain.dreamer_actions import DreamerActionCandidate, validate_dreamer_action
from espresso_rl.domain.models import (
    Recipe,
    Recommendation,
    RecommendationMode,
    RecommendationStatus,
    SafetyBounds,
    ShotRecord,
    new_id,
)
from espresso_rl.domain.optimization import OptimizationContext
from espresso_rl.domain.safety import validate_recommendation
from espresso_rl.domain.training import validate_training_transition
from espresso_rl.dreamer.dataset import (
    DREAMER_CONTEXT_WINDOW_SIZE,
    DreamerEpisodeDatasetError,
    build_dreamer_episode_batch,
    build_dreamer_episodes_from_training_rows,
)


class DreamerRecommendationError(ValueError):
    pass


@dataclass(frozen=True)
class DreamerRecipeProposal:
    grind_delta_steps_from_current: int
    projected_relative_step_from_reference: float
    projected_relative_grind_um_from_reference: float
    next_dose_g: float
    target_yield_g: float
    target_ratio: float
    confidence: float
    safety_errors: tuple[str, ...] = ()

    @property
    def safety_valid(self) -> bool:
        return not self.safety_errors


class DreamerRecommendationService:
    """Builds safety-checked Dreamer recipe candidates from canonical context."""

    def __init__(
        self,
        *,
        session: DreamerShadowInferenceSession,
        safety_bounds: SafetyBounds | None = None,
    ) -> None:
        if not session.status.parity_verified:
            raise ValueError("Dreamer recommendation service requires a parity-verified session")
        self._session = session
        self._safety_bounds = safety_bounds or SafetyBounds()

    def recommend(
        self,
        context: OptimizationContext,
        *,
        mode: RecommendationMode = RecommendationMode.DREAMER_CANDIDATE,
    ) -> Recommendation:
        if mode not in {RecommendationMode.DREAMER_CANDIDATE, RecommendationMode.DREAMER_ACTIVE}:
            raise DreamerRecommendationError("Dreamer recommendation mode is invalid")
        transition, context_transitions, source_shot = local_context_replay_from_optimization_context(context)
        batch, current_episode_index = dreamer_episode_batch_from_training_rows(
            self._session,
            transition,
            context_transitions=context_transitions,
        )
        proposal = dreamer_recipe_proposal(
            self._session,
            batch,
            current_episode_index=current_episode_index,
            current_recipe=context.current_recipe,
            safety_bounds=context.safety_bounds or self._safety_bounds,
            reason="DreamerV3 candidate.",
        )
        if not proposal.safety_valid:
            raise DreamerRecommendationError(
                f"Dreamer proposal failed safety validation: {'; '.join(proposal.safety_errors)}"
            )
        return recommendation_from_dreamer_proposal(
            context=context,
            proposal=proposal,
            source_shot=source_shot,
            mode=mode,
        )


def local_context_replay_from_optimization_context(
    context: OptimizationContext,
) -> tuple[dict[str, Any], list[dict[str, Any]], ShotRecord]:
    expected_key = _context_key_from_optimization_context(context)
    rows: list[tuple[int, str, dict[str, Any], ShotRecord]] = []
    for shot in context.shots:
        transition = local_training_transition_from_shot(shot)
        if transition is None:
            continue
        if _transition_context_key(transition) != expected_key:
            raise DreamerRecommendationError("Dreamer local replay mixes bean or grinder contexts")
        rows.append(
            (
                int(transition["observation"]["timestamp"]),
                str(transition["observation"]["shot_id"]),
                transition,
                shot,
            )
        )
    if not rows:
        raise DreamerRecommendationError("Dreamer recommendation requires at least one valid local episode")
    rows.sort(key=lambda item: (item[0], item[1]))
    _, _, current, source_shot = rows[-1]
    history = [row for _, _, row, _ in rows[:-1]]
    return current, history[-DREAMER_CONTEXT_WINDOW_SIZE:], source_shot


def dreamer_episode_batch_from_training_rows(
    session: DreamerShadowInferenceSession,
    transition: dict[str, Any],
    *,
    context_transitions: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], int]:
    architecture = session.checkpoint.architecture
    if architecture is None:
        raise DreamerRecommendationError("checkpoint runtime architecture is missing")
    try:
        replay_rows = dreamer_context_replay_rows(transition, context_transitions or [])
        episodes = build_dreamer_episodes_from_training_rows(replay_rows)
        batch = build_dreamer_episode_batch(
            episodes,
            control_spec=architecture.control_spec,
            device="cpu",
        )
    except DreamerEpisodeDatasetError as exc:
        raise DreamerRecommendationError(f"Dreamer episode conversion failed: {exc}") from exc
    if (
        batch["observations"].shape[-1] != architecture.observation_dim
        or batch["static_context"].shape[-1] != architecture.static_dim
        or batch["dynamic_actions"].shape[-1] != architecture.dynamic_action_dim
        or batch["context_static"].shape[-1] != architecture.context_encoder.static_dim
        or batch["context_terminal"].shape[-1] != architecture.context_encoder.terminal_dim
        or batch["context_time"].shape[-1] != architecture.context_encoder.time_dim
        or batch["context_trajectory_embedding"].shape[-1] != architecture.context_encoder.trajectory_dim
    ):
        raise DreamerRecommendationError("Dreamer episode feature layout is incompatible with checkpoint")
    source_ids = [int(item) for item in batch["source_training_row_ids"].tolist()]
    current_row_id = int(transition["training_row_id"])
    if current_row_id not in source_ids:
        raise DreamerRecommendationError("Dreamer context replay omitted the current transition")
    return batch, source_ids.index(current_row_id)


@torch.no_grad()
def dreamer_recipe_proposal(
    session: DreamerShadowInferenceSession,
    batch: dict[str, Any],
    *,
    current_episode_index: int,
    current_recipe: Recipe,
    safety_bounds: SafetyBounds,
    reason: str,
) -> DreamerRecipeProposal:
    models = session.models
    context_state = models.context_encoder(batch)
    observed = models.world_model.observe(batch, context_state=context_state, sample=False)
    valid_index = int(batch["step_mask"][current_episode_index].sum().item()) - 1
    if valid_index < 0:
        raise DreamerRecommendationError("Dreamer episode contains no valid observation steps")
    features = observed["features"][current_episode_index : current_episode_index + 1, valid_index]
    control_mask = batch["control_action_mask"][current_episode_index : current_episode_index + 1, valid_index]
    actor_output = models.actor(features, control_mask)
    static_actions = actor_output["static_actions"][0].detach().cpu()
    static_logits = actor_output["static_logits"][0].detach().cpu()
    grind_delta_steps = int(round(float(static_actions[0].item())))
    next_dose_g = float(current_recipe.dose_g + static_actions[1].item())
    target_yield_g = float(current_recipe.target_yield_g + static_actions[2].item())
    target_ratio = target_yield_g / next_dose_g
    confidence = float(F.softmax(static_logits, dim=-1).amax(dim=-1).mean().item())

    safety_errors: tuple[str, ...] = ()
    try:
        candidate = DreamerActionCandidate(
            grind_delta_steps_from_current=grind_delta_steps,
            next_dose_g=next_dose_g,
            target_yield_g=target_yield_g,
            target_ratio=target_ratio,
            confidence=confidence,
            reason=reason,
        )
        validate_dreamer_action(candidate, current=current_recipe, bounds=safety_bounds)
    except ValueError as exc:
        safety_errors = (str(exc),)

    projected_step = current_recipe.relative_grind_steps_from_reference + grind_delta_steps
    return DreamerRecipeProposal(
        grind_delta_steps_from_current=grind_delta_steps,
        projected_relative_step_from_reference=projected_step,
        projected_relative_grind_um_from_reference=(
            projected_step * current_recipe.microns_per_step * current_recipe.grinder_direction_sign
        ),
        next_dose_g=next_dose_g,
        target_yield_g=target_yield_g,
        target_ratio=target_ratio,
        confidence=max(0.0, min(1.0, confidence)),
        safety_errors=safety_errors,
    )


def recommendation_from_dreamer_proposal(
    *,
    context: OptimizationContext,
    proposal: DreamerRecipeProposal,
    source_shot: ShotRecord,
    mode: RecommendationMode,
) -> Recommendation:
    if not proposal.safety_valid:
        raise DreamerRecommendationError("unsafe Dreamer proposal cannot become a recommendation")
    projected_absolute_step = None
    if source_shot.current_absolute_step is not None:
        projected_absolute_step = source_shot.current_absolute_step + proposal.grind_delta_steps_from_current
    recommendation = Recommendation(
        recommendation_id=new_id("rec"),
        created_at=context.now,
        updated_at=context.now,
        expires_at=context.now + 12 * 60 * 60,
        install_id=context.install_id,
        machine_id=context.machine_id,
        bean_context_id=context.bean_context_id,
        grinder_context_id=context.grinder_context_id,
        profile_id=source_shot.profile_id,
        raw_profile_hash=source_shot.raw_profile_hash,
        grind_delta_steps_from_current=proposal.grind_delta_steps_from_current,
        grind_delta_um_from_current=(
            proposal.grind_delta_steps_from_current
            * context.current_recipe.microns_per_step
            * context.current_recipe.grinder_direction_sign
        ),
        projected_relative_step_from_reference=proposal.projected_relative_step_from_reference,
        projected_relative_grind_um_from_reference=proposal.projected_relative_grind_um_from_reference,
        next_dose_g=proposal.next_dose_g,
        target_yield_g=proposal.target_yield_g,
        target_ratio=proposal.target_ratio,
        mode=mode,
        confidence=proposal.confidence,
        reason="DreamerV3 candidate from parity-verified context inference.",
        status=RecommendationStatus.PENDING,
        source_shot_id=source_shot.shot_id,
        grinder_calibration_mode=source_shot.grinder_calibration_mode,
        grinder_step_direction=context.current_recipe.grinder_step_direction,
        grinder_reference_label=source_shot.grinder_reference_label,
        current_absolute_step=source_shot.current_absolute_step,
        absolute_reference_step=source_shot.absolute_reference_step,
        projected_absolute_step=projected_absolute_step,
    )
    validate_recommendation(context.current_recipe, recommendation, context.safety_bounds)
    return recommendation


def dreamer_context_replay_rows(
    transition: dict[str, Any],
    context_transitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not context_transitions:
        return [transition]
    errors = validate_training_transition(transition)
    if errors:
        raise DreamerRecommendationError(f"Dreamer transition is invalid: {'; '.join(errors[:10])}")
    current_context = _transition_context_key(transition)
    current_row_id = int(transition["training_row_id"])
    current_timestamp = float(transition["observation"]["timestamp"])
    history: list[dict[str, Any]] = []
    seen_row_ids: set[int] = {current_row_id}
    seen_shot_ids: set[str] = {str(transition["observation"]["shot_id"])}
    for candidate in context_transitions:
        errors = validate_training_transition(candidate)
        if errors:
            raise DreamerRecommendationError(
                f"Dreamer context transition is invalid: {'; '.join(errors[:10])}"
            )
        row_id = int(candidate["training_row_id"])
        shot_id = str(candidate["observation"]["shot_id"])
        if row_id in seen_row_ids or shot_id in seen_shot_ids:
            raise DreamerRecommendationError("Dreamer context replay contains duplicate shots")
        if _transition_context_key(candidate) != current_context:
            raise DreamerRecommendationError("Dreamer context replay mixes bean or grinder contexts")
        if float(candidate["observation"]["timestamp"]) >= current_timestamp:
            raise DreamerRecommendationError("Dreamer context replay contains stale or future context")
        seen_row_ids.add(row_id)
        seen_shot_ids.add(shot_id)
        history.append(candidate)
    history = sorted(
        history,
        key=lambda row: (float(row["observation"]["timestamp"]), int(row["training_row_id"])),
    )
    return [*history[-DREAMER_CONTEXT_WINDOW_SIZE:], transition]


def _transition_context_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    source = row["source"]
    context = row["context"]
    return (
        _required_context(source.get("install_id"), "install_id"),
        _required_context(context.get("machine_id"), "machine_id"),
        _required_context(context.get("bean_context_id"), "bean_context_id"),
        _required_context(context.get("grinder_context_id"), "grinder_context_id"),
    )


def _context_key_from_optimization_context(context: OptimizationContext) -> tuple[str, str, str, str]:
    return (
        _required_context(context.install_id, "install_id"),
        _required_context(context.machine_id, "machine_id"),
        _required_context(context.bean_context_id, "bean_context_id"),
        _required_context(context.grinder_context_id, "grinder_context_id"),
    )


def _required_context(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DreamerRecommendationError(f"Dreamer recommendation requires {field_name}")
    return value

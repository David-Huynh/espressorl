from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from espresso_rl.application.preference_optimization import (
    ConsecutivePreferenceOptimizationService,
)
from espresso_rl.domain.cpbo import (
    ComparisonMode,
    OptimizationRunContext,
    PhysicalShotStatus,
    PreferenceLabel,
    Suggestion,
)
from espresso_rl.domain.events import PreferenceFeedbackEvent
from espresso_rl.domain.community import PairwiseShotComparison
from espresso_rl.domain.models import Recipe, Recommendation, SafetyBounds, ShotRecord, ShotType
from espresso_rl.ports.repositories import ShotRepository


RunContextFactory = Callable[[ShotRecord], OptimizationRunContext]
RecommendationSink = Callable[[Recommendation], None]
ComparisonSink = Callable[[PairwiseShotComparison], None]


@dataclass(frozen=True)
class CPBOShotOutcome:
    optimization_run_id: str | None
    recommendation: Recommendation | None
    awaiting_preference: bool
    skipped_reason: str | None = None


class CPBORuntimeBridge:
    """Maps stored canonical shots to the stateful CPBO application API."""

    def __init__(
        self,
        optimizer: ConsecutivePreferenceOptimizationService,
        shots: ShotRepository,
        recommendation_sink: RecommendationSink,
        context_factory: RunContextFactory,
        comparison_sink: ComparisonSink | None = None,
        *,
        comparison_mode: ComparisonMode,
        safety_bounds: SafetyBounds,
    ) -> None:
        self._optimizer = optimizer
        self._shots = shots
        self._recommendation_sink = recommendation_sink
        self._context_factory = context_factory
        self._comparison_sink = comparison_sink
        self._comparison_mode = ComparisonMode(comparison_mode)
        self._safety_bounds = safety_bounds

    def handle_shot(self, shot: ShotRecord) -> CPBOShotOutcome:
        if shot.shot_type != ShotType.ESPRESSO or shot.exclude_from_local_optimization:
            return CPBOShotOutcome(None, None, False, "shot_not_locally_optimizable")
        current_recipe = _observed_recipe(shot)
        if current_recipe is None:
            return CPBOShotOutcome(None, None, False, "recipe_controls_not_fully_observed")
        try:
            context = self._context_factory(shot)
        except ValueError as exc:
            return CPBOShotOutcome(None, None, False, str(exc))

        run = self._optimizer.active_run(context)
        if run is None:
            request = self._optimizer.initialize(
                context,
                current_recipe,
                comparison_mode=self._comparison_mode,
            )
            run_id = request.optimization_run_id
        else:
            run_id = run.run_id

        state_before = self._optimizer.get_state(run_id)
        has_valid_baseline = state_before.previous_valid_shot_id is not None
        if has_valid_baseline and state_before.pending_shot_id is not None:
            return CPBOShotOutcome(
                run_id,
                None,
                True,
                "preference_feedback_pending",
            )
        if has_valid_baseline and state_before.pending_recipe_id is None:
            # A physical shot may arrive after a process restart or without the
            # displayed proposal. Establish the correct anchor first, then keep
            # the actual observed recipe instead of fabricating follow-through.
            self._optimizer.suggest_next(run_id)

        status = _physical_status(shot)
        self._optimizer.record_shot(
            run_id,
            current_recipe,
            status,
            telemetry=shot.fixed_cadence_sequence,
            metadata=_shot_metadata(shot),
            shot_id=shot.shot_id,
            started_at=_shot_started_at(shot),
            completed_at=(shot.timestamp if status == PhysicalShotStatus.VALID else None),
            raw_telemetry_reference=shot.shot_id if shot.fixed_cadence_sequence is not None else None,
            allow_recipe_deviation=has_valid_baseline,
        )

        if status == PhysicalShotStatus.VALID and has_valid_baseline:
            return CPBOShotOutcome(run_id, None, True)
        if status != PhysicalShotStatus.VALID and not has_valid_baseline:
            return CPBOShotOutcome(run_id, None, False, status.value)

        suggestion = self._optimizer.suggest_next(run_id)
        recommendation = self._machine_recommendation(suggestion, shot, current_recipe)
        self._recommendation_sink(recommendation)
        return CPBOShotOutcome(run_id, recommendation, False)

    def handle_preference(self, event: PreferenceFeedbackEvent) -> Recommendation:
        run = self._optimizer.get_run(event.optimization_run_id)
        if run.context.install_id != event.install_id:
            raise ValueError("preference install_id does not own the CPBO run")
        if not _same_machine_id(run.context.machine_id, event.machine_id):
            raise ValueError("preference machine_id does not own the CPBO run")
        if event.comparison_mode is not None and event.comparison_mode != run.comparison_mode:
            raise ValueError("preference comparison_mode does not match the optimization run")
        self._optimizer.record_preference(
            event.optimization_run_id,
            event.new_shot_id,
            event.anchor_shot_id,
            event.label,
        )
        comparison = self._optimizer.get_comparison(
            event.optimization_run_id,
            event.new_shot_id,
            event.anchor_shot_id,
        )
        shot = self._shots.get(event.new_shot_id)
        if shot is None:
            raise ValueError("canonical shot for CPBO preference is missing")
        current_recipe = _observed_recipe(shot)
        if current_recipe is None:
            raise ValueError("CPBO preference shot no longer has complete recipe controls")
        if self._comparison_sink is not None:
            self._comparison_sink(
                PairwiseShotComparison(
                    comparison_id=comparison.comparison_id,
                    optimization_run_id=comparison.optimization_run_id,
                    new_shot_id=comparison.new_shot_id,
                    anchor_shot_id=comparison.anchor_shot_id,
                    label=comparison.label.value,
                    comparison_mode=comparison.comparison_mode.value,
                    created_at=comparison.created_at,
                    install_id=run.context.install_id,
                    machine_id=run.context.machine_id,
                    machine_adapter=shot.machine_adapter,
                    recommendation_id=shot.recommendation_id,
                    bean_context_id=run.context.bean_context_id,
                    grinder_context_id=run.context.grinder_context_id,
                    profile_id=run.context.profile_id,
                    raw_profile_hash=run.context.raw_profile_hash,
                )
            )
        suggestion = self._optimizer.suggest_next(event.optimization_run_id)
        recommendation = self._machine_recommendation(suggestion, shot, current_recipe)
        self._recommendation_sink(recommendation)
        return recommendation

    def reset_owner(self, install_id: str, machine_id: str) -> dict[str, int]:
        return self._optimizer.reset_owner(install_id, machine_id)

    def _machine_recommendation(
        self,
        suggestion: Suggestion,
        shot: ShotRecord,
        current_recipe: Recipe,
    ) -> Recommendation:
        return self._optimizer.suggestion_to_machine_recommendation(
            suggestion,
            current_recipe=current_recipe,
            safety_bounds=self._safety_bounds,
            install_id=shot.install_id,
            machine_id=shot.machine_id,
            bean_context_id=shot.bean_context_id,
            grinder_context_id=shot.grinder_context_id,
            profile_id=shot.profile_id,
            raw_profile_hash=(shot.raw_profile_hash if shot.profile_id is None else None),
        )


def strict_context_from_shot(shot: ShotRecord) -> OptimizationRunContext:
    if not shot.bean_context_id:
        raise ValueError("CPBO requires a bean context")
    if not shot.grinder_context_id:
        raise ValueError("CPBO requires a grinder context")
    profile_id = shot.profile_id or shot.profile_label
    if not profile_id:
        raise ValueError("CPBO requires a stable profile context")
    return OptimizationRunContext(
        install_id=shot.install_id,
        machine_id=shot.machine_id,
        bean_context_id=shot.bean_context_id,
        grinder_context_id=shot.grinder_context_id,
        profile_id=profile_id,
        raw_profile_hash=shot.raw_profile_hash,
        basket_id=f"basket_ml:{shot.basket_size_ml:.6g}",
        user_id=shot.user_id or None,
    )


def _observed_recipe(shot: ShotRecord) -> Recipe | None:
    if not (
        shot.grind_observed
        and shot.dose_observed
        and shot.target_yield_observed
        and shot.relative_grind_steps_from_reference is not None
    ):
        return None
    values = (
        shot.relative_grind_steps_from_reference,
        shot.microns_per_step,
        shot.dose_in_g,
        shot.target_yield_g,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return None
    return shot.to_recipe()


def _physical_status(shot: ShotRecord) -> PhysicalShotStatus:
    state = (shot.shot_end_state or "finished").strip().lower()
    if state in {"aborted", "abort", "cancelled", "canceled"}:
        return PhysicalShotStatus.ABORTED
    if state in {"failed", "failure", "machine_failure", "error"}:
        return PhysicalShotStatus.MACHINE_FAILURE
    return PhysicalShotStatus.VALID


def _shot_started_at(shot: ShotRecord) -> int:
    duration = max(0, int(math.ceil(shot.shot_time_s or 0.0)))
    return max(0, shot.timestamp - duration)


def _shot_metadata(shot: ShotRecord) -> dict[str, object]:
    return {
        "canonical_shot_id": shot.shot_id,
        "beverage_out_g": shot.beverage_out_g,
        "shot_time_s": shot.shot_time_s,
        "weight_source": shot.weight_source,
        "profile_id": shot.profile_id,
    }


def _same_machine_id(left: str, right: str) -> bool:
    if left == right:
        return True
    if left.startswith("gaggimate:") and right.startswith("gaggimate:"):
        return left.removeprefix("gaggimate:").casefold() == right.removeprefix(
            "gaggimate:"
        ).casefold()
    return False

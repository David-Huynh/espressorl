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
    PendingPreferenceRequest,
    PhysicalShotStatus,
    PreferenceLabel,
    RecipeDomain,
    Suggestion,
    TrustRegionAction,
)
from espresso_rl.domain.events import (
    MachineStateEvent,
    OptimizerControlEvent,
    OptimizerSettingsEvent,
    PreferenceFeedbackEvent,
)
from espresso_rl.domain.community import PairwiseShotComparison
from espresso_rl.domain.models import Recipe, Recommendation, ShotRecord, ShotType
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
    preference_request: PendingPreferenceRequest | None = None


@dataclass(frozen=True)
class CPBOLocalOptimizationStatus:
    optimization_run_id: str
    locally_converged: bool
    trust_region_length: float
    trust_region_success_count: int
    trust_region_failure_count: int
    last_transition_action: TrustRegionAction | None


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
    ) -> None:
        self._optimizer = optimizer
        self._shots = shots
        self._recommendation_sink = recommendation_sink
        self._context_factory = context_factory
        self._comparison_sink = comparison_sink
        self._comparison_mode = ComparisonMode(comparison_mode)

    def configure_recipe_domain(self, recipe_domain: RecipeDomain) -> None:
        self._optimizer.configure_recipe_domain(recipe_domain)

    def configure_optimizer(
        self,
        optimizer: ConsecutivePreferenceOptimizationService,
        *,
        comparison_mode: ComparisonMode,
    ) -> None:
        self._optimizer = optimizer
        self._comparison_mode = ComparisonMode(comparison_mode)

    def refresh_after_optimizer_settings(
        self,
        event: OptimizerSettingsEvent,
        current_machine_state: MachineStateEvent | None = None,
    ) -> CPBOShotOutcome:
        recent = self._shots.list_recent(
            event.install_id,
            event.machine_id,
            event.bean_context_id,
            limit=200,
            grinder_context_id=event.grinder_context_id,
        )
        for shot in reversed(recent):
            if not _shot_matches_optimizer_settings(shot, event):
                continue
            current_recipe = _known_recipe(shot)
            if current_recipe is None:
                continue
            try:
                context = self._context_factory(shot)
            except ValueError:
                continue
            run = self._optimizer.active_run(
                context,
                comparison_mode=self._comparison_mode,
            )
            if run is None:
                continue
            state = self._optimizer.get_state(run.run_id)
            if state.pending_shot_id is not None:
                return CPBOShotOutcome(
                    run.run_id,
                    None,
                    True,
                    "existing_preference_feedback_pending",
                )
            if state.trust_region_state.locally_converged:
                return CPBOShotOutcome(
                    run.run_id,
                    None,
                    False,
                    "local_optimization_converged",
                )
            if state.previous_valid_shot_id is None:
                return CPBOShotOutcome(
                    run.run_id,
                    None,
                    False,
                    "optimization_run_has_no_valid_shot",
                )
            suggestion = self._optimizer.suggest_next(run.run_id)
            recommendation = self._machine_recommendation(
                suggestion,
                shot,
                current_recipe,
                current_machine_state=current_machine_state,
            )
            self._recommendation_sink(recommendation)
            return CPBOShotOutcome(run.run_id, recommendation, False)
        return CPBOShotOutcome(
            None,
            None,
            False,
            "no_matching_optimization_history",
        )

    def handle_shot(self, shot: ShotRecord) -> CPBOShotOutcome:
        if shot.shot_type != ShotType.ESPRESSO or shot.exclude_from_local_optimization:
            return CPBOShotOutcome(None, None, False, "shot_not_locally_optimizable")
        existing_preference_shot = self._optimizer.find_shot(shot.shot_id)
        if existing_preference_shot is not None:
            state = self._optimizer.get_state(existing_preference_shot.optimization_run_id)
            preference_request = self._pending_preference_request(
                existing_preference_shot.optimization_run_id,
                shot.shot_id,
                shot.recommendation_id,
            )
            return CPBOShotOutcome(
                existing_preference_shot.optimization_run_id,
                None,
                state.pending_shot_id == shot.shot_id,
                "shot_already_processed",
                preference_request,
            )
        current_recipe = _known_recipe(shot)
        if current_recipe is None:
            return CPBOShotOutcome(None, None, False, "recipe_controls_not_fully_known")
        try:
            context = self._context_factory(shot)
        except ValueError as exc:
            return CPBOShotOutcome(None, None, False, str(exc))

        run = self._optimizer.active_run(context, comparison_mode=self._comparison_mode)
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
        if (
            has_valid_baseline
            and state_before.trust_region_state.locally_converged
        ):
            return CPBOShotOutcome(
                run_id,
                None,
                False,
                "local_optimization_converged",
            )
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
            return CPBOShotOutcome(
                run_id,
                None,
                True,
                preference_request=self._pending_preference_request(
                    run_id,
                    shot.shot_id,
                    shot.recommendation_id,
                ),
            )
        if status != PhysicalShotStatus.VALID and not has_valid_baseline:
            return CPBOShotOutcome(run_id, None, False, status.value)

        suggestion = self._optimizer.suggest_next(run_id)
        recommendation = self._machine_recommendation(suggestion, shot, current_recipe)
        self._recommendation_sink(recommendation)
        return CPBOShotOutcome(run_id, recommendation, False)

    def _pending_preference_request(
        self,
        run_id: str,
        shot_id: str,
        recommendation_id: str | None = None,
    ) -> PendingPreferenceRequest | None:
        run = self._optimizer.get_run(run_id)
        state = self._optimizer.get_state(run_id)
        if state.pending_shot_id != shot_id or state.pending_anchor_shot_id is None:
            return None
        return PendingPreferenceRequest(
            install_id=run.context.install_id,
            machine_id=run.context.machine_id,
            optimization_run_id=run_id,
            new_shot_id=shot_id,
            anchor_shot_id=state.pending_anchor_shot_id,
            comparison_mode=run.comparison_mode,
            taste_goal=run.context.taste_goal,
            recommendation_id=recommendation_id,
        )

    def handle_preference(self, event: PreferenceFeedbackEvent) -> Recommendation | None:
        run = self._optimizer.get_run(event.optimization_run_id)
        if run.context.install_id != event.install_id:
            raise ValueError("preference install_id does not own the CPBO run")
        if not _same_machine_id(run.context.machine_id, event.machine_id):
            raise ValueError("preference machine_id does not own the CPBO run")
        pending = self._optimizer.get_pending_suggestion(event.optimization_run_id)
        if pending is None:
            raise ValueError("preference optimization run has no pending comparison")
        if event.comparison_mode is not None and event.comparison_mode != pending.comparison_mode:
            raise ValueError("preference comparison_mode does not match the optimization run")
        if event.taste_goal.fingerprint != run.context.taste_goal.fingerprint:
            raise ValueError("preference taste goal does not match the optimization run")
        updated_state = self._optimizer.record_preference(
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
        current_recipe = _known_recipe(shot)
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
                    taste_goal=run.context.taste_goal,
                )
            )
        if updated_state.trust_region_state.locally_converged:
            return None
        suggestion = self._optimizer.suggest_next(event.optimization_run_id)
        recommendation = self._machine_recommendation(suggestion, shot, current_recipe)
        self._recommendation_sink(recommendation)
        return recommendation

    def handle_shot_correction(
        self,
        shot: ShotRecord,
        current_machine_state: MachineStateEvent | None = None,
    ) -> CPBOShotOutcome:
        preference_shot = self._optimizer.find_shot(shot.shot_id)
        if preference_shot is None:
            return CPBOShotOutcome(None, None, False, "shot_not_processed_by_cpbo")
        if shot.exclude_from_local_optimization:
            if preference_shot.status == PhysicalShotStatus.EXCLUDED:
                return CPBOShotOutcome(
                    preference_shot.optimization_run_id,
                    None,
                    False,
                    "shot_already_excluded",
                )
            self._optimizer.exclude_shot(shot.shot_id)
            state = self._optimizer.get_state(preference_shot.optimization_run_id)
            current_shot_id = state.previous_valid_shot_id
            if current_shot_id is None:
                return CPBOShotOutcome(
                    preference_shot.optimization_run_id,
                    None,
                    False,
                    "optimization_run_has_no_valid_shot",
                )
            current_shot = self._shots.get(current_shot_id)
            if current_shot is None:
                raise ValueError("canonical current shot for rebuilt CPBO run is missing")
            current_recipe = _known_recipe(current_shot)
            if current_recipe is None:
                raise ValueError("canonical current shot no longer has complete recipe controls")
            if state.trust_region_state.locally_converged:
                return CPBOShotOutcome(
                    preference_shot.optimization_run_id,
                    None,
                    False,
                    "local_optimization_converged",
                )
            suggestion = self._optimizer.suggest_next(
                preference_shot.optimization_run_id
            )
            recommendation = self._machine_recommendation(
                suggestion,
                current_shot,
                current_recipe,
                current_machine_state=current_machine_state,
            )
            self._recommendation_sink(recommendation)
            return CPBOShotOutcome(
                preference_shot.optimization_run_id,
                recommendation,
                False,
            )
        corrected_recipe = _known_recipe(shot)
        if corrected_recipe is None:
            return CPBOShotOutcome(
                preference_shot.optimization_run_id,
                None,
                False,
                "corrected_recipe_controls_not_fully_known",
            )
        correction = self._optimizer.correct_shot_recipe(
            shot.shot_id,
            corrected_recipe,
            metadata=_shot_metadata(shot),
        )
        if not correction.recipe_changed:
            return CPBOShotOutcome(
                preference_shot.optimization_run_id,
                None,
                correction.awaiting_preference,
                "recipe_unchanged_after_quantization",
            )
        if correction.awaiting_preference:
            return CPBOShotOutcome(
                preference_shot.optimization_run_id,
                None,
                True,
                "existing_preference_feedback_pending",
            )
        state = self._optimizer.get_state(preference_shot.optimization_run_id)
        current_shot_id = state.previous_valid_shot_id
        current_shot = self._shots.get(current_shot_id) if current_shot_id else None
        if current_shot is None:
            raise ValueError("canonical current shot for corrected CPBO run is missing")
        current_recipe = _known_recipe(current_shot)
        if current_recipe is None:
            raise ValueError("canonical current shot no longer has complete recipe controls")
        if state.trust_region_state.locally_converged:
            return CPBOShotOutcome(
                preference_shot.optimization_run_id,
                None,
                False,
                "local_optimization_converged",
            )
        suggestion = self._optimizer.suggest_next(preference_shot.optimization_run_id)
        recommendation = self._machine_recommendation(
            suggestion,
            current_shot,
            current_recipe,
            current_machine_state=current_machine_state,
        )
        self._recommendation_sink(recommendation)
        return CPBOShotOutcome(
            preference_shot.optimization_run_id,
            recommendation,
            False,
        )

    def reset_owner(self, install_id: str, machine_id: str) -> dict[str, int]:
        return self._optimizer.reset_owner(install_id, machine_id)

    def resume_local_exploration(
        self,
        run_id: str,
        *,
        control_event_id: str | None = None,
    ) -> CPBOLocalOptimizationStatus:
        state = self._optimizer.resume_local_exploration(
            run_id,
            control_event_id=control_event_id,
        )
        return self._status_from_state(run_id, state)

    def handle_optimizer_control(
        self,
        event: OptimizerControlEvent,
        current_machine_state: MachineStateEvent | None = None,
    ) -> CPBOShotOutcome:
        if event.action.value != "resume_local_exploration":
            raise ValueError("unsupported optimizer control action")
        run = self._optimizer.get_run(event.optimization_run_id)
        if run.context.install_id != event.install_id:
            raise ValueError("optimizer control install_id does not own the CPBO run")
        if not _same_machine_id(run.context.machine_id, event.machine_id):
            raise ValueError("optimizer control machine_id does not own the CPBO run")
        self.resume_local_exploration(
            run.run_id,
            control_event_id=event.request_id,
        )
        state = self._optimizer.get_state(run.run_id)
        current_shot_id = state.previous_valid_shot_id
        if current_shot_id is None:
            raise ValueError("resumed CPBO run has no current physical shot")
        shot = self._shots.get(current_shot_id)
        if shot is None:
            raise ValueError("canonical current shot for resumed CPBO run is missing")
        current_recipe = _known_recipe(shot)
        if current_recipe is None:
            raise ValueError("resumed CPBO shot no longer has complete recipe controls")
        suggestion = (
            self._optimizer.get_pending_suggestion(run.run_id)
            or self._optimizer.suggest_next(run.run_id)
        )
        recommendation = self._machine_recommendation(
            suggestion,
            shot,
            current_recipe,
            current_machine_state=current_machine_state,
        )
        self._recommendation_sink(recommendation)
        return CPBOShotOutcome(run.run_id, recommendation, False)

    def local_optimization_status(
        self,
        run_id: str,
    ) -> CPBOLocalOptimizationStatus:
        return self._status_from_state(run_id, self._optimizer.get_state(run_id))

    def local_optimization_status_for_shot(
        self,
        shot: ShotRecord,
    ) -> CPBOLocalOptimizationStatus | None:
        try:
            context = self._context_factory(shot)
        except ValueError:
            return None
        run = self._optimizer.active_run(
            context,
            comparison_mode=self._comparison_mode,
        )
        if run is None:
            return None
        return self.local_optimization_status(run.run_id)

    @staticmethod
    def _status_from_state(run_id: str, state) -> CPBOLocalOptimizationStatus:
        transitions = state.trust_region_state.transitions
        return CPBOLocalOptimizationStatus(
            optimization_run_id=run_id,
            locally_converged=state.trust_region_state.locally_converged,
            trust_region_length=state.trust_region_state.length,
            trust_region_success_count=state.trust_region_state.success_count,
            trust_region_failure_count=state.trust_region_state.failure_count,
            last_transition_action=(
                transitions[-1].action if transitions else None
            ),
        )

    def _machine_recommendation(
        self,
        suggestion: Suggestion,
        shot: ShotRecord,
        current_recipe: Recipe,
        *,
        current_machine_state: MachineStateEvent | None = None,
    ) -> Recommendation:
        projection_recipe = current_recipe
        grinder_calibration_mode = shot.grinder_calibration_mode
        grinder_reference_label = shot.grinder_reference_label
        current_absolute_step = shot.current_absolute_step
        absolute_reference_step = shot.absolute_reference_step
        if current_machine_state is not None and _machine_state_matches_shot(current_machine_state, shot):
            active_recipe = current_machine_state.current_recipe()
            if active_recipe is not None:
                projection_recipe = active_recipe
                grinder_calibration_mode = current_machine_state.grinder_calibration_mode
                grinder_reference_label = current_machine_state.grinder_reference_label
                current_absolute_step = current_machine_state.current_absolute_step
                absolute_reference_step = current_machine_state.absolute_reference_step
        return self._optimizer.suggestion_to_machine_recommendation(
            suggestion,
            current_recipe=projection_recipe,
            install_id=shot.install_id,
            machine_id=shot.machine_id,
            bean_context_id=shot.bean_context_id,
            grinder_context_id=shot.grinder_context_id,
            profile_id=shot.profile_id,
            raw_profile_hash=(shot.raw_profile_hash if shot.profile_id is None else None),
            grinder_calibration_mode=grinder_calibration_mode,
            grinder_reference_label=grinder_reference_label,
            current_absolute_step=current_absolute_step,
            absolute_reference_step=absolute_reference_step,
        )


def strict_context_from_shot(shot: ShotRecord) -> OptimizationRunContext:
    if not shot.bean_context_id:
        raise ValueError("CPBO requires a bean context")
    if not shot.grinder_context_id:
        raise ValueError("CPBO requires a grinder context")
    profile_id = shot.profile_id or shot.profile_label
    raw_profile_hash = None if profile_id else shot.raw_profile_hash
    if not profile_id and not raw_profile_hash:
        raise ValueError("CPBO requires a stable profile context")
    return OptimizationRunContext(
        install_id=shot.install_id,
        machine_id=shot.machine_id,
        bean_context_id=shot.bean_context_id,
        grinder_context_id=shot.grinder_context_id,
        profile_id=profile_id,
        raw_profile_hash=raw_profile_hash,
        basket_id=f"basket_ml:{shot.basket_size_ml:.6g}",
        user_id=shot.user_id or None,
        taste_goal=shot.taste_goal,
    )


def _machine_state_matches_shot(event: MachineStateEvent, shot: ShotRecord) -> bool:
    return (
        event.install_id == shot.install_id
        and _same_machine_id(event.machine_id, shot.machine_id)
        and event.bean_context_id == shot.bean_context_id
        and event.grinder_context_id == shot.grinder_context_id
        and event.profile_id == shot.profile_id
        and (shot.profile_id is not None or event.raw_profile_hash == shot.raw_profile_hash)
        and event.taste_goal.fingerprint == shot.taste_goal.fingerprint
    )


def _shot_matches_optimizer_settings(
    shot: ShotRecord,
    event: OptimizerSettingsEvent,
) -> bool:
    shot_profile = shot.profile_id or shot.profile_label
    event_profile = event.profile_id or event.profile_label
    return (
        shot.install_id == event.install_id
        and _same_machine_id(shot.machine_id, event.machine_id)
        and shot.bean_context_id == event.bean_context_id
        and shot.grinder_context_id == event.grinder_context_id
        and shot_profile == event_profile
        and shot.taste_goal.fingerprint == event.taste_goal.fingerprint
        and shot.shot_type == ShotType.ESPRESSO
        and not shot.exclude_from_local_optimization
    )


def _known_recipe(shot: ShotRecord) -> Recipe | None:
    if not (
        shot.grind_observed
        and shot.dose_target_g is not None
        and (shot.dose_observed or shot.dose_target_confirmed)
        and shot.target_yield_observed
        and shot.relative_grind_steps_from_reference is not None
    ):
        return None
    values = (
        shot.relative_grind_steps_from_reference,
        shot.microns_per_step,
        shot.dose_target_g,
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
        "dose_measured": shot.dose_observed,
        "dose_target_g": shot.dose_target_g,
        "dose_target_confirmed": shot.dose_target_confirmed,
        "predicted_final_beverage_out_g": shot.predicted_final_beverage_out_g,
        "predictive_stop_applied": shot.predictive_stop_applied,
    }


def _same_machine_id(left: str, right: str) -> bool:
    if left == right:
        return True
    if left.startswith("gaggimate:") and right.startswith("gaggimate:"):
        return left.removeprefix("gaggimate:").casefold() == right.removeprefix(
            "gaggimate:"
        ).casefold()
    return False

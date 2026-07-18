from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Callable

from espresso_rl.application.upload_payloads import (
    make_comparison_upload_item,
    make_recommendation_upload_item,
    make_shot_upload_item,
)
from espresso_rl.domain.community import PairwiseShotComparison
from espresso_rl.domain.cpbo import PendingPreferenceRequest
from espresso_rl.domain.events import (
    MachineStateEvent,
    RecommendationApplyEvent,
    RecommendationDecisionEvent,
    ShotCorrectionEvent,
    ShotProfileEvent,
)
from espresso_rl.domain.follow_through import FollowThroughTolerances, infer_follow_through
from espresso_rl.domain.models import (
    FollowThroughState,
    MachineState,
    Recommendation,
    RecommendationApplyStatus,
    RecommendationDecision,
    RecommendationStatus,
    ShotRecord,
    ShotType,
    now_ts,
)
from espresso_rl.domain.profile import (
    build_fixed_cadence_sequence,
    profile_hash,
    resample_profile_with_quality,
    resample_shot_metadata,
)
from espresso_rl.domain.staleness import check_recommendation_staleness
from espresso_rl.domain.utility import classify_shot_profile_event
from espresso_rl.ports.repositories import (
    RecommendationRepository,
    ShotRepository,
    UploadQueueRepository,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    shot: ShotRecord | None
    recommendation: Recommendation | None
    dropped_reason: str | None = None
    replayed: bool = False
    preference_request: PendingPreferenceRequest | None = None

    @property
    def stored(self) -> bool:
        return self.shot is not None


def _recommendation_signature(recommendation: Recommendation) -> tuple:
    return (
        recommendation.status.value,
        recommendation.apply_status.value,
        recommendation.shown_count > 0,
        tuple(sorted(recommendation.applied_fields.items())),
        tuple(sorted(recommendation.manual_fields)),
        recommendation.apply_error,
        recommendation.grind_delta_steps_from_current,
        round(recommendation.grind_delta_um_from_current, 4),
        round(recommendation.projected_relative_step_from_reference, 4),
        round(recommendation.projected_relative_grind_um_from_reference, 4),
        round(recommendation.next_dose_g, 4),
        round(recommendation.target_yield_g, 4),
        round(recommendation.target_ratio, 4),
        recommendation.mode.value,
        recommendation.grinder_context_id,
        recommendation.optimization_run_id,
        recommendation.comparison_anchor_shot_id,
        recommendation.comparison_mode,
        recommendation.preference_feedback_required,
        round(recommendation.confidence, 4),
        recommendation.reason,
    )


def _normal_context_key(value: str | None) -> str:
    return "" if value is None else " ".join(value.casefold().split())


def _profile_scope_key(
    *,
    profile_id: str | None,
    raw_profile_hash: str | None,
) -> tuple[str, str] | None:
    if profile_id:
        return ("id", _normal_context_key(profile_id))
    if raw_profile_hash:
        return ("hash", raw_profile_hash)
    return None


def _recommendation_matches_profile_scope(
    recommendation: Recommendation,
    scope: tuple[str, str] | None,
) -> bool:
    if scope is None:
        return True
    kind, value = scope
    if kind == "hash":
        return recommendation.raw_profile_hash == value
    return _normal_context_key(recommendation.profile_id) == value


class EspressoRLService:
    """Physical-shot and recommendation lifecycle use cases.

    Preference optimization is stateful and is deliberately orchestrated by
    ``CPBORuntimeBridge``. This service never synthesizes scalar taste rewards.
    """

    def __init__(
        self,
        shots: ShotRepository,
        recommendations: RecommendationRepository,
        upload_queue: UploadQueueRepository | None = None,
        clock: Callable[[], int] = now_ts,
        community_upload_enabled_default: bool = False,
    ) -> None:
        self._shots = shots
        self._recommendations = recommendations
        self._upload_queue = upload_queue
        self._clock = clock
        self._community_upload_enabled_default = bool(community_upload_enabled_default)
        self._community_upload_enabled_by_machine: dict[tuple[str, str], bool] = {}

    def ingest_shot_profile(self, event: ShotProfileEvent) -> IngestResult:
        now = self._clock()
        classification = classify_shot_profile_event(event)
        if not classification.locally_optimizable:
            reason = (
                "local_optimization_disabled"
                if not event.local_optimization_enabled or event.exclude_from_local_optimization
                else f"not_locally_optimizable:{classification.shot_type.value}"
            )
            return IngestResult(shot=None, recommendation=None, dropped_reason=reason)

        profile_quality = resample_profile_with_quality(event)
        shot_metadata = resample_shot_metadata(event)
        fixed_cadence_sequence = build_fixed_cadence_sequence(event)
        effective_profile_hash = event.raw_profile_hash or profile_hash(profile_quality.profile)
        existing = self._shots.get(event.shot_id)
        if existing is not None:
            if not _same_immutable_shot_event(existing, event, effective_profile_hash):
                raise ValueError(f"shot_id {event.shot_id} conflicts with an existing immutable shot")
            return IngestResult(shot=existing, recommendation=None, replayed=True)
        recommendation = self._recommendation_for_event(
            event,
            now,
            raw_profile_hash=effective_profile_hash,
        )
        shot = ShotRecord(
            shot_id=event.shot_id,
            timestamp=int(event.timestamp),
            install_id=event.install_id,
            machine_id=event.machine_id,
            machine_adapter=event.machine_adapter,
            profile=profile_quality.profile,
            microns_per_step=event.microns_per_step,
            dose_in_g=event.dose_in_g,
            target_yield_g=event.target_yield_g,
            grind_observed=event.grind_observed,
            dose_observed=event.dose_observed,
            dose_target_g=event.dose_target_g,
            dose_target_confirmed=event.dose_target_confirmed,
            target_yield_observed=event.target_yield_observed,
            relative_grind_steps_from_reference=event.relative_grind_steps_from_reference,
            beverage_out_g=event.beverage_out_g,
            beverage_out_observation=event.beverage_out_observation,
            predicted_final_beverage_out_g=event.predicted_final_beverage_out_g,
            predictive_stop_applied=event.predictive_stop_applied,
            predictive_stop_delay_ms=event.predictive_stop_delay_ms,
            predictive_stop_rate_g_per_s=event.predictive_stop_rate_g_per_s,
            predictive_stop_lead_g=event.predictive_stop_lead_g,
            shot_time_s=event.shot_time_s,
            bean_context_id=event.bean_context_id,
            bean_context_name=event.bean_context_name,
            grinder_context_id=event.grinder_context_id,
            taste_goal=event.taste_goal,
            grinder_calibration_mode=event.grinder_calibration_mode,
            grinder_step_direction=event.grinder_step_direction,
            grinder_adjustment_mode=event.grinder_adjustment_mode,
            grinder_reference_label=event.grinder_reference_label,
            current_absolute_step=event.current_absolute_step,
            absolute_reference_step=event.absolute_reference_step,
            recommendation_id=(
                recommendation.recommendation_id
                if recommendation is not None
                else event.recommendation_id
            ),
            raw_profile_available=len(event.time_ms) >= 2,
            raw_profile_hash=effective_profile_hash,
            shot_type=classification.shot_type,
            exclude_from_local_optimization=classification.exclude_from_local_optimization,
            optimization_weight=classification.optimization_weight,
            weight_source=event.weight_source,
            flow_source=event.flow_source,
            flow_units=event.flow_units,
            pump_flow_source=event.pump_flow_source,
            pump_flow_units=event.pump_flow_units,
            pump_flow_calibration_required=event.pump_flow_calibration_required,
            profile_flow_valid=profile_quality.flow_valid,
            profile_flow_masked=profile_quality.flow_masked,
            profile_id=event.profile_id,
            profile_label=event.profile_label,
            profile_type=event.profile_type,
            profile_phase_count=event.profile_phase_count,
            final_phase_index=event.final_phase_index,
            final_phase_name=event.final_phase_name,
            final_phase_type=event.final_phase_type,
            final_phase_elapsed_s=event.final_phase_elapsed_s,
            final_pump_target=event.final_pump_target,
            final_target_pressure=event.final_target_pressure,
            final_target_flow=event.final_target_flow,
            final_valve_open=event.final_valve_open,
            profile_temperature_c=event.profile_temperature_c,
            final_phase_temperature_c=event.final_phase_temperature_c,
            beverage_flow_profile=shot_metadata.beverage_flow_profile,
            temperature_profile=shot_metadata.temperature_profile,
            target_temperature_profile=shot_metadata.target_temperature_profile,
            pump_target_mode_profile=shot_metadata.pump_target_mode_profile,
            fixed_cadence_sequence=fixed_cadence_sequence,
            shot_end_state=event.shot_end_state,
            created_at=now,
            updated_at=now,
        )

        decision = self._decision_from_recommendation(recommendation)
        if recommendation is not None:
            shot.recommended_grind_delta_steps_from_current = (
                recommendation.grind_delta_steps_from_current
            )
            shot.recommended_grind_delta_um_from_current = (
                recommendation.grind_delta_um_from_current
            )
            shot.recommended_projected_relative_step_from_reference = (
                recommendation.projected_relative_step_from_reference
            )
            shot.recommended_dose_g = recommendation.next_dose_g
            shot.recommended_target_yield_g = recommendation.target_yield_g
            shot.recommended_target_ratio = recommendation.target_ratio
            shot.recommendation_decision = decision
            followed = infer_follow_through(shot, recommendation, decision)
            shot.recommendation_followed = followed.state
            shot.recommendation_attribution_weight = followed.attribution_weight
            self._mark_recommendation_used_if_followed(recommendation, followed.state, now)

        self._store_shot(
            shot,
            now,
            community_upload_enabled=event.community_upload_enabled,
        )
        return IngestResult(shot=shot, recommendation=None)

    def record_shot_correction(
        self,
        event: ShotCorrectionEvent,
    ) -> ShotRecord:
        now = self._clock()
        shot = self._shots.get(event.shot_id)
        if shot is None:
            raise ValueError(f"unknown shot_id {event.shot_id}")
        if shot.install_id != event.install_id or shot.machine_id != event.machine_id:
            raise ValueError("shot correction does not match the stored shot owner")
        shot = copy.copy(shot)

        tags = set(event.correction_tags)
        if event.shot_type is not None:
            shot.shot_type = event.shot_type
        if "utility_brew" in tags:
            shot.shot_type = ShotType.UTILITY_FLUSH

        should_exclude = event.exclude_from_local_optimization is True
        if tags.intersection({"bad_puck_prep", "utility_brew"}) or shot.shot_type != ShotType.ESPRESSO:
            should_exclude = True

        if event.grind_followed is not None:
            shot.grind_followed = event.grind_followed
            shot.grind_recommendation_trust = 1.0 if event.grind_followed else 0.0
            if event.grind_followed:
                if shot.recommended_projected_relative_step_from_reference is not None:
                    shot.relative_grind_steps_from_reference = (
                        shot.recommended_projected_relative_step_from_reference
                    )
                    shot.relative_grind_um_from_reference = (
                        shot.relative_grind_steps_from_reference
                        * shot.microns_per_step
                        * shot.grinder_direction_sign
                    )
                shot.grind_observed = shot.relative_grind_steps_from_reference is not None
            else:
                shot.grind_observed = False
        if event.dose_followed is not None:
            shot.dose_followed = event.dose_followed
            shot.dose_recommendation_trust = 1.0 if event.dose_followed else 0.0
            if event.dose_followed:
                if shot.recommended_dose_g is not None:
                    shot.dose_in_g = shot.recommended_dose_g
                    shot.dose_target_g = shot.recommended_dose_g
                shot.dose_target_confirmed = True
                shot.brew_ratio = (
                    shot.beverage_out_g / shot.dose_in_g
                    if shot.beverage_out_g is not None
                    else None
                )
                shot.target_ratio = shot.target_yield_g / shot.dose_in_g
            else:
                shot.dose_observed = False
                shot.dose_target_confirmed = False
                shot.brew_ratio = None
        if event.yield_followed is not None:
            shot.yield_followed = event.yield_followed
            shot.yield_recommendation_trust = 1.0 if event.yield_followed else 0.0

        if "did_not_follow_grind" in tags:
            shot.grind_followed = False
            shot.grind_recommendation_trust = 0.0
            shot.grind_observed = False
        if "did_not_follow_dose" in tags:
            shot.dose_followed = False
            shot.dose_recommendation_trust = 0.0
            shot.dose_observed = False
            shot.brew_ratio = None
        if "did_not_follow_yield" in tags:
            shot.yield_followed = False
            shot.yield_recommendation_trust = 0.0

        corrected_relative_grind = event.relative_grind_steps_from_reference
        if event.current_absolute_step is not None:
            if shot.absolute_reference_step is None:
                raise ValueError("shot correction requires an absolute grinder reference")
            derived_relative = event.current_absolute_step - shot.absolute_reference_step
            if corrected_relative_grind is not None and not math.isclose(
                corrected_relative_grind,
                derived_relative,
                rel_tol=0.0,
                abs_tol=0.01,
            ):
                raise ValueError("absolute and relative grind corrections disagree")
            corrected_relative_grind = derived_relative
            shot.current_absolute_step = event.current_absolute_step
        if corrected_relative_grind is not None:
            shot.relative_grind_steps_from_reference = corrected_relative_grind
            shot.relative_grind_um_from_reference = (
                corrected_relative_grind
                * shot.microns_per_step
                * shot.grinder_direction_sign
            )
            shot.grind_observed = True
            if shot.absolute_reference_step is not None:
                shot.current_absolute_step = (
                    shot.absolute_reference_step + corrected_relative_grind
                )
        if event.dose_in_g is not None:
            shot.dose_in_g = event.dose_in_g
            shot.dose_target_g = event.dose_in_g
            shot.dose_observed = True
            shot.dose_target_confirmed = False
            if shot.recommended_dose_g is not None:
                shot.dose_followed = (
                    abs(event.dose_in_g - shot.recommended_dose_g)
                    <= FollowThroughTolerances().dose_g
                )
                shot.dose_recommendation_trust = 1.0 if shot.dose_followed else 0.0
        if event.target_yield_g is not None:
            shot.target_yield_g = event.target_yield_g
            shot.target_yield_observed = True
        if event.beverage_out_g is not None:
            shot.beverage_out_g = event.beverage_out_g
            shot.beverage_out_observation = "user_corrected"

        ratio_dose = (
            shot.dose_in_g
            if shot.dose_observed
            else shot.dose_target_g if shot.dose_target_confirmed else None
        )
        shot.brew_ratio = (
            shot.beverage_out_g / ratio_dose
            if shot.beverage_out_g is not None and ratio_dose is not None
            else None
        )
        shot.target_ratio = shot.target_yield_g / (shot.dose_target_g or shot.dose_in_g)

        variable_follow = [shot.grind_followed, shot.dose_followed, shot.yield_followed]
        any_not_followed = any(value is False for value in variable_follow)
        if should_exclude:
            shot.exclude_from_local_optimization = True
            shot.optimization_weight = 0.0
            shot.recommendation_attribution_weight = 0.0
            shot.grind_recommendation_trust = 0.0
            shot.dose_recommendation_trust = 0.0
            shot.yield_recommendation_trust = 0.0
            shot.recommendation_followed = FollowThroughState.NOT_FOLLOWED
        elif any_not_followed:
            known = [value for value in variable_follow if value is not None]
            all_known_variables_rejected = len(known) == 3 and all(value is False for value in known)
            shot.recommendation_followed = (
                FollowThroughState.NOT_FOLLOWED
                if all_known_variables_rejected
                else FollowThroughState.PARTIALLY_FOLLOWED
            )
            if shot.recommendation_followed == FollowThroughState.NOT_FOLLOWED:
                shot.recommendation_attribution_weight = 0.0
            else:
                followed_share = sum(1 for value in known if value) / max(1, len(known))
                shot.recommendation_attribution_weight = min(
                    shot.recommendation_attribution_weight or 0.5,
                    max(0.25, followed_share),
                )

        shot.updated_at = now
        self._store_shot(shot, now)
        return shot

    def record_recommendation_decision(
        self,
        event: RecommendationDecisionEvent,
    ) -> Recommendation:
        now = self._clock()
        recommendation = self._recommendations.get(event.recommendation_id)
        if recommendation is None:
            raise ValueError(f"unknown recommendation_id {event.recommendation_id}")
        self._require_recommendation_owner(recommendation, event.install_id, event.machine_id)

        updated = copy.copy(recommendation)
        updated.updated_at = now
        if event.decision == RecommendationDecision.ACCEPTED:
            updated.status = RecommendationStatus.ACCEPTED
            updated.accepted_at = now
        elif event.decision == RecommendationDecision.EDITED:
            updated.status = RecommendationStatus.EDITED
            updated.edited_at = now
            self._apply_edits(updated, event.edited_fields)
        elif event.decision in {RecommendationDecision.IGNORED, RecommendationDecision.DISMISSED}:
            updated.status = RecommendationStatus.IGNORED
            updated.ignored_at = now
        else:
            updated.status = RecommendationStatus.SHOWN
            updated.shown_count += 1
        self._store_recommendation(updated, now)
        return updated

    def record_recommendation_apply(self, event: RecommendationApplyEvent) -> Recommendation:
        now = self._clock()
        recommendation = self._recommendations.get(event.recommendation_id)
        if recommendation is None:
            raise ValueError(f"unknown recommendation_id {event.recommendation_id}")
        self._require_recommendation_owner(recommendation, event.install_id, event.machine_id)

        updated = copy.copy(recommendation)
        updated.updated_at = now
        updated.apply_status = event.status
        updated.apply_acknowledged_at = now
        updated.applied_fields = dict(event.applied_fields)
        updated.manual_fields = list(event.manual_fields)
        updated.apply_error = event.message if event.status == RecommendationApplyStatus.FAILED else None
        self._store_recommendation(updated, now)
        return updated

    def get_current_recommendation(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        grinder_context_id: str | None = None,
        profile_id: str | None = None,
        raw_profile_hash: str | None = None,
        taste_goal_fingerprint: str | None = None,
    ) -> Recommendation | None:
        return self._recommendations.get_current(
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            now=self._clock(),
            grinder_context_id=grinder_context_id,
            profile_id=profile_id,
            raw_profile_hash=raw_profile_hash if profile_id is None else None,
            taste_goal_fingerprint=taste_goal_fingerprint,
        )

    def handle_machine_state(self, event: MachineStateEvent) -> Recommendation | None:
        if event.community_upload_enabled is not None:
            self.set_community_upload_enabled(
                event.install_id,
                event.machine_id,
                event.community_upload_enabled,
            )
        if event.state not in {MachineState.WAKE, MachineState.IDLE, MachineState.STANDBY}:
            return None
        if not event.bean_context_id:
            return None

        now = self._clock()
        current = self._recommendations.get_current(
            install_id=event.install_id,
            machine_id=event.machine_id,
            bean_context_id=event.bean_context_id,
            now=now,
            grinder_context_id=event.grinder_context_id,
            profile_id=event.profile_id,
            raw_profile_hash=event.raw_profile_hash if event.profile_id is None else None,
            taste_goal_fingerprint=event.taste_goal.fingerprint,
        )
        if current is None:
            return None
        stale = check_recommendation_staleness(
            current,
            now=now,
            bean_context_id=event.bean_context_id,
            grinder_context_id=event.grinder_context_id,
            taste_goal=event.taste_goal,
        )
        if stale.stale:
            self._expire_recommendation(current, now)
            return None
        if current.status in {RecommendationStatus.ACCEPTED, RecommendationStatus.EDITED}:
            return current
        return self._mark_recommendation_shown(current, now)

    def set_community_upload_enabled(
        self,
        install_id: str,
        machine_id: str,
        enabled: bool,
    ) -> None:
        self._community_upload_enabled_by_machine[(install_id, machine_id)] = bool(enabled)

    def persist_generated_recommendation(self, recommendation: Recommendation) -> None:
        now = self._clock()
        recommendation.updated_at = now
        self._recommendations.supersede_active(
            install_id=recommendation.install_id,
            machine_id=recommendation.machine_id,
            bean_context_id=recommendation.bean_context_id,
            now=now,
            except_recommendation_id=recommendation.recommendation_id,
            grinder_context_id=recommendation.grinder_context_id,
            profile_id=recommendation.profile_id,
            raw_profile_hash=(
                recommendation.raw_profile_hash
                if recommendation.profile_id is None
                else None
            ),
            taste_goal_fingerprint=recommendation.taste_goal.fingerprint,
        )
        self._store_recommendation(recommendation, now)

    def supersede_active_recommendation(
        self,
        *,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        grinder_context_id: str | None,
        profile_id: str | None,
        taste_goal_fingerprint: str,
    ) -> None:
        self._recommendations.supersede_active(
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            grinder_context_id=grinder_context_id,
            profile_id=profile_id,
            taste_goal_fingerprint=taste_goal_fingerprint,
            now=self._clock(),
        )

    def community_upload_enabled_for(self, install_id: str, machine_id: str) -> bool:
        return self._community_upload_enabled_by_machine.get(
            (install_id, machine_id),
            self._community_upload_enabled_default,
        )

    def enqueue_comparison_upload(self, comparison: PairwiseShotComparison) -> None:
        if self._upload_queue is None:
            return
        if not self.community_upload_enabled_for(comparison.install_id, comparison.machine_id):
            return
        try:
            self._upload_queue.enqueue(make_comparison_upload_item(comparison, self._clock()))
        except Exception as exc:
            logger.warning(
                "Community upload enqueue failed for comparison %s; local comparison was retained: %s",
                comparison.comparison_id,
                exc,
            )

    def _recommendation_for_event(
        self,
        event: ShotProfileEvent,
        now: int,
        *,
        raw_profile_hash: str | None = None,
    ) -> Recommendation | None:
        profile_scope = _profile_scope_key(
            profile_id=event.profile_id,
            raw_profile_hash=raw_profile_hash,
        )
        if event.recommendation_id:
            recommendation = self._recommendations.get(event.recommendation_id)
            if (
                recommendation is not None
                and recommendation.install_id == event.install_id
                and recommendation.machine_id == event.machine_id
                and _recommendation_matches_profile_scope(recommendation, profile_scope)
                and not check_recommendation_staleness(
                    recommendation,
                    now=now,
                    bean_context_id=event.bean_context_id,
                    grinder_context_id=event.grinder_context_id,
                    taste_goal=event.taste_goal,
                ).stale
            ):
                return recommendation
        recommendation = self._recommendations.get_current(
            install_id=event.install_id,
            machine_id=event.machine_id,
            bean_context_id=event.bean_context_id,
            now=now,
            grinder_context_id=event.grinder_context_id,
            profile_id=event.profile_id,
            raw_profile_hash=raw_profile_hash if event.profile_id is None else None,
            taste_goal_fingerprint=event.taste_goal.fingerprint,
        )
        if recommendation is None or not _recommendation_matches_profile_scope(recommendation, profile_scope):
            return None
        if check_recommendation_staleness(
            recommendation,
            now=now,
            bean_context_id=event.bean_context_id,
            grinder_context_id=event.grinder_context_id,
            taste_goal=event.taste_goal,
        ).stale:
            return None
        return recommendation

    def _decision_from_recommendation(
        self,
        recommendation: Recommendation | None,
    ) -> RecommendationDecision:
        if recommendation is None:
            return RecommendationDecision.UNKNOWN
        if recommendation.status == RecommendationStatus.ACCEPTED:
            return RecommendationDecision.ACCEPTED
        if recommendation.status == RecommendationStatus.EDITED:
            return RecommendationDecision.EDITED
        if recommendation.status == RecommendationStatus.IGNORED:
            return RecommendationDecision.IGNORED
        return RecommendationDecision.UNKNOWN

    def _mark_recommendation_used_if_followed(
        self,
        recommendation: Recommendation,
        state: FollowThroughState,
        now: int,
    ) -> None:
        if state not in {FollowThroughState.FOLLOWED, FollowThroughState.PARTIALLY_FOLLOWED}:
            return
        updated = copy.copy(recommendation)
        updated.status = RecommendationStatus.USED
        updated.used_at = now
        updated.updated_at = now
        self._store_recommendation(updated, now)

    def _mark_recommendation_shown(
        self,
        recommendation: Recommendation,
        now: int,
    ) -> Recommendation:
        updated = copy.copy(recommendation)
        updated.status = RecommendationStatus.SHOWN
        updated.shown_count += 1
        updated.updated_at = now
        self._store_recommendation(updated, now)
        return updated

    def _expire_recommendation(self, recommendation: Recommendation, now: int) -> None:
        updated = copy.copy(recommendation)
        updated.status = RecommendationStatus.EXPIRED
        updated.updated_at = now
        self._store_recommendation(updated, now)

    def _apply_edits(self, recommendation: Recommendation, edited_fields: dict) -> None:
        if "projected_relative_step_from_reference" in edited_fields:
            projected = float(edited_fields["projected_relative_step_from_reference"])
            microns_per_step = (
                recommendation.grind_delta_um_from_current
                / recommendation.grind_delta_steps_from_current
                if recommendation.grind_delta_steps_from_current
                else 0.0
            )
            recommendation.projected_relative_step_from_reference = projected
            if microns_per_step > 0:
                recommendation.projected_relative_grind_um_from_reference = (
                    projected * microns_per_step
                )
        if "next_dose_g" in edited_fields:
            recommendation.next_dose_g = float(edited_fields["next_dose_g"])
        if "target_yield_g" in edited_fields:
            recommendation.target_yield_g = float(edited_fields["target_yield_g"])
        recommendation.target_ratio = recommendation.target_yield_g / recommendation.next_dose_g

    def _store_shot(
        self,
        shot: ShotRecord,
        now: int,
        *,
        community_upload_enabled: bool | None = None,
    ) -> None:
        if community_upload_enabled is not None:
            self.set_community_upload_enabled(
                shot.install_id,
                shot.machine_id,
                community_upload_enabled,
            )
        self._shots.upsert(shot)
        if (
            self._upload_queue is not None
            and self.community_upload_enabled_for(shot.install_id, shot.machine_id)
            and _shot_is_community_uploadable(shot)
        ):
            try:
                self._upload_queue.enqueue(make_shot_upload_item(shot, now))
            except Exception as exc:
                logger.warning(
                    "Community upload enqueue failed for shot %s; local shot was retained: %s",
                    shot.shot_id,
                    exc,
                )

    def _store_recommendation(self, recommendation: Recommendation, now: int) -> None:
        prior = self._recommendations.get(recommendation.recommendation_id)
        self._recommendations.upsert(recommendation)
        if self._upload_queue is None:
            return
        if not self.community_upload_enabled_for(recommendation.install_id, recommendation.machine_id):
            return
        if prior is not None and _recommendation_signature(prior) == _recommendation_signature(recommendation):
            return
        try:
            self._upload_queue.enqueue(make_recommendation_upload_item(recommendation, now))
        except Exception as exc:
            logger.warning(
                "Community upload enqueue failed for recommendation %s; local recommendation was retained: %s",
                recommendation.recommendation_id,
                exc,
            )

    @staticmethod
    def _require_recommendation_owner(
        recommendation: Recommendation,
        install_id: str | None,
        machine_id: str | None,
    ) -> None:
        if install_id is not None and recommendation.install_id != install_id:
            raise ValueError("recommendation event does not match the stored install owner")
        if machine_id is not None and recommendation.machine_id != machine_id:
            raise ValueError("recommendation event does not match the stored machine owner")


def _shot_is_community_uploadable(shot: ShotRecord) -> bool:
    return (
        shot.shot_type == ShotType.ESPRESSO
        and not shot.exclude_from_local_optimization
        and shot.optimization_weight > 0.0
    )


def _same_immutable_shot_event(
    shot: ShotRecord,
    event: ShotProfileEvent,
    effective_profile_hash: str,
) -> bool:
    exact_pairs = (
        (shot.shot_id, event.shot_id),
        (shot.timestamp, int(event.timestamp)),
        (shot.install_id, event.install_id),
        (shot.machine_id, event.machine_id),
        (shot.machine_adapter, event.machine_adapter),
        (shot.raw_profile_hash, effective_profile_hash),
        (shot.bean_context_id, event.bean_context_id),
        (shot.grinder_context_id, event.grinder_context_id),
        (shot.profile_id, event.profile_id),
        (shot.grind_observed, event.grind_observed),
        (shot.dose_observed, event.dose_observed),
        (shot.dose_target_confirmed, event.dose_target_confirmed),
        (shot.target_yield_observed, event.target_yield_observed),
        (shot.weight_source, event.weight_source),
        (shot.beverage_out_observation, event.beverage_out_observation),
        (shot.predictive_stop_applied, event.predictive_stop_applied),
        (shot.shot_end_state, event.shot_end_state),
        (shot.taste_goal.fingerprint, event.taste_goal.fingerprint),
    )
    if any(left != right for left, right in exact_pairs):
        return False
    if event.recommendation_id is not None and shot.recommendation_id != event.recommendation_id:
        return False
    numeric_pairs = (
        (shot.microns_per_step, event.microns_per_step),
        (shot.relative_grind_steps_from_reference, event.relative_grind_steps_from_reference),
        (shot.dose_in_g, event.dose_in_g),
        (shot.dose_target_g, event.dose_target_g),
        (shot.target_yield_g, event.target_yield_g),
        (shot.beverage_out_g, event.beverage_out_g),
        (shot.shot_time_s, event.shot_time_s),
        (shot.predicted_final_beverage_out_g, event.predicted_final_beverage_out_g),
        (shot.predictive_stop_delay_ms, event.predictive_stop_delay_ms),
        (shot.predictive_stop_rate_g_per_s, event.predictive_stop_rate_g_per_s),
        (shot.predictive_stop_lead_g, event.predictive_stop_lead_g),
    )
    return all(_same_optional_number(left, right) for left, right in numeric_pairs)


def _same_optional_number(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)

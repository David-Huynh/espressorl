from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

from espresso_rl.domain.events import (
    MachineStateEvent,
    RecommendationApplyEvent,
    RecommendationDecisionEvent,
    ShotCorrectionEvent,
    ShotFeedbackEvent,
    ShotProfileEvent,
)
from espresso_rl.domain.follow_through import infer_follow_through
from espresso_rl.domain.models import (
    FollowThroughState,
    MachineState,
    Recipe,
    Recommendation,
    RecommendationApplyStatus,
    RecommendationDecision,
    RecommendationStatus,
    SafetyBounds,
    ShotRecord,
    ShotType,
    now_ts,
)
from espresso_rl.domain.optimization import OptimizationContext
from espresso_rl.domain.profile import profile_hash, profile_mse, profile_score, resample_profile_with_quality
from espresso_rl.domain.reward import compute_reward
from espresso_rl.domain.staleness import check_recommendation_staleness
from espresso_rl.domain.utility import classify_shot_profile_event
from espresso_rl.application.upload_payloads import (
    make_recommendation_upload_item,
    make_shot_upload_item,
)
from espresso_rl.ports.optimizers import Optimizer, PriorProvider
from espresso_rl.ports.repositories import (
    RecommendationRepository,
    ShotRepository,
    UploadQueueRepository,
)


@dataclass(frozen=True)
class IngestResult:
    shot: ShotRecord | None
    recommendation: Recommendation | None
    dropped_reason: str | None = None

    @property
    def stored(self) -> bool:
        return self.shot is not None


def _recommendation_signature(recommendation: Recommendation) -> tuple:
    """The *meaningful* state of a recommendation for upload deduplication.

    Two recommendations with the same signature differ only in incidental
    bookkeeping — `shown_count` beyond the first show, `updated_at`, and the
    per-transition timestamps — so they should not produce a fresh community
    upload. Real lifecycle transitions (status/apply changes, the first show, or
    a change to the recommended values/reason) all change the signature.
    """
    return (
        recommendation.status.value,
        recommendation.apply_status.value,
        recommendation.shown_count > 0,  # was_shown: first show matters; later bumps do not
        tuple(sorted(recommendation.applied_fields.items())),
        tuple(sorted(recommendation.manual_fields)),
        recommendation.apply_error,
        recommendation.grind_delta_steps,
        round(recommendation.grind_delta_um, 4),
        round(recommendation.next_grind_steps, 4),
        round(recommendation.next_grind_um, 4),
        round(recommendation.next_dose_g, 4),
        round(recommendation.target_yield_g, 4),
        round(recommendation.target_ratio, 4),
        recommendation.mode.value,
        round(recommendation.confidence, 4),
        recommendation.reason,
    )


class EspressoRLService:
    """Application service over canonical EspressoRL events."""

    def __init__(
        self,
        shots: ShotRepository,
        recommendations: RecommendationRepository,
        optimizer: Optimizer,
        upload_queue: UploadQueueRepository | None = None,
        prior_provider: PriorProvider | None = None,
        safety_bounds: SafetyBounds | None = None,
        clock: Callable[[], int] = now_ts,
    ) -> None:
        self._shots = shots
        self._recommendations = recommendations
        self._optimizer = optimizer
        self._upload_queue = upload_queue
        self._prior_provider = prior_provider
        self._safety_bounds = safety_bounds or SafetyBounds()
        self._clock = clock

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
        recommendation = self._recommendation_for_event(event, now)
        profile_quality = resample_profile_with_quality(event)
        profile = profile_quality.profile
        mse = profile_mse(profile)
        score = profile_score(profile)

        shot = ShotRecord(
            shot_id=event.shot_id,
            timestamp=int(event.timestamp),
            install_id=event.install_id,
            machine_id=event.machine_id,
            machine_adapter=event.machine_adapter,
            profile=profile,
            grinder_step_size_um=event.grinder_step_size_um,
            dose_in_g=event.dose_in_g,
            target_yield_g=event.target_yield_g,
            grind_steps=event.grind_steps,
            beverage_out_g=event.beverage_out_g,
            shot_time_s=event.shot_time_s,
            bean_context_id=event.bean_context_id,
            recommendation_id=recommendation.recommendation_id if recommendation else event.recommendation_id,
            raw_profile_available=len(event.time_ms) >= 2,
            raw_profile_hash=profile_hash(profile),
            profile_mse=mse,
            profile_score=score,
            shot_type=classification.shot_type,
            exclude_from_local_optimization=classification.exclude_from_local_optimization,
            optimization_weight=classification.optimization_weight,
            rating_prompt_allowed=classification.rating_prompt_allowed,
            weight_source=event.weight_source,
            flow_source=event.flow_source,
            flow_units=event.flow_units,
            pump_flow_source=event.pump_flow_source,
            pump_flow_units=event.pump_flow_units,
            pump_flow_calibration_required=event.pump_flow_calibration_required,
            profile_flow_valid=profile_quality.flow_valid,
            profile_flow_masked=profile_quality.flow_masked,
            created_at=now,
            updated_at=now,
        )

        decision = self._decision_from_recommendation(recommendation)
        if recommendation is not None:
            shot.recommended_grind_delta_steps = recommendation.grind_delta_steps
            shot.recommended_grind_delta_um = recommendation.grind_delta_um
            shot.recommended_next_grind_steps = recommendation.next_grind_steps
            shot.recommended_dose_g = recommendation.next_dose_g
            shot.recommended_target_yield_g = recommendation.target_yield_g
            shot.recommended_target_ratio = recommendation.target_ratio
            shot.recommendation_decision = decision
            followed = infer_follow_through(shot, recommendation, decision)
            shot.recommendation_followed = followed.state
            shot.recommendation_attribution_weight = followed.attribution_weight
            self._mark_recommendation_used_if_followed(recommendation, followed.state, now)

        reward = compute_reward(
            human_rating=None,
            profile_score=score,
            follow_through=shot.recommendation_followed,
            taste_tags=shot.taste_tags,
            profile_complete=shot.raw_profile_available,
        )
        shot.reward = reward.reward
        shot.reward_confidence = reward.confidence
        self._store_shot(shot, now)

        if not classification.locally_optimizable:
            return IngestResult(shot=shot, recommendation=None)

        next_rec = self.generate_recommendation(
            install_id=event.install_id,
            machine_id=event.machine_id,
            bean_context_id=event.bean_context_id,
            current_recipe=shot.to_recipe(),
            now=now,
        )
        return IngestResult(shot=shot, recommendation=next_rec)

    def record_feedback(self, event: ShotFeedbackEvent) -> ShotRecord:
        now = self._clock()
        shot = self._shots.get(event.shot_id)
        if shot is None:
            raise ValueError(f"unknown shot_id {event.shot_id}")
        if event.recommendation_id and shot.recommendation_id is None:
            shot.recommendation_id = event.recommendation_id

        shot.human_rating = None if event.skipped else event.rating
        shot.taste_tags = list(event.taste_tags)
        reward = compute_reward(
            human_rating=shot.human_rating,
            profile_score=shot.profile_score or 0.0,
            follow_through=shot.recommendation_followed,
            taste_tags=shot.taste_tags,
            profile_complete=shot.raw_profile_available,
        )
        shot.reward = reward.reward
        shot.reward_confidence = reward.confidence
        shot.updated_at = now
        self._store_shot(shot, now)
        return shot

    def record_shot_correction(self, event: ShotCorrectionEvent) -> ShotRecord:
        now = self._clock()
        shot = self._shots.get(event.shot_id)
        if shot is None:
            raise ValueError(f"unknown shot_id {event.shot_id}")
        if shot.install_id != event.install_id or shot.machine_id != event.machine_id:
            raise ValueError("shot correction does not match the stored shot owner")

        tags = set(event.correction_tags)
        if event.shot_type is not None:
            shot.shot_type = event.shot_type
        if "utility_brew" in tags:
            shot.shot_type = ShotType.UTILITY_FLUSH

        should_exclude = event.exclude_from_local_optimization is True
        if tags.intersection({"bad_puck_prep", "utility_brew"}):
            should_exclude = True
        if shot.shot_type != ShotType.ESPRESSO:
            should_exclude = True

        if event.grind_followed is not None:
            shot.grind_followed = event.grind_followed
            shot.grind_recommendation_trust = 1.0 if event.grind_followed else 0.0
        if event.dose_followed is not None:
            shot.dose_followed = event.dose_followed
            shot.dose_recommendation_trust = 1.0 if event.dose_followed else 0.0
        if event.yield_followed is not None:
            shot.yield_followed = event.yield_followed
            shot.yield_recommendation_trust = 1.0 if event.yield_followed else 0.0

        if "did_not_follow_grind" in tags:
            shot.grind_followed = False
            shot.grind_recommendation_trust = 0.0
        if "did_not_follow_dose" in tags:
            shot.dose_followed = False
            shot.dose_recommendation_trust = 0.0
        if "did_not_follow_yield" in tags:
            shot.yield_followed = False
            shot.yield_recommendation_trust = 0.0

        if "channeling_suspected" in tags or "bad_puck_prep" in tags:
            taste_tags = set(shot.taste_tags)
            taste_tags.add("channeling_suspected")
            shot.taste_tags = sorted(taste_tags)

        variable_follow = [shot.grind_followed, shot.dose_followed, shot.yield_followed]
        any_not_followed = any(value is False for value in variable_follow)
        any_followed = any(value is True for value in variable_follow)

        if should_exclude:
            shot.exclude_from_local_optimization = True
            shot.optimization_weight = 0.0
            shot.recommendation_attribution_weight = 0.0
            shot.grind_recommendation_trust = 0.0
            shot.dose_recommendation_trust = 0.0
            shot.yield_recommendation_trust = 0.0
            if shot.shot_type != ShotType.ESPRESSO:
                shot.rating_prompt_allowed = False
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

        reward = compute_reward(
            human_rating=shot.human_rating,
            profile_score=shot.profile_score or 0.0,
            follow_through=shot.recommendation_followed,
            taste_tags=shot.taste_tags,
            profile_complete=shot.raw_profile_available,
        )
        shot.reward = reward.reward
        shot.reward_confidence = reward.confidence
        shot.updated_at = now
        self._store_shot(shot, now)
        return shot

    def record_recommendation_decision(self, event: RecommendationDecisionEvent) -> Recommendation:
        now = self._clock()
        recommendation = self._recommendations.get(event.recommendation_id)
        if recommendation is None:
            raise ValueError(f"unknown recommendation_id {event.recommendation_id}")

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
    ) -> Recommendation | None:
        return self._recommendations.get_current(
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            now=self._clock(),
        )

    def handle_machine_state(self, event: MachineStateEvent) -> Recommendation | None:
        if event.state not in {MachineState.WAKE, MachineState.IDLE, MachineState.STANDBY}:
            return None
        if not event.bean_context_id:
            return None

        now = self._clock()
        current_recipe = event.current_recipe()
        current = self._recommendations.get_current(
            install_id=event.install_id,
            machine_id=event.machine_id,
            bean_context_id=event.bean_context_id,
            now=now,
        )
        if current is not None:
            stale = check_recommendation_staleness(
                current,
                now=now,
                bean_context_id=event.bean_context_id,
                current_recipe=current_recipe,
            )
            if stale.stale:
                self._expire_recommendation(current, now)
                current = None
            elif current.status in {RecommendationStatus.ACCEPTED, RecommendationStatus.EDITED}:
                return current
            else:
                return self._mark_recommendation_shown(current, now)

        if current_recipe is None:
            return None
        recommendation = self.generate_recommendation(
            install_id=event.install_id,
            machine_id=event.machine_id,
            bean_context_id=event.bean_context_id,
            current_recipe=current_recipe,
            now=now,
        )
        return self._mark_recommendation_shown(recommendation, now)

    def generate_recommendation(
        self,
        install_id: str,
        machine_id: str,
        bean_context_id: str | None,
        current_recipe: Recipe,
        now: int | None = None,
    ) -> Recommendation:
        timestamp = self._clock() if now is None else now
        recent = self._shots.list_recent(
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            limit=200,
        )
        machine_adapter = recent[-1].machine_adapter if recent else None
        last_recommendation = self._recommendations.get_current(
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            now=timestamp,
        ) or self._recommendations.get_latest(
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
        )
        context = OptimizationContext(
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            machine_adapter=machine_adapter,
            current_recipe=current_recipe,
            shots=recent,
            safety_bounds=self._safety_bounds,
            now=timestamp,
            last_recommendation=last_recommendation,
        )
        if self._prior_provider is not None:
            prior_points = self._prior_provider.get_prior_points(context)
            context = OptimizationContext(
                install_id=context.install_id,
                machine_id=context.machine_id,
                bean_context_id=context.bean_context_id,
                machine_adapter=context.machine_adapter,
                current_recipe=context.current_recipe,
                shots=context.shots,
                safety_bounds=context.safety_bounds,
                now=context.now,
                last_recommendation=context.last_recommendation,
                prior_points=tuple(prior_points),
            )
        recommendation = self._optimizer.recommend(context)
        self._recommendations.supersede_active(
            install_id=install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            now=timestamp,
            except_recommendation_id=recommendation.recommendation_id,
        )
        self._store_recommendation(recommendation, timestamp)
        return recommendation

    def _recommendation_for_event(
        self,
        event: ShotProfileEvent,
        now: int,
    ) -> Recommendation | None:
        current_recipe = None
        if event.grind_steps is not None:
            current_recipe = Recipe(
                grind_steps=event.grind_steps,
                grinder_step_size_um=event.grinder_step_size_um,
                dose_g=event.dose_in_g,
                target_yield_g=event.target_yield_g,
            )
        if event.recommendation_id:
            recommendation = self._recommendations.get(event.recommendation_id)
            if recommendation is not None and not check_recommendation_staleness(
                recommendation,
                now=now,
                bean_context_id=event.bean_context_id,
                current_recipe=current_recipe,
            ).stale:
                return recommendation
        recommendation = self._recommendations.get_current(
            install_id=event.install_id,
            machine_id=event.machine_id,
            bean_context_id=event.bean_context_id,
            now=now,
        )
        if recommendation is None:
            return None
        if check_recommendation_staleness(
            recommendation,
            now=now,
            bean_context_id=event.bean_context_id,
            current_recipe=current_recipe,
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

    def _expire_recommendation(
        self,
        recommendation: Recommendation,
        now: int,
    ) -> None:
        updated = copy.copy(recommendation)
        updated.status = RecommendationStatus.EXPIRED
        updated.updated_at = now
        self._store_recommendation(updated, now)

    def _apply_edits(self, recommendation: Recommendation, edited_fields: dict) -> None:
        if "next_grind_steps" in edited_fields:
            next_grind_steps = float(edited_fields["next_grind_steps"])
            step_size_um = recommendation.grind_delta_um / recommendation.grind_delta_steps if recommendation.grind_delta_steps else 0.0
            recommendation.next_grind_steps = next_grind_steps
            if step_size_um > 0:
                recommendation.next_grind_um = next_grind_steps * step_size_um
        if "next_dose_g" in edited_fields:
            recommendation.next_dose_g = float(edited_fields["next_dose_g"])
        if "target_yield_g" in edited_fields:
            recommendation.target_yield_g = float(edited_fields["target_yield_g"])
        if "target_ratio" in edited_fields:
            recommendation.target_ratio = float(edited_fields["target_ratio"])
        else:
            recommendation.target_ratio = recommendation.target_yield_g / recommendation.next_dose_g

    def _store_shot(self, shot: ShotRecord, now: int) -> None:
        self._shots.upsert(shot)
        if self._upload_queue is not None and _shot_is_community_uploadable(shot):
            self._upload_queue.enqueue(make_shot_upload_item(shot, now))

    def _store_recommendation(
        self,
        recommendation: Recommendation,
        now: int,
    ) -> None:
        prior = self._recommendations.get(recommendation.recommendation_id)
        self._recommendations.upsert(recommendation)
        if self._upload_queue is None:
            return
        # Upload only meaningful lifecycle transitions (created/shown/accepted/
        # ignored/edited/used/applied/expired). Skip incidental churn such as
        # repeated shown_count bumps, updated_at-only changes, and idle re-marks.
        if prior is not None and _recommendation_signature(prior) == _recommendation_signature(recommendation):
            return
        self._upload_queue.enqueue(make_recommendation_upload_item(recommendation, now))


def _shot_is_community_uploadable(shot: ShotRecord) -> bool:
    return (
        shot.shot_type == ShotType.ESPRESSO
        and not shot.exclude_from_local_optimization
        and shot.optimization_weight > 0.0
    )

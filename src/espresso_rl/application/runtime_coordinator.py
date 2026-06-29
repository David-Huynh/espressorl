from __future__ import annotations

import logging
from collections.abc import Callable

from espresso_rl.application.services import EspressoRLService, FeedbackResult, IngestResult
from espresso_rl.domain.events import MachineStateEvent, ShotFeedbackEvent, ShotProfileEvent
from espresso_rl.domain.models import Recommendation, ShotRecord
from espresso_rl.ports.runtime import AutoTuningRuntimePublisher

logger = logging.getLogger(__name__)

OutcomeObserver = Callable[[ShotRecord, Recommendation | None], None]


class AutoTuningRuntimeCoordinator:
    """Coordinates canonical auto-tuning use cases and outbound runtime events."""

    def __init__(
        self,
        service: EspressoRLService,
        publisher: AutoTuningRuntimePublisher,
        outcome_observer: OutcomeObserver | None = None,
    ) -> None:
        self._service = service
        self._publisher = publisher
        self._outcome_observer = outcome_observer

    def handle_shot(self, event: ShotProfileEvent) -> IngestResult:
        result = self._service.ingest_shot_profile(event)
        if result.shot is None:
            logger.info(
                "Shot %s dropped before local storage reason=%s",
                event.shot_id,
                result.dropped_reason or "unknown",
            )
            return result

        recommendation = result.recommendation
        if recommendation is None:
            logger.info(
                "Shot %s stored type=%s local_optimization=%s; waiting for feedback before recommendation",
                result.shot.shot_id,
                result.shot.shot_type.value,
                "included" if not result.shot.exclude_from_local_optimization else "excluded",
            )
        else:
            logger.info(
                "Shot %s stored; next rec %s mode=%s grind=%+d dose=%.1f yield=%.1f",
                result.shot.shot_id,
                recommendation.recommendation_id,
                recommendation.mode.value,
                recommendation.grind_delta_steps_from_current,
                recommendation.next_dose_g,
                recommendation.target_yield_g,
            )
            self._observe(result.shot, recommendation)
            self._publisher.publish_recommendation(recommendation)

        self._publisher.publish_status(
            event.machine_id,
            event.bean_context_id,
            event.grinder_context_id,
            profile_id=event.profile_id,
            profile_label=event.profile_label,
            last_shot_id=result.shot.shot_id,
            last_shot_at=result.shot.timestamp,
            last_recommendation_id=(recommendation.recommendation_id if recommendation else None),
            last_recommendation_at=(recommendation.created_at if recommendation else None),
            mode=recommendation.mode.value if recommendation else None,
        )
        return result

    def handle_feedback(self, event: ShotFeedbackEvent) -> FeedbackResult:
        result = self._service.record_feedback(event)
        shot = result.shot
        recommendation = result.recommendation
        logger.info(
            "Feedback for shot %s stored rating=%s reward=%.3f confidence=%.3f",
            shot.shot_id,
            shot.human_rating,
            shot.reward or 0.0,
            shot.reward_confidence,
        )
        if recommendation is not None:
            logger.info(
                "Feedback for shot %s produced next rec %s mode=%s grind=%+d dose=%.1f yield=%.1f",
                shot.shot_id,
                recommendation.recommendation_id,
                recommendation.mode.value,
                recommendation.grind_delta_steps_from_current,
                recommendation.next_dose_g,
                recommendation.target_yield_g,
            )
            self._publisher.publish_recommendation(recommendation)

        self._observe(shot, recommendation)
        self._publisher.publish_status(
            shot.machine_id,
            shot.bean_context_id,
            shot.grinder_context_id,
            profile_id=shot.profile_id,
            profile_label=shot.profile_label,
            last_shot_id=shot.shot_id,
            last_shot_at=shot.timestamp,
            last_recommendation_id=(recommendation.recommendation_id if recommendation else None),
            last_recommendation_at=(recommendation.created_at if recommendation else None),
            mode=recommendation.mode.value if recommendation else None,
        )
        return result

    def handle_machine_state(self, event: MachineStateEvent) -> Recommendation | None:
        recommendation = self._service.handle_machine_state(event)
        if recommendation is not None:
            logger.info(
                "Machine %s state=%s showing recommendation %s",
                event.machine_id,
                event.state.value,
                recommendation.recommendation_id,
            )
            self._publisher.publish_recommendation(recommendation)

        self._publisher.publish_status(
            event.machine_id,
            event.bean_context_id,
            event.grinder_context_id,
            profile_id=event.profile_id,
            profile_label=event.profile_label,
            last_recommendation_id=(recommendation.recommendation_id if recommendation else None),
            last_recommendation_at=(recommendation.updated_at if recommendation else None),
            mode=recommendation.mode.value if recommendation else None,
        )
        return recommendation

    def _observe(self, shot: ShotRecord, recommendation: Recommendation | None) -> None:
        if self._outcome_observer is not None:
            self._outcome_observer(shot, recommendation)

from __future__ import annotations

import logging
from collections.abc import Callable

from espresso_rl.application.live_telemetry import LiveShotTelemetryService
from espresso_rl.application.services import EspressoRLService, IngestResult
from espresso_rl.domain.events import MachineStateEvent, ShotProfileEvent
from espresso_rl.domain.models import Recommendation, ShotRecord
from espresso_rl.ports.runtime import AutoTuningRuntimePublisher

logger = logging.getLogger(__name__)

OutcomeObserver = Callable[[ShotRecord, Recommendation | None], None]
PostShotRecommendation = Callable[[ShotRecord], Recommendation | None]


class AutoTuningRuntimeCoordinator:
    """Coordinates canonical auto-tuning use cases and outbound runtime events."""

    def __init__(
        self,
        service: EspressoRLService,
        publisher: AutoTuningRuntimePublisher,
        outcome_observer: OutcomeObserver | None = None,
        post_shot_recommendation: PostShotRecommendation | None = None,
        live_telemetry: LiveShotTelemetryService | None = None,
    ) -> None:
        self._service = service
        self._publisher = publisher
        self._outcome_observer = outcome_observer
        self._post_shot_recommendation = post_shot_recommendation
        self._live_telemetry = live_telemetry

    def handle_shot(self, event: ShotProfileEvent) -> IngestResult:
        result = self._service.ingest_shot_profile(event)
        if result.shot is None:
            logger.info(
                "Shot %s dropped before local storage reason=%s",
                event.shot_id,
                result.dropped_reason or "unknown",
            )
            return result

        if self._live_telemetry is not None:
            live_session_matched = self._live_telemetry.reconcile_completed_shot(
                result.shot.shot_id,
                result.shot.install_id,
                result.shot.machine_id,
            )
            if not live_session_matched:
                logger.warning(
                    "Discarded conflicting live telemetry for authoritative shot %s",
                    result.shot.shot_id,
                )

        recommendation = result.recommendation
        if recommendation is None and self._post_shot_recommendation is not None:
            recommendation = self._post_shot_recommendation(result.shot)
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

        if not result.replayed and recommendation is None and result.shot.recommendation_id:
            self._publisher.clear_recommendation(event.machine_id)

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
            taste_goal=event.taste_goal,
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
            taste_goal=event.taste_goal,
        )
        return recommendation

    def _observe(self, shot: ShotRecord, recommendation: Recommendation | None) -> None:
        if self._outcome_observer is not None:
            self._outcome_observer(shot, recommendation)

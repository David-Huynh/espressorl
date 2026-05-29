from __future__ import annotations

import logging
import signal
import sys
import threading

from espresso_rl.adapters.gaggimate_mqtt import GaggimateMQTTClient
from espresso_rl.adapters.sqlite_repositories import (
    SQLiteRecommendationRepository,
    SQLiteShotRepository,
    SQLiteStore,
    SQLiteUploadQueueRepository,
)
from espresso_rl.adapters.supabase_upload import (
    SignedSupabaseUploadClient,
    SignedUploadConfig,
    UploadQueueWorker,
)
from espresso_rl.application.services import EspressoRLService
from espresso_rl.config import Config
from espresso_rl.domain.events import (
    MachineStateEvent,
    RecommendationDecisionEvent,
    ShotFeedbackEvent,
    ShotProfileEvent,
)
from espresso_rl.domain.models import Recipe, SafetyBounds
from espresso_rl.optimizers.conservative_bo import ConservativeBOOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-32s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    config = Config.load()
    logger.info("EspressoRL starting [training_mode=%s]", config.training_mode)
    if config.training_mode:
        logger.warning(
            "DreamerV3 training is not wired into the active path yet; BO remains the safe recommendation path."
        )

    store = SQLiteStore(config.data_dir / "espresso_rl.db")
    shot_repo = SQLiteShotRepository(store)
    recommendation_repo = SQLiteRecommendationRepository(store)
    upload_queue_repo = SQLiteUploadQueueRepository(store)
    service = EspressoRLService(
        shots=shot_repo,
        recommendations=recommendation_repo,
        optimizer=ConservativeBOOptimizer(),
        upload_queue=upload_queue_repo if config.community_upload_enabled else None,
        safety_bounds=SafetyBounds(),
        clock=config.now,
    )

    stop_event = threading.Event()
    upload_thread = maybe_start_upload_worker(config, upload_queue_repo, stop_event)

    mqtt_client: GaggimateMQTTClient

    def on_shot(event: ShotProfileEvent) -> None:
        result = service.ingest_shot_profile(event)
        logger.info(
            "Shot %s stored; next rec %s mode=%s grind=%+d dose=%.1f yield=%.1f",
            result.shot.shot_id,
            result.recommendation.recommendation_id,
            result.recommendation.mode.value,
            result.recommendation.grind_delta_steps,
            result.recommendation.next_dose_g,
            result.recommendation.target_yield_g,
        )
        mqtt_client.publish_recommendation(result.recommendation)

    def on_feedback(event: ShotFeedbackEvent) -> None:
        shot = service.record_feedback(event)
        logger.info(
            "Feedback for shot %s stored rating=%s reward=%.3f confidence=%.3f",
            shot.shot_id,
            shot.human_rating,
            shot.reward or 0.0,
            shot.reward_confidence,
        )

    def on_decision(event: RecommendationDecisionEvent) -> None:
        recommendation = service.record_recommendation_decision(event)
        logger.info(
            "Recommendation %s decision stored status=%s",
            recommendation.recommendation_id,
            recommendation.status.value,
        )

    def on_machine_state(event: MachineStateEvent) -> None:
        recommendation = service.handle_machine_state(event)
        if recommendation is None:
            return
        logger.info(
            "Machine %s state=%s showing recommendation %s",
            event.machine_id,
            event.state.value,
            recommendation.recommendation_id,
        )
        mqtt_client.publish_recommendation(recommendation)

    mqtt_client = GaggimateMQTTClient(
        config=config,
        on_shot=on_shot,
        on_feedback=on_feedback,
        on_decision=on_decision,
        on_machine_state=on_machine_state,
    )

    def shutdown(sig: int, frame: object) -> None:
        logger.info("Shutting down (signal %d)", sig)
        stop_event.set()
        mqtt_client.stop()
        if upload_thread is not None:
            upload_thread.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    maybe_publish_startup_recommendation(config, service, mqtt_client)
    mqtt_client.start()
    logger.info("Listening for canonical events via Gaggimate MQTT adapter")
    signal.pause()


def maybe_publish_startup_recommendation(
    config: Config,
    service: EspressoRLService,
    mqtt_client: GaggimateMQTTClient,
) -> None:
    if config.machine_id == "gaggimate:local":
        return
    current = service.get_current_recommendation(
        install_id=config.install_id,
        machine_id=config.machine_id,
        bean_context_id=config.bean_context_id,
    )
    if current is not None:
        mqtt_client.publish_recommendation(current)
        return
    recipe = Recipe(
        grind_steps=config.initial_grind_steps,
        grinder_step_size_um=config.grinder_step_size_um,
        dose_g=config.initial_dose_g,
        target_yield_g=config.initial_target_yield_g,
    )
    recommendation = service.generate_recommendation(
        install_id=config.install_id,
        machine_id=config.machine_id,
        bean_context_id=config.bean_context_id,
        current_recipe=recipe,
    )
    mqtt_client.publish_recommendation(recommendation)


def maybe_start_upload_worker(
    config: Config,
    upload_queue_repo: SQLiteUploadQueueRepository,
    stop_event: threading.Event,
) -> threading.Thread | None:
    if not config.community_upload_enabled:
        logger.info("Community upload disabled; local shot history will still accumulate.")
        return None
    if not config.supabase_ingest_url or not config.upload_secret:
        logger.warning(
            "Community upload enabled but no ingest URL/secret configured; records will queue locally only."
        )
        return None

    client = SignedSupabaseUploadClient(
        SignedUploadConfig(
            ingest_url=config.supabase_ingest_url,
            install_id=config.install_id,
            upload_secret=config.upload_secret,
            upload_token_id=config.upload_token_id,
            max_payload_bytes=config.upload_max_payload_bytes,
        )
    )
    worker = UploadQueueWorker(upload_queue_repo, client, clock=config.now)

    def loop() -> None:
        logger.info("Community upload worker started")
        while not stop_event.is_set():
            try:
                uploaded = worker.run_once()
                if uploaded:
                    logger.info("Uploaded %d queued EspressoRL records", uploaded)
            except Exception:
                logger.exception("Upload worker cycle failed")
            stop_event.wait(config.upload_worker_interval_s)

    thread = threading.Thread(target=loop, name="espresso-rl-upload", daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    main()

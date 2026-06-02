from __future__ import annotations

import logging
import signal
import sys
import threading

from espresso_rl.adapters.gaggimate_mqtt import GaggimateMQTTClient
from espresso_rl.adapters.postgres_repositories import (
    PostgresCommunityWarehouse,
    PostgresRecommendationRepository,
    PostgresShotRepository,
    PostgresStore,
    PostgresUploadQueueRepository,
)
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
from espresso_rl.adapters.supabase_community_queue import (
    SupabaseCommunityQueueClient,
    SupabaseCommunityQueueConfig,
)
from espresso_rl.adapters.supabase_credentials import (
    JsonCommunityCredentialStore,
    SupabaseCredentialRegistrar,
    SupabaseCredentialRegistrarConfig,
)
from espresso_rl.application.admin_pipeline import AdminPipelineService
from espresso_rl.application.community_credentials import CommunityCredentialService
from espresso_rl.application.community_mirror import CommunityMirrorService
from espresso_rl.application.community_priors import CommunityPriorGenerationService
from espresso_rl.application.community_validation import CommunityValidationService
from espresso_rl.application.prior_providers import (
    CommunityPriorProvider,
    CompositePriorProvider,
    LocalHistoryPriorProvider,
    RuleBasedPriorProvider,
)
from espresso_rl.application.services import EspressoRLService
from espresso_rl.application.upload_maintenance import UploadQueueMaintenanceService
from espresso_rl.config import Config
from espresso_rl.domain.community import CommunityUploadCredentials
from espresso_rl.domain.events import (
    MachineStateEvent,
    RecommendationApplyEvent,
    RecommendationDecisionEvent,
    ShotCorrectionEvent,
    ShotFeedbackEvent,
    ShotProfileEvent,
    UploadQueueMaintenanceEvent,
)
from espresso_rl.domain.models import Recipe, SafetyBounds, UploadQueueStatus
from espresso_rl.optimizers.conservative_bo import ConservativeBOOptimizer
from espresso_rl.ports.community import CommunityCredentialRegistrar, CommunityCredentialStore
from espresso_rl.ports.repositories import RecommendationRepository, ShotRepository, UploadQueueRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-32s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    config = Config.load()
    if config.deployment_role == "admin":
        run_admin(config)
        return

    maybe_resolve_community_upload_credentials(config)
    logger.info("EspressoRL starting [training_mode=%s]", config.training_mode)
    if config.training_mode:
        logger.warning(
            "DreamerV3 training is not wired into the active path yet; BO remains the safe recommendation path."
        )

    shot_repo, recommendation_repo, upload_queue_repo = open_repositories(config)
    service = EspressoRLService(
        shots=shot_repo,
        recommendations=recommendation_repo,
        optimizer=ConservativeBOOptimizer(),
        upload_queue=upload_queue_for_service(config, upload_queue_repo),
        prior_provider=open_prior_provider(config),
        safety_bounds=SafetyBounds(),
        clock=config.now,
    )
    upload_maintenance = UploadQueueMaintenanceService(upload_queue_repo, clock=config.now)

    stop_event = threading.Event()
    admin_pipeline = maybe_build_admin_pipeline_service(config)
    upload_thread = maybe_start_upload_worker(config, upload_queue_repo, stop_event)
    collector_thread = maybe_start_admin_collector_worker(
        config,
        stop_event,
        admin_pipeline=admin_pipeline,
    )
    dashboard_thread = maybe_start_admin_dashboard(
        config,
        stop_event,
        admin_pipeline=admin_pipeline,
    )

    mqtt_client: GaggimateMQTTClient

    def publish_status(
        machine_id: str,
        bean_context_id: str | None,
        *,
        last_shot_id: str | None = None,
        last_shot_at: int | None = None,
        last_recommendation_id: str | None = None,
        last_recommendation_at: int | None = None,
        mode: str | None = None,
    ) -> None:
        status = build_status_payload(
            config=config,
            service=service,
            upload_maintenance=upload_maintenance,
            shot_repo=shot_repo,
            upload_queue_repo=upload_queue_repo,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            last_shot_id=last_shot_id,
            last_shot_at=last_shot_at,
            last_recommendation_id=last_recommendation_id,
            last_recommendation_at=last_recommendation_at,
            mode=mode,
        )
        mqtt_client.publish_status(machine_id, status)

    def on_shot(event: ShotProfileEvent) -> None:
        result = service.ingest_shot_profile(event)
        if result.recommendation is None:
            logger.info(
                "Shot %s stored type=%s local_optimization=%s; no BO recommendation generated",
                result.shot.shot_id,
                result.shot.shot_type.value,
                "included" if not result.shot.exclude_from_local_optimization else "excluded",
            )
            publish_status(
                event.machine_id,
                event.bean_context_id,
                last_shot_id=result.shot.shot_id,
                last_shot_at=result.shot.timestamp,
            )
            return
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
        publish_status(
            event.machine_id,
            event.bean_context_id,
            last_shot_id=result.shot.shot_id,
            last_shot_at=result.shot.timestamp,
            last_recommendation_id=result.recommendation.recommendation_id,
            last_recommendation_at=result.recommendation.created_at,
            mode=result.recommendation.mode.value,
        )

    def on_feedback(event: ShotFeedbackEvent) -> None:
        shot = service.record_feedback(event)
        logger.info(
            "Feedback for shot %s stored rating=%s reward=%.3f confidence=%.3f",
            shot.shot_id,
            shot.human_rating,
            shot.reward or 0.0,
            shot.reward_confidence,
        )
        publish_status(
            shot.machine_id,
            shot.bean_context_id,
            last_shot_id=shot.shot_id,
            last_shot_at=shot.timestamp,
        )

    def on_correction(event: ShotCorrectionEvent) -> None:
        shot = service.record_shot_correction(event)
        logger.info(
            "Correction for shot %s stored type=%s excluded=%s followed=%s attribution=%.2f",
            shot.shot_id,
            shot.shot_type.value,
            shot.exclude_from_local_optimization,
            shot.recommendation_followed.value,
            shot.recommendation_attribution_weight,
        )
        publish_status(
            shot.machine_id,
            shot.bean_context_id,
            last_shot_id=shot.shot_id,
            last_shot_at=shot.timestamp,
        )

    def on_upload_maintenance(event: UploadQueueMaintenanceEvent) -> None:
        result = upload_maintenance.requeue_valid_rejected(limit=event.limit)
        logger.info(
            "Upload queue maintenance action=%s inspected=%d requeued=%d skipped=%d",
            event.action,
            result.inspected,
            result.requeued,
            result.skipped,
        )
        publish_status(event.machine_id, event.bean_context_id)

    def on_decision(event: RecommendationDecisionEvent) -> None:
        recommendation = service.record_recommendation_decision(event)
        logger.info(
            "Recommendation %s decision stored status=%s",
            recommendation.recommendation_id,
            recommendation.status.value,
        )
        publish_status(
            recommendation.machine_id,
            recommendation.bean_context_id,
            last_recommendation_id=recommendation.recommendation_id,
            last_recommendation_at=recommendation.updated_at,
            mode=recommendation.mode.value,
        )

    def on_apply(event: RecommendationApplyEvent) -> None:
        recommendation = service.record_recommendation_apply(event)
        logger.info(
            "Recommendation %s apply acknowledgement stored apply_status=%s manual_fields=%s",
            recommendation.recommendation_id,
            recommendation.apply_status.value,
            ",".join(recommendation.manual_fields),
        )
        publish_status(
            recommendation.machine_id,
            recommendation.bean_context_id,
            last_recommendation_id=recommendation.recommendation_id,
            last_recommendation_at=recommendation.updated_at,
            mode=recommendation.mode.value,
        )

    def on_machine_state(event: MachineStateEvent) -> None:
        recommendation = service.handle_machine_state(event)
        if recommendation is None:
            publish_status(event.machine_id, event.bean_context_id)
            return
        logger.info(
            "Machine %s state=%s showing recommendation %s",
            event.machine_id,
            event.state.value,
            recommendation.recommendation_id,
        )
        mqtt_client.publish_recommendation(recommendation)
        publish_status(
            event.machine_id,
            event.bean_context_id,
            last_recommendation_id=recommendation.recommendation_id,
            last_recommendation_at=recommendation.updated_at,
            mode=recommendation.mode.value,
        )

    mqtt_client = GaggimateMQTTClient(
        config=config,
        on_shot=on_shot,
        on_feedback=on_feedback,
        on_correction=on_correction,
        on_upload_maintenance=on_upload_maintenance,
        on_decision=on_decision,
        on_apply=on_apply,
        on_machine_state=on_machine_state,
    )

    def shutdown(sig: int, frame: object) -> None:
        logger.info("Shutting down (signal %d)", sig)
        stop_event.set()
        mqtt_client.stop()
        if upload_thread is not None:
            upload_thread.join(timeout=5)
        if collector_thread is not None:
            collector_thread.join(timeout=5)
        if dashboard_thread is not None:
            dashboard_thread.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    mqtt_client.start()
    maybe_publish_startup_recommendation(
        config,
        service,
        mqtt_client,
        upload_maintenance=upload_maintenance,
        shot_repo=shot_repo,
        upload_queue_repo=upload_queue_repo,
    )
    logger.info("Listening for canonical events via Gaggimate MQTT adapter")
    signal.pause()


def run_admin(config: Config) -> None:
    logger.info("EspressoRL admin runtime starting")
    stop_event = threading.Event()
    admin_pipeline = maybe_build_admin_pipeline_service(config)
    collector_thread = maybe_start_admin_collector_worker(
        config,
        stop_event,
        admin_pipeline=admin_pipeline,
    )
    dashboard_thread = maybe_start_admin_dashboard(
        config,
        stop_event,
        admin_pipeline=admin_pipeline,
    )

    def shutdown(sig: int, frame: object) -> None:
        logger.info("Shutting down admin runtime (signal %d)", sig)
        stop_event.set()
        if collector_thread is not None:
            collector_thread.join(timeout=5)
        if dashboard_thread is not None:
            dashboard_thread.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    if collector_thread is None and dashboard_thread is None:
        logger.warning("Admin runtime has no enabled collector or dashboard; waiting for shutdown.")
    signal.pause()


def maybe_publish_startup_recommendation(
    config: Config,
    service: EspressoRLService,
    mqtt_client: GaggimateMQTTClient,
    upload_maintenance: UploadQueueMaintenanceService | None = None,
    shot_repo: ShotRepository | None = None,
    upload_queue_repo: UploadQueueRepository | None = None,
) -> None:
    if config.machine_id == "gaggimate:local":
        return
    if not config.bean_context_id:
        return
    current = service.get_current_recommendation(
        install_id=config.install_id,
        machine_id=config.machine_id,
        bean_context_id=config.bean_context_id,
    )
    if current is not None:
        mqtt_client.publish_recommendation(current)
        mqtt_client.publish_status(
            config.machine_id,
            build_status_payload(
                config=config,
                service=service,
                upload_maintenance=upload_maintenance,
                shot_repo=shot_repo,
                upload_queue_repo=upload_queue_repo,
                machine_id=config.machine_id,
                bean_context_id=config.bean_context_id,
                last_recommendation_id=current.recommendation_id,
                last_recommendation_at=current.updated_at,
                mode=current.mode.value,
            ),
        )
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
    mqtt_client.publish_status(
        config.machine_id,
        build_status_payload(
            config=config,
            service=service,
            upload_maintenance=upload_maintenance,
            shot_repo=shot_repo,
            upload_queue_repo=upload_queue_repo,
            machine_id=config.machine_id,
            bean_context_id=config.bean_context_id,
            last_recommendation_id=recommendation.recommendation_id,
            last_recommendation_at=recommendation.created_at,
            mode=recommendation.mode.value,
        ),
    )


def build_status_payload(
    config: Config,
    service: EspressoRLService,
    upload_maintenance: UploadQueueMaintenanceService | None,
    shot_repo: ShotRepository | None,
    upload_queue_repo: UploadQueueRepository | None,
    machine_id: str,
    bean_context_id: str | None,
    *,
    last_shot_id: str | None = None,
    last_shot_at: int | None = None,
    last_recommendation_id: str | None = None,
    last_recommendation_at: int | None = None,
    mode: str | None = None,
) -> dict:
    now = config.now()
    recent = (
        shot_repo.list_recent(
            install_id=config.install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            limit=1_000_000,
        )
        if shot_repo is not None
        else []
    )
    if recent and last_shot_id is None:
        last_shot = recent[-1]
        last_shot_id = last_shot.shot_id
        last_shot_at = last_shot.timestamp

    optimizer_shots = [
        shot
        for shot in recent
        if shot.shot_type.value == "espresso"
        and not shot.exclude_from_local_optimization
        and shot.optimization_weight > 0.0
    ]
    rated_shots = [shot for shot in optimizer_shots if shot.human_rating is not None]

    current = service.get_current_recommendation(
        install_id=config.install_id,
        machine_id=machine_id,
        bean_context_id=bean_context_id,
    )
    if current is not None:
        last_recommendation_id = last_recommendation_id or current.recommendation_id
        last_recommendation_at = last_recommendation_at or current.updated_at
        mode = mode or current.mode.value
        apply_status = current.apply_status.value
    else:
        apply_status = None

    upload_queue_count = 0
    upload_queue_status_counts: dict[str, int] = {}
    if upload_queue_repo is not None:
        upload_queue_count = len(upload_queue_repo.list_ready(now=now, limit=1_000_000))
        upload_queue_status_counts = {
            status.value: count for status, count in upload_queue_repo.count_by_status().items()
        }
    latest_rejected = upload_maintenance.latest_rejected() if upload_maintenance is not None else None

    return {
        "addon_online": True,
        "install_id": config.install_id,
        "timestamp": now,
        "last_shot_id": last_shot_id,
        "last_shot_at": last_shot_at,
        "last_recommendation_id": last_recommendation_id,
        "last_recommendation_at": last_recommendation_at,
        "recommendation_apply_status": apply_status,
        "mode": mode,
        "local_shot_count": len(optimizer_shots),
        "rated_shot_count": len(rated_shots),
        "best_known_recipe": best_known_recipe_payload(optimizer_shots),
        "upload_queue_count": upload_queue_count,
        "upload_queue_pending_count": upload_queue_status_counts.get(UploadQueueStatus.PENDING.value, 0),
        "upload_queue_failed_count": upload_queue_status_counts.get(UploadQueueStatus.FAILED.value, 0),
        "upload_queue_rejected_count": upload_queue_status_counts.get(UploadQueueStatus.REJECTED.value, 0),
        "upload_queue_uploaded_count": upload_queue_status_counts.get(UploadQueueStatus.UPLOADED.value, 0),
        "upload_queue_last_rejected_id": latest_rejected.upload_id if latest_rejected else None,
        "upload_queue_last_rejected_record_type": latest_rejected.local_record_type if latest_rejected else None,
        "upload_queue_last_rejected_record_id": latest_rejected.local_record_id if latest_rejected else None,
        "upload_queue_last_rejected_error": latest_rejected.error_message if latest_rejected else None,
        "upload_queue_last_rejected_at": latest_rejected.updated_at if latest_rejected else None,
        "community_upload_enabled": config.should_enqueue_community_uploads(),
    }


def best_known_recipe_payload(shots: list) -> dict | None:
    if not shots:
        return None

    def score(shot) -> float:
        if shot.human_rating is not None:
            return 10.0 + shot.human_rating + (shot.reward or 0.0)
        if shot.reward is not None:
            return shot.reward * max(shot.reward_confidence, 0.05)
        return shot.profile_score or 0.0

    best = max(shots, key=score)
    return {
        "shot_id": best.shot_id,
        "rating": best.human_rating,
        "grind_steps": best.grind_steps,
        "dose_g": best.dose_in_g,
        "target_yield_g": best.target_yield_g,
        "target_ratio": best.target_ratio,
        "reward": best.reward,
    }


def maybe_start_upload_worker(
    config: Config,
    upload_queue_repo: UploadQueueRepository,
    stop_event: threading.Event,
) -> threading.Thread | None:
    if config.deployment_role == "admin":
        logger.info("Admin deployment role selected; community upload push is disabled to prevent duplicate data.")
        return None
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


def maybe_resolve_community_upload_credentials(
    config: Config,
    *,
    store: CommunityCredentialStore | None = None,
    registrar: CommunityCredentialRegistrar | None = None,
) -> CommunityUploadCredentials | None:
    if config.deployment_role == "admin":
        logger.info("Admin deployment role selected; community upload registration is disabled.")
        return None
    if not config.community_upload_enabled:
        return None

    configured = _configured_upload_credentials(config)
    if configured is not None:
        return _apply_upload_credentials(config, configured)

    credential_store = store or JsonCommunityCredentialStore(config.data_dir / "community_upload_credentials.json")
    stored = credential_store.load()
    if stored is not None:
        logger.info("Loaded stored community upload credentials for install_id=%s", stored.install_id)
        return _apply_upload_credentials(config, stored)

    if registrar is None and not config.supabase_registration_url:
        logger.warning(
            "Community upload enabled but no upload credentials or registration URL are configured; "
            "records will queue locally only."
        )
        return None

    credential_registrar = registrar or SupabaseCredentialRegistrar(
        SupabaseCredentialRegistrarConfig(registration_url=config.supabase_registration_url)
    )
    try:
        credentials = CommunityCredentialService(
            store=credential_store,
            registrar=credential_registrar,
        ).resolve_for_upload(allow_registration=True)
    except Exception as exc:
        logger.warning(
            "Community upload registration failed; records will queue locally only: %s",
            exc,
        )
        return None
    if credentials is None:
        return None
    logger.info("Registered community upload credentials for install_id=%s", credentials.install_id)
    return _apply_upload_credentials(config, credentials)


def _configured_upload_credentials(config: Config) -> CommunityUploadCredentials | None:
    if not config.upload_secret:
        return None
    return CommunityUploadCredentials(
        install_id=config.install_id,
        upload_token_id=config.upload_token_id,
        upload_secret=config.upload_secret,
    )


def _apply_upload_credentials(
    config: Config,
    credentials: CommunityUploadCredentials,
) -> CommunityUploadCredentials:
    config.install_id = credentials.install_id
    config.upload_token_id = credentials.upload_token_id
    config.upload_secret = credentials.upload_secret
    return credentials


def maybe_start_admin_collector_worker(
    config: Config,
    stop_event: threading.Event,
    admin_pipeline: AdminPipelineService | None = None,
) -> threading.Thread | None:
    if config.deployment_role != "admin":
        return None
    if not config.admin_collector_enabled:
        logger.info("Admin collector disabled.")
        return None
    if config.storage_backend != "postgres":
        logger.warning("Admin collector requires Postgres storage.")
        return None
    if not config.supabase_rest_url or not config.supabase_service_role_key:
        logger.warning("Admin collector enabled but Supabase REST URL/service-role key is missing.")
        return None

    pipeline = admin_pipeline or build_admin_pipeline_service(config)

    def loop() -> None:
        logger.info("Admin community mirror/validation worker started")
        while not stop_event.is_set():
            try:
                mirror_action = pipeline.mirror_once(
                    limit=config.admin_collector_batch_size,
                    requested_by="admin_collector",
                )
                result = mirror_action.mirror
                if mirror_action.already_running:
                    logger.info("Admin mirror skipped because the job is already running")
                elif result is not None and result.claimed:
                    logger.info(
                        "Mirrored community uploads claimed=%d mirrored=%d failed=%d",
                        result.claimed,
                        result.mirrored,
                        result.failed,
                    )
                validation_action = pipeline.validate_once(
                    limit=config.admin_collector_batch_size,
                    requested_by="admin_collector",
                )
                validation = validation_action.validation
                if validation_action.already_running:
                    logger.info("Admin validation skipped because the job is already running")
                elif validation is not None and validation.processed:
                    logger.info(
                        "Validated community uploads processed=%d shots=%d recommendations=%d rejected=%d training_rows=%d",
                        validation.processed,
                        validation.validated_shots,
                        validation.stored_recommendations,
                        validation.rejected,
                        validation.training_rows,
                    )
                prior_action = pipeline.generate_priors_once(
                    limit=max(config.admin_collector_batch_size * 50, 5000),
                    requested_by="admin_collector",
                )
                priors = prior_action.priors
                if prior_action.already_running:
                    logger.info("Admin prior generation skipped because the job is already running")
                elif priors is not None and priors.priors_written:
                    logger.info(
                        "Generated community priors examined=%d eligible=%d rejected=%d contexts=%d written=%d",
                        priors.examined,
                        priors.eligible,
                        priors.rejected,
                        priors.contexts_seen,
                        priors.priors_written,
                    )
            except Exception:
                logger.exception("Admin community mirror/validation cycle failed")
            stop_event.wait(config.admin_collector_interval_s)

    thread = threading.Thread(target=loop, name="espresso-rl-admin-mirror", daemon=True)
    thread.start()
    return thread


def maybe_start_admin_dashboard(
    config: Config,
    stop_event: threading.Event,
    admin_pipeline: AdminPipelineService | None = None,
) -> threading.Thread | None:
    if config.deployment_role != "admin":
        return None
    if not config.admin_dashboard_enabled:
        logger.info("Admin dashboard disabled.")
        return None
    if config.storage_backend != "postgres":
        logger.warning("Admin dashboard requires Postgres storage.")
        return None
    if len(config.admin_dashboard_token) < 32:
        logger.warning(
            "Admin dashboard enabled but ESPRESSORL_ADMIN_DASHBOARD_TOKEN/admin_dashboard_token is missing or too short."
        )
        return None

    service = admin_pipeline or build_admin_pipeline_service(config)
    from espresso_rl.adapters.admin_dashboard import start_admin_dashboard

    logger.info(
        "Starting admin dashboard on %s:%d",
        config.admin_dashboard_host,
        config.admin_dashboard_port,
    )
    return start_admin_dashboard(
        service,
        admin_token=config.admin_dashboard_token,
        host=config.admin_dashboard_host,
        port=config.admin_dashboard_port,
        stop_event=stop_event,
    )


def maybe_build_admin_pipeline_service(config: Config) -> AdminPipelineService | None:
    if config.deployment_role != "admin":
        return None
    if config.storage_backend != "postgres":
        return None
    collector_ready = (
        config.admin_collector_enabled
        and bool(config.supabase_rest_url)
        and bool(config.supabase_service_role_key)
    )
    dashboard_ready = config.admin_dashboard_enabled and len(config.admin_dashboard_token) >= 32
    if not collector_ready and not dashboard_ready:
        return None
    return build_admin_pipeline_service(config)


def build_admin_pipeline_service(config: Config) -> AdminPipelineService:
    warehouse = PostgresCommunityWarehouse(PostgresStore(config.postgres_dsn))
    mirror = None
    if config.supabase_rest_url and config.supabase_service_role_key:
        source = SupabaseCommunityQueueClient(
            SupabaseCommunityQueueConfig(
                rest_url=config.supabase_rest_url,
                service_role_key=config.supabase_service_role_key,
                admin_id=config.admin_collector_id,
                claim_lease_seconds=config.admin_collector_lease_seconds,
            )
        )
        mirror = CommunityMirrorService(source=source, warehouse=warehouse)
    return AdminPipelineService(
        warehouse=warehouse,
        mirror=mirror,
        validator=CommunityValidationService(warehouse=warehouse),
        prior_generator=CommunityPriorGenerationService(warehouse=warehouse),
        clock=config.now,
    )


def upload_queue_for_service(
    config: Config,
    upload_queue_repo: UploadQueueRepository,
) -> UploadQueueRepository | None:
    if not config.should_enqueue_community_uploads():
        return None
    return upload_queue_repo


def open_prior_provider(config: Config) -> CompositePriorProvider:
    providers = [LocalHistoryPriorProvider(), RuleBasedPriorProvider()]
    if config.storage_backend == "postgres":
        providers.append(
            CommunityPriorProvider(
                PostgresCommunityWarehouse(PostgresStore(config.postgres_dsn)),
            )
        )
    return CompositePriorProvider(providers)


def open_repositories(
    config: Config,
) -> tuple[ShotRepository, RecommendationRepository, UploadQueueRepository]:
    if config.storage_backend == "postgres":
        logger.info("Using Postgres storage backend")
        store = PostgresStore(config.postgres_dsn)
        return (
            PostgresShotRepository(store),
            PostgresRecommendationRepository(store),
            PostgresUploadQueueRepository(store),
        )

    logger.warning("Using SQLite storage backend; Postgres is the intended container/admin runtime backend.")
    store = SQLiteStore(config.data_dir / "espresso_rl.db")
    return (
        SQLiteShotRepository(store),
        SQLiteRecommendationRepository(store),
        SQLiteUploadQueueRepository(store),
    )


if __name__ == "__main__":
    main()

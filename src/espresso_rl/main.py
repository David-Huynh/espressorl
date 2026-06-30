from __future__ import annotations

import logging
import signal
import sys
import threading
from urllib.parse import urlsplit, urlunsplit

from espresso_rl.adapters.gaggimate_mqtt import GaggimateMQTTClient
from espresso_rl.adapters.postgres_repositories import (
    PostgresCommunityWarehouse,
    PostgresLocalDataRepository,
    PostgresRecommendationRepository,
    PostgresShadowEvaluationRepository,
    PostgresShadowQualityReportRepository,
    PostgresShotRepository,
    PostgresStore,
    PostgresUploadQueueRepository,
)
from espresso_rl.adapters.sqlite_repositories import (
    SQLiteLocalDataRepository,
    SQLiteRecommendationRepository,
    SQLiteShadowEvaluationRepository,
    SQLiteShadowQualityReportRepository,
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
from espresso_rl.adapters.file_artifacts import LocalTextArtifactWriter
from espresso_rl.adapters.local_model_store import LocalModelArtifactStore
from espresso_rl.application.admin_pipeline import AdminPipelineService
from espresso_rl.application.checkpoint_loading import CheckpointLoadError, load_verified_dreamer_checkpoint
from espresso_rl.application.dreamer_shadow_inference import (
    DreamerShadowInferenceError,
    build_dreamer_shadow_inference_session,
)
from espresso_rl.application.dreamer_recommendations import DreamerRecommendationService
from espresso_rl.application.dreamer_shadow_evaluation import (
    DreamerShadowEvaluationError,
    DreamerShadowEvaluationService,
)
from espresso_rl.application.dreamer_shadow_quality import (
    DreamerShadowQualityError,
    DreamerShadowQualityReportService,
)
from espresso_rl.application.community_credentials import CommunityCredentialService
from espresso_rl.application.community_mirror import CommunityMirrorService
from espresso_rl.application.community_priors import CommunityPriorGenerationService
from espresso_rl.application.community_validation import CommunityValidationService
from espresso_rl.application.local_data import LocalDataService
from espresso_rl.application.training_export import TrainingDatasetExportService
from espresso_rl.application.training_export import local_training_transition_from_shot
from espresso_rl.application.prior_providers import (
    CommunityPriorProvider,
    CompositePriorProvider,
    LocalHistoryPriorProvider,
)
from espresso_rl.application.runtime_coordinator import AutoTuningRuntimeCoordinator
from espresso_rl.application.services import EspressoRLService
from espresso_rl.application.upload_maintenance import UploadQueueMaintenanceService
from espresso_rl.config import Config
from espresso_rl.domain.community import CommunityUploadCredentials
from espresso_rl.domain.events import (
    LocalResetEvent,
    OptimizerSettingsEvent,
    RecommendationApplyEvent,
    RecommendationDecisionEvent,
    ShotCorrectionEvent,
    UploadQueueMaintenanceEvent,
)
from espresso_rl.domain.models import Recommendation, SafetyBounds, UploadQueueStatus
from espresso_rl.domain.model_checkpoint import VerifiedDreamerCheckpoint
from espresso_rl.domain.optimization import (
    DEFAULT_OPTIMIZER_MODE,
    OPTIMIZER_MODE_DREAMER_V3_ACTIVE,
    OPTIMIZER_MODE_DREAMER_V3_SHADOW,
)
from espresso_rl.dreamer.dataset import DREAMER_CONTEXT_WINDOW_SIZE
from espresso_rl.optimizers.runtime import RuntimeOptimizer, verify_model_artifact, verify_model_manifest_file
from espresso_rl.ports.community import CommunityCredentialRegistrar, CommunityCredentialStore
from espresso_rl.ports.repositories import LocalDataRepository, RecommendationRepository, ShotRepository, UploadQueueRepository
from espresso_rl.ports.shadow_evaluations import ShadowEvaluationRepository
from espresso_rl.ports.shadow_quality_reports import ShadowQualityReportRepository

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
    logger.info(
        "EspressoRL starting [training_mode=%s optimizer_mode=%s]",
        config.training_mode,
        config.optimizer_mode,
    )
    if config.training_mode:
        logger.warning(
            "DreamerV3 training is not wired into the active path yet; BO remains the safe recommendation path."
        )

    (
        shot_repo,
        recommendation_repo,
        upload_queue_repo,
        local_data_repo,
        shadow_evaluation_repo,
        shadow_quality_report_repo,
    ) = open_repositories(config)
    verified_checkpoint, checkpoint_unavailable_reason = load_configured_dreamer_checkpoint(config)
    checkpoint_inference_parity_verified = False
    checkpoint_inference_parity_reason = None
    dreamer_shadow_session = None
    if verified_checkpoint is not None:
        try:
            dreamer_shadow_session = build_dreamer_shadow_inference_session(verified_checkpoint)
            checkpoint_inference_parity_verified = dreamer_shadow_session.status.parity_verified
        except DreamerShadowInferenceError as exc:
            checkpoint_inference_parity_reason = str(exc)
            logger.warning("DreamerV3 checkpoint inference parity failed: %s", exc)
    dreamer_recommendation_service = (
        DreamerRecommendationService(
            session=dreamer_shadow_session,
            safety_bounds=SafetyBounds(),
        )
        if dreamer_shadow_session is not None
        else None
    )
    runtime_optimizer = RuntimeOptimizer(
        optimizer_mode=config.optimizer_mode,
        model_artifact_path=config.optimizer_model_artifact_path,
        model_artifact_sha256=config.optimizer_model_artifact_sha256,
        model_manifest_path=config.optimizer_model_manifest_path,
        model_artifact_max_bytes=config.optimizer_model_artifact_max_bytes,
        verified_checkpoint=verified_checkpoint,
        checkpoint_unavailable_reason=checkpoint_unavailable_reason,
        checkpoint_inference_parity_verified=checkpoint_inference_parity_verified,
        checkpoint_inference_parity_reason=checkpoint_inference_parity_reason,
        dreamer_optimizer=dreamer_recommendation_service,
    )
    shadow_evaluation_service = (
        DreamerShadowEvaluationService(
            session=dreamer_shadow_session,
            repository=shadow_evaluation_repo,
            safety_bounds=SafetyBounds(),
            clock=config.now,
        )
        if dreamer_shadow_session is not None and config.optimizer_mode != OPTIMIZER_MODE_DREAMER_V3_ACTIVE
        else None
    )
    shadow_quality_service = (
        DreamerShadowQualityReportService(
            evaluations=shadow_evaluation_repo,
            reports=shadow_quality_report_repo,
            checkpoint_artifact_sha256=dreamer_shadow_session.status.checkpoint_artifact_sha256,
            checkpoint_inference_probe_sha256=dreamer_shadow_session.status.inference_probe_sha256,
            inference_contract_id=dreamer_shadow_session.status.inference_contract_id,
            clock=config.now,
        )
        if dreamer_shadow_session is not None and config.optimizer_mode != OPTIMIZER_MODE_DREAMER_V3_ACTIVE
        else None
    )
    service = EspressoRLService(
        shots=shot_repo,
        recommendations=recommendation_repo,
        optimizer=runtime_optimizer,
        upload_queue=upload_queue_for_service(config, upload_queue_repo),
        prior_provider=open_prior_provider(config),
        safety_bounds=SafetyBounds(),
        clock=config.now,
        community_upload_enabled_default=False,
    )
    upload_maintenance = UploadQueueMaintenanceService(upload_queue_repo, clock=config.now)
    local_data_service = LocalDataService(
        local_data_repo,
        install_id=config.install_id,
        machine_id=config.machine_id,
        clock=config.now,
    )

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
    local_dashboard_thread = maybe_start_local_dashboard(
        config,
        stop_event,
        local_data_service=local_data_service,
        upload_maintenance=upload_maintenance,
    )

    mqtt_client: GaggimateMQTTClient

    def record_shadow_evaluation(shot, recommendation) -> None:
        result = try_record_shadow_evaluation(
            shadow_evaluation_service,
            shot=shot,
            recommendation=recommendation,
            shot_repo=shot_repo,
        )
        if result is not None:
            try_build_shadow_quality_report(shadow_quality_service, result.evaluation)

    def publish_status(
        machine_id: str,
        bean_context_id: str | None,
        grinder_context_id: str | None = None,
        *,
        profile_id: str | None = None,
        profile_label: str | None = None,
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
            grinder_context_id=grinder_context_id,
            profile_id=profile_id,
            profile_label=profile_label,
            last_shot_id=last_shot_id,
            last_shot_at=last_shot_at,
            last_recommendation_id=last_recommendation_id,
            last_recommendation_at=last_recommendation_at,
            mode=mode,
            optimizer_status=runtime_optimizer.status().to_dict(),
            shadow_evaluation_service=shadow_evaluation_service,
            shadow_quality_service=shadow_quality_service,
        )
        mqtt_client.publish_status(machine_id, status)

    class RuntimePublisher:
        def publish_recommendation(self, recommendation: Recommendation) -> None:
            mqtt_client.publish_recommendation(recommendation)

        def publish_status(
            self,
            machine_id: str,
            bean_context_id: str | None,
            grinder_context_id: str | None,
            **kwargs,
        ) -> None:
            publish_status(machine_id, bean_context_id, grinder_context_id, **kwargs)

    runtime_coordinator = AutoTuningRuntimeCoordinator(
        service=service,
        publisher=RuntimePublisher(),
        outcome_observer=record_shadow_evaluation,
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
            shot.grinder_context_id,
            profile_id=shot.profile_id,
            profile_label=shot.profile_label,
            last_shot_id=shot.shot_id,
            last_shot_at=shot.timestamp,
        )

    def on_upload_maintenance(event: UploadQueueMaintenanceEvent) -> None:
        if event.action == "purge_rejected":
            result = upload_maintenance.purge_rejected(limit=event.limit, local_record_id=event.local_record_id)
            logger.info(
                "Upload queue maintenance action=%s local_record_id=%s inspected=%d purged_uploads=%d purged_shots=%d "
                "purged_recommendations=%d kept_linked_records=%d",
                event.action,
                event.local_record_id,
                result.inspected,
                result.purged_uploads,
                result.purged_shots,
                result.purged_recommendations,
                result.kept_linked_records,
            )
        else:
            result = upload_maintenance.requeue_valid_rejected(limit=event.limit)
            logger.info(
                "Upload queue maintenance action=%s inspected=%d requeued=%d skipped=%d",
                event.action,
                result.inspected,
                result.requeued,
                result.skipped,
            )
        publish_status(event.machine_id, event.bean_context_id, event.grinder_context_id)

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
            recommendation.grinder_context_id,
            profile_id=recommendation.profile_id,
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
            recommendation.grinder_context_id,
            profile_id=recommendation.profile_id,
            last_recommendation_id=recommendation.recommendation_id,
            last_recommendation_at=recommendation.updated_at,
            mode=recommendation.mode.value,
        )

    def on_optimizer_settings(event: OptimizerSettingsEvent) -> None:
        if event.install_id != config.install_id or not _same_machine_id(event.machine_id, config.machine_id):
            logger.warning(
                "Ignoring optimizer settings for unexpected owner install=%s machine=%s",
                event.install_id,
                event.machine_id,
            )
            return
        service.configure_prior_policy(
            event.install_id,
            event.machine_id,
            event.prior_mode,
            event.prior_rules,
        )
        status = runtime_optimizer.configure(
            optimizer_mode=event.optimizer_mode,
            model_artifact_path=event.model_artifact_path,
            model_artifact_sha256=event.model_artifact_sha256,
        )
        logger.info(
            "Optimizer settings accepted machine=%s configured=%s effective=%s prior_mode=%s rules=%d",
            event.machine_id,
            status.configured_mode,
            status.effective_mode,
            event.prior_mode.value,
            len(event.prior_rules),
        )
        publish_status(event.machine_id, event.bean_context_id, event.grinder_context_id)

    def on_local_reset(event: LocalResetEvent) -> None:
        if event.install_id != config.install_id or not _same_machine_id(event.machine_id, config.machine_id):
            logger.warning(
                "Ignoring local reset for unexpected owner install=%s machine=%s",
                event.install_id,
                event.machine_id,
            )
            return
        result = local_data_service.reset_all(dry_run=event.dry_run)
        logger.warning(
            "Local reset requested machine=%s dry_run=%s counts=%s",
            event.machine_id,
            event.dry_run,
            result.counts,
        )
        if not event.dry_run:
            mqtt_client.clear_recommendation(event.machine_id)
        publish_status(event.machine_id, None, None)

    mqtt_client = GaggimateMQTTClient(
        config=config,
        on_shot=runtime_coordinator.handle_shot,
        on_feedback=runtime_coordinator.handle_feedback,
        on_correction=on_correction,
        on_upload_maintenance=on_upload_maintenance,
        on_decision=on_decision,
        on_apply=on_apply,
        on_machine_state=runtime_coordinator.handle_machine_state,
        on_optimizer_settings=on_optimizer_settings,
        on_local_reset=on_local_reset,
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
        if local_dashboard_thread is not None:
            local_dashboard_thread.join(timeout=5)
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
        optimizer_status=runtime_optimizer.status().to_dict(),
        shadow_evaluation_service=shadow_evaluation_service,
        shadow_quality_service=shadow_quality_service,
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
    optimizer_status: dict[str, object] | None = None,
    shadow_evaluation_service: DreamerShadowEvaluationService | None = None,
    shadow_quality_service: DreamerShadowQualityReportService | None = None,
) -> None:
    if config.machine_id == "gaggimate:local":
        return
    if not config.bean_context_id:
        mqtt_client.clear_recommendation(config.machine_id)
        mqtt_client.publish_status(
            config.machine_id,
            build_status_payload(
                config=config,
                service=service,
                upload_maintenance=upload_maintenance,
                shot_repo=shot_repo,
                upload_queue_repo=upload_queue_repo,
                machine_id=config.machine_id,
                bean_context_id=None,
                grinder_context_id=config.grinder_context_id,
                optimizer_status=optimizer_status,
                shadow_evaluation_service=shadow_evaluation_service,
                shadow_quality_service=shadow_quality_service,
            ),
        )
        return
    current = service.get_current_recommendation(
        install_id=config.install_id,
        machine_id=config.machine_id,
        bean_context_id=config.bean_context_id,
        grinder_context_id=config.grinder_context_id,
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
                grinder_context_id=config.grinder_context_id,
                last_recommendation_id=current.recommendation_id,
                last_recommendation_at=current.updated_at,
                mode=current.mode.value,
                optimizer_status=optimizer_status,
                shadow_evaluation_service=shadow_evaluation_service,
                shadow_quality_service=shadow_quality_service,
            ),
        )
        return
    mqtt_client.clear_recommendation(config.machine_id)
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
            grinder_context_id=config.grinder_context_id,
            optimizer_status=optimizer_status,
            shadow_evaluation_service=shadow_evaluation_service,
            shadow_quality_service=shadow_quality_service,
        ),
    )


def _same_machine_id(left: str, right: str) -> bool:
    if left == right:
        return True
    if left.startswith("gaggimate:") and right.startswith("gaggimate:"):
        return left.removeprefix("gaggimate:").lower() == right.removeprefix("gaggimate:").lower()
    return False


def _profile_ids_match(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.strip().casefold() == right.strip().casefold()


def build_status_payload(
    config: Config,
    service: EspressoRLService,
    upload_maintenance: UploadQueueMaintenanceService | None,
    shot_repo: ShotRepository | None,
    upload_queue_repo: UploadQueueRepository | None,
    machine_id: str,
    bean_context_id: str | None,
    grinder_context_id: str | None = None,
    *,
    profile_id: str | None = None,
    profile_label: str | None = None,
    last_shot_id: str | None = None,
    last_shot_at: int | None = None,
    last_recommendation_id: str | None = None,
    last_recommendation_at: int | None = None,
    mode: str | None = None,
    optimizer_status: dict[str, object] | None = None,
    shadow_evaluation_service: DreamerShadowEvaluationService | None = None,
    shadow_quality_service: DreamerShadowQualityReportService | None = None,
) -> dict:
    now = config.now()
    all_recent = (
        shot_repo.list_recent(
            install_id=config.install_id,
            machine_id=machine_id,
            bean_context_id=bean_context_id,
            grinder_context_id=grinder_context_id,
            limit=1_000_000,
        )
        if shot_repo is not None
        else []
    )
    current = service.get_current_recommendation(
        install_id=config.install_id,
        machine_id=machine_id,
        bean_context_id=bean_context_id,
        grinder_context_id=grinder_context_id,
    )
    last_shot_record = (
        next((shot for shot in reversed(all_recent) if shot.shot_id == last_shot_id), None)
        if last_shot_id is not None
        else None
    )
    optimizer_profile_id = (
        profile_id
        or (last_shot_record.profile_id if last_shot_record is not None else None)
        or (current.profile_id if current is not None else None)
        or (all_recent[-1].profile_id if all_recent else None)
    )
    recent = [
        shot
        for shot in all_recent
        if optimizer_profile_id is None or _profile_ids_match(shot.profile_id, optimizer_profile_id)
    ]
    if last_shot_record is not None and not _profile_ids_match(
        last_shot_record.profile_id,
        optimizer_profile_id,
    ):
        last_shot_record = None
    if last_shot_id is None and recent:
        last_shot_record = recent[-1]
        last_shot_id = last_shot_record.shot_id
        last_shot_at = last_shot_record.timestamp
    optimizer_profile_label = profile_label or next(
        (
            shot.profile_label
            for shot in reversed(recent)
            if shot.profile_label and _profile_ids_match(shot.profile_id, optimizer_profile_id)
        ),
        None,
    )

    optimizer_shots = [
        shot
        for shot in recent
        if shot.shot_type.value == "espresso"
        and not shot.exclude_from_local_optimization
        and shot.optimization_weight > 0.0
    ]
    rated_shots = [shot for shot in optimizer_shots if shot.human_rating is not None]

    if current is not None and optimizer_profile_id is not None and not _profile_ids_match(
        current.profile_id,
        optimizer_profile_id,
    ):
        current = None
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
    rejected_summaries = upload_maintenance.list_rejected(limit=20) if upload_maintenance is not None else []
    latest_rejected = rejected_summaries[0] if rejected_summaries else None
    rejected_record_ids = {
        item.local_record_id
        for item in rejected_summaries
        if item.local_record_type == "shot" and item.local_record_id
    }
    model_artifact_status = verify_model_artifact(
        config.optimizer_model_artifact_path,
        config.optimizer_model_artifact_sha256,
        max_bytes=config.optimizer_model_artifact_max_bytes,
    )
    model_manifest_status = verify_model_manifest_file(
        config.optimizer_model_manifest_path,
        expected_model_sha256=config.optimizer_model_artifact_sha256,
    )
    if model_manifest_status.model_artifact_sha256 and not model_artifact_status.verified:
        model_artifact_status = verify_model_artifact(
            config.optimizer_model_artifact_path,
            model_manifest_status.model_artifact_sha256,
            max_bytes=config.optimizer_model_artifact_max_bytes,
        )
    # Status construction must not promote files that have only passed the legacy
    # outer hash/manifest checks. Runtime availability requires the typed loader
    # result supplied through optimizer_status.
    config_dreamer_v3_available = False
    config_dreamer_v3_shadow_available = False
    config_optimizer_mode = (
        config.optimizer_mode
        if config.optimizer_mode
        in {DEFAULT_OPTIMIZER_MODE, OPTIMIZER_MODE_DREAMER_V3_ACTIVE, OPTIMIZER_MODE_DREAMER_V3_SHADOW}
        else DEFAULT_OPTIMIZER_MODE
    )
    optimizer_status = optimizer_status or {
        "configured_mode": config_optimizer_mode,
        "effective_mode": DEFAULT_OPTIMIZER_MODE,
        "model_artifact_path": model_artifact_status.path,
        "model_artifact_sha256": model_artifact_status.expected_sha256,
        "model_artifact_actual_sha256": model_artifact_status.actual_sha256,
        "model_artifact_size_bytes": model_artifact_status.size_bytes,
        "model_artifact_verified": model_artifact_status.verified,
        "model_artifact_unavailable_reason": model_artifact_status.unavailable_reason,
        "model_manifest_path": model_manifest_status.path,
        "model_manifest_sha256": model_manifest_status.actual_sha256,
        "model_manifest_size_bytes": model_manifest_status.size_bytes,
        "model_manifest_verified": model_manifest_status.verified,
        "model_manifest_unavailable_reason": model_manifest_status.unavailable_reason,
        "model_manifest_model_family": model_manifest_status.model_family,
        "model_manifest_artifact_format": model_manifest_status.model_artifact_format,
        "model_manifest_dataset_sha256": model_manifest_status.dataset_sha256,
        "model_manifest_dataset_manifest_sha256": model_manifest_status.dataset_manifest_sha256,
        "model_manifest_trainer_git_sha": model_manifest_status.trainer_git_sha,
        "model_manifest_training_config_sha256": model_manifest_status.training_config_sha256,
        "model_manifest_state_schema_version": model_manifest_status.state_schema_version,
        "model_manifest_action_schema_version": model_manifest_status.action_schema_version,
        "model_manifest_reward_schema_version": model_manifest_status.reward_schema_version,
        "checkpoint_verified": False,
        "checkpoint_inference_ready": False,
        "checkpoint_tensor_count": 0,
        "checkpoint_component_names": [],
        "checkpoint_architecture_sha256": None,
        "checkpoint_inference_probe_sha256": None,
        "checkpoint_heldout_inference_sha256": None,
        "checkpoint_unavailable_reason": "DreamerV3 checkpoint has not passed strict tensor verification.",
        "checkpoint_inference_parity_verified": False,
        "checkpoint_inference_parity_reason": "DreamerV3 checkpoint has not been materialized.",
        "dreamer_v3_available": config_dreamer_v3_available,
        "dreamer_v3_shadow_available": config_dreamer_v3_shadow_available,
        "dreamer_v3_active_available": config_dreamer_v3_available,
        "available_modes": [DEFAULT_OPTIMIZER_MODE]
        + ([OPTIMIZER_MODE_DREAMER_V3_SHADOW] if config_dreamer_v3_shadow_available else [])
        + ([OPTIMIZER_MODE_DREAMER_V3_ACTIVE] if config_dreamer_v3_available else []),
        "unavailable_modes": {}
        if config_dreamer_v3_available
        else {
            OPTIMIZER_MODE_DREAMER_V3_SHADOW: model_manifest_status.unavailable_reason
            or model_artifact_status.unavailable_reason
            or "DreamerV3 model artifact is not verified.",
            OPTIMIZER_MODE_DREAMER_V3_ACTIVE: model_manifest_status.unavailable_reason
            or model_artifact_status.unavailable_reason
            or "DreamerV3 active model artifact is not verified.",
        },
        "fallback_reason": None
        if config_optimizer_mode == DEFAULT_OPTIMIZER_MODE
        else "Bayesian Optimization is serving recommendations.",
        "dreamer_v3_active_recommendation_count": 0,
        "dreamer_v3_bo_fallback_count": 0,
        "dreamer_v3_bo_fallback_reason_counts": {},
        "dreamer_v3_last_runtime_event": None,
        "dreamer_v3_last_bo_fallback_reason": None,
    }
    shadow_summary = {
        "inference_contract_id": None,
        "record_count": 0,
        "pending_count": 0,
        "observed_count": 0,
        "safe_proposal_count": 0,
        "unsafe_proposal_count": 0,
        "dreamer_matched_count": 0,
        "bo_matched_count": 0,
        "mean_dreamer_followed_reward_delta": None,
        "mean_bo_followed_reward_delta": None,
        "shadow_only": True,
    }
    if shadow_evaluation_service is not None and bean_context_id and grinder_context_id:
        try:
            shadow_summary = shadow_evaluation_service.context_summary(
                install_id=config.install_id,
                machine_id=machine_id,
                bean_context_id=bean_context_id,
                grinder_context_id=grinder_context_id,
            )
        except Exception as exc:
            logger.warning("DreamerV3 shadow evaluation status unavailable: %s", exc)
    shadow_quality_summary = {
        "report_id": None,
        "generated_at": None,
        "checkpoint_artifact_sha256": None,
        "checkpoint_inference_probe_sha256": None,
        "inference_contract_id": None,
        "overall_status": "insufficient_data",
        "evaluated_record_count": 0,
        "stale_checkpoint_record_count": 0,
        "observed_count": 0,
        "safety_rate": None,
        "outcome_coverage": None,
        "dreamer_reward_delta_advantage": None,
        "gates": [],
        "observational_only": True,
        "shadow_only": True,
        "recommendation_enabled": False,
        "machine_control_enabled": False,
    }
    if shadow_quality_service is not None and bean_context_id and grinder_context_id:
        try:
            quality_report = shadow_quality_service.build_context_report(
                install_id=config.install_id,
                machine_id=machine_id,
                bean_context_id=bean_context_id,
                grinder_context_id=grinder_context_id,
            )
            shadow_quality_summary = quality_report.status_summary()
        except Exception as exc:
            logger.warning("DreamerV3 shadow quality status unavailable: %s", exc)

    community_upload_enabled = config.should_enqueue_community_uploads() and service.community_upload_enabled_for(
        config.install_id,
        machine_id,
    )
    runtime_health = _runtime_health_payload(
        config=config,
        bean_context_id=bean_context_id,
        grinder_context_id=grinder_context_id,
        optimizer_shot_count=len(optimizer_shots),
        rated_shot_count=len(rated_shots),
        last_recommendation_id=last_recommendation_id,
        upload_queue_status_counts=upload_queue_status_counts,
        upload_queue_available=upload_queue_repo is not None,
        community_upload_requested=community_upload_enabled,
    )
    diagnostic_steps = _auto_tuning_diagnostic_steps(
        bean_context_id=bean_context_id,
        grinder_context_id=grinder_context_id,
        last_shot_id=last_shot_id,
        last_shot_record=last_shot_record,
        rated_shot_count=len(rated_shots),
        last_recommendation_id=last_recommendation_id,
        mode=mode,
        community_upload_enabled=community_upload_enabled,
        upload_queue_status_counts=upload_queue_status_counts,
    )

    return {
        "addon_online": True,
        "install_id": config.install_id,
        "bean_context_id": bean_context_id,
        "grinder_context_id": grinder_context_id,
        "optimizer_profile_id": optimizer_profile_id,
        "optimizer_profile_label": optimizer_profile_label,
        "prior_mode": service.prior_mode_for(config.install_id, machine_id).value,
        "selected_prior_rule_count": len(service.prior_rules_for(config.install_id, machine_id)),
        **runtime_health,
        "auto_tuning_diagnostic_steps": diagnostic_steps,
        "timestamp": now,
        "last_shot_id": last_shot_id,
        "last_shot_at": last_shot_at,
        "last_shot_type": last_shot_record.shot_type.value if last_shot_record else None,
        "last_shot_time_s": last_shot_record.shot_time_s if last_shot_record else None,
        "last_shot_beverage_out_g": last_shot_record.beverage_out_g if last_shot_record else None,
        "last_shot_target_yield_g": last_shot_record.target_yield_g if last_shot_record else None,
        "last_shot_human_rating": last_shot_record.human_rating if last_shot_record else None,
        "recent_shots": recent_shot_summaries(recent, rejected_record_ids),
        "last_recommendation_id": last_recommendation_id,
        "last_recommendation_at": last_recommendation_at,
        "recommendation_apply_status": apply_status,
        "mode": mode,
        "optimizer_configured_mode": optimizer_status.get("configured_mode"),
        "optimizer_effective_mode": optimizer_status.get("effective_mode"),
        "optimizer_model_artifact_path": optimizer_status.get("model_artifact_path"),
        "optimizer_model_artifact_sha256": optimizer_status.get("model_artifact_sha256"),
        "optimizer_model_artifact_actual_sha256": optimizer_status.get("model_artifact_actual_sha256"),
        "optimizer_model_artifact_size_bytes": optimizer_status.get("model_artifact_size_bytes"),
        "optimizer_model_artifact_verified": bool(optimizer_status.get("model_artifact_verified")),
        "optimizer_model_artifact_unavailable_reason": optimizer_status.get("model_artifact_unavailable_reason"),
        "optimizer_model_manifest_path": optimizer_status.get("model_manifest_path"),
        "optimizer_model_manifest_sha256": optimizer_status.get("model_manifest_sha256"),
        "optimizer_model_manifest_size_bytes": optimizer_status.get("model_manifest_size_bytes"),
        "optimizer_model_manifest_verified": bool(optimizer_status.get("model_manifest_verified")),
        "optimizer_model_manifest_unavailable_reason": optimizer_status.get("model_manifest_unavailable_reason"),
        "optimizer_model_manifest_model_family": optimizer_status.get("model_manifest_model_family"),
        "optimizer_model_manifest_artifact_format": optimizer_status.get("model_manifest_artifact_format"),
        "optimizer_model_manifest_dataset_sha256": optimizer_status.get("model_manifest_dataset_sha256"),
        "optimizer_model_manifest_dataset_manifest_sha256": optimizer_status.get("model_manifest_dataset_manifest_sha256"),
        "optimizer_model_manifest_trainer_git_sha": optimizer_status.get("model_manifest_trainer_git_sha"),
        "optimizer_model_manifest_training_config_sha256": optimizer_status.get("model_manifest_training_config_sha256"),
        "optimizer_model_manifest_state_schema_version": optimizer_status.get("model_manifest_state_schema_version"),
        "optimizer_model_manifest_action_schema_version": optimizer_status.get("model_manifest_action_schema_version"),
        "optimizer_model_manifest_reward_schema_version": optimizer_status.get("model_manifest_reward_schema_version"),
        "optimizer_checkpoint_verified": bool(optimizer_status.get("checkpoint_verified")),
        "optimizer_checkpoint_inference_ready": bool(optimizer_status.get("checkpoint_inference_ready")),
        "optimizer_checkpoint_tensor_count": int(optimizer_status.get("checkpoint_tensor_count") or 0),
        "optimizer_checkpoint_component_names": optimizer_status.get("checkpoint_component_names") or [],
        "optimizer_checkpoint_architecture_sha256": optimizer_status.get("checkpoint_architecture_sha256"),
        "optimizer_checkpoint_inference_probe_sha256": optimizer_status.get("checkpoint_inference_probe_sha256"),
        "optimizer_checkpoint_heldout_inference_sha256": optimizer_status.get(
            "checkpoint_heldout_inference_sha256"
        ),
        "optimizer_checkpoint_unavailable_reason": optimizer_status.get("checkpoint_unavailable_reason"),
        "optimizer_checkpoint_inference_parity_verified": bool(
            optimizer_status.get("checkpoint_inference_parity_verified")
        ),
        "optimizer_checkpoint_inference_parity_reason": optimizer_status.get(
            "checkpoint_inference_parity_reason"
        ),
        "optimizer_dreamer_v3_available": bool(optimizer_status.get("dreamer_v3_available")),
        "optimizer_dreamer_v3_shadow_available": bool(
            optimizer_status.get("dreamer_v3_shadow_available")
        ),
        "optimizer_dreamer_v3_active_available": bool(
            optimizer_status.get("dreamer_v3_active_available")
        ),
        "optimizer_available_modes": optimizer_status.get("available_modes") or [DEFAULT_OPTIMIZER_MODE],
        "optimizer_unavailable_modes": optimizer_status.get("unavailable_modes") or {},
        "optimizer_fallback_reason": optimizer_status.get("fallback_reason"),
        "optimizer_dreamer_v3_active_recommendation_count": int(
            optimizer_status.get("dreamer_v3_active_recommendation_count") or 0
        ),
        "optimizer_dreamer_v3_bo_fallback_count": int(
            optimizer_status.get("dreamer_v3_bo_fallback_count") or 0
        ),
        "optimizer_dreamer_v3_bo_fallback_reason_counts": (
            optimizer_status.get("dreamer_v3_bo_fallback_reason_counts") or {}
        ),
        "optimizer_dreamer_v3_last_runtime_event": optimizer_status.get("dreamer_v3_last_runtime_event"),
        "optimizer_dreamer_v3_last_bo_fallback_reason": optimizer_status.get(
            "dreamer_v3_last_bo_fallback_reason"
        ),
        "dreamer_shadow_evaluation": shadow_summary,
        "dreamer_shadow_quality_report": shadow_quality_summary,
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
        "community_upload_enabled": community_upload_enabled,
        "grinder_catalog_search_url": grinder_catalog_search_url(config),
        "prior_rule_catalog_search_url": prior_rule_catalog_search_url(config),
    }


def _auto_tuning_diagnostic_steps(
    *,
    bean_context_id: str | None,
    grinder_context_id: str | None,
    last_shot_id: str | None,
    last_shot_record,
    rated_shot_count: int,
    last_recommendation_id: str | None,
    mode: str | None,
    community_upload_enabled: bool,
    upload_queue_status_counts: dict[str, int],
) -> list[dict[str, str]]:
    def step(key: str, label: str, state: str, detail: str) -> dict[str, str]:
        return {"key": key, "label": label, "state": state, "detail": detail}

    steps = [
        step(
            "context",
            "Context",
            "ok" if bean_context_id and grinder_context_id else "waiting",
            "Bean and grinder selected." if bean_context_id and grinder_context_id else "Select a bean and grinder.",
        ),
        step(
            "shot_observed",
            "Shot observed",
            "ok" if last_shot_id else "waiting",
            "A shot has been seen for this context." if last_shot_id else "Waiting for a shot.",
        ),
    ]

    if last_shot_record is not None:
        shot_usable = (
            last_shot_record.shot_type.value == "espresso"
            and not last_shot_record.exclude_from_local_optimization
            and last_shot_record.optimization_weight > 0.0
        )
        steps.append(step("shot_stored", "Local storage", "ok", "Last shot is in local history."))
        steps.append(
            step(
                "shot_usable",
                "Optimizer input",
                "ok" if shot_usable else "attention",
                "Last shot is usable for optimization."
                if shot_usable
                else "Last shot was excluded or has zero optimization weight.",
            )
        )
        steps.append(
            step(
                "rating",
                "Rating",
                "ok" if last_shot_record.human_rating is not None else "waiting",
                "Rating is recorded." if last_shot_record.human_rating is not None else "Waiting for shot rating.",
            )
        )
    else:
        steps.append(
            step(
                "shot_stored",
                "Local storage",
                "attention" if last_shot_id else "waiting",
                "Observed shot was not found in local history."
                if last_shot_id
                else "No shot is available to store yet.",
            )
        )
        steps.append(step("shot_usable", "Optimizer input", "waiting", "Waiting for a stored shot."))
        steps.append(step("rating", "Rating", "waiting", "Waiting for a stored shot."))

    if last_recommendation_id:
        detail = "Recommendation is available."
        if mode == "zero_observe":
            detail = "Baseline observation is active."
        steps.append(step("recommendation", "Recommendation", "ok", detail))
    else:
        detail = "Waiting for enough rated data."
        if rated_shot_count > 0:
            detail = "Waiting for the next recommendation cycle."
        steps.append(step("recommendation", "Recommendation", "waiting", detail))

    pending_uploads = upload_queue_status_counts.get(UploadQueueStatus.PENDING.value, 0)
    failed_uploads = upload_queue_status_counts.get(UploadQueueStatus.FAILED.value, 0)
    rejected_uploads = upload_queue_status_counts.get(UploadQueueStatus.REJECTED.value, 0)
    if not community_upload_enabled:
        upload_state = "off"
        upload_detail = "Community upload is disabled."
    elif rejected_uploads:
        upload_state = "attention"
        upload_detail = f"{rejected_uploads} upload record(s) rejected."
    elif failed_uploads:
        upload_state = "attention"
        upload_detail = f"{failed_uploads} upload record(s) waiting to retry."
    elif pending_uploads:
        upload_state = "waiting"
        upload_detail = f"{pending_uploads} upload record(s) queued."
    else:
        upload_state = "ok"
        upload_detail = "Upload queue is clear."
    steps.append(step("community_upload", "Community upload", upload_state, upload_detail))
    steps.append(step("status_published", "Status published", "ok", "This diagnostic status was published."))
    return steps


def _runtime_health_payload(
    *,
    config: Config,
    bean_context_id: str | None,
    grinder_context_id: str | None,
    optimizer_shot_count: int,
    rated_shot_count: int,
    last_recommendation_id: str | None,
    upload_queue_status_counts: dict[str, int],
    upload_queue_available: bool,
    community_upload_requested: bool,
) -> dict[str, object]:
    upload_configured = bool(config.supabase_ingest_url and config.upload_secret)
    pending_uploads = upload_queue_status_counts.get(UploadQueueStatus.PENDING.value, 0)
    failed_uploads = upload_queue_status_counts.get(UploadQueueStatus.FAILED.value, 0)
    rejected_uploads = upload_queue_status_counts.get(UploadQueueStatus.REJECTED.value, 0)

    waiting_reasons: list[str] = []
    warnings: list[str] = []
    if not bean_context_id:
        waiting_reasons.append("Select a bean context.")
    if not grinder_context_id:
        waiting_reasons.append("Select a grinder context.")
    if community_upload_requested and not upload_queue_available:
        warnings.append("Community upload is enabled but the local upload queue is unavailable.")
    if community_upload_requested and not upload_configured:
        warnings.append("Community upload is enabled but upload credentials are incomplete.")
    if rejected_uploads:
        warnings.append(f"{rejected_uploads} community upload record(s) were rejected by the backend.")
    if failed_uploads:
        warnings.append(f"{failed_uploads} community upload record(s) are waiting to retry.")

    if warnings:
        status = "attention"
        summary = warnings[0]
    elif waiting_reasons:
        status = "waiting"
        summary = "Setup incomplete"
    elif optimizer_shot_count <= 0:
        status = "waiting"
        summary = "Ready - waiting for the first shot"
    elif rated_shot_count <= 0:
        status = "waiting"
        summary = "Ready - waiting for the first rating"
    elif not last_recommendation_id:
        status = "waiting"
        summary = "Ready - building the next recommendation"
    else:
        status = "ok"
        summary = "Connected"

    return {
        "runtime_health_status": status,
        "runtime_health_summary": summary,
        "runtime_health_warnings": warnings,
        "runtime_health_waiting_reasons": waiting_reasons,
        "runtime_health_storage_backend": config.storage_backend,
        "runtime_health_storage_available": True,
        "runtime_health_upload_configured": upload_configured,
        "runtime_health_community_upload_requested": community_upload_requested,
        "runtime_health_pending_upload_count": pending_uploads,
        "runtime_health_failed_upload_count": failed_uploads,
        "runtime_health_rejected_upload_count": rejected_uploads,
    }


def grinder_catalog_search_url(config: Config) -> str:
    for source_url in (config.supabase_ingest_url, config.supabase_registration_url):
        derived = _derive_supabase_function_url(source_url, "espresso-rl-grinder-search")
        if derived:
            return derived
    return ""


def prior_rule_catalog_search_url(config: Config) -> str:
    for source_url in (config.supabase_ingest_url, config.supabase_registration_url):
        derived = _derive_supabase_function_url(source_url, "espresso-rl-prior-rule-search")
        if derived:
            return derived
    return ""


def _derive_supabase_function_url(source_url: str, function_name: str) -> str:
    text = str(source_url or "").strip().rstrip("/")
    if not text:
        return ""
    parts = urlsplit(text)
    if not parts.scheme or not parts.netloc:
        return ""

    path_parts = [part for part in parts.path.split("/") if part]
    if len(path_parts) >= 3 and path_parts[-3] == "functions" and path_parts[-2] == "v1":
        path_parts[-1] = function_name
    else:
        path_parts.extend(["functions", "v1", function_name])

    return urlunsplit((parts.scheme, parts.netloc, "/" + "/".join(path_parts), "", ""))


def recent_shot_summaries(shots: list, rejected_record_ids: set[str], limit: int = 10) -> list[dict]:
    recent = list(reversed(shots[-limit:]))
    return [
        {
            "shot_id": shot.shot_id,
            "timestamp": shot.timestamp,
            "bean_context_id": shot.bean_context_id,
            "grinder_context_id": shot.grinder_context_id,
            "shot_type": shot.shot_type.value,
            "shot_time_s": shot.shot_time_s,
            "beverage_out_g": shot.beverage_out_g,
            "target_yield_g": shot.target_yield_g,
            "human_rating": shot.human_rating,
            "exclude_from_local_optimization": shot.exclude_from_local_optimization,
            "optimization_weight": shot.optimization_weight,
            "profile_label": shot.profile_label,
            "profile_type": shot.profile_type,
            "final_phase_index": shot.final_phase_index,
            "final_phase_name": shot.final_phase_name,
            "final_phase_type": shot.final_phase_type,
            "final_phase_elapsed_s": shot.final_phase_elapsed_s,
            "final_pump_target": shot.final_pump_target,
            "shot_end_state": shot.shot_end_state,
            "profile_flow_valid": shot.profile_flow_valid,
            "profile_flow_masked": shot.profile_flow_masked,
            "rejected_upload": shot.shot_id in rejected_record_ids,
        }
        for shot in recent
    ]


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
        "relative_grind_steps_from_reference": best.relative_grind_steps_from_reference,
        "relative_grind_um_from_reference": best.relative_grind_um_from_reference,
        "grinder_calibration_mode": best.grinder_calibration_mode.value,
        "step_direction": best.grinder_step_direction.value,
        "reference_label": best.grinder_reference_label,
        "current_absolute_step": best.current_absolute_step,
        "absolute_reference_step": best.absolute_reference_step,
        "restorable_absolute_step": (
            best.absolute_reference_step + best.relative_grind_steps_from_reference
            if best.absolute_reference_step is not None and best.relative_grind_steps_from_reference is not None
            else None
        ),
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


def load_configured_dreamer_checkpoint(
    config: Config,
) -> tuple[VerifiedDreamerCheckpoint | None, str | None]:
    if not config.optimizer_model_artifact_path:
        return None, "DreamerV3 checkpoint artifact path is not configured."
    if not config.optimizer_model_manifest_path:
        return None, "DreamerV3 checkpoint manifest path is not configured."
    if not config.optimizer_model_artifact_sha256:
        return None, "DreamerV3 checkpoint artifact SHA-256 is not configured."
    try:
        checkpoint = load_verified_dreamer_checkpoint(
            LocalModelArtifactStore(),
            artifact_reference=config.optimizer_model_artifact_path,
            manifest_reference=config.optimizer_model_manifest_path,
            expected_artifact_sha256=config.optimizer_model_artifact_sha256,
            max_checkpoint_bytes=config.optimizer_model_artifact_max_bytes,
        )
    except CheckpointLoadError as exc:
        logger.warning("DreamerV3 checkpoint verification failed: %s", exc)
        return None, str(exc)
    return checkpoint, None


def try_record_shadow_evaluation(
    shadow_service: DreamerShadowEvaluationService | None,
    *,
    shot,
    recommendation,
    shot_repo: ShotRepository | None = None,
):
    if shadow_service is None:
        return None
    try:
        transition = local_training_transition_from_shot(shot)
        if transition is None:
            return None
        context_transitions = local_context_transitions_for_shadow_replay(
            shot,
            shot_repo=shot_repo,
        )
        result = shadow_service.evaluate_transition(
            transition,
            bo_recommendation=recommendation,
            context_transitions=context_transitions,
        )
    except DreamerShadowEvaluationError as exc:
        logger.warning("DreamerV3 shadow evaluation skipped for shot %s: %s", shot.shot_id, exc)
        return None
    except Exception:
        logger.exception("DreamerV3 shadow evaluation failed for shot %s; BO remains active", shot.shot_id)
        return None
    logger.info(
        "DreamerV3 shadow evaluation stored shot=%s evaluation=%s safe=%s parity_only=true",
        shot.shot_id,
        result.evaluation.evaluation_id,
        result.evaluation.dreamer_proposal.safety_valid,
    )
    return result


def local_context_transitions_for_shadow_replay(
    shot,
    *,
    shot_repo: ShotRepository | None,
    limit: int = DREAMER_CONTEXT_WINDOW_SIZE,
) -> list[dict]:
    if shot_repo is None or not shot.bean_context_id or not shot.grinder_context_id:
        return []
    recent = shot_repo.list_recent(
        shot.install_id,
        shot.machine_id,
        bean_context_id=shot.bean_context_id,
        grinder_context_id=shot.grinder_context_id,
        limit=limit + 1,
    )
    context_rows = []
    for candidate in sorted(recent, key=lambda item: (item.timestamp, item.shot_id)):
        if candidate.shot_id == shot.shot_id or candidate.timestamp >= shot.timestamp:
            continue
        transition = local_training_transition_from_shot(candidate)
        if transition is not None:
            context_rows.append(transition)
    return context_rows[-limit:]


def try_build_shadow_quality_report(
    quality_service: DreamerShadowQualityReportService | None,
    evaluation,
):
    if quality_service is None:
        return None
    try:
        report = quality_service.build_context_report(
            install_id=evaluation.install_id,
            machine_id=evaluation.machine_id,
            bean_context_id=evaluation.bean_context_id,
            grinder_context_id=evaluation.grinder_context_id,
        )
    except DreamerShadowQualityError as exc:
        logger.warning("DreamerV3 shadow quality report skipped: %s", exc)
        return None
    except Exception:
        logger.exception("DreamerV3 shadow quality report failed; BO remains active")
        return None
    logger.info(
        "DreamerV3 shadow quality report stored report=%s status=%s shadow_only=true",
        report.report_id,
        report.overall_status.value,
    )
    return report


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


def maybe_start_local_dashboard(
    config: Config,
    stop_event: threading.Event,
    *,
    local_data_service: LocalDataService,
    upload_maintenance: UploadQueueMaintenanceService,
) -> threading.Thread | None:
    if config.deployment_role != "public":
        return None
    if not config.local_dashboard_enabled:
        logger.info("Local dashboard disabled.")
        return None
    if len(config.local_dashboard_token) < 32:
        logger.warning(
            "Local dashboard enabled but ESPRESSORL_LOCAL_DASHBOARD_TOKEN/local_dashboard_token is missing or too short."
        )
        return None

    from espresso_rl.adapters.local_dashboard import start_local_dashboard

    logger.info(
        "Starting local dashboard on %s:%d",
        config.local_dashboard_host,
        config.local_dashboard_port,
    )
    return start_local_dashboard(
        local_data_service,
        upload_maintenance,
        local_token=config.local_dashboard_token,
        host=config.local_dashboard_host,
        port=config.local_dashboard_port,
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
        training_exporter=TrainingDatasetExportService(
            warehouse=warehouse,
            writer=LocalTextArtifactWriter(config.training_export_dir),
            source_git_sha=config.build_git_sha,
            max_rows=config.training_export_max_rows,
            clock=config.now,
        ),
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
    providers = [LocalHistoryPriorProvider()]
    if config.storage_backend == "postgres":
        providers.append(
            CommunityPriorProvider(
                PostgresCommunityWarehouse(PostgresStore(config.postgres_dsn)),
            )
        )
    return CompositePriorProvider(providers)


def open_repositories(
    config: Config,
) -> tuple[
    ShotRepository,
    RecommendationRepository,
    UploadQueueRepository,
    LocalDataRepository,
    ShadowEvaluationRepository,
    ShadowQualityReportRepository,
]:
    if config.storage_backend == "postgres":
        logger.info("Using Postgres storage backend")
        store = PostgresStore(config.postgres_dsn)
        return (
            PostgresShotRepository(store),
            PostgresRecommendationRepository(store),
            PostgresUploadQueueRepository(store),
            PostgresLocalDataRepository(store),
            PostgresShadowEvaluationRepository(store),
            PostgresShadowQualityReportRepository(store),
        )

    logger.warning("Using SQLite storage backend; Postgres is the intended container/admin runtime backend.")
    store = SQLiteStore(config.data_dir / "espresso_rl.db")
    return (
        SQLiteShotRepository(store),
        SQLiteRecommendationRepository(store),
        SQLiteUploadQueueRepository(store),
        SQLiteLocalDataRepository(store),
        SQLiteShadowEvaluationRepository(store),
        SQLiteShadowQualityReportRepository(store),
    )


if __name__ == "__main__":
    main()

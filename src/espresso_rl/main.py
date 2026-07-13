from __future__ import annotations

import logging
import signal
import sys
import threading
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

from espresso_rl.adapters.gaggimate_mqtt import GaggimateMQTTClient
from espresso_rl.adapters.postgres_repositories import (
    PostgresCommunityWarehouse,
    PostgresLocalDataRepository,
    PostgresPreferentialOptimizationRepository,
    PostgresRecommendationRepository,
    PostgresShotRepository,
    PostgresStore,
    PostgresUploadQueueRepository,
)
from espresso_rl.adapters.sqlite_repositories import (
    SQLiteLocalDataRepository,
    SQLitePreferentialOptimizationRepository,
    SQLiteRecommendationRepository,
    SQLiteShotRepository,
    SQLiteStore,
    SQLiteUploadQueueRepository,
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
from espresso_rl.adapters.supabase_upload import (
    SignedSupabaseUploadClient,
    SignedUploadConfig,
    UploadQueueWorker,
)
from espresso_rl.application.admin_pipeline import AdminPipelineService
from espresso_rl.application.community_credentials import CommunityCredentialService
from espresso_rl.application.community_mirror import CommunityMirrorService
from espresso_rl.application.community_validation import CommunityValidationService
from espresso_rl.application.cpbo_runtime import CPBORuntimeBridge, strict_context_from_shot
from espresso_rl.application.local_data import LocalDataService
from espresso_rl.application.offline_dataset_export import OfflineDatasetExportService
from espresso_rl.application.preference_optimization import (
    ConsecutivePreferenceOptimizationService,
)
from espresso_rl.application.runtime_coordinator import AutoTuningRuntimeCoordinator
from espresso_rl.application.services import EspressoRLService
from espresso_rl.application.upload_maintenance import UploadQueueMaintenanceService
from espresso_rl.config import Config
from espresso_rl.domain.community import CommunityUploadCredentials
from espresso_rl.domain.cpbo import RecipeDomain, RecipeParameter, RecipeSpace
from espresso_rl.domain.events import (
    LocalResetEvent,
    MachineStateEvent,
    OptimizerSettingsEvent,
    PreferenceFeedbackEvent,
    RecommendationApplyEvent,
    RecommendationDecisionEvent,
    ShotCorrectionEvent,
    UploadQueueMaintenanceEvent,
)
from espresso_rl.domain.models import (
    GrinderAdjustmentMode,
    Recipe,
    Recommendation,
    UploadQueueStatus,
)
from espresso_rl.domain.optimization import DEFAULT_OPTIMIZER_MODE, OPTIMIZER_MODE_CPBO
from espresso_rl.domain.taste_goal import TasteGoal
from espresso_rl.optimizers.cpbo import ConsecutivePreferentialBayesianOptimizer
from espresso_rl.optimizers.cpbo_trace import TRACE_FEATURE_NAMES, extract_trace_features
from espresso_rl.ports.community import CommunityCredentialRegistrar, CommunityCredentialStore
from espresso_rl.ports.preference_optimization import PreferentialOptimizationRepository
from espresso_rl.ports.repositories import (
    LocalDataRepository,
    RecommendationRepository,
    ShotRepository,
    UploadQueueRepository,
)


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
    run_public(config)


def run_public(config: Config) -> None:
    logger.info("EspressoRL starting [optimizer_mode=%s]", config.optimizer_mode)
    maybe_resolve_community_upload_credentials(config)
    (
        shot_repo,
        recommendation_repo,
        upload_queue_repo,
        local_data_repo,
        preference_optimization_repo,
    ) = open_repositories(config)
    cpbo_service = ConsecutivePreferenceOptimizationService(
        repository=preference_optimization_repo,
        optimizer=ConsecutivePreferentialBayesianOptimizer(config.cpbo),
        recipe_space_factory=lambda baseline, recipe_domain: build_cpbo_recipe_space(
            baseline,
            config=config,
            recipe_domain=recipe_domain,
        ),
        random_seed=config.cpbo.random_seed,
        configuration_version=config.cpbo.effective_configuration_version,
        recipe_domain=config.cpbo.recipe_domain,
        initial_trust_region_length=config.cpbo.trust_region.initial_length,
        trace_feature_extractor=lambda sequence: (
            TRACE_FEATURE_NAMES,
            extract_trace_features(sequence, config.cpbo.trace).values,
        ),
        clock=config.now,
    )
    service = EspressoRLService(
        shots=shot_repo,
        recommendations=recommendation_repo,
        upload_queue=upload_queue_for_service(config, upload_queue_repo),
        clock=config.now,
        community_upload_enabled_default=False,
    )
    cpbo_runtime = CPBORuntimeBridge(
        optimizer=cpbo_service,
        shots=shot_repo,
        recommendation_sink=service.persist_generated_recommendation,
        comparison_sink=service.enqueue_comparison_upload,
        context_factory=strict_context_from_shot,
        comparison_mode=config.cpbo.comparison_mode,
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

    status_context_lock = threading.Lock()
    status_context: dict[str, object | None] = {
        "machine_id": config.machine_id,
        "bean_context_id": config.bean_context_id,
        "grinder_context_id": config.grinder_context_id,
        "taste_goal": TasteGoal.balanced(),
    }
    mqtt_client: GaggimateMQTTClient

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
        taste_goal: TasteGoal | None = None,
    ) -> None:
        with status_context_lock:
            status_context["machine_id"] = machine_id
            status_context["bean_context_id"] = bean_context_id
            status_context["grinder_context_id"] = grinder_context_id
            if taste_goal is not None:
                status_context["taste_goal"] = taste_goal
            active_taste_goal = status_context["taste_goal"]
        if not isinstance(active_taste_goal, TasteGoal):
            active_taste_goal = TasteGoal.balanced()
        mqtt_client.publish_status(
            machine_id,
            build_status_payload(
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
                taste_goal=active_taste_goal,
            ),
        )

    def publish_upload_queue_status() -> None:
        with status_context_lock:
            machine_id = status_context["machine_id"] or config.machine_id
            bean_context_id = status_context["bean_context_id"]
            grinder_context_id = status_context["grinder_context_id"]
        publish_status(machine_id, bean_context_id, grinder_context_id)

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

    def cpbo_recommendation_after_shot(shot) -> Recommendation | None:
        outcome = cpbo_runtime.handle_shot(shot)
        if outcome.skipped_reason == "shot_already_processed":
            logger.info("CPBO ignored idempotent replay of shot %s", shot.shot_id)
        elif outcome.skipped_reason is not None:
            logger.warning("CPBO skipped shot %s reason=%s", shot.shot_id, outcome.skipped_reason)
        elif outcome.awaiting_preference:
            logger.info(
                "CPBO shot %s stored in run %s; awaiting three-outcome preference",
                shot.shot_id,
                outcome.optimization_run_id,
            )
        return outcome.recommendation

    runtime_coordinator = AutoTuningRuntimeCoordinator(
        service=service,
        publisher=RuntimePublisher(),
        post_shot_recommendation=cpbo_recommendation_after_shot,
    )

    def on_preference(event: PreferenceFeedbackEvent) -> None:
        recommendation = cpbo_runtime.handle_preference(event)
        logger.info(
            "CPBO preference stored run=%s new=%s anchor=%s label=%s next=%s",
            event.optimization_run_id,
            event.new_shot_id,
            event.anchor_shot_id,
            event.label.value,
            recommendation.recommendation_id,
        )
        mqtt_client.publish_recommendation(recommendation)
        publish_status(
            recommendation.machine_id,
            recommendation.bean_context_id,
            recommendation.grinder_context_id,
            profile_id=recommendation.profile_id,
            last_shot_id=event.new_shot_id,
            last_recommendation_id=recommendation.recommendation_id,
            last_recommendation_at=recommendation.updated_at,
            mode=recommendation.mode.value,
            taste_goal=recommendation.taste_goal,
        )

    def on_correction(event: ShotCorrectionEvent) -> None:
        shot = service.record_shot_correction(event)
        logger.info(
            "Correction stored shot=%s excluded=%s followed=%s attribution=%.2f",
            shot.shot_id,
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
            taste_goal=shot.taste_goal,
        )

    def on_upload_maintenance(event: UploadQueueMaintenanceEvent) -> None:
        if event.action == "purge_rejected":
            result = upload_maintenance.purge_rejected(
                limit=event.limit,
                local_record_id=event.local_record_id,
            )
            logger.info("Upload purge result=%s", result)
        else:
            result = upload_maintenance.requeue_valid_rejected(limit=event.limit)
            logger.info("Upload requeue result=%s", result)
        publish_status(event.machine_id, event.bean_context_id, event.grinder_context_id)

    def on_decision(event: RecommendationDecisionEvent) -> None:
        recommendation = service.record_recommendation_decision(event)
        publish_status(
            recommendation.machine_id,
            recommendation.bean_context_id,
            recommendation.grinder_context_id,
            profile_id=recommendation.profile_id,
            last_recommendation_id=recommendation.recommendation_id,
            last_recommendation_at=recommendation.updated_at,
            mode=recommendation.mode.value,
            taste_goal=recommendation.taste_goal,
        )

    def on_apply(event: RecommendationApplyEvent) -> None:
        recommendation = service.record_recommendation_apply(event)
        publish_status(
            recommendation.machine_id,
            recommendation.bean_context_id,
            recommendation.grinder_context_id,
            profile_id=recommendation.profile_id,
            last_recommendation_id=recommendation.recommendation_id,
            last_recommendation_at=recommendation.updated_at,
            mode=recommendation.mode.value,
            taste_goal=recommendation.taste_goal,
        )

    def on_optimizer_settings(event: OptimizerSettingsEvent) -> None:
        if event.install_id != config.install_id or not _same_machine_id(event.machine_id, config.machine_id):
            logger.warning(
                "Ignoring optimizer settings for unexpected owner install=%s machine=%s",
                event.install_id,
                event.machine_id,
            )
            return
        if event.optimizer_mode != OPTIMIZER_MODE_CPBO:
            raise ValueError("only CPBO is available")
        if event.recipe_domain is not None:
            cpbo_runtime.configure_recipe_domain(event.recipe_domain)
        publish_status(
            event.machine_id,
            event.bean_context_id,
            event.grinder_context_id,
            profile_id=event.profile_id,
            profile_label=event.profile_label,
            taste_goal=event.taste_goal,
        )

    def on_local_reset(event: LocalResetEvent) -> None:
        if event.install_id != config.install_id or not _same_machine_id(event.machine_id, config.machine_id):
            logger.warning("Ignoring local reset for unexpected owner")
            return
        result = local_data_service.reset_all(dry_run=event.dry_run)
        logger.warning("Local reset machine=%s dry_run=%s counts=%s", event.machine_id, event.dry_run, result.counts)
        if not event.dry_run:
            logger.warning("CPBO reset counts=%s", cpbo_runtime.reset_owner(event.install_id, event.machine_id))
            mqtt_client.clear_recommendation(event.machine_id)
        publish_status(event.machine_id, None, None, taste_goal=TasteGoal.balanced())

    def on_machine_state(event: MachineStateEvent) -> None:
        runtime_coordinator.handle_machine_state(event)

    mqtt_client = GaggimateMQTTClient(
        config=config,
        on_shot=runtime_coordinator.handle_shot,
        on_preference=on_preference,
        on_correction=on_correction,
        on_upload_maintenance=on_upload_maintenance,
        on_decision=on_decision,
        on_apply=on_apply,
        on_machine_state=on_machine_state,
        on_optimizer_settings=on_optimizer_settings,
        on_local_reset=on_local_reset,
    )

    def shutdown(sig: int, frame: object) -> None:
        logger.info("Shutting down (signal %d)", sig)
        stop_event.set()
        mqtt_client.stop()
        for thread in (upload_thread, collector_thread, dashboard_thread, local_dashboard_thread):
            if thread is not None:
                thread.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    mqtt_client.start()
    upload_thread = maybe_start_upload_worker(
        config,
        upload_queue_repo,
        stop_event,
        on_queue_changed=publish_upload_queue_status,
    )
    maybe_publish_startup_recommendation(
        config,
        service,
        mqtt_client,
        upload_maintenance=upload_maintenance,
        shot_repo=shot_repo,
        upload_queue_repo=upload_queue_repo,
    )
    logger.info("Listening for canonical CPBO events via Gaggimate MQTT adapter")
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
        for thread in (collector_thread, dashboard_thread):
            if thread is not None:
                thread.join(timeout=5)
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
    **_ignored: object,
) -> None:
    if config.machine_id == "gaggimate:local":
        return
    startup_taste_goal = TasteGoal.balanced()
    current = None
    if config.bean_context_id:
        current = service.get_current_recommendation(
            install_id=config.install_id,
            machine_id=config.machine_id,
            bean_context_id=config.bean_context_id,
            grinder_context_id=config.grinder_context_id,
            taste_goal_fingerprint=startup_taste_goal.fingerprint,
        )
    if current is not None:
        mqtt_client.publish_recommendation(current)
    else:
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
            last_recommendation_id=current.recommendation_id if current else None,
            last_recommendation_at=current.updated_at if current else None,
            mode=current.mode.value if current else None,
            taste_goal=startup_taste_goal,
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
    grinder_context_id: str | None = None,
    *,
    profile_id: str | None = None,
    profile_label: str | None = None,
    last_shot_id: str | None = None,
    last_shot_at: int | None = None,
    last_recommendation_id: str | None = None,
    last_recommendation_at: int | None = None,
    mode: str | None = None,
    taste_goal: TasteGoal | None = None,
    **_ignored: object,
) -> dict:
    now = config.now()
    active_taste_goal = taste_goal or TasteGoal.balanced()
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
    goal_recent = [
        shot
        for shot in all_recent
        if shot.taste_goal.fingerprint == active_taste_goal.fingerprint
    ]
    optimizer_profile_id = profile_id or next(
        (shot.profile_id for shot in reversed(goal_recent) if shot.profile_id),
        None,
    )
    recent = [
        shot
        for shot in goal_recent
        if optimizer_profile_id is None or _profile_ids_match(shot.profile_id, optimizer_profile_id)
    ]
    last_shot_record = (
        next((shot for shot in reversed(recent) if shot.shot_id == last_shot_id), None)
        if last_shot_id
        else (recent[-1] if recent else None)
    )
    if last_shot_record is not None:
        last_shot_id = last_shot_record.shot_id
        last_shot_at = last_shot_at or last_shot_record.timestamp
    optimizer_profile_label = profile_label or next(
        (shot.profile_label for shot in reversed(recent) if shot.profile_label),
        None,
    )
    optimizer_shots = [
        shot
        for shot in recent
        if shot.shot_type.value == "espresso"
        and not shot.exclude_from_local_optimization
        and shot.optimization_weight > 0.0
    ]
    current = service.get_current_recommendation(
        install_id=config.install_id,
        machine_id=machine_id,
        bean_context_id=bean_context_id,
        grinder_context_id=grinder_context_id,
        profile_id=optimizer_profile_id,
        taste_goal_fingerprint=active_taste_goal.fingerprint,
    )
    if current is not None:
        last_recommendation_id = last_recommendation_id or current.recommendation_id
        last_recommendation_at = last_recommendation_at or current.updated_at
        mode = mode or current.mode.value

    upload_queue_count = 0
    upload_queue_status_counts: dict[str, int] = {}
    if upload_queue_repo is not None:
        upload_queue_count = len(upload_queue_repo.list_ready(now=now, limit=1_000_000))
        upload_queue_status_counts = {
            status.value: count for status, count in upload_queue_repo.count_by_status().items()
        }
    rejected_summaries = (
        upload_maintenance.list_rejected(limit=20)
        if upload_maintenance is not None
        else []
    )
    latest_rejected = rejected_summaries[0] if rejected_summaries else None
    rejected_record_ids = {
        item.local_record_id
        for item in rejected_summaries
        if item.local_record_type == "shot" and item.local_record_id
    }
    community_upload_enabled = (
        config.should_enqueue_community_uploads()
        and service.community_upload_enabled_for(config.install_id, machine_id)
    )
    runtime_health = _runtime_health_payload(
        config=config,
        bean_context_id=bean_context_id,
        grinder_context_id=grinder_context_id,
        optimizer_shot_count=len(optimizer_shots),
        last_recommendation_id=last_recommendation_id,
        upload_queue_status_counts=upload_queue_status_counts,
        upload_queue_available=upload_queue_repo is not None,
        community_upload_requested=community_upload_enabled,
    )
    return {
        "addon_online": True,
        "install_id": config.install_id,
        "bean_context_id": bean_context_id,
        "grinder_context_id": grinder_context_id,
        "taste_goal": active_taste_goal.to_dict(),
        "taste_goal_summary": active_taste_goal.summary,
        "optimizer_profile_id": optimizer_profile_id,
        "optimizer_profile_label": optimizer_profile_label,
        **runtime_health,
        "auto_tuning_diagnostic_steps": _auto_tuning_diagnostic_steps(
            bean_context_id=bean_context_id,
            grinder_context_id=grinder_context_id,
            last_shot_record=last_shot_record,
            last_recommendation_id=last_recommendation_id,
            community_upload_enabled=community_upload_enabled,
            upload_queue_status_counts=upload_queue_status_counts,
        ),
        "timestamp": now,
        "last_shot_id": last_shot_id,
        "last_shot_at": last_shot_at,
        "last_shot_type": last_shot_record.shot_type.value if last_shot_record else None,
        "last_shot_time_s": last_shot_record.shot_time_s if last_shot_record else None,
        "last_shot_beverage_out_g": last_shot_record.beverage_out_g if last_shot_record else None,
        "last_shot_target_yield_g": last_shot_record.target_yield_g if last_shot_record else None,
        "recent_shots": recent_shot_summaries(recent, rejected_record_ids),
        "last_recommendation_id": last_recommendation_id,
        "last_recommendation_at": last_recommendation_at,
        "recommendation_apply_status": current.apply_status.value if current else None,
        "mode": mode,
        "optimizer_configured_mode": OPTIMIZER_MODE_CPBO,
        "optimizer_effective_mode": OPTIMIZER_MODE_CPBO,
        "optimizer_available_modes": [OPTIMIZER_MODE_CPBO],
        "optimizer_unavailable_modes": {},
        "optimizer_fallback_reason": None,
        "local_shot_count": len(optimizer_shots),
        "best_known_recipe": None,
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
    }


def _auto_tuning_diagnostic_steps(
    *,
    bean_context_id: str | None,
    grinder_context_id: str | None,
    last_shot_record,
    last_recommendation_id: str | None,
    community_upload_enabled: bool,
    upload_queue_status_counts: dict[str, int],
) -> list[dict[str, str]]:
    def step(key: str, label: str, state: str, detail: str) -> dict[str, str]:
        return {"key": key, "label": label, "state": state, "detail": detail}

    context_ready = bool(bean_context_id and grinder_context_id)
    steps = [
        step(
            "context",
            "Context",
            "ok" if context_ready else "waiting",
            "Bean and grinder selected." if context_ready else "Select a bean and grinder.",
        ),
        step(
            "shot_observed",
            "Physical shot",
            "ok" if last_shot_record is not None else "waiting",
            "Latest shot is stored." if last_shot_record is not None else "Waiting for a shot.",
        ),
        step(
            "preference",
            "Preference cycle",
            "ok" if last_recommendation_id else "waiting",
            "Candidate recipe is available."
            if last_recommendation_id
            else "A valid baseline or comparison is still required.",
        ),
    ]
    pending = upload_queue_status_counts.get(UploadQueueStatus.PENDING.value, 0)
    failed = upload_queue_status_counts.get(UploadQueueStatus.FAILED.value, 0)
    rejected = upload_queue_status_counts.get(UploadQueueStatus.REJECTED.value, 0)
    if not community_upload_enabled:
        state, detail = "off", "Community upload is disabled."
    elif rejected:
        state, detail = "attention", f"{rejected} upload record(s) rejected."
    elif failed:
        state, detail = "attention", f"{failed} upload record(s) waiting to retry."
    elif pending:
        state, detail = "waiting", f"{pending} upload record(s) queued."
    else:
        state, detail = "ok", "Upload queue is clear."
    steps.append(step("community_upload", "Community upload", state, detail))
    return steps


def _runtime_health_payload(
    *,
    config: Config,
    bean_context_id: str | None,
    grinder_context_id: str | None,
    optimizer_shot_count: int,
    last_recommendation_id: str | None,
    upload_queue_status_counts: dict[str, int],
    upload_queue_available: bool,
    community_upload_requested: bool,
) -> dict[str, object]:
    upload_configured = bool(config.supabase_ingest_url and config.upload_secret)
    pending = upload_queue_status_counts.get(UploadQueueStatus.PENDING.value, 0)
    failed = upload_queue_status_counts.get(UploadQueueStatus.FAILED.value, 0)
    rejected = upload_queue_status_counts.get(UploadQueueStatus.REJECTED.value, 0)
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
    if rejected:
        warnings.append(f"{rejected} community upload record(s) were rejected by the backend.")
    if failed:
        warnings.append(f"{failed} community upload record(s) are waiting to retry.")
    if warnings:
        status, summary = "attention", warnings[0]
    elif waiting_reasons:
        status, summary = "waiting", "Setup incomplete"
    elif optimizer_shot_count <= 0:
        status, summary = "waiting", "Ready - waiting for the first shot"
    elif not last_recommendation_id:
        status, summary = "waiting", "Ready - waiting for preference feedback"
    else:
        status, summary = "ok", "Connected"
    return {
        "runtime_health_status": status,
        "runtime_health_summary": summary,
        "runtime_health_warnings": warnings,
        "runtime_health_waiting_reasons": waiting_reasons,
        "runtime_health_storage_backend": config.storage_backend,
        "runtime_health_storage_available": True,
        "runtime_health_upload_configured": upload_configured,
        "runtime_health_community_upload_requested": community_upload_requested,
        "runtime_health_pending_upload_count": pending,
        "runtime_health_failed_upload_count": failed,
        "runtime_health_rejected_upload_count": rejected,
    }


def recent_shot_summaries(shots: list, rejected_record_ids: set[str], limit: int = 10) -> list[dict]:
    return [
        {
            "shot_id": shot.shot_id,
            "timestamp": shot.timestamp,
            "bean_context_id": shot.bean_context_id,
            "grinder_context_id": shot.grinder_context_id,
            "taste_goal": shot.taste_goal.to_dict(),
            "shot_type": shot.shot_type.value,
            "shot_time_s": shot.shot_time_s,
            "beverage_out_g": shot.beverage_out_g,
            "target_yield_g": shot.target_yield_g,
            "exclude_from_local_optimization": shot.exclude_from_local_optimization,
            "optimization_weight": shot.optimization_weight,
            "profile_label": shot.profile_label,
            "profile_type": shot.profile_type,
            "shot_end_state": shot.shot_end_state,
            "profile_flow_valid": shot.profile_flow_valid,
            "profile_flow_masked": shot.profile_flow_masked,
            "rejected_upload": shot.shot_id in rejected_record_ids,
        }
        for shot in reversed(shots[-limit:])
    ]


def grinder_catalog_search_url(config: Config) -> str:
    for source_url in (config.supabase_ingest_url, config.supabase_registration_url):
        derived = _derive_supabase_function_url(source_url, "espresso-rl-grinder-search")
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


def maybe_start_upload_worker(
    config: Config,
    upload_queue_repo: UploadQueueRepository,
    stop_event: threading.Event,
    *,
    on_queue_changed: Callable[[], None] | None = None,
) -> threading.Thread | None:
    if config.deployment_role == "admin" or not config.community_upload_enabled:
        return None
    if not config.supabase_ingest_url or not config.upload_secret:
        logger.warning("Community upload configured without an ingest URL/credential; records remain local.")
        return None
    worker = UploadQueueWorker(
        upload_queue_repo,
        SignedSupabaseUploadClient(
            SignedUploadConfig(
                ingest_url=config.supabase_ingest_url,
                install_id=config.install_id,
                upload_secret=config.upload_secret,
                upload_token_id=config.upload_token_id,
                max_payload_bytes=config.upload_max_payload_bytes,
            )
        ),
        clock=config.now,
        on_queue_changed=on_queue_changed,
    )

    def loop() -> None:
        while not stop_event.is_set():
            try:
                worker.run_once()
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
    if config.deployment_role == "admin" or not config.community_upload_enabled:
        return None
    configured = _configured_upload_credentials(config)
    if configured is not None:
        return _apply_upload_credentials(config, configured)
    credential_store = store or JsonCommunityCredentialStore(
        config.data_dir / "community_upload_credentials.json"
    )
    stored = credential_store.load()
    if stored is not None:
        return _apply_upload_credentials(config, stored)
    if registrar is None and not config.supabase_registration_url:
        logger.warning("Community upload has no credentials or registration URL.")
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
        logger.warning("Community upload registration failed: %s", exc)
        return None
    return _apply_upload_credentials(config, credentials) if credentials is not None else None


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
    if config.deployment_role != "admin" or not config.admin_collector_enabled:
        return None
    if config.storage_backend != "postgres":
        logger.warning("Admin collector requires Postgres storage.")
        return None
    if not config.supabase_rest_url or not config.supabase_service_role_key:
        logger.warning("Admin collector is missing Supabase admin credentials.")
        return None
    pipeline = admin_pipeline or build_admin_pipeline_service(config)

    def loop() -> None:
        last_source_purge_at = 0
        while not stop_event.is_set():
            try:
                pipeline.mirror_once(
                    limit=config.admin_collector_batch_size,
                    requested_by="admin_collector",
                )
                pipeline.validate_once(
                    limit=config.admin_collector_batch_size,
                    requested_by="admin_collector",
                )
                now = config.now()
                if config.admin_source_purge_enabled and now - last_source_purge_at >= 3600:
                    pipeline.purge_source_once(requested_by="admin_collector")
                    last_source_purge_at = now
            except Exception:
                logger.exception("Admin community pipeline cycle failed")
            stop_event.wait(config.admin_collector_interval_s)

    thread = threading.Thread(target=loop, name="espresso-rl-admin-mirror", daemon=True)
    thread.start()
    return thread


def maybe_start_admin_dashboard(
    config: Config,
    stop_event: threading.Event,
    admin_pipeline: AdminPipelineService | None = None,
) -> threading.Thread | None:
    if config.deployment_role != "admin" or not config.admin_dashboard_enabled:
        return None
    if config.storage_backend != "postgres":
        logger.warning("Admin dashboard requires Postgres storage.")
        return None
    if len(config.admin_dashboard_token) < 32:
        logger.warning("Admin dashboard token is missing or too short.")
        return None
    from espresso_rl.adapters.admin_dashboard import start_admin_dashboard

    return start_admin_dashboard(
        admin_pipeline or build_admin_pipeline_service(config),
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
    if config.deployment_role != "public" or not config.local_dashboard_enabled:
        return None
    if len(config.local_dashboard_token) < 32:
        logger.warning("Local dashboard token is missing or too short.")
        return None
    from espresso_rl.adapters.local_dashboard import start_local_dashboard

    return start_local_dashboard(
        local_data_service,
        upload_maintenance,
        local_token=config.local_dashboard_token,
        host=config.local_dashboard_host,
        port=config.local_dashboard_port,
        stop_event=stop_event,
    )


def maybe_build_admin_pipeline_service(config: Config) -> AdminPipelineService | None:
    if config.deployment_role != "admin" or config.storage_backend != "postgres":
        return None
    collector_ready = (
        config.admin_collector_enabled
        and bool(config.supabase_rest_url)
        and bool(config.supabase_service_role_key)
    )
    dashboard_ready = config.admin_dashboard_enabled and len(config.admin_dashboard_token) >= 32
    return build_admin_pipeline_service(config) if collector_ready or dashboard_ready else None


def build_admin_pipeline_service(config: Config) -> AdminPipelineService:
    warehouse = PostgresCommunityWarehouse(PostgresStore(config.postgres_dsn))
    mirror = None
    if config.supabase_rest_url and config.supabase_service_role_key:
        mirror = CommunityMirrorService(
            source=SupabaseCommunityQueueClient(
                SupabaseCommunityQueueConfig(
                    rest_url=config.supabase_rest_url,
                    service_role_key=config.supabase_service_role_key,
                    admin_id=config.admin_collector_id,
                    claim_lease_seconds=config.admin_collector_lease_seconds,
                )
            ),
            warehouse=warehouse,
        )
    return AdminPipelineService(
        warehouse=warehouse,
        mirror=mirror,
        validator=CommunityValidationService(warehouse=warehouse),
        offline_dataset_exporter=OfflineDatasetExportService(
            warehouse,
            clock=config.now,
            exporter_version=config.build_git_sha or "development",
        ),
        source_mirrored_retention_days=config.admin_source_mirrored_retention_days,
        source_rejected_retention_days=config.admin_source_rejected_retention_days,
        source_failed_retention_days=config.admin_source_failed_retention_days,
        clock=config.now,
    )


def upload_queue_for_service(
    config: Config,
    upload_queue_repo: UploadQueueRepository,
) -> UploadQueueRepository | None:
    return upload_queue_repo if config.should_enqueue_community_uploads() else None


def build_cpbo_recipe_space(
    baseline: Recipe,
    *,
    config: Config,
    recipe_domain: RecipeDomain,
) -> RecipeSpace:
    grind_resolution = (
        config.cpbo.stepless_grind_resolution
        if baseline.grinder_adjustment_mode == GrinderAdjustmentMode.STEPLESS
        else config.cpbo.stepped_grind_resolution
    )
    grind_radius = recipe_domain.grind_radius_steps
    return RecipeSpace(
        grind=RecipeParameter(
            name="grind_size",
            physical_min=baseline.relative_grind_steps_from_reference - grind_radius,
            physical_max=baseline.relative_grind_steps_from_reference + grind_radius,
            resolution=grind_resolution,
            unit="grinder_step_from_reference",
            constraints=("configured_recipe_domain",),
        ),
        dose=RecipeParameter(
            name="dose_g",
            physical_min=recipe_domain.dose_min_g,
            physical_max=recipe_domain.dose_max_g,
            resolution=config.cpbo.dose_resolution_g,
            unit="g",
            constraints=("configured_recipe_domain",),
        ),
        target_output=RecipeParameter(
            name="target_output_g",
            physical_min=recipe_domain.target_output_min_g,
            physical_max=recipe_domain.target_output_max_g,
            resolution=config.cpbo.target_output_resolution_g,
            unit="g",
            constraints=("configured_recipe_domain",),
        ),
        grinder_step_direction=baseline.grinder_step_direction,
        version=recipe_domain.effective_version,
    )


def open_repositories(
    config: Config,
) -> tuple[
    ShotRepository,
    RecommendationRepository,
    UploadQueueRepository,
    LocalDataRepository,
    PreferentialOptimizationRepository,
]:
    if config.storage_backend == "postgres":
        store = PostgresStore(config.postgres_dsn)
        return (
            PostgresShotRepository(store),
            PostgresRecommendationRepository(store),
            PostgresUploadQueueRepository(store),
            PostgresLocalDataRepository(store),
            PostgresPreferentialOptimizationRepository(store),
        )
    store = SQLiteStore(config.data_dir / "espresso_rl.db")
    return (
        SQLiteShotRepository(store),
        SQLiteRecommendationRepository(store),
        SQLiteUploadQueueRepository(store),
        SQLiteLocalDataRepository(store),
        SQLitePreferentialOptimizationRepository(store),
    )


def _same_machine_id(left: str, right: str) -> bool:
    if left == right:
        return True
    if left.startswith("gaggimate:") and right.startswith("gaggimate:"):
        return left.removeprefix("gaggimate:").casefold() == right.removeprefix(
            "gaggimate:"
        ).casefold()
    return False


def _profile_ids_match(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.strip().casefold() == right.strip().casefold()


if __name__ == "__main__":
    main()

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from espresso_rl.domain.optimization import DEFAULT_OPTIMIZER_MODE, normalize_optimizer_mode

_OPTIONS_PATH = Path("/data/options.json")
_DATA_DIR = Path("/data/espresso_rl")
_DEFAULT_DREAMER_V3_MODEL_ARTIFACT_PATH = _DATA_DIR / "models" / "dreamer_v3.pt"
_DEFAULT_DREAMER_V3_MODEL_MANIFEST_PATH = _DATA_DIR / "models" / "dreamer_v3_manifest.json"
_DEFAULT_MODEL_ARTIFACT_MAX_BYTES = 512 * 1024 * 1024
_DEFAULT_TRAINING_EXPORT_DIR = _DATA_DIR / "exports"
_RELEASE_DEFAULT_MODEL_ARTIFACT_SHA256 = os.getenv(
    "ESPRESSORL_RELEASE_MODEL_ARTIFACT_SHA256",
    "",
)


@dataclass
class Config:
    mqtt_host: str
    mqtt_port: int = 1883
    mqtt_user: str = ""
    mqtt_password: str = ""
    # Grinder geometry Ã¢â‚¬â€ user registers once
    microns_per_step: float = 10.0  # ÃŽÂ¼m per click/step
    grinder_model: str = ""
    install_id: str = "local_install"
    machine_id: str = "gaggimate:local"
    bean_context_id: str | None = None
    grinder_context_id: str | None = None
    # Machine geometry
    machine_pressure_bar: float = 9.0
    basket_size_ml: float = 18.0
    # Initial state Ã¢â‚¬â€ user sets these before first run
    initial_relative_grind_steps_from_reference: float | None = None
    initial_relative_grind_um_from_reference: float = 0.0
    initial_dose_g: float = 18.0
    initial_target_yield_g: float = 36.0
    # Reward weighting: r = alpha*human + (1-alpha)*profile_score
    alpha: float = 0.5
    optimizer_mode: str = DEFAULT_OPTIMIZER_MODE
    optimizer_model_artifact_path: str = ""
    optimizer_model_artifact_sha256: str = ""
    optimizer_model_manifest_path: str = ""
    default_optimizer_model_artifact_sha256: str = ""
    optimizer_model_artifact_max_bytes: int = _DEFAULT_MODEL_ARTIFACT_MAX_BYTES
    # When True: run DreamerV3 training thread locally (central-server install only).
    # When False (default): inference + BO only; downloads weights from central server.
    training_mode: bool = False
    community_upload_enabled: bool = False
    supabase_registration_url: str = ""
    supabase_ingest_url: str = ""
    upload_secret: str = ""
    upload_token_id: str = ""
    upload_worker_interval_s: float = 30.0
    upload_max_payload_bytes: int = 2_000_000
    storage_backend: str = "sqlite"
    postgres_dsn: str = ""
    deployment_role: str = "public"
    admin_collector_enabled: bool = False
    supabase_rest_url: str = ""
    supabase_service_role_key: str = ""
    admin_collector_id: str = "espresso-rl-admin"
    admin_collector_lease_seconds: int = 300
    admin_collector_interval_s: float = 30.0
    admin_collector_batch_size: int = 100
    training_export_dir: Path = field(default_factory=lambda: _DEFAULT_TRAINING_EXPORT_DIR)
    training_export_max_rows: int = 50_000
    build_git_sha: str = ""
    admin_dashboard_enabled: bool = False
    admin_dashboard_host: str = "0.0.0.0"
    admin_dashboard_port: int = 8080
    admin_dashboard_token: str = ""
    local_dashboard_enabled: bool = False
    local_dashboard_host: str = "0.0.0.0"
    local_dashboard_port: int = 8081
    local_dashboard_token: str = ""
    data_dir: Path = field(default_factory=lambda: _DATA_DIR)

    def __post_init__(self) -> None:
        self.optimizer_mode = normalize_optimizer_mode(self.optimizer_mode)
        if self.optimizer_model_artifact_max_bytes <= 0:
            raise ValueError("optimizer_model_artifact_max_bytes must be positive")
        if self.training_export_max_rows <= 0:
            raise ValueError("training_export_max_rows must be positive")

    def now(self) -> int:
        return int(time.time())

    def should_enqueue_community_uploads(self) -> bool:
        return self.community_upload_enabled and self.deployment_role != "admin"

    @classmethod
    def load(cls) -> "Config":
        if _OPTIONS_PATH.exists():
            opts: dict = json.loads(_OPTIONS_PATH.read_text())
        else:
            # Local dev fallback: accept env vars or plain defaults
            opts = {}

        data_dir = _DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)

        microns_per_step = float(opts.get("microns_per_step", 10.0))
        initial_relative_grind_um_from_reference = float(opts.get("initial_relative_grind_um_from_reference", 0.0))
        raw_steps = _optional_number(opts.get("initial_relative_grind_steps_from_reference"))
        if raw_steps is None:
            initial_relative_grind_steps_from_reference = initial_relative_grind_um_from_reference / microns_per_step if initial_relative_grind_um_from_reference else 0.0
        else:
            initial_relative_grind_steps_from_reference = raw_steps
        if initial_relative_grind_um_from_reference == 0.0 and initial_relative_grind_steps_from_reference:
            initial_relative_grind_um_from_reference = initial_relative_grind_steps_from_reference * microns_per_step

        storage_backend = str(
            opts.get("storage_backend", os.getenv("ESPRESSORL_STORAGE_BACKEND", "sqlite"))
        ).lower()
        deployment_role = str(
            opts.get(
                "deployment_role",
                os.getenv("ESPRESSORL_DEPLOYMENT_ROLE", "public"),
            )
        ).lower()
        if storage_backend not in {"postgres", "sqlite"}:
            raise ValueError("storage_backend must be 'postgres' or 'sqlite'")
        if deployment_role not in {"public", "admin"}:
            raise ValueError("deployment_role must be 'public' or 'admin'")

        return cls(
            mqtt_host=opts.get("mqtt_host", os.getenv("MQTT_HOST", "localhost")),
            mqtt_port=int(opts.get("mqtt_port", 1883)),
            mqtt_user=opts.get("mqtt_user", os.getenv("MQTT_USER", "")),
            mqtt_password=opts.get("mqtt_password", os.getenv("MQTT_PASSWORD", "")),
            microns_per_step=microns_per_step,
            grinder_model=opts.get("grinder_model", ""),
            install_id=opts.get("install_id", os.getenv("ESPRESSORL_INSTALL_ID", "local_install")),
            machine_id=opts.get("machine_id", "gaggimate:local"),
            bean_context_id=_optional_string(opts.get("bean_context_id")),
            grinder_context_id=_optional_string(
                opts.get("grinder_context_id", os.getenv("ESPRESSORL_GRINDER_CONTEXT_ID"))
            ),
            machine_pressure_bar=float(opts.get("machine_pressure_bar", 9.0)),
            basket_size_ml=float(opts.get("basket_size_ml", 18.0)),
            initial_relative_grind_steps_from_reference=initial_relative_grind_steps_from_reference,
            initial_relative_grind_um_from_reference=initial_relative_grind_um_from_reference,
            initial_dose_g=float(opts.get("initial_dose_g", 18.0)),
            initial_target_yield_g=float(opts.get("initial_target_yield_g", 36.0)),
            alpha=float(opts.get("alpha", 0.5)),
            optimizer_mode=normalize_optimizer_mode(
                opts.get(
                    "optimizer_mode",
                    os.getenv("ESPRESSORL_OPTIMIZER_MODE", DEFAULT_OPTIMIZER_MODE),
                )
            ),
            optimizer_model_artifact_path=_model_artifact_path(opts),
            optimizer_model_artifact_sha256=_model_artifact_sha256(opts),
            optimizer_model_manifest_path=_model_manifest_path(opts),
            default_optimizer_model_artifact_sha256=_option_string_or_env(
                opts,
                "default_optimizer_model_artifact_sha256",
                "ESPRESSORL_DEFAULT_OPTIMIZER_MODEL_ARTIFACT_SHA256",
                _RELEASE_DEFAULT_MODEL_ARTIFACT_SHA256,
            ),
            optimizer_model_artifact_max_bytes=int(
                opts.get(
                    "optimizer_model_artifact_max_bytes",
                    os.getenv(
                        "ESPRESSORL_OPTIMIZER_MODEL_ARTIFACT_MAX_BYTES",
                        _DEFAULT_MODEL_ARTIFACT_MAX_BYTES,
                    ),
                )
            ),
            training_mode=bool(opts.get("training_mode", False)),
            community_upload_enabled=bool(opts.get("community_upload_enabled", False)),
            supabase_registration_url=_option_string_or_env(
                opts,
                "supabase_registration_url",
                "ESPRESSORL_SUPABASE_REGISTRATION_URL",
            ),
            supabase_ingest_url=_option_string_or_env(
                opts,
                "supabase_ingest_url",
                "ESPRESSORL_SUPABASE_INGEST_URL",
            ),
            upload_secret=_option_string_or_env(opts, "upload_secret", "ESPRESSORL_UPLOAD_SECRET"),
            upload_token_id=_option_string_or_env(opts, "upload_token_id", "ESPRESSORL_UPLOAD_TOKEN_ID"),
            upload_worker_interval_s=float(opts.get("upload_worker_interval_s", 30.0)),
            upload_max_payload_bytes=int(opts.get("upload_max_payload_bytes", 2_000_000)),
            storage_backend=storage_backend,
            postgres_dsn=_option_string_or_env(opts, "postgres_dsn", "ESPRESSORL_POSTGRES_DSN"),
            deployment_role=deployment_role,
            admin_collector_enabled=bool(
                opts.get(
                    "admin_collector_enabled",
                    os.getenv("ESPRESSORL_ADMIN_COLLECTOR_ENABLED", "").lower()
                    in {"1", "true", "yes"},
                )
            ),
            supabase_rest_url=_option_string_or_env(
                opts,
                "supabase_rest_url",
                "ESPRESSORL_SUPABASE_REST_URL",
            ),
            supabase_service_role_key=_option_string_or_env(
                opts,
                "supabase_service_role_key",
                "ESPRESSORL_SUPABASE_SERVICE_ROLE_KEY",
            ),
            admin_collector_id=opts.get(
                "admin_collector_id",
                os.getenv("ESPRESSORL_ADMIN_COLLECTOR_ID", "espresso-rl-admin"),
            ),
            admin_collector_lease_seconds=int(
                opts.get(
                    "admin_collector_lease_seconds",
                    os.getenv("ESPRESSORL_ADMIN_COLLECTOR_LEASE_SECONDS", 300),
                )
            ),
            admin_collector_interval_s=float(
                opts.get(
                    "admin_collector_interval_s",
                    os.getenv("ESPRESSORL_ADMIN_COLLECTOR_INTERVAL_S", 30.0),
                )
            ),
            admin_collector_batch_size=int(
                opts.get(
                    "admin_collector_batch_size",
                    os.getenv("ESPRESSORL_ADMIN_COLLECTOR_BATCH_SIZE", 100),
                )
            ),
            training_export_dir=Path(
                _option_string_or_env(
                    opts,
                    "training_export_dir",
                    "ESPRESSORL_TRAINING_EXPORT_DIR",
                    str(_DEFAULT_TRAINING_EXPORT_DIR),
                )
            ),
            training_export_max_rows=int(
                opts.get(
                    "training_export_max_rows",
                    os.getenv("ESPRESSORL_TRAINING_EXPORT_MAX_ROWS", 50_000),
                )
            ),
            build_git_sha=_option_string_or_env(opts, "build_git_sha", "ESPRESSORL_BUILD_GIT_SHA"),
            admin_dashboard_enabled=bool(
                opts.get(
                    "admin_dashboard_enabled",
                    os.getenv("ESPRESSORL_ADMIN_DASHBOARD_ENABLED", "").lower()
                    in {"1", "true", "yes"},
                )
            ),
            admin_dashboard_host=opts.get(
                "admin_dashboard_host",
                os.getenv("ESPRESSORL_ADMIN_DASHBOARD_HOST", "0.0.0.0"),
            ),
            admin_dashboard_port=int(
                opts.get("admin_dashboard_port", os.getenv("ESPRESSORL_ADMIN_DASHBOARD_PORT", 8080))
            ),
            admin_dashboard_token=_option_string_or_env(
                opts,
                "admin_dashboard_token",
                "ESPRESSORL_ADMIN_DASHBOARD_TOKEN",
            ),
            local_dashboard_enabled=bool(
                opts.get(
                    "local_dashboard_enabled",
                    os.getenv("ESPRESSORL_LOCAL_DASHBOARD_ENABLED", "").lower()
                    in {"1", "true", "yes"},
                )
            ),
            local_dashboard_host=opts.get(
                "local_dashboard_host",
                os.getenv("ESPRESSORL_LOCAL_DASHBOARD_HOST", "0.0.0.0"),
            ),
            local_dashboard_port=int(
                opts.get("local_dashboard_port", os.getenv("ESPRESSORL_LOCAL_DASHBOARD_PORT", 8081))
            ),
            local_dashboard_token=_option_string_or_env(
                opts,
                "local_dashboard_token",
                "ESPRESSORL_LOCAL_DASHBOARD_TOKEN",
            ),
            data_dir=data_dir,
        )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _option_string_or_env(
    opts: dict,
    option_name: str,
    env_name: str,
    default: str = "",
) -> str:
    option_value = _optional_string(opts.get(option_name))
    if option_value is not None:
        return option_value
    return os.getenv(env_name, default)


def _model_artifact_sha256(opts: dict) -> str:
    release_default = _option_string_or_env(
        opts,
        "default_optimizer_model_artifact_sha256",
        "ESPRESSORL_DEFAULT_OPTIMIZER_MODEL_ARTIFACT_SHA256",
        _RELEASE_DEFAULT_MODEL_ARTIFACT_SHA256,
    )
    return _option_string_or_env(
        opts,
        "optimizer_model_artifact_sha256",
        "ESPRESSORL_OPTIMIZER_MODEL_ARTIFACT_SHA256",
        release_default,
    )


def _model_artifact_path(opts: dict) -> str:
    default_path = (
        str(_DEFAULT_DREAMER_V3_MODEL_ARTIFACT_PATH)
        if _model_artifact_sha256(opts)
        else ""
    )
    return _option_string_or_env(
        opts,
        "optimizer_model_artifact_path",
        "ESPRESSORL_OPTIMIZER_MODEL_ARTIFACT_PATH",
        default_path,
    )


def _model_manifest_path(opts: dict) -> str:
    default_path = (
        str(_DEFAULT_DREAMER_V3_MODEL_MANIFEST_PATH)
        if _model_artifact_sha256(opts)
        else ""
    )
    return _option_string_or_env(
        opts,
        "optimizer_model_manifest_path",
        "ESPRESSORL_OPTIMIZER_MODEL_MANIFEST_PATH",
        default_path,
    )


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if value == "":
        return None
    return float(value)

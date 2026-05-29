import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

_OPTIONS_PATH = Path("/data/options.json")
_DATA_DIR = Path("/data/espresso_rl")


@dataclass
class Config:
    mqtt_host: str
    mqtt_port: int = 1883
    mqtt_user: str = ""
    mqtt_password: str = ""
    # Grinder geometry — user registers once
    grinder_step_size_um: float = 10.0  # μm per click/step
    grinder_model: str = ""
    install_id: str = "local_install"
    machine_id: str = "gaggimate:local"
    bean_context_id: str | None = None
    # Machine geometry
    machine_pressure_bar: float = 9.0
    basket_size_ml: float = 18.0
    # Initial state — user sets these before first run
    initial_grind_steps: float = 0.0
    initial_grind_um: float = 0.0
    initial_dose_g: float = 18.0
    initial_target_yield_g: float = 36.0
    # Reward weighting: r = alpha*human + (1-alpha)*profile_score
    alpha: float = 0.5
    # When True: run DreamerV3 training thread locally (central-server install only).
    # When False (default): inference + BO only; downloads weights from central server.
    training_mode: bool = False
    community_upload_enabled: bool = False
    supabase_ingest_url: str = ""
    upload_secret: str = ""
    upload_token_id: str = ""
    upload_worker_interval_s: float = 30.0
    upload_max_payload_bytes: int = 2_000_000
    data_dir: Path = field(default_factory=lambda: _DATA_DIR)

    def now(self) -> int:
        return int(time.time())

    @classmethod
    def load(cls) -> "Config":
        if _OPTIONS_PATH.exists():
            opts: dict = json.loads(_OPTIONS_PATH.read_text())
        else:
            # Local dev fallback: accept env vars or plain defaults
            opts = {}

        data_dir = _DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)

        step_size_um = float(opts.get("grinder_step_size_um", 10.0))
        initial_grind_um = float(opts.get("initial_grind_um", 0.0))
        raw_steps = opts.get("initial_grind_steps")
        if raw_steps is None:
            initial_grind_steps = initial_grind_um / step_size_um if initial_grind_um else 0.0
        else:
            initial_grind_steps = float(raw_steps)
        if initial_grind_um == 0.0 and initial_grind_steps:
            initial_grind_um = initial_grind_steps * step_size_um

        return cls(
            mqtt_host=opts.get("mqtt_host", os.getenv("MQTT_HOST", "localhost")),
            mqtt_port=int(opts.get("mqtt_port", 1883)),
            mqtt_user=opts.get("mqtt_user", os.getenv("MQTT_USER", "")),
            mqtt_password=opts.get("mqtt_password", os.getenv("MQTT_PASSWORD", "")),
            grinder_step_size_um=step_size_um,
            grinder_model=opts.get("grinder_model", ""),
            install_id=opts.get("install_id", os.getenv("ESPRESSORL_INSTALL_ID", "local_install")),
            machine_id=opts.get("machine_id", "gaggimate:local"),
            bean_context_id=opts.get("bean_context_id"),
            machine_pressure_bar=float(opts.get("machine_pressure_bar", 9.0)),
            basket_size_ml=float(opts.get("basket_size_ml", 18.0)),
            initial_grind_steps=initial_grind_steps,
            initial_grind_um=initial_grind_um,
            initial_dose_g=float(opts.get("initial_dose_g", 18.0)),
            initial_target_yield_g=float(opts.get("initial_target_yield_g", 36.0)),
            alpha=float(opts.get("alpha", 0.5)),
            training_mode=bool(opts.get("training_mode", False)),
            community_upload_enabled=bool(opts.get("community_upload_enabled", False)),
            supabase_ingest_url=opts.get("supabase_ingest_url", ""),
            upload_secret=opts.get("upload_secret", ""),
            upload_token_id=opts.get("upload_token_id", ""),
            upload_worker_interval_s=float(opts.get("upload_worker_interval_s", 30.0)),
            upload_max_payload_bytes=int(opts.get("upload_max_payload_bytes", 2_000_000)),
            data_dir=data_dir,
        )

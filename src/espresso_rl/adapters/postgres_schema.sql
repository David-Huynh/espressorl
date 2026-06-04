CREATE TABLE IF NOT EXISTS shots (
    shot_id TEXT PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    install_id TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    machine_adapter TEXT NOT NULL,
    bean_context_id TEXT,
    profile_resampled_blob BYTEA NOT NULL,
    raw_profile_available BOOLEAN NOT NULL,
    raw_profile_hash TEXT,
    grind_steps DOUBLE PRECISION,
    grind_um DOUBLE PRECISION,
    grinder_step_size_um DOUBLE PRECISION NOT NULL,
    dose_in_g DOUBLE PRECISION NOT NULL,
    beverage_out_g DOUBLE PRECISION,
    brew_ratio DOUBLE PRECISION,
    target_yield_g DOUBLE PRECISION NOT NULL,
    target_ratio DOUBLE PRECISION,
    shot_time_s DOUBLE PRECISION,
    recommendation_id TEXT,
    recommended_grind_delta_steps INTEGER,
    recommended_grind_delta_um DOUBLE PRECISION,
    recommended_next_grind_steps DOUBLE PRECISION,
    recommended_dose_g DOUBLE PRECISION,
    recommended_target_yield_g DOUBLE PRECISION,
    recommended_target_ratio DOUBLE PRECISION,
    recommendation_decision TEXT NOT NULL,
    recommendation_followed TEXT NOT NULL,
    recommendation_attribution_weight DOUBLE PRECISION NOT NULL,
    human_rating INTEGER,
    taste_tags_json TEXT NOT NULL,
    profile_score DOUBLE PRECISION,
    profile_mse DOUBLE PRECISION,
    reward DOUBLE PRECISION,
    reward_confidence DOUBLE PRECISION NOT NULL,
    shot_type TEXT NOT NULL DEFAULT 'espresso',
    exclude_from_local_optimization BOOLEAN NOT NULL DEFAULT FALSE,
    optimization_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    rating_prompt_allowed BOOLEAN NOT NULL DEFAULT TRUE,
    grind_followed BOOLEAN,
    dose_followed BOOLEAN,
    yield_followed BOOLEAN,
    grind_recommendation_trust DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    dose_recommendation_trust DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    yield_recommendation_trust DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    weight_source TEXT,
    flow_source TEXT,
    flow_units TEXT,
    pump_flow_source TEXT,
    pump_flow_units TEXT,
    pump_flow_calibration_required BOOLEAN NOT NULL DEFAULT FALSE,
    profile_flow_valid BOOLEAN NOT NULL DEFAULT TRUE,
    profile_flow_masked BOOLEAN NOT NULL DEFAULT FALSE,
    profile_id TEXT,
    profile_label TEXT,
    profile_type TEXT,
    profile_phase_count INTEGER,
    final_phase_index INTEGER,
    final_phase_name TEXT,
    final_phase_type TEXT,
    final_phase_elapsed_s DOUBLE PRECISION,
    final_pump_target TEXT,
    final_target_pressure DOUBLE PRECISION,
    final_target_flow DOUBLE PRECISION,
    final_valve_open BOOLEAN,
    profile_temperature_c DOUBLE PRECISION,
    final_phase_temperature_c DOUBLE PRECISION,
    shot_end_state TEXT,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_shots_context_time
    ON shots (install_id, machine_id, bean_context_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id TEXT PRIMARY KEY,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    expires_at BIGINT,
    install_id TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    bean_context_id TEXT,
    grind_delta_steps INTEGER NOT NULL,
    grind_delta_um DOUBLE PRECISION NOT NULL,
    next_grind_steps DOUBLE PRECISION NOT NULL,
    next_grind_um DOUBLE PRECISION NOT NULL,
    next_dose_g DOUBLE PRECISION NOT NULL,
    target_yield_g DOUBLE PRECISION NOT NULL,
    target_ratio DOUBLE PRECISION NOT NULL,
    mode TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    shown_count INTEGER NOT NULL,
    accepted_at BIGINT,
    ignored_at BIGINT,
    edited_at BIGINT,
    used_at BIGINT,
    superseded_at BIGINT,
    source_shot_id TEXT,
    apply_status TEXT NOT NULL DEFAULT 'unknown',
    apply_acknowledged_at BIGINT,
    applied_fields_json TEXT NOT NULL DEFAULT '{}',
    manual_fields_json TEXT NOT NULL DEFAULT '[]',
    apply_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_recommendations_current
    ON recommendations (install_id, machine_id, bean_context_id, created_at DESC);

CREATE TABLE IF NOT EXISTS upload_queue (
    upload_id TEXT PRIMARY KEY,
    local_record_type TEXT NOT NULL,
    local_record_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL,
    last_attempt_at BIGINT,
    next_retry_at BIGINT,
    error_message TEXT,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_upload_queue_ready
    ON upload_queue (status, next_retry_at, created_at);

CREATE INDEX IF NOT EXISTS idx_upload_queue_record
    ON upload_queue (local_record_type, local_record_id);

-- Admin/training warehouse tables. These are populated by an admin collector
-- from the community-fed Supabase raw queue. Public clients must not write here.
CREATE TABLE IF NOT EXISTS community_raw_uploads (
    install_id TEXT NOT NULL,
    upload_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    supabase_received_at TIMESTAMPTZ,
    mirrored_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL DEFAULT 'mirrored',
    validated_at TIMESTAMPTZ,
    rejected_at TIMESTAMPTZ,
    validation_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    validation_errors JSONB NOT NULL DEFAULT '[]'::jsonb,
    PRIMARY KEY (install_id, upload_id),
    UNIQUE (install_id, payload_hash)
);

CREATE INDEX IF NOT EXISTS idx_community_raw_uploads_status
    ON community_raw_uploads (status, mirrored_at);

CREATE TABLE IF NOT EXISTS community_validated_shots (
    validation_id BIGSERIAL PRIMARY KEY,
    install_id TEXT NOT NULL,
    upload_id TEXT NOT NULL,
    shot_id TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    trust_weight DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    validation_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (install_id, shot_id)
);

CREATE TABLE IF NOT EXISTS community_recommendations (
    install_id TEXT NOT NULL,
    recommendation_id TEXT NOT NULL,
    upload_id TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (install_id, recommendation_id)
);

CREATE TABLE IF NOT EXISTS install_trust_scores (
    install_id TEXT PRIMARY KEY,
    trust_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    reason TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS abuse_events (
    event_id BIGSERIAL PRIMARY KEY,
    install_id TEXT,
    upload_id TEXT,
    payload_hash TEXT,
    reason TEXT NOT NULL,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_dataset (
    training_row_id BIGSERIAL PRIMARY KEY,
    source_validation_id BIGINT REFERENCES community_validated_shots(validation_id),
    payload_json JSONB NOT NULL,
    trust_weight DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_validation_id)
);

CREATE TABLE IF NOT EXISTS community_priors (
    prior_id BIGSERIAL PRIMARY KEY,
    context_key TEXT NOT NULL,
    prior_json JSONB NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_community_priors_context_key
    ON community_priors (context_key);

CREATE TABLE IF NOT EXISTS admin_action_log (
    action_id BIGSERIAL PRIMARY KEY,
    action_type TEXT NOT NULL,
    requested_at BIGINT NOT NULL,
    requested_by TEXT NOT NULL,
    dry_run BOOLEAN NOT NULL DEFAULT FALSE,
    status TEXT NOT NULL,
    rows_seen INTEGER NOT NULL DEFAULT 0,
    rows_changed INTEGER NOT NULL DEFAULT 0,
    warnings_count INTEGER NOT NULL DEFAULT 0,
    error_summary TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_admin_action_log_requested_at
    ON admin_action_log (requested_at DESC, action_id DESC);

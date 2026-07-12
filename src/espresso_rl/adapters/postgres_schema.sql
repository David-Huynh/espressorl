CREATE TABLE IF NOT EXISTS shots (
    shot_id TEXT PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    install_id TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    machine_adapter TEXT NOT NULL,
    bean_context_id TEXT,
    bean_context_name TEXT,
    grinder_context_id TEXT,
    taste_goal_json TEXT NOT NULL DEFAULT '{"schema_version":1,"mode":"balanced","targets":{}}',
    taste_goal_fingerprint TEXT NOT NULL DEFAULT 'aba2fe71b86b279da2f84ba25b616fbcdd127274d5f5fa393c8e0576777744a8',
    profile_resampled_blob BYTEA NOT NULL,
    raw_profile_available BOOLEAN NOT NULL,
    raw_profile_hash TEXT,
    relative_grind_steps_from_reference DOUBLE PRECISION,
    relative_grind_um_from_reference DOUBLE PRECISION,
    microns_per_step DOUBLE PRECISION NOT NULL,
    dose_in_g DOUBLE PRECISION NOT NULL,
    grind_observed BOOLEAN NOT NULL DEFAULT TRUE,
    dose_observed BOOLEAN NOT NULL DEFAULT TRUE,
    dose_target_g DOUBLE PRECISION,
    beverage_out_g DOUBLE PRECISION,
    brew_ratio DOUBLE PRECISION,
    target_yield_g DOUBLE PRECISION NOT NULL,
    target_yield_observed BOOLEAN NOT NULL DEFAULT TRUE,
    target_ratio DOUBLE PRECISION,
    shot_time_s DOUBLE PRECISION,
    recommendation_id TEXT,
    recommended_grind_delta_steps_from_current DOUBLE PRECISION,
    recommended_grind_delta_um_from_current DOUBLE PRECISION,
    recommended_projected_relative_step_from_reference DOUBLE PRECISION,
    recommended_dose_g DOUBLE PRECISION,
    recommended_target_yield_g DOUBLE PRECISION,
    recommended_target_ratio DOUBLE PRECISION,
    recommendation_decision TEXT NOT NULL,
    recommendation_followed TEXT NOT NULL,
    recommendation_attribution_weight DOUBLE PRECISION NOT NULL,
    shot_type TEXT NOT NULL DEFAULT 'espresso',
    exclude_from_local_optimization BOOLEAN NOT NULL DEFAULT FALSE,
    optimization_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
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
    beverage_flow_profile_blob BYTEA,
    temperature_profile_blob BYTEA,
    target_temperature_profile_blob BYTEA,
    pump_target_mode_profile_blob BYTEA,
    fixed_cadence_sequence_json TEXT,
    shot_end_state TEXT,
    grinder_calibration_mode TEXT NOT NULL DEFAULT 'relative_calibrated',
    grinder_step_direction TEXT NOT NULL DEFAULT 'higher_is_finer',
    grinder_adjustment_mode TEXT NOT NULL DEFAULT 'stepped',
    grinder_reference_label TEXT NOT NULL DEFAULT 'reference',
    current_absolute_step DOUBLE PRECISION,
    absolute_reference_step DOUBLE PRECISION,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_shots_context_time
    ON shots (install_id, machine_id, bean_context_id, timestamp DESC);

ALTER TABLE shots
    ADD COLUMN IF NOT EXISTS grinder_context_id TEXT;

ALTER TABLE shots
    ADD COLUMN IF NOT EXISTS taste_goal_json TEXT NOT NULL
    DEFAULT '{"schema_version":1,"mode":"balanced","targets":{}}';

ALTER TABLE shots
    ADD COLUMN IF NOT EXISTS taste_goal_fingerprint TEXT NOT NULL
    DEFAULT 'aba2fe71b86b279da2f84ba25b616fbcdd127274d5f5fa393c8e0576777744a8';

ALTER TABLE shots
    ADD COLUMN IF NOT EXISTS dose_target_g DOUBLE PRECISION;

ALTER TABLE shots
    ADD COLUMN IF NOT EXISTS relative_grind_steps_from_reference DOUBLE PRECISION;

ALTER TABLE shots
    ADD COLUMN IF NOT EXISTS relative_grind_um_from_reference DOUBLE PRECISION;

ALTER TABLE shots
    ADD COLUMN IF NOT EXISTS microns_per_step DOUBLE PRECISION NOT NULL DEFAULT 12.5;

ALTER TABLE shots
    ADD COLUMN IF NOT EXISTS recommended_grind_delta_um_from_current DOUBLE PRECISION;

ALTER TABLE shots
    ADD COLUMN IF NOT EXISTS recommended_projected_relative_step_from_reference DOUBLE PRECISION;

ALTER TABLE shots
    ADD COLUMN IF NOT EXISTS beverage_flow_profile_blob BYTEA;

ALTER TABLE shots
    ADD COLUMN IF NOT EXISTS temperature_profile_blob BYTEA;

ALTER TABLE shots
    ADD COLUMN IF NOT EXISTS target_temperature_profile_blob BYTEA;

ALTER TABLE shots
    ADD COLUMN IF NOT EXISTS pump_target_mode_profile_blob BYTEA;

ALTER TABLE shots
    ADD COLUMN IF NOT EXISTS fixed_cadence_sequence_json TEXT;

CREATE INDEX IF NOT EXISTS idx_shots_context_grinder_time
    ON shots (install_id, machine_id, bean_context_id, grinder_context_id, timestamp DESC);

ALTER TABLE shots
    ADD COLUMN IF NOT EXISTS recommended_grind_delta_steps_from_current DOUBLE PRECISION;

ALTER TABLE shots
    ALTER COLUMN recommended_grind_delta_steps_from_current TYPE DOUBLE PRECISION
    USING recommended_grind_delta_steps_from_current::DOUBLE PRECISION;

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id TEXT PRIMARY KEY,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    expires_at BIGINT,
    install_id TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    bean_context_id TEXT,
    grinder_context_id TEXT,
    profile_id TEXT,
    raw_profile_hash TEXT,
    taste_goal_json TEXT NOT NULL DEFAULT '{"schema_version":1,"mode":"balanced","targets":{}}',
    taste_goal_fingerprint TEXT NOT NULL DEFAULT 'aba2fe71b86b279da2f84ba25b616fbcdd127274d5f5fa393c8e0576777744a8',
    grind_delta_steps_from_current DOUBLE PRECISION NOT NULL,
    grind_delta_um_from_current DOUBLE PRECISION NOT NULL,
    projected_relative_step_from_reference DOUBLE PRECISION NOT NULL,
    projected_relative_grind_um_from_reference DOUBLE PRECISION NOT NULL,
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
    optimization_run_id TEXT,
    comparison_anchor_shot_id TEXT,
    comparison_mode TEXT,
    preference_feedback_required BOOLEAN NOT NULL DEFAULT FALSE,
    apply_status TEXT NOT NULL DEFAULT 'unknown',
    apply_acknowledged_at BIGINT,
    applied_fields_json TEXT NOT NULL DEFAULT '{}',
    manual_fields_json TEXT NOT NULL DEFAULT '[]',
    apply_error TEXT,
    grinder_calibration_mode TEXT NOT NULL DEFAULT 'relative_calibrated',
    grinder_step_direction TEXT NOT NULL DEFAULT 'higher_is_finer',
    grinder_adjustment_mode TEXT NOT NULL DEFAULT 'stepped',
    grinder_reference_label TEXT NOT NULL DEFAULT 'reference',
    current_absolute_step DOUBLE PRECISION,
    absolute_reference_step DOUBLE PRECISION,
    projected_absolute_step DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_recommendations_current
    ON recommendations (install_id, machine_id, bean_context_id, created_at DESC);

ALTER TABLE recommendations
    ADD COLUMN IF NOT EXISTS grinder_context_id TEXT;

ALTER TABLE recommendations
    ADD COLUMN IF NOT EXISTS profile_id TEXT;

ALTER TABLE recommendations
    ADD COLUMN IF NOT EXISTS raw_profile_hash TEXT;

ALTER TABLE recommendations
    ADD COLUMN IF NOT EXISTS taste_goal_json TEXT NOT NULL
    DEFAULT '{"schema_version":1,"mode":"balanced","targets":{}}';

ALTER TABLE recommendations
    ADD COLUMN IF NOT EXISTS taste_goal_fingerprint TEXT NOT NULL
    DEFAULT 'aba2fe71b86b279da2f84ba25b616fbcdd127274d5f5fa393c8e0576777744a8';

ALTER TABLE recommendations
    ADD COLUMN IF NOT EXISTS grind_delta_steps_from_current DOUBLE PRECISION NOT NULL DEFAULT 0.0;

ALTER TABLE recommendations
    ADD COLUMN IF NOT EXISTS grind_delta_um_from_current DOUBLE PRECISION NOT NULL DEFAULT 0.0;

ALTER TABLE recommendations
    ADD COLUMN IF NOT EXISTS projected_relative_step_from_reference DOUBLE PRECISION NOT NULL DEFAULT 0.0;

ALTER TABLE recommendations
    ADD COLUMN IF NOT EXISTS projected_relative_grind_um_from_reference DOUBLE PRECISION NOT NULL DEFAULT 0.0;

ALTER TABLE recommendations
    ALTER COLUMN grind_delta_steps_from_current TYPE DOUBLE PRECISION
    USING grind_delta_steps_from_current::DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_recommendations_context_grinder_time
    ON recommendations (install_id, machine_id, bean_context_id, grinder_context_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_recommendations_context_goal_time
    ON recommendations (
        install_id, machine_id, bean_context_id, grinder_context_id,
        taste_goal_fingerprint, created_at DESC
    );

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

-- Algorithm-neutral pairwise feedback. Optimizers and offline learners join
-- these immutable comparisons to the two physical shot trajectories.
CREATE TABLE IF NOT EXISTS community_comparisons (
    install_id TEXT NOT NULL,
    comparison_id TEXT NOT NULL,
    upload_id TEXT NOT NULL,
    optimization_run_id TEXT NOT NULL,
    new_shot_id TEXT NOT NULL,
    anchor_shot_id TEXT NOT NULL,
    label TEXT NOT NULL,
    comparison_mode TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    trust_weight DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    validation_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (install_id, comparison_id),
    CHECK (new_shot_id <> anchor_shot_id),
    CHECK (trust_weight >= 0.0 AND trust_weight <= 1.0),
    CHECK (label IN ('new_better', 'anchor_better', 'tie')),
    CHECK (comparison_mode IN ('global_previous', 'best_incumbent'))
);

CREATE INDEX IF NOT EXISTS idx_community_comparisons_run
    ON community_comparisons (install_id, optimization_run_id, created_at);

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

CREATE TABLE IF NOT EXISTS community_grinder_catalog (
    grinder_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    manufacturer TEXT,
    model TEXT,
    microns_per_step DOUBLE PRECISION,
    min_steps INTEGER,
    max_steps INTEGER,
    step_direction TEXT,
    source TEXT NOT NULL DEFAULT 'community',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (microns_per_step IS NULL OR microns_per_step > 0),
    CHECK (min_steps IS NULL OR min_steps >= 0),
    CHECK (max_steps IS NULL OR max_steps > 0),
    CHECK (min_steps IS NULL OR max_steps IS NULL OR min_steps < max_steps),
    CHECK (step_direction IS NULL OR step_direction IN ('higher_is_finer', 'higher_is_coarser')),
    CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

ALTER TABLE community_grinder_catalog
    ADD COLUMN IF NOT EXISTS min_steps INTEGER;

CREATE TABLE IF NOT EXISTS community_grinder_aliases (
    alias_id BIGSERIAL PRIMARY KEY,
    grinder_id TEXT NOT NULL REFERENCES community_grinder_catalog(grinder_id) ON DELETE CASCADE,
    alias_name TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'community',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (normalized_alias),
    CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

CREATE INDEX IF NOT EXISTS idx_community_grinder_aliases_grinder_id
    ON community_grinder_aliases (grinder_id);

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

CREATE TABLE IF NOT EXISTS cpbo_runs (
    run_id TEXT PRIMARY KEY,
    context_fingerprint TEXT NOT NULL,
    install_id TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    active BOOLEAN NOT NULL,
    created_at BIGINT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_cpbo_runs_active_context
    ON cpbo_runs (context_fingerprint)
    WHERE active = TRUE;

CREATE TABLE IF NOT EXISTS cpbo_recipes (
    recipe_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cpbo_recipes_run
    ON cpbo_recipes (run_id, created_at, recipe_id);

CREATE TABLE IF NOT EXISTS cpbo_shots (
    shot_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    recipe_id TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE (run_id, sequence_number)
);

CREATE INDEX IF NOT EXISTS idx_cpbo_shots_run_sequence
    ON cpbo_shots (run_id, sequence_number);

CREATE TABLE IF NOT EXISTS cpbo_comparisons (
    comparison_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    new_shot_id TEXT NOT NULL UNIQUE,
    anchor_shot_id TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cpbo_comparisons_run_created
    ON cpbo_comparisons (run_id, created_at, comparison_id);

CREATE TABLE IF NOT EXISTS cpbo_states (
    run_id TEXT PRIMARY KEY,
    updated_at BIGINT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cpbo_suggestions (
    suggestion_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    recipe_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cpbo_suggestions_pending
    ON cpbo_suggestions (run_id, status, created_at DESC);

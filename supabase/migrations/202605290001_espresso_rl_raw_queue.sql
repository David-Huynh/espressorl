-- EspressoRL public community upload queue.
--
-- Public add-ons must upload through the Edge Function. The function verifies
-- signatures and writes only to raw_upload_queue. Admin collectors mirror these
-- raw rows into local Postgres for validation, trust scoring, and training.

CREATE TABLE IF NOT EXISTS public.espressorl_upload_credentials (
    install_id TEXT NOT NULL,
    upload_token_id TEXT NOT NULL DEFAULT '',
    upload_secret TEXT NOT NULL,
    community_upload_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (install_id, upload_token_id),
    CHECK (length(install_id) BETWEEN 1 AND 128),
    CHECK (length(upload_token_id) <= 128),
    CHECK (length(upload_secret) >= 32)
);

CREATE TABLE IF NOT EXISTS public.raw_upload_queue (
    install_id TEXT NOT NULL,
    upload_id TEXT NOT NULL,
    upload_token_id TEXT NOT NULL DEFAULT '',
    payload_hash TEXT NOT NULL,
    local_record_type TEXT NOT NULL,
    local_record_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    client_timestamp BIGINT,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL DEFAULT 'queued',
    mirror_error TEXT,
    mirror_claimed_by TEXT,
    mirror_claimed_at TIMESTAMPTZ,
    mirror_claim_expires_at TIMESTAMPTZ,
    mirror_completed_at TIMESTAMPTZ,
    mirror_attempt_count INTEGER NOT NULL DEFAULT 0,
    validation_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_ip_hash TEXT,
    PRIMARY KEY (install_id, upload_id),
    UNIQUE (install_id, payload_hash),
    CHECK (length(upload_id) BETWEEN 1 AND 256),
    CHECK (payload_hash ~ '^[a-f0-9]{64}$'),
    CHECK (local_record_type IN ('shot', 'recommendation')),
    CHECK (event_type IN ('shot_record', 'recommendation_record')),
    CHECK (status IN ('queued', 'mirroring', 'mirrored', 'mirror_failed', 'rejected'))
);

CREATE INDEX IF NOT EXISTS idx_raw_upload_queue_status_received
    ON public.raw_upload_queue (status, received_at);

CREATE INDEX IF NOT EXISTS idx_raw_upload_queue_claim_expiry
    ON public.raw_upload_queue (status, mirror_claim_expires_at, received_at);

CREATE TABLE IF NOT EXISTS public.espressorl_ingest_rate_counters (
    scope TEXT NOT NULL,
    bucket_start TIMESTAMPTZ NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (scope, bucket_start)
);

CREATE TABLE IF NOT EXISTS public.espressorl_abuse_events (
    event_id BIGSERIAL PRIMARY KEY,
    install_id TEXT,
    upload_id TEXT,
    payload_hash TEXT,
    source_ip_hash TEXT,
    reason TEXT NOT NULL,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION public.espressorl_consume_rate_limit(
    p_scope TEXT,
    p_bucket_start TIMESTAMPTZ,
    p_limit INTEGER
) RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    current_count INTEGER;
BEGIN
    INSERT INTO public.espressorl_ingest_rate_counters (scope, bucket_start, count, updated_at)
    VALUES (p_scope, p_bucket_start, 1, now())
    ON CONFLICT (scope, bucket_start)
    DO UPDATE SET
        count = public.espressorl_ingest_rate_counters.count + 1,
        updated_at = now()
    RETURNING count INTO current_count;

    RETURN current_count <= p_limit;
END;
$$;

CREATE OR REPLACE FUNCTION public.espressorl_claim_raw_uploads(
    p_claimed_by TEXT,
    p_limit INTEGER DEFAULT 100,
    p_lease_seconds INTEGER DEFAULT 300
) RETURNS TABLE (
    install_id TEXT,
    upload_id TEXT,
    payload_hash TEXT,
    event_type TEXT,
    payload_json JSONB,
    received_at TIMESTAMPTZ
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    RETURN QUERY
    WITH candidates AS (
        SELECT q.install_id, q.upload_id
        FROM public.raw_upload_queue q
        WHERE q.status = 'queued'
           OR (
                q.status = 'mirroring'
                AND q.mirror_claim_expires_at IS NOT NULL
                AND q.mirror_claim_expires_at < now()
           )
        ORDER BY q.received_at ASC
        FOR UPDATE SKIP LOCKED
        LIMIT LEAST(GREATEST(p_limit, 1), 500)
    ),
    claimed AS (
        UPDATE public.raw_upload_queue q
        SET
            status = 'mirroring',
            mirror_claimed_by = p_claimed_by,
            mirror_claimed_at = now(),
            mirror_claim_expires_at = now() + make_interval(secs => LEAST(GREATEST(p_lease_seconds, 60), 3600)),
            mirror_attempt_count = q.mirror_attempt_count + 1,
            mirror_error = NULL
        FROM candidates c
        WHERE q.install_id = c.install_id
          AND q.upload_id = c.upload_id
        RETURNING
            q.install_id,
            q.upload_id,
            q.payload_hash,
            q.event_type,
            q.payload_json,
            q.received_at
    )
    SELECT
        claimed.install_id,
        claimed.upload_id,
        claimed.payload_hash,
        claimed.event_type,
        claimed.payload_json,
        claimed.received_at
    FROM claimed;
END;
$$;

CREATE OR REPLACE FUNCTION public.espressorl_purge_raw_upload_queue(
    p_mirrored_retention_days INTEGER DEFAULT 14,
    p_rejected_retention_days INTEGER DEFAULT 30,
    p_failed_retention_days INTEGER DEFAULT 90
) RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM public.raw_upload_queue q
    WHERE (
            q.status = 'mirrored'
            AND q.mirror_completed_at < now() - make_interval(days => GREATEST(p_mirrored_retention_days, 1))
        )
       OR (
            q.status = 'rejected'
            AND q.received_at < now() - make_interval(days => GREATEST(p_rejected_retention_days, 1))
        )
       OR (
            q.status = 'mirror_failed'
            AND q.mirror_completed_at < now() - make_interval(days => GREATEST(p_failed_retention_days, 1))
        );

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;

ALTER TABLE public.espressorl_upload_credentials ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.raw_upload_queue ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.espressorl_ingest_rate_counters ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.espressorl_abuse_events ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON public.espressorl_upload_credentials FROM anon, authenticated;
REVOKE ALL ON public.raw_upload_queue FROM anon, authenticated;
REVOKE ALL ON public.espressorl_ingest_rate_counters FROM anon, authenticated;
REVOKE ALL ON public.espressorl_abuse_events FROM anon, authenticated;
REVOKE ALL ON FUNCTION public.espressorl_consume_rate_limit(TEXT, TIMESTAMPTZ, INTEGER)
    FROM anon, authenticated;
REVOKE ALL ON FUNCTION public.espressorl_claim_raw_uploads(TEXT, INTEGER, INTEGER)
    FROM anon, authenticated;
REVOKE ALL ON FUNCTION public.espressorl_purge_raw_upload_queue(INTEGER, INTEGER, INTEGER)
    FROM anon, authenticated;

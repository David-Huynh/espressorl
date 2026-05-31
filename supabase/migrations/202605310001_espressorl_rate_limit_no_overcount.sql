-- Make the ingest rate limiter count *accepted* uploads, not attempts.
--
-- The original espressorl_consume_rate_limit incremented the counter on every
-- call and only then compared against the limit, so a client stuck retrying past
-- its quota drove the bucket far above the limit (observed: day counter 627 with a
-- limit of 50) and every rejected attempt still wrote to the counter. This version
-- only increments while the bucket is under the limit; once at/over the limit it
-- denies without counting. Rejected attempts remain visible in
-- espressorl_abuse_events. The counter now tops out at the limit.

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
    WHERE public.espressorl_ingest_rate_counters.count < p_limit
    RETURNING count INTO current_count;

    -- A pre-existing row at/over the limit matches neither the INSERT nor the
    -- conditional UPDATE, so RETURNING yields nothing: deny without counting.
    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;

    RETURN current_count <= p_limit;
END;
$$;

REVOKE ALL ON FUNCTION public.espressorl_consume_rate_limit(TEXT, TIMESTAMPTZ, INTEGER)
    FROM anon, authenticated;

-- Community grinder catalog metadata.
--
-- The catalog is for grinder-name matching, aliases, and optional display
-- convenience such as microns/step or max step count. Optimizer observations
-- and priors must continue to use normalized relative grind values.

CREATE TABLE IF NOT EXISTS public.espressorl_grinder_catalog (
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
    CHECK (length(grinder_id) BETWEEN 1 AND 128),
    CHECK (length(canonical_name) BETWEEN 1 AND 160),
    CHECK (microns_per_step IS NULL OR microns_per_step > 0),
    CHECK (min_steps IS NULL OR min_steps >= 0),
    CHECK (max_steps IS NULL OR max_steps > 0),
    CHECK (min_steps IS NULL OR max_steps IS NULL OR min_steps < max_steps),
    CHECK (step_direction IS NULL OR step_direction IN ('higher_is_finer', 'higher_is_coarser')),
    CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

CREATE TABLE IF NOT EXISTS public.espressorl_grinder_aliases (
    alias_id BIGSERIAL PRIMARY KEY,
    grinder_id TEXT NOT NULL REFERENCES public.espressorl_grinder_catalog(grinder_id) ON DELETE CASCADE,
    alias_name TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'community',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (normalized_alias),
    CHECK (length(alias_name) BETWEEN 1 AND 160),
    CHECK (length(normalized_alias) BETWEEN 1 AND 160),
    CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

CREATE INDEX IF NOT EXISTS idx_espressorl_grinder_aliases_grinder_id
    ON public.espressorl_grinder_aliases (grinder_id);

ALTER TABLE public.espressorl_grinder_catalog ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.espressorl_grinder_aliases ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON public.espressorl_grinder_catalog FROM anon, authenticated;
REVOKE ALL ON public.espressorl_grinder_aliases FROM anon, authenticated;

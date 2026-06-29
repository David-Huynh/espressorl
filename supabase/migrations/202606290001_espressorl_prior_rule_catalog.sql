-- Read-only community catalog of declarative optimizer prior rule packs.
-- Public clients search through a rate-limited Edge Function. Catalog changes
-- are reviewed SQL migrations; no public role can write rule content.

CREATE TABLE IF NOT EXISTS public.espressorl_prior_rule_catalog (
    rule_pack_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    author TEXT,
    description TEXT NOT NULL,
    rules_json JSONB NOT NULL,
    source_url TEXT,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.2,
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (length(rule_pack_id) BETWEEN 1 AND 96),
    CHECK (length(name) BETWEEN 1 AND 120),
    CHECK (length(normalized_name) BETWEEN 1 AND 120),
    CHECK (length(description) BETWEEN 1 AND 500),
    CHECK (jsonb_typeof(rules_json) = 'array'),
    CHECK (jsonb_array_length(rules_json) BETWEEN 1 AND 16),
    CHECK (confidence >= 0.0 AND confidence <= 0.35)
);

CREATE INDEX IF NOT EXISTS idx_espressorl_prior_rule_catalog_normalized_name
    ON public.espressorl_prior_rule_catalog (normalized_name);

ALTER TABLE public.espressorl_prior_rule_catalog ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.espressorl_prior_rule_catalog FROM anon, authenticated;

INSERT INTO public.espressorl_prior_rule_catalog (
    rule_pack_id,
    name,
    normalized_name,
    author,
    description,
    rules_json,
    source_url,
    confidence
) VALUES
(
    'taste-ratio-classic-v1',
    'Taste correction by ratio',
    'taste correction by ratio',
    'EspressoRL community',
    'A low-confidence starting heuristic that shortens bitter shots and extends sour shots.',
    '[
      {"rule_id":"community_bitter_shorter","name":"Bitter: reduce ratio","metric":"taste_tag","operator":"contains","condition_value":"bitter","ratio_direction":"decrease","confidence":0.2,"enabled":true,"source":"community"},
      {"rule_id":"community_sour_longer","name":"Sour: increase ratio","metric":"taste_tag","operator":"contains","condition_value":"sour","ratio_direction":"increase","confidence":0.2,"enabled":true,"source":"community"}
    ]'::jsonb,
    'https://github.com/brokenankle/espresso-rl',
    0.2
),
(
    'taste-grind-classic-v1',
    'Taste correction by grind',
    'taste correction by grind',
    'EspressoRL community',
    'An alternative low-confidence heuristic that adjusts grind instead of ratio.',
    '[
      {"rule_id":"community_bitter_coarser","name":"Bitter: grind coarser","metric":"taste_tag","operator":"contains","condition_value":"bitter","grind_direction":"coarser","confidence":0.2,"enabled":true,"source":"community"},
      {"rule_id":"community_sour_finer","name":"Sour: grind finer","metric":"taste_tag","operator":"contains","condition_value":"sour","grind_direction":"finer","confidence":0.2,"enabled":true,"source":"community"}
    ]'::jsonb,
    'https://github.com/brokenankle/espresso-rl',
    0.2
),
(
    'thirty-second-grind-v1',
    'Thirty second grind heuristic',
    'thirty second grind heuristic',
    'EspressoRL community',
    'A low-confidence traditional timing heuristic for users intentionally targeting about 30 seconds.',
    '[
      {"rule_id":"community_fast_finer","name":"Under 25s: grind finer","metric":"shot_time_s","operator":"lt","condition_value":25,"grind_direction":"finer","confidence":0.15,"enabled":true,"source":"community"},
      {"rule_id":"community_slow_coarser","name":"Over 35s: grind coarser","metric":"shot_time_s","operator":"gt","condition_value":35,"grind_direction":"coarser","confidence":0.15,"enabled":true,"source":"community"}
    ]'::jsonb,
    'https://github.com/brokenankle/espresso-rl',
    0.15
)
ON CONFLICT (rule_pack_id) DO UPDATE SET
    name = EXCLUDED.name,
    normalized_name = EXCLUDED.normalized_name,
    author = EXCLUDED.author,
    description = EXCLUDED.description,
    rules_json = EXCLUDED.rules_json,
    source_url = EXCLUDED.source_url,
    confidence = EXCLUDED.confidence,
    enabled = EXCLUDED.enabled,
    updated_at = now();

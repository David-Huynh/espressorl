-- Starter grinder catalog rows.
--
-- This seed makes live grinder search useful on a fresh Supabase project. The
-- catalog is metadata for matching, display, and local calibration defaults.
-- Optimizer observations and priors must continue to use normalized relative
-- grind values from the active grinder context.

ALTER TABLE public.espressorl_grinder_catalog
    ADD COLUMN IF NOT EXISTS min_steps INTEGER;

DO $$
BEGIN
    ALTER TABLE public.espressorl_grinder_catalog
        ADD CONSTRAINT espressorl_grinder_catalog_min_steps_check
        CHECK (min_steps IS NULL OR min_steps >= 0);
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    ALTER TABLE public.espressorl_grinder_catalog
        ADD CONSTRAINT espressorl_grinder_catalog_step_range_check
        CHECK (min_steps IS NULL OR max_steps IS NULL OR min_steps < max_steps);
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

WITH seed_grinders (
    grinder_id,
    canonical_name,
    manufacturer,
    model,
    microns_per_step,
    min_steps,
    max_steps,
    step_direction,
    source,
    confidence,
    metadata_json
) AS (
    VALUES
        (
            '1zpresso_jx_pro', '1Zpresso JX-Pro', '1Zpresso', 'JX-Pro',
            12.5, 0, 200, 'higher_is_coarser', 'project_seed', 0.85,
            '{"adjustment_unit":"click","source_quality":"official_click_size_user_range","source_urls":["https://1zpresso.coffee/grind-setting/"],"range_notes":"Max step count is a project/user practical range, not treated as optimizer truth."}'::jsonb
        ),
        (
            '1zpresso_j_ultra', '1Zpresso J-Ultra', '1Zpresso', 'J-Ultra',
            8.0, 0, 500, 'higher_is_coarser', 'project_seed', 0.85,
            '{"adjustment_unit":"click","source_quality":"official_click_size","source_urls":["https://1zpresso.coffee/grind-setting/"]}'::jsonb
        ),
        (
            '1zpresso_j_max', '1Zpresso J-Max', '1Zpresso', 'J-Max',
            8.8, 0, 450, 'higher_is_coarser', 'project_seed', 0.8,
            '{"adjustment_unit":"click","source_quality":"official_click_size","source_urls":["https://1zpresso.coffee/grind-setting/"]}'::jsonb
        ),
        (
            '1zpresso_k_ultra', '1Zpresso K-Ultra', '1Zpresso', 'K-Ultra',
            20.0, 0, 200, 'higher_is_coarser', 'project_seed', 0.85,
            '{"adjustment_unit":"click","source_quality":"official_click_size","source_urls":["https://1zpresso.coffee/grind-setting/"],"applies_to":["K-Ultra","K-Plus","K-Pro","K-Max"]}'::jsonb
        ),
        (
            '1zpresso_x_ultra', '1Zpresso X-Ultra', '1Zpresso', 'X-Ultra',
            12.5, 0, 270, 'higher_is_coarser', 'project_seed', 0.8,
            '{"adjustment_unit":"click","source_quality":"official_click_size","source_urls":["https://1zpresso.coffee/grind-setting/"]}'::jsonb
        ),
        (
            '1zpresso_q2_s', '1Zpresso Q2 S', '1Zpresso', 'Q2 S',
            25.0, 0, 90, 'higher_is_coarser', 'project_seed', 0.8,
            '{"adjustment_unit":"click","source_quality":"official_click_size","source_urls":["https://1zpresso.coffee/grind-setting/"]}'::jsonb
        ),
        (
            'baratza_encore_esp', 'Baratza Encore ESP', 'Baratza', 'Encore ESP',
            20.0, 1, 40, 'higher_is_coarser', 'project_seed', 0.75,
            '{"adjustment_model":"piecewise_single_axis","adjustment_unit":"click","source_quality":"manufacturer_engineer_quote","source_urls":["https://www.wired.com/review/baratza-encore-esp/"],"default_range":{"min":1,"max":20,"microns_per_step":20.0},"segments":[{"min":1,"max":20,"microns_per_step":20.0,"output_adjustment_range_microns":400.0,"burr_vertical_range_microns":70.0,"burr_vertical_microns_per_click":3.6},{"min":21,"max":40,"nonlinear":true,"approx_step_range_microns":[60.0,150.0]}],"notes":"One physical grinder context; espresso and coarse sides are metadata segments so history is not fragmented."}'::jsonb
        ),
        (
            'baratza_sette_270', 'Baratza Sette 270', 'Baratza', 'Sette 270',
            NULL, 1, 270, 'higher_is_coarser', 'project_seed', 0.45,
            '{"adjustment_unit":"macro_micro_index","source_quality":"model_name_and_common_index_count","notes":"Macro/micro index count is useful for display, but no catalog micron-per-marker value is seeded."}'::jsonb
        ),
        (
            'df64_gen_2', 'DF64 Gen 2', 'DF64', 'Gen 2',
            10.8, 0, 90, 'higher_is_coarser', 'project_seed', 0.55,
            '{"adjustment_unit":"dial_marker","source_quality":"user_measured","notes":"Treats one dial marker as one discrete optimizer/display step for a stepless grinder."}'::jsonb
        ),
        (
            'eureka_mignon', 'Eureka Mignon', 'Eureka', 'Mignon',
            24.0, 0, NULL, 'higher_is_coarser', 'project_seed', 0.25,
            '{"adjustment_unit":"dial_marker","source_quality":"unverified_marker_scale","notes":"Stepless micrometric grinder. Add marker pitch by PR when measured or sourced."}'::jsonb
        ),
        (
            'fellow_opus', 'Fellow Opus', 'Fellow', 'Opus',
            16.7, NULL, NULL, 'higher_is_coarser', 'project_seed', 0.5,
            '{"adjustment_model":"compound_dual_axis","adjustment_unit":"inner_micro_notch","source_quality":"user_provided_design_note","default_calibration_unit":"inner_micro_notch","primary_axis":{"name":"outer_macro_ring","min":1,"max":41,"burr_carrier_microns_per_step":50.0,"effective_burr_gap_microns_per_step":30.0},"secondary_axis":{"name":"inner_micro_ring","microns_per_step":16.7,"outer_setting_fraction_per_notch":0.6667},"notes":"One physical grinder context; outer macro and inner micro settings combine into one effective relative grind coordinate."}'::jsonb
        ),
        (
            'niche_zero', 'Niche Zero', 'Niche', 'Zero',
            20.0, 0, 50, 'higher_is_coarser', 'project_seed', 0.25,
            '{"adjustment_unit":"dial_marker","source_quality":"unverified_marker_scale","notes":"Stepless grinder. Add marker pitch by PR when measured or sourced."}'::jsonb
        ),
        (
            'timemore_sculptor_064s', 'Timemore Sculptor 064S', 'Timemore', 'Sculptor 064S',
            5, 0, 180, 'higher_is_coarser', 'project_seed', 0.25,
            '{"adjustment_unit":"dial_marker","source_quality":"unverified_marker_scale","notes":"Stepless grinder. Add marker pitch by PR when measured or sourced."}'::jsonb
        )
)
INSERT INTO public.espressorl_grinder_catalog (
    grinder_id,
    canonical_name,
    manufacturer,
    model,
    microns_per_step,
    min_steps,
    max_steps,
    step_direction,
    source,
    confidence,
    metadata_json
)
SELECT
    grinder_id,
    canonical_name,
    manufacturer,
    model,
    microns_per_step,
    min_steps,
    max_steps,
    step_direction,
    source,
    confidence,
    metadata_json
FROM seed_grinders
ON CONFLICT (grinder_id) DO UPDATE SET
    canonical_name = EXCLUDED.canonical_name,
    manufacturer = EXCLUDED.manufacturer,
    model = EXCLUDED.model,
    microns_per_step = EXCLUDED.microns_per_step,
    min_steps = EXCLUDED.min_steps,
    max_steps = EXCLUDED.max_steps,
    step_direction = EXCLUDED.step_direction,
    source = EXCLUDED.source,
    confidence = GREATEST(public.espressorl_grinder_catalog.confidence, EXCLUDED.confidence),
    metadata_json = public.espressorl_grinder_catalog.metadata_json || EXCLUDED.metadata_json,
    updated_at = now();

DELETE FROM public.espressorl_grinder_catalog
WHERE grinder_id IN ('baratza_encore_esp_coarse_range', 'fellow_opus_micro_ring');

WITH seed_aliases (
    grinder_id,
    alias_name,
    normalized_alias,
    source,
    confidence
) AS (
    VALUES
        ('1zpresso_jx_pro', '1Zpresso JX-Pro', '1zpresso jx pro', 'project_seed', 0.85),
        ('1zpresso_jx_pro', 'JX-Pro', 'jx pro', 'project_seed', 0.85),
        ('1zpresso_jx_pro', 'JXPro', 'jxpro', 'project_seed', 0.85),
        ('1zpresso_j_ultra', '1Zpresso J-Ultra', '1zpresso j ultra', 'project_seed', 0.85),
        ('1zpresso_j_ultra', 'J-Ultra', 'j ultra', 'project_seed', 0.85),
        ('1zpresso_j_max', '1Zpresso J-Max', '1zpresso j max', 'project_seed', 0.8),
        ('1zpresso_j_max', 'J-Max', 'j max', 'project_seed', 0.8),
        ('1zpresso_k_ultra', '1Zpresso K-Ultra', '1zpresso k ultra', 'project_seed', 0.85),
        ('1zpresso_k_ultra', 'K-Ultra', 'k ultra', 'project_seed', 0.85),
        ('1zpresso_k_ultra', 'K-Plus', 'k plus', 'project_seed', 0.75),
        ('1zpresso_k_ultra', 'K-Pro', 'k pro', 'project_seed', 0.75),
        ('1zpresso_k_ultra', 'K-Max', 'k max', 'project_seed', 0.75),
        ('1zpresso_x_ultra', '1Zpresso X-Ultra', '1zpresso x ultra', 'project_seed', 0.8),
        ('1zpresso_x_ultra', 'X-Ultra', 'x ultra', 'project_seed', 0.8),
        ('1zpresso_x_ultra', 'X-Pro S', 'x pro s', 'project_seed', 0.75),
        ('1zpresso_q2_s', '1Zpresso Q2 S', '1zpresso q2 s', 'project_seed', 0.8),
        ('1zpresso_q2_s', 'Q2 S', 'q2 s', 'project_seed', 0.8),
        ('1zpresso_q2_s', 'Q Air', 'q air', 'project_seed', 0.75),
        ('baratza_encore_esp', 'Baratza Encore ESP', 'baratza encore esp', 'project_seed', 0.75),
        ('baratza_encore_esp', 'Encore ESP', 'encore esp', 'project_seed', 0.75),
        ('baratza_encore_esp', 'Baratza Encore ESP espresso', 'baratza encore esp espresso', 'project_seed', 0.75),
        ('baratza_encore_esp', 'Encore ESP 1-20', 'encore esp 1 20', 'project_seed', 0.75),
        ('baratza_encore_esp', 'Encore ESP espresso range', 'encore esp espresso range', 'project_seed', 0.75),
        ('baratza_encore_esp', 'Baratza Encore ESP coarse', 'baratza encore esp coarse', 'project_seed', 0.6),
        ('baratza_encore_esp', 'Encore ESP coarse', 'encore esp coarse', 'project_seed', 0.6),
        ('baratza_encore_esp', 'Encore ESP 21-40', 'encore esp 21 40', 'project_seed', 0.6),
        ('baratza_encore_esp', 'Encore ESP filter range', 'encore esp filter range', 'project_seed', 0.6),
        ('baratza_sette_270', 'Baratza Sette 270', 'baratza sette 270', 'project_seed', 0.45),
        ('baratza_sette_270', 'Sette 270', 'sette 270', 'project_seed', 0.45),
        ('df64_gen_2', 'DF64 Gen 2', 'df64 gen 2', 'project_seed', 0.55),
        ('df64_gen_2', 'DF64', 'df64', 'project_seed', 0.55),
        ('df64_gen_2', 'G-IOTA', 'g iota', 'project_seed', 0.35),
        ('eureka_mignon', 'Eureka Mignon', 'eureka mignon', 'project_seed', 0.25),
        ('eureka_mignon', 'Mignon', 'mignon', 'project_seed', 0.25),
        ('eureka_mignon', 'Specialita', 'specialita', 'project_seed', 0.2),
        ('fellow_opus', 'Fellow Opus', 'fellow opus', 'project_seed', 0.5),
        ('fellow_opus', 'Opus', 'opus', 'project_seed', 0.5),
        ('fellow_opus', 'Fellow Opus macro', 'fellow opus macro', 'project_seed', 0.5),
        ('fellow_opus', 'Opus outer ring', 'opus outer ring', 'project_seed', 0.5),
        ('fellow_opus', 'Opus coarse ring', 'opus coarse ring', 'project_seed', 0.5),
        ('fellow_opus', 'Fellow Opus micro', 'fellow opus micro', 'project_seed', 0.5),
        ('fellow_opus', 'Opus micro', 'opus micro', 'project_seed', 0.5),
        ('fellow_opus', 'Opus inner ring', 'opus inner ring', 'project_seed', 0.5),
        ('fellow_opus', 'Opus fine adjustment', 'opus fine adjustment', 'project_seed', 0.5),
        ('niche_zero', 'Niche Zero', 'niche zero', 'project_seed', 0.25),
        ('timemore_sculptor_064s', 'Timemore Sculptor 064S', 'timemore sculptor 064s', 'project_seed', 0.25),
        ('timemore_sculptor_064s', 'Sculptor 064S', 'sculptor 064s', 'project_seed', 0.25),
        ('timemore_sculptor_064s', '064S', '064s', 'project_seed', 0.25)
)
INSERT INTO public.espressorl_grinder_aliases (
    grinder_id,
    alias_name,
    normalized_alias,
    source,
    confidence
)
SELECT
    grinder_id,
    alias_name,
    normalized_alias,
    source,
    confidence
FROM seed_aliases
ON CONFLICT (normalized_alias) DO UPDATE SET
    grinder_id = EXCLUDED.grinder_id,
    alias_name = EXCLUDED.alias_name,
    source = EXCLUDED.source,
    confidence = GREATEST(public.espressorl_grinder_aliases.confidence, EXCLUDED.confidence);

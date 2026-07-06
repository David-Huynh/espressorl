from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase" / "migrations" / "202605290001_espresso_rl_raw_queue.sql"
FUNCTION = ROOT / "supabase" / "functions" / "espresso-rl-ingest" / "index.ts"
GRINDER_SEARCH_FUNCTION = (
    ROOT / "supabase" / "functions" / "espresso-rl-grinder-search" / "index.ts"
)
CONFIG = ROOT / "supabase" / "config.toml"
RATE_LIMIT_MIGRATION = (
    ROOT / "supabase" / "migrations" / "202605310001_espressorl_rate_limit_no_overcount.sql"
)
GRINDER_CATALOG_MIGRATION = (
    ROOT / "supabase" / "migrations" / "202606250001_espressorl_grinder_catalog.sql"
)
GRINDER_CATALOG_SEED_MIGRATION = (
    ROOT / "supabase" / "migrations" / "202606270001_espressorl_grinder_catalog_seed.sql"
)
PRIOR_RULE_CATALOG_MIGRATION = (
    ROOT / "supabase" / "migrations" / "202606290001_espressorl_prior_rule_catalog.sql"
)
PRIOR_RULE_SEARCH_FUNCTION = (
    ROOT / "supabase" / "functions" / "espresso-rl-prior-rule-search" / "index.ts"
)


class SupabaseIngestionContractTests(unittest.TestCase):
    def test_ingest_function_disables_platform_jwt_check_for_hmac_uploads(self) -> None:
        config = CONFIG.read_text()

        self.assertIn("[functions.espresso-rl-ingest]", config)
        self.assertIn("verify_jwt = false", config)

    def test_grinder_search_function_disables_jwt_and_exposes_only_bounded_search(self) -> None:
        config = CONFIG.read_text()
        source = GRINDER_SEARCH_FUNCTION.read_text()

        self.assertIn("[functions.espresso-rl-grinder-search]", config)
        self.assertIn("verify_jwt = false", config)
        self.assertIn("request.method !== 'GET'", source)
        self.assertIn("MAX_QUERY_LENGTH = 80", source)
        self.assertIn("MAX_LIMIT = 10", source)
        self.assertIn("espressorl_grinder_aliases", source)
        self.assertIn("espressorl_grinder_catalog", source)
        self.assertIn("min_steps", source)
        self.assertIn("metadata_json", source)
        self.assertIn("espressorl_consume_rate_limit", source)
        self.assertIn("'Access-Control-Allow-Origin': '*'", source)

    def test_prior_rule_catalog_is_read_only_bounded_and_rate_limited(self) -> None:
        config = CONFIG.read_text()
        source = PRIOR_RULE_SEARCH_FUNCTION.read_text()
        sql = PRIOR_RULE_CATALOG_MIGRATION.read_text()

        self.assertIn("[functions.espresso-rl-prior-rule-search]", config)
        self.assertIn("request.method !== 'GET'", source)
        self.assertIn("MAX_QUERY_LENGTH = 80", source)
        self.assertIn("MAX_LIMIT = 10", source)
        self.assertIn("espressorl_consume_rate_limit", source)
        self.assertIn("espressorl_prior_rule_catalog", source)
        self.assertIn("Math.min(0.35", source)
        self.assertIn("CREATE TABLE IF NOT EXISTS public.espressorl_prior_rule_catalog", sql)
        self.assertIn("jsonb_array_length(rules_json) BETWEEN 1 AND 16", sql)
        self.assertIn("confidence >= 0.0 AND confidence <= 0.35", sql)
        self.assertIn("ENABLE ROW LEVEL SECURITY", sql)
        self.assertIn("REVOKE ALL ON public.espressorl_prior_rule_catalog FROM anon, authenticated", sql)
        self.assertIn("taste-ratio-classic-v1", sql)
        self.assertIn("taste-grind-classic-v1", sql)
        self.assertIn("thirty-second-grind-v1", sql)
        self.assertIn('"ratio_direction":"decrease"', sql)
        self.assertIn('"grind_direction":"finer"', sql)
        self.assertNotIn("target_ratio_delta", sql)
        self.assertNotIn("grind_delta_steps", sql)

    def test_raw_queue_migration_blocks_public_table_access(self) -> None:
        sql = MIGRATION.read_text()

        self.assertIn("CREATE TABLE IF NOT EXISTS public.raw_upload_queue", sql)
        self.assertIn("PRIMARY KEY (install_id, upload_id)", sql)
        self.assertIn("UNIQUE (install_id, payload_hash)", sql)
        self.assertIn("CHECK (status IN ('queued', 'mirroring', 'mirrored', 'mirror_failed', 'rejected'))", sql)
        self.assertIn("mirror_claimed_by TEXT", sql)
        self.assertIn("mirror_claim_expires_at TIMESTAMPTZ", sql)
        self.assertIn("mirror_completed_at TIMESTAMPTZ", sql)
        self.assertIn("ALTER TABLE public.raw_upload_queue ENABLE ROW LEVEL SECURITY", sql)
        self.assertIn("REVOKE ALL ON public.raw_upload_queue FROM anon, authenticated", sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS public.espressorl_upload_credentials", sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS public.espressorl_abuse_events", sql)
        self.assertIn("espressorl_consume_rate_limit", sql)
        self.assertIn("espressorl_claim_raw_uploads", sql)
        self.assertIn("FOR UPDATE SKIP LOCKED", sql)
        self.assertIn("espressorl_purge_raw_upload_queue", sql)
        self.assertIn("q.status = 'mirrored'", sql)

    def test_edge_function_verifies_signed_upload_before_queue_insert(self) -> None:
        source = FUNCTION.read_text()

        for required_header in (
            "x-espressorl-install-id",
            "x-espressorl-upload-id",
            "x-espressorl-timestamp",
            "x-espressorl-signature",
            "x-espressorl-payload-hash",
        ):
            self.assertIn(required_header, source)

        self.assertIn("lookupCredential", source)
        self.assertIn("hmacSha256Hex", source)
        self.assertIn("constantTimeEqual", source)
        self.assertIn("payload hash mismatch", source)
        self.assertIn("timestamp out of range", source)
        self.assertIn("payload install_id does not match upload credential", source)
        self.assertIn("request.headers.get('content-length')", source)
        self.assertIn("SHA256_HEX.test(payloadHash)", source)
        self.assertIn("sanitizedPayload.install_id = installId", source)
        self.assertIn("consumeRateLimits", source)
        self.assertIn(".from('raw_upload_queue').insert", source)
        self.assertIn(".from('raw_upload_queue').upsert", source)
        self.assertIn("onConflict: 'install_id,upload_id'", source)
        self.assertIn("payload_json: sanitizedPayload", source)
        self.assertNotIn(".from('validated_shots').insert", source)
        self.assertNotIn(".from('community_priors').insert", source)
        self.assertNotIn(".from('training_dataset').insert", source)

    def test_edge_function_validates_physical_espresso_bounds(self) -> None:
        source = FUNCTION.read_text()

        self.assertIn("validateShotRecord", source)
        self.assertIn("validateRecommendationRecord", source)
        self.assertIn("SUPPORTED_SCHEMA_VERSION = 1", source)
        self.assertIn("rejectUnknownFields(payload, allowedShotFields, errors)", source)
        self.assertIn("rejectUnknownFields(payload, allowedRecommendationFields, errors)", source)
        self.assertIn("requireNumberRange(payload, 'schema_version', SUPPORTED_SCHEMA_VERSION", source)
        self.assertIn("requireNumberRange(payload, 'dose_in_g', 5, 30", source)
        self.assertIn("optionalNumberRange(payload, 'beverage_out_g', 0, 120", source)
        self.assertIn("requireNumberRange(payload, 'target_yield_g', 5, 100", source)
        self.assertIn("optionalNumberRange(payload, 'shot_time_s', 0, 180", source)
        self.assertIn("requireNumberRange(payload, 'target_ratio', 1.2, 3.5", source)
        self.assertIn("requireNumberRange(payload, 'profile_temperature_c', 0, 160", source)
        self.assertIn("requireNumberRange(payload, 'final_phase_temperature_c', 0, 160", source)
        self.assertIn("optionalNumericProfileVector(payload, 'beverage_flow_profile', 0, 20", source)
        self.assertIn("optionalNumericProfileVector(payload, 'temperature_profile', 0, 160", source)
        self.assertIn("optionalNumericProfileVector(payload, 'target_temperature_profile', 0, 160", source)
        self.assertIn("optionalPumpTargetModeProfile(payload, 'pump_target_mode_profile'", source)
        self.assertIn("optionalFixedCadenceSequence(payload.fixed_cadence_sequence", source)
        self.assertIn("fixed_cadence_sequence.sample_interval_ms must be 250", source)
        self.assertIn("fixed_cadence_sequence channels must have matching lengths", source)
        self.assertIn("optionalEnum(payload, 'shot_type', ['espresso', 'utility_flush', 'cleaning', 'calibration', 'unknown']", source)
        self.assertIn("optionalBoolean(payload, 'exclude_from_local_optimization'", source)
        self.assertIn("optionalBoolean(payload, 'feedback_recorded'", source)
        self.assertIn("'action_observed'", source)
        self.assertIn("optionalActionObserved(payload, errors)", source)
        self.assertIn("action_observed.grind cannot be true without a grind measurement", source)
        self.assertIn("'grinder_adjustment_mode'", source)
        self.assertGreaterEqual(
            source.count("optionalEnum(payload, 'grinder_adjustment_mode', ['stepped', 'stepless']"),
            2,
        )
        self.assertIn("optionalString(payload, 'profile_id', 120", source)
        self.assertIn("optionalSha256(payload, 'raw_profile_hash'", source)
        self.assertIn("optionalNumberRange(payload, 'optimization_weight', 0, 1", source)
        self.assertIn("[0, 20, 'pressure']", source)
        self.assertIn("[0, 15, 'target_pressure']", source)
        self.assertIn("[-1, 120, 'weight']", source)
        self.assertIn("profile_resampled must have 5 channels", source)
        self.assertIn("profile_resampled ${label} channel must have exactly 100 samples", source)
        self.assertNotIn("final profile weight does not match beverage_out_g", source)

    def test_edge_function_dedups_before_consuming_rate_limit(self) -> None:
        source = FUNCTION.read_text()

        self.assertIn("uploadAlreadyQueued", source)
        self.assertIn("status: 'duplicate'", source)
        self.assertIn(".eq('payload_hash', payloadHash)", source)
        # A re-send must be detected before any rate-limit budget is spent.
        self.assertLess(
            source.index("uploadAlreadyQueued("),
            source.index("consumeRateLimits("),
        )

    def test_recommendation_uploads_update_one_remote_raw_row(self) -> None:
        source = FUNCTION.read_text()

        self.assertIn("validation.localRecordType === 'recommendation'", source)
        self.assertIn("mirror_claimed_by: null", source)
        self.assertIn("mirror_completed_at: null", source)
        self.assertIn("mirror_attempt_count: 0", source)
        self.assertIn("status: 'queued'", source)

    def test_edge_function_returns_retry_after_and_raised_limits(self) -> None:
        source = FUNCTION.read_text()

        self.assertIn("'Retry-After'", source)
        self.assertIn("secondsToNextUtcDay", source)
        self.assertIn("INSTALL_DAY_LIMIT = 500", source)
        self.assertIn("INSTALL_MINUTE_LIMIT = 30", source)

    def test_rate_limit_migration_counts_only_accepted_uploads(self) -> None:
        sql = RATE_LIMIT_MIGRATION.read_text()

        self.assertIn("CREATE OR REPLACE FUNCTION public.espressorl_consume_rate_limit", sql)
        # Increment only while under the limit; deny without counting once at/over it.
        self.assertIn("WHERE public.espressorl_ingest_rate_counters.count < p_limit", sql)
        self.assertIn("IF NOT FOUND THEN", sql)
        self.assertIn("RETURN FALSE", sql)

    def test_grinder_catalog_migration_defines_locked_down_metadata_tables(self) -> None:
        sql = GRINDER_CATALOG_MIGRATION.read_text()

        self.assertIn("CREATE TABLE IF NOT EXISTS public.espressorl_grinder_catalog", sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS public.espressorl_grinder_aliases", sql)
        self.assertIn("microns_per_step DOUBLE PRECISION", sql)
        self.assertIn("min_steps INTEGER", sql)
        self.assertIn("max_steps INTEGER", sql)
        self.assertIn("normalized_alias TEXT NOT NULL", sql)
        self.assertIn("ALTER TABLE public.espressorl_grinder_catalog ENABLE ROW LEVEL SECURITY", sql)
        self.assertIn("ALTER TABLE public.espressorl_grinder_aliases ENABLE ROW LEVEL SECURITY", sql)
        self.assertIn("REVOKE ALL ON public.espressorl_grinder_catalog FROM anon, authenticated", sql)
        self.assertIn("REVOKE ALL ON public.espressorl_grinder_aliases FROM anon, authenticated", sql)

    def test_grinder_catalog_seed_populates_practical_step_metadata(self) -> None:
        sql = GRINDER_CATALOG_SEED_MIGRATION.read_text()

        self.assertIn("ADD COLUMN IF NOT EXISTS min_steps", sql)
        self.assertIn("'1zpresso_jx_pro'", sql)
        self.assertIn("12.5, 0, 200, 'higher_is_coarser'", sql)
        self.assertIn("'df64_gen_2'", sql)
        self.assertIn("10.8, 0, 90, 'higher_is_coarser'", sql)
        self.assertIn('"adjustment_model":"piecewise_single_axis"', sql)
        self.assertIn("20.0, 1, 40, 'higher_is_coarser'", sql)
        self.assertIn('"default_range":{"min":1,"max":20,"microns_per_step":20.0}', sql)
        self.assertIn('"adjustment_model":"compound_dual_axis"', sql)
        self.assertIn('"primary_axis":{"name":"outer_macro_ring","min":1,"max":41', sql)
        self.assertIn("16.7, NULL, NULL, 'higher_is_coarser'", sql)
        self.assertIn('"adjustment_unit":"dial_marker"', sql)
        self.assertIn('"source_quality":"user_measured"', sql)
        self.assertIn("('baratza_encore_esp', 'Encore ESP coarse'", sql)
        self.assertIn("('fellow_opus', 'Opus micro'", sql)

    def test_grinder_catalog_seed_has_unique_normalized_aliases(self) -> None:
        sql = GRINDER_CATALOG_SEED_MIGRATION.read_text()
        alias_values = sql.split("WITH seed_aliases", 1)[1].split(")\nINSERT INTO", 1)[0]
        aliases: list[str] = []
        for line in alias_values.splitlines():
            stripped = line.strip()
            if not stripped.startswith("('"):
                continue
            parts = stripped.split("'")
            if len(parts) >= 6:
                aliases.append(parts[5])

        self.assertEqual(len(aliases), len(set(aliases)))


if __name__ == "__main__":
    unittest.main()

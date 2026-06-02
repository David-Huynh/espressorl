from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase" / "migrations" / "202605290001_espresso_rl_raw_queue.sql"
FUNCTION = ROOT / "supabase" / "functions" / "espresso-rl-ingest" / "index.ts"
CONFIG = ROOT / "supabase" / "config.toml"
RATE_LIMIT_MIGRATION = (
    ROOT / "supabase" / "migrations" / "202605310001_espressorl_rate_limit_no_overcount.sql"
)


class SupabaseIngestionContractTests(unittest.TestCase):
    def test_ingest_function_disables_platform_jwt_check_for_hmac_uploads(self) -> None:
        config = CONFIG.read_text()

        self.assertIn("[functions.espresso-rl-ingest]", config)
        self.assertIn("verify_jwt = false", config)

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
        self.assertIn("payload.install_id = installId", source)
        self.assertIn("consumeRateLimits", source)
        self.assertIn(".from('raw_upload_queue').insert", source)
        self.assertNotIn(".from('validated_shots').insert", source)
        self.assertNotIn(".from('community_priors').insert", source)
        self.assertNotIn(".from('training_dataset').insert", source)

    def test_edge_function_validates_physical_espresso_bounds(self) -> None:
        source = FUNCTION.read_text()

        self.assertIn("validateShotRecord", source)
        self.assertIn("validateRecommendationRecord", source)
        self.assertIn("requireNumberRange(payload, 'dose_in_g', 5, 30", source)
        self.assertIn("optionalNumberRange(payload, 'beverage_out_g', 5, 100", source)
        self.assertIn("requireNumberRange(payload, 'target_yield_g', 5, 100", source)
        self.assertIn("optionalNumberRange(payload, 'shot_time_s', 5, 90", source)
        self.assertIn("requireNumberRange(payload, 'target_ratio', 1.2, 3.5", source)
        self.assertIn("optionalEnum(payload, 'shot_type', ['espresso', 'utility_flush', 'cleaning', 'calibration', 'unknown']", source)
        self.assertIn("optionalBoolean(payload, 'exclude_from_local_optimization'", source)
        self.assertIn("optionalNumberRange(payload, 'optimization_weight', 0, 1", source)
        self.assertIn("profile_resampled must have 5 channels", source)
        self.assertIn("profile_resampled ${label} channel must have exactly 100 samples", source)
        self.assertIn("final profile weight does not match beverage_out_g", source)

    def test_edge_function_dedups_before_consuming_rate_limit(self) -> None:
        source = FUNCTION.read_text()

        self.assertIn("uploadAlreadyQueued", source)
        self.assertIn("status: 'duplicate'", source)
        # A re-send must be detected before any rate-limit budget is spent.
        self.assertLess(
            source.index("uploadAlreadyQueued("),
            source.index("consumeRateLimits("),
        )

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


if __name__ == "__main__":
    unittest.main()

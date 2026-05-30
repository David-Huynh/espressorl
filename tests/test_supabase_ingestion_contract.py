from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase" / "migrations" / "202605290001_espresso_rl_raw_queue.sql"
FUNCTION = ROOT / "supabase" / "functions" / "espresso-rl-ingest" / "index.ts"


class SupabaseIngestionContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

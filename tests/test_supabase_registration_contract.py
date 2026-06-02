from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTER_FUNCTION = ROOT / "supabase" / "functions" / "espresso-rl-register" / "index.ts"
CONFIG = ROOT / "supabase" / "config.toml"


class SupabaseRegistrationContractTests(unittest.TestCase):
    def test_registration_function_disables_platform_jwt_check_for_hmac_actions(self) -> None:
        config = CONFIG.read_text()

        self.assertIn("[functions.espresso-rl-register]", config)
        self.assertIn("verify_jwt = false", config)

    def test_registration_function_issues_rotates_and_revokes_credentials(self) -> None:
        source = REGISTER_FUNCTION.read_text()

        self.assertIn("Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')", source)
        self.assertIn("action === 'register'", source)
        self.assertIn("action === 'rotate'", source)
        self.assertIn("action === 'revoke'", source)
        self.assertIn(".from('espressorl_upload_credentials').insert", source)
        self.assertIn(".update({ revoked_at:", source)
        self.assertIn("community_upload_enabled: false", source)
        self.assertIn("crypto.randomUUID()", source)
        self.assertIn("randomHex(32)", source)

    def test_rotation_and_revoke_require_signed_existing_credentials(self) -> None:
        source = REGISTER_FUNCTION.read_text()

        self.assertIn("requireSignedCredential", source)
        self.assertIn("x-espressorl-install-id", source)
        self.assertIn("x-espressorl-token-id", source)
        self.assertIn("x-espressorl-signature", source)
        self.assertIn("hmacSha256Hex", source)
        self.assertIn("constantTimeEqual(signature, expectedSignature)", source)
        self.assertIn("unknown or revoked upload credential", source)

    def test_registration_is_rate_limited_by_ip(self) -> None:
        source = REGISTER_FUNCTION.read_text()

        self.assertIn("consumeRegistrationRateLimits", source)
        self.assertIn("registration-ip:", source)
        self.assertIn("espressorl_consume_rate_limit", source)


if __name__ == "__main__":
    unittest.main()

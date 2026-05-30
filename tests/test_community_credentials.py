from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from espresso_rl.adapters.supabase_credentials import (
    HttpResponse,
    JsonCommunityCredentialStore,
    SupabaseCredentialRegistrar,
    SupabaseCredentialRegistrarConfig,
)
from espresso_rl.application.community_credentials import CommunityCredentialService
from espresso_rl.config import Config
from espresso_rl.domain.community import CommunityUploadCredentials
from espresso_rl.main import maybe_resolve_community_upload_credentials


class CommunityCredentialTests(unittest.TestCase):
    def test_service_prefers_configured_credentials_without_registering(self) -> None:
        store = FakeCredentialStore()
        registrar = FakeCredentialRegistrar()
        configured = CommunityUploadCredentials(
            install_id="configured_install",
            upload_token_id="configured_token",
            upload_secret="c" * 32,
        )

        resolved = CommunityCredentialService(store, registrar).resolve_for_upload(configured)

        self.assertEqual(resolved, configured)
        self.assertEqual(registrar.registered, 0)
        self.assertIsNone(store.saved)

    def test_service_registers_and_stores_when_no_credentials_exist(self) -> None:
        store = FakeCredentialStore()
        registrar = FakeCredentialRegistrar()

        resolved = CommunityCredentialService(store, registrar).resolve_for_upload()

        self.assertEqual(resolved.install_id, "registered_install")  # type: ignore[union-attr]
        self.assertEqual(registrar.registered, 1)
        self.assertEqual(store.saved, resolved)

    def test_json_credential_store_round_trips_without_logging_secret(self) -> None:
        credentials = CommunityUploadCredentials(
            install_id="install_1",
            upload_token_id="token_1",
            upload_secret="s" * 32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonCommunityCredentialStore(Path(tmp) / "credentials.json")
            store.save(credentials)

            self.assertEqual(store.load(), credentials)
            mode = (Path(tmp) / "credentials.json").stat().st_mode & 0o777
            self.assertEqual(mode, 0o600)

    def test_supabase_registrar_registers_rotates_and_revokes(self) -> None:
        calls = []

        def transport(req, timeout_s):
            calls.append(req)
            if len(calls) == 1:
                self.assertNotIn("x-espressorl-signature", {k.lower(): v for k, v in req.headers.items()})
                return HttpResponse(
                    201,
                    json.dumps(
                        {
                            "install_id": "install_1",
                            "upload_token_id": "token_1",
                            "upload_secret": "s" * 32,
                        }
                    ),
                )
            if len(calls) == 2:
                headers = {k.lower(): v for k, v in req.headers.items()}
                self.assertEqual(headers["x-espressorl-install-id"], "install_1")
                self.assertEqual(headers["x-espressorl-token-id"], "token_1")
                self.assertIn("x-espressorl-signature", headers)
                return HttpResponse(
                    200,
                    json.dumps(
                        {
                            "install_id": "install_1",
                            "upload_token_id": "token_2",
                            "upload_secret": "r" * 32,
                        }
                    ),
                )
            return HttpResponse(200, json.dumps({"status": "revoked"}))

        registrar = SupabaseCredentialRegistrar(
            SupabaseCredentialRegistrarConfig("https://example.invalid/functions/v1/espresso-rl-register"),
            transport=transport,
            clock=lambda: 123,
        )

        registered = registrar.register_install()
        rotated = registrar.rotate_credentials(registered)
        registrar.revoke_credentials(rotated)

        self.assertEqual(registered.install_id, "install_1")
        self.assertEqual(rotated.upload_token_id, "token_2")
        self.assertEqual(len(calls), 3)
        self.assertEqual(json.loads(calls[0].data.decode("utf-8")), {"action": "register"})  # type: ignore[union-attr]
        self.assertEqual(json.loads(calls[1].data.decode("utf-8")), {"action": "rotate"})  # type: ignore[union-attr]
        self.assertEqual(json.loads(calls[2].data.decode("utf-8")), {"action": "revoke"})  # type: ignore[union-attr]

    def test_public_config_resolves_stored_credentials_before_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonCommunityCredentialStore(Path(tmp) / "credentials.json")
            store.save(
                CommunityUploadCredentials(
                    install_id="stored_install",
                    upload_token_id="stored_token",
                    upload_secret="s" * 32,
                )
            )
            config = Config(
                mqtt_host="localhost",
                community_upload_enabled=True,
                data_dir=Path(tmp),
            )

            resolved = maybe_resolve_community_upload_credentials(config, store=store)

        self.assertEqual(resolved.install_id, "stored_install")  # type: ignore[union-attr]
        self.assertEqual(config.install_id, "stored_install")
        self.assertEqual(config.upload_token_id, "stored_token")
        self.assertEqual(config.upload_secret, "s" * 32)

    def test_admin_config_never_registers_credentials(self) -> None:
        registrar = FakeCredentialRegistrar()
        config = Config(
            mqtt_host="localhost",
            community_upload_enabled=True,
            deployment_role="admin",
            supabase_registration_url="https://example.invalid/functions/v1/espresso-rl-register",
        )

        resolved = maybe_resolve_community_upload_credentials(
            config,
            store=FakeCredentialStore(),
            registrar=registrar,
        )

        self.assertIsNone(resolved)
        self.assertEqual(registrar.registered, 0)

    def test_registration_failure_does_not_block_local_startup(self) -> None:
        config = Config(
            mqtt_host="localhost",
            community_upload_enabled=True,
            supabase_registration_url="https://example.invalid/functions/v1/espresso-rl-register",
        )

        resolved = maybe_resolve_community_upload_credentials(
            config,
            store=FakeCredentialStore(),
            registrar=FailingCredentialRegistrar(),
        )

        self.assertIsNone(resolved)
        self.assertEqual(config.upload_secret, "")


class FakeCredentialStore:
    def __init__(self) -> None:
        self.saved: CommunityUploadCredentials | None = None

    def load(self) -> CommunityUploadCredentials | None:
        return self.saved

    def save(self, credentials: CommunityUploadCredentials) -> None:
        self.saved = credentials

    def clear(self) -> None:
        self.saved = None


class FakeCredentialRegistrar:
    def __init__(self) -> None:
        self.registered = 0
        self.rotated = 0
        self.revoked = 0

    def register_install(self) -> CommunityUploadCredentials:
        self.registered += 1
        return CommunityUploadCredentials(
            install_id="registered_install",
            upload_token_id="registered_token",
            upload_secret="r" * 32,
        )

    def rotate_credentials(self, current: CommunityUploadCredentials) -> CommunityUploadCredentials:
        self.rotated += 1
        return CommunityUploadCredentials(
            install_id=current.install_id,
            upload_token_id="rotated_token",
            upload_secret="n" * 32,
        )

    def revoke_credentials(self, current: CommunityUploadCredentials) -> None:
        self.revoked += 1


class FailingCredentialRegistrar:
    def register_install(self) -> CommunityUploadCredentials:
        raise RuntimeError("registration failed")

    def rotate_credentials(self, current: CommunityUploadCredentials) -> CommunityUploadCredentials:
        raise RuntimeError("rotation failed")

    def revoke_credentials(self, current: CommunityUploadCredentials) -> None:
        raise RuntimeError("revoke failed")


if __name__ == "__main__":
    unittest.main()

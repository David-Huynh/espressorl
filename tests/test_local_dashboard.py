from __future__ import annotations

import importlib.util
import unittest

from espresso_rl.application.local_data import LocalDataStatus


@unittest.skipUnless(
    importlib.util.find_spec("fastapi")
    and (importlib.util.find_spec("httpx2") or importlib.util.find_spec("httpx")),
    "FastAPI TestClient dependencies are not installed in the local test env",
)
class LocalDashboardTests(unittest.TestCase):
    def test_status_requires_bearer_token_and_does_not_render_secret_fields(self) -> None:
        from fastapi.testclient import TestClient

        from espresso_rl.adapters.local_dashboard import create_local_dashboard_app

        client = TestClient(create_local_dashboard_app(FakeLocalService(), FakeUploadMaintenance(), "a" * 32))

        self.assertEqual(client.get("/api/status").status_code, 401)
        response = client.get("/api/status", headers={"Authorization": f"Bearer {'a' * 32}"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertNotIn("local_dashboard_token", body)
        self.assertNotIn("service_role_key", body)
        self.assertNotIn("upload_secret", body)
        self.assertNotIn("profile_resampled", body)


class FakeLocalService:
    def status(self, limit: int = 500) -> LocalDataStatus:
        return LocalDataStatus(
            install_id="install_1",
            machine_id="machine_1",
            contexts=[],
            recent_shots=[],
        )


class FakeUploadMaintenance:
    pass


if __name__ == "__main__":
    unittest.main()

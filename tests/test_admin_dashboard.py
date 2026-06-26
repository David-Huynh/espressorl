from __future__ import annotations

import importlib.util
import unittest

from espresso_rl.application.admin_pipeline import AdminPipelineStatus


@unittest.skipUnless(
    importlib.util.find_spec("fastapi")
    and (importlib.util.find_spec("httpx2") or importlib.util.find_spec("httpx")),
    "FastAPI TestClient dependencies are not installed in the local test env",
)
class AdminDashboardTests(unittest.TestCase):
    def test_status_requires_bearer_token_and_does_not_render_secret_fields(self) -> None:
        from fastapi.testclient import TestClient

        from espresso_rl.adapters.admin_dashboard import create_admin_dashboard_app

        client = TestClient(
            create_admin_dashboard_app(FakeAdminService(), "a" * 32)
        )

        self.assertEqual(client.get("/api/status").status_code, 401)
        response = client.get("/api/status", headers={"Authorization": f"Bearer {'a' * 32}"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertNotIn("admin_dashboard_token", body)
        self.assertNotIn("service_role_key", body)
        self.assertNotIn("upload_secret", body)
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertIn("frame-ancestors 'none'", response.headers["content-security-policy"])

    def test_dashboard_rejects_oversized_request_body_before_action_dispatch(self) -> None:
        from fastapi.testclient import TestClient

        from espresso_rl.adapters.admin_dashboard import create_admin_dashboard_app

        client = TestClient(
            create_admin_dashboard_app(FakeAdminService(), "a" * 32)
        )

        response = client.post(
            "/api/validation/run",
            headers={"Authorization": f"Bearer {'a' * 32}"},
            content='{"padding":"' + ("x" * 70_000) + '"}',
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")


class FakeAdminService:
    def status(self) -> AdminPipelineStatus:
        return AdminPipelineStatus(
            raw_upload_counts={},
            local_raw_upload_purge_eligible_counts={},
            validated_shot_count=0,
            training_row_count=0,
            community_prior_count=0,
            abuse_event_count=0,
            latest_rejections=[],
            latest_admin_actions=[],
            mirror_enabled=False,
        )


if __name__ == "__main__":
    unittest.main()

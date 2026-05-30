from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable
from urllib import error, parse, request

from espresso_rl.domain.community import CommunityRawUpload


@dataclass(frozen=True)
class SupabaseCommunityQueueConfig:
    rest_url: str
    service_role_key: str
    admin_id: str = "espresso-rl-admin"
    raw_queue_table: str = "raw_upload_queue"
    claim_lease_seconds: int = 300
    timeout_s: float = 10.0


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: str


HttpTransport = Callable[[request.Request, float], HttpResponse]


class SupabaseCommunityQueueClient:
    """
    Admin-only adapter for polling the community-fed Supabase raw queue.

    This adapter must use a service-role key and must not be used by public
    clients. It mirrors raw queue rows into local Postgres for validation and
    training jobs.
    """

    def __init__(
        self,
        config: SupabaseCommunityQueueConfig,
        transport: HttpTransport | None = None,
    ) -> None:
        if not config.rest_url:
            raise ValueError("supabase rest_url is required")
        if not config.service_role_key:
            raise ValueError("supabase service_role_key is required")
        self._config = config
        self._transport = transport or _default_transport

    def claim_batch(self, limit: int = 100) -> list[CommunityRawUpload]:
        rows = self._request_json(
            "POST",
            self._rpc_url("espressorl_claim_raw_uploads"),
            {
                "p_claimed_by": self._config.admin_id,
                "p_limit": limit,
                "p_lease_seconds": self._config.claim_lease_seconds,
            },
        )
        return [_upload_from_row(row) for row in rows]

    def mark_mirrored(self, upload: CommunityRawUpload) -> None:
        self._patch_status(
            upload,
            "mirroring",
            {
                "status": "mirrored",
                "mirror_error": None,
                "mirror_completed_at": _utc_now(),
            },
        )

    def mark_failed(self, upload: CommunityRawUpload, error_message: str) -> None:
        self._patch_status(
            upload,
            "mirroring",
            {
                "status": "mirror_failed",
                "mirror_error": error_message[:500],
                "mirror_completed_at": _utc_now(),
            },
        )

    def _patch_status(
        self,
        upload: CommunityRawUpload,
        expected_status: str,
        patch: dict,
    ) -> bool:
        rows = self._request_json(
            "PATCH",
            self._table_url(
                {
                    "install_id": f"eq.{upload.install_id}",
                    "upload_id": f"eq.{upload.upload_id}",
                    "status": f"eq.{expected_status}",
                    "mirror_claimed_by": f"eq.{self._config.admin_id}",
                    "select": "install_id,upload_id",
                }
            ),
            patch,
            prefer="return=representation",
        )
        return bool(rows)

    def _request_json(
        self,
        method: str,
        url: str,
        payload: dict | None = None,
        prefer: str | None = None,
    ):
        headers = {
            "apikey": self._config.service_role_key,
            "Authorization": f"Bearer {self._config.service_role_key}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        req = request.Request(url, data=body, headers=headers, method=method)
        response = self._transport(req, self._config.timeout_s)
        if response.status >= 400:
            raise RuntimeError(f"supabase queue request failed with HTTP {response.status}: {response.body[:500]}")
        if not response.body:
            return []
        return json.loads(response.body)

    def _table_url(self, params: dict[str, str]) -> str:
        base = self._config.rest_url.rstrip("/")
        table = parse.quote(self._config.raw_queue_table)
        query = parse.urlencode(params)
        return f"{base}/{table}?{query}"

    def _rpc_url(self, function_name: str) -> str:
        base = self._config.rest_url.rstrip("/")
        return f"{base}/rpc/{parse.quote(function_name)}"


def _default_transport(req: request.Request, timeout_s: float) -> HttpResponse:
    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            return HttpResponse(
                status=response.status,
                body=response.read().decode("utf-8"),
            )
    except error.HTTPError as exc:
        return HttpResponse(
            status=exc.code,
            body=exc.read().decode("utf-8"),
        )


def _upload_from_row(row: dict) -> CommunityRawUpload:
    return CommunityRawUpload(
        install_id=str(row.get("install_id") or ""),
        upload_id=str(row.get("upload_id") or ""),
        payload_hash=str(row.get("payload_hash") or ""),
        event_type=str(row.get("event_type") or ""),
        payload_json=dict(row.get("payload_json") or {}),
        received_at=row.get("received_at"),
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()

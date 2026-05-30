from __future__ import annotations

import hmac
import json
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable
from urllib import error, request

from espresso_rl.domain.community import CommunityUploadCredentials


@dataclass(frozen=True)
class SupabaseCredentialRegistrarConfig:
    registration_url: str
    timeout_s: float = 10.0


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: str


HttpTransport = Callable[[request.Request, float], HttpResponse]


class SupabaseCredentialRegistrar:
    def __init__(
        self,
        config: SupabaseCredentialRegistrarConfig,
        transport: HttpTransport | None = None,
        clock: Callable[[], int] = lambda: int(time.time()),
    ) -> None:
        if not config.registration_url:
            raise ValueError("registration_url is required")
        self._config = config
        self._transport = transport or _default_transport
        self._clock = clock

    def register_install(self) -> CommunityUploadCredentials:
        return self._request_credentials({"action": "register"})

    def rotate_credentials(self, current: CommunityUploadCredentials) -> CommunityUploadCredentials:
        return self._request_credentials({"action": "rotate"}, current=current)

    def revoke_credentials(self, current: CommunityUploadCredentials) -> None:
        self._request_json({"action": "revoke"}, current=current)

    def _request_credentials(
        self,
        payload: dict,
        current: CommunityUploadCredentials | None = None,
    ) -> CommunityUploadCredentials:
        data = self._request_json(payload, current=current)
        return CommunityUploadCredentials(
            install_id=str(data.get("install_id") or ""),
            upload_token_id=str(data.get("upload_token_id") or ""),
            upload_secret=str(data.get("upload_secret") or ""),
        )

    def _request_json(
        self,
        payload: dict,
        current: CommunityUploadCredentials | None = None,
    ) -> dict:
        body = json.dumps(payload, separators=(",", ":"))
        headers = {
            "Content-Type": "application/json",
        }
        if current is not None:
            timestamp = str(int(self._clock()))
            headers.update(
                {
                    "X-EspressoRL-Install-ID": current.install_id,
                    "X-EspressoRL-Token-ID": current.upload_token_id,
                    "X-EspressoRL-Timestamp": timestamp,
                    "X-EspressoRL-Signature": _signature(current.upload_secret, timestamp, body),
                }
            )
        req = request.Request(
            self._config.registration_url,
            data=body.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        response = self._transport(req, self._config.timeout_s)
        if response.status >= 400:
            raise RuntimeError(f"supabase credential request failed with HTTP {response.status}: {response.body[:500]}")
        if not response.body:
            return {}
        return json.loads(response.body)


class JsonCommunityCredentialStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> CommunityUploadCredentials | None:
        if not self._path.exists():
            return None
        try:
            payload = json.loads(self._path.read_text())
            return CommunityUploadCredentials(
                install_id=str(payload.get("install_id") or ""),
                upload_token_id=str(payload.get("upload_token_id") or ""),
                upload_secret=str(payload.get("upload_secret") or ""),
            )
        except Exception:
            return None

    def save(self, credentials: CommunityUploadCredentials) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                {
                    "install_id": credentials.install_id,
                    "upload_token_id": credentials.upload_token_id,
                    "upload_secret": credentials.upload_secret,
                },
                indent=2,
                sort_keys=True,
            )
        )
        self._path.chmod(0o600)

    def clear(self) -> None:
        try:
            self._path.unlink()
        except FileNotFoundError:
            return


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


def _signature(secret: str, timestamp: str, payload_json: str) -> str:
    message = f"{timestamp}.{payload_json}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), message, sha256).hexdigest()

from __future__ import annotations

import hmac
import json
import logging
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Callable
from urllib import error, request

from espresso_rl.domain.models import UploadQueueItem, UploadQueueStatus
from espresso_rl.ports.repositories import UploadQueueRepository

logger = logging.getLogger(__name__)


class UploadRejected(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class SignedUploadConfig:
    ingest_url: str
    install_id: str
    upload_secret: str
    upload_token_id: str = ""
    timeout_s: float = 10.0
    max_payload_bytes: int = 2_000_000


class SignedSupabaseUploadClient:
    """
    Adapter for a Supabase Edge Function or compatible ingestion endpoint.

    The core only writes upload queue items. This adapter signs each queued
    payload with an install-local secret and posts it to the configured endpoint.
    """

    def __init__(self, config: SignedUploadConfig) -> None:
        if not config.ingest_url:
            raise ValueError("ingest_url is required")
        if not config.upload_secret:
            raise ValueError("upload_secret is required")
        self._config = config

    def upload(self, item: UploadQueueItem) -> None:
        body = item.payload_json.encode("utf-8")
        if len(body) > self._config.max_payload_bytes:
            raise UploadRejected(413, "payload too large")

        timestamp = str(int(time.time()))
        signature = self._signature(timestamp, item.payload_json)
        headers = {
            "Content-Type": "application/json",
            "X-EspressoRL-Install-ID": self._config.install_id,
            "X-EspressoRL-Timestamp": timestamp,
            "X-EspressoRL-Signature": signature,
            "X-EspressoRL-Payload-Hash": item.payload_hash,
            "X-EspressoRL-Upload-ID": item.upload_id,
        }
        if self._config.upload_token_id:
            headers["X-EspressoRL-Token-ID"] = self._config.upload_token_id

        req = request.Request(
            self._config.ingest_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._config.timeout_s) as response:
                if response.status >= 400:
                    raise UploadRejected(response.status, response.reason)
        except error.HTTPError as exc:
            message = _read_error_message(exc)
            if 400 <= exc.code < 500 and exc.code != 429:
                raise UploadRejected(exc.code, message) from exc
            raise RuntimeError(f"upload failed with HTTP {exc.code}: {message}") from exc

    def _signature(self, timestamp: str, payload_json: str) -> str:
        message = f"{timestamp}.{payload_json}".encode("utf-8")
        return hmac.new(
            self._config.upload_secret.encode("utf-8"),
            message,
            sha256,
        ).hexdigest()


class UploadQueueWorker:
    def __init__(
        self,
        queue: UploadQueueRepository,
        client: SignedSupabaseUploadClient,
        clock: Callable[[], int] = lambda: int(time.time()),
    ) -> None:
        self._queue = queue
        self._client = client
        self._clock = clock

    def run_once(self, limit: int = 25) -> int:
        now = int(self._clock())
        uploaded = 0
        for item in self._queue.list_ready(now=now, limit=limit):
            try:
                self._queue.update_status(
                    item.upload_id,
                    UploadQueueStatus.UPLOADING,
                    now=now,
                )
                self._client.upload(item)
                self._queue.update_status(
                    item.upload_id,
                    UploadQueueStatus.UPLOADED,
                    now=int(self._clock()),
                )
                uploaded += 1
            except UploadRejected as exc:
                self._queue.update_status(
                    item.upload_id,
                    UploadQueueStatus.REJECTED,
                    now=int(self._clock()),
                    error_message=f"HTTP {exc.status}: {exc}",
                )
            except Exception as exc:
                retry_at = int(self._clock()) + _retry_delay_s(item.attempt_count + 1)
                self._queue.update_status(
                    item.upload_id,
                    UploadQueueStatus.FAILED,
                    now=int(self._clock()),
                    error_message=str(exc),
                    next_retry_at=retry_at,
                )
                logger.warning("Upload failed for %s; retry at %s", item.upload_id, retry_at)
        return uploaded


def _retry_delay_s(attempt_count: int) -> int:
    return min(3600, 60 * (2 ** min(attempt_count, 5)))


def _read_error_message(exc: error.HTTPError) -> str:
    try:
        payload = exc.read().decode("utf-8")
        parsed = json.loads(payload)
        if isinstance(parsed, dict) and "error" in parsed:
            return str(parsed["error"])
        return payload[:500]
    except Exception:
        return exc.reason

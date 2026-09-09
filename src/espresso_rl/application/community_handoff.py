from collections.abc import Callable
from typing import Any

from espresso_rl.application.upload_payloads import (
    make_upload_item,
    recommendation_upload_payload,
    shot_upload_payload,
)
from espresso_rl.application.upload_validation import validate_upload_payload
from espresso_rl.ports.repositories import (
    RecommendationRepository,
    ShotRepository,
    UploadQueueRepository,
)


class CommunityHandoffService:
    """Transfer device backlog without using community records as optimizer evidence."""

    def __init__(
        self,
        queue: UploadQueueRepository,
        *,
        enabled: bool,
        install_id: str,
        clock: Callable[[], int],
        shots: ShotRepository | None = None,
        recommendations: RecommendationRepository | None = None,
    ):
        self._queue = queue
        self._enabled = enabled
        self._install_id = install_id
        self._clock = clock
        self._shots = shots
        self._recommendations = recommendations

    def accept(self, payload: dict[str, Any]) -> None:
        if not self._enabled:
            raise RuntimeError("Enable container community uploads before transferring the device backlog")
        # The container signs under its registered identity; never transfer secrets.
        payload = {**payload, "install_id": self._install_id}
        validation = validate_upload_payload(payload)
        if not validation.ok:
            raise ValueError("Invalid community handoff: " + "; ".join(validation.errors))
        record_type = payload["event_type"].removesuffix("_record")
        record_id = payload[f"{record_type}_id"]
        # Local corrections win even when uploads were disabled when recorded.
        if record_type == "shot" and self._shots is not None:
            shot = self._shots.get(record_id)
            if shot is not None:
                payload = shot_upload_payload(shot)
        elif record_type == "recommendation" and self._recommendations is not None:
            recommendation = self._recommendations.get(record_id)
            if recommendation is not None:
                payload = recommendation_upload_payload(recommendation)
        self._queue.enqueue(
            make_upload_item(record_type, record_id, payload, self._clock()), only_if_new=True,
        )

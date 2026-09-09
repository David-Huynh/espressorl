from typing import Protocol


class DeliveryReceiptRepository(Protocol):
    """Durable outcomes for adapter deliveries; independent of broker QoS."""

    def get(self, delivery_id: str) -> str | None: ...

    def record(self, delivery_id: str, outcome: str) -> None: ...

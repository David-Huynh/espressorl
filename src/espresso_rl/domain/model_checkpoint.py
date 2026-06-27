from __future__ import annotations

from dataclasses import dataclass

_HEX_CHARS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class DreamerCheckpointCompatibility:
    """Runtime-owned hashes used to reject structurally valid but incompatible models."""

    feature_layout_sha256: str | None = None
    control_spec_sha256: str | None = None
    tensor_contract_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "feature_layout_sha256",
            "control_spec_sha256",
            "tensor_contract_sha256",
        ):
            value = getattr(self, field_name)
            if value is not None and not _is_sha256(value):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class DreamerCheckpointTensor:
    name: str
    component: str
    dtype: str
    shape: tuple[int, ...]
    element_count: int
    sha256: str
    data_start: int
    data_end: int


@dataclass(frozen=True)
class VerifiedDreamerCheckpoint:
    """Authenticated checkpoint bytes that are not yet executable runtime state."""

    artifact_reference: str
    manifest_reference: str
    artifact_sha256: str
    manifest_sha256: str
    dataset_sha256: str
    dataset_manifest_sha256: str
    training_config_sha256: str
    tensor_contract_sha256: str
    feature_layout_sha256: str
    control_spec_sha256: str
    evaluation_report_sha256: str | None
    artifact_stage: str
    inference_ready: bool
    tensors: tuple[DreamerCheckpointTensor, ...]
    payload: bytes

    def __post_init__(self) -> None:
        if self.inference_ready:
            raise ValueError("runtime inference readiness is not enabled by the checkpoint loader contract")

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(sorted({tensor.component for tensor in self.tensors}))

    def tensor(self, name: str) -> DreamerCheckpointTensor:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)

    def tensor_bytes(self, name: str) -> memoryview:
        tensor = self.tensor(name)
        return memoryview(self.payload)[tensor.data_start : tensor.data_end]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_CHARS for character in value)
    )

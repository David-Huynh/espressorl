from __future__ import annotations

SHADOW_INFERENCE_CONTRACT_LEARNED_CONTEXT_ENCODER_V1 = "dreamer_v3_learned_context_encoder_v1"
SHADOW_INFERENCE_CONTRACT_LEGACY_V1 = "dreamer_v3_legacy_shadow_v1"

SUPPORTED_SHADOW_INFERENCE_CONTRACTS = frozenset(
    {
        SHADOW_INFERENCE_CONTRACT_LEARNED_CONTEXT_ENCODER_V1,
        SHADOW_INFERENCE_CONTRACT_LEGACY_V1,
    }
)


def validate_shadow_inference_contract_id(value: object, label: str = "shadow inference contract") -> str:
    if not isinstance(value, str) or value not in SUPPORTED_SHADOW_INFERENCE_CONTRACTS:
        raise ValueError(f"{label} is unsupported")
    return value

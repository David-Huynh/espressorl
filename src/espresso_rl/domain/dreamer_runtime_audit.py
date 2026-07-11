from __future__ import annotations

from dataclasses import dataclass

DREAMER_RUNTIME_EVENT_ACTIVE_RECOMMENDATION = "active_recommendation"
DREAMER_RUNTIME_EVENT_CPBO_FALLBACK = "cpbo_fallback"

DREAMER_FALLBACK_REASON_ACTIVE_UNAVAILABLE = "dreamer_active_unavailable"
DREAMER_FALLBACK_REASON_CANDIDATE_REJECTED = "dreamer_candidate_rejected"
DREAMER_FALLBACK_REASON_UNKNOWN = "dreamer_fallback_unknown"

_ALLOWED_FALLBACK_REASONS = {
    DREAMER_FALLBACK_REASON_ACTIVE_UNAVAILABLE,
    DREAMER_FALLBACK_REASON_CANDIDATE_REJECTED,
    DREAMER_FALLBACK_REASON_UNKNOWN,
}


@dataclass(frozen=True)
class DreamerRuntimeAuditSummary:
    active_recommendation_count: int = 0
    cpbo_fallback_count: int = 0
    cpbo_fallback_reason_counts: tuple[tuple[str, int], ...] = ()
    last_runtime_event: str | None = None
    last_cpbo_fallback_reason: str | None = None

    def record_active_recommendation(self) -> "DreamerRuntimeAuditSummary":
        active_count = _safe_nonnegative_int(self.active_recommendation_count)
        fallback_count = _safe_nonnegative_int(self.cpbo_fallback_count)
        return DreamerRuntimeAuditSummary(
            active_recommendation_count=active_count + 1,
            cpbo_fallback_count=fallback_count,
            cpbo_fallback_reason_counts=self.cpbo_fallback_reason_counts,
            last_runtime_event=DREAMER_RUNTIME_EVENT_ACTIVE_RECOMMENDATION,
            last_cpbo_fallback_reason=self.last_cpbo_fallback_reason,
        )

    def record_cpbo_fallback(self, reason: str) -> "DreamerRuntimeAuditSummary":
        fallback_reason = normalize_dreamer_fallback_reason(reason)
        counts = self.fallback_reason_counts_dict()
        counts[fallback_reason] = counts.get(fallback_reason, 0) + 1
        return DreamerRuntimeAuditSummary(
            active_recommendation_count=_safe_nonnegative_int(self.active_recommendation_count),
            cpbo_fallback_count=_safe_nonnegative_int(self.cpbo_fallback_count) + 1,
            cpbo_fallback_reason_counts=tuple(sorted(counts.items())),
            last_runtime_event=DREAMER_RUNTIME_EVENT_CPBO_FALLBACK,
            last_cpbo_fallback_reason=fallback_reason,
        )

    def fallback_reason_counts_dict(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for reason, count in self.cpbo_fallback_reason_counts:
            clean_count = _safe_nonnegative_int(count)
            if clean_count <= 0:
                continue
            clean_reason = normalize_dreamer_fallback_reason(reason)
            counts[clean_reason] = counts.get(clean_reason, 0) + clean_count
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "active_recommendation_count": _safe_nonnegative_int(self.active_recommendation_count),
            "cpbo_fallback_count": _safe_nonnegative_int(self.cpbo_fallback_count),
            "cpbo_fallback_reason_counts": self.fallback_reason_counts_dict(),
            "last_runtime_event": self.last_runtime_event,
            "last_cpbo_fallback_reason": self.last_cpbo_fallback_reason,
        }


def normalize_dreamer_fallback_reason(reason: str | None) -> str:
    text = str(reason or "").strip()
    if text in _ALLOWED_FALLBACK_REASONS:
        return text
    return DREAMER_FALLBACK_REASON_UNKNOWN


def _safe_nonnegative_int(value: object) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, integer)

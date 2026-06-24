from __future__ import annotations

import math
import re
import hashlib
from dataclasses import dataclass
from statistics import median
from typing import Any

from espresso_rl.application.upload_validation import validate_upload_payload
from espresso_rl.domain.community import CommunityPrior, CommunityTrainingRow
from espresso_rl.ports.community import CommunityWarehouseRepository


MAX_COMMUNITY_PRIOR_CONFIDENCE = 0.18
COMMUNITY_PRIOR_OBSERVATION_NOISE = 0.5
DEFAULT_MIN_INDEPENDENT_INSTALLS = 3
DEFAULT_MIN_CONTEXT_POINTS = 6
DEFAULT_MIN_DIVERSE_BUCKETS_FOR_SINGLE_INSTALL = 6
DEFAULT_MAX_POINTS_PER_INSTALL_PER_BUCKET = 3


@dataclass(frozen=True)
class CommunityPriorGenerationResult:
    examined: int
    eligible: int
    rejected: int
    contexts_seen: int
    priors_written: int


@dataclass(frozen=True)
class _PriorCandidate:
    row: CommunityTrainingRow
    context_key: str
    dose_g: float
    target_yield_g: float
    target_ratio: float
    reward: float
    reward_confidence: float
    trust_weight: float
    contribution_bucket: str

    @property
    def quality(self) -> float:
        return self.trust_weight * max(self.reward_confidence, 0.05) * max(self.reward, 0.0)


class CommunityPriorGenerationService:
    def __init__(
        self,
        warehouse: CommunityWarehouseRepository,
        *,
        min_independent_installs: int = DEFAULT_MIN_INDEPENDENT_INSTALLS,
        min_context_points: int = DEFAULT_MIN_CONTEXT_POINTS,
        min_diverse_buckets_for_single_install: int = DEFAULT_MIN_DIVERSE_BUCKETS_FOR_SINGLE_INSTALL,
        max_points_per_install_per_bucket: int = DEFAULT_MAX_POINTS_PER_INSTALL_PER_BUCKET,
        max_points_per_install_per_context: int | None = None,
    ) -> None:
        if min_independent_installs < 2:
            raise ValueError("min_independent_installs must be at least 2")
        if min_context_points < min_independent_installs:
            raise ValueError("min_context_points must be >= min_independent_installs")
        if min_diverse_buckets_for_single_install < 1:
            raise ValueError("min_diverse_buckets_for_single_install must be positive")
        if max_points_per_install_per_context is not None:
            max_points_per_install_per_bucket = max_points_per_install_per_context
        if max_points_per_install_per_bucket < 1:
            raise ValueError("max_points_per_install_per_bucket must be positive")
        self._warehouse = warehouse
        self._min_independent_installs = min_independent_installs
        self._min_context_points = min_context_points
        self._min_diverse_buckets_for_single_install = min_diverse_buckets_for_single_install
        self._max_points_per_install_per_bucket = max_points_per_install_per_bucket

    def generate_once(self, limit: int = 5000, *, dry_run: bool = False) -> CommunityPriorGenerationResult:
        rows = self._warehouse.list_training_rows(limit=limit)
        groups: dict[str, list[_PriorCandidate]] = {}
        rejected = 0
        eligible = 0

        for row in rows:
            candidate = self._candidate_from_row(row)
            if candidate is None:
                rejected += 1
                continue
            eligible += 1
            groups.setdefault(candidate.context_key, []).append(candidate)

        priors_written = 0
        for context_key, candidates in groups.items():
            capped = self._apply_install_cap(candidates)
            install_count = len({candidate.row.install_id for candidate in capped})
            diversity_count = len({candidate.contribution_bucket for candidate in capped})
            if len(capped) < self._min_context_points:
                continue
            if not self._has_sufficient_public_support(install_count, diversity_count):
                continue
            prior = aggregate_community_prior(
                context_key=context_key,
                candidates=capped,
                per_install_bucket_cap=self._max_points_per_install_per_bucket,
                min_independent_installs=self._min_independent_installs,
                min_diverse_buckets_for_single_install=self._min_diverse_buckets_for_single_install,
            )
            if prior.confidence <= 0:
                continue
            if not dry_run:
                self._warehouse.upsert_community_prior(prior)
            priors_written += 1

        return CommunityPriorGenerationResult(
            examined=len(rows),
            eligible=eligible,
            rejected=rejected,
            contexts_seen=len(groups),
            priors_written=priors_written,
        )

    def _candidate_from_row(self, row: CommunityTrainingRow) -> _PriorCandidate | None:
        if row.trust_weight <= 0:
            return None
        payload = dict(row.payload_json)
        if payload.get("install_id") != row.install_id:
            return None
        if payload.get("shot_type", "espresso") != "espresso":
            return None
        if payload.get("exclude_from_local_optimization") is True:
            return None
        validation = validate_upload_payload(payload)
        if not validation.ok:
            return None

        dose = _number(payload.get("dose_in_g"))
        target_yield = _number(payload.get("target_yield_g"))
        target_ratio = _number(payload.get("target_ratio"))
        if target_ratio is None and dose is not None and target_yield is not None:
            target_ratio = target_yield / dose
        reward = _number(payload.get("reward"))
        if reward is None and isinstance(payload.get("human_rating"), (int, float)):
            reward = (float(payload["human_rating"]) - 1.0) / 4.0
        reward_confidence = _number(payload.get("reward_confidence")) or 0.0
        if dose is None or target_yield is None or target_ratio is None or reward is None:
            return None
        if reward_confidence <= 0:
            return None

        return _PriorCandidate(
            row=row,
            context_key=community_prior_context_key(payload),
            dose_g=dose,
            target_yield_g=target_yield,
            target_ratio=target_ratio,
            reward=max(0.0, min(1.0, reward)),
            reward_confidence=max(0.0, min(1.0, reward_confidence)),
            trust_weight=max(0.0, min(0.25, row.trust_weight)),
            contribution_bucket=community_prior_contribution_bucket(payload),
        )

    def _apply_install_cap(self, candidates: list[_PriorCandidate]) -> list[_PriorCandidate]:
        by_install_bucket: dict[tuple[str, str], list[_PriorCandidate]] = {}
        for candidate in candidates:
            by_install_bucket.setdefault(
                (candidate.row.install_id, candidate.contribution_bucket),
                [],
            ).append(candidate)

        capped: list[_PriorCandidate] = []
        for install_candidates in by_install_bucket.values():
            capped.extend(
                sorted(install_candidates, key=lambda candidate: candidate.quality, reverse=True)[
                    : self._max_points_per_install_per_bucket
                ]
            )
        return sorted(capped, key=lambda candidate: candidate.quality, reverse=True)

    def _has_sufficient_public_support(self, install_count: int, diversity_count: int) -> bool:
        if install_count >= self._min_independent_installs:
            return True
        return diversity_count >= self._min_diverse_buckets_for_single_install


def community_prior_context_key(payload: dict[str, Any]) -> str:
    adapter = _slug(str(payload.get("machine_adapter") or "unknown"))
    dose = _number(payload.get("dose_in_g"))
    target_ratio = _number(payload.get("target_ratio"))
    target_yield = _number(payload.get("target_yield_g"))
    if target_ratio is None and dose is not None and target_yield is not None:
        target_ratio = target_yield / dose
    dose_bucket = _bucket(dose or 0.0, 0.5)
    ratio_bucket = _bucket(target_ratio or 0.0, 0.1)
    return f"adapter:{adapter}|dose:{dose_bucket:.1f}|ratio:{ratio_bucket:.1f}"


def community_prior_contribution_bucket(payload: dict[str, Any]) -> str:
    bean_context = str(payload.get("bean_context_id") or "none")
    grinder_context = str(payload.get("grinder_context_id") or "none")
    dose = _bucket(_number(payload.get("dose_in_g")) or 0.0, 0.5)
    target_yield = _bucket(_number(payload.get("target_yield_g")) or 0.0, 2.0)
    target_ratio = _number(payload.get("target_ratio"))
    if target_ratio is None:
        dose_value = _number(payload.get("dose_in_g"))
        yield_value = _number(payload.get("target_yield_g"))
        if dose_value is not None and yield_value is not None:
            target_ratio = yield_value / dose_value
    ratio = _bucket(target_ratio or 0.0, 0.1)

    grind_delta_um = _number(payload.get("recommended_grind_delta_um")) or 0.0
    dose_delta = 0.0
    recommended_dose = _number(payload.get("recommended_dose_g"))
    dose_value = _number(payload.get("dose_in_g"))
    if recommended_dose is not None and dose_value is not None:
        dose_delta = recommended_dose - dose_value
    yield_delta = 0.0
    recommended_yield = _number(payload.get("recommended_target_yield_g"))
    yield_value = _number(payload.get("target_yield_g"))
    if recommended_yield is not None and yield_value is not None:
        yield_delta = recommended_yield - yield_value

    return "|".join(
        [
            community_prior_context_key(payload),
            f"bean:{_fingerprint(bean_context)}",
            f"grinder:{_fingerprint(grinder_context)}",
            f"recipe:d{dose:.1f}:y{target_yield:.0f}:r{ratio:.1f}",
            f"action:g{_bucket(grind_delta_um, 25.0):.0f}:d{_bucket(dose_delta, 0.5):.1f}:y{_bucket(yield_delta, 2.0):.0f}",
            f"profile:{_profile_shape_bucket(payload)}",
        ]
    )


def aggregate_community_prior(
    *,
    context_key: str,
    candidates: list[_PriorCandidate],
    per_install_bucket_cap: int,
    min_independent_installs: int = DEFAULT_MIN_INDEPENDENT_INSTALLS,
    min_diverse_buckets_for_single_install: int = DEFAULT_MIN_DIVERSE_BUCKETS_FOR_SINGLE_INSTALL,
) -> CommunityPrior:
    if not candidates:
        raise ValueError("candidates are required")

    install_count = len({candidate.row.install_id for candidate in candidates})
    diversity_count = len({candidate.contribution_bucket for candidate in candidates})
    effective_weights = _diminished_install_weights(candidates)
    total_effective_weight = sum(effective_weights.values()) or 1.0
    dominant_install_share = max(
        sum(effective_weights[candidate.row.training_row_id] for candidate in candidates if candidate.row.install_id == install_id)
        / total_effective_weight
        for install_id in {candidate.row.install_id for candidate in candidates}
    )
    avg_trust = _weighted_average(
        [(candidate.trust_weight, effective_weights[candidate.row.training_row_id]) for candidate in candidates]
    )
    avg_confidence = _weighted_average(
        [(candidate.reward_confidence, effective_weights[candidate.row.training_row_id]) for candidate in candidates]
    )
    predicted_reward = _trimmed_weighted_mean(
        [(candidate.reward, effective_weights[candidate.row.training_row_id]) for candidate in candidates]
    )
    install_support = min(1.0, math.sqrt(install_count / 10.0))
    diversity_support = min(1.0, 0.25 * math.sqrt(diversity_count / 12.0))
    dominance_penalty = max(0.25, 1.0 - max(0.0, dominant_install_share - 0.5))
    confidence = min(
        MAX_COMMUNITY_PRIOR_CONFIDENCE,
        avg_trust * avg_confidence * max(install_support, diversity_support) * dominance_penalty,
    )
    confidence = round(max(0.0, confidence), 6)

    dose = round(float(median(candidate.dose_g for candidate in candidates)) * 2.0) / 2.0
    target_ratio = round(float(median(candidate.target_ratio for candidate in candidates)), 3)
    target_yield = round(float(median(candidate.target_yield_g for candidate in candidates)), 1)

    prior_json = {
        "schema_version": 1,
        "source": "community_validated_training_v1",
        "context_key": context_key,
        "zero_trust": {
            "validated_training_rows_only": True,
            "revalidated_before_aggregation": True,
            "per_install_contribution_bucket_cap": per_install_bucket_cap,
            "min_independent_installs": min_independent_installs,
            "min_diverse_buckets_for_single_install": min_diverse_buckets_for_single_install,
            "max_confidence": MAX_COMMUNITY_PRIOR_CONFIDENCE,
        },
        "aggregation": {
            "support": len(candidates),
            "independent_install_count": install_count,
            "contribution_bucket_count": diversity_count,
            "dominant_install_share": round(dominant_install_share, 6),
            "effective_weight_sum": round(total_effective_weight, 6),
            "avg_trust_weight": round(avg_trust, 6),
            "avg_reward_confidence": round(avg_confidence, 6),
            "method": "per-install-bucket-cap+diminishing-install-weight+median+trimmed-weighted-mean",
        },
        "points": [
            {
                "grind_delta_um": 0.0,
                "dose_g": dose,
                "target_yield_g": target_yield,
                "target_ratio": target_ratio,
                "predicted_reward": round(predicted_reward, 6),
                "confidence": confidence,
                "observation_noise": COMMUNITY_PRIOR_OBSERVATION_NOISE,
                "support": len(candidates),
                "independent_install_count": install_count,
                "contribution_bucket_count": diversity_count,
            }
        ],
    }
    return CommunityPrior(
        context_key=context_key,
        prior_json=prior_json,
        confidence=confidence,
    )


def _trimmed_weighted_mean(values: list[tuple[float, float]]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, key=lambda item: item[0])
    trim = int(len(ordered) * 0.1)
    if trim and len(ordered) - trim > trim:
        ordered = ordered[trim:-trim]
    weighted_sum = sum(value * weight for value, weight in ordered)
    weight_sum = sum(weight for _, weight in ordered)
    return weighted_sum / weight_sum if weight_sum else 0.0


def _weighted_average(values: list[tuple[float, float]]) -> float:
    if not values:
        return 0.0
    weight_sum = sum(weight for _, weight in values)
    if weight_sum <= 0:
        return 0.0
    return sum(value * weight for value, weight in values) / weight_sum


def _diminished_install_weights(candidates: list[_PriorCandidate]) -> dict[int, float]:
    by_install: dict[str, list[_PriorCandidate]] = {}
    for candidate in candidates:
        by_install.setdefault(candidate.row.install_id, []).append(candidate)

    weights: dict[int, float] = {}
    for install_candidates in by_install.values():
        ordered = sorted(install_candidates, key=lambda candidate: candidate.quality, reverse=True)
        for index, candidate in enumerate(ordered, start=1):
            base = max(candidate.trust_weight * candidate.reward_confidence, 0.001)
            weights[candidate.row.training_row_id] = base / math.sqrt(index)
    return weights


def _number(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def _bucket(value: float, step: float) -> float:
    return round(float(value) / step) * step


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _profile_shape_bucket(payload: dict[str, Any]) -> str:
    profile = payload.get("profile_resampled")
    shot_time = _bucket(_number(payload.get("shot_time_s")) or 0.0, 5.0)
    if not isinstance(profile, list) or len(profile) != 5:
        return f"noprof:t{shot_time:.0f}"

    pressure_peak = _channel_peak(profile[0])
    flow_peak = _channel_peak(profile[2])
    final_weight = _channel_last(profile[4])
    return (
        f"p{_bucket(pressure_peak, 1.0):.0f}:"
        f"f{_bucket(flow_peak, 1.0):.0f}:"
        f"w{_bucket(final_weight, 5.0):.0f}:"
        f"t{shot_time:.0f}"
    )


def _channel_peak(channel: Any) -> float:
    if not isinstance(channel, list):
        return 0.0
    values = [_number(value) for value in channel]
    numeric = [value for value in values if value is not None]
    return max(numeric) if numeric else 0.0


def _channel_last(channel: Any) -> float:
    if not isinstance(channel, list) or not channel:
        return 0.0
    value = _number(channel[-1])
    return value if value is not None else 0.0


def _slug(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9_-]+", "_", lowered)
    return slug.strip("_") or "unknown"

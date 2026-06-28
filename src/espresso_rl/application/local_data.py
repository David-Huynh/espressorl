from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from espresso_rl.domain.models import ShotRecord, ShotType, now_ts
from espresso_rl.ports.repositories import LocalDataRepository


@dataclass(frozen=True)
class LocalContextSummary:
    bean_context_id: str | None
    grinder_context_id: str | None
    shot_count: int
    optimizer_shot_count: int
    rated_shot_count: int
    rejected_upload_count: int
    latest_shot_at: int | None


@dataclass(frozen=True)
class LocalDataStatus:
    install_id: str
    machine_id: str
    contexts: list[LocalContextSummary]
    recent_shots: list[dict]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["contexts"] = [asdict(context) for context in self.contexts]
        return data


@dataclass(frozen=True)
class LocalDataActionResult:
    action: str
    dry_run: bool
    counts: dict[str, int]
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class LocalDataService:
    def __init__(
        self,
        repository: LocalDataRepository,
        *,
        install_id: str,
        machine_id: str,
        clock: Callable[[], int] = now_ts,
    ) -> None:
        self._repository = repository
        self._install_id = install_id
        self._machine_id = machine_id
        self._clock = clock

    def status(self, limit: int = 200) -> LocalDataStatus:
        shots = self._repository.list_machine_shots(
            self._install_id,
            self._machine_id,
            limit=_safe_limit(limit, default=200, maximum=2000),
        )
        return LocalDataStatus(
            install_id=self._install_id,
            machine_id=self._machine_id,
            contexts=_context_summaries(shots),
            recent_shots=[_shot_summary(shot) for shot in reversed(shots[-50:])],
        )

    def delete_shot(self, shot_id: str, *, dry_run: bool = False) -> LocalDataActionResult:
        counts = self._repository.delete_shot(
            self._install_id,
            self._machine_id,
            _safe_id(shot_id),
            dry_run=dry_run,
        )
        return LocalDataActionResult(
            action="delete_shot",
            dry_run=dry_run,
            counts=counts,
            warnings=[],
        )

    def exclude_shot(self, shot_id: str, *, dry_run: bool = False) -> LocalDataActionResult:
        counts = self._repository.exclude_shot_from_optimization(
            self._install_id,
            self._machine_id,
            _safe_id(shot_id),
            now=self._clock(),
            dry_run=dry_run,
        )
        return LocalDataActionResult(
            action="exclude_shot_from_optimization",
            dry_run=dry_run,
            counts=counts,
            warnings=[],
        )

    def purge_useless_shots(
        self,
        *,
        bean_context_id: str | None = None,
        grinder_context_id: str | None = None,
        limit: int = 100,
        dry_run: bool = False,
    ) -> LocalDataActionResult:
        counts = self._repository.purge_useless_shots(
            self._install_id,
            self._machine_id,
            _optional_safe_id(bean_context_id),
            limit=_safe_limit(limit, default=100, maximum=1000),
            dry_run=dry_run,
            grinder_context_id=_optional_safe_id(grinder_context_id),
        )
        return LocalDataActionResult(
            action="purge_useless_shots",
            dry_run=dry_run,
            counts=counts,
            warnings=[
                "keeps shots that are still marked as local optimizer evidence"
            ],
        )

    def reset_optimizer_context(
        self,
        bean_context_id: str,
        *,
        grinder_context_id: str | None = None,
        dry_run: bool = False,
    ) -> LocalDataActionResult:
        counts = self._repository.reset_optimizer_context(
            self._install_id,
            self._machine_id,
            _safe_id(bean_context_id),
            now=self._clock(),
            dry_run=dry_run,
            grinder_context_id=_optional_safe_id(grinder_context_id),
        )
        return LocalDataActionResult(
            action="reset_optimizer_context",
            dry_run=dry_run,
            counts=counts,
            warnings=[
                "preserves shot history but removes this context from local BO evidence"
            ],
        )

    def reset_all(self, *, dry_run: bool = False) -> LocalDataActionResult:
        counts = self._repository.reset_all(
            self._install_id,
            self._machine_id,
            dry_run=dry_run,
        )
        return LocalDataActionResult(
            action="reset_all",
            dry_run=dry_run,
            counts=counts,
            warnings=[
                "deletes all local shots, recommendations, queued uploads, shadow evaluations, and quality reports for this machine"
            ],
        )


def _context_summaries(shots: list[ShotRecord]) -> list[LocalContextSummary]:
    grouped: dict[tuple[str | None, str | None], list[ShotRecord]] = {}
    for shot in shots:
        grouped.setdefault((shot.bean_context_id, shot.grinder_context_id), []).append(shot)
    contexts = []
    for (bean_context_id, grinder_context_id), context_shots in grouped.items():
        contexts.append(
            LocalContextSummary(
                bean_context_id=bean_context_id,
                grinder_context_id=grinder_context_id,
                shot_count=len(context_shots),
                optimizer_shot_count=sum(1 for shot in context_shots if _included_in_optimizer(shot)),
                rated_shot_count=sum(1 for shot in context_shots if _included_in_optimizer(shot) and shot.human_rating is not None),
                rejected_upload_count=sum(1 for shot in context_shots if getattr(shot, "_rejected_upload", False)),
                latest_shot_at=max((shot.timestamp for shot in context_shots), default=None),
            )
        )
    return sorted(contexts, key=lambda item: item.latest_shot_at or 0, reverse=True)


def _shot_summary(shot: ShotRecord) -> dict:
    return {
        "shot_id": shot.shot_id,
        "timestamp": shot.timestamp,
        "bean_context_id": shot.bean_context_id,
        "grinder_context_id": shot.grinder_context_id,
        "shot_type": shot.shot_type.value,
        "shot_time_s": shot.shot_time_s,
        "beverage_out_g": shot.beverage_out_g,
        "target_yield_g": shot.target_yield_g,
        "human_rating": shot.human_rating,
        "feedback_recorded": shot.feedback_recorded,
        "profile_label": shot.profile_label,
        "final_phase_name": shot.final_phase_name,
        "shot_end_state": shot.shot_end_state,
        "exclude_from_local_optimization": shot.exclude_from_local_optimization,
        "optimization_weight": shot.optimization_weight,
        "included_in_optimizer": _included_in_optimizer(shot),
        "profile_flow_valid": shot.profile_flow_valid,
        "profile_flow_masked": shot.profile_flow_masked,
        "rejected_upload": bool(getattr(shot, "_rejected_upload", False)),
    }


def _included_in_optimizer(shot: ShotRecord) -> bool:
    return (
        shot.shot_type == ShotType.ESPRESSO
        and not shot.exclude_from_local_optimization
        and shot.optimization_weight > 0.0
        and shot.feedback_recorded
        and shot.reward is not None
    )


def _safe_limit(value: object, *, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(maximum, parsed))


def _safe_id(value: object) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 180:
        raise ValueError("id is required")
    return text


def _optional_safe_id(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _safe_id(text)

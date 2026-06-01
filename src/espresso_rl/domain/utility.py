from __future__ import annotations

from dataclasses import dataclass

from espresso_rl.domain.events import ShotProfileEvent
from espresso_rl.domain.models import ShotType


UTILITY_TYPES = {ShotType.UTILITY_FLUSH, ShotType.CLEANING, ShotType.CALIBRATION}


@dataclass(frozen=True)
class ShotClassification:
    shot_type: ShotType
    exclude_from_local_optimization: bool
    optimization_weight: float
    rating_prompt_allowed: bool

    @property
    def locally_optimizable(self) -> bool:
        return self.shot_type == ShotType.ESPRESSO and not self.exclude_from_local_optimization and self.optimization_weight > 0.0


def classify_shot_profile_event(event: ShotProfileEvent) -> ShotClassification:
    requested_type = ShotType(event.shot_type or ShotType.ESPRESSO)
    is_utility = bool(event.utility) or requested_type in UTILITY_TYPES
    if event.shot_time_s is not None and event.shot_time_s < 7.5:
        is_utility = True
    if event.beverage_out_g is not None and event.beverage_out_g < 3.0:
        is_utility = True

    if is_utility:
        shot_type = requested_type if requested_type in UTILITY_TYPES else ShotType.UTILITY_FLUSH
        return ShotClassification(
            shot_type=shot_type,
            exclude_from_local_optimization=True,
            optimization_weight=0.0,
            rating_prompt_allowed=False,
        )

    if event.exclude_from_local_optimization or not event.local_optimization_enabled:
        return ShotClassification(
            shot_type=requested_type,
            exclude_from_local_optimization=True,
            optimization_weight=0.0,
            rating_prompt_allowed=event.rating_prompt_allowed,
        )

    weight = 1.0 if event.optimization_weight is None else float(event.optimization_weight)
    return ShotClassification(
        shot_type=requested_type,
        exclude_from_local_optimization=False,
        optimization_weight=max(0.0, min(1.0, weight)),
        rating_prompt_allowed=event.rating_prompt_allowed,
    )

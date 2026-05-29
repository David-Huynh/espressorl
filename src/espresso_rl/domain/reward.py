from __future__ import annotations

from .models import FollowThroughState, RewardResult


def compute_reward(
    human_rating: int | None,
    profile_score: float,
    follow_through: FollowThroughState,
    taste_tags: list[str] | None = None,
    profile_complete: bool = True,
    low_confidence_weight: float = 0.4,
) -> RewardResult:
    if not 0.0 <= profile_score <= 1.0:
        raise ValueError("profile_score must be in [0, 1]")
    if human_rating is not None and not 1 <= human_rating <= 5:
        raise ValueError("human_rating must be 1..5 or None")

    if human_rating is None:
        reward = low_confidence_weight * profile_score
        confidence = 0.4
    else:
        human_norm = (human_rating - 1) / 4.0
        reward = 0.8 * human_norm + 0.2 * profile_score
        confidence = 1.0

    if human_rating is None and taste_tags:
        confidence *= 1.25

    if follow_through == FollowThroughState.PARTIALLY_FOLLOWED:
        confidence *= 0.6
    elif follow_through == FollowThroughState.NOT_FOLLOWED:
        confidence *= 0.2
    elif follow_through == FollowThroughState.UNKNOWN:
        confidence *= 0.5

    if taste_tags and "channeling_suspected" in taste_tags:
        confidence *= 0.7
    if not profile_complete:
        confidence *= 0.5

    return RewardResult(
        reward=max(0.0, min(1.0, reward)),
        confidence=max(0.0, min(1.0, confidence)),
    )

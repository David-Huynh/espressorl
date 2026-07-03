from __future__ import annotations

import re
from collections.abc import Iterable


PROFILE_TASTE_TAGS = (
    "fruity",
    "citrus",
    "floral",
    "sweet",
    "nutty_cocoa",
    "roasted",
    "spice",
    "fermented",
)
FAULT_TASTE_TAGS = (
    "sour",
    "green_vegetative",
    "bitter",
    "astringent_harsh",
    "papery_stale",
    "salty",
)
USER_TASTE_TAGS = PROFILE_TASTE_TAGS + FAULT_TASTE_TAGS
SYSTEM_TASTE_TAGS = ("channeling_suspected",)
VALID_TASTE_TAGS = frozenset(USER_TASTE_TAGS + SYSTEM_TASTE_TAGS)

TASTE_TAG_ALIASES = {
    "fruit": "fruity",
    "fruitiness": "fruity",
    "berry": "fruity",
    "berries": "fruity",
    "sweetness": "sweet",
    "overall_sweet": "sweet",
    "balanced": "sweet",
    "balance": "sweet",
    "chocolate": "nutty_cocoa",
    "chocolatiness": "nutty_cocoa",
    "cocoa": "nutty_cocoa",
    "nutty": "nutty_cocoa",
    "good_body": "nutty_cocoa",
    "roast": "roasted",
    "roastiness": "roasted",
    "spices": "spice",
    "acidic": "citrus",
    "acidity": "citrus",
    "vegetative": "green_vegetative",
    "green": "green_vegetative",
    "under_ripe": "green_vegetative",
    "underripe": "green_vegetative",
    "astringent": "astringent_harsh",
    "harsh": "astringent_harsh",
    "dry": "astringent_harsh",
    "papery": "papery_stale",
    "stale": "papery_stale",
    "weak": "green_vegetative",
    "thin": "green_vegetative",
    "muddy": "papery_stale",
}


def normalize_taste_tag(value: object, *, allow_system: bool = True) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = value.strip().lower()
    parsed = re.sub(r"[^a-z0-9_ /-]+", "", parsed)
    parsed = re.sub(r"[\s/-]+", "_", parsed).strip("_")
    canonical = TASTE_TAG_ALIASES.get(parsed, parsed)
    valid = VALID_TASTE_TAGS if allow_system else frozenset(USER_TASTE_TAGS)
    return canonical if canonical in valid else None


def normalize_taste_tags(values: Iterable[object], *, allow_system: bool = True) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    invalid: list[object] = []
    for value in values:
        tag = normalize_taste_tag(value, allow_system=allow_system)
        if tag is None:
            invalid.append(value)
            continue
        if tag in seen:
            continue
        normalized.append(tag)
        seen.add(tag)
    if invalid:
        raise ValueError(f"invalid taste tags: {sorted(str(value) for value in invalid)}")
    return normalized

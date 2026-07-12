from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


TASTE_GOAL_SCHEMA_VERSION = 1
PROFILE_TASTE_ATTRIBUTES = (
    "fruity",
    "citrus",
    "floral",
    "sweet",
    "nutty_cocoa",
    "roasted",
    "spice",
    "fermented",
)
FAULT_TASTE_ATTRIBUTES = (
    "sour",
    "green_vegetative",
    "bitter",
    "astringent_harsh",
    "papery_stale",
    "salty",
)
TASTE_GOAL_ATTRIBUTES = PROFILE_TASTE_ATTRIBUTES + FAULT_TASTE_ATTRIBUTES


class TasteGoalMode(str, Enum):
    BALANCED = "balanced"
    CUSTOM = "custom"


class TasteGoalLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class TasteGoal:
    """Immutable rubric that defines what a preference comparison means."""

    mode: TasteGoalMode = TasteGoalMode.BALANCED
    targets: tuple[tuple[str, TasteGoalLevel], ...] = ()
    schema_version: int = TASTE_GOAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TASTE_GOAL_SCHEMA_VERSION:
            raise ValueError("taste goal schema_version is unsupported")
        object.__setattr__(self, "mode", TasteGoalMode(self.mode))

        raw_targets: Any = self.targets
        if isinstance(raw_targets, Mapping):
            entries = raw_targets.items()
        else:
            entries = raw_targets
        normalized: dict[str, TasteGoalLevel] = {}
        try:
            for attribute, level in entries:
                attribute = str(attribute)
                if attribute not in TASTE_GOAL_ATTRIBUTES:
                    raise ValueError(f"taste goal attribute is unsupported: {attribute}")
                if attribute in normalized:
                    raise ValueError(f"taste goal attribute is duplicated: {attribute}")
                normalized[attribute] = TasteGoalLevel(level)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("taste goal"):
                raise
            raise ValueError("taste goal targets are invalid") from exc

        ordered = tuple(
            (attribute, normalized[attribute])
            for attribute in TASTE_GOAL_ATTRIBUTES
            if attribute in normalized
        )
        if self.mode == TasteGoalMode.BALANCED and ordered:
            raise ValueError("balanced taste goal cannot contain custom targets")
        if self.mode == TasteGoalMode.CUSTOM and not ordered:
            raise ValueError("custom taste goal requires at least one target")
        object.__setattr__(self, "targets", ordered)

    @classmethod
    def balanced(cls) -> "TasteGoal":
        return cls()

    @classmethod
    def custom(cls, targets: Mapping[str, str | TasteGoalLevel]) -> "TasteGoal":
        return cls(mode=TasteGoalMode.CUSTOM, targets=tuple(targets.items()))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "TasteGoal":
        if value is None:
            return cls.balanced()
        if not isinstance(value, Mapping):
            raise ValueError("taste goal must be an object")
        allowed = {"schema_version", "mode", "targets"}
        unknown = sorted(set(value) - allowed)
        missing = sorted(allowed - set(value))
        if unknown or missing:
            details = []
            if missing:
                details.append(f"missing={','.join(missing)}")
            if unknown:
                details.append(f"unknown={','.join(unknown)}")
            raise ValueError(f"taste goal fields are invalid ({'; '.join(details)})")
        targets = value.get("targets")
        if not isinstance(targets, Mapping):
            raise ValueError("taste goal targets must be an object")
        return cls(
            schema_version=value.get("schema_version"),
            mode=value.get("mode"),
            targets=tuple(targets.items()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "targets": {attribute: level.value for attribute, level in self.targets},
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def summary(self) -> str:
        if self.mode == TasteGoalMode.BALANCED:
            return "Balanced"
        return ", ".join(
            f"{attribute.replace('_', ' ')} {level.value}"
            for attribute, level in self.targets
        )

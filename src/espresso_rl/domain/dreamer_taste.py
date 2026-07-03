from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from espresso_rl.domain.taste import USER_TASTE_TAGS, normalize_taste_tag

DREAMER_TASTE_OBJECTIVE_SPEC_FORMAT = "espresso_rl_dreamer_taste_objective_spec_v2"
DREAMER_TASTE_OBJECTIVE_SPEC_SCHEMA_VERSION = 2
DREAMER_TASTE_OBJECTIVE_MODES = ("auto", "custom")
DREAMER_TASTE_OBJECTIVE_ATTRIBUTES = USER_TASTE_TAGS
DREAMER_TASTE_OBJECTIVE_LEVELS = ("unspecified", "low", "medium", "high")
DREAMER_TASTE_OBJECTIVE_LEVEL_ENCODING = {
    "unspecified": 0.0,
    "low": 1.0 / 3.0,
    "medium": 2.0 / 3.0,
    "high": 1.0,
}

_SPEC_FIELDS = frozenset(
    {
        "format",
        "schema_version",
        "modes",
        "attributes",
        "levels",
        "level_encoding",
    }
)


@dataclass(frozen=True)
class DreamerTasteObjectiveSpec:
    format: str = DREAMER_TASTE_OBJECTIVE_SPEC_FORMAT
    schema_version: int = DREAMER_TASTE_OBJECTIVE_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.format != DREAMER_TASTE_OBJECTIVE_SPEC_FORMAT:
            raise ValueError("Dreamer taste-objective spec format is unsupported")
        if self.schema_version != DREAMER_TASTE_OBJECTIVE_SPEC_SCHEMA_VERSION:
            raise ValueError("Dreamer taste-objective spec schema_version is unsupported")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "modes": list(DREAMER_TASTE_OBJECTIVE_MODES),
            "attributes": list(DREAMER_TASTE_OBJECTIVE_ATTRIBUTES),
            "levels": list(DREAMER_TASTE_OBJECTIVE_LEVELS),
            "level_encoding": dict(DREAMER_TASTE_OBJECTIVE_LEVEL_ENCODING),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "DreamerTasteObjectiveSpec":
        if not isinstance(value, dict):
            raise ValueError("Dreamer taste-objective spec must be an object")
        unknown = sorted(str(key) for key in value if key not in _SPEC_FIELDS)
        missing = sorted(key for key in _SPEC_FIELDS if key not in value)
        if unknown or missing:
            raise ValueError("Dreamer taste-objective spec fields are incompatible")
        if value.get("modes") != list(DREAMER_TASTE_OBJECTIVE_MODES):
            raise ValueError("Dreamer taste-objective modes are incompatible")
        if value.get("attributes") != list(DREAMER_TASTE_OBJECTIVE_ATTRIBUTES):
            raise ValueError("Dreamer taste-objective attribute ordering is incompatible")
        if value.get("levels") != list(DREAMER_TASTE_OBJECTIVE_LEVELS):
            raise ValueError("Dreamer taste-objective levels are incompatible")
        if value.get("level_encoding") != DREAMER_TASTE_OBJECTIVE_LEVEL_ENCODING:
            raise ValueError("Dreamer taste-objective level encoding is incompatible")
        return cls(format=value.get("format"), schema_version=value.get("schema_version"))


DEFAULT_DREAMER_TASTE_OBJECTIVE_SPEC = DreamerTasteObjectiveSpec()


def dreamer_taste_objective_spec_sha256(spec: DreamerTasteObjectiveSpec) -> str:
    payload = json.dumps(
        spec.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


DEFAULT_DREAMER_TASTE_OBJECTIVE_SPEC_SHA256 = dreamer_taste_objective_spec_sha256(
    DEFAULT_DREAMER_TASTE_OBJECTIVE_SPEC
)


def validate_dreamer_taste_objective(value: object, *, path: str = "taste_objective") -> list[str]:
    if not isinstance(value, dict):
        return [f"{path} must be an object"]
    normalized, unknown = _canonical_objective_fields(value)
    errors = [f"{path} contains unsupported fields: {', '.join(unknown[:5])}"] if unknown else []
    if normalized.get("mode") not in DREAMER_TASTE_OBJECTIVE_MODES:
        errors.append(f"{path}.mode is invalid")
    selected_count = 0
    for attribute in DREAMER_TASTE_OBJECTIVE_ATTRIBUTES:
        level = normalized.get(attribute)
        if level is not None and level not in DREAMER_TASTE_OBJECTIVE_LEVELS:
            errors.append(f"{path}.{attribute} is invalid")
        elif level in {"low", "medium", "high"}:
            selected_count += 1
    if normalized.get("mode") == "auto" and selected_count:
        errors.append(f"{path} auto mode must not include custom targets")
    if normalized.get("mode") == "custom" and selected_count == 0:
        errors.append(f"{path} custom mode requires at least one target")
    return errors


def normalize_dreamer_taste_objective(value: object) -> dict[str, str]:
    errors = validate_dreamer_taste_objective(value)
    if errors:
        raise ValueError("; ".join(errors[:10]))
    assert isinstance(value, dict)
    objective, _ = _canonical_objective_fields(value)
    if objective["mode"] == "auto":
        return {"mode": "auto"}
    return {
        "mode": "custom",
        **{
            attribute: str(objective[attribute])
            for attribute in DREAMER_TASTE_OBJECTIVE_ATTRIBUTES
            if objective.get(attribute) in {"low", "medium", "high"}
        },
    }


def encode_dreamer_taste_objective(value: object) -> tuple[float, ...]:
    objective = normalize_dreamer_taste_objective(value)
    return (
        1.0 if objective["mode"] == "auto" else 0.0,
        *(
            DREAMER_TASTE_OBJECTIVE_LEVEL_ENCODING.get(objective.get(attribute, "unspecified"), 0.0)
            for attribute in DREAMER_TASTE_OBJECTIVE_ATTRIBUTES
        ),
    )


def _canonical_objective_fields(value: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    normalized: dict[str, Any] = {"mode": value.get("mode")}
    unknown: list[str] = []
    for key, level in value.items():
        if key == "mode":
            continue
        tag = normalize_taste_tag(str(key), allow_system=False)
        if tag is None:
            unknown.append(str(key))
            continue
        if tag not in normalized or normalized[tag] in (None, "unspecified"):
            normalized[tag] = level
    return normalized, sorted(unknown)

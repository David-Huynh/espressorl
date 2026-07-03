from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from espresso_rl.domain.taste import normalize_taste_tag


MAX_PRIOR_RULES = 16
MAX_RULE_NAME_LENGTH = 80
MAX_RULE_TAG_LENGTH = 40


class PriorSelectionMode(str, Enum):
    NO_PRIORS = "no_priors"
    COMMUNITY_ONLY = "community_only"
    RULES_AND_COMMUNITY = "rules_and_community"


class PriorRuleMetric(str, Enum):
    TASTE_TAG = "taste_tag"
    SHOT_TIME_S = "shot_time_s"
    BREW_RATIO = "brew_ratio"
    HUMAN_RATING = "human_rating"


class PriorRuleOperator(str, Enum):
    CONTAINS = "contains"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"


class PriorRuleSource(str, Enum):
    USER = "user"
    COMMUNITY = "community"


class PriorGrindDirection(str, Enum):
    NONE = "none"
    FINER = "finer"
    COARSER = "coarser"


class PriorValueDirection(str, Enum):
    NONE = "none"
    INCREASE = "increase"
    DECREASE = "decrease"


@dataclass(frozen=True)
class PriorRule:
    rule_id: str
    name: str
    metric: PriorRuleMetric
    operator: PriorRuleOperator
    condition_value: str | float
    grind_direction: PriorGrindDirection = PriorGrindDirection.NONE
    ratio_direction: PriorValueDirection = PriorValueDirection.NONE
    dose_direction: PriorValueDirection = PriorValueDirection.NONE
    confidence: float = 0.35
    enabled: bool = True
    source: PriorRuleSource = PriorRuleSource.USER

    def __post_init__(self) -> None:
        rule_id = _short_identifier(self.rule_id, "rule_id", 96)
        name = _short_text(self.name, "name", MAX_RULE_NAME_LENGTH)
        metric = PriorRuleMetric(self.metric)
        operator = PriorRuleOperator(self.operator)
        source = PriorRuleSource(self.source)
        grind_direction = PriorGrindDirection(self.grind_direction)
        ratio_direction = PriorValueDirection(self.ratio_direction)
        dose_direction = PriorValueDirection(self.dose_direction)
        if metric == PriorRuleMetric.TASTE_TAG:
            if operator != PriorRuleOperator.CONTAINS:
                raise ValueError("taste_tag rules require the contains operator")
            condition_value: str | float = _normalized_tag(self.condition_value)
        else:
            if operator == PriorRuleOperator.CONTAINS:
                raise ValueError("numeric rules cannot use the contains operator")
            condition_value = _finite_number(self.condition_value, "condition_value")
            _validate_numeric_condition(metric, condition_value)

        confidence = _finite_number(self.confidence, "confidence")
        if not 0.05 <= confidence <= 0.65:
            raise ValueError("confidence must be between 0.05 and 0.65")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
        if (
            grind_direction == PriorGrindDirection.NONE
            and ratio_direction == PriorValueDirection.NONE
            and dose_direction == PriorValueDirection.NONE
        ):
            raise ValueError("a prior rule must define at least one direction")

        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "grind_direction", grind_direction)
        object.__setattr__(self, "ratio_direction", ratio_direction)
        object.__setattr__(self, "dose_direction", dose_direction)
        object.__setattr__(self, "condition_value", condition_value)
        object.__setattr__(self, "confidence", confidence)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PriorRule:
        if not isinstance(value, dict):
            raise ValueError("prior rule must be an object")
        allowed = {
            "rule_id",
            "name",
            "metric",
            "operator",
            "condition_value",
            "grind_direction",
            "ratio_direction",
            "dose_direction",
            "confidence",
            "enabled",
            "source",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"prior rule contains unknown fields: {', '.join(unknown[:5])}")
        return cls(
            rule_id=value.get("rule_id", ""),
            name=value.get("name", ""),
            metric=value.get("metric", ""),
            operator=value.get("operator", ""),
            condition_value=value.get("condition_value", ""),
            grind_direction=value.get("grind_direction", PriorGrindDirection.NONE.value),
            ratio_direction=value.get("ratio_direction", PriorValueDirection.NONE.value),
            dose_direction=value.get("dose_direction", PriorValueDirection.NONE.value),
            confidence=value.get("confidence", 0.35),
            enabled=value.get("enabled", True),
            source=value.get("source", PriorRuleSource.USER.value),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "metric": self.metric.value,
            "operator": self.operator.value,
            "condition_value": self.condition_value,
            "grind_direction": self.grind_direction.value,
            "ratio_direction": self.ratio_direction.value,
            "dose_direction": self.dose_direction.value,
            "confidence": self.confidence,
            "enabled": self.enabled,
            "source": self.source.value,
        }


def parse_prior_rules(values: object) -> tuple[PriorRule, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError("prior_rules must be an array")
    if len(values) > MAX_PRIOR_RULES:
        raise ValueError(f"prior_rules cannot contain more than {MAX_PRIOR_RULES} rules")
    rules = tuple(value if isinstance(value, PriorRule) else PriorRule.from_dict(value) for value in values)
    rule_ids = [rule.rule_id for rule in rules]
    if len(set(rule_ids)) != len(rule_ids):
        raise ValueError("prior rule IDs must be unique")
    return rules


def _short_identifier(value: object, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    parsed = value.strip()
    if not parsed or len(parsed) > max_length or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", parsed):
        raise ValueError(f"{field_name} must be a safe identifier")
    return parsed


def _short_text(value: object, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    parsed = value.strip()
    if not parsed or len(parsed) > max_length or any(ord(char) < 32 for char in parsed):
        raise ValueError(f"{field_name} must be a safe short string")
    return parsed


def _normalized_tag(value: object) -> str:
    parsed = _short_text(value, "condition_value", MAX_RULE_TAG_LENGTH).lower()
    parsed = re.sub(r"[^a-z0-9_ -]+", "", parsed).strip().replace(" ", "_")
    canonical = normalize_taste_tag(parsed, allow_system=False)
    if canonical is None:
        raise ValueError("taste tag condition is invalid")
    return canonical


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _validate_numeric_condition(metric: PriorRuleMetric, value: float) -> None:
    bounds = {
        PriorRuleMetric.SHOT_TIME_S: (0.0, 180.0),
        PriorRuleMetric.BREW_RATIO: (0.1, 10.0),
        PriorRuleMetric.HUMAN_RATING: (1.0, 5.0),
    }
    lower, upper = bounds[metric]
    if not lower <= value <= upper:
        raise ValueError(f"condition_value for {metric.value} must be between {lower} and {upper}")

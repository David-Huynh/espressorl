from __future__ import annotations

from espresso_rl.domain.models import ShotRecord
from espresso_rl.domain.optimization import OptimizationContext, PriorSignal
from espresso_rl.domain.prior_rules import (
    PriorGrindDirection,
    PriorRule,
    PriorRuleMetric,
    PriorRuleOperator,
    PriorRuleSource,
    PriorValueDirection,
)


def rule_prior_signals(
    context: OptimizationContext,
    rules: tuple[PriorRule, ...],
) -> list[PriorSignal]:
    if not context.shots:
        return []
    latest = context.shots[-1]
    if not latest.feedback_recorded:
        return []

    signals: list[PriorSignal] = []
    for rule in rules:
        if not rule.enabled or not _matches(rule, latest):
            continue
        source = "user_rule" if rule.source == PriorRuleSource.USER else "community_rule"
        confidence_cap = 0.65 if rule.source == PriorRuleSource.USER else 0.35
        signals.append(
            PriorSignal(
                grind_direction={
                    PriorGrindDirection.NONE: 0,
                    PriorGrindDirection.FINER: 1,
                    PriorGrindDirection.COARSER: -1,
                }[rule.grind_direction],
                ratio_direction=_value_direction(rule.ratio_direction),
                dose_direction=_value_direction(rule.dose_direction),
                confidence=min(rule.confidence, confidence_cap),
                observation_noise=0.3 if rule.source == PriorRuleSource.USER else 0.6,
                source=source,
                reason=f"Matched prior rule: {rule.name}",
            )
        )
    return signals


def _value_direction(direction: PriorValueDirection) -> int:
    return {
        PriorValueDirection.NONE: 0,
        PriorValueDirection.INCREASE: 1,
        PriorValueDirection.DECREASE: -1,
    }[direction]


def _matches(rule: PriorRule, shot: ShotRecord) -> bool:
    if rule.metric == PriorRuleMetric.TASTE_TAG:
        expected = str(rule.condition_value)
        tags = {str(tag).strip().lower().replace(" ", "_") for tag in shot.taste_tags}
        return expected in tags

    actual = {
        PriorRuleMetric.SHOT_TIME_S: shot.shot_time_s,
        PriorRuleMetric.BREW_RATIO: shot.brew_ratio,
        PriorRuleMetric.HUMAN_RATING: shot.human_rating,
    }[rule.metric]
    if actual is None:
        return False
    expected = float(rule.condition_value)
    if rule.operator == PriorRuleOperator.LT:
        return actual < expected
    if rule.operator == PriorRuleOperator.LTE:
        return actual <= expected
    if rule.operator == PriorRuleOperator.GT:
        return actual > expected
    if rule.operator == PriorRuleOperator.GTE:
        return actual >= expected
    return False

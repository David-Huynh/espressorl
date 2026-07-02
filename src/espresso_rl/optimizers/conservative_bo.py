from __future__ import annotations

import math

from espresso_rl.domain.models import (
    FollowThroughState,
    GrinderAdjustmentMode,
    Recommendation,
    RecommendationDecision,
    RecommendationMode,
    RecommendationStatus,
    ShotRecord,
    ShotType,
    new_id,
)
from espresso_rl.domain.optimization import OptimizationContext, PriorPoint, PriorSignal
from espresso_rl.domain.safety import clamp_candidate_recipe, validate_recommendation

MAX_USABLE_PRIOR_POINTS = 64


class ConservativeBOOptimizer:
    """
    Small-step BO-style optimizer for cold start and fallback.

    This intentionally uses a bounded candidate set and a conservative local
    surrogate instead of allowing BoTorch to optimize a continuous space from a
    handful of noisy espresso shots. It is machine-agnostic and only consumes
    canonical shot records.
    """

    def recommend(self, context: OptimizationContext) -> Recommendation:
        shots = self._optimizer_shots(list(context.shots))
        prior_points = self._usable_prior_points(context, len(shots))
        prior_signals = self._usable_prior_signals(context, len(shots))
        current = context.current_recipe

        if not shots:
            recipe = current
            mode = RecommendationMode.ZERO_OBSERVE
            reason = "Use the current recipe for a baseline shot."
            confidence = 0.25
            source_shot_id = None
        else:
            radius_steps, radius_yield_g, dose_radius_g, mode = self._trust_region(
                shots,
                prior_points,
                prior_signals,
                context,
            )
            if (prior_points or prior_signals) and len(shots) <= 4:
                mode = RecommendationMode.WARM_STARTED_BO
            center = self._center_recipe(shots)
            recipe = self._choose_candidate(
                shots=shots,
                prior_points=prior_points,
                prior_signals=prior_signals,
                context=context,
                center=center,
                radius_steps=radius_steps,
                radius_yield_g=radius_yield_g,
                dose_radius_g=dose_radius_g,
            )
            reason = self._reason(
                recipe.relative_grind_steps_from_reference
                - current.relative_grind_steps_from_reference,
                recipe.target_yield_g - current.target_yield_g,
            )
            if mode == RecommendationMode.WARM_STARTED_BO:
                reason = self._warm_start_reason(prior_points, prior_signals)
            confidence = self._confidence(shots)
            if (prior_points or prior_signals) and mode == RecommendationMode.WARM_STARTED_BO:
                confidence = min(0.65, confidence + 0.05)
            source_shot_id = shots[-1].shot_id

        grind_delta_steps = recipe.relative_grind_steps_from_reference - current.relative_grind_steps_from_reference
        recommendation = Recommendation(
            recommendation_id=new_id("rec"),
            created_at=context.now,
            updated_at=context.now,
            expires_at=context.now + 12 * 60 * 60,
            install_id=context.install_id,
            machine_id=context.machine_id,
            bean_context_id=context.bean_context_id,
            grinder_context_id=context.grinder_context_id,
            grind_delta_steps_from_current=grind_delta_steps,
            grind_delta_um_from_current=grind_delta_steps
            * current.microns_per_step
            * current.grinder_direction_sign,
            projected_relative_step_from_reference=recipe.relative_grind_steps_from_reference,
            projected_relative_grind_um_from_reference=recipe.relative_grind_um_from_reference,
            next_dose_g=recipe.dose_g,
            target_yield_g=recipe.target_yield_g,
            target_ratio=recipe.target_ratio or recipe.target_yield_g / recipe.dose_g,
            mode=mode,
            confidence=confidence,
            reason=reason,
            status=RecommendationStatus.PENDING,
            source_shot_id=source_shot_id,
            grinder_step_direction=current.grinder_step_direction,
            grinder_adjustment_mode=current.grinder_adjustment_mode,
        )
        validate_recommendation(current, recommendation, context.safety_bounds)
        return recommendation

    def _trust_region(
        self,
        shots: list[ShotRecord],
        prior_points: list[PriorPoint],
        prior_signals: list[PriorSignal],
        context: OptimizationContext,
    ) -> tuple[int, float, float, RecommendationMode]:
        n = len(shots)
        if n <= 4:
            radius_steps, radius_yield_g, dose_radius_g, mode = 2, 4.0, 0.0, RecommendationMode.ZERO_IMMEDIATE_BO
        elif n <= 14:
            radius_steps, radius_yield_g, dose_radius_g, mode = 3, 6.0, 0.5, RecommendationMode.LOCAL_BO
        else:
            radius_steps, radius_yield_g, dose_radius_g, mode = 5, 8.0, 1.0, RecommendationMode.LOCAL_BO

        prior_strength = max(
            self._prior_strength(prior_points),
            self._prior_signal_strength(prior_signals),
        )
        near_good = self._near_good_signal(shots[-1] if shots else None)
        evidence_strength = prior_strength

        if n <= 4 and evidence_strength > 0:
            radius_steps = max(radius_steps, int(round(2 + 3 * evidence_strength)))
            radius_yield_g = max(radius_yield_g, 4.0 + 4.0 * evidence_strength)
        elif evidence_strength > 0:
            radius_steps = max(radius_steps, int(round(3 + 2 * evidence_strength)))
            radius_yield_g = max(radius_yield_g, 6.0 + 2.0 * evidence_strength)

        if near_good and prior_strength < 0.5:
            radius_steps = min(radius_steps, 2)
            radius_yield_g = min(radius_yield_g, 4.0)

        radius_steps = max(1, min(context.safety_bounds.max_grind_delta_steps_from_current, radius_steps))
        radius_yield_g = max(2.0, min(context.safety_bounds.max_yield_delta_g, radius_yield_g))
        dose_radius_g = min(context.safety_bounds.max_dose_delta_g, dose_radius_g)
        return radius_steps, radius_yield_g, dose_radius_g, mode

    def _eligible_shots(self, shots: list[ShotRecord]) -> list[ShotRecord]:
        return [shot for shot in shots if shot.reward is not None]

    def _center_recipe(self, shots: list[ShotRecord]) -> ShotRecord:
        eligible = self._eligible_shots(shots)

        def score(shot: ShotRecord) -> float:
            if shot.human_rating is not None:
                return 10.0 + shot.human_rating + (shot.reward or 0.0)
            if shot.reward is not None:
                return shot.reward * max(shot.reward_confidence, 0.05)
            return shot.profile_score or 0.0

        return max(eligible, key=lambda shot: (score(shot), shot.timestamp))

    def _choose_candidate(
        self,
        shots: list[ShotRecord],
        prior_points: list[PriorPoint],
        prior_signals: list[PriorSignal],
        context: OptimizationContext,
        center: ShotRecord,
        radius_steps: int,
        radius_yield_g: float,
        dose_radius_g: float,
    ):
        current = context.current_recipe
        if len(shots) == 1 and not prior_points and not prior_signals:
            grind_delta, yield_delta = self._single_point_probe(context)
            return clamp_candidate_recipe(
                current=current,
                candidate_relative_grind_steps_from_reference=current.relative_grind_steps_from_reference + grind_delta,
                candidate_dose_g=current.dose_g,
                candidate_target_yield_g=current.target_yield_g + yield_delta,
                bounds=context.safety_bounds,
            )

        center_grind = (
            center.relative_grind_steps_from_reference
            if center.grind_observed and center.relative_grind_steps_from_reference is not None
            else current.relative_grind_steps_from_reference
        )
        center_dose = center.dose_in_g if center.dose_observed else current.dose_g
        center_yield = center.realized_yield_g if center.realized_yield_observed else current.target_yield_g
        dose_offsets = self._dose_offsets(dose_radius_g, context)
        grind_offsets = self._grind_offsets(radius_steps, context)
        yield_offsets = self._yield_offsets(radius_yield_g, context)

        best_recipe = current
        best_score = -math.inf
        for step_offset in grind_offsets:
            for dose_offset in dose_offsets:
                for yield_offset in yield_offsets:
                    candidate = clamp_candidate_recipe(
                        current=current,
                        candidate_relative_grind_steps_from_reference=center_grind + step_offset,
                        candidate_dose_g=center_dose + dose_offset,
                        candidate_target_yield_g=center_yield + yield_offset,
                        bounds=context.safety_bounds,
                    )
                    candidate_score = self._candidate_score(
                        candidate,
                        shots,
                        prior_points,
                        prior_signals,
                        context,
                        radius_steps,
                        radius_yield_g,
                        max(dose_radius_g, 0.5),
                    )
                    if self._repeats_ignored_recommendation(candidate, context):
                        candidate_score -= 10.0
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_recipe = candidate
        return best_recipe

    def _optimizer_shots(self, shots: list[ShotRecord]) -> list[ShotRecord]:
        return [
            shot
            for shot in shots
            if shot.shot_type == ShotType.ESPRESSO
            and not shot.exclude_from_local_optimization
            and shot.optimization_weight > 0.0
            and shot.feedback_recorded
            and shot.reward is not None
        ]

    def _single_point_probe(self, context: OptimizationContext) -> tuple[float, float]:
        if context.current_recipe.grinder_adjustment_mode == GrinderAdjustmentMode.STEPLESS:
            return 0.5, 0.0
        return 1.0, 0.0

    def _near_good_signal(self, shot: ShotRecord | None) -> bool:
        if shot is None:
            return False
        if shot.human_rating is not None and shot.human_rating >= 4:
            return True
        return bool(shot.reward is not None and shot.reward >= 0.72)

    def _prior_strength(self, prior_points: list[PriorPoint]) -> float:
        strength = 0.0
        for point in prior_points:
            if point.confidence <= 0 or point.observation_noise <= 0:
                continue
            source_scale = self._prior_source_scale(point.source)
            point_strength = min(
                1.0,
                point.confidence * source_scale * self._prior_action_coverage(point) / point.observation_noise,
            )
            strength = max(strength, point_strength)
        return strength

    def _grind_offsets(self, radius_steps: int, context: OptimizationContext) -> list[float]:
        if context.current_recipe.grinder_adjustment_mode != GrinderAdjustmentMode.STEPLESS:
            return [float(value) for value in range(-radius_steps, radius_steps + 1)]
        spacing = 0.5
        count = int(round(radius_steps / spacing))
        return [round(index * spacing, 1) for index in range(-count, count + 1)]

    def _dose_offsets(self, radius_dose_g: float, context: OptimizationContext) -> list[float]:
        if radius_dose_g <= 0:
            return [0.0]
        if context.current_recipe.grinder_adjustment_mode != GrinderAdjustmentMode.STEPLESS:
            return [-radius_dose_g, 0.0, radius_dose_g]
        spacing = 0.1
        count = int(round(radius_dose_g / spacing))
        return [round(index * spacing, 1) for index in range(-count, count + 1)]

    def _yield_offsets(self, radius_yield_g: float, context: OptimizationContext) -> list[float]:
        values: list[float] = []
        step = 0.5 if context.current_recipe.grinder_adjustment_mode == GrinderAdjustmentMode.STEPLESS else 2.0
        k = int(radius_yield_g / step)
        for i in range(-k, k + 1):
            values.append(round(i * step, 1))
        return values

    def _candidate_score(
        self,
        candidate,
        shots: list[ShotRecord],
        prior_points: list[PriorPoint],
        prior_signals: list[PriorSignal],
        context: OptimizationContext,
        radius_steps: int,
        radius_yield_g: float,
        dose_radius_g: float,
    ) -> float:
        eligible = self._eligible_shots(shots)
        weighted_reward = 0.0
        weight_sum = 0.0
        min_distance = math.inf
        for shot in eligible:
            distance = self._distance(candidate, shot, radius_steps, radius_yield_g, dose_radius_g)
            min_distance = min(min_distance, distance)
            coverage = self._action_coverage(shot)
            weight = ((shot.reward_confidence or 0.1) * shot.optimization_weight * coverage) / (0.15 + distance)
            observation_reward = shot.reward if shot.reward is not None else (shot.profile_score or 0.0)
            weighted_reward += weight * observation_reward
            weight_sum += weight
        predicted_reward = weighted_reward / weight_sum if weight_sum else 0.0
        predicted_reward = self._blend_prior_reward(
            candidate=candidate,
            local_predicted_reward=predicted_reward,
            local_weight_sum=weight_sum,
            prior_points=prior_points,
            context=context,
            local_shot_count=len(shots),
            radius_steps=radius_steps,
            radius_yield_g=radius_yield_g,
            dose_radius_g=dose_radius_g,
        )
        predicted_reward += self._directional_prior_adjustment(
            candidate=candidate,
            signals=prior_signals,
            context=context,
            local_shot_count=len(shots),
            radius_steps=radius_steps,
            radius_yield_g=radius_yield_g,
            dose_radius_g=dose_radius_g,
        )

        distance_from_current = abs(candidate.relative_grind_steps_from_reference - context.current_recipe.relative_grind_steps_from_reference) / max(radius_steps, 1)
        distance_from_current += abs(candidate.target_yield_g - context.current_recipe.target_yield_g) / max(radius_yield_g, 1.0)
        distance_from_current += abs(candidate.dose_g - context.current_recipe.dose_g) / max(dose_radius_g, 0.5)

        exploration_bonus = 0.05 * min(min_distance if math.isfinite(min_distance) else 0.0, 1.0)
        evidence_strength = max(
            self._prior_strength(prior_points),
            self._prior_signal_strength(prior_signals),
        )
        distance_penalty_rate = 0.03 - 0.015 * evidence_strength
        distance_penalty = distance_penalty_rate * distance_from_current
        oscillation_penalty = self._oscillation_penalty(candidate, context)
        return predicted_reward + exploration_bonus - distance_penalty - oscillation_penalty

    def _blend_prior_reward(
        self,
        candidate,
        local_predicted_reward: float,
        local_weight_sum: float,
        prior_points: list[PriorPoint],
        context: OptimizationContext,
        local_shot_count: int,
        radius_steps: int,
        radius_yield_g: float,
        dose_radius_g: float,
    ) -> float:
        if not prior_points:
            return local_predicted_reward
        prior_decay = max(0.0, (5 - local_shot_count) / 5.0)
        if prior_decay <= 0:
            return local_predicted_reward

        prior_weighted_reward = 0.0
        prior_weight_sum = 0.0
        for point in prior_points:
            prior_relative_grind_steps_from_reference = (
                context.current_recipe.relative_grind_steps_from_reference
                + point.grind_delta_um_from_current
                / (context.current_recipe.microns_per_step * context.current_recipe.grinder_direction_sign)
            )
            distances: list[float] = []
            if point.grind_observed:
                distances.append(
                    abs(candidate.relative_grind_steps_from_reference - prior_relative_grind_steps_from_reference)
                    / max(radius_steps, 1)
                )
            if point.dose_observed:
                distances.append(abs(candidate.dose_g - point.dose_g) / max(dose_radius_g, 0.5))
            if point.target_yield_observed:
                distances.append(
                    abs(candidate.target_yield_g - point.target_yield_g) / max(radius_yield_g, 1.0)
                )
            distance = math.sqrt(sum(value * value for value in distances))
            weight = (
                min(point.confidence, 0.8)
                * prior_decay
                * self._prior_source_scale(point.source)
                * self._prior_action_coverage(point)
                / (point.observation_noise + 0.25 + distance)
            )
            prior_weighted_reward += weight * point.predicted_reward
            prior_weight_sum += weight

        if prior_weight_sum <= 0:
            return local_predicted_reward

        combined_weight = local_weight_sum + prior_weight_sum
        if combined_weight <= 0:
            return prior_weighted_reward / prior_weight_sum
        return (local_predicted_reward * local_weight_sum + prior_weighted_reward) / combined_weight

    def _usable_prior_points(
        self,
        context: OptimizationContext,
        local_shot_count: int,
    ) -> list[PriorPoint]:
        if local_shot_count <= 0:
            return []
        if local_shot_count >= 5:
            return []
        points: list[PriorPoint] = []
        for point in context.prior_points:
            if point.confidence <= 0:
                continue
            if point.observation_noise <= 0:
                continue
            if not context.safety_bounds.dose_min_g <= point.dose_g <= context.safety_bounds.dose_max_g:
                continue
            if not context.safety_bounds.target_yield_min_g <= point.target_yield_g <= context.safety_bounds.target_yield_max_g:
                continue
            if not context.safety_bounds.target_ratio_min <= point.target_ratio <= context.safety_bounds.target_ratio_max:
                continue
            points.append(point)
        return points[:MAX_USABLE_PRIOR_POINTS]

    def _usable_prior_signals(
        self,
        context: OptimizationContext,
        local_shot_count: int,
    ) -> list[PriorSignal]:
        if local_shot_count <= 0 or local_shot_count >= 5:
            return []
        return [
            signal
            for signal in context.prior_signals
            if signal.confidence > 0 and signal.observation_noise > 0
        ][:16]

    def _warm_start_reason(
        self,
        prior_points: list[PriorPoint],
        prior_signals: list[PriorSignal],
    ) -> str:
        if any(point.source == "local_bean_history" for point in prior_points):
            return "Same-bean previous bag history plus local shot data; staying inside the trust region."
        if any(signal.source == "user_rule" for signal in prior_signals):
            return "User rules guide direction while BO selects a bounded step from local evidence."
        if prior_signals:
            return "Community rules guide direction while BO selects a bounded step from local evidence."
        return "Warm-start prior plus local shot data; staying inside the trust region."

    def _prior_signal_strength(self, signals: list[PriorSignal]) -> float:
        strength = 0.0
        for signal in signals:
            signal_strength = min(
                1.0,
                signal.confidence
                * self._prior_source_scale(signal.source)
                / signal.observation_noise,
            )
            strength = max(strength, signal_strength)
        return strength

    def _directional_prior_adjustment(
        self,
        *,
        candidate,
        signals: list[PriorSignal],
        context: OptimizationContext,
        local_shot_count: int,
        radius_steps: int,
        radius_yield_g: float,
        dose_radius_g: float,
    ) -> float:
        if not signals:
            return 0.0
        prior_decay = max(0.0, (5 - local_shot_count) / 5.0)
        if prior_decay <= 0:
            return 0.0

        current = context.current_recipe
        numeric_grind_delta = (
            candidate.relative_grind_steps_from_reference
            - current.relative_grind_steps_from_reference
        )
        physical_grind_delta = numeric_grind_delta * current.grinder_direction_sign
        normalized_movements = {
            "grind": max(-1.0, min(1.0, physical_grind_delta / max(radius_steps, 1))),
            "ratio": max(
                -1.0,
                min(
                    1.0,
                    ((candidate.target_ratio or candidate.target_yield_g / candidate.dose_g) - (current.target_ratio or current.target_yield_g / current.dose_g))
                    / max(radius_yield_g / current.dose_g, 0.1),
                ),
            ),
            "dose": max(
                -1.0,
                min(1.0, (candidate.dose_g - current.dose_g) / max(dose_radius_g, 0.5)),
            ),
        }

        weighted_alignment = 0.0
        total_weight = 0.0
        for signal in signals:
            alignments = []
            if signal.grind_direction:
                alignments.append(signal.grind_direction * normalized_movements["grind"])
            if signal.ratio_direction:
                alignments.append(signal.ratio_direction * normalized_movements["ratio"])
            if signal.dose_direction:
                alignments.append(signal.dose_direction * normalized_movements["dose"])
            if not alignments:
                continue
            weight = min(
                1.0,
                signal.confidence
                * self._prior_source_scale(signal.source)
                / signal.observation_noise,
            )
            weighted_alignment += weight * (sum(alignments) / len(alignments))
            total_weight += weight
        if total_weight <= 0:
            return 0.0
        return 0.12 * prior_decay * weighted_alignment / total_weight

    def _prior_source_scale(self, source: str) -> float:
        if source == "local_bean_history":
            return 0.75
        if source == "local_history":
            return 0.6
        if source == "user_rule":
            return 0.65
        if source == "community_rule":
            return 0.35
        return 0.35

    def _distance(
        self,
        candidate,
        shot: ShotRecord,
        radius_steps: int,
        radius_yield_g: float,
        dose_radius_g: float,
    ) -> float:
        distances: list[float] = []
        if shot.grind_observed and shot.relative_grind_steps_from_reference is not None:
            distances.append(
                abs(candidate.relative_grind_steps_from_reference - shot.relative_grind_steps_from_reference)
                / max(radius_steps, 1)
            )
        if shot.dose_observed:
            distances.append(abs(candidate.dose_g - shot.dose_in_g) / max(dose_radius_g, 0.5))
        if shot.realized_yield_observed:
            distances.append(
                abs(candidate.target_yield_g - shot.realized_yield_g) / max(radius_yield_g, 1.0)
            )
        return math.sqrt(sum(distance * distance for distance in distances))

    def _action_coverage(self, shot: ShotRecord) -> float:
        observed = sum(
            (
                shot.grind_observed and shot.relative_grind_steps_from_reference is not None,
                shot.dose_observed,
                shot.realized_yield_observed,
            )
        )
        return observed / 3.0

    def _prior_action_coverage(self, point: PriorPoint) -> float:
        return sum((point.grind_observed, point.dose_observed, point.target_yield_observed)) / 3.0

    def _repeats_ignored_recommendation(self, candidate, context: OptimizationContext) -> bool:
        last = context.last_recommendation
        if last is None or last.status != RecommendationStatus.IGNORED:
            return False
        return (
            abs(candidate.relative_grind_steps_from_reference - last.projected_relative_step_from_reference) < 0.5
            and abs(candidate.dose_g - last.next_dose_g) < 0.05
            and abs(candidate.target_yield_g - last.target_yield_g) < 0.1
        )

    def _oscillation_penalty(self, candidate, context: OptimizationContext) -> float:
        last = context.last_recommendation
        if last is None:
            return 0.0
        new_delta = candidate.relative_grind_steps_from_reference - context.current_recipe.relative_grind_steps_from_reference
        if new_delta == 0 or last.grind_delta_steps_from_current == 0:
            return 0.0
        if (new_delta > 0) != (last.grind_delta_steps_from_current > 0):
            return 0.03
        return 0.0

    def _reason(self, grind_delta_steps_from_current: float, yield_delta_g: float) -> str:
        if grind_delta_steps_from_current == 0 and abs(yield_delta_g) < 0.1:
            return "Hold near the best known recipe while more feedback is collected."
        return "Small trust-region BO step near the best known recipe."

    def _confidence(self, shots: list[ShotRecord]) -> float:
        rated = sum(1 for shot in shots if shot.human_rating is not None)
        followed = sum(
            1
            for shot in shots
            if shot.recommendation_followed
            in {FollowThroughState.FOLLOWED, FollowThroughState.PARTIALLY_FOLLOWED}
        )
        base = 0.25 + min(0.35, rated * 0.04) + min(0.25, followed * 0.03)
        return max(0.0, min(0.85, base))

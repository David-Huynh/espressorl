from __future__ import annotations

import math

from espresso_rl.domain.models import (
    FollowThroughState,
    Recommendation,
    RecommendationDecision,
    RecommendationMode,
    RecommendationStatus,
    ShotRecord,
    ShotType,
    new_id,
)
from espresso_rl.domain.optimization import OptimizationContext, PriorPoint
from espresso_rl.domain.safety import clamp_candidate_recipe, validate_recommendation


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
        current = context.current_recipe

        if not shots:
            recipe = current
            mode = RecommendationMode.ZERO_OBSERVE
            reason = "Use the current recipe for a baseline shot."
            confidence = 0.25
            source_shot_id = None
        else:
            radius_steps, radius_yield_g, dose_radius_g, mode = self._trust_region(shots)
            if prior_points and any(point.source != "local_history" for point in prior_points) and len(shots) <= 4:
                mode = RecommendationMode.WARM_STARTED_BO
            center = self._center_recipe(shots)
            recipe = self._choose_candidate(
                shots=shots,
                prior_points=prior_points,
                context=context,
                center=center,
                radius_steps=radius_steps,
                radius_yield_g=radius_yield_g,
                dose_radius_g=dose_radius_g,
            )
            reason = self._reason(shots, recipe.grind_steps - current.grind_steps, recipe.target_yield_g - current.target_yield_g)
            if mode == RecommendationMode.WARM_STARTED_BO:
                reason = "Weak warm-start prior plus local shot data; staying inside the trust region."
            confidence = self._confidence(shots)
            if prior_points and mode == RecommendationMode.WARM_STARTED_BO:
                confidence = min(0.65, confidence + 0.05)
            source_shot_id = shots[-1].shot_id

        recommendation = Recommendation(
            recommendation_id=new_id("rec"),
            created_at=context.now,
            updated_at=context.now,
            expires_at=context.now + 12 * 60 * 60,
            install_id=context.install_id,
            machine_id=context.machine_id,
            bean_context_id=context.bean_context_id,
            grind_delta_steps=round(recipe.grind_steps - current.grind_steps),
            grind_delta_um=(recipe.grind_steps - current.grind_steps) * current.grinder_step_size_um,
            next_grind_steps=recipe.grind_steps,
            next_grind_um=recipe.grind_um,
            next_dose_g=recipe.dose_g,
            target_yield_g=recipe.target_yield_g,
            target_ratio=recipe.target_ratio or recipe.target_yield_g / recipe.dose_g,
            mode=mode,
            confidence=confidence,
            reason=reason,
            status=RecommendationStatus.PENDING,
            source_shot_id=source_shot_id,
        )
        validate_recommendation(current, recommendation, context.safety_bounds)
        return recommendation

    def _trust_region(self, shots: list[ShotRecord]) -> tuple[int, float, float, RecommendationMode]:
        n = len(shots)
        if n <= 4:
            return 2, 4.0, 0.0, RecommendationMode.ZERO_IMMEDIATE_BO
        if n <= 14:
            return 3, 6.0, 0.5, RecommendationMode.LOCAL_BO
        return 5, 8.0, 1.0, RecommendationMode.LOCAL_BO

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
        context: OptimizationContext,
        center: ShotRecord,
        radius_steps: int,
        radius_yield_g: float,
        dose_radius_g: float,
    ):
        current = context.current_recipe
        if len(shots) == 1 and not prior_points:
            grind_delta, yield_delta = self._single_point_probe(shots[0])
            return clamp_candidate_recipe(
                current=current,
                candidate_grind_steps=current.grind_steps + grind_delta,
                candidate_dose_g=current.dose_g,
                candidate_target_yield_g=current.target_yield_g + yield_delta,
                bounds=context.safety_bounds,
            )

        center_grind = center.grind_steps if center.grind_steps is not None else current.grind_steps
        center_dose = center.dose_in_g
        center_yield = center.target_yield_g
        dose_offsets = [0.0] if dose_radius_g == 0 else [-dose_radius_g, 0.0, dose_radius_g]
        yield_offsets = self._yield_offsets(radius_yield_g)

        best_recipe = current
        best_score = -math.inf
        for step_offset in range(-radius_steps, radius_steps + 1):
            for dose_offset in dose_offsets:
                for yield_offset in yield_offsets:
                    candidate = clamp_candidate_recipe(
                        current=current,
                        candidate_grind_steps=center_grind + step_offset,
                        candidate_dose_g=center_dose + dose_offset,
                        candidate_target_yield_g=center_yield + yield_offset,
                        bounds=context.safety_bounds,
                    )
                    candidate_score = self._candidate_score(
                        candidate,
                        shots,
                        prior_points,
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
            and shot.recommendation_decision
            not in {RecommendationDecision.IGNORED, RecommendationDecision.DISMISSED}
            and shot.recommendation_followed != FollowThroughState.NOT_FOLLOWED
        ]

    def _single_point_probe(self, shot: ShotRecord) -> tuple[int, float]:
        tags = set(shot.taste_tags)
        if {"sour", "weak", "thin", "too_fast"} & tags or (shot.shot_time_s is not None and shot.shot_time_s < 25):
            return 1, 2.0
        if {"bitter", "harsh", "astringent", "dry", "muddy", "too_slow"} & tags or (
            shot.shot_time_s is not None and shot.shot_time_s > 35
        ):
            return -1, -2.0
        return 1, 0.0

    def _yield_offsets(self, radius_yield_g: float) -> list[float]:
        values: list[float] = []
        step = 2.0
        k = int(radius_yield_g / step)
        for i in range(-k, k + 1):
            values.append(i * step)
        return values

    def _candidate_score(
        self,
        candidate,
        shots: list[ShotRecord],
        prior_points: list[PriorPoint],
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
            if shot.grind_steps is None:
                continue
            distance = self._distance(candidate, shot, radius_steps, radius_yield_g, dose_radius_g)
            min_distance = min(min_distance, distance)
            weight = ((shot.reward_confidence or 0.1) * shot.optimization_weight) / (0.15 + distance)
            observation_reward = shot.reward if shot.reward is not None else (shot.profile_score or 0.0)
            observation_reward += self._taste_candidate_adjustment(
                candidate,
                shot,
                radius_steps,
                radius_yield_g,
                dose_radius_g,
            )
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

        distance_from_current = abs(candidate.grind_steps - context.current_recipe.grind_steps) / max(radius_steps, 1)
        distance_from_current += abs(candidate.target_yield_g - context.current_recipe.target_yield_g) / max(radius_yield_g, 1.0)
        distance_from_current += abs(candidate.dose_g - context.current_recipe.dose_g) / max(dose_radius_g, 0.5)

        exploration_bonus = 0.05 * min(min_distance if math.isfinite(min_distance) else 0.0, 1.0)
        distance_penalty = 0.025 * distance_from_current
        oscillation_penalty = self._oscillation_penalty(candidate, context)
        return predicted_reward + exploration_bonus - distance_penalty - oscillation_penalty

    def _taste_candidate_adjustment(
        self,
        candidate,
        shot: ShotRecord,
        radius_steps: int,
        radius_yield_g: float,
        dose_radius_g: float,
    ) -> float:
        tags = set(shot.taste_tags)
        if not tags:
            return 0.0

        shot_grind = shot.grind_steps if shot.grind_steps is not None else candidate.grind_steps
        grind_delta = (candidate.grind_steps - shot_grind) / max(radius_steps, 1)
        yield_delta = (candidate.target_yield_g - shot.target_yield_g) / max(radius_yield_g, 1.0)
        extraction_direction = max(-1.0, min(1.0, 0.65 * grind_delta + 0.35 * yield_delta))

        if {"sour", "weak", "thin", "too_fast"} & tags:
            return 0.12 * extraction_direction
        if {"bitter", "harsh", "astringent", "dry", "muddy", "too_slow"} & tags:
            return -0.12 * extraction_direction
        if {"balanced", "sweet", "good_body"} & tags:
            distance = self._distance(candidate, shot, radius_steps, radius_yield_g, dose_radius_g)
            return -0.08 * min(distance, 1.0)
        return 0.0

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
            prior_grind_steps = (
                context.current_recipe.grind_steps
                + point.grind_delta_um / context.current_recipe.grinder_step_size_um
            )
            grind_d = abs(candidate.grind_steps - prior_grind_steps) / max(radius_steps, 1)
            dose_d = abs(candidate.dose_g - point.dose_g) / max(dose_radius_g, 0.5)
            yield_d = abs(candidate.target_yield_g - point.target_yield_g) / max(radius_yield_g, 1.0)
            distance = math.sqrt(grind_d * grind_d + dose_d * dose_d + yield_d * yield_d)
            source_scale = 0.6 if point.source == "local_history" else 0.35
            weight = (
                min(point.confidence, 0.8)
                * prior_decay
                * source_scale
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
        return points[:10]

    def _distance(
        self,
        candidate,
        shot: ShotRecord,
        radius_steps: int,
        radius_yield_g: float,
        dose_radius_g: float,
    ) -> float:
        shot_grind = shot.grind_steps if shot.grind_steps is not None else candidate.grind_steps
        grind_d = abs(candidate.grind_steps - shot_grind) / max(radius_steps, 1)
        dose_d = abs(candidate.dose_g - shot.dose_in_g) / max(dose_radius_g, 0.5)
        yield_d = abs(candidate.target_yield_g - shot.target_yield_g) / max(radius_yield_g, 1.0)
        return math.sqrt(grind_d * grind_d + dose_d * dose_d + yield_d * yield_d)

    def _repeats_ignored_recommendation(self, candidate, context: OptimizationContext) -> bool:
        last = context.last_recommendation
        if last is None or last.status != RecommendationStatus.IGNORED:
            return False
        return (
            abs(candidate.grind_steps - last.next_grind_steps) < 0.5
            and abs(candidate.dose_g - last.next_dose_g) < 0.05
            and abs(candidate.target_yield_g - last.target_yield_g) < 0.1
        )

    def _oscillation_penalty(self, candidate, context: OptimizationContext) -> float:
        last = context.last_recommendation
        if last is None:
            return 0.0
        new_delta = candidate.grind_steps - context.current_recipe.grind_steps
        if new_delta == 0 or last.grind_delta_steps == 0:
            return 0.0
        if (new_delta > 0) != (last.grind_delta_steps > 0):
            return 0.03
        return 0.0

    def _reason(self, shots: list[ShotRecord], grind_delta_steps: float, yield_delta_g: float) -> str:
        last = shots[-1]
        tags = set(last.taste_tags)
        if {"sour", "weak", "thin", "too_fast"} & tags:
            return "Last shot looked under-extracted; try a small finer/longer adjustment."
        if {"bitter", "harsh", "astringent", "dry", "muddy", "too_slow"} & tags:
            return "Last shot looked over-extracted or slow; try a small coarser/shorter adjustment."
        if grind_delta_steps == 0 and abs(yield_delta_g) < 0.1:
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

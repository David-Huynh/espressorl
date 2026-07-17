from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from espresso_rl.domain.cpbo import RecipePoint, RecipeSpace
from espresso_rl.optimizers.cpbo_config import MESConfig


@dataclass(frozen=True)
class CandidateDomain:
    proposal_recipes: tuple[RecipePoint, ...]
    discretization_recipes: tuple[RecipePoint, ...]
    proposal_indices: tuple[int, ...]
    maximum_indices: tuple[int, ...]
    anchor_index: int
    lower_bounds: tuple[float, float, float]
    upper_bounds: tuple[float, float, float]

    @property
    def normalized_x(self) -> Tensor:
        return torch.tensor(
            [recipe.normalized_x for recipe in self.discretization_recipes],
            dtype=torch.float64,
        )


def build_candidate_domain(
    *,
    run_id: str,
    recipe_space: RecipeSpace,
    evaluated_recipes: Sequence[RecipePoint],
    anchor_recipe: RecipePoint,
    config: MESConfig,
    seed: int,
    lower_bounds: tuple[float, float, float] = (0.0, 0.0, 0.0),
    upper_bounds: tuple[float, float, float] = (1.0, 1.0, 1.0),
    created_at: int,
) -> CandidateDomain:
    lower = torch.tensor(lower_bounds, dtype=torch.float64)
    upper = torch.tensor(upper_bounds, dtype=torch.float64)
    if lower.shape != (3,) or upper.shape != (3,) or torch.any(lower < 0) or torch.any(upper > 1):
        raise ValueError("candidate bounds must lie inside [0, 1]^3")
    if torch.any(lower > upper):
        raise ValueError("candidate lower bounds exceed upper bounds")
    if anchor_recipe.optimization_run_id != run_id:
        raise ValueError("anchor recipe belongs to another optimization run")

    points: list[Tensor] = []
    sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
    points.extend(lower + sobol.draw(config.sobol_candidate_count).to(torch.float64) * (upper - lower))
    points.extend(_local_perturbations(anchor_recipe.normalized_x, lower, upper, config, seed))
    points.extend(_boundary_points(lower, upper))

    evaluated_normalized = [torch.tensor(recipe.normalized_x, dtype=torch.float64) for recipe in evaluated_recipes]
    anchor_normalized = torch.tensor(anchor_recipe.normalized_x, dtype=torch.float64)
    candidates: list[RecipePoint] = []
    seen_recipe_ids: set[str] = set()
    for point in points:
        try:
            physical = recipe_space.inverse_recipe(point.tolist(), quantize=True)
            recipe = RecipePoint.create(
                run_id,
                recipe_space,
                *physical,
                created_at=created_at,
            )
        except ValueError:
            continue
        if recipe.recipe_id == anchor_recipe.recipe_id:
            continue
        if recipe.recipe_id in seen_recipe_ids:
            continue
        normalized = torch.tensor(recipe.normalized_x, dtype=torch.float64)
        if not _inside_bounds(normalized, lower, upper):
            continue
        if not config.allow_repeat_recipes and any(
            torch.linalg.vector_norm(normalized - observed) <= config.near_duplicate_tolerance
            for observed in evaluated_normalized
        ):
            continue
        if torch.linalg.vector_norm(normalized - anchor_normalized) <= config.near_duplicate_tolerance:
            continue
        seen_recipe_ids.add(recipe.recipe_id)
        candidates.append(recipe)

    if not candidates:
        candidates = _fallback_grid_candidates(
            run_id=run_id,
            recipe_space=recipe_space,
            evaluated_recipes=evaluated_recipes,
            anchor_recipe=anchor_recipe,
            config=config,
            lower=lower,
            upper=upper,
            created_at=created_at,
        )
    if not candidates:
        raise ValueError("no feasible nonduplicate CPBO candidate exists")

    discretization: list[RecipePoint] = []
    index_by_id: dict[str, int] = {}
    feasible_evaluated = [
        recipe for recipe in evaluated_recipes if recipe.inside_search_space
    ]
    for recipe in [*candidates, *feasible_evaluated, anchor_recipe]:
        if recipe.recipe_id not in index_by_id:
            index_by_id[recipe.recipe_id] = len(discretization)
            discretization.append(recipe)
    maximum_indices = tuple(
        dict.fromkeys(
            index_by_id[recipe.recipe_id]
            for recipe in [*candidates, *feasible_evaluated]
        )
    )
    return CandidateDomain(
        proposal_recipes=tuple(candidates),
        discretization_recipes=tuple(discretization),
        proposal_indices=tuple(index_by_id[recipe.recipe_id] for recipe in candidates),
        maximum_indices=maximum_indices,
        anchor_index=index_by_id[anchor_recipe.recipe_id],
        lower_bounds=tuple(float(value) for value in lower),
        upper_bounds=tuple(float(value) for value in upper),
    )


def _local_perturbations(
    anchor: tuple[float, float, float],
    lower: Tensor,
    upper: Tensor,
    config: MESConfig,
    seed: int,
) -> list[Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1_000_003)
    center = torch.tensor(anchor, dtype=torch.float64)
    width = (upper - lower).clamp_min(1e-6)
    noise = torch.randn((config.local_candidate_count, 3), dtype=torch.float64, generator=generator)
    return list(torch.clamp(center + 0.15 * width * noise, lower, upper))


def _boundary_points(lower: Tensor, upper: Tensor) -> list[Tensor]:
    midpoint = (lower + upper) / 2.0
    points = [torch.tensor(values, dtype=torch.float64) for values in itertools.product(*zip(lower, upper))]
    points.append(midpoint)
    for dimension in range(3):
        for boundary in (lower[dimension], upper[dimension]):
            point = midpoint.clone()
            point[dimension] = boundary
            points.append(point)
    return points


def _fallback_grid_candidates(
    *,
    run_id: str,
    recipe_space: RecipeSpace,
    evaluated_recipes: Sequence[RecipePoint],
    anchor_recipe: RecipePoint,
    config: MESConfig,
    lower: Tensor,
    upper: Tensor,
    created_at: int,
) -> list[RecipePoint]:
    evaluated_ids = {recipe.recipe_id for recipe in evaluated_recipes}
    anchor = torch.tensor(anchor_recipe.normalized_x, dtype=torch.float64)
    candidates: list[RecipePoint] = []
    seen: set[str] = set()
    for offset in (0.02, 0.05, 0.10, 0.20, 0.35, 0.50):
        for dimension in range(3):
            for sign in (-1.0, 1.0):
                point = anchor.clone()
                point[dimension] += sign * offset
                point = torch.clamp(point, lower, upper)
                try:
                    recipe = RecipePoint.create(
                        run_id,
                        recipe_space,
                        *recipe_space.inverse_recipe(point.tolist(), quantize=True),
                        created_at=created_at,
                    )
                except ValueError:
                    continue
                if recipe.recipe_id == anchor_recipe.recipe_id or recipe.recipe_id in seen:
                    continue
                if not config.allow_repeat_recipes and recipe.recipe_id in evaluated_ids:
                    continue
                seen.add(recipe.recipe_id)
                candidates.append(recipe)
    return candidates


def _inside_bounds(point: Tensor, lower: Tensor, upper: Tensor) -> bool:
    return bool(torch.all(point >= lower - 1e-12) and torch.all(point <= upper + 1e-12))

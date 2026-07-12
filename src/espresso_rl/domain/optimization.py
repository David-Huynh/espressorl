from __future__ import annotations


OPTIMIZER_MODE_CPBO = "cpbo"
DEFAULT_OPTIMIZER_MODE = OPTIMIZER_MODE_CPBO
VALID_OPTIMIZER_MODES = frozenset({OPTIMIZER_MODE_CPBO})
OPTIMIZER_MODE_ALIASES = {
    "preferential_bo": OPTIMIZER_MODE_CPBO,
    "consecutive_preferential_bo": OPTIMIZER_MODE_CPBO,
    # One-way configuration migration from the removed scalar BO modes.
    "bayesian_optimization": OPTIMIZER_MODE_CPBO,
    "bo": OPTIMIZER_MODE_CPBO,
}


def normalize_optimizer_mode(value: object) -> str:
    mode = str(value or DEFAULT_OPTIMIZER_MODE).strip().lower()
    mode = OPTIMIZER_MODE_ALIASES.get(mode, mode)
    if mode not in VALID_OPTIMIZER_MODES:
        raise ValueError("optimizer_mode must be 'cpbo'")
    return mode

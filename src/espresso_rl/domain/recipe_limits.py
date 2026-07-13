"""Broad integrity limits for physical recipe and shot data.

These limits reject malformed values at trust boundaries. An optimizer's
active recipe domain and trust region are narrower, configurable policies.
"""

RECIPE_DOMAIN_GRIND_RADIUS_MIN_STEPS = 0.1
RECIPE_DOMAIN_GRIND_RADIUS_MAX_STEPS = 1_000.0
RECIPE_DOMAIN_DOSE_MIN_G = 0.1
RECIPE_DOMAIN_DOSE_MAX_G = 100.0
RECIPE_DOMAIN_OUTPUT_MIN_G = 0.1
RECIPE_DOMAIN_OUTPUT_MAX_G = 1_000.0

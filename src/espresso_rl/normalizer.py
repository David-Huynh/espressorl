"""Compatibility wrapper for older imports.

New code should use espresso_rl.domain.grind and espresso_rl.domain.profile.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from espresso_rl.domain.events import ShotProfileEvent
from espresso_rl.domain.grind import GrindNormalizer
from espresso_rl.domain.profile import CHANNEL_NAMES, profile_mse, resample_profile

CHANNEL_KEYS = list(CHANNEL_NAMES)
N_CHANNELS = len(CHANNEL_KEYS)
N_POINTS = 100


class ProfileResampler:
    def __init__(self, n_points: int = N_POINTS) -> None:
        self.n_points = n_points

    def resample(self, payload: dict[str, Any]) -> np.ndarray:
        event = ShotProfileEvent(
            shot_id=str(payload.get("shot_id", "compat")),
            install_id=str(payload.get("install_id", "compat")),
            machine_id=str(payload.get("machine_id", "compat")),
            machine_adapter=str(payload.get("machine_adapter", "compat")),
            timestamp=int(payload.get("timestamp", 0)),
            time_ms=list(payload.get("time_ms", [])),
            pressure=list(payload.get("pressure", [])),
            target_pressure=list(payload.get("target_pressure", [])),
            flow=list(payload.get("flow", [])),
            target_flow=list(payload.get("target_flow", [])),
            weight=list(payload.get("weight", [])),
            grinder_step_size_um=float(payload.get("grinder_step_size_um", 10.0)),
            dose_in_g=float(payload.get("dose_in_g", 18.0)),
            target_yield_g=float(payload.get("target_yield_g", 36.0)),
        )
        return resample_profile(event, self.n_points)

    def compute_profile_mse(self, profile: np.ndarray) -> float:
        return profile_mse(profile)


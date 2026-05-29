from __future__ import annotations

import hashlib

import numpy as np

from .events import ShotProfileEvent
from .models import PROFILE_DTYPE, PROFILE_SHAPE

CHANNEL_NAMES = ("pressure", "target_pressure", "flow", "target_flow", "weight")


def resample_profile(event: ShotProfileEvent, n_points: int = PROFILE_SHAPE[1]) -> np.ndarray:
    """Resample a canonical shot profile event to the fixed Dreamer/BO shape."""
    time_ms = np.asarray(event.time_ms, dtype=np.float64)
    if len(time_ms) < 2:
        return np.zeros((len(CHANNEL_NAMES), n_points), dtype=PROFILE_DTYPE)
    if np.any(np.diff(time_ms) <= 0):
        raise ValueError("time_ms must be strictly increasing")

    t_uniform = np.linspace(time_ms[0], time_ms[-1], n_points)
    channels: list[np.ndarray] = []
    for name in CHANNEL_NAMES:
        raw = np.asarray(getattr(event, name), dtype=np.float64)
        channels.append(np.interp(t_uniform, time_ms, raw).astype(PROFILE_DTYPE))
    return np.stack(channels)


def profile_mse(profile: np.ndarray) -> float:
    profile = np.asarray(profile, dtype=PROFILE_DTYPE)
    if profile.shape != PROFILE_SHAPE:
        raise ValueError(f"profile must have shape {PROFILE_SHAPE}")
    pressure_mse = float(np.mean((profile[0] - profile[1]) ** 2))
    flow_mse = float(np.mean((profile[2] - profile[3]) ** 2))
    return (pressure_mse + flow_mse) / 2.0


def profile_score(profile: np.ndarray) -> float:
    return 1.0 / (1.0 + profile_mse(profile))


def profile_hash(profile: np.ndarray) -> str:
    profile = np.asarray(profile, dtype=PROFILE_DTYPE)
    return hashlib.sha256(profile.tobytes()).hexdigest()


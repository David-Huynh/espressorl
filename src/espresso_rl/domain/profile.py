from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .events import ShotProfileEvent
from .models import PROFILE_DTYPE, PROFILE_SHAPE

CHANNEL_NAMES = ("pressure", "target_pressure", "flow", "target_flow", "weight")
PRESSURE_RANGE = (0.0, 15.0)
FLOW_RANGE = (0.0, 20.0)
TARGET_ACTIVE_EPSILON = 1e-6


@dataclass(frozen=True)
class ProfileQuality:
    profile: np.ndarray
    flow_valid: bool
    flow_masked: bool


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


def sanitize_profile(profile: np.ndarray) -> ProfileQuality:
    """Mask untrusted flow channels before local storage/model input.

    Gaggimate may publish pump-derived flow telemetry that is not the same unit
    as beverage flow. Keep pressure/weight usable while preventing a corrupt
    flow pair from poisoning profile scoring or model input.
    """
    sanitized = np.asarray(profile, dtype=PROFILE_DTYPE).copy()
    if sanitized.shape != PROFILE_SHAPE:
        raise ValueError(f"profile must have shape {PROFILE_SHAPE}")

    flow_valid = _channel_values_valid(sanitized[2], *FLOW_RANGE)
    target_flow_valid = _channel_values_valid(sanitized[3], *FLOW_RANGE)
    flow_masked = not (flow_valid and target_flow_valid)
    if flow_masked:
        sanitized[2] = 0.0
        sanitized[3] = 0.0

    return ProfileQuality(
        profile=sanitized,
        flow_valid=flow_valid,
        flow_masked=flow_masked,
    )


def resample_profile_with_quality(
    event: ShotProfileEvent,
    n_points: int = PROFILE_SHAPE[1],
) -> ProfileQuality:
    return sanitize_profile(resample_profile(event, n_points=n_points))


def profile_mse(profile: np.ndarray) -> float:
    profile = np.asarray(profile, dtype=PROFILE_DTYPE)
    if profile.shape != PROFILE_SHAPE:
        raise ValueError(f"profile must have shape {PROFILE_SHAPE}")
    channel_mses: list[float] = []
    if _channel_pair_usable(profile[0], profile[1], PRESSURE_RANGE):
        channel_mses.append(float(np.mean((profile[0] - profile[1]) ** 2)))
    if _channel_pair_usable(profile[2], profile[3], FLOW_RANGE):
        channel_mses.append(float(np.mean((profile[2] - profile[3]) ** 2)))
    if not channel_mses:
        return 1.0
    return float(np.mean(channel_mses))


def profile_score(profile: np.ndarray) -> float:
    return 1.0 / (1.0 + profile_mse(profile))


def profile_hash(profile: np.ndarray) -> str:
    profile = np.asarray(profile, dtype=PROFILE_DTYPE)
    return hashlib.sha256(profile.tobytes()).hexdigest()


def _channel_pair_usable(
    actual: np.ndarray,
    target: np.ndarray,
    allowed_range: tuple[float, float],
) -> bool:
    minimum, maximum = allowed_range
    if not _channel_values_valid(actual, minimum, maximum):
        return False
    if not _channel_values_valid(target, minimum, maximum):
        return False
    return bool(np.any(np.abs(target) > TARGET_ACTIVE_EPSILON))


def _channel_values_valid(channel: np.ndarray, minimum: float, maximum: float) -> bool:
    return bool(np.all(np.isfinite(channel)) and np.all(channel >= minimum) and np.all(channel <= maximum))

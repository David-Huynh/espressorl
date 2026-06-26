from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .events import ShotProfileEvent
from .models import PROFILE_DTYPE, PROFILE_SHAPE

CHANNEL_NAMES = ("pressure", "target_pressure", "pump_flow", "target_flow", "weight")
PUMP_TARGET_MODE_SIMPLE = 0
PUMP_TARGET_MODE_PRESSURE = 1
PUMP_TARGET_MODE_FLOW = 2
PRESSURE_RANGE = (0.0, 15.0)
FLOW_RANGE = (0.0, 20.0)
TEMPERATURE_RANGE = (0.0, 160.0)
TARGET_ACTIVE_EPSILON = 1e-6


@dataclass(frozen=True)
class ProfileQuality:
    profile: np.ndarray
    flow_valid: bool
    flow_masked: bool


@dataclass(frozen=True)
class ResampledShotMetadata:
    beverage_flow_profile: np.ndarray | None
    temperature_profile: np.ndarray | None
    target_temperature_profile: np.ndarray | None
    pump_target_mode_profile: np.ndarray | None


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


def resample_shot_metadata(event: ShotProfileEvent, n_points: int = PROFILE_SHAPE[1]) -> ResampledShotMetadata:
    time_ms = np.asarray(event.time_ms, dtype=np.float64)
    beverage_flow_profile = _resample_optional_numeric_channel(
        time_ms,
        event.beverage_flow,
        n_points,
        allowed_range=FLOW_RANGE,
    )
    temperature_profile = _resample_optional_numeric_channel(
        time_ms,
        event.temperature,
        n_points,
        allowed_range=TEMPERATURE_RANGE,
    )
    target_temperature_profile = _resample_optional_numeric_channel(
        time_ms,
        event.target_temperature,
        n_points,
        allowed_range=TEMPERATURE_RANGE,
    )
    pump_target_mode_profile = _resample_optional_step_channel(time_ms, event.pump_target_mode, n_points)
    return ResampledShotMetadata(
        beverage_flow_profile=beverage_flow_profile,
        temperature_profile=temperature_profile,
        target_temperature_profile=target_temperature_profile,
        pump_target_mode_profile=pump_target_mode_profile,
    )


def sanitize_profile(profile: np.ndarray, *, force_flow_mask: bool = False) -> ProfileQuality:
    """Mask untrusted pump-flow channels before storage/model input.

    The fixed profile stores pump flow in ml/s so it can be compared with the
    pump-flow target. Beverage mass flow is retained separately and is never
    compared with the pump target.
    """
    sanitized = np.asarray(profile, dtype=PROFILE_DTYPE).copy()
    if sanitized.shape != PROFILE_SHAPE:
        raise ValueError(f"profile must have shape {PROFILE_SHAPE}")

    flow_valid = not force_flow_mask and _channel_values_valid(sanitized[2], *FLOW_RANGE)
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
    return sanitize_profile(
        resample_profile(event, n_points=n_points),
        force_flow_mask=event.pump_flow_calibration_required,
    )


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


def _resample_optional_numeric_channel(
    time_ms: np.ndarray,
    values: list[float] | None,
    n_points: int,
    *,
    allowed_range: tuple[float, float],
) -> np.ndarray | None:
    if values is None or len(time_ms) < 2:
        return None
    raw = np.asarray(values, dtype=np.float64)
    if len(raw) != len(time_ms) or np.any(np.diff(time_ms) <= 0):
        return None
    t_uniform = np.linspace(time_ms[0], time_ms[-1], n_points)
    resampled = np.interp(t_uniform, time_ms, raw).astype(PROFILE_DTYPE)
    if not _channel_values_valid(resampled, *allowed_range):
        return None
    return resampled


def _resample_optional_step_channel(
    time_ms: np.ndarray,
    values: list[int] | None,
    n_points: int,
) -> np.ndarray | None:
    if values is None or len(time_ms) < 2:
        return None
    raw = np.asarray(values, dtype=np.uint8)
    if len(raw) != len(time_ms) or np.any(np.diff(time_ms) <= 0):
        return None
    t_uniform = np.linspace(time_ms[0], time_ms[-1], n_points)
    indexes = np.searchsorted(time_ms, t_uniform, side="right") - 1
    indexes = np.clip(indexes, 0, len(raw) - 1)
    return raw[indexes].astype(np.uint8)

"""Infer coarse contact-style phases from a tracked object's motion."""

import math

import numpy as np
from scipy.signal import savgol_filter

from .schema import PhaseInterval, Trajectory2D


def _first_sustained(condition: np.ndarray, start: int, length: int) -> int | None:
    """Return the first index at which ``condition`` holds for ``length`` frames."""

    run_length = 0
    for index in range(start, len(condition)):
        run_length = run_length + 1 if condition[index] else 0
        if run_length >= length:
            return index - length + 1
    return None


def _phase_confidence(
    speed: np.ndarray, is_motion: bool, speed_on_px_s: float, speed_off_px_s: float
) -> float:
    observed_speed = float(np.mean(speed)) if len(speed) else 0.0
    if is_motion:
        separation = (observed_speed - speed_off_px_s) / max(speed_on_px_s, 1e-6)
    else:
        separation = (speed_on_px_s - observed_speed) / max(speed_on_px_s, 1e-6)
    return float(np.clip(separation, 0.0, 1.0))


def infer_motion_phases(
    trajectory: Trajectory2D,
    fps: float,
    speed_on_px_s: float = 8.0,
    speed_off_px_s: float = 3.0,
    min_phase_s: float = 0.3,
) -> tuple[PhaseInterval, ...]:
    """Segment a trajectory into approach, hold, release, and settle phases.

    Sustained speed above ``speed_on_px_s`` begins ``hold``; sustained speed
    below ``speed_off_px_s`` after that begins ``release``.  The initial
    minimum-duration portion of release is kept distinct from ``settle``.
    """

    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if not (0.0 <= speed_off_px_s < speed_on_px_s):
        raise ValueError("speed thresholds must satisfy 0 <= off < on")
    if min_phase_s <= 0.0:
        raise ValueError("min_phase_s must be positive")
    frame_count = len(trajectory.centers_px)
    if frame_count < 4:
        raise ValueError("trajectory must contain at least four frames")

    centers = trajectory.centers_px
    if frame_count >= 7:
        window = min(7, frame_count if frame_count % 2 else frame_count - 1)
        centers = savgol_filter(centers, window_length=window, polyorder=2, axis=0)
    speed = np.linalg.norm(np.gradient(centers, axis=0), axis=1) * fps
    minimum_frames = max(1, math.ceil(min_phase_s * fps))

    hold_start = _first_sustained(speed >= speed_on_px_s, 0, minimum_frames)
    if hold_start is None:
        hold_start = max(1, frame_count // 4)
    release_start = _first_sustained(
        speed <= speed_off_px_s, hold_start + minimum_frames, minimum_frames
    )
    if release_start is None:
        release_start = max(hold_start + 1, (3 * frame_count) // 4)
    release_start = min(release_start, frame_count - 2)
    settle_start = min(release_start + minimum_frames, frame_count - 1)

    approach_end = max(0, hold_start - 1)
    hold_end = max(hold_start, release_start - 1)
    release_end = max(release_start, settle_start - 1)
    if not (
        approach_end < hold_start <= hold_end < release_start <= release_end < settle_start
    ):
        raise ValueError("trajectory cannot form four non-empty, non-overlapping phases")

    return (
        PhaseInterval(
            "approach",
            0,
            approach_end,
            _phase_confidence(speed[: approach_end + 1], False, speed_on_px_s, speed_off_px_s),
            ("object_still",),
        ),
        PhaseInterval(
            "hold",
            hold_start,
            hold_end,
            _phase_confidence(
                speed[hold_start : hold_end + 1], True, speed_on_px_s, speed_off_px_s
            ),
            ("object_motion",),
        ),
        PhaseInterval(
            "release",
            release_start,
            release_end,
            _phase_confidence(
                speed[release_start : release_end + 1], False, speed_on_px_s, speed_off_px_s
            ),
            ("object_still",),
        ),
        PhaseInterval(
            "settle",
            settle_start,
            frame_count - 1,
            _phase_confidence(speed[settle_start:], False, speed_on_px_s, speed_off_px_s),
            ("object_settled",),
        ),
    )

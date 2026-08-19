"""Validated data contracts shared across the video-to-data pipeline."""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


class RunStatus(str, Enum):
    """Terminal status of an auditable pipeline run."""

    COMPLETED = "completed"
    NOT_RUN = "not_run"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True)
class VideoMetadata:
    """Basic media facts measured from a source video."""

    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float


@dataclass(frozen=True)
class PhaseInterval:
    """A labeled, inclusive interval of video frames."""

    label: str
    start_frame: int
    end_frame: int
    confidence: float
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class Trajectory2D:
    """Per-frame 2D centers and confidences indexed by timestamps."""

    timestamps_s: NDArray[np.float64] | Sequence[float]
    centers_px: NDArray[np.float64] | Sequence[Sequence[float]]
    confidence: NDArray[np.float64] | Sequence[float]

    def __post_init__(self) -> None:
        timestamps_s = np.asarray(self.timestamps_s, dtype=float)
        centers_px = np.asarray(self.centers_px, dtype=float)
        confidence = np.asarray(self.confidence, dtype=float)

        if timestamps_s.ndim != 1:
            raise ValueError("timestamps_s must have shape [T]")
        if centers_px.ndim != 2 or centers_px.shape[1:] != (2,):
            raise ValueError("centers_px must have shape [T, 2]")
        if confidence.ndim != 1:
            raise ValueError("confidence must have shape [T]")
        if not (
            len(timestamps_s) == len(centers_px) == len(confidence)
        ):
            raise ValueError("timestamps_s, centers_px, and confidence must have the same length")
        if not all(
            np.isfinite(values).all()
            for values in (timestamps_s, centers_px, confidence)
        ):
            raise ValueError("trajectory values must be finite")
        if np.any(np.diff(timestamps_s) < 0.0):
            raise ValueError("timestamps_s must be monotonic")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("confidence must be in [0, 1]")

        object.__setattr__(self, "timestamps_s", timestamps_s)
        object.__setattr__(self, "centers_px", centers_px)
        object.__setattr__(self, "confidence", confidence)

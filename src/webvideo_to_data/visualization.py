"""Diagnostic video overlays for lightweight object tracking."""

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .schema import PhaseInterval, Trajectory2D


_PHASE_COLORS = {
    "approach": (0, 220, 255),
    "hold": (0, 190, 0),
    "release": (0, 100, 255),
    "settle": (255, 190, 0),
}


def _phase_for_frame(phases: Sequence[PhaseInterval], frame_index: int) -> PhaseInterval | None:
    return next(
        (
            phase
            for phase in phases
            if phase.start_frame <= frame_index <= phase.end_frame
        ),
        None,
    )


def render_tracking_overlay(
    video_path: str | Path,
    trajectory: Trajectory2D,
    phases: Sequence[PhaseInterval],
    output_path: str | Path,
    *,
    roi_size: Sequence[float] | None = None,
) -> Path:
    """Render a readable tracking diagnostic video and return its path.

    ``roi_size`` is ``[width, height]``.  When it is omitted, a compact
    16-pixel square is used around each tracked center.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    roi = np.asarray(roi_size if roi_size is not None else (16.0, 16.0), dtype=float)
    if roi.shape != (2,) or np.any(roi <= 0.0):
        raise ValueError("roi_size must be [width, height] with positive values")

    capture = cv2.VideoCapture(str(video_path))
    writer: cv2.VideoWriter | None = None
    try:
        if not capture.isOpened():
            raise ValueError("video cannot be opened")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count != len(trajectory.centers_px):
            raise ValueError("trajectory frame count must match source video")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0:
            raise ValueError("video fps must be positive")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = "mp4v" if output.suffix.lower() in {".mp4", ".m4v"} else "MJPG"
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if not writer.isOpened():
            raise ValueError("overlay video writer cannot be opened")

        trail: list[tuple[int, int]] = []
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise ValueError("source video ended before its declared frame count")
            center = tuple(np.rint(trajectory.centers_px[frame_index]).astype(int))
            phase = _phase_for_frame(phases, frame_index)
            label = phase.label if phase is not None else "unclassified"
            color = _PHASE_COLORS.get(label, (220, 220, 220))

            x0 = int(round(center[0] - roi[0] / 2.0))
            y0 = int(round(center[1] - roi[1] / 2.0))
            x1 = int(round(center[0] + roi[0] / 2.0))
            y1 = int(round(center[1] + roi[1] / 2.0))
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
            trail.append(center)
            for start, end in zip(trail[-30:-1], trail[-29:]):
                cv2.line(frame, start, end, color, 1, cv2.LINE_AA)
            cv2.circle(frame, center, 3, color, -1, cv2.LINE_AA)

            confidence = float(trajectory.confidence[frame_index])
            bar_width = 50
            cv2.rectangle(frame, (5, height - 12), (5 + bar_width, height - 6), (40, 40, 40), -1)
            cv2.rectangle(
                frame,
                (5, height - 12),
                (5 + round(bar_width * confidence), height - 6),
                color,
                -1,
            )
            cv2.putText(frame, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{trajectory.timestamps_s[frame_index]:.2f}s",
                (5, 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    return output

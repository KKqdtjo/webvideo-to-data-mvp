"""Honest, aspect-preserving diagnostic media composition."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray

from .config import MediaConfig
from .schema import PhaseInterval, Trajectory2D


_PHASE_COLORS = {
    "approach": (0, 220, 255),
    "hold": (0, 190, 0),
    "release": (0, 100, 255),
    "settle": (255, 190, 0),
}
_COMPARISON_DURATION_TOLERANCE_S = 0.001


@dataclass(frozen=True)
class MediaLabels:
    """Terminal labels that prevent diagnostics from resembling action evidence."""

    status: str
    mode: str
    metric_warning: str
    time_alignment: str


def labels_for_metrics(metrics: Mapping[str, object]) -> MediaLabels:
    """Derive conservative display labels from trusted terminal metrics."""

    status = str(metrics.get("status", "unknown")).strip().lower()
    variant = str(metrics.get("variant", "unknown")).strip().upper()
    status_label = {
        "rejected": "REJECTED — NOT ACTION DATA",
        "failed": "FAILED — NOT ACTION DATA",
        "not_run": "NOT RUN — NOT ACTION DATA",
    }.get(status, "UNVERIFIED — NOT ACTION DATA")
    mode = {
        "B0": "MANUAL PHYSICS BASELINE",
        "B1": "KINEMATIC OBJECT-POSE OVERRIDE",
    }.get(variant, "METRIC DEPTH UNAVAILABLE")
    warning = (
        "availability != semantic accuracy"
        if variant == "B1"
        else (
            "manual baseline != video-grounded action"
            if variant == "B0"
            else "metric depth not available"
        )
    )
    return MediaLabels(
        status=status_label,
        mode=mode,
        metric_warning=warning,
        time_alignment="TIME-WARPED FOR COMPARISON",
    )


def letterbox_frame(
    frame: NDArray[np.uint8],
    panel_size: tuple[int, int],
    color: tuple[int, int, int],
) -> NDArray[np.uint8]:
    """Fit one BGR frame inside ``(width, height)`` without stretching it."""

    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or min(frame.shape[:2]) <= 0
    ):
        raise ValueError("frame must be a nonempty uint8 BGR image")
    panel_width, panel_height = panel_size
    if min(panel_width, panel_height) <= 0:
        raise ValueError("panel size must be positive")
    if len(color) != 3 or any(type(value) is not int or not 0 <= value <= 255 for value in color):
        raise ValueError("letterbox color must contain three bytes")
    scale = min(panel_width / frame.shape[1], panel_height / frame.shape[0])
    width = max(1, round(frame.shape[1] * scale))
    height = max(1, round(frame.shape[0] * scale))
    interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (width, height), interpolation=interpolation)
    canvas = np.full((panel_height, panel_width, 3), color, dtype=np.uint8)
    x = (panel_width - width) // 2
    y = (panel_height - height) // 2
    canvas[y : y + height, x : x + width] = resized
    return canvas


def _fit_scale(text: str, maximum_width: int, preferred: float) -> float:
    scale = preferred
    minimum = 0.08
    while scale >= minimum:
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if width <= maximum_width:
            return scale
        scale -= 0.02
    return minimum


def _draw_status_text(frame: NDArray[np.uint8], text: str, y: int) -> None:
    """Draw the required Unicode em dash as a real line between ASCII segments."""

    left, separator, right = text.partition(" — ")
    ascii_measure = f"{left} - {right}" if separator else left
    scale = _fit_scale(ascii_measure, frame.shape[1] - 24, 0.72)
    thickness = max(1, round(scale * 2.0))
    x = 12
    cv2.putText(
        frame, left, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
        (255, 255, 255), thickness, cv2.LINE_AA,
    )
    if not separator:
        return
    left_width = cv2.getTextSize(left, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    dash_x0 = x + left_width + max(5, round(7 * scale))
    dash_width = max(8, round(16 * scale))
    dash_y = y - max(3, round(5 * scale))
    cv2.line(
        frame,
        (dash_x0, dash_y),
        (dash_x0 + dash_width, dash_y),
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        right,
        (dash_x0 + dash_width + max(5, round(7 * scale)), y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _label_band_heights(height: int) -> tuple[int, int]:
    return (
        min(height, max(22, min(70, round(height * 0.22)))),
        min(height, max(14, min(38, round(height * 0.12)))),
    )


def _compose_status_banner(
    frame: NDArray[np.uint8],
    labels: MediaLabels,
    *,
    source_time_s: float | None = None,
    simulation_time_s: float | None = None,
    top_height: int,
    clock_height: int,
) -> NDArray[np.uint8]:
    output = frame.copy()
    height, width = output.shape[:2]
    cv2.rectangle(output, (0, 0), (width, top_height), (26, 26, 150), -1)
    _draw_status_text(
        output,
        labels.status,
        min(27, max(8, round(top_height * 0.45))),
    )
    detail = f"{labels.mode} · {labels.metric_warning}"
    detail_ascii = detail.replace("·", "|")
    detail_scale = _fit_scale(detail_ascii, width - 24, 0.42)
    cv2.putText(
        output,
        detail_ascii,
        (12, top_height - max(4, round(top_height * 0.12))),
        cv2.FONT_HERSHEY_SIMPLEX,
        detail_scale,
        (238, 238, 238),
        1,
        cv2.LINE_AA,
    )
    clock_parts: list[str] = []
    if source_time_s is not None:
        clock_parts.append(f"VIDEO t={source_time_s:.2f}s")
    if simulation_time_s is not None:
        clock_parts.append(f"SIM t={simulation_time_s:.2f}s")
    if (
        source_time_s is not None
        and simulation_time_s is not None
        and labels.time_alignment
    ):
        clock_parts.append(labels.time_alignment)
    if clock_parts:
        clock = "  |  ".join(clock_parts)
        cv2.rectangle(
            output,
            (0, height - clock_height),
            (width, height),
            (20, 20, 20),
            -1,
        )
        clock_scale = _fit_scale(clock, width - 24, 0.5)
        cv2.putText(
            output,
            clock,
            (12, height - max(8, round(clock_height * 0.25))),
            cv2.FONT_HERSHEY_SIMPLEX,
            clock_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def compose_status_banner(
    frame: NDArray[np.uint8],
    labels: MediaLabels,
    *,
    source_time_s: float | None = None,
    simulation_time_s: float | None = None,
) -> NDArray[np.uint8]:
    """Overlay permanent terminal status, mode, warning, and readable clocks."""

    top_height, clock_height = _label_band_heights(frame.shape[0])
    return _compose_status_banner(
        frame,
        labels,
        source_time_s=source_time_s,
        simulation_time_s=simulation_time_s,
        top_height=top_height,
        clock_height=clock_height,
    )


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
    media_labels: MediaLabels | None = None,
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
            if media_labels is not None:
                frame = compose_status_banner(
                    frame,
                    media_labels,
                    source_time_s=float(trajectory.timestamps_s[frame_index]),
                )
            writer.write(frame)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    return output


def _video_frames(path: Path) -> tuple[list[NDArray[np.uint8]], float]:
    capture = cv2.VideoCapture(str(path))
    frames: list[NDArray[np.uint8]] = []
    try:
        if not capture.isOpened():
            raise ValueError("video cannot be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("video fps must be positive")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise ValueError("video contains an empty frame")
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError("video contains no decodable frames")
    return frames, fps


def _sample_frames(
    frames: Sequence[NDArray[np.uint8]],
    source_fps: float,
    *,
    fps: int,
    maximum_duration_s: float,
) -> list[tuple[NDArray[np.uint8], float]]:
    if type(fps) is not int or fps <= 0:
        raise ValueError("GIF fps must be a positive integer")
    if not np.isfinite(maximum_duration_s) or maximum_duration_s <= 0.0:
        raise ValueError("maximum GIF duration must be positive")
    source_duration = len(frames) / source_fps
    duration = min(source_duration, maximum_duration_s)
    count = max(1, min(len(frames), int(round(duration * fps))))
    samples: list[tuple[NDArray[np.uint8], float]] = []
    for output_index in range(count):
        timestamp = output_index / fps
        source_index = min(len(frames) - 1, round(timestamp * source_fps))
        samples.append((frames[source_index], timestamp))
    return samples


def _write_gif(
    frames_bgr: Sequence[NDArray[np.uint8]], output: Path, *, fps: int
) -> Path:
    if not frames_bgr:
        raise ValueError("cannot write a GIF without frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        output,
        [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr],
        format="GIF",
        duration=1000 / fps,
        loop=0,
        subrectangles=False,
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise ValueError("GIF writer produced no output")
    return output


def render_comparison_video(
    source_path: str | Path,
    overlay_path: str | Path,
    simulation_frames: Sequence[NDArray[np.uint8]] | NDArray[np.uint8],
    output_path: str | Path,
    media_config: MediaConfig,
    metrics: Mapping[str, object],
    source_duration_s: float,
    simulation_duration_s: float,
) -> Path:
    """Render a labeled source/tracking/simulation comparison without stretching."""

    if len(simulation_frames) == 0:
        raise ValueError("comparison requires simulation frames")
    if min(source_duration_s, simulation_duration_s) <= 0.0:
        raise ValueError("comparison durations must be positive")
    source_capture = cv2.VideoCapture(str(source_path))
    overlay_capture = cv2.VideoCapture(str(overlay_path))
    writer: cv2.VideoWriter | None = None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not source_capture.isOpened() or not overlay_capture.isOpened():
            raise ValueError("source and overlay videos must be readable")
        source_count = int(source_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        overlay_count = int(overlay_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if source_count <= 0 or overlay_count != source_count:
            raise ValueError("source and overlay frame counts must match")
        source_fps = float(source_capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(source_fps) or source_fps <= 0.0:
            raise ValueError("source video fps must be positive")
        canvas_width, canvas_height = media_config.canvas_size
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            media_config.output_fps,
            (canvas_width, canvas_height),
        )
        if not writer.isOpened():
            raise ValueError("comparison video writer cannot be opened")
        labels = labels_for_metrics(metrics)
        if (
            abs(source_duration_s - simulation_duration_s)
            <= _COMPARISON_DURATION_TOLERANCE_S
        ):
            labels = replace(labels, time_alignment="")
        simulation_fps = (
            (len(simulation_frames) - 1) / simulation_duration_s
            if len(simulation_frames) > 1
            else None
        )
        output_count = max(1, round(source_duration_s * media_config.output_fps))
        for output_index in range(output_count):
            fraction = output_index / max(1, output_count - 1)
            source_index = min(source_count - 1, round(fraction * (source_count - 1)))
            source_capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            overlay_capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            source_ok, source_frame = source_capture.read()
            overlay_ok, overlay_frame = overlay_capture.read()
            if not source_ok or not overlay_ok:
                raise ValueError("source or overlay ended before declared frame count")
            simulation_index = min(
                len(simulation_frames) - 1,
                round(fraction * (len(simulation_frames) - 1)),
            )
            simulation_frame = cv2.cvtColor(
                np.asarray(simulation_frames[simulation_index]), cv2.COLOR_RGB2BGR
            )
            panels = [
                letterbox_frame(
                    np.asarray(panel),
                    media_config.panel_size,
                    media_config.letterbox_bgr,
                )
                for panel in (source_frame, overlay_frame, simulation_frame)
            ]
            for panel, name in zip(panels, ("SOURCE", "TRACKING", "SIMULATION")):
                cv2.putText(
                    panel,
                    name,
                    (8, min(panel.shape[0] - 8, 78)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            assembled = np.hstack(panels)
            canvas = letterbox_frame(
                assembled,
                media_config.canvas_size,
                media_config.letterbox_bgr,
            )
            writer.write(
                compose_status_banner(
                    canvas,
                    labels,
                    source_time_s=source_index / source_fps,
                    simulation_time_s=(
                        simulation_index / simulation_fps
                        if simulation_fps is not None
                        else 0.0
                    ),
                )
            )
    finally:
        source_capture.release()
        overlay_capture.release()
        if writer is not None:
            writer.release()
    return output


def render_preview_gif(
    comparison_path: str | Path,
    output_path: str | Path,
    width_px: int = 960,
    fps: int = 8,
    maximum_duration_s: float = 12.0,
) -> Path:
    """Convert labeled comparison frames to a GitHub-inline GIF."""

    if type(width_px) is not int or width_px <= 0:
        raise ValueError("GIF width must be positive")
    frames, source_fps = _video_frames(Path(comparison_path))
    resized: list[NDArray[np.uint8]] = []
    for frame, _ in _sample_frames(
        frames, source_fps, fps=fps, maximum_duration_s=maximum_duration_s
    ):
        height = max(1, round(frame.shape[0] * width_px / frame.shape[1]))
        interpolation = cv2.INTER_AREA if width_px <= frame.shape[1] else cv2.INTER_LINEAR
        resized.append(cv2.resize(frame, (width_px, height), interpolation=interpolation))
    return _write_gif(resized, Path(output_path), fps=fps)


def render_public_simulation_preview(
    simulation_path: str | Path,
    output_path: str | Path,
    labels: MediaLabels,
    media_config: MediaConfig,
) -> Path:
    """Render a labeled, simulation-only GIF suitable for a public README."""

    frames, source_fps = _video_frames(Path(simulation_path))
    sampled = _sample_frames(frames, source_fps, fps=8, maximum_duration_s=12.0)
    output_frames: list[NDArray[np.uint8]] = []
    width = media_config.canvas_size[0]
    for frame, timestamp in sampled:
        height = max(1, round(frame.shape[0] * width / frame.shape[1]))
        resized = cv2.resize(
            frame,
            (width, height),
            interpolation=cv2.INTER_AREA if width <= frame.shape[1] else cv2.INTER_LINEAR,
        )
        scale = width / frame.shape[1]
        source_top, source_clock = _label_band_heights(frame.shape[0])
        top_height = min(
            resized.shape[0] - 1,
            max(1, int(np.ceil((source_top + 1) * scale)) - 1),
        )
        clock_height = min(
            resized.shape[0],
            max(1, int(np.ceil(source_clock * scale))),
        )
        output_frames.append(
            _compose_status_banner(
                resized,
                labels,
                simulation_time_s=timestamp,
                top_height=top_height,
                clock_height=clock_height,
            )
        )
    return _write_gif(output_frames, Path(output_path), fps=8)

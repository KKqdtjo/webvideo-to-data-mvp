from pathlib import Path

import cv2
import numpy as np

from webvideo_to_data.schema import PhaseInterval, Trajectory2D
from webvideo_to_data.visualization import render_tracking_overlay


def _write_input_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
    )
    assert writer.isOpened()
    try:
        for _ in range(5):
            writer.write(np.full((48, 64, 3), 40, dtype=np.uint8))
    finally:
        writer.release()


def test_renders_readable_overlay_at_tracked_center(tmp_path: Path) -> None:
    """Removing the center marker would leave the tracked neighborhood unchanged."""

    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "overlay.avi"
    _write_input_video(input_path)
    trajectory = Trajectory2D(
        timestamps_s=np.arange(5, dtype=float) / 10.0,
        centers_px=np.array([[20 + index, 24] for index in range(5)], dtype=float),
        confidence=np.full(5, 0.9),
    )
    phases = (
        PhaseInterval("approach", 0, 1, 0.8, ("object_still",)),
        PhaseInterval("hold", 2, 2, 0.8, ("object_motion",)),
        PhaseInterval("release", 3, 3, 0.8, ("object_still",)),
        PhaseInterval("settle", 4, 4, 0.8, ("object_settled",)),
    )

    returned_path = render_tracking_overlay(
        input_path, trajectory, phases, output_path, roi_size=[16, 20]
    )

    assert returned_path == output_path
    assert output_path.is_file()
    input_capture = cv2.VideoCapture(str(input_path))
    output_capture = cv2.VideoCapture(str(output_path))
    try:
        ok_input, first_input = input_capture.read()
        ok_output, first_output = output_capture.read()
        readable_frames = int(ok_output)
        while output_capture.read()[0]:
            readable_frames += 1
    finally:
        input_capture.release()
        output_capture.release()
    assert ok_input and ok_output
    assert readable_frames == 5
    assert not np.array_equal(first_input[22:27, 18:23], first_output[22:27, 18:23])

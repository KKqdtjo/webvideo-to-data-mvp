from pathlib import Path

import cv2
import numpy as np
import pytest

from webvideo_to_data.tracking import track_roi_lk


def _write_moving_rectangle_video(path: Path) -> None:
    """Write a deterministic textured rectangle translating two pixels/frame."""

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 72)
    )
    assert writer.isOpened()
    try:
        for frame_index in range(20):
            frame = np.full((72, 96, 3), 24, dtype=np.uint8)
            x = 10 + 2 * frame_index
            cv2.rectangle(frame, (x, 20), (x + 15, 39), (255, 0, 0), -1)
            for offset_x, offset_y in ((3, 3), (11, 4), (5, 13), (12, 15)):
                cv2.circle(frame, (x + offset_x, 20 + offset_y), 1, (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def test_tracks_deterministic_textured_rectangle(tmp_path: Path) -> None:
    """A missing LK tracker would lose the known 38-pixel translation."""

    video_path = tmp_path / "moving.avi"
    _write_moving_rectangle_video(video_path)

    trajectory = track_roi_lk(video_path, [10, 20, 16, 20])

    displacement_x = trajectory.centers_px[-1, 0] - trajectory.centers_px[0, 0]
    assert displacement_x == pytest.approx(38.0, abs=3.0)
    assert np.count_nonzero(trajectory.confidence > 0.0) >= 18
    assert np.all(np.diff(trajectory.timestamps_s) >= 0.0)

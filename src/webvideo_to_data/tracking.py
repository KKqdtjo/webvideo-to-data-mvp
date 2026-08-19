"""Lightweight Lucas-Kanade tracking for a configured object ROI."""

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .schema import Trajectory2D


_FEATURE_PARAMS = {
    "maxCorners": 32,
    "qualityLevel": 0.01,
    "minDistance": 3,
    "blockSize": 3,
}
_LK_PARAMS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    ),
}


def _features_in_roi(gray: np.ndarray, roi: np.ndarray) -> np.ndarray:
    x, y, width, height = np.rint(roi).astype(int)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[y : y + height, x : x + width] = 255
    points = cv2.goodFeaturesToTrack(gray, mask=mask, **_FEATURE_PARAMS)
    if points is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return points.astype(np.float32)


def _inside_frame(roi: np.ndarray, frame_width: int, frame_height: int) -> bool:
    x, y, width, height = roi
    return (
        width > 0
        and height > 0
        and x >= 0
        and y >= 0
        and x + width <= frame_width
        and y + height <= frame_height
    )


def track_roi_lk(
    video_path: str | Path,
    initial_roi: Sequence[float],
    *,
    forward_backward_threshold_px: float = 1.5,
    minimum_live_points: int = 8,
) -> Trajectory2D:
    """Track an object's ROI center using forward-backward Lucas-Kanade flow.

    The supplied ROI is ``[x, y, width, height]`` in the first frame.  A
    ``ValueError`` is raised when that ROI is not usable for tracking.
    """

    roi = np.asarray(initial_roi, dtype=float)
    if roi.shape != (4,):
        raise ValueError("initial_roi must have shape [x, y, w, h]")
    if forward_backward_threshold_px < 0.0:
        raise ValueError("forward_backward_threshold_px must be nonnegative")
    if minimum_live_points <= 0:
        raise ValueError("minimum_live_points must be positive")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError("video cannot be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0:
            raise ValueError("video fps must be positive")

        ok, first_frame = capture.read()
        if not ok:
            raise ValueError("video contains no readable frames")
        frame_height, frame_width = first_frame.shape[:2]
        if not _inside_frame(roi, frame_width, frame_height):
            raise ValueError("initial_roi must be inside the first frame")

        previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        points = _features_in_roi(previous_gray, roi)
        if len(points) < 4:
            raise ValueError("initial_roi must contain at least four features")

        center = roi[:2] + roi[2:] / 2.0
        centers = [center.copy()]
        timestamps = [0.0]
        confidences = [min(1.0, len(points) / 24.0)]

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            points_before = len(points)
            next_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
                previous_gray, current_gray, points, None, **_LK_PARAMS
            )
            if next_points is None or forward_status is None:
                valid_previous = np.empty((0, 1, 2), dtype=np.float32)
                valid_next = valid_previous.copy()
                forward_fraction = 0.0
            else:
                backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                    current_gray, previous_gray, next_points, None, **_LK_PARAMS
                )
                if backward_points is None or backward_status is None:
                    valid_previous = np.empty((0, 1, 2), dtype=np.float32)
                    valid_next = valid_previous.copy()
                    forward_fraction = 0.0
                else:
                    forward_valid = forward_status.reshape(-1).astype(bool)
                    backward_valid = backward_status.reshape(-1).astype(bool)
                    backward_error = np.linalg.norm(
                        points.reshape(-1, 2) - backward_points.reshape(-1, 2), axis=1
                    )
                    valid_mask = forward_valid & backward_valid & (
                        backward_error < forward_backward_threshold_px
                    )
                    valid_previous = points[valid_mask]
                    valid_next = next_points[valid_mask]
                    forward_fraction = (
                        float(np.count_nonzero(valid_mask)) / points_before
                        if points_before
                        else 0.0
                    )

            valid_count = len(valid_next)
            if valid_count:
                displacement = np.median(
                    valid_next.reshape(-1, 2) - valid_previous.reshape(-1, 2), axis=0
                )
                center = center + displacement
                roi[:2] = center - roi[2:] / 2.0
                points = valid_next.reshape(-1, 1, 2)
            else:
                points = np.empty((0, 1, 2), dtype=np.float32)

            if (
                _inside_frame(roi, frame_width, frame_height)
                and len(points) < minimum_live_points
            ):
                detected = _features_in_roi(current_gray, roi)
                if len(detected):
                    points = detected

            centers.append(center.copy())
            timestamps.append(frame_index / fps)
            confidences.append(min(1.0, valid_count / 24.0) * forward_fraction)
            previous_gray = current_gray
    finally:
        capture.release()

    return Trajectory2D(
        timestamps_s=np.asarray(timestamps, dtype=float),
        centers_px=np.asarray(centers, dtype=float),
        confidence=np.asarray(confidences, dtype=float),
    )

from pathlib import Path

import cv2
import numpy as np
import pytest

import webvideo_to_data.visualization as visualization
from webvideo_to_data.config import MediaConfig
from webvideo_to_data.schema import PhaseInterval, Trajectory2D
from webvideo_to_data.visualization import (
    compose_status_banner,
    labels_for_metrics,
    letterbox_frame,
    render_comparison_video,
    render_preview_gif,
    render_public_simulation_preview,
    render_tracking_overlay,
)


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


def test_letterbox_preserves_vertical_source_aspect_ratio() -> None:
    """Replacing letterboxing with a direct square resize would erase the side bars."""

    frame = np.zeros((120, 60, 3), dtype=np.uint8)
    frame[:, :, 1] = 255

    output = letterbox_frame(frame, (120, 120), (16, 16, 16))

    assert output.shape == (120, 120, 3)
    assert np.all(output[:, :30] == 16)
    assert np.all(output[:, 90:] == 16)
    assert np.all(output[:, 30:90, 1] == 255)


def test_rejected_b1_media_labels_cannot_look_like_action_success() -> None:
    """A generic success caption would misrepresent a kinematic diagnostic as actions."""

    labels = labels_for_metrics({"status": "rejected", "variant": "B1"})

    assert labels.status == "REJECTED — NOT ACTION DATA"
    assert labels.mode == "KINEMATIC OBJECT-POSE OVERRIDE"
    assert labels.metric_warning == "availability != semantic accuracy"


def test_small_simulation_banner_keeps_detail_text_inside_frame() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    output = compose_status_banner(
        frame,
        labels_for_metrics({"status": "rejected", "variant": "B0"}),
        simulation_time_s=0.0,
    )

    assert np.all(output[35:52, -5:] == np.array([26, 26, 150], dtype=np.uint8))


def test_tiny_diagnostic_frame_retains_visible_scene_between_label_bands() -> None:
    frame = np.full((72, 96, 3), 210, dtype=np.uint8)

    output = compose_status_banner(
        frame,
        labels_for_metrics({"status": "rejected", "variant": "B1"}),
        source_time_s=1.0,
    )

    assert np.all(output[24:56] == 210)


def _write_labeled_comparison_fixture(
    path: Path, *, size: tuple[int, int], frames: int, fps: float
) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size
    )
    assert writer.isOpened()
    labels = labels_for_metrics({"status": "rejected", "variant": "B1"})
    try:
        width, height = size
        for index in range(frames):
            frame = np.full((height, width, 3), 24, dtype=np.uint8)
            cv2.circle(frame, (40 + index * 12, height // 2), 18, (0, 180, 255), -1)
            writer.write(
                compose_status_banner(
                    frame,
                    labels,
                    source_time_s=index / fps,
                    simulation_time_s=index / (fps * 2.0),
                )
            )
    finally:
        writer.release()
    return path


def _decode_all_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        assert capture.isOpened()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame is not None and frame.size
            frames.append(frame)
    finally:
        capture.release()
    return frames


def test_preview_gif_preserves_comparison_aspect_and_frame_count(
    tmp_path: Path,
) -> None:
    """Sampling at the source FPS must not flatten the panorama or drop changing frames."""

    comparison = _write_labeled_comparison_fixture(
        tmp_path / "comparison.avi", size=(960, 320), frames=24, fps=8
    )

    output = render_preview_gif(comparison, tmp_path / "preview.gif", fps=8)
    frames = _decode_all_frames(output)

    assert len(frames) == 24
    assert frames[0].shape[:2] == (320, 960)
    assert np.mean(
        np.abs(frames[0].astype(float) - frames[-1].astype(float))
    ) > 1.0


def test_public_preview_replaces_mismatched_existing_labels(
    tmp_path: Path,
) -> None:
    simulation = _write_labeled_comparison_fixture(
        tmp_path / "simulation.avi", size=(320, 240), frames=3, fps=8
    )
    media = MediaConfig(
        canvas_size=(960, 320),
        panel_size=(320, 320),
        letterbox_bgr=(16, 16, 16),
        output_fps=30.0,
        comparison_alignment="time_warped_for_comparison",
    )
    output = render_public_simulation_preview(
        simulation,
        tmp_path / "public.gif",
        labels_for_metrics({"status": "rejected", "variant": "B0"}),
        media,
    )
    source_frame = _decode_all_frames(simulation)[0]
    preview_frame = _decode_all_frames(output)[0]
    expected = cv2.resize(source_frame, (960, 720), interpolation=cv2.INTER_LINEAR)

    assert np.mean(
        np.abs(expected[:160].astype(float) - preview_frame[:160].astype(float))
    ) > 10.0


def test_public_preview_labels_solid_red_unlabeled_video(tmp_path: Path) -> None:
    simulation = tmp_path / "solid-red.avi"
    writer = cv2.VideoWriter(
        str(simulation), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (320, 240)
    )
    assert writer.isOpened()
    try:
        for _ in range(3):
            writer.write(np.full((240, 320, 3), (0, 0, 255), dtype=np.uint8))
    finally:
        writer.release()
    media = MediaConfig(
        canvas_size=(960, 320),
        panel_size=(320, 320),
        letterbox_bgr=(16, 16, 16),
        output_fps=30.0,
        comparison_alignment="time_warped_for_comparison",
    )

    output = render_public_simulation_preview(
        simulation,
        tmp_path / "public.gif",
        labels_for_metrics({"status": "rejected", "variant": "B0"}),
        media,
    )
    frame = _decode_all_frames(output)[0]

    assert np.all(frame[-5, 5] < 60)
    assert np.count_nonzero(np.all(frame[-100:] > 180, axis=2)) > 20


def _write_clock_fixture(path: Path, *, frames: int = 10, fps: float = 10.0) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48)
    )
    assert writer.isOpened()
    try:
        for index in range(frames):
            writer.write(np.full((48, 64, 3), 20 + index, dtype=np.uint8))
    finally:
        writer.release()
    return path


def _clock_test_media() -> MediaConfig:
    return MediaConfig(
        canvas_size=(192, 96),
        panel_size=(64, 64),
        letterbox_bgr=(16, 16, 16),
        output_fps=5.0,
        comparison_alignment="time_warped_for_comparison",
    )


def test_comparison_clocks_use_selected_frame_timestamps_and_equal_duration_is_not_warped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_clock_fixture(tmp_path / "source.avi")
    overlay = _write_clock_fixture(tmp_path / "overlay.avi")
    simulation = np.full((6, 48, 64, 3), 90, dtype=np.uint8)
    observed: list[tuple[float | None, float | None, str]] = []
    original = visualization.compose_status_banner

    def capture_clocks(
        frame: np.ndarray,
        labels: visualization.MediaLabels,
        *,
        source_time_s: float | None = None,
        simulation_time_s: float | None = None,
    ) -> np.ndarray:
        observed.append((source_time_s, simulation_time_s, labels.time_alignment))
        return original(
            frame,
            labels,
            source_time_s=source_time_s,
            simulation_time_s=simulation_time_s,
        )

    monkeypatch.setattr(visualization, "compose_status_banner", capture_clocks)
    render_comparison_video(
        source,
        overlay,
        simulation,
        tmp_path / "comparison.mp4",
        _clock_test_media(),
        {"status": "rejected", "variant": "B0"},
        source_duration_s=1.0,
        simulation_duration_s=1.0,
    )

    assert observed[-1][0] == pytest.approx(0.9)
    assert observed[-1][1] == pytest.approx(1.0)
    assert {alignment for _, _, alignment in observed} == {""}


@pytest.mark.parametrize(
    ("simulation_duration_s", "expected_alignment"),
    [
        (1.0005, ""),
        (1.0011, "TIME-WARPED FOR COMPARISON"),
    ],
)
def test_comparison_time_warp_label_has_one_millisecond_tolerance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    simulation_duration_s: float,
    expected_alignment: str,
) -> None:
    source = _write_clock_fixture(tmp_path / "source.avi")
    overlay = _write_clock_fixture(tmp_path / "overlay.avi")
    simulation = np.full((6, 48, 64, 3), 90, dtype=np.uint8)
    observed: list[str] = []
    original = visualization.compose_status_banner

    def capture_alignment(
        frame: np.ndarray,
        labels: visualization.MediaLabels,
        **clocks: float | None,
    ) -> np.ndarray:
        observed.append(labels.time_alignment)
        return original(frame, labels, **clocks)

    monkeypatch.setattr(visualization, "compose_status_banner", capture_alignment)
    render_comparison_video(
        source,
        overlay,
        simulation,
        tmp_path / "comparison.mp4",
        _clock_test_media(),
        {"status": "rejected", "variant": "B0"},
        source_duration_s=1.0,
        simulation_duration_s=simulation_duration_s,
    )

    assert set(observed) == {expected_alignment}

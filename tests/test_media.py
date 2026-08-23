import cv2
import numpy as np
import pytest

from webvideo_to_data.media import probe_video, sha256_file


@pytest.fixture
def fixture_path(tmp_path):
    path = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
    )
    assert writer.isOpened()
    for index in range(10):
        frame = np.full((48, 64, 3), index, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_probe_video_reads_generated_fixture_metadata(fixture_path):
    metadata = probe_video(fixture_path)

    assert metadata.width == 64
    assert metadata.height == 48
    assert metadata.frame_count == 10
    assert metadata.fps == pytest.approx(10.0, rel=0.1)
    assert len(sha256_file(fixture_path)) == 64


def test_probe_video_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_video(tmp_path / "missing.avi")


def test_probe_video_rejects_unreadable_file(tmp_path):
    path = tmp_path / "not-a-video.avi"
    path.write_text("not video data", encoding="utf-8")

    with pytest.raises(ValueError, match="video cannot be opened"):
        probe_video(path)

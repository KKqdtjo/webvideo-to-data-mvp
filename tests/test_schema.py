import numpy as np
import pytest

from webvideo_to_data.schema import PhaseInterval, RunStatus, Trajectory2D


def test_phase_interval_rejects_reversed_frames():
    with pytest.raises(ValueError, match="end_frame"):
        PhaseInterval("hold", 8, 4, 0.8, ("motion",))


def test_phase_interval_rejects_confidence_outside_unit_interval():
    with pytest.raises(ValueError, match="confidence"):
        PhaseInterval("hold", 4, 8, 1.1, ("motion",))


def test_trajectory_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        Trajectory2D(
            timestamps_s=np.array([0.0, 0.1]),
            centers_px=np.array([[10.0, 20.0]]),
            confidence=np.array([0.9, 0.8]),
        )


@pytest.mark.parametrize(
    ("timestamps_s", "centers_px", "confidence", "message"),
    [
        ([0.0], [[1.0, 2.0, 3.0]], [0.9], "shape"),
        ([0.1, 0.0], [[1.0, 2.0], [3.0, 4.0]], [0.9, 0.8], "monotonic"),
        ([0.0], [[np.nan, 2.0]], [0.9], "finite"),
        ([0.0], [[1.0, 2.0]], [1.1], "confidence"),
    ],
)
def test_trajectory_rejects_invalid_values(
    timestamps_s, centers_px, confidence, message
):
    with pytest.raises(ValueError, match=message):
        Trajectory2D(
            timestamps_s=timestamps_s,
            centers_px=centers_px,
            confidence=confidence,
        )


def test_trajectory_converts_sequences_to_numpy_arrays():
    trajectory = Trajectory2D(
        timestamps_s=[0.0, 0.1],
        centers_px=[[10.0, 20.0], [11.0, 21.0]],
        confidence=[0.9, 0.8],
    )

    assert isinstance(trajectory.timestamps_s, np.ndarray)
    assert isinstance(trajectory.centers_px, np.ndarray)
    assert isinstance(trajectory.confidence, np.ndarray)


def test_run_status_has_auditable_terminal_values():
    assert {status.value for status in RunStatus} == {
        "completed",
        "not_run",
        "rejected",
        "failed",
    }

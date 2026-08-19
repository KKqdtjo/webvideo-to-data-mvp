import numpy as np
import pytest

from webvideo_to_data.contact import infer_motion_phases
from webvideo_to_data.schema import Trajectory2D


def test_infers_motion_onset_and_post_motion_release() -> None:
    """Missing hysteresis would not locate the handoff from rest to motion."""

    centers_x = np.concatenate(
        (np.zeros(10), np.arange(1, 21, dtype=float), np.full(10, 20.0))
    )
    trajectory = Trajectory2D(
        timestamps_s=np.arange(40, dtype=float) / 10.0,
        centers_px=np.column_stack((centers_x, np.full(40, 24.0))),
        confidence=np.ones(40),
    )

    phases = infer_motion_phases(
        trajectory,
        fps=10.0,
        speed_on_px_s=8.0,
        speed_off_px_s=3.0,
        min_phase_s=0.3,
    )

    assert tuple(phase.label for phase in phases) == (
        "approach",
        "hold",
        "release",
        "settle",
    )
    hold = phases[1]
    release = phases[2]
    assert hold.start_frame == pytest.approx(10, abs=1)
    assert release.start_frame == pytest.approx(30, abs=2)
    assert all(
        earlier.end_frame < later.start_frame
        for earlier, later in zip(phases, phases[1:])
    )
    assert phases[0].evidence == ("object_still",)
    assert phases[1].evidence == ("object_motion",)
    assert phases[3].evidence == ("object_settled",)


def test_rejects_late_motion_without_room_for_four_non_overlapping_phases() -> None:
    """Clamping release before hold must not fabricate overlapping intervals."""

    trajectory = Trajectory2D(
        timestamps_s=np.arange(6, dtype=float),
        centers_px=np.column_stack(([0.0, 0.0, 0.0, 0.0, 0.0, 100.0], np.zeros(6))),
        confidence=np.ones(6),
    )

    with pytest.raises(ValueError, match="four non-empty, non-overlapping phases"):
        infer_motion_phases(trajectory, fps=1.0)

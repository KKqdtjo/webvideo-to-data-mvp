import numpy as np
import pytest

from webvideo_to_data.retargeting import (
    RobotReference,
    build_pick_place_reference,
    map_pixels_to_scene,
)
from webvideo_to_data.schema import PhaseInterval, Trajectory2D


def test_map_pixels_to_scene_uses_canonical_bounds() -> None:
    points_px = np.array(
        [
            [270.0, 480.0],
            [0.0, 0.0],
            [-10.0, -20.0],
            [600.0, 1000.0],
        ]
    )

    mapped = map_pixels_to_scene(
        points_px,
        image_size=(540, 960),
        x_bounds=(-0.15, 0.15),
        y_bounds=(0.35, 0.65),
    )

    np.testing.assert_allclose(
        mapped,
        np.array(
            [
                [0.0, 0.5],
                [-0.15, 0.35],
                [-0.15, 0.35],
                [0.15, 0.65],
            ]
        ),
    )


def _trajectory_and_phases() -> tuple[Trajectory2D, tuple[PhaseInterval, ...]]:
    trajectory = Trajectory2D(
        timestamps_s=[0.0, 0.1, 0.2, 0.3, 0.4],
        centers_px=[
            [90.0, 160.0],
            [90.0, 160.0],
            [270.0, 700.0],
            [450.0, 800.0],
            [450.0, 800.0],
        ],
        confidence=[1.0, 1.0, 0.9, 1.0, 1.0],
    )
    phases = (
        PhaseInterval("approach", 0, 1, 0.9, ("object_still",)),
        PhaseInterval("hold", 2, 2, 0.9, ("object_motion",)),
        PhaseInterval("settle", 3, 4, 0.9, ("object_settled",)),
    )
    return trajectory, phases


def test_b0_reference_has_all_phases_fixed_poses_and_bounded_speed() -> None:
    trajectory, phases = _trajectory_and_phases()

    reference = build_pick_place_reference(trajectory, phases, variant="B0")

    assert isinstance(reference, RobotReference)
    assert reference.source_variant == "B0"
    assert set(reference.phase) == {
        "approach",
        "close",
        "lift",
        "transport",
        "lower",
        "open",
        "retreat",
    }
    assert reference.ee_positions.shape[1:] == (3,)
    assert reference.quaternion_wxyz.shape == (len(reference.timestamps_s), 4)
    assert reference.gripper_width.shape == (len(reference.timestamps_s),)
    close_index = reference.phase.index("close")
    open_index = reference.phase.index("open")
    np.testing.assert_allclose(reference.ee_positions[close_index], [0.12, 0.45, 0.04])
    np.testing.assert_allclose(reference.ee_positions[open_index], [-0.05, 0.55, 0.13])
    assert reference.timestamps_s[-1] - reference.timestamps_s[open_index] >= 1.0
    speed = np.linalg.norm(
        np.diff(reference.ee_positions, axis=0)
        / np.diff(reference.timestamps_s)[:, None],
        axis=1,
    )
    assert float(np.max(speed)) <= 0.35 + 1e-9


def test_b0_manual_baseline_is_independent_of_video_track_length() -> None:
    """Catch a fixed B0 baseline silently changing with the source frame count."""

    short_trajectory, phases = _trajectory_and_phases()
    long_trajectory = Trajectory2D(
        timestamps_s=np.arange(210) / 30.0,
        centers_px=np.zeros((210, 2)),
        confidence=np.ones(210),
    )

    short_reference = build_pick_place_reference(short_trajectory, phases, variant="B0")
    long_reference = build_pick_place_reference(long_trajectory, phases, variant="B0")

    np.testing.assert_array_equal(long_reference.timestamps_s, short_reference.timestamps_s)
    np.testing.assert_allclose(long_reference.ee_positions, short_reference.ee_positions)
    assert long_reference.phase == short_reference.phase


def test_b1_maps_stable_endpoints_and_preserves_tracked_path_shape() -> None:
    trajectory, phases = _trajectory_and_phases()

    reference = build_pick_place_reference(trajectory, phases, variant="B1")

    assert reference.source_variant == "B1"
    close_index = reference.phase.index("close")
    open_index = reference.phase.index("open")
    np.testing.assert_allclose(reference.ee_positions[close_index, :2], [-0.10, 0.40])
    np.testing.assert_allclose(reference.ee_positions[open_index, :2], [0.10, 0.60])
    transport_xy = reference.ee_positions[
        np.asarray(reference.phase) == "transport", :2
    ]
    assert np.max(transport_xy[:, 1] - np.linspace(0.40, 0.60, len(transport_xy))) > 0.03


def test_robot_reference_rejects_empty_timeline() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        RobotReference(
            timestamps_s=np.empty(0),
            ee_positions=np.empty((0, 3)),
            quaternion_wxyz=np.empty((0, 4)),
            gripper_width=np.empty(0),
            phase=(),
            source_variant="B0",
        )


def test_robot_reference_rejects_zero_norm_quaternion() -> None:
    with pytest.raises(ValueError, match="nonzero unit orientations"):
        RobotReference(
            timestamps_s=np.array([0.0]),
            ee_positions=np.array([[0.0, 0.0, 0.0]]),
            quaternion_wxyz=np.zeros((1, 4)),
            gripper_width=np.array([0.08]),
            phase=("approach",),
            source_variant="B0",
        )


def test_robot_reference_rejects_nonfinite_quaternion() -> None:
    with pytest.raises(ValueError, match="finite"):
        RobotReference(
            timestamps_s=np.array([0.0]),
            ee_positions=np.array([[0.0, 0.0, 0.0]]),
            quaternion_wxyz=np.array([[np.nan, 0.0, 0.0, 0.0]]),
            gripper_width=np.array([0.08]),
            phase=("approach",),
            source_variant="B0",
        )

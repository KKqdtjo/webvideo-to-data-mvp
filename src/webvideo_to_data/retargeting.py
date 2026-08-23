"""Map tracked image motion into canonical robot references."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .config import ControlConfig
from .schema import PhaseInterval, Trajectory2D


TrajectoryVariant = Literal["B0", "B1"]


@dataclass(frozen=True)
class RobotReference:
    """Time-indexed Cartesian and gripper targets for robot replay."""

    timestamps_s: NDArray[np.float64]
    ee_positions: NDArray[np.float64]
    quaternion_wxyz: NDArray[np.float64]
    gripper_width: NDArray[np.float64]
    phase: tuple[str, ...]
    source_variant: TrajectoryVariant

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_s, dtype=float)
        positions = np.asarray(self.ee_positions, dtype=float)
        quaternions = np.asarray(self.quaternion_wxyz, dtype=float)
        widths = np.asarray(self.gripper_width, dtype=float)
        frame_count = len(timestamps)
        if frame_count == 0:
            raise ValueError("RobotReference must contain at least one frame")
        if timestamps.ndim != 1 or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("timestamps_s must be a strictly increasing [T] array")
        if positions.shape != (frame_count, 3):
            raise ValueError("ee_positions must have shape [T, 3]")
        if quaternions.shape != (frame_count, 4):
            raise ValueError("quaternion_wxyz must have shape [T, 4]")
        if widths.shape != (frame_count,):
            raise ValueError("gripper_width must have shape [T]")
        if len(self.phase) != frame_count:
            raise ValueError("phase must contain one label per frame")
        if self.source_variant not in ("B0", "B1"):
            raise ValueError("source_variant must be B0 or B1")
        if not all(
            np.isfinite(values).all()
            for values in (timestamps, positions, quaternions, widths)
        ):
            raise ValueError("reference values must be finite")
        if not np.allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-6):
            raise ValueError("quaternion_wxyz must contain nonzero unit orientations")
        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "ee_positions", positions)
        object.__setattr__(self, "quaternion_wxyz", quaternions)
        object.__setattr__(self, "gripper_width", widths)


def _quaternion_matrix(quaternion_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    w, x, y, z = quaternion_wxyz / np.linalg.norm(quaternion_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _matrix_quaternion(rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    quaternion = np.empty(4, dtype=float)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion[:] = (
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        )
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion[:] = (
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            )
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion[:] = (
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion[:] = (
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion if quaternion[0] >= 0.0 else -quaternion


def _grasp_frame_quaternion(
    approach_axis: NDArray[np.float64],
    convention_wxyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Derive a TCP frame from an approach axis and a convention selector."""

    local_z = approach_axis / np.linalg.norm(approach_axis)
    convention = _quaternion_matrix(convention_wxyz)
    local_x = convention[:, 0] - np.dot(convention[:, 0], local_z) * local_z
    if np.linalg.norm(local_x) < 1e-9:
        local_x = convention[:, 1] - np.dot(convention[:, 1], local_z) * local_z
    local_x /= np.linalg.norm(local_x)
    local_y = np.cross(local_z, local_x)
    return _matrix_quaternion(np.column_stack((local_x, local_y, local_z)))


def build_manual_b0_reference(
    start_m: Sequence[float],
    goal_m: Sequence[float],
    control: ControlConfig,
    grasp_quaternion_wxyz: Sequence[float],
) -> RobotReference:
    """Build the fixed ten-phase manual baseline from can/TCP geometry."""

    start = np.asarray(start_m, dtype=float)
    goal = np.asarray(goal_m, dtype=float)
    convention = np.asarray(grasp_quaternion_wxyz, dtype=float)
    if start.shape != (3,) or goal.shape != (3,):
        raise ValueError("manual start and goal must be three-vectors")
    if not np.isfinite(start).all() or not np.isfinite(goal).all():
        raise ValueError("manual start and goal must be finite")
    if convention.shape != (4,) or not np.isfinite(convention).all():
        raise ValueError("grasp quaternion must be finite")
    if np.linalg.norm(convention) == 0.0:
        raise ValueError("grasp quaternion must be nonzero")

    # The can axis is vertical.  Grasp phases put the TCP offset on that axis,
    # pointing local TCP z down toward the can.  Home uses its own TCP-to-can
    # offset, while the quaternion input selects only the free radial yaw.
    cylinder_axis = np.array([0.0, 0.0, 1.0])
    home = start + np.array([0.0, -0.20, 0.35])
    pregrasp = start + 0.16 * cylinder_axis
    approach = start + 0.08 * cylinder_axis
    lifted = start + 0.18 * cylinder_axis
    transported = goal + 0.18 * cylinder_axis
    retreat = goal + 0.15 * cylinder_axis
    grasp_orientation = _grasp_frame_quaternion(-cylinder_axis, convention)
    home_orientation = _grasp_frame_quaternion(start - home, convention)

    phase_keyframes = (
        ("home", home, home_orientation, control.gripper_open_width_m),
        ("pregrasp", pregrasp, grasp_orientation, control.gripper_open_width_m),
        ("approach", approach, grasp_orientation, control.gripper_open_width_m),
        ("close", start, grasp_orientation, control.gripper_closed_width_m),
        ("lift", lifted, grasp_orientation, control.gripper_closed_width_m),
        ("transport", transported, grasp_orientation, control.gripper_closed_width_m),
        ("lower", goal, grasp_orientation, control.gripper_closed_width_m),
        ("open", goal, grasp_orientation, control.gripper_open_width_m),
        ("retreat", retreat, grasp_orientation, control.gripper_open_width_m),
        ("settle", retreat, grasp_orientation, control.gripper_open_width_m),
    )
    timestamps = [0.0]
    for phase, *_ in phase_keyframes:
        timestamps.append(timestamps[-1] + control.phase_duration_s[phase])
    final_phase, final_position, final_quaternion, final_width = phase_keyframes[-1]
    return RobotReference(
        timestamps_s=np.asarray(timestamps, dtype=float),
        ee_positions=np.asarray(
            [item[1] for item in phase_keyframes] + [final_position], dtype=float
        ),
        quaternion_wxyz=np.asarray(
            [item[2] for item in phase_keyframes] + [final_quaternion], dtype=float
        ),
        gripper_width=np.asarray(
            [item[3] for item in phase_keyframes] + [final_width], dtype=float
        ),
        phase=tuple(item[0] for item in phase_keyframes) + (final_phase,),
        source_variant="B0",
    )


def map_pixels_to_scene(
    points_px: NDArray[np.float64] | Sequence[Sequence[float]],
    image_size: tuple[int, int],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> NDArray[np.float64]:
    """Map ``[x, y]`` pixels linearly into canonical scene x/y bounds."""

    points = np.asarray(points_px, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_px must have shape [T, 2]")
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_size must contain positive dimensions")
    if x_bounds[0] >= x_bounds[1] or y_bounds[0] >= y_bounds[1]:
        raise ValueError("scene bounds must be increasing")
    clipped = np.clip(points, (0.0, 0.0), (image_width, image_height))
    x = x_bounds[0] + clipped[:, 0] / image_width * (x_bounds[1] - x_bounds[0])
    y = y_bounds[0] + clipped[:, 1] / image_height * (y_bounds[1] - y_bounds[0])
    return np.column_stack((x, y))


def _stable_center(
    trajectory: Trajectory2D, interval: PhaseInterval
) -> NDArray[np.float64]:
    indices = np.arange(interval.start_frame, interval.end_frame + 1)
    indices = indices[indices < len(trajectory.centers_px)]
    if not len(indices):
        raise ValueError("phase intervals must overlap the trajectory")
    confident = indices[trajectory.confidence[indices] >= 0.5]
    selected = confident if len(confident) else indices
    return np.median(trajectory.centers_px[selected], axis=0)


def _densify_segment(
    output_positions: list[NDArray[np.float64]],
    output_phases: list[str],
    output_widths: list[float],
    targets: Sequence[NDArray[np.float64]],
    phase: str,
    width: float,
    maximum_step: float,
) -> None:
    for target in targets:
        start = output_positions[-1]
        distance = float(np.linalg.norm(target - start))
        steps = max(1, int(np.ceil(distance / maximum_step)))
        for fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
            output_positions.append(start + fraction * (target - start))
            output_phases.append(phase)
            output_widths.append(width)


def build_pick_place_reference(
    trajectory: Trajectory2D,
    phases: Sequence[PhaseInterval],
    variant: TrajectoryVariant,
    *,
    image_size: tuple[int, int] = (540, 960),
    x_bounds: tuple[float, float] = (-0.15, 0.15),
    y_bounds: tuple[float, float] = (0.35, 0.65),
    max_speed_m_s: float = 0.35,
    sample_period_s: float = 0.1,
    post_release_hold_s: float = 1.0,
    b0_start_m: Sequence[float] = (0.12, 0.45, 0.04),
    b0_goal_m: Sequence[float] = (-0.05, 0.55, 0.13),
) -> RobotReference:
    """Build a speed-limited, seven-phase object-centric pick/place reference."""

    if variant not in ("B0", "B1"):
        raise ValueError("variant must be B0 or B1")
    if not phases:
        raise ValueError("at least one phase interval is required")
    if max_speed_m_s <= 0.0 or sample_period_s <= 0.0:
        raise ValueError("speed and sample period must be positive")
    if post_release_hold_s < 1.0:
        raise ValueError("post_release_hold_s must be at least 1.0")

    if variant == "B0":
        start = np.asarray(b0_start_m, dtype=float)
        goal = np.asarray(b0_goal_m, dtype=float)
        if start.shape != (3,) or goal.shape != (3,):
            raise ValueError("B0 start and goal must each contain x, y, z")
        if not np.isfinite(start).all() or not np.isfinite(goal).all():
            raise ValueError("B0 start and goal must be finite")
        start_xy = start[:2]
        goal_xy = goal[:2]
        tracked_xy = np.linspace(start_xy, goal_xy, 5)
    else:
        first_interval = min(phases, key=lambda item: item.start_frame)
        last_interval = max(phases, key=lambda item: item.end_frame)
        first_center = _stable_center(trajectory, first_interval)
        last_center = _stable_center(trajectory, last_interval)
        endpoints = map_pixels_to_scene(
            np.vstack((first_center, last_center)), image_size, x_bounds, y_bounds
        )
        start_xy, goal_xy = endpoints
        tracked_xy = map_pixels_to_scene(
            trajectory.centers_px, image_size, x_bounds, y_bounds
        )
        start = np.r_[start_xy, 0.04]
        goal = np.r_[goal_xy, 0.13]

    lift_height = max(start[2], goal[2]) + 0.10
    approach = start + np.array([0.0, 0.0, 0.10])
    lifted = np.r_[start_xy, lift_height]
    transport = [np.r_[xy, lift_height] for xy in tracked_xy]
    lowered = goal.copy()
    retreat = goal + np.array([0.0, 0.0, 0.10])

    positions: list[NDArray[np.float64]] = [approach]
    labels = ["approach"]
    widths = [0.08]
    maximum_step = max_speed_m_s * sample_period_s * 0.95
    _densify_segment(positions, labels, widths, [start], "approach", 0.08, maximum_step)
    _densify_segment(positions, labels, widths, [start], "close", 0.0, maximum_step)
    _densify_segment(positions, labels, widths, [lifted], "lift", 0.0, maximum_step)
    _densify_segment(positions, labels, widths, transport, "transport", 0.0, maximum_step)
    _densify_segment(positions, labels, widths, [lowered], "lower", 0.0, maximum_step)
    _densify_segment(positions, labels, widths, [goal], "open", 0.08, maximum_step)
    hold_steps = math.ceil(post_release_hold_s / sample_period_s)
    _densify_segment(
        positions,
        labels,
        widths,
        [goal] * hold_steps,
        "open",
        0.08,
        maximum_step,
    )
    _densify_segment(positions, labels, widths, [retreat], "retreat", 0.08, maximum_step)

    positions_array = np.asarray(positions, dtype=float)
    frame_count = len(positions_array)
    quaternions = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (frame_count, 1))
    return RobotReference(
        timestamps_s=np.arange(frame_count, dtype=float) * sample_period_s,
        ee_positions=positions_array,
        quaternion_wxyz=quaternions,
        gripper_width=np.asarray(widths, dtype=float),
        phase=tuple(labels),
        source_variant=variant,
    )

"""Offline converged inverse kinematics and Panda joint-control planning."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from .config import ControlConfig, IKConfig
from .retargeting import RobotReference
from .scene import PandaSceneIds


class IKPlanningError(RuntimeError):
    """A Cartesian phase boundary could not be solved offline."""

    def __init__(self, phase: str) -> None:
        self.phase = phase
        super().__init__(f"IK failed for phase {phase}")


class ControlLimitError(RuntimeError):
    """A planned phase cannot satisfy one configured joint limit."""

    def __init__(self, phase: str, limit: str) -> None:
        self.phase = phase
        self.limit = limit
        super().__init__(f"control limit {limit} failed for phase {phase}")


@dataclass(frozen=True)
class IKResult:
    arm_qpos: NDArray[np.float64]
    converged: bool
    iterations: int
    position_error_m: float
    orientation_error_rad: float


@dataclass(frozen=True)
class JointControlProgram:
    timestamps_s: NDArray[np.float64]
    arm_qpos_targets: NDArray[np.float64]
    gripper_ctrl: NDArray[np.float64]
    ee_positions: NDArray[np.float64]
    quaternion_wxyz: NDArray[np.float64]
    phase: tuple[str, ...]
    keyframe_ik: tuple[IKResult, ...]


def solve_pose_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: PandaSceneIds,
    position_m: NDArray[np.float64],
    quaternion_wxyz: NDArray[np.float64],
    options: IKConfig,
    initial_arm_qpos: NDArray[np.float64] | None,
) -> IKResult:
    """Solve a full TCP pose without changing the caller's MuJoCo state."""

    position = np.asarray(position_m, dtype=float)
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("position_m must be a finite three-vector")
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion_wxyz must be a finite quaternion")
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm == 0.0:
        raise ValueError("quaternion_wxyz must be nonzero")

    working = mujoco.MjData(model)
    working.qpos[:] = data.qpos
    working.qvel[:] = data.qvel
    qpos_addresses = np.asarray(
        [model.jnt_qposadr[joint_id] for joint_id in ids.arm_joint_ids]
    )
    dof_addresses = np.asarray(
        [model.jnt_dofadr[joint_id] for joint_id in ids.arm_joint_ids]
    )
    if initial_arm_qpos is not None:
        initial = np.asarray(initial_arm_qpos, dtype=float)
        if initial.shape != (len(ids.arm_joint_ids),) or not np.isfinite(initial).all():
            raise ValueError("initial_arm_qpos must be a finite arm joint vector")
        working.qpos[qpos_addresses] = initial

    desired_quaternion = quaternion / quaternion_norm
    joint_ids = np.asarray(ids.arm_joint_ids)
    lower = model.jnt_range[joint_ids, 0]
    upper = model.jnt_range[joint_ids, 1]
    center = 0.5 * (lower + upper)
    span = np.maximum(upper - lower, 1e-9)
    position_norm = np.inf
    orientation_norm = np.inf

    for iteration in range(1, options.maximum_iterations + 1):
        mujoco.mj_forward(model, working)
        current_quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(
            current_quaternion, working.site_xmat[ids.tcp_site_id]
        )
        current_inverse = current_quaternion * np.array(
            [1.0, -1.0, -1.0, -1.0]
        )
        quaternion_error = np.empty(4, dtype=float)
        mujoco.mju_mulQuat(
            quaternion_error, desired_quaternion, current_inverse
        )
        if quaternion_error[0] < 0.0:
            quaternion_error *= -1.0
        rotation_error = np.empty(3, dtype=float)
        mujoco.mju_quat2Vel(rotation_error, quaternion_error, 1.0)
        position_error = position - working.site_xpos[ids.tcp_site_id]
        position_norm = float(np.linalg.norm(position_error))
        orientation_norm = float(np.linalg.norm(rotation_error))
        if (
            position_norm <= options.position_tolerance_m
            and orientation_norm <= options.orientation_tolerance_rad
        ):
            return IKResult(
                working.qpos[qpos_addresses].copy(),
                True,
                iteration,
                position_norm,
                orientation_norm,
            )

        jacobian_position = np.zeros((3, model.nv), dtype=float)
        jacobian_rotation = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacSite(
            model,
            working,
            jacobian_position,
            jacobian_rotation,
            ids.tcp_site_id,
        )
        jacobian = np.vstack(
            (
                jacobian_position[:, dof_addresses],
                options.orientation_weight
                * jacobian_rotation[:, dof_addresses],
            )
        )
        error = np.r_[
            position_error, options.orientation_weight * rotation_error
        ]
        normal = (
            jacobian @ jacobian.T + options.damping**2 * np.eye(6)
        )
        damped_pseudoinverse = jacobian.T @ np.linalg.solve(
            normal, np.eye(6)
        )
        task_delta = damped_pseudoinverse @ error
        joint_limit_gradient = (
            center - working.qpos[qpos_addresses]
        ) / span
        nullspace = (
            np.eye(len(ids.arm_joint_ids))
            - damped_pseudoinverse @ jacobian
        )
        delta = task_delta + nullspace @ (
            options.joint_limit_weight * joint_limit_gradient
        )
        next_qpos = (
            working.qpos[qpos_addresses] + options.step_size * delta
        )
        working.qpos[qpos_addresses] = np.clip(next_qpos, lower, upper)

    mujoco.mj_forward(model, working)
    current_quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(
        current_quaternion, working.site_xmat[ids.tcp_site_id]
    )
    current_inverse = current_quaternion * np.array(
        [1.0, -1.0, -1.0, -1.0]
    )
    quaternion_error = np.empty(4, dtype=float)
    mujoco.mju_mulQuat(
        quaternion_error, desired_quaternion, current_inverse
    )
    if quaternion_error[0] < 0.0:
        quaternion_error *= -1.0
    rotation_error = np.empty(3, dtype=float)
    mujoco.mju_quat2Vel(rotation_error, quaternion_error, 1.0)
    position_norm = float(
        np.linalg.norm(position - working.site_xpos[ids.tcp_site_id])
    )
    orientation_norm = float(np.linalg.norm(rotation_error))
    converged = bool(
        position_norm <= options.position_tolerance_m
        and orientation_norm <= options.orientation_tolerance_rad
    )
    return IKResult(
        working.qpos[qpos_addresses].copy(),
        converged,
        options.maximum_iterations,
        position_norm,
        orientation_norm,
    )


def _keyframe_indices(reference: RobotReference) -> tuple[int, ...]:
    indices = [0]
    indices.extend(
        index
        for index in range(1, len(reference.phase))
        if reference.phase[index] != reference.phase[index - 1]
    )
    if indices[-1] != len(reference.phase) - 1:
        indices.append(len(reference.phase) - 1)
    return tuple(indices)


def _slerp(
    first: NDArray[np.float64], second: NDArray[np.float64], fraction: float
) -> NDArray[np.float64]:
    start = first / np.linalg.norm(first)
    end = second / np.linalg.norm(second)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = start + fraction * (end - start)
        return result / np.linalg.norm(result)
    angle = float(np.arccos(dot))
    scale = np.sin(angle)
    return (
        np.sin((1.0 - fraction) * angle) / scale * start
        + np.sin(fraction * angle) / scale * end
    )


def _minimum_quintic_duration(
    delta: NDArray[np.float64], control: ControlConfig
) -> float:
    maximum_delta = float(np.max(np.abs(delta)))
    velocity_duration = (
        1.875 * maximum_delta / control.maximum_joint_velocity_rad_s
    )
    acceleration_duration = np.sqrt(
        (10.0 / np.sqrt(3.0))
        * maximum_delta
        / control.maximum_joint_acceleration_rad_s2
    )
    return 1.000001 * max(velocity_duration, acceleration_duration)


def _first_limit_violation(
    targets: NDArray[np.float64],
    timestamps: NDArray[np.float64],
    model: mujoco.MjModel,
    ids: PandaSceneIds,
    control: ControlConfig,
) -> tuple[int, str] | None:
    joint_ids = np.asarray(ids.arm_joint_ids)
    lower = model.jnt_range[joint_ids, 0]
    upper = model.jnt_range[joint_ids, 1]
    position_bad = np.argwhere((targets < lower) | (targets > upper))
    if len(position_bad):
        return int(position_bad[0, 0]), "position"
    if len(targets) < 2:
        return None
    intervals = np.diff(timestamps)
    velocity = np.diff(targets, axis=0) / intervals[:, None]
    velocity_bad = np.argwhere(
        np.abs(velocity) > control.maximum_joint_velocity_rad_s + 1e-9
    )
    if len(velocity_bad):
        return int(velocity_bad[0, 0] + 1), "velocity"
    if len(targets) < 3:
        return None
    acceleration = np.diff(velocity, axis=0) / intervals[1:, None]
    acceleration_bad = np.argwhere(
        np.abs(acceleration)
        > control.maximum_joint_acceleration_rad_s2 + 1e-9
    )
    if len(acceleration_bad):
        return int(acceleration_bad[0, 0] + 2), "acceleration"
    return None


def plan_joint_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: PandaSceneIds,
    reference: RobotReference,
    ik: IKConfig,
    control: ControlConfig,
) -> JointControlProgram:
    """Solve phase boundaries, then build a smooth limit-bounded program."""

    keyframe_indices = _keyframe_indices(reference)
    results: list[IKResult] = []
    previous: NDArray[np.float64] | None = None
    for index in keyframe_indices:
        result = solve_pose_ik(
            model,
            data,
            ids,
            reference.ee_positions[index],
            reference.quaternion_wxyz[index],
            ik,
            previous,
        )
        if not result.converged:
            raise IKPlanningError(reference.phase[index])
        results.append(result)
        previous = result.arm_qpos

    arm_qpos_addresses = model.jnt_qposadr[np.asarray(ids.arm_joint_ids)]
    initial_arm_qpos = data.qpos[arm_qpos_addresses].copy()
    times = [0.0]
    targets = [initial_arm_qpos]
    phases = [reference.phase[keyframe_indices[0]]]
    widths = [float(reference.gripper_width[keyframe_indices[0]])]
    sample_period = 1.0 / control.control_hz

    initial_duration = float(control.phase_duration_s[phases[0]])
    initial_steps = max(
        1,
        int(
            np.ceil(
                max(
                    initial_duration,
                    _minimum_quintic_duration(
                        results[0].arm_qpos - initial_arm_qpos, control
                    ),
                )
                * control.control_hz
            )
        ),
    )
    for step in range(1, initial_steps + 1):
        fraction = step / initial_steps
        blend = fraction**3 * (10.0 - 15.0 * fraction + 6.0 * fraction**2)
        times.append(times[-1] + sample_period)
        targets.append(
            initial_arm_qpos + blend * (results[0].arm_qpos - initial_arm_qpos)
        )
        phases.append(reference.phase[keyframe_indices[0]])
        widths.append(float(reference.gripper_width[keyframe_indices[0]]))

    for segment in range(len(keyframe_indices) - 1):
        first_index = keyframe_indices[segment]
        second_index = keyframe_indices[segment + 1]
        first_qpos = results[segment].arm_qpos
        second_qpos = results[segment + 1].arm_qpos
        requested_duration = float(
            reference.timestamps_s[second_index]
            - reference.timestamps_s[first_index]
        )
        required_duration = _minimum_quintic_duration(
            second_qpos - first_qpos, control
        )
        steps = max(
            1,
            int(
                np.ceil(
                    max(requested_duration, required_duration)
                    * control.control_hz
                )
            ),
        )
        for step in range(1, steps + 1):
            fraction = step / steps
            blend = fraction**3 * (
                10.0 - 15.0 * fraction + 6.0 * fraction**2
            )
            times.append(times[-1] + sample_period)
            targets.append(first_qpos + blend * (second_qpos - first_qpos))
            at_boundary = step == steps
            label_index = second_index if at_boundary else first_index
            phases.append(reference.phase[label_index])
            widths.append(float(reference.gripper_width[label_index]))

    timestamp_array = np.asarray(times, dtype=float)
    target_array = np.asarray(targets, dtype=float)
    violation = _first_limit_violation(
        target_array, timestamp_array, model, ids, control
    )
    if violation is not None:
        sample_index, limit = violation
        raise ControlLimitError(phases[sample_index], limit)

    widths_array = np.asarray(widths, dtype=float)
    width_span = control.gripper_open_width_m - control.gripper_closed_width_m
    if width_span <= 0.0:
        raise ControlLimitError(phases[0], "position")
    gripper = 255.0 * (
        widths_array - control.gripper_closed_width_m
    ) / width_span
    gripper = np.clip(gripper, 0.0, 255.0)
    positions = np.empty((len(target_array), 3), dtype=float)
    quaternions = np.empty((len(target_array), 4), dtype=float)
    working = mujoco.MjData(model)
    for index, target in enumerate(target_array):
        working.qpos[:] = data.qpos
        working.qvel[:] = data.qvel
        working.qpos[arm_qpos_addresses] = target
        mujoco.mj_forward(model, working)
        positions[index] = working.site_xpos[ids.tcp_site_id]
        mujoco.mju_mat2Quat(
            quaternions[index], working.site_xmat[ids.tcp_site_id]
        )
    return JointControlProgram(
        timestamps_s=timestamp_array,
        arm_qpos_targets=target_array,
        gripper_ctrl=gripper,
        ee_positions=positions,
        quaternion_wxyz=quaternions,
        phase=tuple(phases),
        keyframe_ik=tuple(results),
    )

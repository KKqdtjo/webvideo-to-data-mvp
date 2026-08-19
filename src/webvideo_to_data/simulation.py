"""Headless MuJoCo replay of Cartesian Panda-like pick/place references."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
from numpy.typing import NDArray

from .retargeting import RobotReference


ReplayMode = Literal["kinematic_replay", "physics_grasp"]
_DEFAULT_MODEL_PATH = Path(__file__).with_name("assets") / "panda_pick_place.xml"


@dataclass(frozen=True)
class SimulationResult:
    """Recorded MuJoCo rollout and conservative physical placement verdict."""

    mode: ReplayMode
    qpos: NDArray[np.float64]
    qvel: NDArray[np.float64]
    can_pose: NDArray[np.float64]
    contact_count: NDArray[np.int64]
    minimum_distance_m: float
    target_error_m: float
    rendered_rgb: NDArray[np.uint8]
    invalid_numerical_state: bool
    placed_successfully: bool
    maximum_lift_m: float
    support_contact_duration_s: float


def _named_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    identifier = mujoco.mj_name2id(model, object_type, name)
    if identifier < 0:
        raise ValueError(f"MuJoCo model is missing required name: {name}")
    return identifier


def _reference_index(reference: RobotReference, time_s: float) -> int:
    return min(
        len(reference.timestamps_s) - 1,
        max(0, int(np.searchsorted(reference.timestamps_s, time_s, side="right") - 1)),
    )


def _apply_damped_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    desired_position: NDArray[np.float64],
    desired_quaternion_wxyz: NDArray[np.float64],
    site_id: int,
    joint_ids: list[int],
    actuator_ids: list[int],
    damping: float,
) -> None:
    jacobian_position = np.zeros((3, model.nv), dtype=float)
    jacobian_rotation = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(
        model, data, jacobian_position, jacobian_rotation, site_id
    )
    dof_indices = np.array([model.jnt_dofadr[joint_id] for joint_id in joint_ids])
    current_quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(current_quaternion, data.site_xmat[site_id])
    current_inverse = current_quaternion * np.array([1.0, -1.0, -1.0, -1.0])
    desired_quaternion = desired_quaternion_wxyz / np.linalg.norm(
        desired_quaternion_wxyz
    )
    quaternion_error = np.empty(4, dtype=float)
    mujoco.mju_mulQuat(quaternion_error, desired_quaternion, current_inverse)
    if quaternion_error[0] < 0.0:
        quaternion_error *= -1.0
    rotation_error = np.empty(3, dtype=float)
    mujoco.mju_quat2Vel(rotation_error, quaternion_error, 1.0)

    rotation_weight = 0.25
    jacobian = np.vstack(
        (
            jacobian_position[:, dof_indices],
            rotation_weight * jacobian_rotation[:, dof_indices],
        )
    )
    error = np.r_[
        desired_position - data.site_xpos[site_id],
        rotation_weight * rotation_error,
    ]
    normal = jacobian @ jacobian.T + damping**2 * np.eye(6)
    delta = jacobian.T @ np.linalg.solve(normal, error)
    for joint_id, actuator_id, change in zip(joint_ids, actuator_ids, delta):
        qpos_address = model.jnt_qposadr[joint_id]
        target = data.qpos[qpos_address] + float(np.clip(change, -0.08, 0.08))
        if model.jnt_limited[joint_id]:
            target = float(np.clip(target, *model.jnt_range[joint_id]))
        data.ctrl[actuator_id] = target


def _minimum_hand_can_distance(
    model: mujoco.MjModel, data: mujoco.MjData, can_geom_id: int
) -> float:
    distances = []
    for name in ("left_finger_geom", "right_finger_geom"):
        finger_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        from_to = np.empty(6, dtype=float)
        distance = mujoco.mj_geomDistance(
            model, data, can_geom_id, finger_id, 10.0, from_to
        )
        distances.append(max(0.0, float(distance)))
    return min(distances)


def _has_support_contact(
    data: mujoco.MjData, can_geom_id: int, support_geom_id: int
) -> bool:
    required_pair = {can_geom_id, support_geom_id}
    return any(
        {int(data.contact[index].geom1), int(data.contact[index].geom2)}
        == required_pair
        for index in range(data.ncon)
    )


def run_mujoco_replay(
    reference: RobotReference,
    *,
    mode: ReplayMode = "physics_grasp",
    model_path: str | Path = _DEFAULT_MODEL_PATH,
    max_steps: int | None = None,
    render_every: int = 20,
    render_size: tuple[int, int] = (320, 240),
    ik_damping: float = 0.05,
) -> SimulationResult:
    """Replay a reference with DLS Jacobian IK in a real headless MuJoCo model."""

    if mode not in ("kinematic_replay", "physics_grasp"):
        raise ValueError("mode must be kinematic_replay or physics_grasp")
    if render_every <= 0 or min(render_size) <= 0:
        raise ValueError("render interval and dimensions must be positive")
    if ik_damping <= 0.0:
        raise ValueError("ik_damping must be positive")

    model = mujoco.MjModel.from_xml_path(str(Path(model_path)))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    timestep = float(model.opt.timestep)
    requested_steps = max_steps
    if requested_steps is None:
        requested_steps = max(1, int(np.ceil(reference.timestamps_s[-1] / timestep)) + 1)
    if requested_steps <= 0:
        raise ValueError("max_steps must be positive")

    site_id = _named_id(model, mujoco.mjtObj.mjOBJ_SITE, "panda_tcp")
    can_body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "can")
    box_body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    can_geom_id = _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "can_geom")
    support_geom_id = _named_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "box_floor_geom"
    )
    can_joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "can_free")
    can_qpos_address = model.jnt_qposadr[can_joint_id]
    joint_ids = [
        _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"panda_joint{index}")
        for index in range(1, 8)
    ]
    actuator_ids = [
        _named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{index}_act")
        for index in range(1, 8)
    ]
    finger_actuator_ids = [
        _named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in ("left_finger_act", "right_finger_act")
    ]

    initial_can_z = float(data.xpos[can_body_id, 2])
    qpos_history = np.empty((requested_steps, model.nq), dtype=float)
    qvel_history = np.empty((requested_steps, model.nv), dtype=float)
    can_pose_history = np.empty((requested_steps, 7), dtype=float)
    contact_history = np.empty(requested_steps, dtype=np.int64)
    minimum_distance = np.inf
    current_support_duration = 0.0
    maximum_support_duration = 0.0
    frames: list[NDArray[np.uint8]] = []

    render_width, render_height = render_size
    renderer = mujoco.Renderer(model, height=render_height, width=render_width)
    try:
        for step in range(requested_steps):
            reference_index = _reference_index(reference, float(data.time))
            _apply_damped_ik(
                model,
                data,
                reference.ee_positions[reference_index],
                reference.quaternion_wxyz[reference_index],
                site_id,
                joint_ids,
                actuator_ids,
                ik_damping,
            )
            finger_target = float(np.clip(reference.gripper_width[reference_index] / 2.0, 0.0, 0.04))
            for actuator_id in finger_actuator_ids:
                data.ctrl[actuator_id] = finger_target
            if mode == "kinematic_replay":
                data.qpos[can_qpos_address : can_qpos_address + 3] = reference.ee_positions[
                    reference_index
                ]
                data.qpos[can_qpos_address + 3 : can_qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
                data.qvel[model.jnt_dofadr[can_joint_id] : model.jnt_dofadr[can_joint_id] + 6] = 0.0
                mujoco.mj_forward(model, data)

            mujoco.mj_step(model, data)
            qpos_history[step] = data.qpos
            qvel_history[step] = data.qvel
            can_pose_history[step, :3] = data.xpos[can_body_id]
            can_pose_history[step, 3:] = data.xquat[can_body_id]
            contact_history[step] = data.ncon
            if reference.phase[reference_index] == "open":
                if _has_support_contact(data, can_geom_id, support_geom_id):
                    current_support_duration += timestep
                    maximum_support_duration = max(
                        maximum_support_duration, current_support_duration
                    )
                else:
                    current_support_duration = 0.0
            minimum_distance = min(
                minimum_distance,
                _minimum_hand_can_distance(model, data, can_geom_id),
            )
            if (step + 1) % render_every == 0:
                renderer.update_scene(data, camera="overview")
                frames.append(renderer.render().copy())
    finally:
        renderer.close()

    invalid = not (
        np.isfinite(qpos_history).all()
        and np.isfinite(qvel_history).all()
        and np.isfinite(can_pose_history).all()
    )
    invalid = invalid or any(warning.number > 0 for warning in data.warning)
    maximum_lift = float(np.max(can_pose_history[:, 2]) - initial_can_z)
    target_error = float(np.linalg.norm(can_pose_history[-1, :3] - data.xpos[box_body_id]))
    box_center_xy = data.geom_xpos[support_geom_id, :2]
    available_margin = model.geom_size[support_geom_id, :2] - model.geom_size[
        can_geom_id, 0
    ]
    finishes_inside_box = bool(
        np.all(np.abs(can_pose_history[-1, :2] - box_center_xy) <= available_margin)
    )
    placed_successfully = bool(
        mode == "physics_grasp"
        and maximum_lift >= 0.03
        and finishes_inside_box
        and maximum_support_duration >= 1.0
        and not invalid
    )
    rendered = (
        np.asarray(frames, dtype=np.uint8)
        if frames
        else np.empty((0, render_height, render_width, 3), dtype=np.uint8)
    )
    return SimulationResult(
        mode=mode,
        qpos=qpos_history,
        qvel=qvel_history,
        can_pose=can_pose_history,
        contact_count=contact_history,
        minimum_distance_m=float(minimum_distance),
        target_error_m=target_error,
        rendered_rgb=rendered,
        invalid_numerical_state=invalid,
        placed_successfully=placed_successfully,
        maximum_lift_m=maximum_lift,
        support_contact_duration_s=maximum_support_duration,
    )

"""Pinned MuJoCo Menagerie Panda scene contract for EXP-001."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_SCENE_PATH = (
    Path(__file__).with_name("assets")
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "exp001_scene.xml"
)


@dataclass(frozen=True)
class ScenePerturbation:
    can_dx_m: float
    can_dy_m: float
    can_yaw_rad: float
    mass_scale: float
    friction_scale: float


@dataclass(frozen=True)
class PandaSceneIds:
    arm_joint_ids: tuple[int, ...]
    arm_actuator_ids: tuple[int, ...]
    finger_joint_ids: tuple[int, int]
    finger_body_ids: tuple[int, int]
    robot_body_ids: frozenset[int]
    robot_geom_ids: frozenset[int]
    tcp_site_id: int
    gripper_actuator_id: int
    can_body_id: int
    can_joint_id: int
    can_geom_id: int
    box_body_id: int
    box_floor_geom_id: int
    box_geom_ids: frozenset[int]
    table_geom_id: int


def _required_id(model: mujoco.MjModel, object_type: str, name: str) -> int:
    object_kind = {
        "joint": mujoco.mjtObj.mjOBJ_JOINT,
        "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
        "site": mujoco.mjtObj.mjOBJ_SITE,
    }[object_type]
    identifier = mujoco.mj_name2id(model, object_kind, name)
    if identifier < 0:
        raise ValueError(f"Panda scene is missing required {object_type}: {name}")
    return identifier


def _robot_body_ids(model: mujoco.MjModel, root_body_id: int) -> frozenset[int]:
    body_ids: set[int] = set()
    for body_id in range(model.nbody):
        ancestor = body_id
        while ancestor != 0 and ancestor != root_body_id:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == root_body_id:
            body_ids.add(body_id)
    return frozenset(body_ids)


def _resolve_scene_ids(model: mujoco.MjModel) -> PandaSceneIds:
    arm_joint_ids = tuple(
        _required_id(model, "joint", f"joint{index}") for index in range(1, 8)
    )
    arm_actuator_ids = tuple(
        _required_id(model, "actuator", f"actuator{index}") for index in range(1, 8)
    )
    finger_joint_ids = tuple(
        _required_id(model, "joint", name)
        for name in ("finger_joint1", "finger_joint2")
    )
    finger_body_ids = tuple(
        _required_id(model, "body", name) for name in ("left_finger", "right_finger")
    )
    root_body_id = _required_id(model, "body", "link0")
    robot_body_ids = _robot_body_ids(model, root_body_id)
    robot_geom_ids = frozenset(
        geom_id
        for geom_id, body_id in enumerate(model.geom_bodyid)
        if int(body_id) in robot_body_ids
    )
    box_floor_geom_id = _required_id(model, "geom", "box_floor_geom")
    box_geom_ids = frozenset(
        [box_floor_geom_id]
        + [
            _required_id(model, "geom", name)
            for name in (
                "box_wall_pos_x",
                "box_wall_neg_x",
                "box_wall_pos_y",
                "box_wall_neg_y",
            )
        ]
    )
    return PandaSceneIds(
        arm_joint_ids=arm_joint_ids,
        arm_actuator_ids=arm_actuator_ids,
        finger_joint_ids=finger_joint_ids,  # type: ignore[arg-type]
        finger_body_ids=finger_body_ids,  # type: ignore[arg-type]
        robot_body_ids=robot_body_ids,
        robot_geom_ids=robot_geom_ids,
        tcp_site_id=_required_id(model, "site", "panda_tcp"),
        gripper_actuator_id=_required_id(model, "actuator", "actuator8"),
        can_body_id=_required_id(model, "body", "can"),
        can_joint_id=_required_id(model, "joint", "can_free"),
        can_geom_id=_required_id(model, "geom", "can_geom"),
        box_body_id=_required_id(model, "body", "box"),
        box_floor_geom_id=box_floor_geom_id,
        box_geom_ids=box_geom_ids,
        table_geom_id=_required_id(model, "geom", "table_geom"),
    )


def load_panda_scene(
    path: str | Path = DEFAULT_SCENE_PATH,
) -> tuple[mujoco.MjModel, mujoco.MjData, PandaSceneIds]:
    """Load the installed pinned Panda scene and resolve stable named IDs."""
    model = mujoco.MjModel.from_xml_path(str(Path(path)))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    # The upstream robot keyframe naturally has no values for this scene's
    # added free can joint.  Seed that joint from its declared world pose.
    can_body_id = _required_id(model, "body", "can")
    can_joint_id = _required_id(model, "joint", "can_free")
    qpos_address = int(model.jnt_qposadr[can_joint_id])
    data.qpos[qpos_address : qpos_address + 3] = model.body_pos[can_body_id]
    data.qpos[qpos_address + 3 : qpos_address + 7] = model.body_quat[can_body_id]
    mujoco.mj_forward(model, data)
    return model, data, _resolve_scene_ids(model)


def apply_scene_perturbation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: PandaSceneIds,
    perturbation: ScenePerturbation,
) -> None:
    """Apply one B0 can pose, mass, and contact-friction perturbation in place."""
    qpos_address = int(model.jnt_qposadr[ids.can_joint_id])
    model.body_mass[ids.can_body_id] *= perturbation.mass_scale
    model.body_inertia[ids.can_body_id] *= perturbation.mass_scale
    model.geom_friction[ids.can_geom_id] *= perturbation.friction_scale
    # mj_setConst resets its MjData argument.  Recompute model constants from
    # a scratch state so the caller's non-can qpos, qvel, and ctrl survive.
    mujoco.mj_setConst(model, mujoco.MjData(model))
    data.qpos[qpos_address] += perturbation.can_dx_m
    data.qpos[qpos_address + 1] += perturbation.can_dy_m
    yaw_quaternion = np.array(
        [np.cos(perturbation.can_yaw_rad / 2.0), 0.0, 0.0, np.sin(perturbation.can_yaw_rad / 2.0)]
    )
    current_quaternion = data.qpos[qpos_address + 3 : qpos_address + 7].copy()
    mujoco.mju_mulQuat(
        data.qpos[qpos_address + 3 : qpos_address + 7],
        yaw_quaternion,
        current_quaternion,
    )
    mujoco.mj_forward(model, data)

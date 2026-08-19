from pathlib import Path

import mujoco
import numpy as np

from webvideo_to_data.retargeting import RobotReference
from webvideo_to_data.simulation import run_mujoco_replay


ASSET_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "webvideo_to_data"
    / "assets"
    / "panda_pick_place.xml"
)


def _stationary_reference(mode_variant: str = "B0") -> RobotReference:
    return RobotReference(
        timestamps_s=np.array([0.0, 0.2]),
        ee_positions=np.array([[0.12, 0.45, 0.20], [0.12, 0.45, 0.20]]),
        quaternion_wxyz=np.array([[0.0, 1.0, 0.0, 0.0]] * 2),
        gripper_width=np.array([0.08, 0.08]),
        phase=("approach", "approach"),
        source_variant=mode_variant,
    )


def test_headless_mujoco_scene_and_runner_smoke() -> None:
    model = mujoco.MjModel.from_xml_path(str(ASSET_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for _ in range(100):
        mujoco.mj_step(model, data)

    body_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        for index in range(model.nbody)
    }
    joint_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    }
    assert {"can", "box", "panda_hand"} <= body_names
    assert {f"panda_joint{index}" for index in range(1, 8)} <= joint_names
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()

    result = run_mujoco_replay(
        _stationary_reference(),
        mode="physics_grasp",
        model_path=ASSET_PATH,
        max_steps=100,
        render_every=50,
        render_size=(96, 72),
    )

    assert result.mode == "physics_grasp"
    assert result.qpos.shape[0] == 100
    assert result.qvel.shape[0] == 100
    assert result.can_pose.shape == (100, 7)
    assert result.contact_count.shape == (100,)
    np.testing.assert_allclose(result.can_pose[0, :2], [0.12, 0.45], atol=1e-5)
    assert result.can_pose[0, 2] >= 0.044
    assert np.isfinite(result.qpos).all()
    assert np.isfinite(result.qvel).all()
    assert not result.invalid_numerical_state
    assert result.minimum_distance_m >= 0.0
    assert result.target_error_m >= 0.0
    assert result.rendered_rgb.shape == (2, 72, 96, 3)


def test_kinematic_object_replay_is_never_physics_success() -> None:
    reference = RobotReference(
        timestamps_s=np.array([0.0, 0.2, 0.4, 1.6]),
        ee_positions=np.array(
            [
                [0.12, 0.45, 0.045],
                [0.12, 0.45, 0.13],
                [-0.05, 0.55, 0.068],
                [-0.05, 0.55, 0.068],
            ]
        ),
        quaternion_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]] * 4),
        gripper_width=np.array([0.0, 0.0, 0.08, 0.08]),
        phase=("close", "lift", "open", "open"),
        source_variant="B1",
    )

    result = run_mujoco_replay(
        reference,
        mode="kinematic_replay",
        model_path=ASSET_PATH,
        render_every=1000,
    )

    assert result.maximum_lift_m >= 0.03
    assert result.support_contact_duration_s >= 1.0
    assert result.mode == "kinematic_replay"
    assert not result.placed_successfully


def test_stationary_physics_scene_is_not_a_successful_placement() -> None:
    result = run_mujoco_replay(
        _stationary_reference(),
        mode="physics_grasp",
        model_path=ASSET_PATH,
        max_steps=100,
        render_every=1000,
    )

    assert result.maximum_lift_m < 0.03
    assert result.mode == "physics_grasp"
    assert not result.placed_successfully


def test_replay_ik_uses_end_effector_orientation_reference() -> None:
    base = _stationary_reference()
    identity = RobotReference(
        timestamps_s=base.timestamps_s,
        ee_positions=base.ee_positions,
        quaternion_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]] * 2),
        gripper_width=base.gripper_width,
        phase=base.phase,
        source_variant="B0",
    )
    rotated = RobotReference(
        timestamps_s=identity.timestamps_s,
        ee_positions=identity.ee_positions,
        quaternion_wxyz=np.array([[0.0, 1.0, 0.0, 0.0]] * 2),
        gripper_width=identity.gripper_width,
        phase=identity.phase,
        source_variant="B0",
    )

    identity_result = run_mujoco_replay(
        identity,
        model_path=ASSET_PATH,
        max_steps=20,
        render_every=1000,
    )
    rotated_result = run_mujoco_replay(
        rotated,
        model_path=ASSET_PATH,
        max_steps=20,
        render_every=1000,
    )

    assert not np.allclose(identity_result.qpos[-1, 7:14], rotated_result.qpos[-1, 7:14])

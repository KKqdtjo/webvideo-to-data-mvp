from pathlib import Path

import mujoco
import numpy as np

from webvideo_to_data.retargeting import RobotReference, build_pick_place_reference
from webvideo_to_data.schema import PhaseInterval, Trajectory2D
from webvideo_to_data.simulation import _placement_success, run_mujoco_replay


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


def _initial_tcp_pose() -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(ASSET_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "panda_tcp")
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
    return data.site_xpos[site_id].copy(), quaternion


def _b0_reference() -> RobotReference:
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
    return build_pick_place_reference(trajectory, phases, variant="B0")


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
        timestamps_s=np.array([0.0, 0.2, 0.4, 0.5, 1.6]),
        ee_positions=np.array(
            [
                [0.12, 0.45, 0.045],
                [0.12, 0.45, 0.13],
                [-0.05, 0.55, 0.068],
                [-0.05, 0.55, 0.068],
                [-0.05, 0.55, 0.068],
            ]
        ),
        quaternion_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]] * 5),
        gripper_width=np.array([0.0, 0.0, 0.08, 0.08, 0.08]),
        phase=("close", "lift", "open", "retreat", "retreat"),
        source_variant="B1",
    )

    result = run_mujoco_replay(
        reference,
        mode="kinematic_replay",
        model_path=ASSET_PATH,
        render_every=1000,
    )

    assert result.maximum_can_height_gain_m >= 0.03
    assert result.maximum_lift_m == 0.0
    assert not result.grasp_contact.any()
    assert result.support_contact_duration_s >= 1.0
    assert result.final_support_contact
    assert result.target_error_m < 1e-4
    assert result.target_height_error_m > 0.05
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


def test_gripper_closed_evidence_uses_actual_finger_state() -> None:
    initial_position, initial_quaternion = _initial_tcp_pose()
    reference = RobotReference(
        timestamps_s=np.array([0.0]),
        ee_positions=np.array([initial_position]),
        quaternion_wxyz=np.array([initial_quaternion]),
        gripper_width=np.array([0.0]),
        phase=("lift",),
        source_variant="B0",
    )

    result = run_mujoco_replay(
        reference,
        mode="physics_grasp",
        model_path=ASSET_PATH,
        max_steps=1,
        render_every=1000,
    )

    assert result.gripper_width_m[0] > 0.07
    assert not result.gripper_closed[0]


def test_support_duration_resets_if_contact_is_lost_after_release() -> None:
    reference = RobotReference(
        timestamps_s=np.array([0.0, 1.1, 1.2]),
        ee_positions=np.array(
            [
                [-0.05, 0.55, 0.068],
                [-0.05, 0.55, 0.068],
                [0.20, 0.20, 0.20],
            ]
        ),
        quaternion_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]] * 3),
        gripper_width=np.array([0.08, 0.08, 0.08]),
        phase=("open", "open", "retreat"),
        source_variant="B1",
    )

    result = run_mujoco_replay(
        reference,
        mode="kinematic_replay",
        model_path=ASSET_PATH,
        render_every=1000,
    )

    assert result.support_contact_duration_s == 0.0
    assert not result.final_support_contact
    assert not result.placed_successfully


def test_box_placement_uses_wall_interior_minus_can_radius() -> None:
    reference = RobotReference(
        timestamps_s=np.array([0.0, 1.1]),
        ee_positions=np.array([[0.01, 0.55, 0.068]] * 2),
        quaternion_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]] * 2),
        gripper_width=np.array([0.08, 0.08]),
        phase=("open", "open"),
        source_variant="B1",
    )

    result = run_mujoco_replay(
        reference,
        mode="kinematic_replay",
        model_path=ASSET_PATH,
        render_every=1000,
    )

    assert not result.finishes_inside_box


def test_replay_ik_uses_end_effector_orientation_reference() -> None:
    initial_position, initial_quaternion = _initial_tcp_pose()
    rotated_quaternion = np.empty(4)
    mujoco.mju_mulQuat(
        rotated_quaternion,
        np.array([0.0, 1.0, 0.0, 0.0]),
        initial_quaternion,
    )
    identity = RobotReference(
        timestamps_s=np.array([0.0, 0.1]),
        ee_positions=np.tile(initial_position, (2, 1)),
        quaternion_wxyz=np.tile(initial_quaternion, (2, 1)),
        gripper_width=np.array([0.08, 0.08]),
        phase=("approach", "approach"),
        source_variant="B0",
    )
    rotated = RobotReference(
        timestamps_s=identity.timestamps_s,
        ee_positions=identity.ee_positions,
        quaternion_wxyz=np.tile(rotated_quaternion, (2, 1)),
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
    assert identity_result.ik_orientation_error_rad[0] < 1e-8
    assert identity_result.reachability_ratio >= 0.95
    assert rotated_result.ik_orientation_error_rad[0] > 3.0
    assert not rotated_result.ik_converged[0]


def test_ik_residuals_distinguish_reachable_and_unreachable_references() -> None:
    initial_position, initial_quaternion = _initial_tcp_pose()
    reachable = RobotReference(
        timestamps_s=np.array([0.0, 0.1]),
        ee_positions=np.tile(initial_position, (2, 1)),
        quaternion_wxyz=np.tile(initial_quaternion, (2, 1)),
        gripper_width=np.array([0.08, 0.08]),
        phase=("approach", "approach"),
        source_variant="B0",
    )
    unreachable = RobotReference(
        timestamps_s=reachable.timestamps_s,
        ee_positions=np.array([[5.0, 5.0, 5.0]] * 2),
        quaternion_wxyz=reachable.quaternion_wxyz,
        gripper_width=reachable.gripper_width,
        phase=reachable.phase,
        source_variant="B0",
    )

    reachable_result = run_mujoco_replay(
        reachable, model_path=ASSET_PATH, max_steps=20, render_every=1000
    )
    unreachable_result = run_mujoco_replay(
        unreachable, model_path=ASSET_PATH, max_steps=20, render_every=1000
    )

    assert reachable_result.ik_position_error_m.shape == (20,)
    assert reachable_result.ik_orientation_error_rad.shape == (20,)
    assert reachable_result.ik_converged.shape == (20,)
    assert reachable_result.reachability_ratio >= 0.95
    assert unreachable_result.reachability_ratio <= 0.05
    assert reachable_result.ik_position_error_m.max() < 0.03
    assert unreachable_result.ik_position_error_m.min() > 1.0


def test_physics_success_requires_at_least_95_percent_reachability() -> None:
    evidence = dict(
        mode="physics_grasp",
        qualifying_lift_m=0.03,
        finishes_inside_box=True,
        support_contact_duration_s=1.0,
        final_support_contact=True,
        invalid_numerical_state=False,
    )

    assert _placement_success(**evidence, reachability_ratio=0.95)
    assert not _placement_success(**evidence, reachability_ratio=0.949)


def test_real_b0_physics_replay_reports_measured_controller_failure() -> None:
    result = run_mujoco_replay(
        _b0_reference(),
        mode="physics_grasp",
        model_path=ASSET_PATH,
        render_every=100000,
    )

    assert result.mode == "physics_grasp"
    assert not result.invalid_numerical_state
    assert result.reachability_ratio < 0.10
    assert not result.grasp_contact.any()
    assert result.maximum_lift_m == 0.0
    assert not result.final_support_contact
    assert result.target_error_m > 0.10
    assert not result.placed_successfully

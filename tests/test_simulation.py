from pathlib import Path

import mujoco
import numpy as np
import pytest

from webvideo_to_data.retargeting import RobotReference, build_pick_place_reference
from webvideo_to_data.schema import PhaseInterval, Trajectory2D
from webvideo_to_data.simulation import (
    _GraspLiftTracker,
    _set_joint_control,
    _finger_contact_state,
    _placement_success,
    run_mujoco_replay,
    run_joint_control_program,
)
from webvideo_to_data.ik import JointControlProgram
from webvideo_to_data.config import load_experiment_config
from webvideo_to_data.physics_validation import (
    InvalidNumericalStateError,
    validate_rollout,
)
from tests.helpers import write_complete_config


ASSET_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "webvideo_to_data"
    / "assets"
    / "panda_pick_place.xml"
)

GRASP_CONTACT_FIXTURE_XML = """
<mujoco model="grasp_contact_fixture">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="can" pos="0 0 0.10">
      <freejoint name="can_free"/>
      <geom name="can_geom" type="cylinder" size="0.025 0.045" mass="0.1"/>
    </body>
    <body name="left_finger" pos="0 0.02 0.10">
      <geom name="left_finger_geom" type="box" size="0.012 0.008 0.045"/>
    </body>
    <body name="right_finger" pos="0 -0.02 0.10">
      <geom name="right_finger_geom" type="box" size="0.012 0.008 0.045"/>
    </body>
  </worldbody>
</mujoco>
"""


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


def _real_finger_contacts(can_y: float) -> tuple[bool, bool]:
    model = mujoco.MjModel.from_xml_string(GRASP_CONTACT_FIXTURE_XML)
    data = mujoco.MjData(model)
    can_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "can_free")
    can_qpos_address = model.jnt_qposadr[can_joint_id]
    data.qpos[can_qpos_address : can_qpos_address + 3] = [0.0, can_y, 0.10]
    data.qpos[can_qpos_address + 3 : can_qpos_address + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    can_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "can_geom")
    finger_geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("left_finger_geom", "right_finger_geom")
    ]
    return _finger_contact_state(data, can_geom_id, finger_geom_ids)


@pytest.mark.requires_renderer
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
        render=False,
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
        render=False,
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
        render=False,
    )

    assert result.gripper_width_m[0] > 0.07
    assert not result.gripper_closed[0]


def test_single_finger_or_two_frame_contact_cannot_qualify_lift() -> None:
    single_finger = _real_finger_contacts(can_y=0.03)
    both_fingers = _real_finger_contacts(can_y=0.0)
    assert sum(single_finger) == 1
    assert both_fingers == (True, True)
    tracker = _GraspLiftTracker(initial_can_z=0.10)

    tracker.observe("lift", True, single_finger, can_z=0.14)
    tracker.observe("lift", True, both_fingers, can_z=0.15)
    tracker.observe("transport", True, both_fingers, can_z=0.16)

    assert tracker.maximum_lift_m == 0.0


def test_three_frame_double_finger_contact_qualifies_and_loss_resets_streak() -> None:
    both_fingers = _real_finger_contacts(can_y=0.0)
    single_finger = _real_finger_contacts(can_y=0.03)
    tracker = _GraspLiftTracker(initial_can_z=0.10)

    tracker.observe("lift", True, both_fingers, can_z=0.11)
    tracker.observe("lift", True, both_fingers, can_z=0.12)
    tracker.observe("lift", True, single_finger, can_z=0.20)
    tracker.observe("transport", True, both_fingers, can_z=0.13)
    tracker.observe("transport", True, both_fingers, can_z=0.14)
    assert tracker.maximum_lift_m == 0.0

    tracker.observe("transport", True, both_fingers, can_z=0.15)

    assert tracker.maximum_lift_m == pytest.approx(0.05)


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
        render=False,
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
        render=False,
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
        render=False,
    )
    rotated_result = run_mujoco_replay(
        rotated,
        model_path=ASSET_PATH,
        max_steps=20,
        render_every=1000,
        render=False,
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
        reachable,
        model_path=ASSET_PATH,
        max_steps=20,
        render_every=1000,
        render=False,
    )
    unreachable_result = run_mujoco_replay(
        unreachable,
        model_path=ASSET_PATH,
        max_steps=20,
        render_every=1000,
        render=False,
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
        render=False,
    )

    assert result.mode == "physics_grasp"
    assert not result.invalid_numerical_state
    assert result.reachability_ratio < 0.10
    assert not result.grasp_contact.any()
    assert result.maximum_lift_m == 0.0
    assert not result.final_support_contact
    assert result.target_error_m > 0.10
    assert not result.placed_successfully


def test_render_false_does_not_construct_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch --no-render still constructing an OpenGL renderer/context."""

    def fail_renderer(*args: object, **kwargs: object) -> object:
        raise AssertionError("Renderer must not be constructed")

    monkeypatch.setattr(mujoco, "Renderer", fail_renderer)

    result = run_mujoco_replay(
        _stationary_reference(),
        mode="physics_grasp",
        max_steps=2,
        render=False,
    )

    assert result.rendered_rgb.shape == (0, 240, 320, 3)


def test_b0_joint_control_writes_precomputed_official_actuator_targets() -> None:
    from webvideo_to_data.scene import load_panda_scene

    model, data, ids = load_panda_scene()
    arm_targets = np.array([[0.1, -0.2, 0.3, -1.2, 0.4, 1.1, -0.5]])
    program = JointControlProgram(
        timestamps_s=np.array([0.0]),
        arm_qpos_targets=arm_targets,
        gripper_ctrl=np.array([255.0]),
        ee_positions=np.array([[0.12, 0.45, 0.20]]),
        quaternion_wxyz=np.array([[0.0, 1.0, 0.0, 0.0]]),
        phase=("home",),
        keyframe_ik=(),
    )

    _set_joint_control(data, ids, program, 0)

    np.testing.assert_allclose(data.ctrl[list(ids.arm_actuator_ids)], arm_targets[0])
    assert data.ctrl[ids.gripper_actuator_id] == 255.0


def _stationary_joint_program(
    model: mujoco.MjModel, data: mujoco.MjData, ids: object, *, rotated: bool = False
) -> JointControlProgram:
    arm_joint_ids = np.asarray(ids.arm_joint_ids)
    arm_qpos = data.qpos[model.jnt_qposadr[arm_joint_ids]].copy()
    position = data.site_xpos[ids.tcp_site_id].copy()
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, data.site_xmat[ids.tcp_site_id])
    if rotated:
        desired = np.empty(4)
        mujoco.mju_mulQuat(
            desired, np.array([0.0, 1.0, 0.0, 0.0]), quaternion
        )
        quaternion = desired
    return JointControlProgram(
        timestamps_s=np.array([0.0, 0.01, 0.02]),
        arm_qpos_targets=np.tile(arm_qpos, (3, 1)),
        gripper_ctrl=np.full(3, 255.0),
        ee_positions=np.tile(position, (3, 1)),
        quaternion_wxyz=np.tile(quaternion, (3, 1)),
        phase=("home", "home", "home"),
        keyframe_ik=(),
    )


def test_joint_program_records_measured_control_contact_and_safety_arrays(
    tmp_path: Path,
) -> None:
    from webvideo_to_data.scene import load_panda_scene

    config = load_experiment_config(write_complete_config(tmp_path))
    model, data, ids = load_panda_scene()
    program = _stationary_joint_program(model, data, ids)

    result = run_joint_control_program(
        model,
        data,
        ids,
        program,
        ik=config.ik,
        control_config=config.control,
        collision_config=config.collision,
        max_steps=5,
        render=False,
    )

    assert result.timestamps_s.shape == (5,)
    assert result.control.shape == (5, model.nu)
    assert result.qpos.shape == (5, model.nq)
    assert result.qvel.shape == (5, model.nv)
    assert result.tcp_position.shape == (5, 3)
    assert result.tcp_quaternion_wxyz.shape == (5, 4)
    assert result.phase == ("home",) * 5
    for values in (
        result.contact_count,
        result.forbidden_contact,
        result.maximum_penetration_m,
        result.bilateral_contact,
        result.box_support_contact,
        result.tcp_position_within_tolerance,
        result.tcp_orientation_within_tolerance,
        result.joint_position_violation,
        result.joint_velocity_violation,
        result.joint_acceleration_violation,
        result.valid_numerical_state,
    ):
        assert values.shape == (5,)
    np.testing.assert_allclose(
        result.control[:, list(ids.arm_actuator_ids)],
        program.arm_qpos_targets[[0, 0, 0, 0, 0]],
    )
    assert result.tcp_position_within_tolerance.all()
    assert result.tcp_orientation_within_tolerance.all()
    assert result.execution_tracking_ratio == 1.0
    assert result.valid_numerical_state.all()
    assert result.placed_successfully is validate_rollout(
        result, config.collision
    ).passed


def test_execution_tracking_ratio_requires_measured_position_and_orientation(
    tmp_path: Path,
) -> None:
    from webvideo_to_data.scene import load_panda_scene

    config = load_experiment_config(write_complete_config(tmp_path))
    model, data, ids = load_panda_scene()
    program = _stationary_joint_program(model, data, ids, rotated=True)

    result = run_joint_control_program(
        model,
        data,
        ids,
        program,
        ik=config.ik,
        control_config=config.control,
        collision_config=config.collision,
        max_steps=2,
        render=False,
    )

    assert result.tcp_position_within_tolerance.all()
    assert not result.tcp_orientation_within_tolerance.any()
    assert result.execution_tracking_ratio == 0.0


def test_first_step_acceleration_uses_initial_measured_velocity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from webvideo_to_data.scene import load_panda_scene

    config = load_experiment_config(write_complete_config(tmp_path))
    model, data, ids = load_panda_scene()
    program = _stationary_joint_program(model, data, ids)
    arm_dofs = model.jnt_dofadr[np.asarray(ids.arm_joint_ids)]
    real_step = mujoco.mj_step

    def fixed_poststep_velocity(
        step_model: mujoco.MjModel, step_data: mujoco.MjData
    ) -> None:
        real_step(step_model, step_data)
        step_data.qvel[arm_dofs] = 0.1

    monkeypatch.setattr(mujoco, "mj_step", fixed_poststep_velocity)

    result = run_joint_control_program(
        model,
        data,
        ids,
        program,
        ik=config.ik,
        control_config=config.control,
        collision_config=config.collision,
        max_steps=2,
        render=False,
    )

    assert result.joint_acceleration_violation.tolist() == [True, False]
    assert not result.joint_velocity_violation.any()


def test_joint_program_raises_immediately_on_first_nonfinite_measured_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from webvideo_to_data.scene import load_panda_scene

    config = load_experiment_config(write_complete_config(tmp_path))
    model, data, ids = load_panda_scene()
    program = _stationary_joint_program(model, data, ids)
    real_step = mujoco.mj_step

    def corrupt_first_step(step_model: mujoco.MjModel, step_data: mujoco.MjData) -> None:
        real_step(step_model, step_data)
        step_data.qvel[0] = np.nan

    monkeypatch.setattr(mujoco, "mj_step", corrupt_first_step)

    with pytest.raises(InvalidNumericalStateError) as caught:
        run_joint_control_program(
            model,
            data,
            ids,
            program,
            ik=config.ik,
            control_config=config.control,
            collision_config=config.collision,
            max_steps=5,
            render=False,
        )

    assert caught.value.timestamp_s == pytest.approx(model.opt.timestep)

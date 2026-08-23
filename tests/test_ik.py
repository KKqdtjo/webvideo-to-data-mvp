from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from helpers import write_complete_config
from webvideo_to_data.config import load_experiment_config
from webvideo_to_data.ik import (
    ControlLimitError,
    IKPlanningError,
    plan_joint_control,
    solve_pose_ik,
)
from webvideo_to_data.retargeting import build_manual_b0_reference
from webvideo_to_data.scene import load_panda_scene


def _config_path(tmp_path: Path) -> Path:
    return write_complete_config(
        tmp_path, {"source.sha256": "0" * 64}, with_source=False
    )


def _tcp_pose(
    model: mujoco.MjModel, data: mujoco.MjData, site_id: int
) -> tuple[np.ndarray, np.ndarray]:
    mujoco.mj_forward(model, data)
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
    return data.site_xpos[site_id].copy(), quaternion


def _independent_pose_residual(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_ids: tuple[int, ...],
    site_id: int,
    arm_qpos: np.ndarray,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
) -> tuple[float, float]:
    working = mujoco.MjData(model)
    working.qpos[:] = data.qpos
    qpos_addresses = np.asarray(
        [model.jnt_qposadr[joint_id] for joint_id in joint_ids]
    )
    working.qpos[qpos_addresses] = arm_qpos
    actual_position, actual_quaternion = _tcp_pose(model, working, site_id)
    actual_inverse = actual_quaternion * np.array([1.0, -1.0, -1.0, -1.0])
    quaternion_error = np.empty(4, dtype=float)
    mujoco.mju_mulQuat(
        quaternion_error,
        target_quaternion / np.linalg.norm(target_quaternion),
        actual_inverse,
    )
    if quaternion_error[0] < 0.0:
        quaternion_error *= -1.0
    rotation_error = np.empty(3, dtype=float)
    mujoco.mju_quat2Vel(rotation_error, quaternion_error, 1.0)
    return (
        float(np.linalg.norm(target_position - actual_position)),
        float(np.linalg.norm(rotation_error)),
    )


def _normalized_joint_center_cost(
    model: mujoco.MjModel, joint_ids: tuple[int, ...], arm_qpos: np.ndarray
) -> float:
    limits = model.jnt_range[np.asarray(joint_ids)]
    center = np.mean(limits, axis=1)
    span = limits[:, 1] - limits[:, 0]
    return float(np.sum(((arm_qpos - center) / span) ** 2))


def test_pose_ik_converges_to_reachable_target_and_respects_limits(
    tmp_path: Path,
) -> None:
    model, data, ids = load_panda_scene()
    config = load_experiment_config(
        _config_path(tmp_path)
    )
    target = np.array([0.12, 0.45, 0.20])
    quaternion = np.array([0.0, 1.0, 0.0, 0.0])

    result = solve_pose_ik(
        model, data, ids, target, quaternion, config.ik, None
    )

    assert result.converged
    assert result.position_error_m <= 0.005
    assert result.orientation_error_rad <= np.deg2rad(5.0)
    for value, joint_id in zip(result.arm_qpos, ids.arm_joint_ids):
        assert model.jnt_range[joint_id, 0] <= value <= model.jnt_range[joint_id, 1]


def test_pose_ik_rejects_unreachable_target_after_exact_iteration_cap(
    tmp_path: Path,
) -> None:
    model, data, ids = load_panda_scene()
    config = load_experiment_config(
        _config_path(tmp_path)
    )

    result = solve_pose_ik(
        model,
        data,
        ids,
        np.array([5.0, 5.0, 5.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        config.ik,
        None,
    )

    assert not result.converged
    assert result.iterations == 200
    assert result.position_error_m > 1.0


def test_pose_ik_cap_residual_matches_returned_joint_pose(tmp_path: Path) -> None:
    model, data, ids = load_panda_scene()
    config = load_experiment_config(_config_path(tmp_path))
    target_position = np.array([0.12, 0.45, 0.20])
    target_quaternion = np.array([0.0, 1.0, 0.0, 0.0])

    result = solve_pose_ik(
        model,
        data,
        ids,
        target_position,
        target_quaternion,
        replace(config.ik, maximum_iterations=1),
        None,
    )
    position_error, orientation_error = _independent_pose_residual(
        model,
        data,
        ids.arm_joint_ids,
        ids.tcp_site_id,
        result.arm_qpos,
        target_position,
        target_quaternion,
    )

    assert result.iterations == 1
    assert result.position_error_m == pytest.approx(position_error, abs=1e-12)
    assert result.orientation_error_rad == pytest.approx(
        orientation_error, abs=1e-12
    )


def test_pose_ik_last_update_at_iteration_cap_can_converge(tmp_path: Path) -> None:
    model, data, ids = load_panda_scene()
    config = load_experiment_config(_config_path(tmp_path))
    initial_position, initial_quaternion = _tcp_pose(model, data, ids.tcp_site_id)
    target_position = initial_position + np.array([0.01, 0.0, 0.0])
    options = replace(
        config.ik,
        maximum_iterations=1,
        position_tolerance_m=0.009,
    )

    result = solve_pose_ik(
        model,
        data,
        ids,
        target_position,
        initial_quaternion,
        options,
        None,
    )

    assert result.iterations == 1
    assert result.converged
    assert result.position_error_m <= 0.009
    assert result.orientation_error_rad <= options.orientation_tolerance_rad


def test_pose_ik_uses_target_orientation_not_only_position(tmp_path: Path) -> None:
    model, data, ids = load_panda_scene()
    config = load_experiment_config(
        _config_path(tmp_path)
    )
    target = np.array([0.12, 0.45, 0.20])

    downward = solve_pose_ik(
        model,
        data,
        ids,
        target,
        np.array([0.0, 1.0, 0.0, 0.0]),
        config.ik,
        None,
    )
    yawed = solve_pose_ik(
        model,
        data,
        ids,
        target,
        np.array([0.0, 0.0, 1.0, 0.0]),
        config.ik,
        None,
    )

    assert downward.converged and yawed.converged
    assert not np.allclose(downward.arm_qpos, yawed.arm_qpos, atol=1e-3)
    assert downward.orientation_error_rad != pytest.approx(
        yawed.orientation_error_rad, abs=1e-6
    )


def test_pose_ik_projects_joint_center_gradient_into_task_nullspace(
    tmp_path: Path,
) -> None:
    model, data, ids = load_panda_scene()
    config = load_experiment_config(
        _config_path(tmp_path)
    )
    qpos_addresses = np.asarray(
        [model.jnt_qposadr[joint_id] for joint_id in ids.arm_joint_ids]
    )
    first_initial = data.qpos[qpos_addresses].copy()
    second_initial = first_initial + np.array([0.35, -0.25, 0.30, -0.20, 0.25, 0.20, -0.30])
    limits = model.jnt_range[np.asarray(ids.arm_joint_ids)]
    second_initial = np.clip(second_initial, limits[:, 0], limits[:, 1])
    target = np.array([0.12, 0.45, 0.20])
    quaternion = np.array([0.0, 1.0, 0.0, 0.0])

    weighted = [
        solve_pose_ik(
            model, data, ids, target, quaternion, config.ik, initial
        )
        for initial in (first_initial, second_initial)
    ]
    unweighted = solve_pose_ik(
        model,
        data,
        ids,
        target,
        quaternion,
        replace(config.ik, joint_limit_weight=0.0),
        second_initial,
    )

    assert all(result.converged for result in weighted)
    assert unweighted.converged
    assert _normalized_joint_center_cost(
        model, ids.arm_joint_ids, weighted[1].arm_qpos
    ) < _normalized_joint_center_cost(model, ids.arm_joint_ids, unweighted.arm_qpos)


def test_control_program_refuses_any_unconverged_key_pose(tmp_path: Path) -> None:
    model, data, ids = load_panda_scene()
    config = load_experiment_config(
        _config_path(tmp_path)
    )
    reference = build_manual_b0_reference(
        config.scene.b0_start_m,
        config.scene.b0_goal_m,
        config.control,
        config.scene.grasp_quaternion_wxyz,
    )
    reference = replace(
        reference, ee_positions=np.full_like(reference.ee_positions, 5.0)
    )

    with pytest.raises(IKPlanningError, match="IK failed for phase home") as caught:
        plan_joint_control(model, data, ids, reference, config.ik, config.control)

    assert caught.value.phase == "home"
    assert "5.0" not in str(caught.value)


def test_control_program_is_100_hz_limit_bounded_and_maps_gripper(
    tmp_path: Path,
) -> None:
    model, data, ids = load_panda_scene()
    config = load_experiment_config(
        _config_path(tmp_path)
    )
    reference = build_manual_b0_reference(
        config.scene.b0_start_m,
        config.scene.b0_goal_m,
        config.control,
        config.scene.grasp_quaternion_wxyz,
    )

    program = plan_joint_control(
        model, data, ids, reference, config.ik, config.control
    )

    np.testing.assert_allclose(np.diff(program.timestamps_s), 0.01, atol=1e-12)
    velocity = np.diff(program.arm_qpos_targets, axis=0) / 0.01
    acceleration = np.diff(velocity, axis=0) / 0.01
    assert np.max(np.abs(velocity)) <= 2.0 + 1e-9
    assert np.max(np.abs(acceleration)) <= 8.0 + 1e-9
    assert set(program.gripper_ctrl) == {0.0, 255.0}
    assert tuple(dict.fromkeys(program.phase)) == (
        "home",
        "pregrasp",
        "approach",
        "close",
        "lift",
        "transport",
        "lower",
        "open",
        "retreat",
        "settle",
    )
    assert all(result.converged for result in program.keyframe_ik)


def test_planning_exceptions_have_stable_redacted_fields() -> None:
    ik_error = IKPlanningError("transport")
    limit_error = ControlLimitError("lower", "acceleration")

    assert ik_error.phase == "transport"
    assert str(ik_error) == "IK failed for phase transport"
    assert limit_error.phase == "lower"
    assert limit_error.limit == "acceleration"
    assert str(limit_error) == "control limit acceleration failed for phase lower"

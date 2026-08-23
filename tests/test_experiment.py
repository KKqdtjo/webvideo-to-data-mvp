from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
import shutil
import threading
import time

import cv2
import mujoco
import numpy as np
import pytest
import yaml

import webvideo_to_data.artifacts as artifacts_module
import webvideo_to_data.experiment as experiment_module
from webvideo_to_data.artifacts import verify_run_directory
from webvideo_to_data.experiment import _action_gate, _validate_mp4, run_experiment
from webvideo_to_data.media import sha256_file
from webvideo_to_data.physics_validation import PhysicsValidationResult
from webvideo_to_data.ik import plan_joint_control
from webvideo_to_data.retargeting import build_manual_b0_reference
from webvideo_to_data.scene import load_panda_scene
from tests.helpers import VALID_CONFIG


def test_b0_planned_interpolation_starts_continuously_and_records_target_fk(
    tmp_path: Path,
) -> None:
    """Catch an initial actuator jump or a Cartesian trace unrelated to joint targets."""

    config_path = _synthetic_config(tmp_path)
    config = experiment_module.load_experiment_config(config_path)
    model, data, ids = load_panda_scene()
    arm_qpos_addresses = model.jnt_qposadr[np.asarray(ids.arm_joint_ids)]
    initial_arm_qpos = data.qpos[arm_qpos_addresses].copy()
    reference = build_manual_b0_reference(
        config.scene.b0_start_m,
        config.scene.b0_goal_m,
        config.control,
        config.scene.grasp_quaternion_wxyz,
    )

    program = plan_joint_control(
        model, data, ids, reference, config.ik, config.control
    )

    np.testing.assert_allclose(program.arm_qpos_targets[0], initial_arm_qpos)
    scratch = mujoco.MjData(model)
    for index in range(len(program.timestamps_s)):
        scratch.qpos[:] = data.qpos
        scratch.qpos[arm_qpos_addresses] = program.arm_qpos_targets[index]
        mujoco.mj_forward(model, scratch)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, scratch.site_xmat[ids.tcp_site_id])
        np.testing.assert_allclose(
            program.ee_positions[index], scratch.site_xpos[ids.tcp_site_id], atol=1e-10
        )
        assert abs(float(np.dot(program.quaternion_wxyz[index], quaternion))) == pytest.approx(
            1.0, abs=1e-10
        )


def _write_moving_object_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 72)
    )
    assert writer.isOpened()
    try:
        for frame_index in range(24):
            frame = np.zeros((72, 96, 3), dtype=np.uint8)
            x = 18 + min(10, max(0, frame_index - 4))
            cv2.rectangle(frame, (x, 24), (x + 19, 43), (230, 230, 230), -1)
            cv2.line(frame, (x, 24), (x + 19, 43), (10, 10, 10), 2)
            cv2.line(frame, (x + 19, 24), (x, 43), (10, 10, 10), 2)
            writer.write(frame)
    finally:
        writer.release()


def _write_config(
    path: Path,
    source: Path,
    *,
    minimum_valid_ratio: float = 0.1,
    forward_backward_threshold_px: float = 1.5,
    b0_start_m: tuple[float, float, float] = (0.12, 0.45, 0.04),
    b0_goal_m: tuple[float, float, float] = (-0.05, 0.55, 0.13),
) -> None:
    config = deepcopy(VALID_CONFIG)
    config["source"].update(
        path=str(source), sha256=sha256_file(source), roi_xywh=[18, 24, 20, 20]
    )
    config["tracking"].update(
        forward_backward_threshold_px=forward_backward_threshold_px,
        minimum_valid_ratio=minimum_valid_ratio,
    )
    config["scene"].update(b0_start_m=list(b0_start_m), b0_goal_m=list(b0_goal_m))
    config["simulation"].update(render_size=[96, 72], render_every=20)
    path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _synthetic_config(
    tmp_path: Path,
    *,
    minimum_valid_ratio: float = 0.1,
    forward_backward_threshold_px: float = 1.5,
    b0_start_m: tuple[float, float, float] = (0.12, 0.45, 0.04),
    b0_goal_m: tuple[float, float, float] = (-0.05, 0.55, 0.13),
) -> Path:
    source = tmp_path / "moving.mp4"
    _write_moving_object_video(source)
    config_path = tmp_path / "experiment.yaml"
    _write_config(
        config_path,
        source,
        minimum_valid_ratio=minimum_valid_ratio,
        forward_backward_threshold_px=forward_backward_threshold_px,
        b0_start_m=b0_start_m,
        b0_goal_m=b0_goal_m,
    )
    return config_path


def _assert_readable_video(path: Path) -> None:
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) > 0
        assert capture.get(cv2.CAP_PROP_FPS) > 0
    finally:
        capture.release()


def _trusted_no_action_output(tmp_path: Path) -> tuple[Path, Path]:
    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "trusted-no-action-run"
    run_experiment(config_path, output_dir, variant="B0", no_render=True)
    assert experiment_module._trusted_run_snapshot(output_dir)
    assert not (output_dir / "actions.npz").exists()
    return config_path, output_dir


def test_run_experiment_writes_auditable_orchestration_outputs(tmp_path: Path) -> None:
    """Catch a runner that skips a pipeline stage or omits its audit metrics."""

    config_path = _synthetic_config(tmp_path)
    source = tmp_path / "moving.mp4"
    output_dir = tmp_path / "run"

    metrics = run_experiment(config_path, output_dir, variant="B1")

    expected = {
        "provenance.json",
        "trajectory_2d.npz",
        "trajectory_2d.schema.json",
        "phases.json",
        "metrics.json",
        "tracking_overlay.mp4",
        "trajectory_2d.png",
        "mujoco_replay.mp4",
        "side_by_side.mp4",
        "contact_sheet.png",
        "robot_reference.npz",
        "robot_reference.schema.json",
        "simulation.npz",
        "simulation.schema.json",
        "rejection.json",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    with np.load(output_dir / "trajectory_2d.npz") as trajectory:
        assert set(trajectory.files) == {"timestamps_s", "centers_px", "confidence"}
    on_disk = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics == on_disk
    assert {
        "source_sha256",
        "lk_point_availability_ratio",
        "phase_count",
        "variant",
        "simulation_mode",
        "placed_successfully",
    } <= on_disk.keys()
    assert "valid_track_ratio" not in on_disk
    assert on_disk["lk_metric_scope"] == "point_availability_not_semantic_accuracy"
    assert on_disk["semantic_accuracy_status"] == "not_measured"
    assert on_disk["source_sha256"] == sha256_file(source)
    assert on_disk["variant"] == "B1"
    assert on_disk["simulation_mode"] == "kinematic_replay"
    assert on_disk["placed_successfully"] is False
    assert on_disk["status"] == "rejected"
    assert on_disk["reason"] == "kinematic_replay_not_action"
    assert on_disk["action_export_eligible"] is False
    assert on_disk["collision_validation"] == "not_applicable_kinematic"
    assert on_disk["physics_validation"] == "not_applicable_kinematic"
    assert on_disk["action_export_reason"] == "kinematic_replay_not_action"
    assert not (output_dir / "actions.npz").exists()
    for name in ("tracking_overlay.mp4", "mujoco_replay.mp4", "side_by_side.mp4"):
        _assert_readable_video(output_dir / name)
    comparison_capture = cv2.VideoCapture(str(output_dir / "side_by_side.mp4"))
    try:
        assert int(comparison_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 960
        assert int(comparison_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 320
    finally:
        comparison_capture.release()
    for name in ("trajectory_2d.png", "contact_sheet.png"):
        assert cv2.imread(str(output_dir / name)) is not None
    provenance = json.loads(
        (output_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["generator"]["git_commit"] == subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    assert isinstance(provenance["generator"]["git_dirty"], bool)
    assert provenance["config"]["sha256"] == sha256_file(config_path)
    assert provenance["config"]["resolved"]["experiment_id"] == "SYNTHETIC"
    assert str(tmp_path.resolve()) not in json.dumps(provenance, sort_keys=True)
    assert provenance["source"]["id"] == "synthetic-moving-object"
    assert provenance["source"]["sha256"] == sha256_file(source)
    assert provenance["model"]["sha256"] == artifacts_module.pinned_model_identity("B1")
    assert provenance["model"]["id"] == "primitive_7dof_panda_like_diagnostic"
    assert provenance["model"]["description"] == "primitive_7dof_panda_like_diagnostic"
    assert provenance["runtime"]["python"]
    assert provenance["runtime"]["packages"]["mujoco"]
    assert provenance["runtime"]["packages"]["opencv-python"]
    source_probe = provenance["ffprobe"]["source"]
    assert source_probe["codec_name"]
    assert source_probe["width"] == 96
    assert source_probe["height"] == 72
    assert source_probe["duration_s"] > 0.0
    assert set(provenance["ffprobe"]["generated_media"]) == {
        "mujoco_replay.mp4",
        "side_by_side.mp4",
        "tracking_overlay.mp4",
    }
    verified = verify_run_directory(output_dir)
    manifest = verified.manifest
    assert manifest["format_version"] == 4
    assert {
        "status": "rejected",
        "reason": "kinematic_replay_not_action",
        "action_export_eligible": False,
        "action_exported": False,
        "config_sha256": sha256_file(config_path),
        "source_sha256": sha256_file(source),
        "model_sha256": artifacts_module.pinned_model_identity("B1"),
    }.items() <= manifest.items()
    assert all(manifest[key] > 0 for key in ("model_nq", "model_nv", "model_nu"))
    for name, entry in manifest["files"].items():
        assert entry["size"] == (output_dir / name).stat().st_size
        assert entry["sha256"] == sha256_file(output_dir / name)
        if Path(name).suffix in {".mp4", ".png", ".gif"}:
            assert isinstance(entry["media_role"], str)
            assert isinstance(entry["contains_private_source_frames"], bool)
        else:
            assert "media_role" not in entry
            assert "contains_private_source_frames" not in entry
    assert manifest["files"]["mujoco_replay.mp4"]["media_role"] == "simulation_only"
    assert manifest["files"]["mujoco_replay.mp4"][
        "contains_private_source_frames"
    ] is False
    assert manifest["files"]["tracking_overlay.mp4"][
        "contains_private_source_frames"
    ] is True
    manifest_path = output_dir / "run_manifest.json"
    original_manifest_text = manifest_path.read_text(encoding="utf-8")
    for mutate in (
        lambda value: value.update(format_version=4.0),
        lambda value: value.pop("config_sha256"),
        lambda value: value.update(model_nu=value["model_nu"] + 1),
        lambda value: value["files"]["metrics.json"].update(size=True),
        lambda value: value["files"]["metrics.json"].update(sha256="0" * 64),
        lambda value: value["files"]["tracking_overlay.mp4"].update(
            contains_private_source_frames="yes"
        ),
        lambda value: value["files"]["mujoco_replay.mp4"].update(
            contains_private_source_frames=True
        ),
        lambda value: value["files"]["mujoco_replay.mp4"].update(
            media_role="source_tracking_overlay"
        ),
    ):
        altered = deepcopy(manifest)
        mutate(altered)
        manifest_path.write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises(ValueError, match="trusted run directory"):
            verify_run_directory(output_dir)
        manifest_path.write_text(original_manifest_text, encoding="utf-8")
    assert verify_run_directory(output_dir).manifest == manifest

    sidecar_path = output_dir / "trajectory_2d.schema.json"
    original_sidecar = sidecar_path.read_bytes()
    sidecar = json.loads(original_sidecar)
    sidecar["provenance"]["producer"] = "untrusted.producer"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    experiment_module._write_run_manifest(output_dir, dict(verified.metrics))
    with pytest.raises(ValueError, match="trusted run directory"):
        verify_run_directory(output_dir)
    sidecar_path.write_bytes(original_sidecar)
    manifest_path.write_text(original_manifest_text, encoding="utf-8")


def test_metric_depth_variants_are_not_run_instead_of_copying_b1(tmp_path: Path) -> None:
    """Catch unsupported depth variants being mislabeled with copied B1 results."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "B2"

    metrics = run_experiment(config_path, output_dir, variant="B2")

    assert {
        key: metrics[key]
        for key in ("status", "variant", "reason", "source_sha256")
    } == {
        "status": "not_run",
        "variant": "B2",
        "reason": "metric_depth_not_available",
        "source_sha256": sha256_file(tmp_path / "moving.mp4"),
    }
    assert metrics["runtime_s"] >= 0.0
    assert metrics["action_export_eligible"] is False
    assert metrics["collision_validation"] == "not_run"
    assert metrics["physics_validation"] == "not_run"
    assert metrics["action_export_reason"] == "metric_depth_not_available"
    assert not (output_dir / "actions.npz").exists()
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["format_version"] == 4
    assert manifest["source_sha256"] == "not_used"
    assert manifest["status"] == "not_run"
    assert manifest["reason"] == "metric_depth_not_available"
    assert verify_run_directory(output_dir).metrics == metrics


def test_failed_physics_validation_does_not_export_actions(tmp_path: Path) -> None:
    """Catch a failed physics replay being exported as robot action data."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "B0"

    metrics = run_experiment(config_path, output_dir, variant="B0", no_render=True)

    assert metrics["status"] == "rejected"
    assert metrics["reason"] == "physics_validation_failed"
    assert metrics["execution_tracking_ratio"] < 0.95
    assert metrics["bilateral_close_contact_duration_s"] >= 0.0
    assert metrics["bilateral_lift_contact_duration_s"] >= 0.0
    assert metrics["settle_duration_s"] >= 0.0
    assert metrics["action_export_eligible"] is False
    assert metrics["collision_validation"] == "failed"
    assert metrics["physics_validation"] == "failed"
    assert metrics["action_export_reason"] == "physics_validation_failed"
    assert (output_dir / "robot_reference.npz").is_file()
    assert (output_dir / "baseline_control_trace.npz").is_file()
    assert (output_dir / "robot_reference.schema.json").is_file()
    assert (output_dir / "baseline_control_trace.schema.json").is_file()
    assert (output_dir / "simulation.schema.json").is_file()
    assert (output_dir / "rejection.json").is_file()
    assert not (output_dir / "actions.npz").exists()
    with np.load(output_dir / "baseline_control_trace.npz") as control_trace:
        assert set(control_trace.files) == {
            "timestamps_s",
            "control",
            "phase",
        }
    with np.load(output_dir / "simulation.npz") as simulation:
        assert set(simulation.files) == {
            "timestamps_s",
            "control",
            "qpos",
            "qvel",
            "can_pose",
            "tcp_position",
            "tcp_quaternion_wxyz",
            "phase",
            "contact_count",
            "forbidden_contact",
            "maximum_penetration_m",
            "bilateral_contact",
            "box_support_contact",
            "tcp_position_within_tolerance",
            "tcp_orientation_within_tolerance",
            "joint_position_violation",
            "joint_velocity_violation",
            "joint_acceleration_violation",
            "valid_numerical_state",
        }
    sidecar = json.loads(
        (output_dir / "simulation.schema.json").read_text(encoding="utf-8")
    )
    assert sidecar["quaternion_order"] == "wxyz"
    assert sidecar["provenance"]["terminal_status"] == "rejected"
    assert sidecar["provenance"]["terminal_reason"] == "physics_validation_failed"
    assert sidecar["provenance"]["action_export_eligible"] is False
    assert verify_run_directory(output_dir).manifest["source_sha256"] == "not_used"


def test_action_gate_keeps_passing_manual_baseline_out_of_action_data() -> None:
    passed = PhysicsValidationResult(True, (), 0, 0.0)
    failed = PhysicsValidationResult(False, ("maximum_lift_m",), 0, 0.0)

    assert _action_gate("B0", passed) == {
        "collision_validation": "passed",
        "physics_validation": "passed",
        "action_export_eligible": False,
        "action_export_reason": "manual_baseline_not_video_grounded",
        "action_exported": False,
    }
    assert _action_gate("B0", failed)["action_export_reason"] == (
        "physics_validation_failed"
    )
    assert _action_gate("B1", None)["action_export_reason"] == (
        "kinematic_replay_not_action"
    )
    assert _action_gate("B4", None)["action_export_reason"] == (
        "metric_depth_not_available"
    )


@pytest.mark.parametrize("no_render", [True, False])
def test_b0_never_touches_source_in_render_or_no_render_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_render: bool
) -> None:
    config_path = _synthetic_config(tmp_path)
    source = (tmp_path / "moving.mp4").resolve()
    source.unlink()
    real_hash = experiment_module.sha256_file
    real_ffprobe = experiment_module._ffprobe_video_facts

    def source_sentinel(path: str | Path) -> str:
        if Path(path).resolve() == source:
            raise AssertionError("B0 touched source")
        return real_hash(path)

    def ffprobe_sentinel(path: str | Path) -> dict[str, object]:
        if Path(path).resolve() == source:
            raise AssertionError("B0 probed source")
        return real_ffprobe(path)

    monkeypatch.setattr(experiment_module, "sha256_file", source_sentinel)
    monkeypatch.setattr(experiment_module, "_ffprobe_video_facts", ffprobe_sentinel)
    for name in (
        "probe_video",
        "track_roi_lk",
        "render_tracking_overlay",
        "render_comparison_video",
        "_render_contact_sheet",
    ):
        def fail_source_helper(
            *args: object, _name: str = name, **kwargs: object
        ) -> object:
            raise AssertionError(f"B0 entered source-only helper {_name}")

        monkeypatch.setattr(experiment_module, name, fail_source_helper)

    output_dir = tmp_path / f"B0-{no_render}"
    metrics = run_experiment(
        config_path, output_dir, variant="B0", no_render=no_render
    )

    assert metrics["status"] == "rejected"
    assert metrics["reason"] in {
        "physics_validation_failed",
        "manual_baseline_not_video_grounded",
    }
    assert (output_dir / "baseline_control_trace.npz").is_file()
    assert not (output_dir / "trajectory_2d.npz").exists()
    assert not (output_dir / "actions.npz").exists()


@pytest.mark.parametrize("variant", ["B2", "B3", "B4"])
@pytest.mark.parametrize("no_render", [True, False])
def test_metric_depth_variants_never_touch_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    no_render: bool,
) -> None:
    config_path = _synthetic_config(tmp_path)
    source = (tmp_path / "moving.mp4").resolve()
    source.unlink()
    real_hash = experiment_module.sha256_file

    def source_sentinel(path: str | Path) -> str:
        if Path(path).resolve() == source:
            raise AssertionError(f"{variant} touched source")
        return real_hash(path)

    def fail_source_helper(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"{variant} entered a source-only helper")

    monkeypatch.setattr(experiment_module, "sha256_file", source_sentinel)
    for name in ("probe_video", "_ffprobe_video_facts", "track_roi_lk"):
        monkeypatch.setattr(experiment_module, name, fail_source_helper)

    metrics = run_experiment(
        config_path,
        tmp_path / f"{variant}-{no_render}",
        variant=variant,
        no_render=no_render,
    )

    assert metrics["status"] == "not_run"
    assert metrics["reason"] == "metric_depth_not_available"


def test_b0_reference_uses_start_and_goal_from_yaml(tmp_path: Path) -> None:
    """Catch parsed B0 scene coordinates being ignored by retargeting."""

    start = (0.08, 0.42, 0.05)
    goal = (-0.02, 0.52, 0.11)
    config_path = _synthetic_config(
        tmp_path, b0_start_m=start, b0_goal_m=goal
    )
    output_dir = tmp_path / "configured-B0"

    run_experiment(config_path, output_dir, variant="B0", no_render=True)

    with np.load(output_dir / "robot_reference.npz") as reference:
        phases = reference["phase"].tolist()
        np.testing.assert_allclose(reference["ee_positions"][phases.index("close")], start)
        np.testing.assert_allclose(reference["ee_positions"][phases.index("open")], goal)


def test_rejected_tracking_writes_diagnostics_but_not_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch a rejected perception run leaking unvalidated robot actions."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "rejected"

    real_tracking = experiment_module.track_roi_lk

    def tracking_without_evidence(*args: object, **kwargs: object) -> object:
        trajectory = real_tracking(*args, **kwargs)
        return replace(trajectory, confidence=np.zeros_like(trajectory.confidence))

    monkeypatch.setattr(experiment_module, "track_roi_lk", tracking_without_evidence)

    metrics = run_experiment(config_path, output_dir, variant="B1")

    assert metrics["status"] == "rejected"
    assert metrics["rejection_stage"] == "tracking"
    assert metrics["reason"] == "kinematic_replay_not_action"
    assert metrics["perception_rejection_reason"] == (
        "lk_point_availability_ratio_below_minimum"
    )
    assert (output_dir / "rejection.json").is_file()
    assert (output_dir / "trajectory_2d.npz").is_file()
    assert (output_dir / "trajectory_2d.schema.json").is_file()
    assert (output_dir / "tracking_overlay.mp4").is_file()
    assert not (output_dir / "actions.npz").exists()


def test_runner_applies_configured_forward_backward_threshold(tmp_path: Path) -> None:
    """Catch the locked tracking threshold being parsed but silently ignored."""

    config_path = _synthetic_config(tmp_path, forward_backward_threshold_px=0.0)

    metrics = run_experiment(config_path, tmp_path / "strict", variant="B1")

    assert metrics["status"] == "rejected"
    assert metrics["rejection_stage"] == "tracking"
    assert metrics["reason"] == "kinematic_replay_not_action"


def test_command_line_runner_accepts_variant_and_no_render(tmp_path: Path) -> None:
    """Catch the documented CLI flags being absent or writing to the wrong run."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "cli-B2"
    repository = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_exp001.py"),
            "--config",
            str(config_path),
            "--variant",
            "B2",
            "--output-dir",
            str(output_dir),
            "--no-render",
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "not_run"
    assert metrics["reason"] == "metric_depth_not_available"


def test_b2_replaces_reused_b1_output_instead_of_leaving_stale_files(
    tmp_path: Path,
) -> None:
    """Catch a B2 not-run publication retaining B1 media or stale actions."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "reused"
    run_experiment(config_path, output_dir, variant="B1")

    metrics = run_experiment(config_path, output_dir, variant="B2")

    assert metrics["status"] == "not_run"
    assert {path.name for path in output_dir.iterdir()} == {
        "metrics.json",
        "provenance.json",
        "run_manifest.json",
    }


def test_no_render_replaces_prior_rendered_output_without_stale_media(
    tmp_path: Path,
) -> None:
    """Catch --no-render leaving media from a prior rendered run visible."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "no-render-reuse"
    run_experiment(config_path, output_dir, variant="B1")

    metrics = run_experiment(config_path, output_dir, variant="B1", no_render=True)

    assert metrics["status"] == "rejected"
    assert not list(output_dir.glob("*.mp4"))
    assert not list(output_dir.glob("*.png"))
    assert not list(output_dir.parent.glob(f".{output_dir.name}.staging-*"))


def test_parse_failure_publishes_failed_metrics_and_rejection(tmp_path: Path) -> None:
    """Catch malformed YAML escaping without an auditable failed run."""

    config_path = tmp_path / "broken.yaml"
    config_path.write_text("source: [", encoding="utf-8")
    output_dir = tmp_path / "parse-failure"

    metrics = run_experiment(config_path, output_dir, variant="B1")

    assert metrics["status"] == "failed"
    assert metrics["failure_stage"] == "config"
    assert json.loads((output_dir / "rejection.json").read_text(encoding="utf-8"))[
        "stage"
    ] == "config"
    assert (output_dir / "metrics.json").is_file()
    assert not (output_dir / "actions.npz").exists()


def test_config_failure_artifacts_redact_workspace_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch config failure artifacts persisting a local config path."""
    config_path = tmp_path / "config-with-path.yaml"
    config_path.write_text("schema_version: 2", encoding="utf-8")

    def fail_config(path: Path) -> object:
        raise ValueError(f"cannot load {Path(path).resolve()}")

    monkeypatch.setattr(experiment_module, "load_experiment_config", fail_config)
    output_dir = tmp_path / "config-path-failure"

    metrics = run_experiment(config_path, output_dir, variant="B1")

    persisted = [
        json.loads((output_dir / name).read_text(encoding="utf-8"))
        for name in ("metrics.json", "rejection.json")
    ]
    assert metrics["status"] == "failed"
    assert all(str(tmp_path.resolve()) not in item["error_message"] for item in persisted)


def test_source_failure_artifacts_redact_workspace_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch source-probe failure artifacts persisting a resolved source path."""
    config_path = _synthetic_config(tmp_path)
    source = tmp_path / "moving.mp4"
    real_sha256_file = experiment_module.sha256_file

    def fail_source_hash(path: Path) -> str:
        if Path(path).resolve() == source.resolve():
            raise ValueError(f"cannot read {source.resolve()}")
        return real_sha256_file(path)

    monkeypatch.setattr(experiment_module, "sha256_file", fail_source_hash)
    output_dir = tmp_path / "source-path-failure"

    metrics = run_experiment(config_path, output_dir, variant="B1")

    persisted = [
        json.loads((output_dir / name).read_text(encoding="utf-8"))
        for name in ("metrics.json", "rejection.json")
    ]
    assert metrics["status"] == "failed"
    assert metrics["failure_stage"] == "source_probe"
    assert all(str(tmp_path.resolve()) not in item["error_message"] for item in persisted)


def test_hash_failure_replaces_old_output_with_failed_run(tmp_path: Path) -> None:
    """Catch source-integrity failure leaving a previous trusted rejected run."""

    config_path, output_dir = _trusted_no_action_output(tmp_path)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["source"]["sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    metrics = run_experiment(config_path, output_dir, variant="B1")

    assert metrics["status"] == "failed"
    assert metrics["failure_stage"] == "source_probe"
    assert not (output_dir / "actions.npz").exists()
    assert (output_dir / "rejection.json").is_file()


def test_tracking_exception_publishes_stage_failure_without_partial_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch an unexpected stage exception bypassing failed-run publication."""

    config_path = _synthetic_config(tmp_path)

    def fail_tracking(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic tracking crash")

    monkeypatch.setattr(experiment_module, "track_roi_lk", fail_tracking)
    output_dir = tmp_path / "tracking-crash"

    metrics = run_experiment(config_path, output_dir, variant="B1")

    assert metrics["status"] == "failed"
    assert metrics["failure_stage"] == "tracking"
    assert metrics["error_type"] == "RuntimeError"
    assert not (output_dir / "actions.npz").exists()
    assert not list(output_dir.glob("*.npz"))
    assert not list(output_dir.glob("*.schema.json"))
    assert (output_dir / "rejection.json").is_file()
    assert experiment_module._trusted_run_snapshot(output_dir)


def test_output_replacement_refuses_unrecognized_directory(tmp_path: Path) -> None:
    """Catch transaction cleanup recursively deleting an arbitrary user directory."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "not-a-generated-run"
    output_dir.mkdir()
    protected = output_dir / "keep.txt"
    protected.write_text("user data", encoding="utf-8")

    with pytest.raises(ValueError, match="unrecognized files"):
        run_experiment(config_path, output_dir, variant="B2")

    assert protected.read_text(encoding="utf-8") == "user data"


def test_output_replacement_refuses_directory_disguised_as_generated_file(
    tmp_path: Path,
) -> None:
    """Catch recursive cleanup following an allowed filename into user data."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "disguised-directory"
    protected_dir = output_dir / "metrics.json"
    protected_dir.mkdir(parents=True)
    protected = protected_dir / "keep.txt"
    protected.write_text("user data", encoding="utf-8")

    with pytest.raises(ValueError, match="unrecognized files"):
        run_experiment(config_path, output_dir, variant="B2")

    assert protected.read_text(encoding="utf-8") == "user data"


def test_relative_source_is_resolved_from_config_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch a reproducible config depending on the caller's working directory."""

    project = tmp_path / "project"
    config_dir = project / "configs"
    config_dir.mkdir(parents=True)
    source = project / "moving.mp4"
    _write_moving_object_video(source)
    config_path = config_dir / "experiment.yaml"
    _write_config(config_path, source)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["source"]["path"] = "../moving.mp4"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    metrics = run_experiment(config_path, tmp_path / "relative-run", variant="B2")

    assert metrics["status"] == "not_run"
    provenance = json.loads(
        (tmp_path / "relative-run" / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["source"]["id"] == "synthetic-moving-object"
    assert str(tmp_path.resolve()) not in json.dumps(provenance, sort_keys=True)


def test_mp4_validation_rejects_non_video_binary(tmp_path: Path) -> None:
    """Catch a nonempty but undecodable file passing the media publication gate."""

    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"not an mp4")

    with pytest.raises(ValueError, match="ffprobe|decodable"):
        _validate_mp4(invalid)


def test_invalid_rendered_mp4_publishes_failure_without_partial_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch a broken encoder artifact being published as a completed run."""

    config_path = _synthetic_config(tmp_path)

    def write_invalid(path: Path, frames: np.ndarray, fps: float) -> None:
        path.write_bytes(b"broken replay")

    monkeypatch.setattr(experiment_module, "_write_rgb_video", write_invalid)
    output_dir = tmp_path / "invalid-media"

    metrics = run_experiment(config_path, output_dir, variant="B1")

    assert metrics["status"] == "failed"
    assert metrics["failure_stage"] == "visualization"
    assert (output_dir / "rejection.json").is_file()
    assert not list(output_dir.glob("*.mp4"))
    assert not list(output_dir.glob("*.png"))
    assert not (output_dir / "actions.npz").exists()


def test_zero_confidence_contact_phases_are_recorded_as_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch unreliable release/settle inference being reported without a warning."""

    config_path = _synthetic_config(tmp_path)
    real_infer = experiment_module.infer_motion_phases

    def infer_with_zero_confidence(trajectory: object, fps: float) -> tuple[object, ...]:
        phases = real_infer(trajectory, fps)
        return tuple(
            replace(phase, confidence=0.0)
            if phase.label in ("release", "settle")
            else phase
            for phase in phases
        )

    monkeypatch.setattr(
        experiment_module, "infer_motion_phases", infer_with_zero_confidence
    )

    metrics = run_experiment(
        config_path, tmp_path / "degraded-phases", variant="B1", no_render=True
    )

    assert metrics["perception_status"] == "degraded"
    assert metrics["perception_warnings"] == [
        "zero_confidence_phase:release",
        "zero_confidence_phase:settle",
    ]


def test_personal_metrics_file_without_trusted_marker_is_never_replaced(
    tmp_path: Path,
) -> None:
    """Catch whitelist-only validation deleting a user's personal metrics file."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "personal-metrics"
    output_dir.mkdir()
    personal = output_dir / "metrics.json"
    personal.write_text('{"owner":"user"}', encoding="utf-8")

    with pytest.raises(ValueError, match="trusted generated-run marker"):
        run_experiment(config_path, output_dir, variant="B2")

    assert personal.read_text(encoding="utf-8") == '{"owner":"user"}'


def test_legacy_manifest_without_producer_or_digests_is_not_trusted(
    tmp_path: Path,
) -> None:
    """Catch an unverifiable v1 marker authorizing deletion of personal files."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "personal-legacy-shaped-output"
    output_dir.mkdir()
    (output_dir / "metrics.json").write_text(
        json.dumps({"status": "not_run", "variant": "B2"}), encoding="utf-8"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps({"owner": "user"}), encoding="utf-8"
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "variant": "B2",
                "status": "not_run",
                "files": ["metrics.json", "provenance.json"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="trusted generated-run marker"):
        run_experiment(config_path, output_dir, variant="B2")

    assert json.loads((output_dir / "provenance.json").read_text(encoding="utf-8")) == {
        "owner": "user"
    }


def test_v3_manifest_remains_read_only_verifiable(tmp_path: Path) -> None:
    """Catch a v4 upgrade rewriting or refusing an already trusted v3 run."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "v3-read-only"
    run_experiment(config_path, output_dir, variant="B2", no_render=True)
    current = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    v3 = {
        "producer": "webvideo_to_data.experiment",
        "format_version": 3,
        "variant": current["variant"],
        "status": current["status"],
        "files": {
            name: {"size": entry["size"], "sha256": entry["sha256"]}
            for name, entry in current["files"].items()
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(v3, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    before = manifest_path.read_bytes()

    verified = verify_run_directory(output_dir)

    assert verified.manifest == v3
    assert experiment_module._trusted_run_snapshot(output_dir)
    assert manifest_path.read_bytes() == before


def test_verify_run_directory_rejects_file_added_during_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "verify-race"
    run_experiment(config_path, output_dir, variant="B2", no_render=True)
    real_verify_npz = artifacts_module._verify_v4_npz_artifacts

    def verify_then_add(*args: object, **kwargs: object) -> None:
        real_verify_npz(*args, **kwargs)
        (output_dir / "externally-added.txt").write_text("preserve", encoding="utf-8")

    monkeypatch.setattr(artifacts_module, "_verify_v4_npz_artifacts", verify_then_add)

    with pytest.raises(ValueError, match="trusted run directory"):
        verify_run_directory(output_dir)
    assert (output_dir / "externally-added.txt").read_text(encoding="utf-8") == "preserve"


def _run_file_bytes(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def _restore_run_file_bytes(
    directory: Path,
    contents: dict[str, bytes],
    timestamps: dict[str, tuple[int, int]],
) -> None:
    for name, content in contents.items():
        path = directory / name
        path.write_bytes(content)
        os.utime(path, ns=timestamps[name])


def test_run_verifier_rejects_initial_malicious_sidecar_aba(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "sidecar-aba"
    run_experiment(config_path, output_dir, variant="B0", no_render=True)
    legal_bytes = _run_file_bytes(output_dir)
    sidecar_path = output_dir / "simulation.schema.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    trusted_producer = sidecar["provenance"]["producer"]
    sidecar["provenance"]["producer"] = "x" * len(trusted_producer)
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    (output_dir / "run_manifest.json").unlink()
    experiment_module._write_run_manifest(output_dir, metrics)
    malicious_bytes = _run_file_bytes(output_dir)
    timestamps = {
        path.name: (path.stat().st_atime_ns, path.stat().st_mtime_ns)
        for path in output_dir.iterdir()
    }
    real_capture = artifacts_module._capture_directory_bytes
    capture_count = 0

    def malicious_legal_malicious(path: Path) -> object:
        nonlocal capture_count
        capture_count += 1
        if capture_count == 1:
            snapshot = real_capture(path)
            _restore_run_file_bytes(output_dir, legal_bytes, timestamps)
            return snapshot
        _restore_run_file_bytes(output_dir, malicious_bytes, timestamps)
        return real_capture(path)

    monkeypatch.setattr(
        artifacts_module, "_capture_directory_bytes", malicious_legal_malicious
    )

    with pytest.raises(ValueError, match="trusted run directory"):
        verify_run_directory(output_dir)
    assert json.loads(sidecar_path.read_text(encoding="utf-8"))["provenance"][
        "producer"
    ] != trusted_producer


def test_run_verifier_rejects_legal_to_final_malicious_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "sidecar-final-malicious"
    run_experiment(config_path, output_dir, variant="B0", no_render=True)
    legal_bytes = _run_file_bytes(output_dir)
    timestamps = {
        path.name: (path.stat().st_atime_ns, path.stat().st_mtime_ns)
        for path in output_dir.iterdir()
    }
    sidecar_path = output_dir / "simulation.schema.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["provenance"]["producer"] = "x" * len(
        sidecar["provenance"]["producer"]
    )
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    (output_dir / "run_manifest.json").unlink()
    experiment_module._write_run_manifest(output_dir, metrics)
    malicious_bytes = _run_file_bytes(output_dir)
    _restore_run_file_bytes(output_dir, legal_bytes, timestamps)
    real_capture = artifacts_module._capture_directory_bytes
    capture_count = 0

    def legal_then_malicious(path: Path) -> object:
        nonlocal capture_count
        capture_count += 1
        snapshot = real_capture(path)
        if capture_count == 1:
            _restore_run_file_bytes(output_dir, malicious_bytes, timestamps)
        return snapshot

    monkeypatch.setattr(artifacts_module, "_capture_directory_bytes", legal_then_malicious)

    with pytest.raises(ValueError, match="trusted run directory"):
        verify_run_directory(output_dir)
    assert json.loads(sidecar_path.read_text(encoding="utf-8"))["provenance"][
        "producer"
    ] != "webvideo_to_data.experiment"


@pytest.mark.skipif(os.name != "nt", reason="Windows zero-inode fallback")
def test_run_verifier_rejects_same_byte_replacements_when_fstat_inode_is_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "same-byte-run-replacement"
    run_experiment(config_path, output_dir, variant="B2", no_render=True)
    replaced_names = {"metrics.json", "run_manifest.json"}
    file_count = len(list(output_dir.iterdir()))
    real_fstat = artifacts_module.os.fstat
    real_windows_identity = artifacts_module._windows_file_identity
    fallback_calls = 0

    def zero_inode(file_descriptor: int) -> object:
        value = real_fstat(file_descriptor)
        return type(
            "ZeroInodeStat",
            (),
            {
                "st_mode": value.st_mode,
                "st_size": value.st_size,
                "st_mtime_ns": value.st_mtime_ns,
                "st_dev": value.st_dev,
                "st_ino": 0,
            },
        )()

    def fallback_identity(file_descriptor: int) -> tuple[int, int]:
        nonlocal fallback_calls
        fallback_calls += 1
        return real_windows_identity(file_descriptor)

    real_capture = artifacts_module._capture_directory_bytes
    captures = 0

    def capture_then_replace(path: Path) -> object:
        nonlocal captures
        snapshot = real_capture(path)
        captures += 1
        if captures == 1:
            for name in replaced_names:
                target = output_dir / name
                timestamps = (target.stat().st_atime_ns, target.stat().st_mtime_ns)
                replacement = target.with_name(f".{target.name}.same-bytes")
                replacement.write_bytes(target.read_bytes())
                os.utime(replacement, ns=timestamps)
                replacement.replace(target)
                os.utime(target, ns=timestamps)
        return snapshot

    monkeypatch.setattr(artifacts_module.os, "fstat", zero_inode)
    monkeypatch.setattr(
        artifacts_module, "_windows_file_identity", fallback_identity
    )
    monkeypatch.setattr(artifacts_module, "_capture_directory_bytes", capture_then_replace)

    with pytest.raises(ValueError, match="trusted run directory"):
        verify_run_directory(output_dir)
    assert fallback_calls == file_count * 4


def test_trusted_snapshot_adopts_verified_snapshot_without_resampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "verified-snapshot"
    run_experiment(config_path, output_dir, variant="B2", no_render=True)

    def forbidden_resample(path: Path) -> object:
        raise AssertionError(f"unexpected resample of {path}")

    monkeypatch.setattr(experiment_module, "_directory_snapshot", forbidden_resample)

    assert experiment_module._trusted_run_snapshot(output_dir)


@pytest.mark.parametrize(
    "metrics_update",
    [
        {"status": "mystery", "reason": "unknown_status"},
        {"status": "not_run", "reason": None},
        {"status": "rejected", "reason": None},
        {"status": "rejected", "reason": "metric_depth_not_available"},
    ],
)
def test_trusted_manifest_rejects_unknown_or_reasonless_terminal_status(
    tmp_path: Path, metrics_update: dict[str, object]
) -> None:
    """Catch malformed terminal metrics authorizing generated-output replacement."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "invalid-terminal-status"
    run_experiment(config_path, output_dir, variant="B2")
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(metrics_update)
    if metrics.get("reason") is None:
        metrics.pop("reason", None)
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    if metrics["status"] == "rejected":
        (output_dir / "rejection.json").write_text(
            json.dumps({"stage": "test", "reason": "test"}), encoding="utf-8"
        )
    (output_dir / "run_manifest.json").unlink()
    experiment_module._write_run_manifest(output_dir, metrics)

    with pytest.raises(ValueError, match="trusted run directory"):
        verify_run_directory(output_dir)
    with pytest.raises(ValueError, match="trusted generated-run marker"):
        run_experiment(config_path, output_dir, variant="B2")


@pytest.mark.parametrize("failed_validation", ["collision_validation", "physics_validation"])
def test_trusted_manifest_rejects_completed_without_physical_validation(
    tmp_path: Path, failed_validation: str,
) -> None:
    """Catch a partially validated completed marker legitimizing unsafe actions."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "completed-without-collision-validation"
    run_experiment(config_path, output_dir, variant="B0", no_render=True)
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        status="completed",
        placed_successfully=True,
        reachability_ratio=1.0,
        collision_validation="passed",
        physics_validation="passed",
        action_export_eligible=True,
        action_exported=True,
    )
    metrics[failed_validation] = "failed"
    metrics.pop("reason", None)
    metrics.pop("rejection_stage", None)
    metrics.pop("action_export_reason", None)
    (output_dir / "rejection.json").unlink()
    shutil.copyfile(output_dir / "robot_reference.npz", output_dir / "actions.npz")
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    (output_dir / "run_manifest.json").unlink()
    experiment_module._write_run_manifest(output_dir, metrics)

    with pytest.raises(ValueError, match="trusted generated-run marker"):
        run_experiment(config_path, output_dir, variant="B2")


def test_refreshed_manifest_cannot_trust_forged_b0_completed_actions(
    tmp_path: Path,
) -> None:
    """Catch valid hashes legitimizing a terminal state B0 can never produce."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "forged-b0-completed"
    run_experiment(config_path, output_dir, variant="B0", no_render=True)
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        status="completed",
        placed_successfully=True,
        reachability_ratio=1.0,
        collision_validation="passed",
        physics_validation="passed",
        action_export_eligible=True,
        action_exported=True,
    )
    metrics.pop("reason", None)
    metrics.pop("rejection_stage", None)
    metrics.pop("action_export_reason", None)
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    (output_dir / "rejection.json").unlink()
    shutil.copyfile(output_dir / "robot_reference.npz", output_dir / "actions.npz")
    (output_dir / "run_manifest.json").unlink()
    experiment_module._write_run_manifest(output_dir, metrics)

    with pytest.raises(ValueError, match="trusted run directory"):
        verify_run_directory(output_dir)

    assert (output_dir / "actions.npz").is_file()


def test_v4_verifier_binds_provenance_experiment_id_to_resolved_config(
    tmp_path: Path,
) -> None:
    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "mismatched-experiment-provenance"
    run_experiment(config_path, output_dir, variant="B2", no_render=True)
    provenance_path = output_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["experiment_id"] = "OTHER-EXPERIMENT"
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["provenance.json"].update(
        size=provenance_path.stat().st_size,
        sha256=sha256_file(provenance_path),
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="trusted run directory verification failed"):
        verify_run_directory(output_dir)


def test_variant_model_identity_is_unique_and_dependency_complete(tmp_path: Path) -> None:
    assert artifacts_module.pinned_model_identity("B0") != artifacts_module.pinned_model_identity("B1")
    with pytest.raises(ValueError, match="pinned"):
        artifacts_module.pinned_model_dimensions(
            "B0", artifacts_module.pinned_model_identity("B1")
        )

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "asset.bin").write_bytes(b"asset-v1")
    (model_dir / "included.xml").write_text(
        '<mujoco><asset><mesh name="m" file="asset.bin"/></asset></mujoco>',
        encoding="utf-8",
    )
    root = model_dir / "root.xml"
    root.write_text('<mujoco><include file="included.xml"/></mujoco>', encoding="utf-8")
    before = artifacts_module._pinned_dependency_identity(root, model_dir)
    (model_dir / "asset.bin").write_bytes(b"asset-v2")
    after = artifacts_module._pinned_dependency_identity(root, model_dir)

    assert after != before


def test_terminal_metrics_enforce_variant_specific_outcomes() -> None:
    b0_pass = {
        "status": "rejected",
        "variant": "B0",
        "reason": "manual_baseline_not_video_grounded",
        "rejection_stage": "simulation",
        "placed_successfully": True,
        "collision_validation": "passed",
        "physics_validation": "passed",
        "action_export_eligible": False,
        "action_export_reason": "manual_baseline_not_video_grounded",
        "action_exported": False,
    }
    b0_fail = {
        **b0_pass,
        "reason": "physics_validation_failed",
        "placed_successfully": False,
        "collision_validation": "failed",
        "physics_validation": "failed",
        "action_export_reason": "physics_validation_failed",
    }
    b1 = {
        "status": "rejected",
        "variant": "B1",
        "reason": "kinematic_replay_not_action",
        "rejection_stage": "simulation",
        "placed_successfully": False,
        "collision_validation": "not_applicable_kinematic",
        "physics_validation": "not_applicable_kinematic",
        "action_export_eligible": False,
        "action_export_reason": "kinematic_replay_not_action",
        "action_exported": False,
    }
    b2 = {
        "status": "not_run",
        "variant": "B2",
        "reason": "metric_depth_not_available",
        "collision_validation": "not_run",
        "physics_validation": "not_run",
        "action_export_eligible": False,
        "action_export_reason": "metric_depth_not_available",
        "action_exported": False,
    }
    assert experiment_module._terminal_metrics_error(
        b0_pass, {"metrics.json", "rejection.json"}
    ) is None
    assert experiment_module._terminal_metrics_error(
        b0_fail, {"metrics.json", "rejection.json"}
    ) is None
    assert experiment_module._terminal_metrics_error(
        b1, {"metrics.json", "rejection.json"}
    ) is None
    assert experiment_module._terminal_metrics_error(
        b2, {"metrics.json"}
    ) is None

    forged = (
        ({**b0_pass, "status": "completed"}, {"metrics.json", "actions.npz"}),
        ({**b0_pass, "placed_successfully": False}, {"metrics.json", "rejection.json"}),
        ({**b0_fail, "physics_validation": "passed"}, {"metrics.json", "rejection.json"}),
        ({**b1, "reason": "tracking_failed"}, {"metrics.json", "rejection.json"}),
        ({**b1, "physics_validation": "passed"}, {"metrics.json", "rejection.json"}),
        ({**b2, "status": "rejected"}, {"metrics.json", "rejection.json"}),
        ({**b2, "collision_validation": "passed"}, {"metrics.json"}),
    )
    for metrics, files in forged:
        assert experiment_module._terminal_metrics_error(metrics, files) is not None


@pytest.mark.parametrize(
    ("variant", "metrics_update"),
    [
        ("B0", {}),
        ("B1", {}),
        ("B2", {"collision_validation": "passed"}),
        ("B3", {"physics_validation": "passed"}),
        ("B4", {"status": "rejected", "rejection_stage": "test"}),
        (
            "B2",
            {
                "status": "failed",
                "reason": "stage_exception",
                "failure_stage": "test",
                "error_type": True,
                "error_message": "synthetic",
            },
        ),
    ],
)
def test_public_verifier_enforces_exact_variant_terminal_schema(
    tmp_path: Path, variant: str, metrics_update: dict[str, object]
) -> None:
    config_path = _synthetic_config(tmp_path)
    original = tmp_path / "terminal-base"
    run_experiment(config_path, original, variant="B2", no_render=True)
    manifest_seed = dict(verify_run_directory(original).manifest)
    candidate = tmp_path / "terminal-candidate"
    shutil.copytree(original, candidate)
    (candidate / "provenance.json").unlink()
    metrics_path = candidate / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["variant"] = variant
    metrics.update(metrics_update)
    if metrics["status"] in {"rejected", "failed"}:
        (candidate / "rejection.json").write_text(
            json.dumps({"stage": "test", "reason": metrics.get("reason")}),
            encoding="utf-8",
        )
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    (candidate / "run_manifest.json").unlink()
    experiment_module._write_run_manifest(
        candidate, metrics, manifest_seed=manifest_seed
    )

    with pytest.raises(ValueError, match="trusted run directory"):
        verify_run_directory(candidate)


def _failed_terminal_metrics(variant: str) -> dict[str, object]:
    return {
        "status": "failed",
        "variant": variant,
        "reason": "stage_exception",
        "failure_stage": "simulation",
        "error_type": "RuntimeError",
        "error_message": "synthetic failure",
        "placed_successfully": False,
        **experiment_module._action_gate(variant, None),
    }


@pytest.mark.parametrize(
    ("variant", "mutation"),
    [
        ("B0", {"placed_successfully": True}),
        ("B0", {"physics_validation": "passed"}),
        ("B0", {"collision_validation": "passed"}),
        ("B0", {"action_export_reason": "arbitrary_reason"}),
        ("B0", {"reason": "unknown_failure"}),
        ("B0", {"failure_stage": ""}),
        ("B0", {"error_type": 7}),
        ("B0", {"error_message": None}),
        ("B1", {"physics_validation": "failed"}),
        ("B1", {"action_export_reason": "physics_validation_failed"}),
        ("B2", {"collision_validation": "failed"}),
        ("B2", {"action_export_reason": "arbitrary_reason"}),
    ],
)
def test_failed_terminal_rejects_forged_variant_evidence(
    variant: str, mutation: dict[str, object]
) -> None:
    metrics = _failed_terminal_metrics(variant)
    metrics.update(mutation)

    assert experiment_module._terminal_metrics_error(
        metrics, {"metrics.json", "rejection.json"}
    ) is not None


@pytest.mark.parametrize("variant", ["B0", "B1", "B2", "B3", "B4"])
def test_real_publication_failure_metrics_match_failed_terminal_schema(
    variant: str,
) -> None:
    metrics = experiment_module._publication_failure_metrics(
        variant=variant,
        stage="publication_swap",
        error=OSError("synthetic publication failure"),
        started=time.perf_counter(),
    )

    assert experiment_module._terminal_metrics_error(
        metrics, {"metrics.json", "rejection.json"}
    ) is None


@pytest.mark.parametrize(
    "metrics",
    [
        {"status": "mystery", "variant": "B2", "reason": "unknown"},
        {
            "status": "not_run",
            "variant": "B2",
            "action_export_eligible": False,
        },
    ],
)
def test_publication_contract_rejects_invalid_terminal_metrics(
    tmp_path: Path, metrics: dict[str, object]
) -> None:
    """Catch an internally malformed status reaching manifest publication."""

    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(experiment_module._PipelineFailure, match="terminal metrics"):
        experiment_module._validate_required_run_files(
            tmp_path, metrics, no_render=True
        )


def test_rejected_run_marker_cannot_trust_an_action_artifact(tmp_path: Path) -> None:
    """Catch refreshed hashes legitimizing an untrusted legacy stale action."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "rejected-with-action"
    run_experiment(config_path, output_dir, variant="B1", no_render=True)
    shutil.copyfile(
        output_dir / "robot_reference.npz", output_dir / "actions.npz"
    )
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    (output_dir / "run_manifest.json").unlink()
    experiment_module._write_run_manifest(output_dir, metrics)

    with pytest.raises(ValueError, match="trusted generated-run marker"):
        run_experiment(config_path, output_dir, variant="B2")

    assert (output_dir / "actions.npz").is_file()


def test_publish_revalidates_destination_identity_and_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch destination mutation during a long run being recursively deleted."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "mutated-during-run"
    run_experiment(config_path, output_dir, variant="B2")
    original_execute = experiment_module._execute_run
    user_file = output_dir / "user-added.txt"

    def execute_then_mutate(*args: object, **kwargs: object) -> dict[str, object]:
        metrics = original_execute(*args, **kwargs)
        user_file.write_text("preserve me", encoding="utf-8")
        return metrics

    monkeypatch.setattr(experiment_module, "_execute_run", execute_then_mutate)

    with pytest.raises(ValueError, match="changed during run"):
        run_experiment(config_path, output_dir, variant="B2")

    assert user_file.read_text(encoding="utf-8") == "preserve me"


def test_same_output_runs_are_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch concurrent publishers validating the same stale output snapshot."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "serialized"
    original_execute = experiment_module._execute_run
    state_lock = threading.Lock()
    start_barrier = threading.Barrier(2)
    active = 0
    maximum_active = 0
    errors: list[BaseException] = []

    def measured_execute(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.15)
            return original_execute(*args, **kwargs)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(experiment_module, "_execute_run", measured_execute)

    def worker() -> None:
        try:
            start_barrier.wait()
            run_experiment(config_path, output_dir, variant="B2")
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert not errors
    assert maximum_active == 1
    assert json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))[
        "status"
    ] == "not_run"


def test_swap_failure_restores_as_explicit_failure_without_stale_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch staging swap rollback leaving the old rejected run canonical."""

    config_path, output_dir = _trusted_no_action_output(tmp_path)
    real_replace = Path.replace

    def fail_staging_swap(self: Path, target: Path) -> Path:
        target_path = Path(target)
        if self.name.startswith(f".{output_dir.name}.staging-") and target_path == output_dir:
            raise OSError("synthetic staging swap failure")
        return real_replace(self, target_path)

    monkeypatch.setattr(Path, "replace", fail_staging_swap)

    metrics = run_experiment(config_path, output_dir, variant="B2")

    assert metrics["status"] == "failed"
    assert metrics["failure_stage"] == "publication_swap"
    assert (output_dir / "rejection.json").is_file()
    assert not (output_dir / "actions.npz").exists()
    assert not list(output_dir.parent.glob(f".{output_dir.name}.backup-*"))


def test_old_output_rename_post_success_error_still_publishes_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch an old-output rename that succeeds before surfacing an I/O error."""

    config_path, output_dir = _trusted_no_action_output(tmp_path)
    real_replace = Path.replace

    def move_old_then_error(self: Path, target: Path) -> Path:
        target_path = Path(target)
        if self == output_dir and target_path.name.startswith(
            f".{output_dir.name}.backup-"
        ):
            real_replace(self, target_path)
            raise OSError("synthetic post-rename error")
        return real_replace(self, target_path)

    monkeypatch.setattr(Path, "replace", move_old_then_error)

    metrics = run_experiment(config_path, output_dir, variant="B2")

    assert metrics["status"] == "failed"
    assert metrics["failure_stage"] == "publication_swap"
    assert (output_dir / "rejection.json").is_file()
    assert not (output_dir / "actions.npz").exists()
    assert len(list(output_dir.parent.glob(f".{output_dir.name}.backup-*"))) == 1


def test_fresh_staging_rename_post_success_error_publishes_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch a fresh staging rename succeeding before surfacing an I/O error."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "fresh-post-success"
    real_replace = Path.replace

    def move_staging_then_error(self: Path, target: Path) -> Path:
        target_path = Path(target)
        if self.name.startswith(f".{output_dir.name}.staging-"):
            real_replace(self, target_path)
            raise OSError("synthetic post-staging-rename error")
        return real_replace(self, target_path)

    monkeypatch.setattr(Path, "replace", move_staging_then_error)

    metrics = run_experiment(config_path, output_dir, variant="B2")

    assert metrics["status"] == "failed"
    assert metrics["failure_stage"] == "publication_swap"
    assert (output_dir / "rejection.json").is_file()
    assert not (output_dir / "actions.npz").exists()


def test_existing_staging_rename_post_success_error_publishes_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch a replacement staging rename succeeding before reporting failure."""

    config_path, output_dir = _trusted_no_action_output(tmp_path)
    real_replace = Path.replace

    def move_staging_then_error(self: Path, target: Path) -> Path:
        target_path = Path(target)
        if self.name.startswith(f".{output_dir.name}.staging-"):
            real_replace(self, target_path)
            raise OSError("synthetic post-staging-rename error")
        return real_replace(self, target_path)

    monkeypatch.setattr(Path, "replace", move_staging_then_error)

    metrics = run_experiment(config_path, output_dir, variant="B2")

    assert metrics["status"] == "failed"
    assert metrics["failure_stage"] == "publication_swap"
    assert (output_dir / "rejection.json").is_file()
    assert not (output_dir / "actions.npz").exists()
    assert not list(output_dir.parent.glob(f".{output_dir.name}.backup-*"))


def test_publication_failure_marker_refuses_external_canonical_replacement(
    tmp_path: Path,
) -> None:
    """Catch failure reporting overwriting a canonical directory it no longer owns."""

    _, output_dir = _trusted_no_action_output(tmp_path)
    expected = experiment_module._trusted_run_snapshot(output_dir)
    quarantine = tmp_path / "runner-output-moved-away"
    output_dir.replace(quarantine)
    output_dir.mkdir()
    personal = output_dir / "metrics.json"
    personal.write_text('{"owner":"user"}', encoding="utf-8")

    with pytest.raises(ValueError, match="canonical output changed"):
        experiment_module._mark_canonical_publication_failure(
            output_dir,
            expected=expected,
            variant="B2",
            stage="publication_swap",
            error=OSError("synthetic publication error"),
            started=time.perf_counter(),
        )

    assert personal.read_text(encoding="utf-8") == '{"owner":"user"}'


@pytest.mark.parametrize("failed_name", ["rejection.json", "metrics.json", "run_manifest.json"])
def test_publication_failure_write_fault_restores_old_canonical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_name: str
) -> None:
    _, output_dir = _trusted_no_action_output(tmp_path)
    expected = experiment_module._trusted_run_snapshot(output_dir)
    before = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    real_atomic_json = experiment_module._atomic_json
    failed = False

    def fail_one_write(path: Path, payload: dict[str, object]) -> None:
        nonlocal failed
        if path.name == failed_name and not failed:
            failed = True
            raise OSError(f"synthetic {failed_name} write failure")
        real_atomic_json(path, payload)

    monkeypatch.setattr(experiment_module, "_atomic_json", fail_one_write)

    with pytest.raises((OSError, ValueError), match="write failure"):
        experiment_module._mark_canonical_publication_failure(
            output_dir,
            expected=expected,
            variant="B0",
            stage="publication_swap",
            error=OSError("synthetic publication error"),
            started=time.perf_counter(),
        )

    assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == before


def test_publication_failure_uses_seed_context_without_old_provenance(
    tmp_path: Path,
) -> None:
    _, output_dir = _trusted_no_action_output(tmp_path)
    expected = experiment_module._trusted_run_snapshot(output_dir)
    old_bytes = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    seed = dict(verify_run_directory(output_dir).manifest)
    seed.update(
        variant="B1",
        source_sha256="a" * 64,
        model_sha256=artifacts_module.pinned_model_identity("B1"),
    )

    metrics = experiment_module._mark_canonical_publication_failure(
        output_dir,
        expected=expected,
        variant="B1",
        stage="publication_swap",
        error=OSError("synthetic publication error"),
        started=time.perf_counter(),
        manifest_seed=seed,
    )

    verified = verify_run_directory(output_dir)
    assert metrics["status"] == "failed"
    assert verified.manifest["config_sha256"] == seed["config_sha256"]
    assert verified.manifest["source_sha256"] == seed["source_sha256"]
    assert verified.manifest["model_sha256"] == artifacts_module.pinned_model_identity("B1")
    assert not (output_dir / "provenance.json").exists()
    quarantines = list(
        output_dir.parent.glob(
            f".{output_dir.name}.failure-quarantine-*"
        )
    )
    assert len(quarantines) == 1
    assert {path.name: path.read_bytes() for path in quarantines[0].iterdir()} == old_bytes


def test_failure_staging_unlink_fault_retries_without_partial_hidden_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output_dir = _trusted_no_action_output(tmp_path)
    expected = experiment_module._trusted_run_snapshot(output_dir)
    before = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    real_atomic_json = experiment_module._atomic_json
    real_unlink = Path.unlink
    write_failed = False
    unlink_failed = False

    def fail_metrics(path: Path, payload: dict[str, object]) -> None:
        nonlocal write_failed
        if path.name == "metrics.json" and not write_failed:
            write_failed = True
            raise OSError("synthetic metrics write failure")
        real_atomic_json(path, payload)

    def fail_first_cleanup_unlink(self: Path, *args: object, **kwargs: object) -> None:
        nonlocal unlink_failed
        if ".failure-staging-" in self.parent.name and not unlink_failed:
            unlink_failed = True
            raise OSError("synthetic cleanup unlink failure")
        real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(experiment_module, "_atomic_json", fail_metrics)
    monkeypatch.setattr(Path, "unlink", fail_first_cleanup_unlink)

    with pytest.raises(OSError, match="metrics write failure"):
        experiment_module._mark_canonical_publication_failure(
            output_dir,
            expected=expected,
            variant="B0",
            stage="publication_swap",
            error=OSError("synthetic publication error"),
            started=time.perf_counter(),
        )

    assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == before
    assert not list(output_dir.parent.glob(f".{output_dir.name}.failure-staging-*"))


def test_publication_failure_marker_does_not_mutate_post_snapshot_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch canonical replacement between ownership validation and first mutation."""

    _, output_dir = _trusted_no_action_output(tmp_path)
    expected = experiment_module._trusted_run_snapshot(output_dir)
    runner_output = tmp_path / "runner-output-moved-after-snapshot"
    personal_actions = b"personal action bytes"
    personal_metrics = '{"owner":"user","kind":"metrics"}'
    personal_rejection = '{"owner":"user","kind":"rejection"}'
    real_snapshot = experiment_module._trusted_run_snapshot
    swapped = False

    def snapshot_then_swap(path: Path) -> object:
        nonlocal swapped
        snapshot = real_snapshot(path)
        if path == output_dir and not swapped:
            swapped = True
            output_dir.replace(runner_output)
            output_dir.mkdir()
            (output_dir / "actions.npz").write_bytes(personal_actions)
            (output_dir / "metrics.json").write_text(
                personal_metrics, encoding="utf-8"
            )
            (output_dir / "rejection.json").write_text(
                personal_rejection, encoding="utf-8"
            )
        return snapshot

    monkeypatch.setattr(experiment_module, "_trusted_run_snapshot", snapshot_then_swap)

    with pytest.raises(ValueError, match="canonical output changed"):
        experiment_module._mark_canonical_publication_failure(
            output_dir,
            expected=expected,
            variant="B2",
            stage="publication_swap",
            error=OSError("synthetic publication error"),
            started=time.perf_counter(),
        )

    assert (output_dir / "actions.npz").read_bytes() == personal_actions
    assert (output_dir / "metrics.json").read_text(encoding="utf-8") == personal_metrics
    assert (
        output_dir / "rejection.json"
    ).read_text(encoding="utf-8") == personal_rejection


def test_publication_failure_restore_conflict_leaves_safe_actionless_working_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch failure restoration overwriting a newly occupied canonical path."""

    _, output_dir = _trusted_no_action_output(tmp_path)
    expected = experiment_module._trusted_run_snapshot(output_dir)
    personal_actions = b"personal canonical action"
    personal_metrics = '{"owner":"user","kind":"canonical"}'
    real_replace = Path.replace

    def occupy_canonical_after_isolation(self: Path, target: Path) -> Path:
        target_path = Path(target)
        result = real_replace(self, target_path)
        if self == output_dir and target_path.name.startswith(
            f".{output_dir.name}.failure-backup-"
        ):
            output_dir.mkdir()
            (output_dir / "actions.npz").write_bytes(personal_actions)
            (output_dir / "metrics.json").write_text(
                personal_metrics, encoding="utf-8"
            )
        return result

    monkeypatch.setattr(Path, "replace", occupy_canonical_after_isolation)

    metrics = experiment_module._mark_canonical_publication_failure(
        output_dir,
        expected=expected,
        variant="B2",
        stage="publication_swap",
        error=OSError("synthetic publication error"),
        started=time.perf_counter(),
    )

    assert metrics["status"] == "failed"
    assert (output_dir / "actions.npz").read_bytes() == personal_actions
    assert (output_dir / "metrics.json").read_text(encoding="utf-8") == personal_metrics
    backup = list(output_dir.parent.glob(f".{output_dir.name}.failure-backup-*"))
    staging = list(output_dir.parent.glob(f".{output_dir.name}.failure-staging-*"))
    assert len(backup) == 1
    assert len(staging) == 1
    assert experiment_module._trusted_run_snapshot(backup[0]) == expected
    assert not (staging[0] / "actions.npz").exists()
    assert json.loads((staging[0] / "metrics.json").read_text(encoding="utf-8"))[
        "status"
    ] == "failed"
    assert (staging[0] / "rejection.json").is_file()


def test_rollback_post_success_error_marks_restored_output_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch rollback succeeding physically before surfacing an I/O error."""

    config_path, output_dir = _trusted_no_action_output(tmp_path)
    real_replace = Path.replace

    def fail_swap_and_report_after_restore(self: Path, target: Path) -> Path:
        target_path = Path(target)
        if self.name.startswith(f".{output_dir.name}.staging-"):
            raise OSError("synthetic staging swap failure")
        if self.name.startswith(f".{output_dir.name}.backup-"):
            real_replace(self, target_path)
            raise OSError("synthetic post-rollback error")
        return real_replace(self, target_path)

    monkeypatch.setattr(Path, "replace", fail_swap_and_report_after_restore)

    metrics = run_experiment(config_path, output_dir, variant="B2")

    assert metrics["status"] == "failed"
    assert metrics["failure_stage"] == "publication_swap"
    assert (output_dir / "rejection.json").is_file()
    assert not (output_dir / "actions.npz").exists()
    assert not list(output_dir.parent.glob(f".{output_dir.name}.backup-*"))


def test_parent_path_alias_serializes_across_processes(tmp_path: Path) -> None:
    """Catch lexical lock keys allowing alias paths to publish concurrently."""

    physical_parent = tmp_path / "physical-parent"
    physical_parent.mkdir()
    alias_parent = tmp_path / "parent-alias"
    try:
        alias_parent.symlink_to(physical_parent, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        if sys.platform != "win32":
            pytest.skip(f"directory symlinks are unavailable: {error}")
        alias_parent = Path("\\\\?\\" + str(physical_parent))
        if not alias_parent.samefile(physical_parent):
            pytest.skip(f"directory path aliases are unavailable: {error}")

    entered_first = tmp_path / "entered-first"
    entered_second = tmp_path / "entered-second"
    ready_second = tmp_path / "ready-second"
    release_first = tmp_path / "release-first"
    release_second = tmp_path / "release-second"
    release_second.write_text("go", encoding="utf-8")
    script = """
import sys
import time
from pathlib import Path
from webvideo_to_data.experiment import _output_candidate, _serialized_output

destination = _output_candidate(sys.argv[1])
entered = Path(sys.argv[2])
release = Path(sys.argv[3])
ready = Path(sys.argv[4]) if len(sys.argv) > 4 else None
if ready is not None:
    ready.write_text("ready", encoding="utf-8")
with _serialized_output(destination):
    entered.write_text("entered", encoding="utf-8")
    deadline = time.monotonic() + 10.0
    while not release.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("release sentinel was not created")
        time.sleep(0.01)
"""
    child_env = os.environ.copy()
    child_env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

    first = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(physical_parent / "output"),
            str(entered_first),
            str(release_first),
        ],
        env=child_env,
    )
    second: subprocess.Popen[bytes] | None = None
    try:
        deadline = time.monotonic() + 10.0
        while not entered_first.exists():
            assert first.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        second = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(alias_parent / "output"),
                str(entered_second),
                str(release_second),
                str(ready_second),
            ],
            env=child_env,
        )
        deadline = time.monotonic() + 10.0
        while not ready_second.exists():
            assert second.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        time.sleep(0.25)
        assert not entered_second.exists()
        release_first.write_text("go", encoding="utf-8")
        first.wait(timeout=10.0)
        second.wait(timeout=10.0)
        assert first.returncode == 0
        assert second.returncode == 0
        assert entered_second.is_file()
    finally:
        release_first.touch()
        if first.poll() is None:
            first.kill()
            first.wait(timeout=5.0)
        if second is not None and second.poll() is None:
            second.kill()
            second.wait(timeout=5.0)


def test_backup_cleanup_failure_marks_canonical_run_failed_without_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch post-publish cleanup failure escaping without an explicit rejection."""

    config_path, output_dir = _trusted_no_action_output(tmp_path)
    real_unlink = Path.unlink

    def fail_backup_cleanup(self: Path, *args: object, **kwargs: object) -> None:
        if self.parent.name.startswith(f".{output_dir.name}.backup-"):
            raise OSError("synthetic backup cleanup failure")
        real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_backup_cleanup)

    metrics = run_experiment(config_path, output_dir, variant="B2")

    assert metrics["status"] == "failed"
    assert metrics["failure_stage"] == "publication_cleanup"
    assert (output_dir / "rejection.json").is_file()
    assert not (output_dir / "actions.npz").exists()
    assert len(list(output_dir.parent.glob(f".{output_dir.name}.backup-*"))) == 1


def test_mutated_backup_is_quarantined_and_never_recursively_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch backup mutation between swap and cleanup being recursively deleted."""

    config_path, output_dir = _trusted_no_action_output(tmp_path)
    real_replace = Path.replace

    def mutate_backup_after_rename(self: Path, target: Path) -> Path:
        target_path = Path(target)
        result = real_replace(self, target_path)
        if self == output_dir and target_path.name.startswith(
            f".{output_dir.name}.backup-"
        ):
            (target_path / "user-added.txt").write_text("preserve me", encoding="utf-8")
        return result

    monkeypatch.setattr(Path, "replace", mutate_backup_after_rename)

    metrics = run_experiment(config_path, output_dir, variant="B2")

    assert metrics["status"] == "failed"
    assert metrics["failure_stage"] == "publication_cleanup"
    backups = list(output_dir.parent.glob(f".{output_dir.name}.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "user-added.txt").read_text(encoding="utf-8") == "preserve me"
    assert not (output_dir / "actions.npz").exists()

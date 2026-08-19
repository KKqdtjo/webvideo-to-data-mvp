from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import yaml

from webvideo_to_data.experiment import run_experiment
from webvideo_to_data.media import sha256_file


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
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "SYNTHETIC",
                "source": {
                    "path": str(source),
                    "sha256": sha256_file(source),
                    "fps": 10.0,
                    "roi_xywh": [18, 24, 20, 20],
                },
                "tracking": {
                    "forward_backward_threshold_px": forward_backward_threshold_px,
                    "minimum_live_points": 8,
                    "minimum_valid_ratio": minimum_valid_ratio,
                },
                "scene": {
                    "x_bounds_m": [-0.15, 0.15],
                    "y_bounds_m": [0.35, 0.65],
                    "b0_start_m": [0.12, 0.45, 0.04],
                    "b0_goal_m": [-0.05, 0.55, 0.13],
                },
                "simulation": {
                    "b0_mode": "physics_grasp",
                    "b1_mode": "kinematic_replay",
                    "render_size": [96, 72],
                    "render_every": 20,
                },
                "random_seed": 19,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _synthetic_config(
    tmp_path: Path,
    *,
    minimum_valid_ratio: float = 0.1,
    forward_backward_threshold_px: float = 1.5,
) -> Path:
    source = tmp_path / "moving.mp4"
    _write_moving_object_video(source)
    config_path = tmp_path / "experiment.yaml"
    _write_config(
        config_path,
        source,
        minimum_valid_ratio=minimum_valid_ratio,
        forward_backward_threshold_px=forward_backward_threshold_px,
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


def test_run_experiment_writes_auditable_orchestration_outputs(tmp_path: Path) -> None:
    """Catch a runner that skips a pipeline stage or omits its audit metrics."""

    config_path = _synthetic_config(tmp_path)
    source = tmp_path / "moving.mp4"
    output_dir = tmp_path / "run"

    metrics = run_experiment(config_path, output_dir, variant="B1")

    expected = {
        "provenance.json",
        "trajectory_2d.npz",
        "phases.json",
        "metrics.json",
        "tracking_overlay.mp4",
        "trajectory_2d.png",
        "mujoco_replay.mp4",
        "side_by_side.mp4",
        "contact_sheet.png",
        "robot_reference.npz",
        "rejection.json",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    with np.load(output_dir / "trajectory_2d.npz") as trajectory:
        assert set(trajectory.files) == {"timestamps_s", "centers_px", "confidence"}
    on_disk = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics == on_disk
    assert {
        "source_sha256",
        "valid_track_ratio",
        "phase_count",
        "variant",
        "simulation_mode",
        "placed_successfully",
    } <= on_disk.keys()
    assert on_disk["source_sha256"] == sha256_file(source)
    assert on_disk["variant"] == "B1"
    assert on_disk["simulation_mode"] == "kinematic_replay"
    assert on_disk["placed_successfully"] is False
    assert on_disk["status"] == "rejected"
    assert on_disk["reason"] == "kinematic_replay_not_action"
    assert not (output_dir / "actions.npz").exists()
    for name in ("tracking_overlay.mp4", "mujoco_replay.mp4", "side_by_side.mp4"):
        _assert_readable_video(output_dir / name)
    for name in ("trajectory_2d.png", "contact_sheet.png"):
        assert cv2.imread(str(output_dir / name)) is not None


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
    assert not (output_dir / "actions.npz").exists()


def test_failed_physics_validation_does_not_export_actions(tmp_path: Path) -> None:
    """Catch a failed physics replay being exported as robot action data."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "B0"

    metrics = run_experiment(config_path, output_dir, variant="B0", no_render=True)

    assert metrics["status"] == "rejected"
    assert metrics["reason"] == "physics_validation_failed"
    assert metrics["reachability_ratio"] < 0.95
    assert (output_dir / "robot_reference.npz").is_file()
    assert (output_dir / "rejection.json").is_file()
    assert not (output_dir / "actions.npz").exists()


def test_rejected_tracking_writes_diagnostics_but_not_actions(tmp_path: Path) -> None:
    """Catch a rejected perception run leaking unvalidated robot actions."""

    config_path = _synthetic_config(tmp_path, minimum_valid_ratio=1.01)
    output_dir = tmp_path / "rejected"

    metrics = run_experiment(config_path, output_dir, variant="B1")

    assert metrics["status"] == "rejected"
    assert metrics["rejection_stage"] == "tracking"
    assert (output_dir / "rejection.json").is_file()
    assert (output_dir / "trajectory_2d.npz").is_file()
    assert (output_dir / "tracking_overlay.mp4").is_file()
    assert not (output_dir / "actions.npz").exists()


def test_runner_applies_configured_forward_backward_threshold(tmp_path: Path) -> None:
    """Catch the locked tracking threshold being parsed but silently ignored."""

    config_path = _synthetic_config(tmp_path, forward_backward_threshold_px=0.0)

    metrics = run_experiment(config_path, tmp_path / "strict", variant="B1")

    assert metrics["status"] == "rejected"
    assert metrics["rejection_stage"] == "tracking"


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

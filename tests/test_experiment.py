from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import cv2
import numpy as np
import pytest
import yaml

import webvideo_to_data.experiment as experiment_module
from webvideo_to_data.experiment import _validate_mp4, run_experiment
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
    b0_start_m: tuple[float, float, float] = (0.12, 0.45, 0.04),
    b0_goal_m: tuple[float, float, float] = (-0.05, 0.55, 0.13),
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
                    "b0_start_m": list(b0_start_m),
                    "b0_goal_m": list(b0_goal_m),
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


def test_b2_replaces_reused_b1_output_instead_of_leaving_stale_files(
    tmp_path: Path,
) -> None:
    """Catch a B2 not-run publication retaining B1 media or stale actions."""

    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "reused"
    run_experiment(config_path, output_dir, variant="B1")
    (output_dir / "actions.npz").write_bytes(b"stale action")

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


def test_hash_failure_replaces_old_output_with_failed_run(tmp_path: Path) -> None:
    """Catch source-integrity failure leaving a previous successful-looking run."""

    config_path = _synthetic_config(tmp_path)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["source"]["sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    output_dir = tmp_path / "hash-failure"
    output_dir.mkdir()
    (output_dir / "actions.npz").write_bytes(b"stale action")
    (output_dir / "metrics.json").write_text('{"status":"completed"}', encoding="utf-8")

    metrics = run_experiment(config_path, output_dir, variant="B0")

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
    assert (output_dir / "rejection.json").is_file()


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
    assert Path(provenance["source_path"]) == source.resolve()


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

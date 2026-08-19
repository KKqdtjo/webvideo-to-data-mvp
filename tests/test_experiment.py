from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from dataclasses import replace
import shutil
import threading
import time

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


def _trusted_action_output(tmp_path: Path) -> tuple[Path, Path]:
    config_path = _synthetic_config(tmp_path)
    output_dir = tmp_path / "trusted-action-run"
    run_experiment(config_path, output_dir, variant="B0", no_render=True)
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        status="completed",
        placed_successfully=True,
        reachability_ratio=1.0,
    )
    metrics.pop("reason", None)
    metrics.pop("rejection_stage", None)
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    (output_dir / "rejection.json").unlink()
    shutil.copyfile(
        output_dir / "robot_reference.npz", output_dir / "actions.npz"
    )
    (output_dir / "run_manifest.json").unlink()
    experiment_module._write_run_manifest(output_dir, metrics)
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

    config_path, output_dir = _trusted_action_output(tmp_path)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["source"]["sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

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


def test_rejected_run_marker_cannot_trust_an_action_artifact(tmp_path: Path) -> None:
    """Catch a refreshed marker legitimizing stale actions on a rejected run."""

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
    """Catch staging swap rollback exposing an old action-bearing successful run."""

    config_path, output_dir = _trusted_action_output(tmp_path)
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

    config_path, output_dir = _trusted_action_output(tmp_path)
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

    config_path, output_dir = _trusted_action_output(tmp_path)
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

    _, output_dir = _trusted_action_output(tmp_path)
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


def test_publication_failure_marker_does_not_mutate_post_snapshot_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch canonical replacement between ownership validation and first mutation."""

    _, output_dir = _trusted_action_output(tmp_path)
    expected = experiment_module._trusted_run_snapshot(output_dir)
    runner_output = tmp_path / "runner-output-moved-after-snapshot"
    personal_actions = b"personal action bytes"
    personal_metrics = '{"owner":"user","kind":"metrics"}'
    personal_rejection = '{"owner":"user","kind":"rejection"}'
    real_snapshot = experiment_module._directory_snapshot
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

    monkeypatch.setattr(experiment_module, "_directory_snapshot", snapshot_then_swap)

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

    _, output_dir = _trusted_action_output(tmp_path)
    expected = experiment_module._trusted_run_snapshot(output_dir)
    personal_actions = b"personal canonical action"
    personal_metrics = '{"owner":"user","kind":"canonical"}'
    real_replace = Path.replace

    def occupy_canonical_after_isolation(self: Path, target: Path) -> Path:
        target_path = Path(target)
        result = real_replace(self, target_path)
        if self == output_dir and target_path.name.startswith(
            f".{output_dir.name}.failure-working-"
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
    working = list(output_dir.parent.glob(f".{output_dir.name}.failure-working-*"))
    assert len(working) == 1
    assert not (working[0] / "actions.npz").exists()
    assert json.loads((working[0] / "metrics.json").read_text(encoding="utf-8"))[
        "status"
    ] == "failed"
    assert (working[0] / "rejection.json").is_file()


def test_rollback_post_success_error_marks_restored_output_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch rollback succeeding physically before surfacing an I/O error."""

    config_path, output_dir = _trusted_action_output(tmp_path)
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

    first = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(physical_parent / "output"),
            str(entered_first),
            str(release_first),
        ]
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
            ]
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

    config_path, output_dir = _trusted_action_output(tmp_path)
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

    config_path, output_dir = _trusted_action_output(tmp_path)
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

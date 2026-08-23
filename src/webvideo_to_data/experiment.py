"""Auditable orchestration for the EXP-001 video-to-MuJoCo pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Literal
from uuid import uuid4

import cv2
from importlib import metadata as importlib_metadata
import matplotlib
import mujoco
import numpy as np
import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .artifacts import (
    NPZContract,
    ROBOT_REFERENCE_V1,
    TRAJECTORY_2D_V1,
    baseline_control_contract,
    pinned_model_identity,
    pinned_model_dimensions,
    simulation_contract,
    terminal_metrics_error,
    verify_run_directory,
    write_npz_artifact,
)
from .contact import infer_motion_phases
from .config import ExperimentConfig, load_experiment_config, to_public_resolved_mapping
from .media import probe_video, sha256_file
from .ik import plan_joint_control
from .physics_validation import (
    PhysicsRolloutFailure,
    PhysicsValidationResult,
    validate_rollout,
)
from .retargeting import build_manual_b0_reference, build_pick_place_reference
from .redaction import redact_text
from .scene import DEFAULT_SCENE_PATH, load_panda_scene
from .schema import RunStatus
from .simulation import (
    _DEFAULT_MODEL_PATH,
    run_joint_control_program,
    run_mujoco_replay,
)
from .tracking import track_roi_lk
from .visualization import (
    MediaLabels,
    compose_status_banner,
    labels_for_metrics,
    letterbox_frame,
    render_comparison_video,
    render_tracking_overlay,
)


Variant = Literal["B0", "B1", "B2", "B3", "B4"]
_RUN_MANIFEST_PRODUCER = "webvideo_to_data.experiment"
_RUN_MANIFEST_VERSION = 4
_GENERATED_RUN_FILES = {
    "actions.npz",
    "baseline_control_trace.npz",
    "baseline_control_trace.schema.json",
    "contact_sheet.png",
    "metrics.json",
    "mujoco_replay.mp4",
    "phases.json",
    "provenance.json",
    "rejection.json",
    "robot_reference.npz",
    "robot_reference.schema.json",
    "run_manifest.json",
    "side_by_side.mp4",
    "simulation.npz",
    "simulation.schema.json",
    "tracking_overlay.mp4",
    "trajectory_2d.npz",
    "trajectory_2d.schema.json",
    "trajectory_2d.png",
}
_MEDIA_CLASSIFICATION: dict[str, tuple[str, bool]] = {
    "contact_sheet.png": ("source_contact_sheet", True),
    "mujoco_replay.mp4": ("simulation_only", False),
    "side_by_side.mp4": ("source_simulation_comparison", True),
    "tracking_overlay.mp4": ("source_tracking_overlay", True),
    "trajectory_2d.png": ("derived_trajectory_plot", False),
}
_OUTPUT_LOCKS_GUARD = threading.Lock()
_OUTPUT_LOCKS: dict[str, threading.Lock] = {}


@dataclass(frozen=True)
class _RunSnapshot:
    directory_identity: tuple[int, int]
    files: tuple[tuple[str, int, str, int, int, int], ...]


class _PipelineFailure(Exception):
    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


def _action_gate(
    variant: Variant, physics: PhysicsValidationResult | None
) -> dict[str, Any]:
    if variant == "B0" and physics is not None and physics.passed:
        return {
            "collision_validation": "passed",
            "physics_validation": "passed",
            "action_export_eligible": False,
            "action_export_reason": "manual_baseline_not_video_grounded",
            "action_exported": False,
        }
    if variant == "B0":
        return {
            "collision_validation": "failed",
            "physics_validation": "failed",
            "action_export_eligible": False,
            "action_export_reason": "physics_validation_failed",
            "action_exported": False,
        }
    if variant == "B1":
        return {
            "collision_validation": "not_applicable_kinematic",
            "physics_validation": "not_applicable_kinematic",
            "action_export_eligible": False,
            "action_export_reason": "kinematic_replay_not_action",
            "action_exported": False,
        }
    return {
        "collision_validation": "not_run",
        "physics_validation": "not_run",
        "action_export_eligible": False,
        "action_export_reason": "metric_depth_not_available",
        "action_exported": False,
    }


@contextmanager
def _pipeline_stage(name: str) -> Any:
    try:
        yield
    except _PipelineFailure:
        raise
    except Exception as error:
        raise _PipelineFailure(name, error) from error


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as temporary:
        json.dump(payload, temporary, ensure_ascii=False, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _redact_local_paths(message: str, *paths: Path) -> str:
    """Remove known local paths before persisting a failed-run diagnostic."""
    return redact_text(
        message,
        workspace=Path.cwd().resolve(),
        sensitive_paths=paths,
    )


def _write_rgb_video(
    path: Path,
    frames: np.ndarray,
    fps: float,
    labels: MediaLabels | None = None,
) -> None:
    if len(frames) == 0:
        raise ValueError("cannot write a video without frames")
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise ValueError(f"video writer cannot open {path}")
    try:
        for index, frame in enumerate(frames):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if labels is not None:
                bgr = compose_status_banner(
                    bgr, labels, simulation_time_s=index / fps
                )
            writer.write(bgr)
    finally:
        writer.release()


def _ffprobe_video_facts(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration,format_name",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        duration = float(payload["format"]["duration"])
        width = int(stream["width"])
        height = int(stream["height"])
        numerator, denominator = stream["avg_frame_rate"].split("/", maxsplit=1)
        fps = float(numerator) / float(denominator)
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"ffprobe returned no valid video facts for {path}") from error
    if (
        completed.returncode != 0
        or duration <= 0.0
        or width <= 0
        or height <= 0
        or fps <= 0.0
        or not stream.get("codec_name")
    ):
        raise ValueError(f"ffprobe rejected {path}: {completed.stderr.strip()}")
    frame_count_raw = stream.get("nb_frames")
    try:
        frame_count = int(frame_count_raw)
    except (TypeError, ValueError):
        frame_count = None
    return {
        "codec_name": str(stream["codec_name"]),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": duration,
        "format_name": str(payload["format"].get("format_name", "")),
    }


def _validate_mp4(path: Path) -> dict[str, Any]:
    facts = _ffprobe_video_facts(path)
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        if not capture.isOpened() or not ok or frame is None or frame.size == 0:
            raise ValueError(f"MP4 is not decodable: {path}")
    finally:
        capture.release()
    return facts


def _git_generator_provenance() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repository,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    )
    return {"git_commit": commit, "git_dirty": dirty}


def _runtime_provenance() -> dict[str, Any]:
    package_names = (
        "webvideo-to-data",
        "numpy",
        "scipy",
        "opencv-python",
        "mujoco",
        "imageio",
        "matplotlib",
        "PyYAML",
    )
    packages = {name: importlib_metadata.version(name) for name in package_names}
    packages["mujoco"] = mujoco.__version__
    packages["opencv-python"] = cv2.__version__
    ffprobe_version = subprocess.run(
        ["ffprobe", "-version"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()[0]
    return {
        "python": sys.version,
        "packages": packages,
        "tools": {"ffprobe": ffprobe_version},
    }


def _render_trajectory_plot(
    trajectory: Any,
    phases: Any,
    output: Path,
    labels: MediaLabels | None = None,
) -> None:
    speed = np.linalg.norm(np.gradient(trajectory.centers_px, axis=0), axis=1)
    delta_t = np.gradient(trajectory.timestamps_s)
    speed = np.divide(speed, delta_t, out=np.zeros_like(speed), where=delta_t > 0)
    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(trajectory.timestamps_s, trajectory.centers_px[:, 0], label="x (px)")
    axes[0].plot(trajectory.timestamps_s, trajectory.centers_px[:, 1], label="y (px)")
    axes[0].set_ylabel("position (px)")
    axes[0].legend(loc="best")
    axes[1].plot(trajectory.timestamps_s, speed, color="tab:red", label="speed")
    axes[1].set_ylabel("speed (px/s)")
    axes[1].set_xlabel("time (s)")
    if labels is not None:
        figure.suptitle(
            f"{labels.status}\n{labels.mode} · {labels.metric_warning}",
            color="#8f1425",
            fontweight="bold",
        )
    for phase in phases:
        boundary_s = trajectory.timestamps_s[phase.start_frame]
        for axis in axes:
            axis.axvline(boundary_s, color="0.5", linewidth=0.8, alpha=0.7)
        axes[0].text(boundary_s, axes[0].get_ylim()[1], phase.label, rotation=90, va="top")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def _read_frame(
    capture: cv2.VideoCapture,
    index: int,
    size: tuple[int, int],
    color: tuple[int, int, int],
) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"cannot read checkpoint frame {index}")
    return letterbox_frame(frame, size, color)


def _render_contact_sheet(
    source_path: Path,
    overlay_path: Path,
    simulation_frames_rgb: np.ndarray,
    phases: Any,
    frame_count: int,
    output_path: Path,
    tile_size: tuple[int, int],
    letterbox_bgr: tuple[int, int, int],
    labels: MediaLabels,
    source_duration_s: float,
    simulation_duration_s: float,
) -> None:
    checkpoints = [
        0,
        phases[1].start_frame,
        phases[2].start_frame,
        frame_count - 1,
    ]
    source_capture = cv2.VideoCapture(str(source_path))
    overlay_capture = cv2.VideoCapture(str(overlay_path))
    try:
        rows: list[np.ndarray] = []
        rows.append(
            np.hstack(
                [
                    _read_frame(source_capture, index, tile_size, letterbox_bgr)
                    for index in checkpoints
                ]
            )
        )
        rows.append(
            np.hstack(
                [
                    _read_frame(overlay_capture, index, tile_size, letterbox_bgr)
                    for index in checkpoints
                ]
            )
        )
        simulation_panels = []
        for index in checkpoints:
            sim_index = min(
                len(simulation_frames_rgb) - 1,
                round(index / max(1, frame_count - 1) * (len(simulation_frames_rgb) - 1)),
            )
            frame = cv2.cvtColor(simulation_frames_rgb[sim_index], cv2.COLOR_RGB2BGR)
            simulation_panels.append(letterbox_frame(frame, tile_size, letterbox_bgr))
        rows.append(np.hstack(simulation_panels))
        sheet = compose_status_banner(
            np.vstack(rows),
            labels,
            source_time_s=source_duration_s,
            simulation_time_s=simulation_duration_s,
        )
        if not cv2.imwrite(str(output_path), sheet):
            raise ValueError("contact sheet image writer failed")
    finally:
        source_capture.release()
        overlay_capture.release()


def _rejected_metrics(
    *,
    variant: Variant,
    source_sha256: str,
    lk_point_availability_ratio: float,
    stage: str,
    reason: str,
    started: float,
) -> dict[str, Any]:
    action = _action_gate(variant, None)
    terminal_reason = action["action_export_reason"] if variant == "B1" else reason
    metrics = {
        "status": RunStatus.REJECTED.value,
        "source_sha256": source_sha256,
        "lk_point_availability_ratio": lk_point_availability_ratio,
        "lk_metric_scope": "point_availability_not_semantic_accuracy",
        "semantic_accuracy_status": "not_measured",
        "phase_count": 0,
        "variant": variant,
        "simulation_mode": None,
        "placed_successfully": False,
        "rejection_stage": stage,
        "reason": terminal_reason,
        "runtime_s": time.perf_counter() - started,
        **action,
    }
    if terminal_reason != reason:
        metrics["perception_rejection_reason"] = reason
    return metrics


def _output_candidate(output_dir: str | Path) -> Path:
    lexical_destination = Path(os.path.abspath(os.fspath(output_dir)))
    if lexical_destination == Path(lexical_destination.anchor):
        raise ValueError("output target is too broad for generated-run replacement")
    lexical_destination.parent.mkdir(parents=True, exist_ok=True)
    destination = lexical_destination.parent.resolve(strict=True) / lexical_destination.name
    if destination.exists() and destination.samefile(Path.cwd()):
        raise ValueError("output target is too broad for generated-run replacement")
    return destination


@contextmanager
def _serialized_output(destination: Path) -> Any:
    parent_stat = destination.parent.stat()
    lock_key = ":".join(
        (
            str(int(parent_stat.st_dev)),
            str(int(parent_stat.st_ino)),
            os.path.normcase(destination.name),
        )
    )
    with _OUTPUT_LOCKS_GUARD:
        thread_lock = _OUTPUT_LOCKS.setdefault(lock_key, threading.Lock())
    lock_root = Path(tempfile.gettempdir()) / "webvideo_to_data_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_name = sha256(lock_key.encode("utf-8")).hexdigest() + ".lock"
    with thread_lock:
        with (lock_root / lock_name).open("a+b") as lock_file:
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                while True:
                    try:
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        time.sleep(0.05)
                try:
                    yield
                finally:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path.name}")
    return value


def _snapshot_files(directory: Path) -> tuple[tuple[str, int, str, int, int, int], ...]:
    signatures: list[tuple[str, int, str, int, int, int]] = []
    for child in sorted(directory.iterdir(), key=lambda item: item.name):
        if child.is_symlink() or not child.is_file():
            raise ValueError(f"unrecognized files in generated run: {child.name}")
        stat = child.stat()
        signatures.append(
            (
                child.name,
                stat.st_size,
                sha256_file(child),
                int(stat.st_mtime_ns),
                int(stat.st_dev),
                int(stat.st_ino),
            )
        )
    return tuple(signatures)


def _directory_snapshot(directory: Path) -> _RunSnapshot:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("snapshot target must be a real directory")
    before = directory.stat()
    files = _snapshot_files(directory)
    after = directory.stat()
    before_identity = (int(before.st_dev), int(before.st_ino))
    after_identity = (int(after.st_dev), int(after.st_ino))
    if before_identity != after_identity:
        raise ValueError("snapshot target changed during inspection")
    return _RunSnapshot(directory_identity=after_identity, files=files)


def _terminal_metrics_error(
    metrics: dict[str, Any], file_names: set[str]
) -> str | None:
    return terminal_metrics_error(metrics, file_names)


def _trusted_run_snapshot(directory: Path) -> _RunSnapshot:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("output target must be a real directory")
    children = {child.name: child for child in directory.iterdir()}
    unrecognized = {
        name
        for name, child in children.items()
        if name not in _GENERATED_RUN_FILES or child.is_symlink() or not child.is_file()
    }
    if unrecognized:
        raise ValueError(
            "output target contains unrecognized files: "
            + ", ".join(sorted(unrecognized))
        )
    try:
        verified = verify_run_directory(directory)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("output target lacks a trusted generated-run marker") from error
    current = directory.stat()
    current_identity = (int(current.st_dev), int(current.st_ino))
    if current_identity != verified.directory_identity:
        raise ValueError("output target changed after verification")
    return _RunSnapshot(
        directory_identity=verified.directory_identity,
        files=tuple(
            (name, size, digest, mtime_ns, device, inode)
            for name, (size, digest, mtime_ns, device, inode) in sorted(
                verified.snapshot.items()
            )
        ),
    )


def _validated_output_target(
    destination: Path,
) -> tuple[Path, _RunSnapshot | None]:
    if destination.exists():
        return destination, _trusted_run_snapshot(destination)
    return destination, None


def _manifest_model_path(variant: Variant) -> Path:
    return _DEFAULT_MODEL_PATH if variant == "B1" else DEFAULT_SCENE_PATH


def _manifest_context(
    staging: Path,
    metrics: dict[str, Any],
    *,
    config_file: Path | None,
    manifest_seed: dict[str, Any] | None,
) -> tuple[str, str, str, tuple[int, int, int]]:
    provenance: dict[str, Any] = {}
    provenance_path = staging / "provenance.json"
    if provenance_path.is_file():
        provenance = _json_object(provenance_path)
    prior_manifest: dict[str, Any] = dict(manifest_seed or {})
    manifest_path = staging / "run_manifest.json"
    if not prior_manifest and manifest_path.is_file():
        try:
            prior_manifest = _json_object(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            prior_manifest = {}
    config_sha256 = prior_manifest.get("config_sha256")
    config_record = provenance.get("config", {})
    if not isinstance(config_sha256, str):
        config_sha256 = (
            config_record.get("sha256")
            if isinstance(config_record, dict)
            else None
        )
    if not isinstance(config_sha256, str) and config_file is not None:
        config_sha256 = sha256_file(config_file)
    if not isinstance(config_sha256, str):
        raise ValueError("cannot determine config hash for run manifest")
    model_sha256 = pinned_model_identity(metrics["variant"])
    model_record = provenance.get("model", {})
    if (
        isinstance(model_record, dict)
        and isinstance(model_record.get("sha256"), str)
        and provenance.get("variant") == metrics["variant"]
        and model_record.get("sha256") != model_sha256
    ):
        raise ValueError("provenance model hash does not identify the pinned model")
    dimensions = pinned_model_dimensions(metrics["variant"], model_sha256)
    source_sha256 = "not_used"
    prior_source = prior_manifest.get("source_sha256")
    if metrics["variant"] == "B1" and isinstance(prior_source, str):
        source_sha256 = prior_source
    source_record = provenance.get("source", {})
    if (
        source_sha256 == "not_used"
        and metrics["variant"] == "B1"
        and provenance.get("variant") == metrics["variant"]
        and isinstance(source_record, dict)
    ):
        candidate = source_record.get("sha256")
        if isinstance(candidate, str) and source_record.get("verification") != "not_accessed_for_variant":
            source_sha256 = candidate
    return config_sha256, source_sha256, model_sha256, dimensions


def _write_run_manifest(
    staging: Path,
    metrics: dict[str, Any],
    *,
    config_file: Path | None = None,
    manifest_seed: dict[str, Any] | None = None,
) -> None:
    config_sha256, source_sha256, model_sha256, dimensions = _manifest_context(
        staging,
        metrics,
        config_file=config_file,
        manifest_seed=manifest_seed,
    )
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(staging.iterdir(), key=lambda item: item.name):
        if path.name == "run_manifest.json":
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"cannot manifest non-file artifact: {path.name}")
        entry: dict[str, Any] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        media = _MEDIA_CLASSIFICATION.get(path.name)
        if path.suffix.lower() in {".gif", ".mp4", ".png"}:
            if media is None:
                raise ValueError(f"media artifact lacks classification: {path.name}")
            entry["media_role"], entry["contains_private_source_frames"] = media
        elif media is not None:
            raise ValueError(f"non-media artifact has media classification: {path.name}")
        files[path.name] = entry
    _atomic_json(
        staging / "run_manifest.json",
        {
            "producer": _RUN_MANIFEST_PRODUCER,
            "format_version": _RUN_MANIFEST_VERSION,
            "variant": metrics["variant"],
            "status": metrics["status"],
            "reason": metrics.get("reason"),
            "action_export_eligible": metrics.get("action_export_eligible"),
            "action_export_reason": metrics.get("action_export_reason"),
            "action_exported": metrics.get("action_exported"),
            "config_sha256": config_sha256,
            "source_sha256": source_sha256,
            "model_sha256": model_sha256,
            "model_nq": dimensions[0],
            "model_nv": dimensions[1],
            "model_nu": dimensions[2],
            "files": files,
        },
    )


def _publication_failure_metrics(
    *,
    variant: Variant,
    stage: str,
    error: Exception,
    started: float,
    sensitive_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    return {
        "status": RunStatus.FAILED.value,
        "variant": variant,
        "reason": "publication_exception",
        "failure_stage": stage,
        "error_type": type(error).__name__,
        "error_message": _redact_local_paths(str(error), *sensitive_paths),
        "runtime_s": time.perf_counter() - started,
        **_action_gate(variant, None),
    }


def _quarantine_failure_backup(backup: Path, expected: _RunSnapshot) -> Path:
    if _trusted_run_snapshot(backup) != expected:
        raise ValueError("failure backup changed before quarantine")
    prefix, separator, _ = backup.name.partition(".failure-backup-")
    quarantine = backup.parent / f"{prefix}.failure-quarantine-{uuid4().hex}"
    if (
        not separator
        or quarantine.exists()
        or quarantine.parent != backup.parent
        or not quarantine.name.startswith(f"{prefix}.failure-quarantine-")
    ):
        raise ValueError("unsafe failure backup quarantine target")
    try:
        backup.replace(quarantine)
    except Exception as error:
        if backup.exists():
            raise ValueError("failure backup quarantine failed; backup preserved") from error
        if not quarantine.exists() or _trusted_run_snapshot(quarantine) != expected:
            raise ValueError("failure backup quarantine changed after rename") from error
    return quarantine


def _remove_owned_failure_staging(staging: Path) -> None:
    if not staging.exists():
        return
    allowed = {"rejection.json", "metrics.json", "run_manifest.json"}
    for child in list(staging.iterdir()):
        if child.name not in allowed or child.is_symlink() or not child.is_file():
            raise ValueError("failure staging contains an unknown entry")
        child.unlink()
    staging.rmdir()


def _mark_canonical_publication_failure(
    destination: Path,
    *,
    expected: _RunSnapshot,
    variant: Variant,
    stage: str,
    error: Exception,
    started: float,
    manifest_seed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def canonical_snapshot(path: Path) -> _RunSnapshot:
        return _trusted_run_snapshot(path) if expected.files else _directory_snapshot(path)

    try:
        current = canonical_snapshot(destination)
    except ValueError as validation_error:
        raise ValueError("canonical output changed before failure reporting") from validation_error
    if current != expected:
        raise ValueError("canonical output changed before failure reporting")
    backup = destination.parent / f".{destination.name}.failure-backup-{uuid4().hex}"
    failure_staging = destination.parent / (
        f".{destination.name}.failure-staging-{uuid4().hex}"
    )
    if (
        backup.parent != destination.parent
        or failure_staging.parent != destination.parent
        or not backup.name.startswith(f".{destination.name}.failure-backup-")
        or not failure_staging.name.startswith(f".{destination.name}.failure-staging-")
        or backup.exists()
        or failure_staging.exists()
    ):
        raise ValueError("unsafe publication failure transaction target")
    try:
        destination.replace(backup)
    except Exception as isolation_error:
        if backup.exists() and not destination.exists():
            pass
        elif destination.exists():
            try:
                if canonical_snapshot(destination) == expected:
                    raise ValueError(
                        "canonical output could not be isolated for failure reporting"
                    ) from isolation_error
            except ValueError as validation_error:
                raise ValueError(
                    "canonical output changed before failure reporting"
                ) from validation_error
        else:
            raise ValueError(
                "canonical output could not be isolated for failure reporting"
            ) from isolation_error
    try:
        moved = canonical_snapshot(backup)
    except ValueError as validation_error:
        if not destination.exists() and backup.exists():
            try:
                backup.replace(destination)
            except Exception:
                pass
        raise ValueError(
            "canonical output changed before failure reporting; "
            f"candidate preserved at {backup if backup.exists() else destination}"
        ) from validation_error
    if moved != expected:
        if not destination.exists() and backup.exists():
            try:
                backup.replace(destination)
            except Exception:
                pass
        raise ValueError(
            "canonical output changed before failure reporting; "
            f"candidate preserved at {backup if backup.exists() else destination}"
        )
    metrics = _publication_failure_metrics(
        variant=variant,
        stage=stage,
        error=error,
        started=started,
        sensitive_paths=(destination, backup, failure_staging),
    )
    seed = dict(manifest_seed or _json_object(backup / "run_manifest.json"))
    try:
        failure_staging.mkdir()
        _atomic_json(
            failure_staging / "rejection.json",
            {
                "stage": stage,
                "reason": "publication_exception",
                "error_type": type(error).__name__,
                "error_message": _redact_local_paths(
                    str(error), destination, backup, failure_staging
                ),
                "action_exported": False,
            },
        )
        _atomic_json(failure_staging / "metrics.json", metrics)
        _write_run_manifest(failure_staging, metrics, manifest_seed=seed)
        failed = _trusted_run_snapshot(failure_staging)
    except Exception:
        cleanup_error: Exception | None = None
        try:
            _remove_owned_failure_staging(failure_staging)
        except Exception as staging_error:
            try:
                _remove_owned_failure_staging(failure_staging)
            except Exception:
                cleanup_error = staging_error
        if not destination.exists() and backup.exists():
            try:
                backup.replace(destination)
            except Exception as restore_error:
                if not (
                    destination.exists()
                    and not backup.exists()
                    and canonical_snapshot(destination) == expected
                ):
                    raise ValueError(
                        f"canonical failure rollback failed; backup preserved at {backup}"
                    ) from restore_error
        if destination.exists():
            try:
                restored = canonical_snapshot(destination)
            except ValueError as restore_error:
                raise ValueError(
                    f"canonical failure rollback could not be verified; backup preserved at {backup}"
                ) from restore_error
            if restored != expected:
                raise ValueError(
                    f"canonical output externally occupied; backup preserved at {backup}"
                )
        if cleanup_error is not None:
            raise ValueError("failure staging cleanup failed") from cleanup_error
        raise
    if destination.exists():
        return metrics
    try:
        failure_staging.replace(destination)
    except Exception as publish_error:
        if not failure_staging.exists() and destination.exists():
            try:
                if _trusted_run_snapshot(destination) == failed:
                    if expected.files:
                        _quarantine_failure_backup(backup, expected)
                    elif _directory_snapshot(backup) == expected:
                        backup.rmdir()
                    return metrics
            except ValueError:
                pass
        if destination.exists():
            return metrics
        try:
            backup.replace(destination)
        except Exception as restore_error:
            if not (
                destination.exists()
                and not backup.exists()
                and canonical_snapshot(destination) == expected
            ):
                raise ValueError(
                    f"canonical failure rollback failed; backup preserved at {backup}; "
                    f"failure staging preserved at {failure_staging}"
                ) from restore_error
        if canonical_snapshot(destination) != expected:
            raise ValueError("canonical failure rollback differs from validated output")
        raise ValueError("failure marker could not be published") from publish_error
    if expected.files:
        _quarantine_failure_backup(backup, expected)
    elif _directory_snapshot(backup) == expected:
        backup.rmdir()
    return metrics


def _assert_destination_unchanged(
    destination: Path, expected: _RunSnapshot | None
) -> None:
    if expected is None:
        if destination.exists():
            raise ValueError("output target changed during run")
        return
    if not destination.exists():
        raise ValueError("output target changed during run")
    try:
        current = _trusted_run_snapshot(destination)
    except ValueError as error:
        raise ValueError("output target changed during run") from error
    if current != expected:
        raise ValueError("output target changed during run")


def _remove_trusted_backup(backup: Path, expected: _RunSnapshot) -> None:
    if _trusted_run_snapshot(backup) != expected:
        raise ValueError("generated backup changed before cleanup")
    expected_files = {
        name: (size, digest, mtime_ns, device, inode)
        for name, size, digest, mtime_ns, device, inode in expected.files
    }
    for child in sorted(backup.iterdir(), key=lambda item: item.name):
        signature = expected_files.get(child.name)
        if (
            signature is None
            or child.is_symlink()
            or not child.is_file()
            or (
                child.stat().st_size,
                sha256_file(child),
                int(child.stat().st_mtime_ns),
                int(child.stat().st_dev),
                int(child.stat().st_ino),
            )
            != signature
        ):
            raise ValueError("generated backup changed before cleanup")
        child.unlink()
    backup.rmdir()


def _publish_staging(
    staging: Path,
    destination: Path,
    expected: _RunSnapshot | None,
    metrics: dict[str, Any],
    *,
    variant: Variant,
    started: float,
) -> dict[str, Any]:
    _assert_destination_unchanged(destination, expected)
    staged = _trusted_run_snapshot(staging)
    manifest_seed = _json_object(staging / "run_manifest.json")
    backup = destination.parent / f".{destination.name}.backup-{uuid4().hex}"
    if backup.parent != destination.parent or not backup.name.startswith(
        f".{destination.name}.backup-"
    ):
        raise ValueError("unsafe generated backup target")
    if expected is None:
        try:
            staging.replace(destination)
            return metrics
        except Exception as error:
            if destination.exists():
                try:
                    current = _trusted_run_snapshot(destination)
                except ValueError as validation_error:
                    raise ValueError("output target changed during run") from validation_error
                if current != staged:
                    raise ValueError("output target changed during run") from error
                return _mark_canonical_publication_failure(
                    destination,
                    expected=staged,
                    variant=variant,
                    stage="publication_swap",
                    error=error,
                    started=started,
                    manifest_seed=manifest_seed,
                )
            if not staging.exists():
                raise ValueError("output target changed during run") from error
            destination.mkdir()
            empty = _directory_snapshot(destination)
            return _mark_canonical_publication_failure(
                destination,
                expected=empty,
                variant=variant,
                stage="publication_swap",
                error=error,
                started=started,
                manifest_seed=manifest_seed,
            )
    try:
        destination.replace(backup)
    except Exception as error:
        if destination.exists():
            _assert_destination_unchanged(destination, expected)
            canonical = expected
        elif backup.exists():
            if _trusted_run_snapshot(backup) != expected:
                raise ValueError("output target changed during run") from error
            destination.mkdir()
            canonical = _directory_snapshot(destination)
        else:
            raise ValueError("output target changed during run") from error
        return _mark_canonical_publication_failure(
            destination,
            expected=canonical,
            variant=variant,
            stage="publication_swap",
            error=error,
            started=started,
            manifest_seed=manifest_seed,
        )
    try:
        if _trusted_run_snapshot(backup) != expected:
            raise ValueError("generated backup changed before cleanup")
    except Exception as error:
        if destination.exists():
            raise ValueError("output target changed during run") from error
        destination.mkdir()
        empty = _directory_snapshot(destination)
        return _mark_canonical_publication_failure(
            destination,
            expected=empty,
            variant=variant,
            stage="publication_cleanup",
            error=error,
            started=started,
            manifest_seed=manifest_seed,
        )
    try:
        staging.replace(destination)
    except Exception as error:
        if destination.exists():
            try:
                current = _trusted_run_snapshot(destination)
            except ValueError as validation_error:
                raise ValueError("output target changed during run") from validation_error
            if current != staged:
                raise ValueError("output target changed during run") from error
            try:
                _remove_trusted_backup(backup, expected)
            except Exception as cleanup_error:
                return _mark_canonical_publication_failure(
                    destination,
                    expected=staged,
                    variant=variant,
                    stage="publication_cleanup",
                    error=cleanup_error,
                    started=started,
                    manifest_seed=manifest_seed,
                )
            return _mark_canonical_publication_failure(
                destination,
                expected=staged,
                variant=variant,
                stage="publication_swap",
                error=error,
                started=started,
                manifest_seed=manifest_seed,
            )
        try:
            backup.replace(destination)
            if _trusted_run_snapshot(destination) != expected:
                raise ValueError("restored output differs from validated snapshot")
        except Exception as restore_error:
            if destination.exists():
                try:
                    if _trusted_run_snapshot(destination) != expected:
                        raise ValueError(
                            "restored output differs from validated snapshot"
                        )
                except Exception as validation_error:
                    raise ValueError(
                        "output target changed during rollback"
                    ) from validation_error
                canonical = expected
            else:
                destination.mkdir()
                canonical = _directory_snapshot(destination)
            return _mark_canonical_publication_failure(
                destination,
                expected=canonical,
                variant=variant,
                stage="publication_swap",
                error=error,
                started=started,
                manifest_seed=manifest_seed,
            )
        return _mark_canonical_publication_failure(
            destination,
            expected=expected,
            variant=variant,
            stage="publication_swap",
            error=error,
            started=started,
            manifest_seed=manifest_seed,
        )
    try:
        _remove_trusted_backup(backup, expected)
    except Exception as error:
        return _mark_canonical_publication_failure(
            destination,
            expected=staged,
            variant=variant,
            stage="publication_cleanup",
            error=error,
            started=started,
            manifest_seed=manifest_seed,
        )
    return metrics


def _validate_required_run_files(
    staging: Path, metrics: dict[str, Any], *, no_render: bool
) -> None:
    terminal_error = _terminal_metrics_error(
        metrics, {path.name for path in staging.iterdir()}
    )
    if terminal_error is not None:
        raise _PipelineFailure(
            "publication", ValueError(f"invalid terminal metrics: {terminal_error}")
        )
    required = {"metrics.json"}
    if metrics["status"] == "failed":
        required.add("rejection.json")
    else:
        required.add("provenance.json")
    if metrics["variant"] == "B0" and metrics["status"] != "failed":
        required.update(
            {
                "baseline_control_trace.npz",
                "baseline_control_trace.schema.json",
                "robot_reference.npz",
                "robot_reference.schema.json",
                "simulation.npz",
                "simulation.schema.json",
                "rejection.json",
            }
        )
        if not no_render:
            required.add("mujoco_replay.mp4")
    if metrics["variant"] == "B1" and metrics["status"] != "failed":
        tracking_rejection = (
            metrics["status"] == "rejected"
            and metrics.get("rejection_stage") == "tracking"
        )
        required.add("trajectory_2d.npz")
        required.add("trajectory_2d.schema.json")
        if tracking_rejection:
            required.add("rejection.json")
            if not no_render:
                required.add("tracking_overlay.mp4")
        else:
            required.update(
                {
                    "phases.json",
                    "robot_reference.npz",
                    "robot_reference.schema.json",
                    "simulation.npz",
                    "simulation.schema.json",
                }
            )
            if not no_render:
                required.update(
                    {
                        "tracking_overlay.mp4",
                        "trajectory_2d.png",
                        "mujoco_replay.mp4",
                        "side_by_side.mp4",
                        "contact_sheet.png",
                    }
                )
            if metrics["status"] == "rejected":
                required.add("rejection.json")
            if metrics["status"] == "completed":
                required.add("actions.npz")
    missing = sorted(name for name in required if not (staging / name).is_file())
    if missing:
        raise _PipelineFailure(
            "publication", ValueError("missing required run files: " + ", ".join(missing))
        )


def _variant_provenance(
    config: ExperimentConfig,
    config_sha256: str,
    variant: Variant,
    *,
    model_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    generated_media_ffprobe: dict[str, dict[str, Any]] = {}
    provenance = {
        "experiment_id": config.experiment_id,
        "generator": _git_generator_provenance(),
        "config": {
            "sha256": config_sha256,
            "resolved": to_public_resolved_mapping(config),
        },
        "source": {
            "id": config.source.id,
            "sha256": config.source.sha256,
            "verification": "not_accessed_for_variant",
        },
        "source_sha256": config.source.sha256,
        "variant": variant,
        "random_seed": config.random_seed,
        "model": {
            "id": "mujoco_menagerie_franka_emika_panda_exp001",
            "sha256": pinned_model_identity(variant),
            "description": "pinned_menagerie_panda_physical_scene",
            "collision_validation": "phase_aware_body_ancestry",
        },
        "runtime": _runtime_provenance(),
        "ffprobe": {"source": None, "generated_media": generated_media_ffprobe},
    }
    return provenance, generated_media_ffprobe


def _terminal_artifact_provenance(
    provenance: dict[str, Any],
    metrics: dict[str, Any],
    *,
    source_sha256: str,
) -> dict[str, Any]:
    return {
        "producer": _RUN_MANIFEST_PRODUCER,
        "git_commit": provenance["generator"]["git_commit"],
        "source_sha256": source_sha256,
        "config_sha256": provenance["config"]["sha256"],
        "model_sha256": provenance["model"]["sha256"],
        "terminal_status": metrics["status"],
        "terminal_reason": metrics["reason"],
        "action_export_eligible": metrics["action_export_eligible"],
    }


def _write_terminal_npz_artifacts(
    destination: Path,
    artifacts: tuple[tuple[str, dict[str, np.ndarray], NPZContract], ...],
    provenance: dict[str, Any],
    metrics: dict[str, Any],
    *,
    source_sha256: str,
) -> None:
    artifact_provenance = _terminal_artifact_provenance(
        provenance, metrics, source_sha256=source_sha256
    )
    for name, arrays, contract in artifacts:
        write_npz_artifact(
            destination / name,
            arrays,
            contract,
            artifact_provenance,
        )


def _simulation_artifact_arrays(simulation: Any) -> dict[str, np.ndarray]:
    return {
        "timestamps_s": np.asarray(simulation.timestamps_s, dtype=np.float64),
        "control": np.asarray(simulation.control, dtype=np.float64),
        "qpos": np.asarray(simulation.qpos, dtype=np.float64),
        "qvel": np.asarray(simulation.qvel, dtype=np.float64),
        "can_pose": np.asarray(simulation.can_pose, dtype=np.float64),
        "tcp_position": np.asarray(simulation.tcp_position, dtype=np.float64),
        "tcp_quaternion_wxyz": np.asarray(
            simulation.tcp_quaternion_wxyz, dtype=np.float64
        ),
        "phase": np.asarray(simulation.phase, dtype="<U16"),
        "contact_count": np.asarray(simulation.contact_count, dtype=np.int64),
        "bilateral_contact": np.asarray(simulation.bilateral_contact, dtype=np.bool_),
        "box_support_contact": np.asarray(
            simulation.box_support_contact, dtype=np.bool_
        ),
        "forbidden_contact": np.asarray(simulation.forbidden_contact, dtype=np.bool_),
        "maximum_penetration_m": np.asarray(
            simulation.maximum_penetration_m, dtype=np.float64
        ),
        "tcp_position_within_tolerance": np.asarray(
            simulation.tcp_position_within_tolerance, dtype=np.bool_
        ),
        "tcp_orientation_within_tolerance": np.asarray(
            simulation.tcp_orientation_within_tolerance, dtype=np.bool_
        ),
        "joint_position_violation": np.asarray(
            simulation.joint_position_violation, dtype=np.bool_
        ),
        "joint_velocity_violation": np.asarray(
            simulation.joint_velocity_violation, dtype=np.bool_
        ),
        "joint_acceleration_violation": np.asarray(
            simulation.joint_acceleration_violation, dtype=np.bool_
        ),
        "valid_numerical_state": np.asarray(
            simulation.valid_numerical_state, dtype=np.bool_
        ),
    }


def _longest_rollout_duration(
    timestamps_s: np.ndarray, selected: np.ndarray
) -> float:
    timestep = (
        float(np.median(np.diff(timestamps_s))) if len(timestamps_s) > 1 else 0.0
    )
    longest = current = 0
    for value in selected:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return round(longest * timestep, 12)


def _execute_manual_b0(
    config: ExperimentConfig,
    config_sha256: str,
    destination: Path,
    *,
    no_render: bool,
    started: float,
) -> dict[str, Any]:
    with _pipeline_stage("model_validation"):
        model, data, ids = load_panda_scene()
    provenance, generated_media_ffprobe = _variant_provenance(
        config, config_sha256, "B0", model_path=DEFAULT_SCENE_PATH
    )

    with _pipeline_stage("retargeting"):
        reference = build_manual_b0_reference(
            config.scene.b0_start_m,
            config.scene.b0_goal_m,
            config.control,
            config.scene.grasp_quaternion_wxyz,
        )
        reference_payload = {
            "timestamps_s": np.asarray(reference.timestamps_s, dtype=np.float64),
            "ee_positions": np.asarray(reference.ee_positions, dtype=np.float64),
            "quaternion_wxyz": np.asarray(
                reference.quaternion_wxyz, dtype=np.float64
            ),
            "gripper_width": np.asarray(reference.gripper_width, dtype=np.float64),
            "phase": np.asarray(reference.phase, dtype="<U16"),
        }
    with _pipeline_stage("control_planning"):
        program = plan_joint_control(
            model, data, ids, reference, config.ik, config.control
        )
        planned_control = np.zeros((len(program.timestamps_s), model.nu), dtype=np.float64)
        planned_control[:, list(ids.arm_actuator_ids)] = program.arm_qpos_targets
        planned_control[:, ids.gripper_actuator_id] = program.gripper_ctrl
        control_payload = {
            "timestamps_s": np.asarray(program.timestamps_s, dtype=np.float64),
            "control": planned_control,
            "phase": np.asarray(program.phase, dtype="<U16"),
        }
    with _pipeline_stage("simulation"):
        simulation = run_joint_control_program(
            model,
            data,
            ids,
            program,
            ik=config.ik,
            control_config=config.control,
            collision_config=config.collision,
            render_every=config.simulation.render_every,
            render_size=config.simulation.render_size,
            render=not no_render,
        )
        physics = validate_rollout(simulation, config.collision)
        simulation_payload = _simulation_artifact_arrays(simulation)
    if not no_render:
        with _pipeline_stage("visualization"):
            replay_fps = 1.0 / (
                float(model.opt.timestep) * config.simulation.render_every
            )
            _write_rgb_video(
                destination / "mujoco_replay.mp4",
                simulation.rendered_rgb,
                replay_fps,
                labels_for_metrics({"status": "rejected", "variant": "B0"}),
            )
            generated_media_ffprobe["mujoco_replay.mp4"] = _validate_mp4(
                destination / "mujoco_replay.mp4"
            )
    _atomic_json(destination / "provenance.json", provenance)

    try:
        if not physics.passed:
            raise PhysicsRolloutFailure(physics.failed_checks)
        failed_checks: tuple[str, ...] = ()
    except PhysicsRolloutFailure as failure:
        failed_checks = failure.failed_checks
    action = _action_gate("B0", physics)
    reason = action["action_export_reason"]
    phase = np.asarray(simulation.phase)
    close_contact_duration = _longest_rollout_duration(
        simulation.timestamps_s,
        simulation.bilateral_contact & (phase == "close"),
    )
    lift_contact_duration = _longest_rollout_duration(
        simulation.timestamps_s,
        simulation.bilateral_contact & (phase == "lift"),
    )
    settle_duration = _longest_rollout_duration(
        simulation.timestamps_s,
        simulation.box_support_contact & (phase == "settle"),
    )
    metrics = {
        "status": RunStatus.REJECTED.value,
        "source_sha256": config.source.sha256,
        "source_accessed": False,
        "variant": "B0",
        "simulation_mode": simulation.mode,
        "placed_successfully": simulation.placed_successfully,
        "execution_tracking_ratio": simulation.execution_tracking_ratio,
        "bilateral_close_contact_duration_s": close_contact_duration,
        "bilateral_lift_contact_duration_s": lift_contact_duration,
        "maximum_lift_m": simulation.maximum_lift_m,
        "target_error_m": simulation.target_error_m,
        "settle_duration_s": settle_duration,
        "final_tilt_rad": float(simulation.can_tilt_rad[-1]),
        "final_linear_speed_m_s": float(simulation.can_linear_speed_m_s[-1]),
        "forbidden_contact_count": physics.forbidden_contact_count,
        "maximum_forbidden_penetration_m": physics.maximum_forbidden_penetration_m,
        "joint_position_violation_count": int(
            np.count_nonzero(simulation.joint_position_violation)
        ),
        "joint_velocity_violation_count": int(
            np.count_nonzero(simulation.joint_velocity_violation)
        ),
        "joint_acceleration_violation_count": int(
            np.count_nonzero(simulation.joint_acceleration_violation)
        ),
        "invalid_numerical_state": simulation.invalid_numerical_state,
        "physics_failed_checks": list(failed_checks),
        "rejection_stage": "simulation",
        "reason": reason,
        "runtime_s": time.perf_counter() - started,
        **action,
    }
    with _pipeline_stage("artifact_serialization"):
        _write_terminal_npz_artifacts(
            destination,
            (
                ("robot_reference.npz", reference_payload, ROBOT_REFERENCE_V1),
                (
                    "baseline_control_trace.npz",
                    control_payload,
                    baseline_control_contract(model.nu),
                ),
                (
                    "simulation.npz",
                    simulation_payload,
                    simulation_contract(model.nq, model.nv, model.nu),
                ),
            ),
            provenance,
            metrics,
            source_sha256="not_used",
        )
    _atomic_json(
        destination / "rejection.json",
        {
            "stage": "simulation",
            "reason": reason,
            "physics_failed_checks": list(failed_checks),
            **action,
        },
    )
    _atomic_json(destination / "metrics.json", metrics)
    return metrics


def _execute_run(
    config_file: Path,
    destination: Path,
    *,
    variant: Variant,
    no_render: bool,
    started: float,
) -> dict[str, Any]:
    with _pipeline_stage("config"):
        try:
            config = load_experiment_config(config_file)
            config_sha256 = sha256_file(config_file)
            np.random.seed(config.random_seed)
        except Exception as error:
            raise ValueError(_redact_local_paths(str(error), config_file)) from error

    if variant == "B0":
        return _execute_manual_b0(
            config,
            config_sha256,
            destination,
            no_render=no_render,
            started=started,
        )

    if variant in ("B2", "B3", "B4"):
        with _pipeline_stage("model_validation"):
            provenance, _ = _variant_provenance(
                config, config_sha256, variant, model_path=DEFAULT_SCENE_PATH
            )
            _atomic_json(destination / "provenance.json", provenance)
        metrics = {
            "status": RunStatus.NOT_RUN.value,
            "variant": variant,
            "reason": "metric_depth_not_available",
            "source_sha256": config.source.sha256,
            "source_accessed": False,
            "runtime_s": time.perf_counter() - started,
            **_action_gate(variant, None),
        }
        with _pipeline_stage("metrics"):
            _atomic_json(destination / "metrics.json", metrics)
        return metrics

    with _pipeline_stage("source_probe"):
        source = config.source.path
        try:
            measured_sha256 = sha256_file(source)
            if measured_sha256.lower() != config.source.sha256:
                raise ValueError(
                    f"source SHA-256 mismatch: expected {config.source.sha256}, "
                    f"got {measured_sha256}"
                )
            metadata = probe_video(source)
            source_ffprobe = _ffprobe_video_facts(source)
            if not np.isclose(metadata.fps, config.source.fps, atol=0.05):
                raise ValueError(
                    f"source FPS mismatch: expected {config.source.fps}, got {metadata.fps}"
                )
        except Exception as error:
            raise ValueError(_redact_local_paths(str(error), source)) from error

    with _pipeline_stage("provenance"):
        generated_media_ffprobe: dict[str, dict[str, Any]] = {}
        provenance = {
            "experiment_id": config.experiment_id,
            "generator": _git_generator_provenance(),
            "config": {
                "sha256": config_sha256,
                "resolved": to_public_resolved_mapping(config),
            },
            "source": {"id": config.source.id, "sha256": measured_sha256},
            "source_sha256": measured_sha256,
            "variant": variant,
            "random_seed": config.random_seed,
            "source_metadata": asdict(metadata),
            "model": {
                "id": "primitive_7dof_panda_like_diagnostic",
                "sha256": pinned_model_identity("B1"),
                "description": "primitive_7dof_panda_like_diagnostic",
                "collision_validation": "not_applicable_kinematic",
            },
            "runtime": _runtime_provenance(),
            "ffprobe": {
                "source": source_ffprobe,
                "generated_media": generated_media_ffprobe,
            },
        }
        _atomic_json(
            destination / "provenance.json",
            provenance,
        )

    with _pipeline_stage("tracking"):
        trajectory = track_roi_lk(
            source,
            config.source.roi_xywh,
            forward_backward_threshold_px=config.tracking.forward_backward_threshold_px,
            minimum_live_points=config.tracking.minimum_live_points,
        )
        trajectory_payload = {
            "timestamps_s": np.asarray(trajectory.timestamps_s, dtype=np.float64),
            "centers_px": np.asarray(trajectory.centers_px, dtype=np.float64),
            "confidence": np.asarray(trajectory.confidence, dtype=np.float64),
        }
        lk_point_availability_ratio = float(np.mean(trajectory.confidence > 0.0))
    if lk_point_availability_ratio < config.tracking.minimum_valid_ratio:
        reason = "lk_point_availability_ratio_below_minimum"
        _atomic_json(
            destination / "rejection.json",
            {
                "stage": "tracking",
                "reason": reason,
                "measured_lk_point_availability_ratio": lk_point_availability_ratio,
                "minimum_lk_point_availability_ratio": config.tracking.minimum_valid_ratio,
            },
        )
        if not no_render:
            with _pipeline_stage("visualization"):
                render_tracking_overlay(
                    source,
                    trajectory,
                    (),
                    destination / "tracking_overlay.mp4",
                    roi_size=config.source.roi_xywh[2:],
                    media_labels=labels_for_metrics(
                        {"status": "rejected", "variant": "B1"}
                    ),
                )
                generated_media_ffprobe["tracking_overlay.mp4"] = _validate_mp4(
                    destination / "tracking_overlay.mp4"
                )
        _atomic_json(destination / "provenance.json", provenance)
        metrics = _rejected_metrics(
            variant=variant,
            source_sha256=measured_sha256,
            lk_point_availability_ratio=lk_point_availability_ratio,
            stage="tracking",
            reason=reason,
            started=started,
        )
        with _pipeline_stage("artifact_serialization"):
            _write_terminal_npz_artifacts(
                destination,
                (("trajectory_2d.npz", trajectory_payload, TRAJECTORY_2D_V1),),
                provenance,
                metrics,
                source_sha256=measured_sha256,
            )
        _atomic_json(destination / "metrics.json", metrics)
        return metrics

    with _pipeline_stage("phase_inference"):
        phases = infer_motion_phases(trajectory, metadata.fps)
        _atomic_json(
            destination / "phases.json",
            [{**asdict(phase), "evidence": list(phase.evidence)} for phase in phases],
        )
    if not no_render:
        with _pipeline_stage("visualization"):
            diagnostic_labels = labels_for_metrics(
                {"status": "rejected", "variant": "B1"}
            )
            render_tracking_overlay(
                source,
                trajectory,
                phases,
                destination / "tracking_overlay.mp4",
                roi_size=config.source.roi_xywh[2:],
                media_labels=diagnostic_labels,
            )
            generated_media_ffprobe["tracking_overlay.mp4"] = _validate_mp4(
                destination / "tracking_overlay.mp4"
            )

    with _pipeline_stage("retargeting"):
        retargeting_variant = "B0" if variant == "B0" else "B1"
        reference = build_pick_place_reference(
            trajectory,
            phases,
            retargeting_variant,
            image_size=(metadata.width, metadata.height),
            x_bounds=config.scene.x_bounds_m,
            y_bounds=config.scene.y_bounds_m,
            b0_start_m=config.scene.b0_start_m,
            b0_goal_m=config.scene.b0_goal_m,
        )
        reference_payload = {
            "timestamps_s": np.asarray(reference.timestamps_s, dtype=np.float64),
            "ee_positions": np.asarray(reference.ee_positions, dtype=np.float64),
            "quaternion_wxyz": np.asarray(
                reference.quaternion_wxyz, dtype=np.float64
            ),
            "gripper_width": np.asarray(reference.gripper_width, dtype=np.float64),
            "phase": np.asarray(reference.phase, dtype="<U16"),
        }

    with _pipeline_stage("simulation"):
        simulation_mode = (
            config.simulation.b0_mode
            if variant == "B0"
            else config.simulation.b1_mode
        )
        simulation = run_mujoco_replay(
            reference,
            mode=simulation_mode,
            render_every=config.simulation.render_every,
            render_size=config.simulation.render_size,
            render=not no_render,
        )
        simulation_payload = _simulation_artifact_arrays(simulation)
    if not no_render:
        with _pipeline_stage("visualization"):
            diagnostic_labels = labels_for_metrics(
                {"status": "rejected", "variant": "B1"}
            )
            _render_trajectory_plot(
                trajectory,
                phases,
                destination / "trajectory_2d.png",
                diagnostic_labels,
            )
            replay_fps = 1.0 / (0.002 * config.simulation.render_every)
            _write_rgb_video(
                destination / "mujoco_replay.mp4",
                simulation.rendered_rgb,
                replay_fps,
                diagnostic_labels,
            )
            generated_media_ffprobe["mujoco_replay.mp4"] = _validate_mp4(
                destination / "mujoco_replay.mp4"
            )
            simulation_duration_s = float(simulation.timestamps_s[-1])
            render_comparison_video(
                source,
                destination / "tracking_overlay.mp4",
                simulation.rendered_rgb,
                destination / "side_by_side.mp4",
                config.media,
                {"status": "rejected", "variant": "B1"},
                metadata.duration_s,
                simulation_duration_s,
            )
            generated_media_ffprobe["side_by_side.mp4"] = _validate_mp4(
                destination / "side_by_side.mp4"
            )
            _render_contact_sheet(
                source,
                destination / "tracking_overlay.mp4",
                simulation.rendered_rgb,
                phases,
                metadata.frame_count,
                destination / "contact_sheet.png",
                config.media.panel_size,
                config.media.letterbox_bgr,
                diagnostic_labels,
                metadata.duration_s,
                simulation_duration_s,
            )
    _atomic_json(destination / "provenance.json", provenance)

    perception_warnings = [
        f"zero_confidence_phase:{phase.label}"
        for phase in phases
        if phase.confidence == 0.0
    ]
    metrics: dict[str, Any] = {
        "status": RunStatus.REJECTED.value,
        "source_sha256": measured_sha256,
        "lk_point_availability_ratio": lk_point_availability_ratio,
        "lk_metric_scope": "point_availability_not_semantic_accuracy",
        "semantic_accuracy_status": "not_measured",
        "phase_count": len(phases),
        "variant": variant,
        "simulation_mode": simulation.mode,
        "placed_successfully": simulation.placed_successfully,
        "reachability_ratio": simulation.reachability_ratio,
        "maximum_lift_m": simulation.maximum_lift_m,
        "maximum_can_height_gain_m": simulation.maximum_can_height_gain_m,
        "target_error_m": simulation.target_error_m,
        "target_height_error_m": simulation.target_height_error_m,
        "grasp_contact_frames": int(np.count_nonzero(simulation.grasp_contact)),
        "support_contact_duration_s": simulation.support_contact_duration_s,
        "final_support_contact": simulation.final_support_contact,
        "finishes_inside_box": simulation.finishes_inside_box,
        "invalid_numerical_state": simulation.invalid_numerical_state,
        "perception_status": "degraded" if perception_warnings else "ok",
        "perception_warnings": perception_warnings,
        "runtime_s": time.perf_counter() - started,
        **_action_gate("B1", None),
    }
    rejection_reason = "kinematic_replay_not_action"
    metrics["rejection_stage"] = "simulation"
    metrics["reason"] = rejection_reason
    _atomic_json(
        destination / "rejection.json",
        {
            "stage": "simulation",
            "reason": rejection_reason,
            "simulation_mode": simulation.mode,
            "placed_successfully": simulation.placed_successfully,
            "reachability_ratio": simulation.reachability_ratio,
            "required_reachability_ratio": 0.95,
            **_action_gate("B1", None),
        },
    )
    with _pipeline_stage("artifact_serialization"):
        pinned_nq, pinned_nv, pinned_nu = pinned_model_dimensions(
            "B1", provenance["model"]["sha256"]
        )
        _write_terminal_npz_artifacts(
            destination,
            (
                ("trajectory_2d.npz", trajectory_payload, TRAJECTORY_2D_V1),
                ("robot_reference.npz", reference_payload, ROBOT_REFERENCE_V1),
                (
                    "simulation.npz",
                    simulation_payload,
                    simulation_contract(pinned_nq, pinned_nv, pinned_nu),
                ),
            ),
            provenance,
            metrics,
            source_sha256=measured_sha256,
        )
    with _pipeline_stage("metrics"):
        _atomic_json(destination / "metrics.json", metrics)
    return metrics


def _run_locked_experiment(
    config_file: Path,
    destination: Path,
    expected: _RunSnapshot | None,
    *,
    variant: Variant,
    no_render: bool,
    started: float,
) -> dict[str, Any]:
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        )
    )
    sensitive_paths = (config_file, destination, staging)
    try:
        sensitive_paths += (load_experiment_config(config_file).source.path,)
    except Exception:
        pass
    try:
        try:
            metrics = _execute_run(
                config_file,
                staging,
                variant=variant,
                no_render=no_render,
                started=started,
            )
            _validate_required_run_files(staging, metrics, no_render=no_render)
        except _PipelineFailure as failure:
            for artifact_pattern in ("*.npz", "*.schema.json"):
                for partial_artifact in staging.glob(artifact_pattern):
                    partial_artifact.unlink()
            for media_pattern in ("*.mp4", "*.png"):
                for partial_media in staging.glob(media_pattern):
                    partial_media.unlink()
            metrics = {
                "status": RunStatus.FAILED.value,
                "variant": variant,
                "reason": "stage_exception",
                "failure_stage": failure.stage,
                "error_type": type(failure.cause).__name__,
                "error_message": _redact_local_paths(
                    str(failure.cause), *sensitive_paths
                ),
                "runtime_s": time.perf_counter() - started,
                **_action_gate(variant, None),
            }
            rejection = {
                "stage": failure.stage,
                "reason": "stage_exception",
                "error_type": type(failure.cause).__name__,
                "error_message": _redact_local_paths(
                    str(failure.cause), *sensitive_paths
                ),
                **_action_gate(variant, None),
            }
            _atomic_json(staging / "rejection.json", rejection)
            _atomic_json(staging / "metrics.json", metrics)
            _validate_required_run_files(staging, metrics, no_render=no_render)
        _write_run_manifest(staging, metrics, config_file=config_file)
        return _publish_staging(
            staging,
            destination,
            expected,
            metrics,
            variant=variant,
            started=started,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def run_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    variant: Variant = "B1",
    no_render: bool = False,
) -> dict[str, Any]:
    """Build one run in isolation, then atomically publish its complete directory."""

    if variant not in ("B0", "B1", "B2", "B3", "B4"):
        raise ValueError("variant must be B0, B1, B2, B3, or B4")
    destination = _output_candidate(output_dir)
    config_file = Path(config_path).resolve()
    started = time.perf_counter()
    with _serialized_output(destination):
        destination, expected = _validated_output_target(destination)
        return _run_locked_experiment(
            config_file,
            destination,
            expected,
            variant=variant,
            no_render=no_render,
            started=started,
        )

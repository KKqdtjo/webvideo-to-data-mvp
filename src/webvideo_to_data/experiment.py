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
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .contact import infer_motion_phases
from .media import probe_video, sha256_file
from .retargeting import build_pick_place_reference
from .schema import RunStatus
from .simulation import _DEFAULT_MODEL_PATH, run_mujoco_replay
from .tracking import track_roi_lk
from .visualization import render_tracking_overlay


Variant = Literal["B0", "B1", "B2", "B3", "B4"]
_RUN_MANIFEST_PRODUCER = "webvideo_to_data.experiment"
_RUN_MANIFEST_VERSION = 3
_COLLISION_BLOCK_REASON = "collision_validation_not_implemented"
_ACTION_EXPORT_BLOCK = {
    "collision_validation": "not_implemented",
    "action_export_eligible": False,
    "action_export_reason": _COLLISION_BLOCK_REASON,
    "action_exported": False,
}
_GENERATED_RUN_FILES = {
    "actions.npz",
    "contact_sheet.png",
    "metrics.json",
    "mujoco_replay.mp4",
    "phases.json",
    "provenance.json",
    "rejection.json",
    "robot_reference.npz",
    "run_manifest.json",
    "side_by_side.mp4",
    "simulation.npz",
    "tracking_overlay.mp4",
    "trajectory_2d.npz",
    "trajectory_2d.png",
}
_OUTPUT_LOCKS_GUARD = threading.Lock()
_OUTPUT_LOCKS: dict[str, threading.Lock] = {}


@dataclass(frozen=True)
class _RunSnapshot:
    directory_identity: tuple[int, int]
    files: tuple[tuple[str, int, str], ...]


class _PipelineFailure(Exception):
    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


@contextmanager
def _pipeline_stage(name: str) -> Any:
    try:
        yield
    except _PipelineFailure:
        raise
    except Exception as error:
        raise _PipelineFailure(name, error) from error


@dataclass(frozen=True)
class SourceConfig:
    path: str
    sha256: str
    fps: float
    roi_xywh: tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackingConfig:
    forward_backward_threshold_px: float
    minimum_live_points: int
    minimum_valid_ratio: float


@dataclass(frozen=True)
class SceneConfig:
    x_bounds_m: tuple[float, float]
    y_bounds_m: tuple[float, float]
    b0_start_m: tuple[float, float, float]
    b0_goal_m: tuple[float, float, float]


@dataclass(frozen=True)
class SimulationConfig:
    b0_mode: str
    b1_mode: str
    render_size: tuple[int, int]
    render_every: int


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    source: SourceConfig
    tracking: TrackingConfig
    scene: SceneConfig
    simulation: SimulationConfig
    random_seed: int


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """Parse a YAML experiment description into explicit immutable contracts."""

    path = Path(config_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("experiment config must be a mapping")
    source = document["source"]
    tracking = document["tracking"]
    scene = document["scene"]
    simulation = document["simulation"]
    return ExperimentConfig(
        experiment_id=str(document["experiment_id"]),
        source=SourceConfig(
            path=str(source["path"]),
            sha256=str(source["sha256"]).lower(),
            fps=float(source["fps"]),
            roi_xywh=tuple(float(value) for value in source["roi_xywh"]),
        ),
        tracking=TrackingConfig(
            forward_backward_threshold_px=float(
                tracking["forward_backward_threshold_px"]
            ),
            minimum_live_points=int(tracking["minimum_live_points"]),
            minimum_valid_ratio=float(tracking["minimum_valid_ratio"]),
        ),
        scene=SceneConfig(
            x_bounds_m=tuple(float(value) for value in scene["x_bounds_m"]),
            y_bounds_m=tuple(float(value) for value in scene["y_bounds_m"]),
            b0_start_m=tuple(float(value) for value in scene["b0_start_m"]),
            b0_goal_m=tuple(float(value) for value in scene["b0_goal_m"]),
        ),
        simulation=SimulationConfig(
            b0_mode=str(simulation["b0_mode"]),
            b1_mode=str(simulation["b1_mode"]),
            render_size=tuple(int(value) for value in simulation["render_size"]),
            render_every=int(simulation["render_every"]),
        ),
        random_seed=int(document["random_seed"]),
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as temporary:
        json.dump(payload, temporary, ensure_ascii=False, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _resolve_source(config_path: Path, configured_path: str) -> Path:
    source = Path(configured_path)
    if source.is_absolute():
        return source
    return (config_path.parent / source).resolve()


def _write_rgb_video(path: Path, frames: np.ndarray, fps: float) -> None:
    if len(frames) == 0:
        raise ValueError("cannot write a video without frames")
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise ValueError(f"video writer cannot open {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
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


def _render_trajectory_plot(trajectory: Any, phases: Any, output: Path) -> None:
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
    for phase in phases:
        boundary_s = trajectory.timestamps_s[phase.start_frame]
        for axis in axes:
            axis.axvline(boundary_s, color="0.5", linewidth=0.8, alpha=0.7)
        axes[0].text(boundary_s, axes[0].get_ylim()[1], phase.label, rotation=90, va="top")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def _render_side_by_side(
    source_path: Path,
    overlay_path: Path,
    simulation_frames_rgb: np.ndarray,
    output_path: Path,
    tile_size: tuple[int, int],
) -> None:
    source_capture = cv2.VideoCapture(str(source_path))
    overlay_capture = cv2.VideoCapture(str(overlay_path))
    writer: cv2.VideoWriter | None = None
    try:
        if not source_capture.isOpened() or not overlay_capture.isOpened():
            raise ValueError("source and overlay videos must be readable")
        source_count = int(source_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(source_capture.get(cv2.CAP_PROP_FPS))
        width, height = tile_size
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width * 3, height),
        )
        if not writer.isOpened():
            raise ValueError("side-by-side video writer cannot be opened")
        for index in range(source_count):
            source_ok, source_frame = source_capture.read()
            overlay_ok, overlay_frame = overlay_capture.read()
            if not source_ok or not overlay_ok:
                raise ValueError("source or overlay ended before declared frame count")
            simulation_index = min(
                len(simulation_frames_rgb) - 1,
                round(index / max(1, source_count - 1) * (len(simulation_frames_rgb) - 1)),
            )
            simulation_frame = cv2.cvtColor(
                simulation_frames_rgb[simulation_index], cv2.COLOR_RGB2BGR
            )
            panels = [
                cv2.resize(source_frame, (width, height)),
                cv2.resize(overlay_frame, (width, height)),
                cv2.resize(simulation_frame, (width, height)),
            ]
            for panel, label in zip(panels, ("source", "tracking", "MuJoCo")):
                cv2.putText(
                    panel, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA
                )
            writer.write(np.hstack(panels))
    finally:
        source_capture.release()
        overlay_capture.release()
        if writer is not None:
            writer.release()


def _read_frame(capture: cv2.VideoCapture, index: int, size: tuple[int, int]) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"cannot read checkpoint frame {index}")
    return cv2.resize(frame, size)


def _render_contact_sheet(
    source_path: Path,
    overlay_path: Path,
    simulation_frames_rgb: np.ndarray,
    phases: Any,
    frame_count: int,
    output_path: Path,
    tile_size: tuple[int, int],
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
        rows.append(np.hstack([_read_frame(source_capture, index, tile_size) for index in checkpoints]))
        rows.append(np.hstack([_read_frame(overlay_capture, index, tile_size) for index in checkpoints]))
        simulation_panels = []
        for index in checkpoints:
            sim_index = min(
                len(simulation_frames_rgb) - 1,
                round(index / max(1, frame_count - 1) * (len(simulation_frames_rgb) - 1)),
            )
            frame = cv2.cvtColor(simulation_frames_rgb[sim_index], cv2.COLOR_RGB2BGR)
            simulation_panels.append(cv2.resize(frame, tile_size))
        rows.append(np.hstack(simulation_panels))
        sheet = np.vstack(rows)
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
    return {
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
        "reason": reason,
        "runtime_s": time.perf_counter() - started,
        **_ACTION_EXPORT_BLOCK,
    }


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


def _snapshot_files(directory: Path) -> tuple[tuple[str, int, str], ...]:
    signatures: list[tuple[str, int, str]] = []
    for child in sorted(directory.iterdir(), key=lambda item: item.name):
        if child.is_symlink() or not child.is_file():
            raise ValueError(f"unrecognized files in generated run: {child.name}")
        signatures.append((child.name, child.stat().st_size, sha256_file(child)))
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
    status = metrics.get("status")
    if status not in {member.value for member in RunStatus}:
        return "unknown terminal status"
    if metrics.get("variant") not in ("B0", "B1", "B2", "B3", "B4"):
        return "unknown experiment variant"
    has_action = "actions.npz" in file_names
    reason = metrics.get("reason")
    if status == RunStatus.COMPLETED.value:
        if not (
            has_action
            and reason is None
            and metrics.get("action_export_eligible") is True
            and metrics.get("action_exported") is True
            and metrics.get("collision_validation") == "passed"
            and metrics.get("simulation_mode") == "physics_grasp"
            and metrics.get("placed_successfully") is True
            and float(metrics.get("reachability_ratio", 0.0)) >= 0.95
        ):
            return "completed status lacks validated action eligibility"
    else:
        if has_action:
            return "non-completed status contains an action"
        if not isinstance(reason, str) or not reason.strip():
            return "non-completed status lacks a reason"
        if metrics.get("action_export_eligible") is not False:
            return "non-completed status lacks explicit action ineligibility"
        if metrics.get("action_exported") is not False:
            return "non-completed status must record action_exported=false"
    if status == RunStatus.REJECTED.value:
        if "rejection.json" not in file_names or not metrics.get("rejection_stage"):
            return "rejected status lacks rejection diagnostics"
    elif status == RunStatus.FAILED.value:
        if "rejection.json" not in file_names or not metrics.get("failure_stage"):
            return "failed status lacks failure diagnostics"
    elif status == RunStatus.NOT_RUN.value and "rejection.json" in file_names:
        return "not_run status must not contain rejection diagnostics"
    return None


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
    manifest_path = children.get("run_manifest.json")
    if manifest_path is None:
        raise ValueError("output target lacks a trusted generated-run marker")
    try:
        manifest = _json_object(manifest_path)
        metrics = _json_object(children["metrics.json"])
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("output target lacks a trusted generated-run marker") from error
    actual_names = set(children) - {"run_manifest.json"}
    version = manifest.get("format_version")
    if version == _RUN_MANIFEST_VERSION:
        manifest_files = manifest.get("files")
        if (
            manifest.get("producer") != _RUN_MANIFEST_PRODUCER
            or not isinstance(manifest_files, dict)
            or set(manifest_files) != actual_names
        ):
            raise ValueError("output target lacks a trusted generated-run marker")
        for name, recorded in manifest_files.items():
            child = children[name]
            if (
                not isinstance(recorded, dict)
                or recorded.get("size") != child.stat().st_size
                or recorded.get("sha256") != sha256_file(child)
            ):
                raise ValueError("output target lacks a trusted generated-run marker")
    else:
        raise ValueError("output target lacks a trusted generated-run marker")
    if (
        manifest.get("status") != metrics.get("status")
        or manifest.get("variant") != metrics.get("variant")
    ):
        raise ValueError("output target lacks a trusted generated-run marker")
    if _terminal_metrics_error(metrics, actual_names) is not None:
        raise ValueError("output target lacks a trusted generated-run marker")
    return _directory_snapshot(directory)


def _validated_output_target(
    destination: Path,
) -> tuple[Path, _RunSnapshot | None]:
    if destination.exists():
        return destination, _trusted_run_snapshot(destination)
    return destination, None


def _write_run_manifest(staging: Path, metrics: dict[str, Any]) -> None:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(staging.iterdir(), key=lambda item: item.name):
        if path.name == "run_manifest.json":
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"cannot manifest non-file artifact: {path.name}")
        files[path.name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    _atomic_json(
        staging / "run_manifest.json",
        {
            "producer": _RUN_MANIFEST_PRODUCER,
            "format_version": _RUN_MANIFEST_VERSION,
            "variant": metrics["variant"],
            "status": metrics["status"],
            "files": files,
        },
    )


def _publication_failure_metrics(
    *, variant: Variant, stage: str, error: Exception, started: float
) -> dict[str, Any]:
    return {
        "status": RunStatus.FAILED.value,
        "variant": variant,
        "reason": "publication_exception",
        "failure_stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "runtime_s": time.perf_counter() - started,
        **_ACTION_EXPORT_BLOCK,
    }


def _mark_canonical_publication_failure(
    destination: Path,
    *,
    expected: _RunSnapshot,
    variant: Variant,
    stage: str,
    error: Exception,
    started: float,
) -> dict[str, Any]:
    try:
        current = _directory_snapshot(destination)
    except ValueError as validation_error:
        raise ValueError("canonical output changed before failure reporting") from validation_error
    if current != expected:
        raise ValueError("canonical output changed before failure reporting")
    working = destination.parent / (
        f".{destination.name}.failure-working-{uuid4().hex}"
    )
    if (
        working.parent != destination.parent
        or not working.name.startswith(f".{destination.name}.failure-working-")
        or working.exists()
    ):
        raise ValueError("unsafe publication failure working target")
    try:
        destination.replace(working)
    except Exception as isolation_error:
        if not working.exists():
            raise ValueError(
                "canonical output could not be isolated for failure reporting"
            ) from isolation_error
    try:
        moved = _directory_snapshot(working)
    except ValueError as validation_error:
        if not destination.exists() and working.exists():
            try:
                working.rename(destination)
            except Exception:
                pass
        preserved = working if working.exists() else destination
        raise ValueError(
            "canonical output changed before failure reporting; "
            f"candidate preserved at {preserved}"
        ) from validation_error
    if moved != expected:
        if not destination.exists():
            try:
                working.rename(destination)
            except Exception:
                pass
        preserved = working if working.exists() else destination
        raise ValueError(
            "canonical output changed before failure reporting; "
            f"candidate preserved at {preserved}"
        )
    actions = working / "actions.npz"
    if actions.exists():
        if actions.is_symlink() or not actions.is_file():
            raise ValueError("canonical action artifact is not a regular file")
        actions.unlink()
    metrics = _publication_failure_metrics(
        variant=variant, stage=stage, error=error, started=started
    )
    _atomic_json(
        working / "rejection.json",
        {
            "stage": stage,
            "reason": "publication_exception",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "action_exported": False,
        },
    )
    _atomic_json(working / "metrics.json", metrics)
    _write_run_manifest(working, metrics)
    failed = _trusted_run_snapshot(working)
    if destination.exists():
        return metrics
    try:
        working.rename(destination)
    except Exception as restore_error:
        if not working.exists() and destination.exists():
            try:
                if _trusted_run_snapshot(destination) == failed:
                    return metrics
            except ValueError:
                pass
            raise ValueError(
                "canonical output changed after failure reporting"
            ) from restore_error
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
    expected_files = {name: (size, digest) for name, size, digest in expected.files}
    for child in sorted(backup.iterdir(), key=lambda item: item.name):
        signature = expected_files.get(child.name)
        if (
            signature is None
            or child.is_symlink()
            or not child.is_file()
            or (child.stat().st_size, sha256_file(child)) != signature
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
                )
            return _mark_canonical_publication_failure(
                destination,
                expected=staged,
                variant=variant,
                stage="publication_swap",
                error=error,
                started=started,
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
            )
        return _mark_canonical_publication_failure(
            destination,
            expected=expected,
            variant=variant,
            stage="publication_swap",
            error=error,
            started=started,
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
    if metrics["variant"] in ("B0", "B1") and metrics["status"] != "failed":
        tracking_rejection = (
            metrics["status"] == "rejected"
            and metrics.get("rejection_stage") == "tracking"
        )
        required.add("trajectory_2d.npz")
        if tracking_rejection:
            required.add("rejection.json")
            if not no_render:
                required.add("tracking_overlay.mp4")
        else:
            required.update(
                {"phases.json", "robot_reference.npz", "simulation.npz"}
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


def _execute_run(
    config_file: Path,
    destination: Path,
    *,
    variant: Variant,
    no_render: bool,
    started: float,
) -> dict[str, Any]:
    with _pipeline_stage("config"):
        config = load_experiment_config(config_file)
        config_sha256 = sha256_file(config_file)
        np.random.seed(config.random_seed)

    with _pipeline_stage("source_probe"):
        source = _resolve_source(config_file, config.source.path)
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

    with _pipeline_stage("provenance"):
        generated_media_ffprobe: dict[str, dict[str, Any]] = {}
        provenance = {
            "experiment_id": config.experiment_id,
            "generator": _git_generator_provenance(),
            "config": {
                "path": str(config_file),
                "sha256": config_sha256,
                "resolved": asdict(config),
            },
            "source_path": str(source),
            "source_sha256": measured_sha256,
            "variant": variant,
            "random_seed": config.random_seed,
            "source_metadata": asdict(metadata),
            "model": {
                "path": str(_DEFAULT_MODEL_PATH.resolve()),
                "sha256": sha256_file(_DEFAULT_MODEL_PATH),
                "description": "primitive_7dof_panda_like_diagnostic",
                "collision_validation": "not_implemented",
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

    if variant in ("B2", "B3", "B4"):
        metrics = {
            "status": RunStatus.NOT_RUN.value,
            "variant": variant,
            "reason": "metric_depth_not_available",
            "source_sha256": measured_sha256,
            "runtime_s": time.perf_counter() - started,
            **_ACTION_EXPORT_BLOCK,
        }
        with _pipeline_stage("metrics"):
            _atomic_json(destination / "metrics.json", metrics)
        return metrics

    with _pipeline_stage("tracking"):
        trajectory = track_roi_lk(
            source,
            config.source.roi_xywh,
            forward_backward_threshold_px=config.tracking.forward_backward_threshold_px,
            minimum_live_points=config.tracking.minimum_live_points,
        )
        np.savez(
            destination / "trajectory_2d.npz",
            timestamps_s=trajectory.timestamps_s,
            centers_px=trajectory.centers_px,
            confidence=trajectory.confidence,
        )
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
            render_tracking_overlay(
                source,
                trajectory,
                phases,
                destination / "tracking_overlay.mp4",
                roi_size=config.source.roi_xywh[2:],
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
            "timestamps_s": reference.timestamps_s,
            "ee_positions": reference.ee_positions,
            "quaternion_wxyz": reference.quaternion_wxyz,
            "gripper_width": reference.gripper_width,
            "phase": np.asarray(reference.phase),
        }
        np.savez(destination / "robot_reference.npz", **reference_payload)

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
        np.savez(
            destination / "simulation.npz",
            qpos=simulation.qpos,
            qvel=simulation.qvel,
            can_pose=simulation.can_pose,
            contact_count=simulation.contact_count,
            grasp_contact=simulation.grasp_contact,
            ik_position_error_m=simulation.ik_position_error_m,
            ik_orientation_error_rad=simulation.ik_orientation_error_rad,
            ik_converged=simulation.ik_converged,
        )
    if not no_render:
        with _pipeline_stage("visualization"):
            _render_trajectory_plot(
                trajectory, phases, destination / "trajectory_2d.png"
            )
            replay_fps = 1.0 / (0.002 * config.simulation.render_every)
            _write_rgb_video(
                destination / "mujoco_replay.mp4", simulation.rendered_rgb, replay_fps
            )
            generated_media_ffprobe["mujoco_replay.mp4"] = _validate_mp4(
                destination / "mujoco_replay.mp4"
            )
            _render_side_by_side(
                source,
                destination / "tracking_overlay.mp4",
                simulation.rendered_rgb,
                destination / "side_by_side.mp4",
                config.simulation.render_size,
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
                config.simulation.render_size,
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
        **_ACTION_EXPORT_BLOCK,
    }
    rejection_reason: str | None = None
    if simulation.mode == "kinematic_replay":
        rejection_reason = "kinematic_replay_not_action"
    elif not simulation.placed_successfully or simulation.reachability_ratio < 0.95:
        rejection_reason = "physics_validation_failed"
    else:
        rejection_reason = _COLLISION_BLOCK_REASON
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
            "collision_validation": "not_implemented",
            "action_export_eligible": False,
            "action_export_reason": _COLLISION_BLOCK_REASON,
            "action_exported": False,
        },
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
            (staging / "actions.npz").unlink(missing_ok=True)
            for media_pattern in ("*.mp4", "*.png"):
                for partial_media in staging.glob(media_pattern):
                    partial_media.unlink()
            metrics = {
                "status": RunStatus.FAILED.value,
                "variant": variant,
                "reason": "stage_exception",
                "failure_stage": failure.stage,
                "error_type": type(failure.cause).__name__,
                "error_message": str(failure.cause),
                "runtime_s": time.perf_counter() - started,
                **_ACTION_EXPORT_BLOCK,
            }
            rejection = {
                "stage": failure.stage,
                "reason": "stage_exception",
                "error_type": type(failure.cause).__name__,
                "error_message": str(failure.cause),
                "action_export_eligible": False,
                "action_export_reason": _COLLISION_BLOCK_REASON,
                "action_exported": False,
            }
            _atomic_json(staging / "rejection.json", rejection)
            _atomic_json(staging / "metrics.json", metrics)
            _validate_required_run_files(staging, metrics, no_render=no_render)
        _write_run_manifest(staging, metrics)
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

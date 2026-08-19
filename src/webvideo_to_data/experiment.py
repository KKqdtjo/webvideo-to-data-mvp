"""Auditable orchestration for the EXP-001 video-to-MuJoCo pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Literal

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .contact import infer_motion_phases
from .media import probe_video, sha256_file
from .retargeting import build_pick_place_reference
from .simulation import run_mujoco_replay
from .tracking import track_roi_lk
from .visualization import render_tracking_overlay


Variant = Literal["B0", "B1", "B2", "B3", "B4"]


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
    return (Path.cwd() / source).resolve()


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
    *, variant: Variant, source_sha256: str, valid_track_ratio: float, stage: str,
    reason: str, started: float
) -> dict[str, Any]:
    return {
        "status": "rejected",
        "source_sha256": source_sha256,
        "valid_track_ratio": valid_track_ratio,
        "phase_count": 0,
        "variant": variant,
        "simulation_mode": None,
        "placed_successfully": False,
        "rejection_stage": stage,
        "reason": reason,
        "runtime_s": time.perf_counter() - started,
    }


def run_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    variant: Variant = "B1",
    no_render: bool = False,
) -> dict[str, Any]:
    """Run one reproducible variant and return the same metrics written to disk."""

    if variant not in ("B0", "B1", "B2", "B3", "B4"):
        raise ValueError("variant must be B0, B1, B2, B3, or B4")
    config_file = Path(config_path).resolve()
    config = load_experiment_config(config_file)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    actions_path = destination / "actions.npz"
    actions_path.unlink(missing_ok=True)
    (destination / "rejection.json").unlink(missing_ok=True)
    started = time.perf_counter()
    np.random.seed(config.random_seed)

    source = _resolve_source(config_file, config.source.path)
    measured_sha256 = sha256_file(source)
    if measured_sha256.lower() != config.source.sha256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {config.source.sha256}, got {measured_sha256}"
        )
    metadata = probe_video(source)
    if not np.isclose(metadata.fps, config.source.fps, atol=0.05):
        raise ValueError(
            f"source FPS mismatch: expected {config.source.fps}, got {metadata.fps}"
        )

    _atomic_json(
        destination / "provenance.json",
        {
            "experiment_id": config.experiment_id,
            "config_path": str(config_file),
            "source_path": str(source),
            "source_sha256": measured_sha256,
            "variant": variant,
            "random_seed": config.random_seed,
            "source_metadata": asdict(metadata),
        },
    )

    if variant in ("B2", "B3", "B4"):
        metrics = {
            "status": "not_run",
            "variant": variant,
            "reason": "metric_depth_not_available",
            "source_sha256": measured_sha256,
            "runtime_s": time.perf_counter() - started,
        }
        _atomic_json(destination / "metrics.json", metrics)
        return metrics

    trajectory = track_roi_lk(
        source,
        config.source.roi_xywh,
        forward_backward_threshold_px=(
            config.tracking.forward_backward_threshold_px
        ),
        minimum_live_points=config.tracking.minimum_live_points,
    )
    np.savez(
        destination / "trajectory_2d.npz",
        timestamps_s=trajectory.timestamps_s,
        centers_px=trajectory.centers_px,
        confidence=trajectory.confidence,
    )
    valid_track_ratio = float(np.mean(trajectory.confidence > 0.0))
    if valid_track_ratio < config.tracking.minimum_valid_ratio:
        reason = "valid_track_ratio_below_minimum"
        rejection = {
            "stage": "tracking",
            "reason": reason,
            "measured_valid_track_ratio": valid_track_ratio,
            "minimum_valid_track_ratio": config.tracking.minimum_valid_ratio,
        }
        _atomic_json(destination / "rejection.json", rejection)
        if not no_render:
            render_tracking_overlay(
                source,
                trajectory,
                (),
                destination / "tracking_overlay.mp4",
                roi_size=config.source.roi_xywh[2:],
            )
        metrics = _rejected_metrics(
            variant=variant,
            source_sha256=measured_sha256,
            valid_track_ratio=valid_track_ratio,
            stage="tracking",
            reason=reason,
            started=started,
        )
        _atomic_json(destination / "metrics.json", metrics)
        return metrics
    phases = infer_motion_phases(trajectory, metadata.fps)
    _atomic_json(
        destination / "phases.json",
        [{**asdict(phase), "evidence": list(phase.evidence)} for phase in phases],
    )
    if not no_render:
        render_tracking_overlay(
            source,
            trajectory,
            phases,
            destination / "tracking_overlay.mp4",
            roi_size=config.source.roi_xywh[2:],
        )

    retargeting_variant = "B0" if variant == "B0" else "B1"
    reference = build_pick_place_reference(
        trajectory,
        phases,
        retargeting_variant,
        image_size=(metadata.width, metadata.height),
        x_bounds=config.scene.x_bounds_m,
        y_bounds=config.scene.y_bounds_m,
    )
    reference_payload = {
        "timestamps_s": reference.timestamps_s,
        "ee_positions": reference.ee_positions,
        "quaternion_wxyz": reference.quaternion_wxyz,
        "gripper_width": reference.gripper_width,
        "phase": np.asarray(reference.phase),
    }
    np.savez(
        destination / "robot_reference.npz",
        **reference_payload,
    )
    simulation_mode = (
        config.simulation.b0_mode if variant == "B0" else config.simulation.b1_mode
    )
    simulation = run_mujoco_replay(
        reference,
        mode=simulation_mode,
        render_every=config.simulation.render_every,
        render_size=config.simulation.render_size,
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
        _render_trajectory_plot(trajectory, phases, destination / "trajectory_2d.png")
        replay_fps = 1.0 / (0.002 * config.simulation.render_every)
        _write_rgb_video(
            destination / "mujoco_replay.mp4", simulation.rendered_rgb, replay_fps
        )
        _render_side_by_side(
            source,
            destination / "tracking_overlay.mp4",
            simulation.rendered_rgb,
            destination / "side_by_side.mp4",
            config.simulation.render_size,
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

    metrics: dict[str, Any] = {
        "status": "completed",
        "source_sha256": measured_sha256,
        "valid_track_ratio": valid_track_ratio,
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
        "runtime_s": time.perf_counter() - started,
    }
    rejection_reason: str | None = None
    if simulation.mode == "kinematic_replay":
        rejection_reason = "kinematic_replay_not_action"
    elif not simulation.placed_successfully or simulation.reachability_ratio < 0.95:
        rejection_reason = "physics_validation_failed"
    if rejection_reason is not None:
        metrics["status"] = "rejected"
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
                "action_exported": False,
            },
        )
    else:
        np.savez(actions_path, **reference_payload)
    _atomic_json(destination / "metrics.json", metrics)
    return metrics

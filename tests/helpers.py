from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import yaml

from webvideo_to_data.media import sha256_file


VALID_CONFIG = {
    "schema_version": 2,
    "experiment_id": "SYNTHETIC",
    "source": {
        "id": "synthetic-moving-object",
        "path": "moving.avi",
        "sha256": "set-by-writer",
        "fps": 10.0,
        "roi_xywh": [20, 20, 16, 20],
    },
    "tracking": {
        "forward_backward_threshold_px": 1.5,
        "minimum_live_points": 4,
        "minimum_valid_ratio": 0.5,
    },
    "scene": {
        "x_bounds_m": [-0.15, 0.15],
        "y_bounds_m": [0.35, 0.65],
        "b0_start_m": [0.12, 0.45, 0.045],
        "b0_goal_m": [-0.05, 0.55, 0.115],
        "grasp_quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
    },
    "ik": {
        "position_tolerance_m": 0.005,
        "orientation_tolerance_deg": 5.0,
        "maximum_iterations": 200,
        "damping": 0.05,
        "step_size": 0.2,
        "orientation_weight": 0.25,
        "joint_limit_weight": 0.05,
    },
    "control": {
        "control_hz": 100.0,
        "maximum_joint_velocity_rad_s": 2.0,
        "maximum_joint_acceleration_rad_s2": 8.0,
        "gripper_open_width_m": 0.08,
        "gripper_closed_width_m": 0.0,
        "phase_duration_s": {
            "home": 0.5, "pregrasp": 0.8, "approach": 0.6, "close": 0.8,
            "lift": 0.8, "transport": 1.2, "lower": 0.6, "open": 0.6,
            "retreat": 0.6, "settle": 1.0,
        },
    },
    "collision": {
        "maximum_penetration_m": 0.002,
        "minimum_lift_m": 0.05,
        "maximum_target_error_m": 0.04,
        "settle_duration_s": 1.0,
        "maximum_final_tilt_deg": 15.0,
        "maximum_final_linear_speed_m_s": 0.02,
        "minimum_bilateral_contact_duration_s": 0.2,
        "minimum_lift_contact_duration_s": 0.1,
        "allowed_contact_pairs": {
            "home": [["can", "table"]],
            "pregrasp": [["can", "table"]],
            "approach": [["can", "table"]],
            "close": [["can", "table"], ["left_finger", "can"], ["right_finger", "can"]],
            "lift": [["left_finger", "can"], ["right_finger", "can"]],
            "transport": [["left_finger", "can"], ["right_finger", "can"]],
            "lower": [["left_finger", "can"], ["right_finger", "can"]],
            "open": [["can", "box"]],
            "retreat": [["can", "box"]],
            "settle": [["can", "box"]],
        },
    },
    "perturbation": {
        "rollout_count": 30,
        "xy_half_range_m": 0.01,
        "yaw_half_range_deg": 5.0,
        "mass_fraction": 0.10,
        "friction_fraction": 0.10,
    },
    "simulation": {
        "b0_mode": "physics_grasp",
        "b1_mode": "kinematic_replay",
        "render_size": [320, 320],
        "render_every": 50,
    },
    "media": {
        "canvas_size": [960, 320],
        "panel_size": [320, 320],
        "letterbox_bgr": [16, 16, 16],
        "output_fps": 30.0,
        "comparison_alignment": "time_warped",
    },
    "random_seed": 19,
}


def _write_moving_object_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 72)
    )
    assert writer.isOpened()
    try:
        for frame_index in range(5):
            frame = np.zeros((72, 96, 3), dtype=np.uint8)
            x = 20 + frame_index
            cv2.rectangle(frame, (x, 20), (x + 15, 39), (230, 230, 230), -1)
            writer.write(frame)
    finally:
        writer.release()


def write_complete_config(
    tmp_path: Path,
    overrides: Mapping[str, object] | None = None,
    with_source: bool = True,
) -> Path:
    config = deepcopy(VALID_CONFIG)
    for dotted_key, value in (overrides or {}).items():
        nested = config
        *parents, key = dotted_key.split(".")
        for parent in parents:
            nested = nested[parent]
        nested[key] = value
    source = tmp_path / "moving.avi"
    if with_source:
        _write_moving_object_video(source)
        config["source"]["sha256"] = sha256_file(source)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path

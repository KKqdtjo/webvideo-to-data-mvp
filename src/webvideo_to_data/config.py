"""Strict, immutable schema-v2 experiment configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from math import degrees, isfinite, radians
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

import yaml


_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_WINDOWS_DEVICES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_PHASES = (
    "home", "pregrasp", "approach", "close", "lift", "transport", "lower",
    "open", "retreat", "settle",
)


@dataclass(frozen=True)
class SourceConfig:
    id: str
    path: Path
    sha256: str
    fps: float
    roi_xywh: tuple[int, int, int, int]


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
    grasp_quaternion_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class IKConfig:
    position_tolerance_m: float
    orientation_tolerance_rad: float
    maximum_iterations: int
    damping: float
    step_size: float
    orientation_weight: float
    joint_limit_weight: float


@dataclass(frozen=True)
class ControlConfig:
    control_hz: float
    maximum_joint_velocity_rad_s: float
    maximum_joint_acceleration_rad_s2: float
    gripper_open_width_m: float
    gripper_closed_width_m: float
    phase_duration_s: Mapping[str, float]


@dataclass(frozen=True)
class CollisionConfig:
    maximum_penetration_m: float
    minimum_lift_m: float
    maximum_target_error_m: float
    settle_duration_s: float
    maximum_final_tilt_rad: float
    maximum_final_linear_speed_m_s: float
    minimum_bilateral_contact_duration_s: float
    minimum_lift_contact_duration_s: float
    allowed_contact_pairs: Mapping[str, tuple[tuple[str, str], ...]]


@dataclass(frozen=True)
class PerturbationConfig:
    rollout_count: int
    xy_half_range_m: float
    yaw_half_range_rad: float
    mass_fraction: float
    friction_fraction: float


@dataclass(frozen=True)
class MediaConfig:
    canvas_size: tuple[int, int]
    panel_size: tuple[int, int]
    letterbox_bgr: tuple[int, int, int]
    output_fps: float
    comparison_alignment: str


@dataclass(frozen=True)
class SimulationConfig:
    b0_mode: str
    b1_mode: str
    render_size: tuple[int, int]
    render_every: int


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_id: str
    source: SourceConfig
    tracking: TrackingConfig
    scene: SceneConfig
    ik: IKConfig
    control: ControlConfig
    collision: CollisionConfig
    perturbation: PerturbationConfig
    simulation: SimulationConfig
    media: MediaConfig
    random_seed: int
    config_path: Path


@dataclass(frozen=True)
class ValidatedPublicResolvedConfig:
    config_sha256: str
    experiment_id: str
    source_id: str
    config: ExperimentConfig


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _take(mapping: dict[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(unknown)}")
    missing = sorted(allowed - set(mapping))
    if missing:
        raise ValueError(f"missing {name} fields: {', '.join(missing)}")


def _float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _numbers(value: object, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return tuple(_float(item, name) for item in value)


def _integers(value: object, length: int, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return tuple(_int(item, name) for item in value)


def _positive(value: float, name: str) -> float:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative(value: float, name: str) -> float:
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _unit_interval(value: float, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _phase_durations(value: object) -> Mapping[str, float]:
    raw = _mapping(value, "control.phase_duration_s")
    _take(raw, set(_PHASES), "control.phase_duration_s")
    return MappingProxyType({phase: _positive(_float(raw[phase], f"control.phase_duration_s.{phase}"), f"control.phase_duration_s.{phase}") for phase in _PHASES})


def _contact_pairs(value: object) -> Mapping[str, tuple[tuple[str, str], ...]]:
    raw = _mapping(value, "collision.allowed_contact_pairs")
    _take(raw, set(_PHASES), "collision.allowed_contact_pairs")
    pairs: dict[str, tuple[tuple[str, str], ...]] = {}
    for phase in _PHASES:
        raw_pairs = raw[phase]
        if not isinstance(raw_pairs, (list, tuple)) or not raw_pairs:
            raise ValueError(f"collision.allowed_contact_pairs.{phase} must be a nonempty sequence")
        parsed: list[tuple[str, str]] = []
        for pair in raw_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"collision.allowed_contact_pairs.{phase} pairs must contain two names")
            left = _string(pair[0], f"collision.allowed_contact_pairs.{phase}")
            right = _string(pair[1], f"collision.allowed_contact_pairs.{phase}")
            if not left or not right:
                raise ValueError(f"collision.allowed_contact_pairs.{phase} names must be nonempty")
            parsed.append((left, right))
        pairs[phase] = tuple(parsed)
    return MappingProxyType(pairs)


def _parse_experiment_document(
    value: object, config_path: Path
) -> ExperimentConfig:
    """Apply the one canonical schema-v2 configuration contract."""

    config_path = config_path.resolve()
    document = _mapping(value, "experiment config")
    _take(document, {"schema_version", "experiment_id", "source", "tracking", "scene", "ik", "control", "collision", "perturbation", "simulation", "media", "random_seed"}, "experiment config")

    schema_version = _int(document["schema_version"], "schema_version")
    if schema_version != 2:
        raise ValueError("schema_version must be 2")
    experiment_id = _string(document["experiment_id"], "experiment_id")
    if not _EXPERIMENT_ID.fullmatch(experiment_id) or experiment_id.upper() in _WINDOWS_DEVICES:
        raise ValueError("experiment_id must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ and not be a Windows device name")

    source_raw = _mapping(document["source"], "source")
    _take(source_raw, {"id", "path", "sha256", "fps", "roi_xywh"}, "source")
    source_id = _string(source_raw["id"], "source.id")
    if not _SOURCE_ID.fullmatch(source_id) or source_id.upper() in _WINDOWS_DEVICES:
        raise ValueError("source.id must match ^[a-z0-9][a-z0-9-]{0,63}$ and not be a Windows device name")
    source_name = _string(source_raw["path"], "source.path")
    source_hash = _string(source_raw["sha256"], "source.sha256").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ValueError("source.sha256 must be a SHA-256 hex digest")
    source_fps = _positive(_float(source_raw["fps"], "source.fps"), "source.fps")
    roi = _integers(source_raw["roi_xywh"], 4, "source.roi_xywh")
    if roi[0] < 0 or roi[1] < 0:
        raise ValueError("roi x and y must be nonnegative")
    if roi[2] <= 0 or roi[3] <= 0:
        raise ValueError("roi width and height must be positive")
    source_path = Path(source_name)
    if not source_path.is_absolute():
        source_path = config_path.parent / source_path
    source = SourceConfig(source_id, source_path.resolve(), source_hash, source_fps, roi)

    tracking_raw = _mapping(document["tracking"], "tracking")
    _take(tracking_raw, {"forward_backward_threshold_px", "minimum_live_points", "minimum_valid_ratio"}, "tracking")
    minimum_live_points = _int(tracking_raw["minimum_live_points"], "minimum_live_points")
    if minimum_live_points <= 0:
        raise ValueError("minimum_live_points must be positive")
    tracking = TrackingConfig(
        _nonnegative(_float(tracking_raw["forward_backward_threshold_px"], "forward_backward_threshold_px"), "forward_backward_threshold_px"),
        minimum_live_points,
        _unit_interval(_float(tracking_raw["minimum_valid_ratio"], "minimum_valid_ratio"), "minimum_valid_ratio"),
    )

    scene_raw = _mapping(document["scene"], "scene")
    _take(scene_raw, {"x_bounds_m", "y_bounds_m", "b0_start_m", "b0_goal_m", "grasp_quaternion_wxyz"}, "scene")
    x_bounds = _numbers(scene_raw["x_bounds_m"], 2, "scene.x_bounds_m")
    y_bounds = _numbers(scene_raw["y_bounds_m"], 2, "scene.y_bounds_m")
    if x_bounds[0] >= x_bounds[1] or y_bounds[0] >= y_bounds[1]:
        raise ValueError("scene bounds must be increasing")
    quaternion = _numbers(scene_raw["grasp_quaternion_wxyz"], 4, "scene.grasp_quaternion_wxyz")
    if not any(quaternion):
        raise ValueError("scene.grasp_quaternion_wxyz must be nonzero")
    scene = SceneConfig(x_bounds, y_bounds, _numbers(scene_raw["b0_start_m"], 3, "scene.b0_start_m"), _numbers(scene_raw["b0_goal_m"], 3, "scene.b0_goal_m"), quaternion)

    ik_raw = _mapping(document["ik"], "ik")
    _take(ik_raw, {"position_tolerance_m", "orientation_tolerance_deg", "maximum_iterations", "damping", "step_size", "orientation_weight", "joint_limit_weight"}, "ik")
    maximum_iterations = _int(ik_raw["maximum_iterations"], "maximum_iterations")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    ik = IKConfig(
        _positive(_float(ik_raw["position_tolerance_m"], "position_tolerance_m"), "position_tolerance_m"),
        radians(_positive(_float(ik_raw["orientation_tolerance_deg"], "orientation_tolerance_deg"), "orientation_tolerance_deg")),
        maximum_iterations,
        _positive(_float(ik_raw["damping"], "damping"), "damping"),
        _positive(_float(ik_raw["step_size"], "step_size"), "step_size"),
        _positive(_float(ik_raw["orientation_weight"], "orientation_weight"), "orientation_weight"),
        _positive(_float(ik_raw["joint_limit_weight"], "joint_limit_weight"), "joint_limit_weight"),
    )

    control_raw = _mapping(document["control"], "control")
    _take(control_raw, {"control_hz", "maximum_joint_velocity_rad_s", "maximum_joint_acceleration_rad_s2", "gripper_open_width_m", "gripper_closed_width_m", "phase_duration_s"}, "control")
    open_width = _positive(_float(control_raw["gripper_open_width_m"], "gripper_open_width_m"), "gripper_open_width_m")
    closed_width = _float(control_raw["gripper_closed_width_m"], "gripper_closed_width_m")
    if closed_width < 0.0 or closed_width > open_width:
        raise ValueError("gripper_closed_width_m must be between zero and gripper_open_width_m")
    control = ControlConfig(
        _positive(_float(control_raw["control_hz"], "control_hz"), "control_hz"),
        _positive(_float(control_raw["maximum_joint_velocity_rad_s"], "maximum_joint_velocity_rad_s"), "maximum_joint_velocity_rad_s"),
        _positive(_float(control_raw["maximum_joint_acceleration_rad_s2"], "maximum_joint_acceleration_rad_s2"), "maximum_joint_acceleration_rad_s2"),
        open_width, closed_width, _phase_durations(control_raw["phase_duration_s"]),
    )

    collision_raw = _mapping(document["collision"], "collision")
    _take(collision_raw, {"maximum_penetration_m", "minimum_lift_m", "maximum_target_error_m", "settle_duration_s", "maximum_final_tilt_deg", "maximum_final_linear_speed_m_s", "minimum_bilateral_contact_duration_s", "minimum_lift_contact_duration_s", "allowed_contact_pairs"}, "collision")
    collision = CollisionConfig(
        _positive(_float(collision_raw["maximum_penetration_m"], "maximum_penetration_m"), "maximum_penetration_m"),
        _positive(_float(collision_raw["minimum_lift_m"], "minimum_lift_m"), "minimum_lift_m"),
        _positive(_float(collision_raw["maximum_target_error_m"], "maximum_target_error_m"), "maximum_target_error_m"),
        _positive(_float(collision_raw["settle_duration_s"], "settle_duration_s"), "settle_duration_s"),
        radians(_positive(_float(collision_raw["maximum_final_tilt_deg"], "maximum_final_tilt_deg"), "maximum_final_tilt_deg")),
        _positive(_float(collision_raw["maximum_final_linear_speed_m_s"], "maximum_final_linear_speed_m_s"), "maximum_final_linear_speed_m_s"),
        _positive(_float(collision_raw["minimum_bilateral_contact_duration_s"], "minimum_bilateral_contact_duration_s"), "minimum_bilateral_contact_duration_s"),
        _positive(_float(collision_raw["minimum_lift_contact_duration_s"], "minimum_lift_contact_duration_s"), "minimum_lift_contact_duration_s"),
        _contact_pairs(collision_raw["allowed_contact_pairs"]),
    )

    perturbation_raw = _mapping(document["perturbation"], "perturbation")
    _take(perturbation_raw, {"rollout_count", "xy_half_range_m", "yaw_half_range_deg", "mass_fraction", "friction_fraction"}, "perturbation")
    rollout_count = _int(perturbation_raw["rollout_count"], "rollout_count")
    if rollout_count != 30:
        raise ValueError("rollout_count must be 30")
    perturbation = PerturbationConfig(
        rollout_count,
        _positive(_float(perturbation_raw["xy_half_range_m"], "xy_half_range_m"), "xy_half_range_m"),
        radians(_positive(_float(perturbation_raw["yaw_half_range_deg"], "yaw_half_range_deg"), "yaw_half_range_deg")),
        _unit_interval(_float(perturbation_raw["mass_fraction"], "mass_fraction"), "mass_fraction"),
        _unit_interval(_float(perturbation_raw["friction_fraction"], "friction_fraction"), "friction_fraction"),
    )

    simulation_raw = _mapping(document["simulation"], "simulation")
    _take(simulation_raw, {"b0_mode", "b1_mode", "render_size", "render_every"}, "simulation")
    b0_mode = _string(simulation_raw["b0_mode"], "simulation.b0_mode")
    b1_mode = _string(simulation_raw["b1_mode"], "simulation.b1_mode")
    if b0_mode != "physics_grasp" or b1_mode != "kinematic_replay":
        raise ValueError("simulation modes must be physics_grasp and kinematic_replay")
    render_size = _integers(simulation_raw["render_size"], 2, "simulation.render_size")
    if min(render_size) <= 0:
        raise ValueError("simulation.render_size values must be positive")
    render_every = _int(simulation_raw["render_every"], "simulation.render_every")
    if render_every <= 0:
        raise ValueError("simulation.render_every must be positive")
    simulation = SimulationConfig(b0_mode, b1_mode, render_size, render_every)

    media_raw = _mapping(document["media"], "media")
    _take(media_raw, {"canvas_size", "panel_size", "letterbox_bgr", "output_fps", "comparison_alignment"}, "media")
    canvas = _integers(media_raw["canvas_size"], 2, "media.canvas_size")
    panel = _integers(media_raw["panel_size"], 2, "media.panel_size")
    letterbox = _integers(media_raw["letterbox_bgr"], 3, "media.letterbox_bgr")
    if min(canvas) <= 0 or min(panel) <= 0 or any(item < 0 or item > 255 for item in letterbox):
        raise ValueError("media sizes must be positive and letterbox_bgr must be bytes")
    alignment = _string(media_raw["comparison_alignment"], "media.comparison_alignment")
    if alignment != "time_warped":
        raise ValueError("media.comparison_alignment must be time_warped")
    media = MediaConfig(canvas, panel, letterbox, _positive(_float(media_raw["output_fps"], "media.output_fps"), "media.output_fps"), alignment)

    random_seed = _int(document["random_seed"], "random_seed")
    if random_seed < 0:
        raise ValueError("random_seed must be nonnegative")
    return ExperimentConfig(schema_version, experiment_id, source, tracking, scene, ik, control, collision, perturbation, simulation, media, random_seed, config_path)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a schema-v2 config, rejecting defaults and ambiguous inputs."""

    config_path = Path(path).resolve()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return _parse_experiment_document(document, config_path)


def _public_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: _public_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _public_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_public_value(item) for item in value]
    return value


def to_public_resolved_mapping(config: ExperimentConfig) -> Mapping[str, object]:
    """Return JSON/YAML-safe resolved parameters without local filesystem paths."""
    public = _public_value(config)
    assert isinstance(public, dict)
    public.pop("config_path")
    source = public["source"]
    assert isinstance(source, dict)
    source["path"] = f"registry:{config.source.id}"
    return public


def validate_public_resolved_mapping(
    value: object,
    logical_path: str | Path = "resolved-config.yaml",
) -> ValidatedPublicResolvedConfig:
    """Validate the public representation through the canonical Task 1 parser."""

    raw = _mapping(value, "public resolved config")
    public_fields = {
        field.name for field in fields(ExperimentConfig) if field.name != "config_path"
    } | {"config_sha256"}
    _take(raw, public_fields, "public resolved config")
    digest = _string(raw["config_sha256"], "config_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("config_sha256 must be a SHA-256 hex digest")

    document = deepcopy(raw)
    del document["config_sha256"]
    source = _mapping(document["source"], "source")
    source_id = _string(source.get("id"), "source.id")
    if source.get("path") != f"registry:{source_id}":
        raise ValueError("source.path must be the source registry reference")
    source["path"] = "public-source.registry"
    document["source"] = source

    for section_name, public_name, input_name in (
        ("ik", "orientation_tolerance_rad", "orientation_tolerance_deg"),
        ("collision", "maximum_final_tilt_rad", "maximum_final_tilt_deg"),
        ("perturbation", "yaw_half_range_rad", "yaw_half_range_deg"),
    ):
        section = _mapping(document[section_name], section_name)
        if public_name not in section or input_name in section:
            raise ValueError(f"invalid public {section_name} angle field")
        angle_rad = _float(section.pop(public_name), f"{section_name}.{public_name}")
        section[input_name] = degrees(angle_rad)
        document[section_name] = section

    parsed = _parse_experiment_document(document, Path(logical_path))
    return ValidatedPublicResolvedConfig(
        config_sha256=digest,
        experiment_id=parsed.experiment_id,
        source_id=parsed.source.id,
        config=parsed,
    )

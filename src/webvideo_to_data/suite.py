"""Append-only core suites and the fixed EXP-001 B0 robustness benchmark."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import threading
from typing import Any, Callable, Iterator, Mapping, Sequence
from uuid import uuid4

import cv2
import mujoco
import numpy as np
import yaml

from .artifacts import (
    VerifiedRun,
    pinned_model_identity,
    pinned_model_logical_identity,
    verify_run_directory,
)
from .config import (
    ExperimentConfig,
    ValidatedPublicResolvedConfig,
    load_experiment_config,
    to_public_resolved_mapping,
    validate_public_resolved_mapping,
)
from .dashboard import _build_dashboard
from .experiment import Variant, run_experiment
from .ik import ControlLimitError, IKPlanningError, plan_joint_control
from .physics_validation import (
    InvalidNumericalStateError,
    PhysicsRolloutFailure,
    validate_rollout,
)
from .redaction import (
    _StablePublicationSnapshot,
    _stable_publication_snapshot,
    audit_publication_tree,
)
from .path_security import (
    absolute_windows_filesystem_path,
    validate_windows_path_namespace as _validate_windows_path_namespace,
    windows_path_for_containment as _windows_path_for_containment,
)
from .retargeting import build_manual_b0_reference
from .scene import ScenePerturbation, apply_scene_perturbation, load_panda_scene
from .simulation import run_joint_control_program
from .visualization import labels_for_metrics, render_public_simulation_preview


_RUN_ID = re.compile(r"^\d{8}T\d{12}Z-[0-9a-f]{8}-[0-9a-f]{4}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VARIANTS: tuple[Variant, ...] = ("B0", "B1", "B2", "B3", "B4")
B0_SEEDS: tuple[int, ...] = tuple(range(19, 49))
_MEDIA_SUFFIXES = {".gif", ".mp4", ".png"}
_ENVIRONMENT_FIELDS = {
    "os_name",
    "os_version",
    "architecture",
    "python_version",
    "mujoco_version",
    "numpy_version",
    "opencv_version",
    "ffmpeg_version",
    "ffprobe_version",
    "generator_commit",
    "generator_dirty",
    "model_sha256",
    "renderer_backend",
}
_SUITE_MANIFEST_FIELDS = {
    "schema_version",
    "producer",
    "feature_set",
    "experiment_id",
    "run_id",
    "status",
    "requested_variants",
    "config_sha256",
    "files",
}
_SUITE_LOCKS_GUARD = threading.Lock()
_SUITE_LOCKS: dict[str, threading.Lock] = {}


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    run_dir: Path


@dataclass(frozen=True)
class RolloutRecord:
    seed: int
    perturbation: Mapping[str, float]
    passed: bool
    failed_checks: tuple[str, ...]
    execution_tracking_ratio: float
    maximum_lift_m: float
    target_error_m: float
    final_tilt_rad: float
    final_linear_speed_m_s: float
    forbidden_contact_count: int
    maximum_forbidden_penetration_m: float


@dataclass(frozen=True)
class B0BenchmarkSummary:
    rollouts: int
    successes: int
    passed: bool
    reason: str
    wilson_95_low: float
    wilson_95_high: float
    total_forbidden_contacts: int
    maximum_forbidden_penetration_m: float
    records: tuple[RolloutRecord, ...]


@dataclass(frozen=True)
class SuiteResult:
    run_id: str
    run_dir: Path
    metrics: Mapping[str, Any]
    dashboard_path: Path | None


@dataclass(frozen=True)
class VerifiedSuite:
    path: Path
    metrics: Mapping[str, Any]
    manifest: Mapping[str, Any]
    variant_runs: Mapping[str, VerifiedRun]
    environment: Mapping[str, Any]
    variant_provenance: Mapping[str, Mapping[str, Any]]
    captured_files: Mapping[str, bytes] = field(repr=False)


RolloutExecutor = Callable[
    [ExperimentConfig, int, ScenePerturbation], RolloutRecord
]


@dataclass(frozen=True)
class SuiteDeps:
    now_utc: Callable[[], datetime]
    random_suffix: Callable[[], str]
    run_variant: Callable[
        [ExperimentConfig, Path, Variant, bool], Mapping[str, Any]
    ]
    evaluate_b0: Callable[
        [ExperimentConfig, Sequence[int]], B0BenchmarkSummary
    ]


def validate_run_id(value: str) -> str:
    """Accept only the fixed, path-inert append-only run identifier format."""

    if type(value) is not str or not _RUN_ID.fullmatch(value):
        raise ValueError("invalid run_id")
    return value


def make_run_id(
    now_utc: datetime, config_sha256: str, random_suffix: str
) -> str:
    """Build and revalidate one timestamped identifier."""

    if now_utc.tzinfo is None or now_utc.utcoffset() is None:
        raise ValueError("now_utc must be timezone-aware")
    digest = str(config_sha256).lower()
    suffix = str(random_suffix).lower()
    if not _SHA256.fullmatch(digest) or not re.fullmatch(r"[0-9a-f]{4}", suffix):
        raise ValueError("invalid run_id component")
    stamp = now_utc.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return validate_run_id(f"{stamp}-{digest[:8]}-{suffix}")


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial proportion."""

    if type(successes) is not int or type(trials) is not int:
        raise ValueError("successes and trials must be integers")
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("successes and trials are inconsistent")
    denominator = 1.0 + z * z / trials
    center = (successes / trials + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(
            successes * (trials - successes) / trials**3
            + z * z / (4 * trials**2)
        )
        / denominator
    )
    return float(center - radius), float(center + radius)


def summarize_b0(records: Sequence[RolloutRecord]) -> B0BenchmarkSummary:
    """Apply the immutable fixed-30 B0 aggregate gate."""

    captured = tuple(records)
    successes = sum(record.passed for record in captured)
    total_forbidden = sum(record.forbidden_contact_count for record in captured)
    maximum_forbidden = max(
        (record.maximum_forbidden_penetration_m for record in captured),
        default=0.0,
    )
    if captured:
        low, high = wilson_interval(successes, len(captured))
    else:
        low = high = 0.0
    if total_forbidden:
        reason = "illegal_contact_observed"
    elif maximum_forbidden > 0.002:
        reason = "maximum_forbidden_penetration_exceeded"
    elif len(captured) != 30:
        reason = "fixed_thirty_rollouts_required"
    elif successes < 24:
        reason = "insufficient_successful_rollouts"
    else:
        reason = "passed"
    return B0BenchmarkSummary(
        rollouts=len(captured),
        successes=successes,
        passed=reason == "passed",
        reason=reason,
        wilson_95_low=low,
        wilson_95_high=high,
        total_forbidden_contacts=total_forbidden,
        maximum_forbidden_penetration_m=maximum_forbidden,
        records=captured,
    )


def _sample_perturbation(
    config: ExperimentConfig, seed: int
) -> ScenePerturbation:
    random = np.random.default_rng(seed)
    perturbation = config.perturbation
    # Draw order is fixed and each rollout starts from a freshly loaded nominal model.
    return ScenePerturbation(
        can_dx_m=float(
            random.uniform(-perturbation.xy_half_range_m, perturbation.xy_half_range_m)
        ),
        can_dy_m=float(
            random.uniform(-perturbation.xy_half_range_m, perturbation.xy_half_range_m)
        ),
        can_yaw_rad=float(
            random.uniform(
                -perturbation.yaw_half_range_rad,
                perturbation.yaw_half_range_rad,
            )
        ),
        mass_scale=1.0
        + float(random.uniform(-perturbation.mass_fraction, perturbation.mass_fraction)),
        friction_scale=1.0
        + float(
            random.uniform(
                -perturbation.friction_fraction,
                perturbation.friction_fraction,
            )
        ),
    )


def _perturbation_mapping(perturbation: ScenePerturbation) -> dict[str, float]:
    return {
        "can_dx_m": perturbation.can_dx_m,
        "can_dy_m": perturbation.can_dy_m,
        "can_yaw_rad": perturbation.can_yaw_rad,
        "mass_scale": perturbation.mass_scale,
        "friction_scale": perturbation.friction_scale,
    }


def _real_rollout_executor(
    config: ExperimentConfig,
    seed: int,
    perturbation: ScenePerturbation,
) -> RolloutRecord:
    model, data, ids = load_panda_scene()
    apply_scene_perturbation(model, data, ids, perturbation)
    reference = build_manual_b0_reference(
        config.scene.b0_start_m,
        config.scene.b0_goal_m,
        config.control,
        config.scene.grasp_quaternion_wxyz,
    )
    program = plan_joint_control(model, data, ids, reference, config.ik, config.control)
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
        render=False,
    )
    physics = validate_rollout(simulation, config.collision)
    return RolloutRecord(
        seed=seed,
        perturbation=_perturbation_mapping(perturbation),
        passed=physics.passed,
        failed_checks=physics.failed_checks,
        execution_tracking_ratio=simulation.execution_tracking_ratio,
        maximum_lift_m=simulation.maximum_lift_m,
        target_error_m=simulation.target_error_m,
        final_tilt_rad=float(simulation.can_tilt_rad[-1]),
        final_linear_speed_m_s=float(simulation.can_linear_speed_m_s[-1]),
        forbidden_contact_count=physics.forbidden_contact_count,
        maximum_forbidden_penetration_m=physics.maximum_forbidden_penetration_m,
    )


def _failed_rollout_record(
    seed: int,
    perturbation: ScenePerturbation,
    failed_checks: tuple[str, ...],
) -> RolloutRecord:
    return RolloutRecord(
        seed=seed,
        perturbation=_perturbation_mapping(perturbation),
        passed=False,
        failed_checks=failed_checks,
        execution_tracking_ratio=0.0,
        maximum_lift_m=0.0,
        target_error_m=0.0,
        final_tilt_rad=0.0,
        final_linear_speed_m_s=0.0,
        forbidden_contact_count=0,
        maximum_forbidden_penetration_m=0.0,
    )


def evaluate_b0_robustness(
    config: ExperimentConfig,
    seeds: Sequence[int],
    executor: RolloutExecutor | None = None,
) -> B0BenchmarkSummary:
    """Run every requested seed, converting only expected typed rollout failures."""

    run_one = executor or _real_rollout_executor
    records: list[RolloutRecord] = []
    for seed in seeds:
        if type(seed) is not int or seed < 0:
            raise ValueError("B0 seeds must be nonnegative integers")
        perturbation = _sample_perturbation(config, seed)
        try:
            record = run_one(config, seed, perturbation)
        except IKPlanningError as error:
            record = _failed_rollout_record(
                seed, perturbation, (f"ik_key_pose_{error.phase}",)
            )
        except ControlLimitError as error:
            record = _failed_rollout_record(
                seed,
                perturbation,
                (f"control_limit_{error.phase}_{error.limit}",),
            )
        except PhysicsRolloutFailure as error:
            record = _failed_rollout_record(seed, perturbation, error.failed_checks)
        except InvalidNumericalStateError:
            record = _failed_rollout_record(
                seed, perturbation, ("invalid_numerical_state",)
            )
        if record.seed != seed:
            raise ValueError("rollout executor returned the wrong seed")
        records.append(replace(record, perturbation=_perturbation_mapping(perturbation)))
    return summarize_b0(records)


def _default_run_variant(
    config: ExperimentConfig,
    destination: Path,
    variant: Variant,
    no_render: bool,
) -> Mapping[str, Any]:
    return run_experiment(
        config.config_path,
        destination,
        variant=variant,
        no_render=no_render,
    )


def _default_deps() -> SuiteDeps:
    return SuiteDeps(
        now_utc=lambda: datetime.now(timezone.utc),
        random_suffix=lambda: os.urandom(2).hex(),
        run_variant=_default_run_variant,
        evaluate_b0=evaluate_b0_robustness,
    )


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_bytes_new(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_new(path: Path, payload: Any) -> None:
    _write_bytes_new(path, _json_bytes(payload))


def _write_yaml_new(path: Path, payload: Mapping[str, Any]) -> None:
    content = yaml.safe_dump(dict(payload), sort_keys=False).encode("utf-8")
    _write_bytes_new(path, content)


def _tool_version(name: str) -> str:
    try:
        completed = subprocess.run(
            [name, "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    first = (completed.stdout or completed.stderr).splitlines()
    return first[0].strip() if completed.returncode == 0 and first else "unavailable"


def _git_identity() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repository), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable", True
    return commit, dirty


def _environment_payload() -> dict[str, Any]:
    generator_commit, generator_dirty = _git_identity()
    return {
        "os_name": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "ffmpeg_version": _tool_version("ffmpeg"),
        "ffprobe_version": _tool_version("ffprobe"),
        "generator_commit": generator_commit,
        "generator_dirty": generator_dirty,
        "model_sha256": pinned_model_identity("B0"),
        "renderer_backend": os.environ.get("MUJOCO_GL", "default"),
    }


def _validated_variants(variants: Sequence[Variant]) -> tuple[Variant, ...]:
    values = tuple(variants)
    if not values or any(value not in _VARIANTS for value in values):
        raise ValueError("variants must contain B0, B1, B2, B3, or B4")
    if len(set(values)) != len(values):
        raise ValueError("variants must not contain duplicates")
    return values


def _contained(candidate: Path, root: Path, description: str) -> Path:
    if os.name == "nt":
        _validate_windows_path_namespace(candidate, description)
        _validate_windows_path_namespace(root, f"{description} root")
    resolved = candidate.resolve(strict=False)
    try:
        if os.name == "nt":
            relative = _windows_path_for_containment(resolved).relative_to(
                _windows_path_for_containment(root)
            )
        else:
            relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"unsafe {description} path") from error
    if not relative.parts:
        return root
    return root.joinpath(*resolved.parts[-len(relative.parts) :])


def _run_identity(
    config: ExperimentConfig,
    artifacts_root: str | Path,
    run_id: str,
) -> tuple[RunIdentity, Path, Path, Path]:
    lexical_root = Path(artifacts_root)
    if os.name == "nt":
        _validate_windows_path_namespace(lexical_root, "artifacts root")
    root = lexical_root.resolve(strict=False)
    experiment = _contained(root / config.experiment_id, root, "experiment")
    runs = _contained(experiment / "runs", root, "runs")
    candidate = _contained(runs / validate_run_id(run_id), root, "run")
    latest = _contained(experiment / "latest.json", root, "latest")
    lock_path = _contained(experiment / ".suite.lock", root, "lock")
    return RunIdentity(run_id, candidate), root, latest, lock_path


def _absolute_artifacts_root(artifacts_root: str | Path) -> Path:
    lexical_root = Path(artifacts_root)
    if os.name != "nt":
        return lexical_root
    return absolute_windows_filesystem_path(lexical_root, "artifacts root")


def _record_payload(record: RolloutRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["schema_version"] = 1
    payload["failed_checks"] = list(record.failed_checks)
    payload["perturbation"] = dict(record.perturbation)
    return payload


def _summary_payload(summary: B0BenchmarkSummary) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rollouts": summary.rollouts,
        "successes": summary.successes,
        "passed": summary.passed,
        "reason": summary.reason,
        "wilson_95_low": summary.wilson_95_low,
        "wilson_95_high": summary.wilson_95_high,
        "total_forbidden_contacts": summary.total_forbidden_contacts,
        "maximum_forbidden_penetration_m": summary.maximum_forbidden_penetration_m,
        "yaw_perturbation_observability": (
            "geometrically_unobservable_for_axisymmetric_can"
        ),
        "records_file": "benchmark-rollouts.jsonl",
    }


def _write_b0_benchmark(directory: Path, summary: B0BenchmarkSummary) -> None:
    directory.mkdir(exist_ok=False)
    lines = b"".join(_json_bytes(_record_payload(record)) for record in summary.records)
    _write_bytes_new(directory / "benchmark-rollouts.jsonl", lines)
    _write_json_new(directory / "benchmark-summary.json", _summary_payload(summary))


def _file_signature(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {"size": len(content), "sha256": sha256(content).hexdigest()}


def _verify_decodable_media(path: Path) -> None:
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    expected_size: tuple[int, int] | None = None
    try:
        if not capture.isOpened():
            raise ValueError("suite media cannot be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        declared_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0.0 or declared_count <= 0:
            raise ValueError("suite media reports invalid timing")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise ValueError("suite media contains an empty frame")
            size = (frame.shape[1], frame.shape[0])
            if expected_size is None:
                expected_size = size
            elif size != expected_size:
                raise ValueError("suite media frame dimensions changed")
            decoded += 1
    finally:
        capture.release()
    if decoded <= 0 or decoded != declared_count:
        raise ValueError("suite media frame count is not fully decodable")


def _variant_manifest_hash(directory: Path) -> str:
    return sha256((directory / "run_manifest.json").read_bytes()).hexdigest()


def _suite_manifest(
    run_dir: Path,
    *,
    experiment_id: str,
    run_id: str,
    requested_variants: Sequence[Variant],
    config_sha256: str,
    status: str,
    feature_set: Sequence[str] = ("core",),
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    child_manifests: dict[str, Mapping[str, Any]] = {}
    variants_dir = run_dir / "variants"
    if variants_dir.is_dir():
        for variant in requested_variants:
            child = variants_dir / variant
            if child.is_dir():
                verified = verify_run_directory(child)
                child_manifests[variant] = verified.manifest
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        if relative == "suite-manifest.json":
            continue
        entry = _file_signature(path)
        if path.suffix.lower() in _MEDIA_SUFFIXES:
            parts = Path(relative).parts
            if len(parts) >= 3 and parts[0] == "variants":
                variant = parts[1]
                child_name = Path(*parts[2:]).as_posix()
                child_entry = child_manifests.get(variant, {}).get("files", {}).get(
                    child_name
                )
                if not isinstance(child_entry, dict):
                    raise ValueError("suite media lacks verified variant classification")
                entry["media_role"] = child_entry.get("media_role")
                entry["contains_private_source_frames"] = child_entry.get(
                    "contains_private_source_frames"
                )
            elif (
                list(feature_set) == ["core", "dashboard"]
                and len(parts) == 3
                and parts[:2] == ("dashboard", "media")
                and re.fullmatch(r"B[01]-preview\.gif", parts[2])
            ):
                entry["media_role"] = "public_simulation_preview"
                entry["contains_private_source_frames"] = False
            else:
                raise ValueError("suite contains unclassified media")
        files[relative] = entry
    return {
        "schema_version": 1,
        "producer": "webvideo_to_data.suite",
        "feature_set": list(feature_set),
        "experiment_id": experiment_id,
        "run_id": run_id,
        "status": status,
        "requested_variants": list(requested_variants),
        "config_sha256": config_sha256,
        "files": files,
    }


def _write_suite_manifest(
    run_dir: Path,
    *,
    experiment_id: str,
    run_id: str,
    requested_variants: Sequence[Variant],
    config_sha256: str,
    status: str,
    feature_set: Sequence[str] = ("core",),
) -> None:
    _write_json_new(
        run_dir / "suite-manifest.json",
        _suite_manifest(
            run_dir,
            experiment_id=experiment_id,
            run_id=run_id,
            requested_variants=requested_variants,
            config_sha256=config_sha256,
            status=status,
            feature_set=feature_set,
        ),
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("expected JSON object")
    return value


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _verify_environment_payload(directory: Path) -> dict[str, Any]:
    environment = _load_json_object(directory / "environment.json")
    if set(environment) != _ENVIRONMENT_FIELDS:
        raise ValueError("invalid environment fields")
    for name in _ENVIRONMENT_FIELDS - {"generator_dirty"}:
        value = environment.get(name)
        if type(value) is not str or not value or value != value.strip() or "\n" in value:
            raise ValueError("invalid environment value")
    if type(environment.get("generator_dirty")) is not bool:
        raise ValueError("invalid environment dirty flag")
    commit = environment["generator_commit"]
    if commit != "unavailable" and not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("invalid environment commit")
    model_sha256 = environment["model_sha256"]
    if (
        not _SHA256.fullmatch(model_sha256)
        or model_sha256 != pinned_model_identity("B0")
    ):
        raise ValueError("environment model hash is not the pinned suite model")
    return environment


def _verify_public_resolved_config(
    directory: Path, manifest: Mapping[str, Any]
) -> ValidatedPublicResolvedConfig:
    resolved = yaml.safe_load(
        (directory / "resolved-config.yaml").read_text(encoding="utf-8")
    )
    validated = validate_public_resolved_mapping(
        resolved, directory / "resolved-config.yaml"
    )
    if validated.config_sha256 != manifest.get("config_sha256"):
        raise ValueError("suite config hash disagrees with resolved config")
    if validated.experiment_id != manifest.get("experiment_id"):
        raise ValueError("resolved config experiment mismatch")
    return validated


def _verify_b0_payload(directory: Path, metrics: Mapping[str, Any]) -> None:
    summary = _load_json_object(directory / "B0" / "benchmark-summary.json")
    expected_summary_fields = {
        "schema_version",
        "rollouts",
        "successes",
        "passed",
        "reason",
        "wilson_95_low",
        "wilson_95_high",
        "total_forbidden_contacts",
        "maximum_forbidden_penetration_m",
        "yaw_perturbation_observability",
        "records_file",
    }
    if (
        set(summary) != expected_summary_fields
        or summary.get("schema_version") != 1
        or summary.get("yaw_perturbation_observability")
        != "geometrically_unobservable_for_axisymmetric_can"
        or summary.get("records_file") != "benchmark-rollouts.jsonl"
    ):
        raise ValueError("invalid B0 benchmark summary schema")
    record_fields = {
        "schema_version",
        "seed",
        "perturbation",
        "passed",
        "failed_checks",
        "execution_tracking_ratio",
        "maximum_lift_m",
        "target_error_m",
        "final_tilt_rad",
        "final_linear_speed_m_s",
        "forbidden_contact_count",
        "maximum_forbidden_penetration_m",
    }
    perturbation_fields = {
        "can_dx_m",
        "can_dy_m",
        "can_yaw_rad",
        "mass_scale",
        "friction_scale",
    }
    records: list[RolloutRecord] = []
    raw_lines = (directory / "B0" / "benchmark-rollouts.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    for line in raw_lines:
        value = json.loads(line)
        if type(value) is not dict or set(value) != record_fields:
            raise ValueError("invalid B0 rollout record schema")
        seed = value.get("seed")
        perturbation = value.get("perturbation")
        failed_checks = value.get("failed_checks")
        forbidden_count = value.get("forbidden_contact_count")
        if (
            value.get("schema_version") != 1
            or type(seed) is not int
            or seed < 0
            or type(perturbation) is not dict
            or set(perturbation) != perturbation_fields
            or any(
                type(name) is not str
                or not name
                or not np.isfinite(_finite_float(item, "perturbation"))
                for name, item in perturbation.items()
            )
            or type(value.get("passed")) is not bool
            or type(failed_checks) is not list
            or any(type(item) is not str or not item for item in failed_checks)
            or type(forbidden_count) is not int
            or forbidden_count < 0
        ):
            raise ValueError("invalid B0 rollout record values")
        records.append(
            RolloutRecord(
                seed=seed,
                perturbation={
                    str(name): _finite_float(item, "perturbation")
                    for name, item in perturbation.items()
                },
                passed=value["passed"],
                failed_checks=tuple(failed_checks),
                execution_tracking_ratio=_finite_float(
                    value.get("execution_tracking_ratio"),
                    "execution_tracking_ratio",
                ),
                maximum_lift_m=_finite_float(
                    value.get("maximum_lift_m"), "maximum_lift_m"
                ),
                target_error_m=_finite_float(
                    value.get("target_error_m"), "target_error_m"
                ),
                final_tilt_rad=_finite_float(
                    value.get("final_tilt_rad"), "final_tilt_rad"
                ),
                final_linear_speed_m_s=_finite_float(
                    value.get("final_linear_speed_m_s"),
                    "final_linear_speed_m_s",
                ),
                forbidden_contact_count=forbidden_count,
                maximum_forbidden_penetration_m=_finite_float(
                    value.get("maximum_forbidden_penetration_m"),
                    "maximum_forbidden_penetration_m",
                ),
            )
        )
    if [record.seed for record in records] != list(B0_SEEDS):
        raise ValueError("B0 rollout seeds do not match the fixed seed set")
    aggregate = summarize_b0(records)
    expected_summary = _summary_payload(aggregate)
    if summary != expected_summary:
        raise ValueError("B0 benchmark aggregate mismatch")
    if (
        metrics.get("b0_physics_baseline")
        != ("passed" if aggregate.passed else "failed")
        or metrics.get("b0_rollouts") != aggregate.rollouts
        or metrics.get("b0_successes") != aggregate.successes
    ):
        raise ValueError("B0 suite metrics mismatch")


def _verify_suite_once(directory: Path) -> VerifiedSuite:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("suite path must be a real directory")
    manifest = _load_json_object(directory / "suite-manifest.json")
    metrics = _load_json_object(directory / "suite-metrics.json")
    expected_metrics_fields = {
        "schema_version",
        "experiment_id",
        "run_id",
        "status",
        "reason",
        "requested_variants",
        "variants",
        "b0_physics_baseline",
        "b0_rollouts",
        "b0_successes",
        "actions_exported",
    }
    feature_set = manifest.get("feature_set")
    if (
        set(manifest) != _SUITE_MANIFEST_FIELDS
        or manifest.get("schema_version") != 1
        or manifest.get("producer") != "webvideo_to_data.suite"
        or feature_set not in (["core"], ["core", "dashboard"])
        or manifest.get("status") != "recorded"
    ):
        raise ValueError("invalid suite manifest root")
    config_digest = manifest.get("config_sha256")
    if type(config_digest) is not str or not _SHA256.fullmatch(config_digest):
        raise ValueError("invalid suite config hash")
    environment = _verify_environment_payload(directory)
    validated_config = _verify_public_resolved_config(directory, manifest)
    if (
        set(metrics) != expected_metrics_fields
        or metrics.get("schema_version") != 1
        or metrics.get("reason") != "suite_recorded"
    ):
        raise ValueError("invalid suite metrics schema")
    run_id = manifest.get("run_id")
    if (
        type(run_id) is not str
        or validate_run_id(run_id) != directory.name
        or metrics.get("run_id") != run_id
        or metrics.get("experiment_id") != manifest.get("experiment_id")
        or metrics.get("status") != "recorded"
    ):
        raise ValueError("inconsistent suite identity")
    requested = manifest.get("requested_variants")
    if type(requested) is not list:
        raise ValueError("invalid requested variants")
    requested_variants = _validated_variants(requested)
    if metrics.get("requested_variants") != requested:
        raise ValueError("suite requested variants mismatch")
    if "B0" in requested_variants:
        _verify_b0_payload(directory, metrics)
    elif (
        (directory / "B0").exists()
        or metrics.get("b0_physics_baseline") != "not_requested"
        or metrics.get("b0_rollouts") is not None
        or metrics.get("b0_successes") is not None
    ):
        raise ValueError("unexpected B0 benchmark payload")
    if list(directory.rglob("actions.npz")):
        raise ValueError("suite action exports are forbidden")
    if metrics.get("actions_exported") != 0:
        raise ValueError("suite action count must be zero")
    dashboard_path = directory / "dashboard" / "index.html"
    dashboard_html = ""
    if feature_set == ["core", "dashboard"]:
        if not dashboard_path.is_file():
            raise ValueError("enhanced suite dashboard is missing")
        dashboard_html = dashboard_path.read_text(encoding="utf-8")
        lowered_dashboard = dashboard_html.lower()
        if (
            "NO ACTION EXPORTED" not in dashboard_html
            or "REJECTED — NOT ACTION DATA" not in dashboard_html
            or "<script" in lowered_dashboard
            or "http://" in lowered_dashboard
            or "https://" in lowered_dashboard
            or "file://" in lowered_dashboard
            or re.search(r"[a-zA-Z]:[\\/]", dashboard_html)
        ):
            raise ValueError("dashboard publication contract mismatch")
        if audit_publication_tree(directory / "dashboard"):
            raise ValueError("dashboard privacy audit failed")
    elif (directory / "dashboard").exists():
        raise ValueError("core suite contains dashboard output")
    variant_metrics = metrics.get("variants")
    if type(variant_metrics) is not dict or set(variant_metrics) != set(requested):
        raise ValueError("suite variant metrics mismatch")
    verified_runs: dict[str, VerifiedRun] = {}
    variant_provenance: dict[str, Mapping[str, Any]] = {}
    for variant in requested_variants:
        child_directory = directory / "variants" / variant
        verified = verify_run_directory(child_directory)
        if (
            verified.metrics.get("variant") != variant
            or verified.manifest.get("format_version") != 4
            or verified.manifest.get("config_sha256") != config_digest
        ):
            raise ValueError("verified variant identity mismatch")
        provenance = _load_json_object(child_directory / "provenance.json")
        source = provenance.get("source")
        model = provenance.get("model")
        if (
            provenance.get("experiment_id") != validated_config.experiment_id
            or type(source) is not dict
            or source.get("id") != validated_config.source_id
            or type(model) is not dict
            or model.get("id") != pinned_model_logical_identity(variant)
            or model.get("sha256") != verified.manifest.get("model_sha256")
        ):
            raise ValueError("verified variant logical identity mismatch")
        recorded = variant_metrics[variant]
        if (
            type(recorded) is not dict
            or recorded.get("status") != verified.metrics.get("status")
            or recorded.get("reason") != verified.metrics.get("reason")
            or recorded.get("run_manifest_sha256")
            != _variant_manifest_hash(directory / "variants" / variant)
        ):
            raise ValueError("suite variant summary mismatch")
        verified_runs[variant] = verified
        variant_provenance[variant] = provenance
    files = manifest.get("files")
    actual = {
        path.relative_to(directory).as_posix(): path
        for path in directory.rglob("*")
        if path.is_file()
        and path.relative_to(directory).as_posix() != "suite-manifest.json"
    }
    if type(files) is not dict or set(files) != set(actual):
        raise ValueError("suite manifested file set mismatch")
    for name, path in actual.items():
        recorded = files[name]
        expected_fields = {"size", "sha256"}
        if path.suffix.lower() in _MEDIA_SUFFIXES:
            expected_fields |= {"media_role", "contains_private_source_frames"}
        if type(recorded) is not dict or set(recorded) != expected_fields:
            raise ValueError("invalid suite manifest file entry")
        signature = _file_signature(path)
        if any(recorded.get(field) != signature[field] for field in signature):
            raise ValueError("suite file signature mismatch")
        if path.suffix.lower() in _MEDIA_SUFFIXES:
            parts = Path(name).parts
            if len(parts) >= 3 and parts[0] == "variants":
                child_entry = verified_runs[parts[1]].manifest["files"].get(
                    Path(*parts[2:]).as_posix()
                )
                if (
                    not isinstance(child_entry, dict)
                    or recorded.get("media_role") != child_entry.get("media_role")
                    or recorded.get("contains_private_source_frames")
                    != child_entry.get("contains_private_source_frames")
                    or type(recorded.get("media_role")) is not str
                    or type(recorded.get("contains_private_source_frames")) is not bool
                ):
                    raise ValueError("suite media classification mismatch")
                if (
                    recorded.get("contains_private_source_frames") is True
                    and f"../variants/{parts[1]}/{Path(*parts[2:]).as_posix()}"
                    in dashboard_html
                ):
                    raise ValueError("dashboard links private local media")
            elif (
                feature_set == ["core", "dashboard"]
                and len(parts) == 3
                and parts[:2] == ("dashboard", "media")
                and re.fullmatch(r"B[01]-preview\.gif", parts[2])
            ):
                preview_variant = parts[2][:2]
                if preview_variant not in requested_variants:
                    raise ValueError("dashboard preview variant was not requested")
                classification = (
                    recorded.get("media_role"),
                    recorded.get("contains_private_source_frames"),
                )
                if classification not in {
                    ("public_simulation_preview", False),
                    ("source_simulation_comparison", True),
                }:
                    raise ValueError("dashboard media classification mismatch")
                _verify_decodable_media(path)
            else:
                raise ValueError("untrusted suite media")
    return VerifiedSuite(
        path=directory,
        metrics=dict(metrics),
        manifest=dict(manifest),
        variant_runs=verified_runs,
        environment=environment,
        variant_provenance=variant_provenance,
        captured_files={},
    )


def verify_suite_directory(path: str | Path) -> VerifiedSuite:
    """Verify one immutable schema-v1 core or enhanced suite and every child run."""

    try:
        with _verified_suite_capability(path) as verified:
            return verified
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("suite directory verification failed") from error


@contextmanager
def _verified_suite_capability(
    path: str | Path,
) -> Iterator[VerifiedSuite]:
    directory = Path(path)
    with _stable_publication_snapshot(directory) as snapshot:
        stable_directory = snapshot.materialized_root
        if stable_directory is None:
            raise ValueError("stable suite snapshot was not materialized")
        yield _verify_suite_snapshot(
            stable_directory,
            snapshot,
            display_path=directory,
        )


def _verify_suite_snapshot(
    materialized_directory: Path,
    snapshot: _StablePublicationSnapshot,
    *,
    display_path: Path,
) -> VerifiedSuite:
    if snapshot.kind != "suite":
        raise ValueError("stable snapshot is not a suite")
    verified = _verify_suite_once(materialized_directory)
    return _rebase_verified_suite(verified, display_path, snapshot)


def _rebase_verified_suite(
    verified: VerifiedSuite,
    directory: Path,
    snapshot: _StablePublicationSnapshot,
) -> VerifiedSuite:
    variant_runs: dict[str, VerifiedRun] = {}
    for variant, run in verified.variant_runs.items():
        prefix = f"variants/{variant}/"
        stable_files = {}
        for name, (size, digest) in snapshot.file_signatures.items():
            if not name.startswith(prefix):
                continue
            relative = name[len(prefix) :]
            if "/" in relative:
                continue
            mtime_ns, device, inode = snapshot.file_identities[name]
            stable_files[relative] = (
                size,
                digest,
                mtime_ns,
                device,
                inode,
            )
        child_relative = f"variants/{variant}"
        variant_runs[variant] = replace(
            run,
            path=directory / "variants" / variant,
            directory_identity=snapshot.directory_identities[child_relative],
            snapshot=stable_files,
        )
    return replace(
        verified,
        path=directory,
        variant_runs=variant_runs,
        captured_files=snapshot.captured_files,
    )


@contextmanager
def _experiment_lock(lock_path: Path):
    key = str(lock_path.resolve(strict=False)).casefold()
    with _SUITE_LOCKS_GUARD:
        thread_lock = _SUITE_LOCKS.setdefault(key, threading.Lock())
    with thread_lock:
        with lock_path.open("a+b") as stream:
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                while True:
                    try:
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        import time

                        time.sleep(0.01)
                try:
                    yield
                finally:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _validate_latest(
    latest_path: Path, artifacts_root: Path
) -> tuple[str, str] | None:
    if not latest_path.exists():
        return None
    payload = _load_json_object(latest_path)
    if set(payload) != {"run_path", "run_id", "suite_manifest_sha256"}:
        raise ValueError("invalid latest pointer")
    run_id = validate_run_id(payload.get("run_id"))
    run_path = payload.get("run_path")
    digest = payload.get("suite_manifest_sha256")
    if type(run_path) is not str or Path(run_path).is_absolute() or not _SHA256.fullmatch(
        str(digest)
    ):
        raise ValueError("invalid latest pointer")
    target = _contained(artifacts_root / run_path, artifacts_root, "latest target")
    verified = verify_suite_directory(target)
    if verified.manifest.get("run_id") != run_id:
        raise ValueError("latest pointer run mismatch")
    manifest_bytes = verified.captured_files.get("suite-manifest.json")
    if (
        not isinstance(manifest_bytes, bytes)
        or sha256(manifest_bytes).hexdigest() != digest
    ):
        raise ValueError("latest pointer manifest mismatch")
    return run_id, run_id.split("-", maxsplit=1)[0]


def _replace_latest(latest_path: Path, payload: Mapping[str, Any]) -> None:
    temporary = latest_path.with_name(f".{latest_path.name}.tmp-{uuid4().hex}")
    try:
        _write_bytes_new(temporary, _json_bytes(payload))
        os.replace(temporary, latest_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_latest(
    identity: RunIdentity,
    artifacts_root: Path,
    latest_path: Path,
    lock_path: Path,
) -> None:
    with _experiment_lock(lock_path):
        verified = verify_suite_directory(identity.run_dir)
        current = _validate_latest(latest_path, artifacts_root)
        new_key = identity.run_id.split("-", maxsplit=1)[0]
        if current is not None and current[1] >= new_key:
            return
        manifest_bytes = verified.captured_files.get("suite-manifest.json")
        if not isinstance(manifest_bytes, bytes):
            raise ValueError("verified suite manifest bytes are unavailable")
        manifest_digest = sha256(manifest_bytes).hexdigest()
        relative = identity.run_dir.relative_to(artifacts_root).as_posix()
        _replace_latest(
            latest_path,
            {
                "run_path": relative,
                "run_id": identity.run_id,
                "suite_manifest_sha256": manifest_digest,
            },
        )
        # Preserve the fully verified object until pointer replacement completed.
        if verified.manifest.get("run_id") != identity.run_id:
            raise ValueError("suite identity changed during publication")


def _initial_metrics(
    config: ExperimentConfig,
    run_id: str,
    variants: Sequence[Variant],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "run_id": run_id,
        "status": "building",
        "reason": "suite_in_progress",
        "requested_variants": list(variants),
        "variants": {},
        "b0_physics_baseline": "not_requested",
        "b0_rollouts": None,
        "b0_successes": None,
        "actions_exported": 0,
    }


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        _write_bytes_new(temporary, _json_bytes(payload))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_suite(
    config_path: str | Path,
    artifacts_root: str | Path,
    variants: Sequence[Variant],
    no_render: bool,
    run_id: str | None = None,
    deps: SuiteDeps | None = None,
) -> SuiteResult:
    """Create one immutable core suite and update its monotonic latest pointer."""

    requested = _validated_variants(variants)
    config_file = Path(config_path).resolve()
    config = load_experiment_config(config_file)
    config_digest = sha256(config_file.read_bytes()).hexdigest()
    dependencies = deps or _default_deps()
    selected_run_id = (
        make_run_id(
            dependencies.now_utc(), config_digest, dependencies.random_suffix()
        )
        if run_id is None
        else validate_run_id(run_id)
    )
    identity, root, latest_path, lock_path = _run_identity(
        config, _absolute_artifacts_root(artifacts_root), selected_run_id
    )
    if identity.run_dir.exists():
        raise FileExistsError(f"run already exists: {selected_run_id}")
    metrics = _initial_metrics(config, selected_run_id, requested)
    identity.run_dir.mkdir(parents=True, exist_ok=False)
    active_variant: Variant | None = None
    metrics_path = identity.run_dir / "suite-metrics.json"
    try:
        _write_json_new(metrics_path, metrics)
        variants_dir = identity.run_dir / "variants"
        variants_dir.mkdir(exist_ok=False)
        resolved = {
            "config_sha256": config_digest,
            **to_public_resolved_mapping(config),
        }
        _write_yaml_new(identity.run_dir / "resolved-config.yaml", resolved)
        _write_json_new(
            identity.run_dir / "environment.json", _environment_payload()
        )
        for variant in requested:
            active_variant = variant
            destination = variants_dir / variant
            child_metrics = dependencies.run_variant(
                config, destination, variant, no_render
            )
            verified = verify_run_directory(destination)
            if verified.metrics.get("variant") != variant:
                raise ValueError("variant runner returned a mismatched variant")
            if child_metrics != verified.metrics:
                raise ValueError("variant runner metrics disagree with verified bytes")
            if verified.metrics.get("status") == "failed":
                raise RuntimeError(f"variant {variant} failed")
            metrics["variants"][variant] = {
                "status": verified.metrics.get("status"),
                "reason": verified.metrics.get("reason"),
                "run_manifest_sha256": _variant_manifest_hash(destination),
            }
        if "B0" in requested:
            active_variant = "B0"
            summary = dependencies.evaluate_b0(config, B0_SEEDS)
            _write_b0_benchmark(identity.run_dir / "B0", summary)
            metrics.update(
                b0_physics_baseline="passed" if summary.passed else "failed",
                b0_rollouts=summary.rollouts,
                b0_successes=summary.successes,
            )
        metrics.update(status="recorded", reason="suite_recorded")
        _replace_json(identity.run_dir / "suite-metrics.json", metrics)
        trusted_preview_variants: set[str] = set()
        if not no_render:
            for variant in requested:
                if variant not in {"B0", "B1"}:
                    continue
                child_directory = variants_dir / variant
                with _stable_publication_snapshot(child_directory) as child_snapshot:
                    stable_child = child_snapshot.materialized_root
                    if stable_child is None:
                        raise ValueError("stable child snapshot was not materialized")
                    verified_child = verify_run_directory(stable_child)
                    replay_entry = verified_child.manifest.get("files", {}).get(
                        "mujoco_replay.mp4"
                    )
                    if replay_entry is None:
                        continue
                    if (
                        not isinstance(replay_entry, Mapping)
                        or replay_entry.get("media_role") != "simulation_only"
                        or replay_entry.get("contains_private_source_frames") is not False
                    ):
                        raise ValueError(
                            "suite preview source is not verified simulation-only media"
                        )
                    render_public_simulation_preview(
                        stable_child / "mujoco_replay.mp4",
                        identity.run_dir
                        / "dashboard"
                        / "media"
                        / f"{variant}-preview.gif",
                        labels_for_metrics(verified_child.metrics),
                        config.media,
                    )
                trusted_preview_variants.add(variant)
        dashboard_path = _build_dashboard(
            identity.run_dir,
            trusted_preview_variants=frozenset(trusted_preview_variants),
        )
        _write_suite_manifest(
            identity.run_dir,
            experiment_id=config.experiment_id,
            run_id=selected_run_id,
            requested_variants=requested,
            config_sha256=config_digest,
            status="recorded",
            feature_set=("core", "dashboard"),
        )
        verified_suite = verify_suite_directory(identity.run_dir)
        _publish_latest(identity, root, latest_path, lock_path)
        return SuiteResult(
            run_id=selected_run_id,
            run_dir=identity.run_dir,
            metrics=verified_suite.metrics,
            dashboard_path=dashboard_path,
        )
    except Exception as error:
        metrics.update(
            status="failed",
            reason="suite_infrastructure_failure",
            failure_variant=active_variant,
            error_type=type(error).__name__,
        )
        try:
            if metrics_path.exists():
                _replace_json(metrics_path, metrics)
            else:
                _write_json_new(metrics_path, metrics)
        except Exception:
            pass
        raise

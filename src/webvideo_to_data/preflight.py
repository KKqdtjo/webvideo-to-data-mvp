"""Read-only environment checks for one or more experiment variants."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Sequence

import mujoco

from .config import ExperimentConfig, load_experiment_config
from .media import sha256_file
from .redaction import redact_text
from .scene import DEFAULT_SCENE_PATH, load_panda_scene
from .source_registry import SourceRecord, load_source_registry


_VARIANTS = {"B0", "B1", "B2", "B3", "B4"}


@dataclass(frozen=True)
class PreflightDeps:
    executable_lookup: Callable[[str], str | None]
    version_runner: Callable[[str], str]
    renderer_probe: Callable[[], None]
    source_hashing: Callable[[Path], str]
    scene_loader: Callable[[Path], object]
    writable_parent_probe: Callable[[Path], None]


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    status: str
    code: str
    detail: str
    remediation: str


@dataclass(frozen=True)
class PreflightReport:
    passed: bool
    checks: tuple[PreflightCheck, ...]

    def by_name(self, name: str) -> PreflightCheck:
        matches = tuple(check for check in self.checks if check.name == name)
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]


def _version_runner(executable: str) -> str:
    completed = subprocess.run(
        [executable, "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("version command failed")
    first_line = completed.stdout.splitlines()
    if not first_line:
        raise ValueError("version command returned no output")
    return first_line[0]


def _renderer_probe() -> None:
    model, _, _ = load_panda_scene(DEFAULT_SCENE_PATH)
    renderer = mujoco.Renderer(model, height=16, width=16)
    renderer.close()


def _writable_parent_probe(path: Path) -> None:
    candidate = Path(path).resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.is_dir() or not os.access(candidate, os.W_OK):
        raise PermissionError("output parent is not writable")


def _default_deps() -> PreflightDeps:
    return PreflightDeps(
        executable_lookup=shutil.which,
        version_runner=_version_runner,
        renderer_probe=_renderer_probe,
        source_hashing=lambda path: sha256_file(path),
        scene_loader=lambda path: load_panda_scene(path),
        writable_parent_probe=_writable_parent_probe,
    )


def _clean(value: object) -> str:
    return redact_text(str(value), workspace=Path.cwd().resolve())


def _passed(name: str, status: str, detail: str) -> PreflightCheck:
    return PreflightCheck(name, True, status, "ok", _clean(detail), "")


def _skipped(name: str, status: str, detail: str) -> PreflightCheck:
    return PreflightCheck(name, True, status, "not_required", _clean(detail), "")


def _failed(
    name: str, code: str, error: object, remediation: str
) -> PreflightCheck:
    return PreflightCheck(
        name,
        False,
        "failed",
        code,
        _clean(error),
        _clean(remediation),
    )


def _python_check() -> PreflightCheck:
    if sys.version_info[:2] != (3, 11):
        return _failed(
            "python",
            "unsupported_python",
            f"Python {sys.version_info.major}.{sys.version_info.minor} is unsupported",
            "Use Python 3.11.",
        )
    return _passed("python", "available", "Python 3.11 is available")


def _config_check(path: Path) -> tuple[PreflightCheck, ExperimentConfig | None]:
    try:
        config = load_experiment_config(path)
    except Exception as error:
        return (
            _failed(
                "config",
                "invalid_config",
                error,
                "Provide a complete schema-v2 experiment config.",
            ),
            None,
        )
    return _passed("config", "valid", "schema-v2 config is valid"), config


def _registry_check(
    variants: tuple[str, ...], path: Path
) -> tuple[PreflightCheck, dict[str, SourceRecord] | None]:
    if "B1" not in variants:
        status = (
            "not_required_for_manual_b0"
            if variants == ("B0",)
            else "not_required_for_requested_variants"
        )
        return (
            _skipped(
                "source_registry",
                status,
                "source registry is required only for B1",
            ),
            None,
        )
    try:
        records = dict(load_source_registry(path))
    except Exception as error:
        return (
            _failed(
                "source_registry",
                "invalid_source_registry",
                error,
                "Repair configs/sources.yaml before running B1.",
            ),
            None,
        )
    return _passed("source_registry", "valid", "source registry is valid"), records


def _source_check(
    variants: tuple[str, ...],
    config: ExperimentConfig | None,
    records: dict[str, SourceRecord] | None,
    deps: PreflightDeps,
) -> PreflightCheck:
    if "B1" not in variants:
        if variants == ("B0",):
            return _skipped(
                "source_video",
                "not_required_for_manual_b0",
                "manual B0 does not access private source video",
            )
        return _skipped(
            "source_video",
            "not_required_for_requested_variants",
            "requested variants stop before source use",
        )
    if config is None:
        return _failed(
            "source_video",
            "config_unavailable",
            "source cannot be checked because config is invalid",
            "Repair the config, then rerun preflight.",
        )
    if records is None:
        return _failed(
            "source_video",
            "source_registry_unavailable",
            "source cannot be checked because the registry is invalid",
            "Repair configs/sources.yaml, then rerun preflight.",
        )
    record = records.get(config.source.id)
    if record is None:
        return _failed(
            "source_video",
            "source_id_not_registered",
            "configured source ID is not registered",
            "Add the exact source ID to configs/sources.yaml.",
        )
    try:
        if record.path != config.source.path:
            return _failed(
                "source_video",
                "source_path_mismatch",
                "config and registry source paths disagree",
                "Make the config path match configs/sources.yaml.",
            )
        if not record.path.is_file():
            return _failed(
                "source_video",
                "source_file_missing",
                "registered source file is unavailable",
                "Place the private local file at the registry-relative path.",
            )
        measured = deps.source_hashing(record.path).lower()
        if measured != record.sha256 or measured != config.source.sha256:
            return _failed(
                "source_video",
                "source_sha256_mismatch",
                "source SHA-256 does not match config and registry",
                "Verify the local file and update configs/sources.yaml only from its measured SHA-256.",
            )
    except Exception as error:
        return _failed(
            "source_video",
            "source_check_failed",
            error,
            "Verify the local source file and configs/sources.yaml.",
        )
    return _passed("source_video", "verified", "registered source hash matches")


def _tool_check(name: str, deps: PreflightDeps) -> PreflightCheck:
    try:
        executable = deps.executable_lookup(name)
        if not executable:
            raise FileNotFoundError(f"{name} executable was not found")
        version = deps.version_runner(executable)
    except Exception as error:
        return _failed(
            name,
            f"{name}_unavailable",
            error,
            f"Install {name} and make it available on PATH.",
        )
    return _passed(name, "available", version)


def _scene_check(deps: PreflightDeps) -> PreflightCheck:
    try:
        deps.scene_loader(DEFAULT_SCENE_PATH)
    except Exception as error:
        return _failed(
            "mujoco_scene",
            "mujoco_scene_unavailable",
            error,
            "Restore the packaged pinned MuJoCo scene and assets.",
        )
    return _passed("mujoco_scene", "loadable", "pinned MuJoCo scene loads")


def _renderer_check(no_render: bool, deps: PreflightDeps) -> PreflightCheck:
    if no_render:
        return _skipped("renderer", "not_required_no_render", "rendering was disabled")
    try:
        deps.renderer_probe()
    except Exception as error:
        return _failed(
            "renderer",
            "renderer_unavailable",
            error,
            "Configure a MuJoCo-compatible generated-scene renderer or use --no-render.",
        )
    return _passed("renderer", "available", "generated-scene renderer is available")


def _output_parent_check(deps: PreflightDeps) -> PreflightCheck:
    try:
        deps.writable_parent_probe(Path.cwd())
    except Exception as error:
        return _failed(
            "output_parent",
            "output_parent_not_writable",
            error,
            "Choose an output directory with a writable existing parent.",
        )
    return _passed("output_parent", "writable", "output parent is writable")


def run_preflight(
    config_path: str | Path,
    variants: Sequence[str],
    no_render: bool,
    deps: PreflightDeps | None = None,
    registry_path: str | Path = Path("configs/sources.yaml"),
) -> PreflightReport:
    """Run all checks in stable order without changing the filesystem."""

    requested = tuple(dict.fromkeys(variants))
    if not requested or any(variant not in _VARIANTS for variant in requested):
        raise ValueError("variants must contain only B0, B1, B2, B3, or B4")
    dependencies = deps or _default_deps()
    python = _python_check()
    config_check, config = _config_check(Path(config_path))
    registry_check, records = _registry_check(requested, Path(registry_path))
    checks = (
        python,
        config_check,
        registry_check,
        _source_check(requested, config, records, dependencies),
        _tool_check("ffmpeg", dependencies),
        _tool_check("ffprobe", dependencies),
        _scene_check(dependencies),
        _renderer_check(no_render, dependencies),
        _output_parent_check(dependencies),
    )
    return PreflightReport(all(check.passed for check in checks), checks)

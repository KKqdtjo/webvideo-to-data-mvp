from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Callable

import pytest
import yaml

from helpers import write_complete_config
from webvideo_to_data.preflight import PreflightDeps, run_preflight


def _passing_deps(**overrides: object) -> PreflightDeps:
    values: dict[str, object] = {
        "executable_lookup": lambda name: f"fixture-{name}",
        "version_runner": lambda executable: f"{executable} version 1",
        "renderer_probe": lambda: None,
        "source_hashing": lambda path: __import__(
            "webvideo_to_data.media", fromlist=["sha256_file"]
        ).sha256_file(path),
        "scene_loader": lambda path: object(),
        "writable_parent_probe": lambda path: None,
    }
    values.update(overrides)
    return PreflightDeps(**values)  # type: ignore[arg-type]


def _source_record(source_id: str, path: str, sha256: str) -> dict[str, object]:
    return {
        "id": source_id,
        "path": path,
        "sha256": sha256,
        "origin": "user_recorded",
        "captured_on": None,
        "captured_on_status": "not_recorded",
        "license": "private_not_redistributable",
        "publishable": False,
        "privacy_review": "local_only",
        "access": "local file required; not distributed",
    }


def _write_source_registry(tmp_path: Path, config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = config["source"]
    registry = tmp_path / "sources.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "sources": [_source_record(source["id"], source["path"], source["sha256"])],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return registry


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_b0_preflight_does_not_require_private_video(tmp_path: Path) -> None:
    config = write_complete_config(
        tmp_path,
        {"source.path": "absent-private-video.mp4", "source.sha256": "0" * 64},
        with_source=False,
    )
    report = run_preflight(
        config, variants=("B0",), no_render=False, deps=_passing_deps()
    )
    assert report.passed
    assert report.by_name("source_video").status == "not_required_for_manual_b0"


@pytest.mark.parametrize("variant", ["B2", "B3", "B4"])
def test_pre_source_variants_do_not_load_registry_or_source(
    tmp_path: Path, variant: str
) -> None:
    config = write_complete_config(
        tmp_path, {"source.sha256": "0" * 64}, with_source=False
    )

    def forbidden(_: Path) -> str:
        raise AssertionError("source was accessed")

    report = run_preflight(
        config,
        variants=(variant,),
        no_render=True,
        deps=_passing_deps(source_hashing=forbidden),
        registry_path=tmp_path / "absent-registry.yaml",
    )
    assert report.passed
    assert report.by_name("source_registry").status == "not_required_for_requested_variants"
    assert report.by_name("source_video").status == "not_required_for_requested_variants"


def test_preflight_is_read_only_even_when_checks_fail(tmp_path: Path) -> None:
    config = write_complete_config(tmp_path)
    registry_path = _write_source_registry(tmp_path, config)
    before = _snapshot_tree(tmp_path)

    def fail(label: str) -> Callable[..., object]:
        def raising(*_: object) -> object:
            raise RuntimeError(f"{label} failure")

        return raising

    report = run_preflight(
        config,
        variants=("B1",),
        no_render=False,
        deps=PreflightDeps(
            executable_lookup=fail("lookup"),
            version_runner=fail("version"),
            renderer_probe=fail("renderer"),
            source_hashing=fail("hash"),
            scene_loader=fail("scene"),
            writable_parent_probe=fail("parent"),
        ),
        registry_path=registry_path,
    )
    assert not report.passed
    assert _snapshot_tree(tmp_path) == before
    assert tuple(check.name for check in report.checks) == (
        "python",
        "config",
        "source_registry",
        "source_video",
        "ffmpeg",
        "ffprobe",
        "mujoco_scene",
        "renderer",
        "output_parent",
    )
    assert len([check for check in report.checks if not check.passed]) >= 6


def test_b1_preflight_reports_hash_mismatch_with_fix_hint(tmp_path: Path) -> None:
    config = write_complete_config(tmp_path, {"source.sha256": "0" * 64})
    registry = _write_source_registry(tmp_path, config)
    document = yaml.safe_load(registry.read_text(encoding="utf-8"))
    document["sources"][0]["sha256"] = "f" * 64
    registry.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    report = run_preflight(
        config,
        variants=("B1",),
        no_render=True,
        deps=_passing_deps(),
        registry_path=registry,
    )
    assert not report.passed
    check = report.by_name("source_video")
    assert check.code == "source_sha256_mismatch"
    assert "configs/sources.yaml" in check.remediation


def test_preflight_redacts_each_failure_before_returning_it(tmp_path: Path) -> None:
    config = write_complete_config(
        tmp_path, {"source.sha256": "0" * 64}, with_source=False
    )
    secret = "TEST_" + "PREFLIGHT_SECRET"
    raw = f"Authorization: Bearer {secret} at {tmp_path.resolve()}"

    def fail(*_: object) -> object:
        raise RuntimeError(raw)

    report = run_preflight(
        config,
        variants=("B0",),
        no_render=False,
        deps=PreflightDeps(fail, fail, fail, fail, fail, fail),
    )
    rendered = repr(report)
    assert secret not in rendered
    assert str(tmp_path.resolve()) not in rendered


def test_preflight_dependency_container_and_report_are_frozen(tmp_path: Path) -> None:
    deps = _passing_deps()
    with pytest.raises(FrozenInstanceError):
        deps.renderer_probe = lambda: None  # type: ignore[misc]
    report = run_preflight(
        write_complete_config(
            tmp_path, {"source.sha256": "0" * 64}, with_source=False
        ),
        variants=("B0",),
        no_render=True,
        deps=deps,
    )
    with pytest.raises(FrozenInstanceError):
        report.passed = False  # type: ignore[misc]


def test_default_preflight_dependencies_smoke_without_private_source(tmp_path: Path) -> None:
    report = run_preflight(
        write_complete_config(
            tmp_path, {"source.sha256": "0" * 64}, with_source=False
        ),
        variants=("B0",),
        no_render=True,
    )
    assert report.passed, report


def test_repeated_b0_variants_have_stable_set_semantics(tmp_path: Path) -> None:
    config = write_complete_config(
        tmp_path, {"source.sha256": "0" * 64}, with_source=False
    )
    report = run_preflight(
        config,
        variants=("B0", "B0"),
        no_render=True,
        deps=_passing_deps(),
    )
    assert report.by_name("source_registry").status == "not_required_for_manual_b0"
    assert report.by_name("source_video").status == "not_required_for_manual_b0"

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import pytest
import yaml

import webvideo_to_data.cli as cli_module
import webvideo_to_data.experiment as experiment_module
import webvideo_to_data.dashboard as dashboard_module
import webvideo_to_data.redaction as redaction_module
import webvideo_to_data.suite as suite_module
from webvideo_to_data.cli import main
from webvideo_to_data.dashboard import (
    copy_public_preview,
    generate_dashboard,
    write_dashboard_copy,
)
from webvideo_to_data.experiment import run_experiment
from webvideo_to_data.media import sha256_file
from webvideo_to_data.redaction import audit_publication_tree
from webvideo_to_data.suite import (
    RolloutRecord,
    SuiteDeps,
    run_suite,
    summarize_b0,
    verify_suite_directory,
)
from tests.helpers import write_complete_config


FIXED_RUN_ID = "20260819T120102123456Z-a1b2c3d4-7f29"


def _make_directory_link(link: Path, target: Path) -> None:
    if os.name != "nt":
        link.symlink_to(target, target_is_directory=True)
        return
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def _remove_directory_link(link: Path) -> None:
    if os.name == "nt":
        link.rmdir()
    else:
        link.unlink()


def _extended_windows_alias(path: Path) -> Path:
    return Path("\\\\?\\" + str(path.resolve()))


def _snapshot_tree(root: Path) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (path.relative_to(root).as_posix(), path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def _write_dashboard_config(tmp_path: Path) -> Path:
    config_path = write_complete_config(tmp_path)
    source = tmp_path / "moving.avi"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 72)
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
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["source"]["sha256"] = sha256_file(source)
    config["tracking"]["minimum_valid_ratio"] = 0.1
    config["simulation"]["render_size"] = [96, 72]
    config["simulation"]["render_every"] = 20
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _write_dashboard_input_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Build real v4 children, so dashboard assertions exercise trusted manifests."""

    config = _write_dashboard_config(tmp_path)
    run = tmp_path / "dashboard-input"
    run.mkdir()
    monkeypatch.setattr(
        experiment_module,
        "_git_generator_provenance",
        lambda: {"git_commit": "1" * 40, "git_dirty": False},
    )
    variants: dict[str, dict[str, Any]] = {}
    for variant in ("B0", "B1", "B2", "B3", "B4"):
        destination = run / variant
        metrics = run_experiment(
            config,
            destination,
            variant=variant,  # type: ignore[arg-type]
            no_render=variant not in {"B0", "B1"},
        )
        manifest_bytes = (destination / "run_manifest.json").read_bytes()
        variants[variant] = {
            "status": metrics["status"],
            "reason": metrics["reason"],
            "physics_validation": (
                "passed" if variant == "B0" else metrics["physics_validation"]
            ),
            "run_manifest_sha256": sha256(manifest_bytes).hexdigest(),
        }
    (run / "suite-metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "SYNTHETIC",
                "run_id": FIXED_RUN_ID,
                "status": "recorded",
                "reason": "suite_recorded",
                "requested_variants": ["B0", "B1", "B2", "B3", "B4"],
                "variants": variants,
                "b0_physics_baseline": "passed",
                "b0_rollouts": 30,
                "b0_successes": 0,
                "actions_exported": 0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "environment.json").write_text(
        json.dumps({"generator_commit": "2" * 40}) + "\n", encoding="utf-8"
    )
    return run


@pytest.mark.requires_renderer
def test_dashboard_leads_with_action_outcome_and_uses_relative_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Private comparison frames or local paths must never become dashboard links."""

    run = _write_dashboard_input_fixture(tmp_path, monkeypatch)

    output = dashboard_module._build_dashboard(run)
    html = output.read_text(encoding="utf-8")

    assert "NO ACTION EXPORTED · 0 / 5 eligible" in html
    assert "B0 manual physics baseline" in html
    assert "B1 kinematic diagnostic" in html
    assert "availability != semantic accuracy" in html
    assert "INPUT COMMIT · " + "1" * 12 in html
    assert "GENERATOR COMMIT · " + "2" * 12 in html
    assert "ARTIFACTS VERIFIED" in html
    assert "C:\\Users" not in html
    assert "file://" not in html
    assert "../B0/mujoco_replay.mp4" in html
    assert "../B1/side_by_side.mp4" not in html
    assert "private local media omitted" in html


@pytest.mark.requires_renderer
def test_unfinalized_dashboard_does_not_trust_preexisting_preview_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_dashboard_input_fixture(tmp_path, monkeypatch)
    preview = run / "dashboard" / "media" / "B1-preview.gif"
    preview.parent.mkdir(parents=True)
    imageio.mimsave(
        preview,
        [np.full((24, 24, 3), 180, dtype=np.uint8)],
        format="GIF",
        duration=125,
    )

    html = dashboard_module._build_dashboard(run).read_text(encoding="utf-8")

    assert 'src="media/B1-preview.gif"' not in html
    assert "../B1/mujoco_replay.mp4" in html


def _write_verified_suite_fixture(tmp_path: Path, *, variants: tuple[str, ...]) -> Path:
    config = _write_dashboard_config(tmp_path)
    result = run_suite(
        config,
        tmp_path / "artifacts",
        variants=variants,  # type: ignore[arg-type]
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_quick_suite_deps(),
    )
    return result.run_dir


def _quick_suite_deps() -> SuiteDeps:
    records = tuple(
        RolloutRecord(
            seed=seed,
            perturbation={
                "can_dx_m": 0.0,
                "can_dy_m": 0.0,
                "can_yaw_rad": 0.0,
                "mass_scale": 1.0,
                "friction_scale": 1.0,
            },
            passed=False,
            failed_checks=("forbidden_contact_count",),
            execution_tracking_ratio=0.13,
            maximum_lift_m=0.20,
            target_error_m=0.01,
            final_tilt_rad=0.0,
            final_linear_speed_m_s=0.0,
            forbidden_contact_count=1,
            maximum_forbidden_penetration_m=0.003,
        )
        for seed in range(19, 49)
    )
    return SuiteDeps(
        now_utc=lambda: datetime(2026, 8, 19, tzinfo=timezone.utc),
        random_suffix=lambda: "7f29",
        run_variant=lambda config, destination, variant, no_render: run_experiment(
            config.config_path,
            destination,
            variant=variant,
            no_render=no_render,
        ),
        evaluate_b0=lambda config, seeds: summarize_b0(records),
    )


def test_dashboard_cli_verifies_run_and_returns_output_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The finalized-suite dashboard command must be byte-for-byte read-only."""

    run = _write_verified_suite_fixture(tmp_path, variants=("B2",))
    before = _snapshot_tree(run)

    assert main(("dashboard", "--run", str(run), "--json")) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["dashboard_path"] == "dashboard/index.html"
    assert _snapshot_tree(run) == before


def test_finalized_dashboard_returns_location_path(tmp_path: Path) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B2",))
    dashboard_path = run / "dashboard" / "index.html"

    dashboard = generate_dashboard(run)

    assert isinstance(dashboard, Path)
    assert dashboard == dashboard_path


def test_generate_dashboard_rejects_unfinalized_input_without_rewriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_dashboard_input_fixture(tmp_path, monkeypatch)
    output = run / "dashboard" / "index.html"
    output.parent.mkdir()
    sentinel = b"private draft must not be rewritten or published"
    output.write_bytes(sentinel)

    with pytest.raises(ValueError):
        generate_dashboard(run)

    assert output.read_bytes() == sentinel


def test_dashboard_cli_output_is_detached_readable_and_has_no_broken_media_links(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))
    _install_public_preview(run)
    before = _snapshot_tree(run)
    output = tmp_path / "export" / "detached.html"

    assert (
        main(
            (
                "dashboard",
                "--run",
                str(run),
                "--output",
                str(output),
                "--json",
            )
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    html = output.read_text(encoding="utf-8")

    assert payload["dashboard_path"] == "<external-output>/detached.html"
    assert html.startswith("<!doctype html>")
    assert "NO ACTION EXPORTED · 0 / 5 eligible" in html
    assert "REJECTED — NOT ACTION DATA" in html
    assert "media omitted from detached dashboard copy" in html
    assert "<img " not in html and "<video " not in html
    assert "src=" not in html and "../" not in html and "media/" not in html
    assert "C:\\Users" not in html and "file://" not in html
    assert audit_publication_tree(output.parent) == ()
    assert _snapshot_tree(run) == before


def test_dashboard_public_copies_reject_relative_outputs_inside_relative_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))
    _install_public_preview(run)
    manifest = run / "suite-manifest.json"
    original_manifest = manifest.read_bytes()
    relative_run = run.relative_to(tmp_path)
    relative_output = relative_run / "suite-manifest.json"
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="outside the verified suite"):
        write_dashboard_copy(relative_run, relative_output)
    with pytest.raises(ValueError, match="outside the verified suite"):
        copy_public_preview(relative_run, "B0", relative_output)

    assert manifest.read_bytes() == original_manifest


def test_dashboard_cli_rejects_relative_output_inside_relative_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B2",))
    manifest = run / "suite-manifest.json"
    original_manifest = manifest.read_bytes()
    relative_run = run.relative_to(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert (
        main(
            (
                "dashboard",
                "--run",
                str(relative_run),
                "--output",
                str(relative_run / "suite-manifest.json"),
                "--json",
            )
        )
        == 4
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "verification_failed"
    assert manifest.read_bytes() == original_manifest


def test_dashboard_public_copies_accept_external_relative_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))
    expected_preview = _install_public_preview(run)
    relative_run = run.relative_to(tmp_path)
    monkeypatch.chdir(tmp_path)

    dashboard = write_dashboard_copy(relative_run, "exports/dashboard.html")
    preview = copy_public_preview(relative_run, "B0", "exports/preview.gif")

    assert dashboard == (tmp_path / "exports" / "dashboard.html")
    assert dashboard.read_bytes().startswith(b"<!doctype html>")
    assert preview == (tmp_path / "exports" / "preview.gif")
    assert preview.read_bytes() == expected_preview


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize("operation", ("dashboard", "preview"))
@pytest.mark.parametrize(
    "extended_suite",
    (False, True),
    ids=("standard-suite-extended-output", "extended-suite-standard-output"),
)
def test_dashboard_public_copies_reject_equivalent_windows_alias_inside_suite(
    tmp_path: Path, operation: str, extended_suite: bool
) -> None:
    """Catch DOS/extended aliases overwriting a verified suite manifest."""

    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))
    _install_public_preview(run)
    manifest = run / "suite-manifest.json"
    original_manifest = manifest.read_bytes()
    source = _extended_windows_alias(run) if extended_suite else run
    output = manifest if extended_suite else _extended_windows_alias(manifest)

    with pytest.raises(ValueError, match="outside the verified suite"):
        if operation == "dashboard":
            write_dashboard_copy(source, output)
        else:
            copy_public_preview(source, "B0", output)

    assert manifest.read_bytes() == original_manifest


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize(
    "extended_suite",
    (False, True),
    ids=("standard-suite-extended-output", "extended-suite-standard-output"),
)
def test_dashboard_cli_rejects_equivalent_windows_alias_inside_suite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extended_suite: bool,
) -> None:
    """Catch the CLI overwriting a manifest through an equivalent path alias."""

    run = _write_verified_suite_fixture(tmp_path, variants=("B2",))
    manifest = run / "suite-manifest.json"
    original_manifest = manifest.read_bytes()
    source = _extended_windows_alias(run) if extended_suite else run
    output = manifest if extended_suite else _extended_windows_alias(manifest)

    assert (
        main(
            (
                "dashboard",
                "--run",
                str(source),
                "--output",
                str(output),
                "--json",
            )
        )
        == 4
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "verification_failed"
    assert manifest.read_bytes() == original_manifest


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize(
    "unsafe_output",
    (
        r"\\.\C:\Artifacts\dashboard.html",
        r"\\?\GLOBALROOT\Device\HarddiskVolume1\dashboard.html",
        r"\\?\Volume{00000000-0000-0000-0000-000000000000}\dashboard.html",
        r"\\?\PIPE\webvideo-to-data",
        r"\\Server\mailslot\webvideo-to-data",
        r"\\?\UNC\Server\IPC$\webvideo-to-data",
    ),
)
def test_dashboard_rejects_unsafe_output_namespace_before_filesystem_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_output: str,
) -> None:
    """Catch device/IPC namespaces reaching any link or resolution probe."""

    suite = tmp_path / "suite"
    suite.mkdir()

    def fail_if_filesystem_is_inspected(path: Path) -> bool:
        del path
        raise AssertionError("unsafe output reached filesystem inspection")

    monkeypatch.setattr(
        dashboard_module, "_is_link_or_junction", fail_if_filesystem_is_inspected
    )

    with pytest.raises(ValueError, match="unsafe dashboard copy output path namespace"):
        dashboard_module._external_output_path(
            suite, Path(unsafe_output), "dashboard copy"
        )


_UNSAFE_WINDOWS_DASHBOARD_PATHS = (
    r"\\.\C:\Artifacts\suite",
    r"\??\C:\Artifacts\suite",
    r"\\??\C:\Artifacts\suite",
    r"\\?\GLOBALROOT\Device\HarddiskVolume1\suite",
    r"\\?\Volume{00000000-0000-0000-0000-000000000000}\suite",
    r"\\?\PIPE\webvideo-to-data",
    r"\\Server\mailslot\webvideo-to-data",
    r"\\?\UNC\Server\IPC$\webvideo-to-data",
)


def _valid_windows_dashboard_path(
    tmp_path: Path, spelling: str, leaf: str
) -> Path:
    if spelling == "dos":
        return tmp_path / leaf
    if spelling == "extended-dos":
        return Path("\\\\?\\" + str(tmp_path / leaf))
    if spelling == "unc":
        return Path(r"\\Server\Share") / leaf
    if spelling == "extended-unc":
        return Path(r"\\?\UNC\Server\Share") / leaf
    raise AssertionError(f"unknown spelling: {spelling}")


def _windows_device_component_path(
    tmp_path: Path, spelling: str, component: str
) -> Path:
    if spelling == "relative":
        return Path("ordinary") / component / "suite"
    if spelling == "dos":
        return tmp_path / "ordinary" / component / "suite"
    if spelling == "extended-dos":
        return Path("\\\\?\\" + str(tmp_path / "ordinary" / component / "suite"))
    if spelling == "unc":
        return Path(r"\\Server\Share\ordinary") / component / "suite"
    if spelling == "extended-unc":
        return Path(r"\\?\UNC\Server\Share\ordinary") / component / "suite"
    raise AssertionError(f"unknown spelling: {spelling}")


_WINDOWS_DEVICE_COMPONENT_CASES = (
    ("relative", "NUL.txt"),
    ("dos", "cOn "),
    ("extended-dos", "LPT³:stream"),
    ("unc", "CONOUT$.txt"),
    ("extended-unc", "CLOCK$."),
)


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize("operation", ("generate", "dashboard-copy", "preview-copy"))
@pytest.mark.parametrize("unsafe_run", _UNSAFE_WINDOWS_DASHBOARD_PATHS)
def test_dashboard_public_apis_reject_unsafe_run_before_suite_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    unsafe_run: str,
) -> None:
    """Catch caller-controlled namespaces reaching suite snapshot inspection."""

    @contextmanager
    def fail_if_suite_is_inspected(path: str | Path) -> Any:
        del path
        raise AssertionError("unsafe run reached suite capability")
        yield

    monkeypatch.setattr(
        suite_module, "_verified_suite_capability", fail_if_suite_is_inspected
    )

    with pytest.raises(ValueError, match="unsafe verified suite path namespace"):
        if operation == "generate":
            generate_dashboard(unsafe_run)
        elif operation == "dashboard-copy":
            write_dashboard_copy(unsafe_run, tmp_path / "dashboard.html")
        else:
            copy_public_preview(unsafe_run, "B0", tmp_path / "preview.gif")


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize("operation", ("dashboard-copy", "preview-copy"))
@pytest.mark.parametrize("unsafe_output", _UNSAFE_WINDOWS_DASHBOARD_PATHS)
def test_dashboard_public_copy_rejects_unsafe_output_before_suite_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    unsafe_output: str,
) -> None:
    """Catch caller-controlled outputs reaching source snapshot inspection."""

    @contextmanager
    def fail_if_suite_is_inspected(path: str | Path) -> Any:
        del path
        raise AssertionError("unsafe output reached suite capability")
        yield

    monkeypatch.setattr(
        suite_module, "_verified_suite_capability", fail_if_suite_is_inspected
    )

    description = "dashboard copy" if operation == "dashboard-copy" else "public preview"
    with pytest.raises(ValueError, match=rf"unsafe {description} output path namespace"):
        if operation == "dashboard-copy":
            write_dashboard_copy(tmp_path / "suite", unsafe_output)
        else:
            copy_public_preview(tmp_path / "suite", "B0", unsafe_output)


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize("argument", ("run", "output"))
@pytest.mark.parametrize("unsafe_path", _UNSAFE_WINDOWS_DASHBOARD_PATHS)
def test_dashboard_cli_rejects_unsafe_namespace_before_dashboard_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argument: str,
    unsafe_path: str,
) -> None:
    """Catch the CLI delegating an unvalidated namespace to a public API."""

    def fail_if_dashboard_api_is_called(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        raise AssertionError("unsafe CLI path reached dashboard API")

    monkeypatch.setattr(cli_module, "generate_dashboard", fail_if_dashboard_api_is_called)
    monkeypatch.setattr(
        cli_module, "write_dashboard_copy", fail_if_dashboard_api_is_called
    )
    run = unsafe_path if argument == "run" else str(tmp_path / "suite")
    command = ["dashboard", "--run", run, "--json"]
    if argument == "output":
        command.extend(("--output", unsafe_path))

    assert main(tuple(command)) == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "verification_failed"
    assert "unsafe" in payload["detail"] and "path namespace" in payload["detail"]


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize("operation", ("generate", "dashboard-copy", "preview-copy"))
@pytest.mark.parametrize(
    "spelling", ("dos", "extended-dos", "unc", "extended-unc")
)
def test_dashboard_public_apis_allow_filesystem_run_namespaces_to_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    spelling: str,
) -> None:
    """Catch the lexical whitelist rejecting a supported filesystem spelling."""

    class ReachedCapability(Exception):
        pass

    source = _valid_windows_dashboard_path(tmp_path, spelling, "suite")
    seen: list[Path] = []

    @contextmanager
    def record_suite_capability(path: str | Path) -> Any:
        seen.append(Path(path))
        raise ReachedCapability
        yield

    monkeypatch.setattr(
        suite_module, "_verified_suite_capability", record_suite_capability
    )

    with pytest.raises(ReachedCapability):
        if operation == "generate":
            generate_dashboard(source)
        elif operation == "dashboard-copy":
            write_dashboard_copy(source, tmp_path / "dashboard.html")
        else:
            copy_public_preview(source, "B0", tmp_path / "preview.gif")

    assert seen == [source]


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize("operation", ("dashboard-copy", "preview-copy"))
@pytest.mark.parametrize(
    "spelling", ("dos", "extended-dos", "unc", "extended-unc")
)
def test_dashboard_public_copies_allow_filesystem_output_namespaces_to_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    spelling: str,
) -> None:
    """Catch output preflight rejecting a supported filesystem spelling."""

    class ReachedCapability(Exception):
        pass

    output = _valid_windows_dashboard_path(tmp_path, spelling, "public.html")

    @contextmanager
    def record_suite_capability(path: str | Path) -> Any:
        del path
        raise ReachedCapability
        yield

    monkeypatch.setattr(
        suite_module, "_verified_suite_capability", record_suite_capability
    )

    with pytest.raises(ReachedCapability):
        if operation == "dashboard-copy":
            write_dashboard_copy(tmp_path / "suite", output)
        else:
            copy_public_preview(tmp_path / "suite", "B0", output)


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize("argument", ("run", "output"))
@pytest.mark.parametrize(
    "spelling", ("dos", "extended-dos", "unc", "extended-unc")
)
def test_dashboard_cli_allows_filesystem_namespaces_to_public_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argument: str,
    spelling: str,
) -> None:
    """Catch CLI preflight rejecting a supported filesystem spelling."""

    candidate = _valid_windows_dashboard_path(tmp_path, spelling, "suite")
    seen: list[tuple[Path, ...]] = []

    def record_generate(run: str | Path) -> Path:
        seen.append((Path(run),))
        return tmp_path / "unused.html"

    def record_copy(run: str | Path, output: str | Path) -> Path:
        seen.append((Path(run), Path(output)))
        return tmp_path / "result.html"

    monkeypatch.setattr(cli_module, "generate_dashboard", record_generate)
    monkeypatch.setattr(cli_module, "write_dashboard_copy", record_copy)
    run = candidate if argument == "run" else tmp_path / "suite"
    command = ["dashboard", "--run", str(run), "--json"]
    if argument == "output":
        command.extend(("--output", str(candidate)))

    assert main(tuple(command)) == 0
    json.loads(capsys.readouterr().out)
    if argument == "run":
        assert seen == [(candidate,)]
    else:
        assert seen == [(tmp_path / "suite", candidate)]


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize("operation", ("generate", "dashboard-copy", "preview-copy"))
@pytest.mark.parametrize("spelling,component", _WINDOWS_DEVICE_COMPONENT_CASES)
def test_dashboard_public_apis_reject_device_run_before_filesystem_or_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    spelling: str,
    component: str,
) -> None:
    """Catch a DOS device component reaching source metadata or snapshot APIs."""

    source = _windows_device_component_path(tmp_path, spelling, component)

    @contextmanager
    def fail_if_suite_is_inspected(path: str | Path) -> Any:
        del path
        raise AssertionError("device run reached suite capability")
        yield

    def fail_if_filesystem_is_inspected(path: Path) -> bool:
        del path
        raise AssertionError("device run reached filesystem inspection")

    monkeypatch.setattr(
        suite_module, "_verified_suite_capability", fail_if_suite_is_inspected
    )
    monkeypatch.setattr(
        dashboard_module, "_is_link_or_junction", fail_if_filesystem_is_inspected
    )

    with pytest.raises(ValueError, match="unsafe verified suite path device alias"):
        if operation == "generate":
            generate_dashboard(source)
        elif operation == "dashboard-copy":
            write_dashboard_copy(source, tmp_path / "dashboard.html")
        else:
            copy_public_preview(source, "B0", tmp_path / "preview.gif")


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize("operation", ("dashboard-copy", "preview-copy"))
@pytest.mark.parametrize("spelling,component", _WINDOWS_DEVICE_COMPONENT_CASES)
def test_dashboard_public_copies_reject_device_output_before_source_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    spelling: str,
    component: str,
) -> None:
    """Catch a DOS device output reaching source snapshot or destination metadata."""

    output = _windows_device_component_path(tmp_path, spelling, component)

    @contextmanager
    def fail_if_suite_is_inspected(path: str | Path) -> Any:
        del path
        raise AssertionError("device output reached suite capability")
        yield

    def fail_if_filesystem_is_inspected(path: Path) -> bool:
        del path
        raise AssertionError("device output reached filesystem inspection")

    monkeypatch.setattr(
        suite_module, "_verified_suite_capability", fail_if_suite_is_inspected
    )
    monkeypatch.setattr(
        dashboard_module, "_is_link_or_junction", fail_if_filesystem_is_inspected
    )

    description = "dashboard copy" if operation == "dashboard-copy" else "public preview"
    with pytest.raises(ValueError, match=rf"unsafe {description} output path device alias"):
        if operation == "dashboard-copy":
            write_dashboard_copy(tmp_path / "suite", output)
        else:
            copy_public_preview(tmp_path / "suite", "B0", output)


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize("argument", ("run", "output"))
@pytest.mark.parametrize("spelling,component", _WINDOWS_DEVICE_COMPONENT_CASES)
def test_dashboard_cli_rejects_device_component_before_dashboard_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argument: str,
    spelling: str,
    component: str,
) -> None:
    """Catch CLI device aliases reaching either dashboard public API."""

    def fail_if_dashboard_api_is_called(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        raise AssertionError("device CLI path reached dashboard API")

    monkeypatch.setattr(cli_module, "generate_dashboard", fail_if_dashboard_api_is_called)
    monkeypatch.setattr(
        cli_module, "write_dashboard_copy", fail_if_dashboard_api_is_called
    )
    candidate = _windows_device_component_path(tmp_path, spelling, component)
    run = candidate if argument == "run" else tmp_path / "suite"
    command = ["dashboard", "--run", str(run), "--json"]
    if argument == "output":
        command.extend(("--output", str(candidate)))

    assert main(tuple(command)) == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "verification_failed"
    assert "path device alias" in payload["detail"]


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize(
    "source",
    (
        Path(r"\\Server\NUL\ordinary\suite"),
        Path(r"\\?\UNC\Server\COM1\ordinary\suite"),
    ),
)
def test_dashboard_api_allows_reserved_looking_unc_share_to_capability(
    monkeypatch: pytest.MonkeyPatch, source: Path
) -> None:
    """Catch the public API treating a UNC share root as an object component."""

    class ReachedCapability(Exception):
        pass

    @contextmanager
    def record_suite_capability(path: str | Path) -> Any:
        assert Path(path) == source
        raise ReachedCapability
        yield

    monkeypatch.setattr(
        suite_module, "_verified_suite_capability", record_suite_capability
    )

    with pytest.raises(ReachedCapability):
        generate_dashboard(source)


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize(
    "source",
    (
        Path(r"\\Server\NUL\ordinary\suite"),
        Path(r"\\?\UNC\Server\COM1\ordinary\suite"),
    ),
)
def test_dashboard_cli_allows_reserved_looking_unc_share_to_api(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    source: Path,
) -> None:
    """Catch CLI preflight rejecting a legal reserved-looking UNC share root."""

    seen: list[Path] = []

    def record_generate(run: str | Path) -> Path:
        seen.append(Path(run))
        return Path(run) / "dashboard" / "index.html"

    monkeypatch.setattr(cli_module, "generate_dashboard", record_generate)

    assert main(("dashboard", "--run", str(source), "--json")) == 0
    json.loads(capsys.readouterr().out)
    assert seen == [source]


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize(
    "source",
    (
        Path(r"\\Server\NUL\ordinary\NUL.txt\suite"),
        Path(r"\\?\UNC\Server\COM1\ordinary\COM1:stream\suite"),
    ),
)
def test_dashboard_api_and_cli_reject_unc_object_after_share_before_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    source: Path,
) -> None:
    """Catch share skipping extending into later attacker-controlled objects."""

    @contextmanager
    def fail_if_suite_is_inspected(path: str | Path) -> Any:
        del path
        raise AssertionError("UNC device object reached suite capability")
        yield

    def fail_if_dashboard_api_is_called(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        raise AssertionError("UNC device object reached dashboard API")

    monkeypatch.setattr(
        suite_module, "_verified_suite_capability", fail_if_suite_is_inspected
    )
    with pytest.raises(ValueError, match="unsafe verified suite path device alias"):
        generate_dashboard(source)

    monkeypatch.setattr(cli_module, "generate_dashboard", fail_if_dashboard_api_is_called)
    assert main(("dashboard", "--run", str(source), "--json")) == 4
    payload = json.loads(capsys.readouterr().out)
    assert "path device alias" in payload["detail"]


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
def test_dashboard_public_copies_accept_extended_external_output_aliases(
    tmp_path: Path,
) -> None:
    """Catch canonicalization accidentally rejecting a safe extended DOS alias."""

    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))
    expected_preview = _install_public_preview(run)
    export_root = tmp_path / "extended-exports"
    export_root.mkdir()
    dashboard_path = export_root / "dashboard.html"
    preview_path = export_root / "preview.gif"

    dashboard = write_dashboard_copy(run, _extended_windows_alias(dashboard_path))
    preview = copy_public_preview(run, "B0", _extended_windows_alias(preview_path))

    assert dashboard_path.read_bytes().startswith(b"<!doctype html>")
    assert preview_path.read_bytes() == expected_preview
    assert dashboard == _extended_windows_alias(dashboard_path)
    assert preview == _extended_windows_alias(preview_path)


def test_detached_dashboard_uses_verified_snapshot_after_source_files_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))
    original_verify = suite_module.verify_suite_directory
    private_path = "C:" + "\\Users\\Alice\\private-source.mp4"

    def mutate_after_verification(path: str | Path) -> Any:
        verified = original_verify(path)
        environment_path = run / "environment.json"
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        environment["generator_commit"] = private_path
        environment_path.write_text(json.dumps(environment), encoding="utf-8")
        provenance_path = run / "variants" / "B0" / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["generator"]["git_commit"] = private_path
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        return verified

    monkeypatch.setattr(
        suite_module, "verify_suite_directory", mutate_after_verification
    )
    output_root = tmp_path / "detached"
    output = write_dashboard_copy(run, output_root / "dashboard.html")
    html = output.read_text(encoding="utf-8")

    assert private_path not in html
    assert "C:\\Users" not in html
    assert audit_publication_tree(output_root) == ()


def test_detached_dashboard_audits_rendered_private_value_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))
    output = tmp_path / "detached" / "dashboard.html"
    output.parent.mkdir()
    output.write_bytes(b"existing public dashboard")
    private_path = "C:" + "\\Users\\Alice\\private-source.mp4"
    original_render = dashboard_module._dashboard_html

    def render_with_dynamic_private_value(input_data: Any) -> str:
        return original_render(input_data).replace(
            "</body>", f"<p>{private_path}</p></body>"
        )

    monkeypatch.setattr(
        dashboard_module, "_dashboard_html", render_with_dynamic_private_value
    )

    with pytest.raises(ValueError, match="privacy audit"):
        write_dashboard_copy(run, output)

    assert output.read_bytes() == b"existing public dashboard"
    assert not list(output.parent.glob("*.tmp"))


def test_detached_dashboard_leaves_existing_output_on_audit_parser_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))
    output = tmp_path / "detached" / "dashboard.html"
    output.parent.mkdir()
    output.write_bytes(b"existing public dashboard")

    def fail_audit(*args: object, **kwargs: object) -> tuple[object, ...]:
        raise ValueError("audit parser failed")

    monkeypatch.setattr(
        redaction_module, "audit_publication_bytes", fail_audit, raising=False
    )

    with pytest.raises(ValueError, match="audit parser failed"):
        write_dashboard_copy(run, output)

    assert output.read_bytes() == b"existing public dashboard"
    assert not list(output.parent.glob("*.tmp"))


def test_dashboard_cli_rejects_unverified_run(tmp_path: Path) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B2",))
    (run / "suite-metrics.json").write_text("{}", encoding="utf-8")

    assert main(("dashboard", "--run", str(run))) == 4


def test_task8_suite_includes_dashboard_and_remains_actionless(tmp_path: Path) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "enhanced",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_quick_suite_deps(),
    )

    verified = verify_suite_directory(result.run_dir)

    assert verified.manifest["feature_set"] == ["core", "dashboard"]
    assert "dashboard/index.html" in verified.manifest["files"]
    assert verified.metrics["actions_exported"] == 0
    assert not list(result.run_dir.rglob("actions.npz"))
    html = (result.run_dir / "dashboard" / "index.html").read_text(encoding="utf-8")
    assert html.count("media not generated (variant not requested)") == 4
    assert html.count("media not generated (--no-render)") == 1


def test_public_preview_copy_reports_unrequested_variant(tmp_path: Path) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))

    with pytest.raises(ValueError, match="B1 preview is not present in verified suite"):
        copy_public_preview(run, "B1", tmp_path / "public.gif")


def _install_public_preview(run: Path, variant: str = "B0") -> bytes:
    preview = run / "dashboard" / "media" / f"{variant}-preview.gif"
    preview.parent.mkdir(exist_ok=True)
    imageio.mimsave(
        preview,
        [
            np.full((24, 32, 3), 20, dtype=np.uint8),
            np.full((24, 32, 3), 180, dtype=np.uint8),
        ],
        format="GIF",
        duration=125,
        loop=0,
    )
    content = preview.read_bytes()
    manifest_path = run / "suite-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][f"dashboard/media/{variant}-preview.gif"] = {
        "size": len(content),
        "sha256": sha256(content).hexdigest(),
        "media_role": "public_simulation_preview",
        "contains_private_source_frames": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    verify_suite_directory(run)
    return content


@pytest.mark.parametrize("link_kind", ["root", "ancestor"])
def test_dashboard_public_entries_reject_linked_source_ancestry(
    tmp_path: Path, link_kind: str
) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))
    _install_public_preview(run)
    if link_kind == "root":
        linked = tmp_path / "linked-suite"
        _make_directory_link(linked, run)
        source = linked
    else:
        linked = tmp_path / "linked-parent"
        _make_directory_link(linked, run.parent)
        source = linked / run.name
    try:
        with pytest.raises(ValueError):
            generate_dashboard(source)
        with pytest.raises(ValueError):
            write_dashboard_copy(source, tmp_path / f"{link_kind}-dashboard.html")
        with pytest.raises(ValueError):
            copy_public_preview(source, "B0", tmp_path / f"{link_kind}-preview.gif")
    finally:
        _remove_directory_link(linked)


def test_public_preview_copy_writes_bytes_verified_before_source_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))
    expected = _install_public_preview(run)
    source = run / "dashboard" / "media" / "B0-preview.gif"
    original_write = dashboard_module._write_bytes_atomic
    write_calls = 0

    def swap_before_captured_write(path: Path, content: bytes) -> Path:
        nonlocal write_calls
        write_calls += 1
        source.write_bytes(b"source changed after capability closed")
        return original_write(path, content)

    monkeypatch.setattr(
        dashboard_module, "_write_bytes_atomic", swap_before_captured_write
    )
    output = copy_public_preview(run, "B0", tmp_path / "export" / "public.gif")

    assert write_calls == 1
    assert output.read_bytes() == expected
    assert source.read_bytes() == b"source changed after capability closed"
    assert not list(output.parent.glob("*.tmp"))


def test_public_preview_copy_refuses_existing_symlink_destination(
    tmp_path: Path,
) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))
    _install_public_preview(run)
    target = tmp_path / "target.gif"
    target.write_bytes(b"do not overwrite")
    link = tmp_path / "public.gif"
    try:
        os.symlink(target, link)
    except OSError as error:
        pytest.skip(f"file symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="link|junction"):
        copy_public_preview(run, "B0", link)

    assert target.read_bytes() == b"do not overwrite"


def test_public_preview_copy_refuses_linked_destination_directory(
    tmp_path: Path,
) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))
    _install_public_preview(run)
    target = tmp_path / "external-target"
    target.mkdir()
    linked = tmp_path / "linked-output"
    _make_directory_link(linked, target)
    try:
        with pytest.raises(ValueError, match="links and junctions"):
            copy_public_preview(run, "B0", linked / "public.gif")
    finally:
        _remove_directory_link(linked)

    assert not (target / "public.gif").exists()


def test_stable_posix_parent_normalizes_not_a_directory_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_open = os.open
    open_calls = 0

    def open_after_parent_swap(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal open_calls
        open_calls += 1
        if dir_fd is None:
            return original_open(os.devnull, os.O_RDONLY)
        raise NotADirectoryError("output parent was replaced")

    monkeypatch.setattr(os, "O_DIRECTORY", 0, raising=False)
    monkeypatch.setattr(os, "O_NOFOLLOW", 0, raising=False)
    monkeypatch.setattr(os, "open", open_after_parent_swap)

    with pytest.raises(ValueError, match="link|junction|reparse") as error:
        with dashboard_module._stable_posix_parent(
            tmp_path / "publication" / "public.gif"
        ):
            pytest.fail("a swapped output parent must not be opened")

    assert isinstance(error.value.__cause__, NotADirectoryError)
    assert open_calls == 2


def test_stable_posix_parent_retries_after_concurrent_directory_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = os.open
    relative_open_calls = 0
    mkdir_calls = 0

    def open_during_concurrent_creation(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal relative_open_calls
        if dir_fd is None:
            return original_open(os.devnull, os.O_RDONLY)
        relative_open_calls += 1
        if relative_open_calls == 1:
            raise FileNotFoundError("output parent is not present yet")
        return original_open(os.devnull, os.O_RDONLY)

    def directory_created_by_competitor(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal mkdir_calls
        mkdir_calls += 1
        raise FileExistsError("another process created the directory")

    monkeypatch.setattr(os, "O_DIRECTORY", 0, raising=False)
    monkeypatch.setattr(os, "O_NOFOLLOW", 0, raising=False)
    monkeypatch.setattr(os, "open", open_during_concurrent_creation)
    monkeypatch.setattr(os, "mkdir", directory_created_by_competitor)

    with dashboard_module._stable_posix_parent(Path("/publication/public.gif")):
        pass

    assert relative_open_calls == 2
    assert mkdir_calls == 1


def test_public_preview_copy_rejects_parent_swapped_to_junction_after_path_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))
    _install_public_preview(run)
    parent = tmp_path / "publication"
    parent.mkdir()
    backup = tmp_path / "publication-original"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "public.gif"
    outside_file.write_bytes(b"outside sentinel")
    original_validate = dashboard_module._external_output_path
    swapped = False

    def swap_after_path_check(
        suite_directory: Path, output_path: str | Path, description: str
    ) -> Path:
        nonlocal swapped
        output = original_validate(suite_directory, output_path, description)
        parent.rename(backup)
        _make_directory_link(parent, outside)
        swapped = True
        return output

    monkeypatch.setattr(
        dashboard_module, "_external_output_path", swap_after_path_check
    )
    try:
        with pytest.raises(ValueError, match="link|junction|reparse"):
            copy_public_preview(run, "B0", parent / "public.gif")
    finally:
        if swapped:
            _remove_directory_link(parent)
            backup.rename(parent)

    assert outside_file.read_bytes() == b"outside sentinel"
    assert not list(outside.glob("*.tmp"))


@pytest.mark.skipif(os.name != "nt", reason="Windows junction ABA regression")
def test_public_preview_copy_installs_through_held_parent_after_junction_aba(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))
    expected = _install_public_preview(run)
    parent = tmp_path / "publication"
    parent.mkdir()
    backup = tmp_path / "publication-original"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "public.gif"
    outside_file.write_bytes(b"outside sentinel")
    original_create = redaction_module._win_create_child
    swapped = False

    def swap_parent_before_handle_relative_create(
        directory_handle: object, name: str, *, directory: bool
    ) -> tuple[object, tuple[int, ...]]:
        nonlocal swapped
        if not directory and not swapped:
            parent.rename(backup)
            _make_directory_link(parent, outside)
            swapped = True
        return original_create(directory_handle, name, directory=directory)

    monkeypatch.setattr(
        redaction_module,
        "_win_create_child",
        swap_parent_before_handle_relative_create,
    )
    try:
        copy_public_preview(run, "B0", parent / "public.gif")
    finally:
        if swapped:
            _remove_directory_link(parent)
            backup.rename(parent)

    assert (parent / "public.gif").read_bytes() == expected
    assert outside_file.read_bytes() == b"outside sentinel"
    assert not list(outside.glob("*.tmp"))


@pytest.mark.skipif(os.name != "nt", reason="Windows junction cleanup regression")
def test_public_preview_copy_cleans_failed_temp_by_handle_after_parent_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))
    _install_public_preview(run)
    parent = tmp_path / "publication"
    parent.mkdir()
    backup = tmp_path / "publication-original"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_temp = outside / ".public.gif.fixed.tmp"
    outside_temp.write_bytes(b"outside sentinel")
    original_create = redaction_module._win_create_child
    swapped = False

    def swap_parent_then_create(
        directory_handle: object, name: str, *, directory: bool
    ) -> tuple[object, tuple[int, ...]]:
        nonlocal swapped
        if not directory and not swapped:
            parent.rename(backup)
            _make_directory_link(parent, outside)
            swapped = True
        return original_create(directory_handle, name, directory=directory)

    class FixedUuid:
        hex = "fixed"

    def interrupt_install(*args: object) -> None:
        raise OSError("install interrupted")

    monkeypatch.setattr(dashboard_module, "uuid4", FixedUuid)
    monkeypatch.setattr(
        redaction_module, "_win_create_child", swap_parent_then_create
    )
    monkeypatch.setattr(
        redaction_module,
        "_win_rename_child",
        interrupt_install,
    )
    try:
        with pytest.raises(OSError, match="install interrupted"):
            copy_public_preview(run, "B0", parent / "public.gif")
    finally:
        if swapped:
            _remove_directory_link(parent)
            backup.rename(parent)

    assert outside_temp.read_bytes() == b"outside sentinel"
    assert not (parent / ".public.gif.fixed.tmp").exists()
    assert not (parent / "public.gif").exists()


def test_public_preview_copy_refuses_private_source_frames(
    tmp_path: Path,
) -> None:
    """Even a verified suite cannot export a preview carrying a private-media flag."""

    run = _write_verified_suite_fixture(tmp_path, variants=("B1",))
    verified = verify_suite_directory(run)
    preview = run / "dashboard" / "media" / "B1-preview.gif"
    preview.parent.mkdir(exist_ok=True)
    imageio.mimsave(
        preview,
        [
            np.full((24, 24, 3), 20, dtype=np.uint8),
            np.full((24, 24, 3), 180, dtype=np.uint8),
        ],
        format="GIF",
        duration=125,
        loop=0,
    )
    manifest_path = run / "suite-manifest.json"
    manifest = dict(verified.manifest)
    files = dict(manifest["files"])
    files["dashboard/media/B1-preview.gif"] = {
        "size": preview.stat().st_size,
        "sha256": sha256(preview.read_bytes()).hexdigest(),
        "media_role": "source_simulation_comparison",
        "contains_private_source_frames": True,
    }
    manifest["files"] = files
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    verify_suite_directory(run)

    with pytest.raises(ValueError, match="contains private source frames"):
        copy_public_preview(run, "B1", tmp_path / "public.gif")


def test_suite_verifier_rejects_rehashed_corrupt_public_preview(tmp_path: Path) -> None:
    """A correct outer hash must not make undecodable public media trustworthy."""

    run = _write_verified_suite_fixture(tmp_path, variants=("B0",))
    preview = run / "dashboard" / "media" / "B0-preview.gif"
    preview.parent.mkdir(exist_ok=True)
    preview.write_bytes(b"GIF89a-corrupt")
    manifest_path = run / "suite-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["dashboard/media/B0-preview.gif"] = {
        "size": preview.stat().st_size,
        "sha256": sha256(preview.read_bytes()).hexdigest(),
        "media_role": "public_simulation_preview",
        "contains_private_source_frames": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(run)

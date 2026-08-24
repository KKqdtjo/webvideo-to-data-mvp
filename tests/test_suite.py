from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import tempfile
import time
from typing import TextIO, TypeAlias

import cv2
import numpy as np
import pytest
import yaml

import webvideo_to_data.cli as cli_module
import webvideo_to_data.redaction as redaction_module
import webvideo_to_data.suite as suite_module
from webvideo_to_data.config import load_experiment_config
from webvideo_to_data.experiment import run_experiment
from webvideo_to_data.ik import IKPlanningError
from webvideo_to_data.suite import (
    B0BenchmarkSummary,
    RolloutRecord,
    SuiteDeps,
    evaluate_b0_robustness,
    make_run_id,
    run_suite,
    summarize_b0,
    validate_run_id,
    verify_suite_directory,
)
from tests.helpers import write_complete_config


FIXED_RUN_ID = "20260819T120102123456Z-a1b2c3d4-7f29"
_PARENT_PROCESS_PHASE_TIMEOUT_S = 30.0
_CHILD_PROCESS_FAIL_SAFE_TIMEOUT_S = 120.0

_ProcessCapture: TypeAlias = tuple[
    str, subprocess.Popen[str], TextIO, TextIO
]


def _captured_process_text(stream: TextIO) -> str:
    stream.flush()
    stream.seek(0)
    return stream.read()


def _process_diagnostics(
    role: str,
    returncode: int,
    stdout: TextIO,
    stderr: TextIO,
    *,
    early: bool = False,
) -> str:
    exit_description = "exited early" if early else "exited"
    return (
        f"{role} subprocess {exit_description} with return code {returncode}\n"
        f"stdout:\n{_captured_process_text(stdout)}\n"
        f"stderr:\n{_captured_process_text(stderr)}"
    )


def _assert_process_alive(
    role: str,
    process: subprocess.Popen[str],
    stdout: TextIO,
    stderr: TextIO,
) -> None:
    returncode = process.poll()
    if returncode is None:
        return
    pytest.fail(
        _process_diagnostics(role, returncode, stdout, stderr, early=True),
        pytrace=False,
    )


def _wait_for_process_marker(
    marker: Path,
    processes: tuple[_ProcessCapture, ...],
    *,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while not marker.exists():
        for role, process, stdout, stderr in processes:
            _assert_process_alive(role, process, stdout, stderr)
        if time.monotonic() >= deadline:
            pytest.fail(f"timed out waiting for {marker.name}", pytrace=False)
        time.sleep(0.01)


def _wait_for_process_success(
    role: str,
    process: subprocess.Popen[str],
    stdout: TextIO,
    stderr: TextIO,
    *,
    timeout_s: float,
) -> None:
    try:
        returncode = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        pytest.fail(f"timed out waiting for {role} subprocess", pytrace=False)
    if returncode != 0:
        pytest.fail(
            _process_diagnostics(role, returncode, stdout, stderr), pytrace=False
        )


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


def _snapshot_tree(root: Path) -> tuple[tuple[str, bytes | None], ...]:
    if not root.exists():
        return ()
    return tuple(
        (path.relative_to(root).as_posix(), path.read_bytes() if path.is_file() else None)
        for path in sorted(root.rglob("*"))
    )


def _passing_record(seed: int) -> RolloutRecord:
    return RolloutRecord(
        seed=seed,
        perturbation={},
        passed=True,
        failed_checks=(),
        execution_tracking_ratio=0.96,
        maximum_lift_m=0.051,
        target_error_m=0.039,
        final_tilt_rad=float(np.deg2rad(14.0)),
        final_linear_speed_m_s=0.019,
        forbidden_contact_count=0,
        maximum_forbidden_penetration_m=0.0,
    )


def _failed_record(seed: int, field: str) -> RolloutRecord:
    return replace(
        _passing_record(seed),
        passed=False,
        failed_checks=(field,),
        target_error_m=0.041,
    )


def _passing_summary(seeds: tuple[int, ...] = tuple(range(19, 49))) -> B0BenchmarkSummary:
    return summarize_b0([_passing_record(seed) for seed in seeds])


def _real_variant_fixture(config, destination: Path, variant: str, no_render: bool):
    return run_experiment(
        config.config_path,
        destination,
        variant=variant,
        no_render=no_render,
    )


def _deps(
    *,
    now: datetime = datetime(2026, 8, 19, 12, 1, 2, 123456, tzinfo=timezone.utc),
    run_variant=_real_variant_fixture,
) -> SuiteDeps:
    return SuiteDeps(
        now_utc=lambda: now,
        random_suffix=lambda: "7f29",
        run_variant=run_variant,
        evaluate_b0=lambda config, seeds: evaluate_b0_robustness(
            config,
            seeds,
            executor=lambda config, seed, perturbation: _passing_record(seed),
        ),
    )


def _refresh_suite_manifest_entry(run_dir: Path, relative_name: str) -> None:
    manifest_path = run_dir / "suite-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = run_dir / relative_name
    manifest["files"][relative_name].update(
        size=path.stat().st_size,
        sha256=sha256(path.read_bytes()).hexdigest(),
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _remove_suite_manifest_entry(run_dir: Path, relative_name: str) -> None:
    manifest_path = run_dir / "suite-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["files"][relative_name]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _set_nested_value(document: dict[str, object], dotted_name: str, value: object) -> None:
    target = document
    *parents, name = dotted_name.split(".")
    for parent in parents:
        child = target[parent]
        assert isinstance(child, dict)
        target = child
    target[name] = value


def _refresh_child_provenance_and_suite(run_dir: Path, variant: str) -> None:
    child = run_dir / "variants" / variant
    child_manifest_path = child / "run_manifest.json"
    child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
    provenance_path = child / "provenance.json"
    child_manifest["files"]["provenance.json"].update(
        size=provenance_path.stat().st_size,
        sha256=sha256(provenance_path.read_bytes()).hexdigest(),
    )
    child_manifest_path.write_text(
        json.dumps(child_manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    metrics_path = run_dir / "suite-metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["variants"][variant]["run_manifest_sha256"] = sha256(
        child_manifest_path.read_bytes()
    ).hexdigest()
    metrics_path.write_text(
        json.dumps(metrics, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    for relative_name in (
        f"variants/{variant}/provenance.json",
        f"variants/{variant}/run_manifest.json",
        "suite-metrics.json",
    ):
        _refresh_suite_manifest_entry(run_dir, relative_name)


def test_run_id_is_stable_format_with_config_hash() -> None:
    value = make_run_id(
        datetime(2026, 8, 19, 12, 1, 2, 123456, tzinfo=timezone.utc),
        "a1b2c3d4" + "0" * 56,
        "7f29",
    )
    assert value == FIXED_RUN_ID
    assert validate_run_id(value) == value


@pytest.mark.parametrize(
    "unsafe",
    ("..", ".", "../escape", "a/b", r"a\b", r"C:\escape", "/escape", "CON"),
)
def test_suite_rejects_path_escape_before_any_output_write(
    tmp_path: Path, unsafe: str,
) -> None:
    config_path = write_complete_config(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    before = _snapshot_tree(tmp_path)
    with pytest.raises(ValueError, match="invalid run_id"):
        run_suite(
            config_path,
            artifacts_root,
            variants=("B2",),
            no_render=True,
            run_id=unsafe,
        )
    assert _snapshot_tree(tmp_path) == before


def test_suite_refuses_existing_run_and_never_replaces_it(tmp_path: Path) -> None:
    config_path = write_complete_config(tmp_path)
    config = load_experiment_config(config_path)
    occupied = tmp_path / config.experiment_id / "runs" / FIXED_RUN_ID
    occupied.mkdir(parents=True)
    personal = occupied / "personal.txt"
    personal.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="run already exists"):
        run_suite(
            config_path=config_path,
            artifacts_root=tmp_path,
            variants=("B0",),
            no_render=True,
            run_id=occupied.name,
        )

    assert personal.read_text(encoding="utf-8") == "keep"


def test_benchmark_passes_only_at_twenty_four_of_thirty_with_no_illegal_contact() -> None:
    records = [_passing_record(seed) for seed in range(19, 43)]
    records += [_failed_record(seed, "target_error_m") for seed in range(43, 49)]
    summary = summarize_b0(records)
    assert summary.successes == 24
    assert summary.rollouts == 30
    assert summary.passed
    assert summary.wilson_95_low == pytest.approx(0.6269, abs=1e-4)


def test_any_illegal_contact_fails_benchmark_even_with_thirty_successes() -> None:
    records = [_passing_record(seed) for seed in range(19, 49)]
    records[0] = replace(records[0], forbidden_contact_count=1)
    summary = summarize_b0(records)
    assert not summary.passed
    assert summary.reason == "illegal_contact_observed"


def test_expected_typed_rollout_failure_is_recorded_and_later_seeds_continue(
    tmp_path: Path,
) -> None:
    calls: list[int] = []

    def executor(config, seed, perturbation):
        del config, perturbation
        calls.append(seed)
        if seed == 23:
            raise IKPlanningError("transport")
        return _passing_record(seed)

    config = load_experiment_config(write_complete_config(tmp_path))
    summary = evaluate_b0_robustness(
        config,
        seeds=tuple(range(19, 49)),
        executor=executor,
    )

    assert calls == list(range(19, 49))
    assert [record.seed for record in summary.records] == list(range(19, 49))
    failed = summary.records[4]
    assert not failed.passed
    assert failed.failed_checks == ("ik_key_pose_transport",)


def test_suite_writes_verified_enhanced_manifest_and_relative_latest_pointer(tmp_path: Path) -> None:
    config_path = write_complete_config(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    result = run_suite(
        config_path,
        artifacts_root,
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )

    verified = verify_suite_directory(result.run_dir)
    assert verified.path == result.run_dir
    assert tuple(verified.variant_runs) == ("B2",)
    assert verified.manifest["feature_set"] == ["core", "dashboard"]
    assert verified.metrics["actions_exported"] == 0
    assert verified.metrics["b0_physics_baseline"] == "not_requested"
    assert result.dashboard_path == result.run_dir / "dashboard" / "index.html"
    assert result.dashboard_path.is_file()
    assert not list(result.run_dir.rglob("actions.npz"))
    assert not list(result.run_dir.rglob("*.gif"))
    assert not (result.run_dir / "dashboard.html").exists()
    environment = json.loads(
        (result.run_dir / "environment.json").read_text(encoding="utf-8")
    )
    assert set(environment) == {
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
    resolved = (result.run_dir / "resolved-config.yaml").read_text(encoding="utf-8")
    assert "registry:synthetic-moving-object" in resolved
    assert str(config_path.resolve()) not in resolved
    assert str((tmp_path / "moving.avi").resolve()) not in resolved
    latest = json.loads(
        (artifacts_root / "SYNTHETIC" / "latest.json").read_text(encoding="utf-8")
    )
    assert latest == {
        "run_path": f"SYNTHETIC/runs/{FIXED_RUN_ID}",
        "run_id": FIXED_RUN_ID,
        "suite_manifest_sha256": sha256(
            (result.run_dir / "suite-manifest.json").read_bytes()
        ).hexdigest(),
    }


@pytest.mark.parametrize("variant", ("B0", "B1"))
@pytest.mark.requires_renderer
def test_suite_preview_renderer_uses_verified_replay_after_swap_and_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    """Catch preview rendering reopening the mutable child after verification."""

    original_verify = suite_module.verify_run_directory
    original_render = suite_module.render_public_simulation_preview
    verification_count = 0
    trusted_replay = b""
    rendered_source = b""
    child_replay: Path | None = None
    trusted_backup = tmp_path / f"{variant}-trusted-replay.mp4"
    attacker_staging = tmp_path / f"{variant}-private-source.mp4"
    attacker_preserved = tmp_path / f"{variant}-private-source-preserved.mp4"
    attacker_staging.write_bytes(b"private-source-pixels")

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
    config["source"].update(
        sha256=sha256(source.read_bytes()).hexdigest(), roi_xywh=[18, 24, 20, 20]
    )
    config["tracking"].update(minimum_valid_ratio=0.1)
    config["simulation"].update(render_size=[96, 72], render_every=20)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def verify_then_swap(path: str | Path) -> object:
        nonlocal verification_count, trusted_replay, child_replay
        verified = original_verify(path)
        if Path(path).name == variant:
            verification_count += 1
            if verification_count == 2:
                child_replay = (
                    tmp_path
                    / "artifacts"
                    / "SYNTHETIC"
                    / "runs"
                    / FIXED_RUN_ID
                    / "variants"
                    / variant
                    / "mujoco_replay.mp4"
                )
                trusted_replay = child_replay.read_bytes()
                child_replay.rename(trusted_backup)
                attacker_staging.rename(child_replay)
        return verified

    def render_then_restore(
        source: Path,
        destination: Path,
        labels: object,
        media: object,
    ) -> Path:
        nonlocal rendered_source
        rendered_source = source.read_bytes()
        assert child_replay is not None
        child_replay.rename(attacker_preserved)
        trusted_backup.rename(child_replay)
        return original_render(source, destination, labels, media)  # type: ignore[arg-type]

    monkeypatch.setattr(suite_module, "verify_run_directory", verify_then_swap)
    monkeypatch.setattr(
        suite_module,
        "render_public_simulation_preview",
        render_then_restore,
    )

    result = None
    try:
        result = run_suite(
            config_path,
            tmp_path / "artifacts",
            variants=(variant,),
            no_render=False,
            run_id=FIXED_RUN_ID,
            deps=_deps(),
        )
    except ValueError as error:
        assert "stable snapshot" in str(error)

    assert verification_count >= 2
    assert rendered_source == trusted_replay
    if result is None:
        assert not (tmp_path / "artifacts" / "SYNTHETIC" / "latest.json").exists()
        return
    preview = result.run_dir / "dashboard" / "media" / f"{variant}-preview.gif"
    assert preview.is_file()
    verified = verify_suite_directory(result.run_dir)
    assert verified.manifest["files"][preview.relative_to(result.run_dir).as_posix()] == {
        "size": preview.stat().st_size,
        "sha256": sha256(preview.read_bytes()).hexdigest(),
        "media_role": "public_simulation_preview",
        "contains_private_source_frames": False,
    }


def test_publish_latest_hashes_captured_manifest_not_reopened_mutable_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts_root = tmp_path / "artifacts"
    result = run_suite(
        write_complete_config(tmp_path),
        artifacts_root,
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    manifest_path = result.run_dir / "suite-manifest.json"
    trusted_manifest = manifest_path.read_bytes()
    original_verify = suite_module.verify_suite_directory
    mutated = False

    def mutate_after_verification(path: str | Path) -> object:
        nonlocal mutated
        verified = original_verify(path)
        manifest_path.write_bytes(trusted_manifest[:-1] + b" ")
        mutated = True
        return verified

    monkeypatch.setattr(
        suite_module, "verify_suite_directory", mutate_after_verification
    )
    latest = tmp_path / "isolated-latest.json"
    suite_module._publish_latest(
        suite_module.RunIdentity(FIXED_RUN_ID, result.run_dir),
        artifacts_root,
        latest,
        tmp_path / "isolated-latest.lock",
    )

    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert mutated
    assert payload["suite_manifest_sha256"] == sha256(trusted_manifest).hexdigest()
    with pytest.raises(ValueError, match="latest pointer manifest mismatch"):
        suite_module._validate_latest(latest, artifacts_root)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_verify_suite_rejects_manifested_variant_directory_junction(
    tmp_path: Path,
) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    variant = result.run_dir / "variants" / "B2"
    external_variant = result.run_dir.parent / "outside-suite-B2"
    variant.rename(external_variant)
    _make_directory_link(variant, external_variant)
    try:
        with pytest.raises(ValueError, match="suite directory verification failed"):
            verify_suite_directory(result.run_dir)
    finally:
        _remove_directory_link(variant)


def test_verify_suite_rejects_root_directory_link(tmp_path: Path) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    linked_run = tmp_path / FIXED_RUN_ID
    _make_directory_link(linked_run, result.run_dir)
    try:
        with pytest.raises(ValueError, match="suite directory verification failed"):
            verify_suite_directory(linked_run)
    finally:
        _remove_directory_link(linked_run)


def test_verify_suite_rejects_ancestor_directory_link(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    result = run_suite(
        write_complete_config(real_parent),
        real_parent / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    linked_parent = tmp_path / "linked-parent"
    _make_directory_link(linked_parent, result.run_dir.parent)
    try:
        with pytest.raises(ValueError, match="suite directory verification failed"):
            verify_suite_directory(linked_parent / result.run_dir.name)
    finally:
        _remove_directory_link(linked_parent)


def test_verify_suite_rejects_manifest_mutation_during_stable_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    manifest_path = result.run_dir / "suite-manifest.json"
    original = manifest_path.read_bytes()
    helper_name = (
        "_snapshot_read_windows" if os.name == "nt" else "_snapshot_read_posix"
    )
    original_read = getattr(redaction_module, helper_name)
    mutated = False

    def mutate_after_read(handle: object, size: int) -> bytes:
        nonlocal mutated
        content = original_read(handle, size)
        if not mutated and size == len(original):
            manifest_path.write_bytes(original[:-1] + b" ")
            mutated = True
        return content

    monkeypatch.setattr(redaction_module, helper_name, mutate_after_read)

    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(result.run_dir)
    assert mutated


def test_verify_suite_rejects_ancestor_aba_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_parent = tmp_path / "trusted-parent"
    trusted_parent.mkdir()
    result = run_suite(
        write_complete_config(trusted_parent),
        trusted_parent / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    alternate_parent = tmp_path / "alternate-parent"
    alternate_parent.mkdir()
    backup = tmp_path / "trusted-parent-backup"
    original_verify_snapshot = suite_module._verify_suite_snapshot
    moved = False
    swapped = False

    def swap_ancestor_then_verify(
        materialized: Path, snapshot: object, *, display_path: Path
    ) -> object:
        nonlocal moved, swapped
        trusted_parent.rename(backup)
        moved = True
        _make_directory_link(trusted_parent, alternate_parent)
        swapped = True
        return original_verify_snapshot(
            materialized, snapshot, display_path=display_path  # type: ignore[arg-type]
        )

    monkeypatch.setattr(
        suite_module, "_verify_suite_snapshot", swap_ancestor_then_verify
    )
    try:
        with pytest.raises(ValueError, match="suite directory verification failed"):
            verify_suite_directory(result.run_dir)
    finally:
        if swapped:
            _remove_directory_link(trusted_parent)
        if moved:
            backup.rename(trusted_parent)


def test_verify_suite_rejects_root_rebound_after_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    original_verify_snapshot = suite_module._verify_suite_snapshot
    backup = result.run_dir.with_name(f"{result.run_dir.name}-trusted-backup")
    attacker = result.run_dir.with_name(f"{result.run_dir.name}-attacker-preserved")
    rebound = False

    def rebind_root_then_verify(
        materialized: Path, snapshot: object, *, display_path: Path
    ) -> object:
        nonlocal rebound
        result.run_dir.rename(backup)
        result.run_dir.mkdir()
        (result.run_dir / "attacker.txt").write_text("unverified", encoding="utf-8")
        rebound = True
        return original_verify_snapshot(
            materialized, snapshot, display_path=display_path  # type: ignore[arg-type]
        )

    monkeypatch.setattr(
        suite_module, "_verify_suite_snapshot", rebind_root_then_verify
    )
    try:
        with pytest.raises(ValueError, match="suite directory verification failed"):
            verify_suite_directory(result.run_dir)
    finally:
        if rebound:
            result.run_dir.rename(attacker)
            backup.rename(result.run_dir)
    assert rebound


def test_verify_suite_rejects_original_manifest_changed_after_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    original_verify_snapshot = suite_module._verify_suite_snapshot
    manifest_path = result.run_dir / "suite-manifest.json"
    original = manifest_path.read_bytes()
    mutated = False

    def mutate_original_then_verify(
        materialized: Path, snapshot: object, *, display_path: Path
    ) -> object:
        nonlocal mutated
        manifest_path.write_bytes(original[:-1] + b" ")
        mutated = True
        return original_verify_snapshot(
            materialized, snapshot, display_path=display_path  # type: ignore[arg-type]
        )

    monkeypatch.setattr(
        suite_module,
        "_verify_suite_snapshot",
        mutate_original_then_verify,
    )

    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(result.run_dir)
    assert mutated


def test_verify_suite_rejects_original_manifest_changed_during_result_rebase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    original_rebase = suite_module._rebase_verified_suite
    manifest_path = result.run_dir / "suite-manifest.json"
    original = manifest_path.read_bytes()
    mutated = False

    def mutate_after_rebase(*args: object, **kwargs: object) -> object:
        nonlocal mutated
        verified = original_rebase(*args, **kwargs)  # type: ignore[arg-type]
        manifest_path.write_bytes(original[:-1] + b" ")
        mutated = True
        return verified

    monkeypatch.setattr(suite_module, "_rebase_verified_suite", mutate_after_rebase)

    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(result.run_dir)
    assert mutated


def test_verify_suite_reports_real_child_directory_and_file_identities(
    tmp_path: Path,
) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )

    verified = verify_suite_directory(result.run_dir)
    child = result.run_dir / "variants" / "B2"
    child_stat = child.stat()
    expected_snapshot = {}
    for path in child.iterdir():
        item = path.stat()
        content = path.read_bytes()
        expected_snapshot[path.name] = (
            len(content),
            sha256(content).hexdigest(),
            item.st_mtime_ns,
            item.st_dev,
            item.st_ino,
        )

    assert verified.variant_runs["B2"].directory_identity == (
        child_stat.st_dev,
        child_stat.st_ino,
    )
    assert verified.variant_runs["B2"].snapshot == expected_snapshot


def test_verify_suite_rejects_rehashed_dashboard_secret(tmp_path: Path) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    dashboard_path = result.run_dir / "dashboard" / "index.html"
    dashboard_path.write_text(
        dashboard_path.read_text(encoding="utf-8")
        + "\n<!-- Authorization: Bearer "
        + "TEST_"
        + "DASHBOARD_SECRET -->\n",
        encoding="utf-8",
    )
    _refresh_suite_manifest_entry(result.run_dir, "dashboard/index.html")

    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(result.run_dir)


def test_latest_bytes_do_not_change_when_a_later_variant_raises(tmp_path: Path) -> None:
    config_path = write_complete_config(tmp_path)
    root = tmp_path / "artifacts"
    run_suite(
        config_path,
        root,
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    latest_path = root / "SYNTHETIC" / "latest.json"
    before = latest_path.read_bytes()

    def fail_b1(config, destination, variant, no_render):
        del config, destination, no_render
        if variant == "B1":
            raise OSError("synthetic B1 infrastructure failure")
        raise AssertionError("unexpected variant")

    later = "20260819T120103123456Z-a1b2c3d4-7f30"
    with pytest.raises(OSError, match="synthetic B1 infrastructure failure"):
        run_suite(
            config_path,
            root,
            variants=("B1",),
            no_render=True,
            run_id=later,
            deps=_deps(run_variant=fail_b1),
        )

    assert latest_path.read_bytes() == before
    failed_metrics = json.loads(
        (root / "SYNTHETIC" / "runs" / later / "suite-metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert failed_metrics["status"] == "failed"
    assert failed_metrics["actions_exported"] == 0


def test_equal_utc_run_finishing_later_does_not_replace_latest(tmp_path: Path) -> None:
    config_path = write_complete_config(tmp_path)
    root = tmp_path / "artifacts"
    first_id = "20260819T120102123456Z-a1b2c3d4-7f29"
    equal_utc_id = "20260819T120102123456Z-a1b2c3d4-7f30"
    run_suite(
        config_path,
        root,
        variants=("B2",),
        no_render=True,
        run_id=first_id,
        deps=_deps(),
    )

    run_suite(
        config_path,
        root,
        variants=("B2",),
        no_render=True,
        run_id=equal_utc_id,
        deps=_deps(),
    )

    latest = json.loads((root / "SYNTHETIC" / "latest.json").read_text("utf-8"))
    assert latest["run_id"] == first_id


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
def test_contained_accepts_extended_alias_for_nonexistent_child(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "physical-artifacts").resolve()
    root.mkdir()
    extended = "\\\\?\\" + str(root)
    extended = extended[:4] + extended[4].lower() + extended[5:]
    candidate = Path(extended) / "SYNTHETIC" / "latest.json"

    contained = suite_module._contained(candidate, root, "latest")

    assert contained == root / "SYNTHETIC" / "latest.json"


@pytest.mark.parametrize(
    ("candidate", "expected"),
    (
        (
            r"\\?\C:\Artifacts\SYNTHETIC\..\SYNTHETIC\latest.json",
            r"c:\artifacts\synthetic\latest.json",
        ),
        (
            r"\\?\UNC\Server\Share\Artifacts\SYNTHETIC\latest.json",
            r"\\server\share\artifacts\synthetic\latest.json",
        ),
    ),
)
def test_windows_containment_comparison_normalizes_only_equivalent_namespaces(
    candidate: str, expected: str
) -> None:
    assert suite_module._windows_path_for_containment(Path(candidate)) == (
        PureWindowsPath(expected)
    )


@pytest.mark.parametrize(
    "candidate",
    (
        r"C:\Artifacts\SYNTHETIC\latest.json",
        r"\\Server\Share\Artifacts\SYNTHETIC\latest.json",
        r"\\Server\C$\Artifacts\SYNTHETIC\latest.json",
        r"\\Server\ADMIN$\Artifacts\SYNTHETIC\latest.json",
        r"\\?\C:\Artifacts\SYNTHETIC\latest.json",
        r"\\?\UNC\Server\Share\Artifacts\SYNTHETIC\latest.json",
    ),
)
def test_windows_namespace_validation_allows_only_supported_filesystem_forms(
    candidate: str,
) -> None:
    suite_module._validate_windows_path_namespace(Path(candidate), "latest")


@pytest.mark.parametrize(
    "candidate",
    (
        r"\\Server\pipe\webvideo-to-data",
        r"\\?\UNC\Server\PIPE\webvideo-to-data",
        r"\\Server\IPC$\webvideo-to-data",
        r"\\?\UNC\Server\ipc$\webvideo-to-data",
        r"\\Server\mailslot\webvideo-to-data",
        r"\\*\MAILSLOT\webvideo-to-data",
        r"\\?\UNC\Server\Mailslot\webvideo-to-data",
        r"\\?\UNC\*\mailslot\webvideo-to-data",
    ),
)
def test_windows_namespace_validation_rejects_unc_ipc_shares(
    candidate: str,
) -> None:
    with pytest.raises(ValueError, match="unsafe artifacts root path namespace"):
        suite_module._validate_windows_path_namespace(
            Path(candidate), "artifacts root"
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows UNC IPC namespace regression")
@pytest.mark.parametrize(
    "artifacts_root",
    (
        r"\\Server\pipe\webvideo-to-data",
        r"\\?\UNC\Server\PIPE\webvideo-to-data",
        r"\\Server\IPC$\webvideo-to-data",
        r"\\?\UNC\Server\ipc$\webvideo-to-data",
        r"\\Server\mailslot\webvideo-to-data",
        r"\\*\MAILSLOT\webvideo-to-data",
        r"\\?\UNC\Server\Mailslot\webvideo-to-data",
        r"\\?\UNC\*\mailslot\webvideo-to-data",
    ),
)
def test_run_suite_rejects_unc_ipc_root_before_identity_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifacts_root: str,
) -> None:
    config_path = write_complete_config(tmp_path)

    def fail_if_resolution_boundary_is_reached(*args: object) -> None:
        del args
        raise AssertionError("unsafe root reached identity resolution boundary")

    monkeypatch.setattr(
        suite_module, "_run_identity", fail_if_resolution_boundary_is_reached
    )

    with pytest.raises(ValueError, match="unsafe artifacts root path namespace"):
        run_suite(
            config_path,
            Path(artifacts_root),
            variants=("B2",),
            no_render=True,
            run_id=FIXED_RUN_ID,
            deps=_deps(),
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows device namespace regression")
@pytest.mark.parametrize(
    "namespace",
    (
        "device_alias",
        "globalroot",
        "volume_guid",
        "named_pipe",
        "nt_object_manager",
        "double_nt_object_manager",
        "non_ascii_drive",
    ),
)
def test_contained_rejects_device_namespaces_before_resolve(
    tmp_path: Path, namespace: str
) -> None:
    root = (tmp_path / "artifacts").resolve()
    root.mkdir()
    candidates = {
        "device_alias": Path("\\\\.\\" + str(root)) / "latest.json",
        "globalroot": Path(
            r"\\?\GLOBALROOT\Device\HarddiskVolume1\Artifacts\latest.json"
        ),
        "volume_guid": Path(
            r"\\?\Volume{00000000-0000-0000-0000-000000000000}"
            r"\Artifacts\latest.json"
        ),
        "named_pipe": Path(r"\\?\PIPE\webvideo-to-data\latest.json"),
        "nt_object_manager": Path(r"\??\C:\Artifacts\latest.json"),
        "double_nt_object_manager": Path(r"\\??\C:\Artifacts\latest.json"),
        "non_ascii_drive": Path(r"\\?\É:\Artifacts\latest.json"),
    }

    with pytest.raises(ValueError, match="unsafe latest path namespace"):
        suite_module._contained(candidates[namespace], root, "latest")


@pytest.mark.skipif(os.name != "nt", reason="Windows device namespace regression")
def test_run_suite_rejects_device_namespace_artifacts_root_before_resolve(
    tmp_path: Path,
) -> None:
    config_path = write_complete_config(tmp_path)
    root = (tmp_path / "artifacts").resolve()
    root.mkdir()
    device_root = Path("\\\\.\\" + str(root))

    with pytest.raises(ValueError, match="unsafe artifacts root path namespace"):
        run_suite(
            config_path,
            device_root,
            variants=("B2",),
            no_render=True,
            run_id=FIXED_RUN_ID,
            deps=_deps(),
        )


def test_run_suite_anchors_explicit_relative_artifacts_root_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_complete_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = run_suite(
        config_path,
        Path("relative-artifacts"),
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )

    expected = (
        tmp_path
        / "relative-artifacts"
        / "SYNTHETIC"
        / "runs"
        / FIXED_RUN_ID
    )
    assert result.run_dir == expected
    assert verify_suite_directory(expected)


def test_contained_rejects_dot_segment_escape(tmp_path: Path) -> None:
    root = (tmp_path / "artifacts").resolve()
    root.mkdir()
    candidate_root = Path("\\\\?\\" + str(root)) if os.name == "nt" else root

    with pytest.raises(ValueError, match="unsafe latest path"):
        suite_module._contained(
            candidate_root
            / "nested"
            / ".."
            / ".."
            / "outside"
            / "latest.json",
            root,
            "latest",
        )


def test_contained_rejects_existing_directory_link_escape(tmp_path: Path) -> None:
    root = (tmp_path / "artifacts").resolve()
    root.mkdir()
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    linked = root / "linked"
    _make_directory_link(linked, outside)
    candidate_root = Path("\\\\?\\" + str(root)) if os.name == "nt" else root
    try:
        with pytest.raises(ValueError, match="unsafe latest path"):
            suite_module._contained(
                candidate_root / linked.name / "latest.json", root, "latest"
            )
    finally:
        _remove_directory_link(linked)


def test_process_exit_diagnostics_include_role_returncode_and_streams() -> None:
    with (
        tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout,
        tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "print('stdout sentinel', flush=True); "
                    "print('stderr sentinel', file=sys.stderr, flush=True); "
                    "raise SystemExit(7)"
                ),
            ],
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        process.wait(timeout=10.0)

        with pytest.raises(pytest.fail.Exception) as error:
            _assert_process_alive("old", process, stdout, stderr)

    message = str(error.value)
    assert "old subprocess exited early with return code 7" in message
    assert "stdout sentinel" in message
    assert "stderr sentinel" in message


def test_process_marker_wait_uses_a_fresh_deadline_for_each_phase(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "ready"
    start = tmp_path / "start"
    first = tmp_path / "first"
    second = tmp_path / "second"
    release = tmp_path / "release"
    script = r"""
import sys
import time
from pathlib import Path

ready, start, first, second, release = map(Path, sys.argv[1:])
ready.write_text("ready", encoding="utf-8")
fail_safe = time.monotonic() + 2.0
while not start.exists():
    if time.monotonic() >= fail_safe:
        raise TimeoutError("timed out waiting for start")
    time.sleep(0.01)
time.sleep(0.3)
first.write_text("ready", encoding="utf-8")
time.sleep(0.3)
second.write_text("ready", encoding="utf-8")
deadline = time.monotonic() + 2.0
while not release.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
"""
    with (
        tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout,
        tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(ready),
                str(start),
                str(first),
                str(second),
                str(release),
            ],
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        capture = ("worker", process, stdout, stderr)
        try:
            _wait_for_process_marker(ready, (capture,), timeout_s=10.0)
            start.touch()
            _wait_for_process_marker(first, (capture,), timeout_s=0.5)
            _wait_for_process_marker(second, (capture,), timeout_s=0.5)
        finally:
            release.touch()
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5.0)


def test_physical_alias_suite_lock_serializes_real_process_pointer_publication(
    tmp_path: Path,
) -> None:
    config_path = write_complete_config(tmp_path)
    root = (tmp_path / "physical-artifacts").resolve()
    root.mkdir()
    if os.name == "nt":
        alias = Path("\\\\?\\" + str(root))
    else:
        alias = tmp_path / "physical-artifacts-alias"
        try:
            alias.symlink_to(root, target_is_directory=True)
        except (NotImplementedError, OSError) as error:
            pytest.skip(f"directory symlinks are unavailable: {error}")
    if not alias.samefile(root):
        pytest.skip("a physical parent path alias is unavailable")
    assert _CHILD_PROCESS_FAIL_SAFE_TIMEOUT_S > (
        2 * _PARENT_PROCESS_PHASE_TIMEOUT_S
    )
    ready_old = tmp_path / "ready-old"
    ready_new = tmp_path / "ready-new"
    start = tmp_path / "start-publication"
    old_inside = tmp_path / "old-inside-replace"
    release_old = tmp_path / "release-old"
    new_inside = tmp_path / "new-inside-replace"
    old_id = "20260819T120102123456Z-a1b2c3d4-7f29"
    new_id = "20260819T120103123456Z-a1b2c3d4-7f30"
    script = r"""
import sys
import time
from pathlib import Path
import webvideo_to_data.suite as suite

config, root, run_id, role, ready, start, old_inside, release_old, new_inside, fail_safe_s = sys.argv[1:]
ready = Path(ready)
start = Path(start)
old_inside = Path(old_inside)
release_old = Path(release_old)
new_inside = Path(new_inside)
fail_safe_s = float(fail_safe_s)
real_publish = suite._publish_latest
real_replace = suite._replace_latest

def wait_for(path):
    deadline = time.monotonic() + fail_safe_s
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path.name}")
        time.sleep(0.01)

def gated_publish(*args, **kwargs):
    ready.write_text("ready", encoding="utf-8")
    wait_for(start)
    if role == "new":
        wait_for(old_inside)
    return real_publish(*args, **kwargs)

def gated_replace(path, payload):
    marker = old_inside if role == "old" else new_inside
    marker.write_text("inside", encoding="utf-8")
    if role == "old":
        wait_for(release_old)
    return real_replace(path, payload)

suite._publish_latest = gated_publish
suite._replace_latest = gated_replace
suite.run_suite(config, root, variants=("B2",), no_render=True, run_id=run_id)
"""
    environment = dict(
        os.environ,
        PYTHONPATH=str((Path(__file__).resolve().parents[1] / "src").resolve()),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    common = [
        str(start),
        str(old_inside),
        str(release_old),
        str(new_inside),
        str(_CHILD_PROCESS_FAIL_SAFE_TIMEOUT_S),
    ]
    with (
        tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as old_stdout,
        tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as old_stderr,
        tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as new_stdout,
        tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as new_stderr,
    ):
        old = subprocess.Popen(
            [
                sys.executable, "-c", script, str(config_path), str(root), old_id,
                "old", str(ready_old), *common,
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=old_stdout,
            stderr=old_stderr,
            text=True,
        )
        new: subprocess.Popen[str] | None = None
        try:
            new = subprocess.Popen(
                [
                    sys.executable, "-c", script, str(config_path), str(alias),
                    new_id, "new", str(ready_new), *common,
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdout=new_stdout,
                stderr=new_stderr,
                text=True,
            )
            captures = (
                ("old", old, old_stdout, old_stderr),
                ("new", new, new_stdout, new_stderr),
            )
            _wait_for_process_marker(
                ready_old,
                captures,
                timeout_s=_PARENT_PROCESS_PHASE_TIMEOUT_S,
            )
            _wait_for_process_marker(
                ready_new,
                captures,
                timeout_s=_PARENT_PROCESS_PHASE_TIMEOUT_S,
            )
            start.write_text("go", encoding="utf-8")
            _wait_for_process_marker(
                old_inside,
                captures,
                timeout_s=_PARENT_PROCESS_PHASE_TIMEOUT_S,
            )
            time.sleep(0.25)
            _assert_process_alive("old", old, old_stdout, old_stderr)
            _assert_process_alive("new", new, new_stdout, new_stderr)
            assert not new_inside.exists()
            release_old.write_text("go", encoding="utf-8")
            _wait_for_process_success(
                "old",
                old,
                old_stdout,
                old_stderr,
                timeout_s=_PARENT_PROCESS_PHASE_TIMEOUT_S,
            )
            _wait_for_process_success(
                "new",
                new,
                new_stdout,
                new_stderr,
                timeout_s=_PARENT_PROCESS_PHASE_TIMEOUT_S,
            )
        finally:
            start.touch()
            release_old.touch()
            for process in (old, new):
                if process is not None and process.poll() is None:
                    process.kill()
                if process is not None:
                    process.wait(timeout=5.0)

    experiment = root / "SYNTHETIC"
    latest = json.loads((experiment / "latest.json").read_text(encoding="utf-8"))
    assert latest["run_id"] == new_id
    assert verify_suite_directory(experiment / "runs" / old_id)
    assert verify_suite_directory(experiment / "runs" / new_id)


def test_suite_b0_jsonl_preserves_exact_seed_order_and_schema(tmp_path: Path) -> None:
    config_path = write_complete_config(tmp_path)
    def executor(config, seed, perturbation):
        del config, perturbation
        if seed == 23:
            raise IKPlanningError("transport")
        return _passing_record(seed)

    dependencies = replace(
        _deps(),
        evaluate_b0=lambda config, seeds: evaluate_b0_robustness(
            config, seeds, executor=executor
        ),
    )
    result = run_suite(
        config_path,
        tmp_path / "artifacts",
        variants=("B0",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=dependencies,
    )
    rows = [
        json.loads(line)
        for line in (result.run_dir / "B0" / "benchmark-rollouts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 30
    assert [row["seed"] for row in rows] == list(range(19, 49))
    assert all(row["schema_version"] == 1 for row in rows)
    assert rows[4]["failed_checks"] == ["ik_key_pose_transport"]
    summary = json.loads(
        (result.run_dir / "B0" / "benchmark-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["yaw_perturbation_observability"] == (
        "geometrically_unobservable_for_axisymmetric_can"
    )


def test_verify_suite_rejects_tamper_and_unmanifested_file(tmp_path: Path) -> None:
    config_path = write_complete_config(tmp_path)
    first = run_suite(
        config_path,
        tmp_path / "one",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    (first.run_dir / "environment.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(first.run_dir)

    second = run_suite(
        config_path,
        tmp_path / "two",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    (second.run_dir / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(second.run_dir)


def test_verify_suite_rejects_nested_suite_manifest_that_is_not_root_manifest(
    tmp_path: Path,
) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B0",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    (result.run_dir / "B0" / "suite-manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(result.run_dir)


def test_suite_manifest_includes_verified_child_file_named_suite_manifest(
    tmp_path: Path,
) -> None:
    def variant_with_nested_manifest(config, destination, variant, no_render):
        metrics = run_experiment(
            config.config_path,
            destination,
            variant=variant,
            no_render=no_render,
        )
        nested = destination / "suite-manifest.json"
        nested.write_text("{}\n", encoding="utf-8")
        manifest_path = destination / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][nested.name] = {
            "size": nested.stat().st_size,
            "sha256": sha256(nested.read_bytes()).hexdigest(),
        }
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return metrics

    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(run_variant=variant_with_nested_manifest),
    )

    manifest = json.loads(
        (result.run_dir / "suite-manifest.json").read_text(encoding="utf-8")
    )
    assert "variants/B2/suite-manifest.json" in manifest["files"]
    assert verify_suite_directory(result.run_dir)


@pytest.mark.parametrize("payload_name", ("environment.json", "resolved-config.yaml"))
def test_verify_suite_requires_public_root_payload_even_if_manifest_entry_is_removed(
    tmp_path: Path, payload_name: str,
) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    (result.run_dir / payload_name).unlink()
    _remove_suite_manifest_entry(result.run_dir, payload_name)

    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(result.run_dir)


def test_verify_suite_rejects_environment_path_field_after_rehash(
    tmp_path: Path,
) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    environment_path = result.run_dir / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["workspace_path"] = str(tmp_path.resolve())
    environment_path.write_text(
        json.dumps(environment, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _refresh_suite_manifest_entry(result.run_dir, "environment.json")

    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(result.run_dir)


def test_verify_suite_rejects_suite_config_mixed_with_verified_child_config(
    tmp_path: Path,
) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    other_digest = "f" * 64
    resolved_path = result.run_dir / "resolved-config.yaml"
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    resolved["config_sha256"] = other_digest
    resolved_path.write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    _refresh_suite_manifest_entry(result.run_dir, "resolved-config.yaml")
    manifest_path = result.run_dir / "suite-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_sha256"] = other_digest
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(result.run_dir)


@pytest.mark.parametrize(
    "changes",
    (
        pytest.param(
            {
                "source.id": "../private-source",
                "source.path": "registry:../private-source",
            },
            id="source-path-bearing-logical-id",
        ),
        pytest.param({"source.fps": True}, id="source-bool-as-number"),
        pytest.param(
            {"tracking.minimum_live_points": True},
            id="tracking-bool-as-int",
        ),
        pytest.param(
            {"tracking.minimum_valid_ratio": 1.1},
            id="tracking-unit-range",
        ),
        pytest.param(
            {"scene.x_bounds_m": [0.1, -0.1]},
            id="scene-bounds-relationship",
        ),
        pytest.param(
            {"scene.b0_start_m": "not-a-vector"},
            id="scene-wrong-container",
        ),
        pytest.param(
            {"scene.grasp_quaternion_wxyz": [0.0, 0.0, 0.0, 0.0]},
            id="scene-zero-quaternion",
        ),
        pytest.param({"ik.damping": float("nan")}, id="ik-nonfinite"),
        pytest.param(
            {"ik.maximum_iterations": True},
            id="ik-bool-as-int",
        ),
        pytest.param(
            {"ik.position_tolerance_m": 0.0},
            id="ik-positive-range",
        ),
        pytest.param(
            {"control.gripper_closed_width_m": 0.09},
            id="control-gripper-relationship",
        ),
        pytest.param(
            {"control.phase_duration_s.home": 0.0},
            id="control-nested-positive-range",
        ),
        pytest.param(
            {"control.control_hz": "100"},
            id="control-wrong-scalar-type",
        ),
        pytest.param(
            {"collision.maximum_penetration_m": 0.0},
            id="collision-positive-range",
        ),
        pytest.param(
            {"collision.allowed_contact_pairs.home": [["", "can"]]},
            id="collision-empty-logical-identifier",
        ),
        pytest.param(
            {"collision.maximum_final_linear_speed_m_s": float("inf")},
            id="collision-nonfinite",
        ),
        pytest.param(
            {"perturbation.mass_fraction": 1.1},
            id="perturbation-unit-range",
        ),
        pytest.param(
            {"perturbation.xy_half_range_m": float("nan")},
            id="perturbation-nonfinite",
        ),
        pytest.param(
            {"perturbation.yaw_half_range_rad": True},
            id="perturbation-bool-as-number",
        ),
        pytest.param(
            {"simulation.b0_mode": "kinematic_replay"},
            id="simulation-domain-enum",
        ),
        pytest.param(
            {"simulation.render_size": [0, 320]},
            id="simulation-positive-container-values",
        ),
        pytest.param(
            {"simulation.render_every": True},
            id="simulation-bool-as-int",
        ),
        pytest.param(
            {"media.comparison_alignment": "index_aligned"},
            id="media-domain-enum",
        ),
        pytest.param(
            {"media.letterbox_bgr": [0, 0, 256]},
            id="media-byte-range",
        ),
        pytest.param(
            {"media.output_fps": True},
            id="media-bool-as-number",
        ),
        pytest.param(
            {"media.canvas_size": "960x320"},
            id="media-wrong-container",
        ),
    ),
)
def test_verify_suite_rejects_invalid_public_resolved_values_after_rehash(
    tmp_path: Path, changes: dict[str, object],
) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    resolved_path = result.run_dir / "resolved-config.yaml"
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    for dotted_name, value in changes.items():
        _set_nested_value(resolved, dotted_name, value)
    resolved_path.write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    _refresh_suite_manifest_entry(result.run_dir, "resolved-config.yaml")

    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(result.run_dir)


def test_verify_suite_rejects_unpinned_environment_model_hash_after_rehash(
    tmp_path: Path,
) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    environment_path = result.run_dir / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["model_sha256"] = "f" * 64
    environment_path.write_text(
        json.dumps(environment, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _refresh_suite_manifest_entry(result.run_dir, "environment.json")

    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(result.run_dir)


@pytest.mark.parametrize(
    ("record_name", "replacement"),
    (
        pytest.param("source", "other-logical-source", id="source-id"),
        pytest.param("model", "../path-bearing-model", id="model-id"),
    ),
)
def test_verify_suite_binds_child_source_and_model_logical_identities(
    tmp_path: Path, record_name: str, replacement: str,
) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    provenance_path = result.run_dir / "variants" / "B2" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance[record_name]["id"] = replacement
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _refresh_child_provenance_and_suite(result.run_dir, "B2")

    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(result.run_dir)


def test_verify_suite_binds_child_experiment_id_to_public_config_after_rehash(
    tmp_path: Path,
) -> None:
    result = run_suite(
        write_complete_config(tmp_path),
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    provenance_path = result.run_dir / "variants" / "B2" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["experiment_id"] = "OTHER-EXPERIMENT"
    provenance["config"]["resolved"]["experiment_id"] = "OTHER-EXPERIMENT"
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _refresh_child_provenance_and_suite(result.run_dir, "B2")

    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(result.run_dir)


def test_suite_uses_immutable_b0_seed_set_when_config_seed_differs(
    tmp_path: Path,
) -> None:
    config_path = write_complete_config(tmp_path, {"random_seed": 7})
    observed: list[int] = []

    def evaluate(config, seeds):
        observed.extend(seeds)
        return evaluate_b0_robustness(
            config,
            seeds,
            executor=lambda config, seed, perturbation: _passing_record(seed),
        )

    result = run_suite(
        config_path,
        tmp_path / "artifacts",
        variants=("B0",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=replace(_deps(), evaluate_b0=evaluate),
    )

    assert observed == list(range(19, 49))
    assert verify_suite_directory(result.run_dir)


@pytest.mark.parametrize("fault", ("write", "environment", "model"))
def test_every_post_creation_failure_records_failed_metrics_and_preserves_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    config_path = write_complete_config(tmp_path)
    root = tmp_path / "artifacts"
    run_suite(
        config_path,
        root,
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    latest_path = root / "SYNTHETIC" / "latest.json"
    latest_before = latest_path.read_bytes()

    if fault == "write":
        def fail_write(path, payload):
            del path, payload
            raise OSError("injected resolved-config write failure")

        monkeypatch.setattr(suite_module, "_write_yaml_new", fail_write)
        expected_error: type[Exception] = OSError
        message = "injected resolved-config write failure"
    elif fault == "environment":
        def fail_environment():
            raise AssertionError("injected environment programmer failure")

        monkeypatch.setattr(suite_module, "_environment_payload", fail_environment)
        expected_error = AssertionError
        message = "injected environment programmer failure"
    else:
        def fail_model(variant):
            del variant
            raise RuntimeError("injected model identity failure")

        monkeypatch.setattr(suite_module, "pinned_model_identity", fail_model)
        expected_error = RuntimeError
        message = "injected model identity failure"

    later_id = "20260819T120103123456Z-a1b2c3d4-7f30"
    with pytest.raises(expected_error, match=message):
        run_suite(
            config_path,
            root,
            variants=("B2",),
            no_render=True,
            run_id=later_id,
            deps=_deps(),
        )

    failed_metrics = json.loads(
        (
            root / "SYNTHETIC" / "runs" / later_id / "suite-metrics.json"
        ).read_text(encoding="utf-8")
    )
    assert failed_metrics["status"] == "failed"
    assert failed_metrics["reason"] == "suite_infrastructure_failure"
    assert failed_metrics["error_type"] == expected_error.__name__
    assert latest_path.read_bytes() == latest_before


def test_verify_suite_rejects_unknown_metrics_version_after_rehash(tmp_path: Path) -> None:
    config_path = write_complete_config(tmp_path)
    result = run_suite(
        config_path,
        tmp_path / "artifacts",
        variants=("B2",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    metrics_path = result.run_dir / "suite-metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["schema_version"] = 2
    metrics_path.write_text(
        json.dumps(metrics, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _refresh_suite_manifest_entry(result.run_dir, "suite-metrics.json")

    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(result.run_dir)


def test_verify_suite_rejects_unknown_benchmark_version_after_rehash(
    tmp_path: Path,
) -> None:
    config_path = write_complete_config(tmp_path)
    result = run_suite(
        config_path,
        tmp_path / "artifacts",
        variants=("B0",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    summary_path = result.run_dir / "B0" / "benchmark-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = 2
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _refresh_suite_manifest_entry(result.run_dir, "B0/benchmark-summary.json")

    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(result.run_dir)


@pytest.mark.parametrize("tamper", ("duplicate_seed", "extra_perturbation_field"))
def test_verify_suite_rejects_non_fixed_benchmark_record_semantics_after_rehash(
    tmp_path: Path, tamper: str,
) -> None:
    config_path = write_complete_config(tmp_path)
    result = run_suite(
        config_path,
        tmp_path / "artifacts",
        variants=("B0",),
        no_render=True,
        run_id=FIXED_RUN_ID,
        deps=_deps(),
    )
    records_path = result.run_dir / "B0" / "benchmark-rollouts.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    if tamper == "duplicate_seed":
        records[1]["seed"] = records[0]["seed"]
    else:
        records[0]["perturbation"]["unrecognized"] = 0.0
    records_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    _refresh_suite_manifest_entry(
        result.run_dir, "B0/benchmark-rollouts.jsonl"
    )

    with pytest.raises(ValueError, match="suite directory verification failed"):
        verify_suite_directory(result.run_dir)


def test_cli_append_only_single_variant_returns_exact_relative_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_complete_config(tmp_path)
    root = tmp_path / "artifacts"
    code = cli_module.main(
        [
            "run", "--config", str(config_path), "--variant", "B2",
            "--artifacts-root", str(root), "--run-id", FIXED_RUN_ID,
            "--no-render", "--require-completed", "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 5
    assert json.loads(captured.out) == {
        "run_id": FIXED_RUN_ID,
        "run_path": f"SYNTHETIC/runs/{FIXED_RUN_ID}",
        "requested_variants": ["B2"],
        "status": "not_run",
        "reason": "metric_depth_not_available",
    }
    assert captured.err == ""
    assert str(tmp_path.resolve()) not in captured.out + captured.err
    assert verify_suite_directory(root / "SYNTHETIC" / "runs" / FIXED_RUN_ID)


@pytest.mark.parametrize(
    ("root_arguments", "root_name"),
    (
        ((), "artifacts"),
        (("--artifacts-root", "relative-artifacts"), "relative-artifacts"),
    ),
)
def test_cli_append_only_accepts_default_and_explicit_relative_artifacts_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    root_arguments: tuple[str, ...],
    root_name: str,
) -> None:
    config_path = write_complete_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    code = cli_module.main(
        [
            "run",
            "--config",
            str(config_path),
            "--variant",
            "B2",
            *root_arguments,
            "--run-id",
            FIXED_RUN_ID,
            "--no-render",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["run_path"] == (
        f"SYNTHETIC/runs/{FIXED_RUN_ID}"
    )
    assert captured.err == ""
    assert verify_suite_directory(
        tmp_path / root_name / "SYNTHETIC" / "runs" / FIXED_RUN_ID
    )


@pytest.mark.parametrize(
    "extra",
    (
        ("--all",),
        ("--run-id", FIXED_RUN_ID),
        ("--artifacts-root", "explicit-artifacts"),
    ),
)
def test_cli_output_dir_rejects_append_only_options(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: tuple[str, ...]
) -> None:
    config_path = write_complete_config(tmp_path)
    code = cli_module.main(
        [
            "run", "--config", str(config_path), "--variant", "B2",
            "--output-dir", str(tmp_path / "output"), *extra, "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert json.loads(captured.out)["error"] == "usage_error"
    assert not (tmp_path / "output").exists()

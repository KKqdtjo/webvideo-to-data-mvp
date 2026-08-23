from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import webvideo_to_data.artifacts as artifacts_module
from webvideo_to_data.artifacts import (
    ROBOT_REFERENCE_V1,
    TRAJECTORY_2D_V1,
    baseline_control_contract,
    load_npz_artifact,
    simulation_contract,
    write_npz_artifact,
)


PROVENANCE = {
    "producer": "tests",
    "git_commit": "0" * 40,
    "source_sha256": "1" * 64,
    "config_sha256": "2" * 64,
    "model_sha256": "3" * 64,
    "terminal_status": "rejected",
    "terminal_reason": "kinematic_replay_not_action",
    "action_export_eligible": False,
}


def _kill_and_reap(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _trajectory_arrays() -> dict[str, np.ndarray]:
    return {
        "timestamps_s": np.array([0.0, 0.1], dtype=np.float64),
        "centers_px": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "confidence": np.array([1.0, 0.8], dtype=np.float64),
    }


def _write_valid_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "trajectory_2d.npz"
    write_npz_artifact(path, _trajectory_arrays(), TRAJECTORY_2D_V1, PROVENANCE)
    return path


def _rewrite_sidecar(path: Path, mutate: object) -> None:
    sidecar_path = path.with_suffix(".schema.json")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    mutate(payload)
    sidecar_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _replace_with_same_bytes_and_mtime(path: Path) -> None:
    timestamps = (path.stat().st_atime_ns, path.stat().st_mtime_ns)
    replacement = path.with_name(f".{path.name}.same-bytes")
    replacement.write_bytes(path.read_bytes())
    os.utime(replacement, ns=timestamps)
    replacement.replace(path)
    os.utime(path, ns=timestamps)


def test_npz_round_trip_checks_shape_dtype_units_frame_and_timebase(
    tmp_path: Path,
) -> None:
    path = _write_valid_fixture(tmp_path)

    loaded = load_npz_artifact(path, TRAJECTORY_2D_V1)

    np.testing.assert_array_equal(loaded["centers_px"], _trajectory_arrays()["centers_px"])
    sidecar = json.loads(path.with_suffix(".schema.json").read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == 1
    assert sidecar["contract_name"] == "trajectory_2d"
    assert sidecar["arrays"]["centers_px"] == {
        "coordinate_frame": "source_image_xy",
        "dtype": "float64",
        "semantic": "tracked_object_center",
        "shape": ["T", 2],
        "timebase": "source_seconds",
        "unit": "pixel",
    }
    assert sidecar["arrays"]["timestamps_s"]["timebase"] == "source_seconds"
    assert sidecar["npz_sha256"]
    assert sidecar["provenance"] == PROVENANCE


@pytest.mark.parametrize("mutation", ["dtype", "semantic", "unit", "frame", "schema"])
def test_loader_rejects_semantic_contract_mismatch(
    tmp_path: Path, mutation: str
) -> None:
    path = _write_valid_fixture(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        arrays = payload["arrays"]
        assert isinstance(arrays, dict)
        center = arrays["centers_px"]
        assert isinstance(center, dict)
        if mutation == "schema":
            payload["schema_version"] = 999
        elif mutation == "dtype":
            center["dtype"] = "float32"
        elif mutation == "semantic":
            center["semantic"] = "robot_joint_target"
        elif mutation == "unit":
            center["unit"] = "m"
        else:
            center["coordinate_frame"] = "robot_base"

    _rewrite_sidecar(path, mutate)

    with pytest.raises(ValueError, match="artifact contract mismatch"):
        load_npz_artifact(path, TRAJECTORY_2D_V1)


def test_loader_rejects_npz_hash_or_shape_mutation(tmp_path: Path) -> None:
    path = _write_valid_fixture(tmp_path)
    arrays = _trajectory_arrays()
    arrays["centers_px"] = np.ones((2, 3), dtype=np.float64)
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match="artifact (hash|contract) mismatch"):
        load_npz_artifact(path, TRAJECTORY_2D_V1)


def test_npz_hash_and_load_use_same_byte_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_valid_fixture(tmp_path)
    original_pair = artifacts_module._capture_artifact_pair(
        path, path.with_suffix(".schema.json")
    )
    replacement = _trajectory_arrays()
    replacement["centers_px"] = replacement["centers_px"] + 100.0
    replacement_path = tmp_path / "replacement" / "trajectory_2d.npz"
    replacement_path.parent.mkdir()
    write_npz_artifact(
        replacement_path, replacement, TRAJECTORY_2D_V1, PROVENANCE
    )
    replacement_pair = artifacts_module._capture_artifact_pair(
        replacement_path,
        replacement_path.with_suffix(".schema.json"),
    )
    captures = iter((original_pair, replacement_pair))

    monkeypatch.setattr(
        artifacts_module,
        "_capture_artifact_pair",
        lambda artifact, sidecar: next(captures),
    )

    with pytest.raises(ValueError, match="changed during validation"):
        load_npz_artifact(path, TRAJECTORY_2D_V1)


def test_loader_rejects_same_byte_pair_replacement_with_preserved_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_valid_fixture(tmp_path)
    sidecar_path = path.with_suffix(".schema.json")
    real_capture = artifacts_module._capture_artifact_pair
    captures = 0

    def capture_then_replace(artifact: Path, sidecar: Path) -> object:
        nonlocal captures
        snapshot = real_capture(artifact, sidecar)
        captures += 1
        if captures == 1:
            _replace_with_same_bytes_and_mtime(path)
            _replace_with_same_bytes_and_mtime(sidecar_path)
        return snapshot

    monkeypatch.setattr(artifacts_module, "_capture_artifact_pair", capture_then_replace)

    with pytest.raises(ValueError, match="changed during validation"):
        load_npz_artifact(path, TRAJECTORY_2D_V1)


@pytest.mark.skipif(os.name != "nt", reason="Windows zero-inode fallback")
def test_loader_uses_windows_file_id_when_fstat_inode_is_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_valid_fixture(tmp_path)
    sidecar_path = path.with_suffix(".schema.json")
    real_fstat = artifacts_module.os.fstat
    real_windows_identity = artifacts_module._windows_file_identity
    fallback_calls = 0

    def zero_inode(file_descriptor: int) -> SimpleNamespace:
        value = real_fstat(file_descriptor)
        return SimpleNamespace(
            st_mode=value.st_mode,
            st_size=value.st_size,
            st_mtime_ns=value.st_mtime_ns,
            st_dev=value.st_dev,
            st_ino=0,
        )

    def fallback_identity(file_descriptor: int) -> tuple[int, int]:
        nonlocal fallback_calls
        fallback_calls += 1
        return real_windows_identity(file_descriptor)

    real_capture = artifacts_module._capture_artifact_pair
    captures = 0

    def capture_then_replace(artifact: Path, sidecar: Path) -> object:
        nonlocal captures
        snapshot = real_capture(artifact, sidecar)
        captures += 1
        if captures == 1:
            _replace_with_same_bytes_and_mtime(path)
            _replace_with_same_bytes_and_mtime(sidecar_path)
        return snapshot

    monkeypatch.setattr(artifacts_module.os, "fstat", zero_inode)
    monkeypatch.setattr(
        artifacts_module, "_windows_file_identity", fallback_identity
    )
    monkeypatch.setattr(artifacts_module, "_capture_artifact_pair", capture_then_replace)

    with pytest.raises(ValueError, match="changed during validation"):
        load_npz_artifact(path, TRAJECTORY_2D_V1)
    assert fallback_calls == 8


def test_pair_write_failure_restores_existing_npz_and_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_valid_fixture(tmp_path)
    sidecar_path = path.with_suffix(".schema.json")
    before = (path.read_bytes(), sidecar_path.read_bytes())
    replacement = _trajectory_arrays()
    replacement["centers_px"] = replacement["centers_px"] + 10.0

    real_replace = Path.replace
    failed = False

    def fail_sidecar_replace(self: Path, target: Path) -> Path:
        nonlocal failed
        if Path(target) == sidecar_path and not failed:
            failed = True
            raise OSError("synthetic sidecar write failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_sidecar_replace)

    with pytest.raises(OSError, match="sidecar write failure"):
        write_npz_artifact(path, replacement, TRAJECTORY_2D_V1, PROVENANCE)

    assert (path.read_bytes(), sidecar_path.read_bytes()) == before


def test_pair_write_post_effect_failure_restores_existing_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_valid_fixture(tmp_path)
    sidecar_path = path.with_suffix(".schema.json")
    before = (path.read_bytes(), sidecar_path.read_bytes())
    replacement = _trajectory_arrays()
    replacement["centers_px"] += 10.0
    real_replace = Path.replace
    failed = False

    def replace_sidecar_then_fail(self: Path, target: Path) -> Path:
        nonlocal failed
        result = real_replace(self, target)
        if Path(target) == sidecar_path and not failed:
            failed = True
            raise OSError("synthetic post-effect sidecar failure")
        return result

    monkeypatch.setattr(Path, "replace", replace_sidecar_then_fail)
    with pytest.raises(OSError, match="post-effect"):
        write_npz_artifact(path, replacement, TRAJECTORY_2D_V1, PROVENANCE)
    assert (path.read_bytes(), sidecar_path.read_bytes()) == before


def test_pair_rollback_failure_preserves_backup_and_reader_rejects_mixed_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_valid_fixture(tmp_path)
    sidecar_path = path.with_suffix(".schema.json")
    replacement = _trajectory_arrays()
    replacement["centers_px"] += 10.0
    real_replace = Path.replace
    publish_failed = False
    rollback_failed = False

    def fail_publish_and_rollback(self: Path, target: Path) -> Path:
        nonlocal publish_failed, rollback_failed
        target = Path(target)
        if target == sidecar_path and self.name.endswith(".json.tmp") and not publish_failed:
            publish_failed = True
            raise OSError("synthetic sidecar publish failure")
        if target == path and ".backup-" in self.name and not rollback_failed:
            rollback_failed = True
            raise OSError("synthetic rollback failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_publish_and_rollback)
    with pytest.raises(ValueError, match="rollback failed"):
        write_npz_artifact(path, replacement, TRAJECTORY_2D_V1, PROVENANCE)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_npz_artifact(path, TRAJECTORY_2D_V1)
    assert list(tmp_path.glob(".trajectory_2d.npz.backup-*"))


def test_concurrent_pair_writers_leave_one_coherent_pair(tmp_path: Path) -> None:
    path = tmp_path / "trajectory_2d.npz"
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    payloads = []
    for offset in (10.0, 20.0):
        arrays = _trajectory_arrays()
        arrays["centers_px"] += offset
        payloads.append(arrays)

    def writer(arrays: dict[str, np.ndarray]) -> None:
        try:
            barrier.wait()
            write_npz_artifact(path, arrays, TRAJECTORY_2D_V1, PROVENANCE)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=writer, args=(arrays,)) for arrays in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert not errors
    loaded = load_npz_artifact(path, TRAJECTORY_2D_V1)
    assert any(np.array_equal(loaded["centers_px"], arrays["centers_px"]) for arrays in payloads)


def test_physical_alias_pair_lock_serializes_a_real_second_process(tmp_path: Path) -> None:
    path = (tmp_path / "alias-pair" / "trajectory_2d.npz").resolve()
    path.parent.mkdir()
    lock_ready = tmp_path / "lock-ready"
    call_lock = tmp_path / "call-lock"
    writer_attempting = tmp_path / "writer-attempting"
    writer_acquired = tmp_path / "writer-acquired"
    writer_done = tmp_path / "writer-done"
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = dict(os.environ, PYTHONPATH=str(source_root))
    writer_code = """
import os, sys, time
from pathlib import Path
import numpy as np
import webvideo_to_data.artifacts as artifacts
path, ready, call_lock, attempting, acquired, done = map(Path, sys.argv[1:])
if os.name == 'nt':
    import msvcrt
    real_lock = msvcrt.locking
    first = True
    def gated_lock(fd, mode, count):
        global first
        if mode == msvcrt.LK_NBLCK and first:
            first = False
            ready.write_text('ready', encoding='utf-8')
            while not call_lock.exists():
                time.sleep(0.01)
            attempting.write_text('attempting', encoding='utf-8')
        result = real_lock(fd, mode, count)
        if mode == msvcrt.LK_NBLCK:
            acquired.write_text('acquired', encoding='utf-8')
        return result
    msvcrt.locking = gated_lock
else:
    import fcntl
    real_lock = fcntl.flock
    first = True
    def gated_lock(fd, operation):
        global first
        if operation & fcntl.LOCK_EX and first:
            first = False
            ready.write_text('ready', encoding='utf-8')
            while not call_lock.exists():
                time.sleep(0.01)
            attempting.write_text('attempting', encoding='utf-8')
        result = real_lock(fd, operation)
        if operation & fcntl.LOCK_EX:
            acquired.write_text('acquired', encoding='utf-8')
        return result
    fcntl.flock = gated_lock
arrays = {
    'timestamps_s': np.array([0.0, 0.1], dtype=np.float64),
    'centers_px': np.array([[21.0, 22.0], [23.0, 24.0]], dtype=np.float64),
    'confidence': np.array([1.0, 0.8], dtype=np.float64),
}
provenance = {
    'producer': 'tests', 'git_commit': '0' * 40,
    'source_sha256': '1' * 64, 'config_sha256': '2' * 64,
    'model_sha256': '3' * 64, 'terminal_status': 'rejected',
    'terminal_reason': 'kinematic_replay_not_action',
    'action_export_eligible': False,
}
artifacts.write_npz_artifact(path, arrays, artifacts.TRAJECTORY_2D_V1, provenance)
done.write_text('done', encoding='utf-8')
"""
    alias = Path("\\\\?\\" + str(path)) if os.name == "nt" else path
    writer: subprocess.Popen[bytes] | None = None
    try:
        with artifacts_module._serialized_pair(path):
            writer = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    writer_code,
                    str(alias),
                    str(lock_ready),
                    str(call_lock),
                    str(writer_attempting),
                    str(writer_acquired),
                    str(writer_done),
                ],
                cwd=source_root.parent,
                env=environment,
            )
            deadline = time.monotonic() + 15.0
            while not lock_ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert lock_ready.exists()
            call_lock.write_text("call", encoding="utf-8")
            attempt_deadline = time.monotonic() + 1.0
            while not writer_attempting.exists() and time.monotonic() < attempt_deadline:
                time.sleep(0.01)
            assert writer_attempting.exists()
            with pytest.raises(subprocess.TimeoutExpired):
                writer.wait(timeout=1.0)
            assert not writer_acquired.exists()
            assert not writer_done.exists()
            assert writer.poll() is None
        acquired_deadline = time.monotonic() + 15.0
        while not writer_acquired.exists() and time.monotonic() < acquired_deadline:
            time.sleep(0.01)
        assert writer_acquired.exists()
        done_deadline = time.monotonic() + 15.0
        while not writer_done.exists() and time.monotonic() < done_deadline:
            time.sleep(0.01)
        assert writer_done.exists()
        assert writer.wait(timeout=15.0) == 0
    finally:
        if writer is not None:
            _kill_and_reap(writer)
        for marker in (
            lock_ready,
            call_lock,
            writer_attempting,
            writer_acquired,
            writer_done,
        ):
            marker.unlink(missing_ok=True)
    loaded = load_npz_artifact(path, TRAJECTORY_2D_V1)
    np.testing.assert_array_equal(
        loaded["centers_px"],
        np.array([[21.0, 22.0], [23.0, 24.0]], dtype=np.float64),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 1.0),
        ("npz_sha256", "g" * 64),
        ("provenance", []),
    ],
)
def test_loader_rejects_inexact_json_types(
    tmp_path: Path, field: str, value: object
) -> None:
    path = _write_valid_fixture(tmp_path)
    _rewrite_sidecar(path, lambda payload: payload.update({field: value}))

    with pytest.raises(ValueError, match="artifact contract mismatch"):
        load_npz_artifact(path, TRAJECTORY_2D_V1)


def test_loader_rejects_bool_disguised_as_integer_in_contract_shape(tmp_path: Path) -> None:
    path = _write_valid_fixture(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        payload["arrays"]["centers_px"]["shape"] = ["T", True]

    _rewrite_sidecar(path, mutate)
    with pytest.raises(ValueError, match="artifact contract mismatch"):
        load_npz_artifact(path, TRAJECTORY_2D_V1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("git_commit", "0" * 39),
        ("terminal_reason", " rejected "),
        ("terminal_status", "unknown"),
    ],
)
def test_writer_rejects_inexact_provenance_values(
    tmp_path: Path, field: str, value: object
) -> None:
    provenance = dict(PROVENANCE)
    provenance[field] = value

    with pytest.raises(ValueError, match="provenance"):
        write_npz_artifact(
            tmp_path / "trajectory_2d.npz",
            _trajectory_arrays(),
            TRAJECTORY_2D_V1,
            provenance,
        )


@pytest.mark.parametrize(
    ("status", "field", "value"),
    [
        ("rejected", "rejection_stage", True),
        ("rejected", "rejection_stage", " tracking "),
        ("failed", "failure_stage", True),
        ("failed", "failure_stage", " publication "),
        ("failed", "error_type", " "),
        ("failed", "error_message", True),
    ],
)
def test_terminal_stage_and_error_diagnostics_are_exact_trimmed_strings(
    status: str, field: str, value: object
) -> None:
    if status == "rejected":
        metrics = {
            "status": "rejected",
            "variant": "B1",
            "reason": "kinematic_replay_not_action",
            "rejection_stage": "tracking",
            "placed_successfully": False,
            "collision_validation": "not_applicable_kinematic",
            "physics_validation": "not_applicable_kinematic",
            "action_export_eligible": False,
            "action_export_reason": "kinematic_replay_not_action",
            "action_exported": False,
        }
    else:
        metrics = {
            "status": "failed",
            "variant": "B2",
            "reason": "stage_exception",
            "failure_stage": "publication",
            "error_type": "OSError",
            "error_message": "synthetic",
            "placed_successfully": False,
            "collision_validation": "not_run",
            "physics_validation": "not_run",
            "action_export_eligible": False,
            "action_export_reason": "metric_depth_not_available",
            "action_exported": False,
        }
    metrics[field] = value
    assert artifacts_module.terminal_metrics_error(
        metrics, {"metrics.json", "rejection.json"}
    ) is not None


@pytest.mark.parametrize(
    "namespace",
    (
        pytest.param(
            "src/webvideo_to_data/assets/mujoco_menagerie/franka_emika_panda",
            id="B0-menagerie",
        ),
        pytest.param(
            "src/webvideo_to_data/assets/panda_pick_place.xml",
            id="B1-primitive",
        ),
    ),
)
def test_pinned_asset_checkout_bytes_equal_index_with_autocrlf_true(
    tmp_path: Path, namespace: str
) -> None:
    """Catch checkout conversion changing bytes used by pinned model identity."""

    repository = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--", namespace],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert tracked
    checkout = tmp_path / "autocrlf-checkout"
    checkout.mkdir()
    prefix = str(checkout.resolve()) + os.sep
    subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "checkout-index",
            "--force",
            "--stdin",
            "-z",
            f"--prefix={prefix}",
        ],
        cwd=repository,
        input=("\0".join(tracked) + "\0").encode("utf-8"),
        check=True,
        capture_output=True,
    )

    mismatches = []
    for relative in tracked:
        index_blob = subprocess.run(
            ["git", "show", f":{relative}"],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        if (checkout / relative).read_bytes() != index_blob:
            mismatches.append(relative)
    assert mismatches == []


def test_pinned_dependency_identity_rejects_escape_and_cycles(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    outside = tmp_path / "outside.xml"
    outside.write_text("<mujoco/>", encoding="utf-8")
    include_escape = trusted / "include_escape.xml"
    include_escape.write_text(
        '<mujoco><include file="../outside.xml"/></mujoco>', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="parent path component"):
        artifacts_module._pinned_dependency_identity(include_escape, trusted)

    asset_escape = trusted / "asset_escape.xml"
    asset_escape.write_text(
        '<mujoco><asset><mesh file="../outside.obj"/></asset></mujoco>',
        encoding="utf-8",
    )
    (tmp_path / "outside.obj").write_bytes(b"outside")
    with pytest.raises(ValueError, match="parent path component"):
        artifacts_module._pinned_dependency_identity(asset_escape, trusted)

    first = trusted / "first.xml"
    second = trusted / "second.xml"
    first.write_text('<mujoco><include file="second.xml"/></mujoco>', encoding="utf-8")
    second.write_text('<mujoco><include file="first.xml"/></mujoco>', encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        artifacts_module._pinned_dependency_identity(first, trusted)


def test_pinned_dependency_identity_tracks_assetdir_resources(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted"
    assets = trusted / "assets"
    assets.mkdir(parents=True)
    mesh = assets / "mesh.obj"
    mesh.write_bytes(b"mesh-v1")
    root = trusted / "root.xml"
    root.write_text(
        """<mujoco>
  <compiler assetdir="assets"/>
  <asset>
    <mesh name="mesh" file="mesh.obj"/>
  </asset>
</mujoco>""",
        encoding="utf-8",
    )
    before = artifacts_module._pinned_dependency_identity(root, trusted)
    mesh.write_bytes(b"mesh-v2")
    after_mesh = artifacts_module._pinned_dependency_identity(root, trusted)

    assert before != after_mesh


def test_pinned_dependency_identity_resolves_nested_includes_from_main_directory(
    tmp_path: Path,
) -> None:
    trusted = tmp_path / "trusted"
    (trusted / "sub").mkdir(parents=True)
    root = trusted / "root.xml"
    outer = trusted / "sub" / "outer.xml"
    inner = trusted / "inner.xml"
    root.write_text('<mujoco><include file="sub/outer.xml"/></mujoco>', encoding="utf-8")
    outer.write_text('<mujoco><include file="inner.xml"/></mujoco>', encoding="utf-8")
    inner.write_text("<mujoco/>", encoding="utf-8")

    before = artifacts_module._pinned_dependency_identity(root, trusted)
    inner.write_text('<mujoco model="changed"/>', encoding="utf-8")
    after = artifacts_module._pinned_dependency_identity(root, trusted)

    assert before != after


def test_pinned_dependency_identity_rejects_parent_component_inside_root(
    tmp_path: Path,
) -> None:
    trusted = tmp_path / "trusted"
    (trusted / "sub").mkdir(parents=True)
    included = trusted / "included.xml"
    included.write_text("<mujoco/>", encoding="utf-8")
    root = trusted / "root.xml"
    root.write_text(
        '<mujoco><include file="sub/../included.xml"/></mujoco>', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="parent path component"):
        artifacts_module._pinned_dependency_identity(root, trusted)


def test_pinned_dependency_identity_applies_included_compiler_globally(
    tmp_path: Path,
) -> None:
    trusted = tmp_path / "trusted"
    assets = trusted / "assets"
    assets.mkdir(parents=True)
    mesh = assets / "mesh.obj"
    mesh.write_bytes(b"mesh-v1")
    (trusted / "settings.xml").write_text(
        '<mujoco><compiler assetdir="assets"/></mujoco>', encoding="utf-8"
    )
    root = trusted / "root.xml"
    root.write_text(
        '<mujoco><include file="settings.xml"/><asset><mesh file="mesh.obj"/></asset></mujoco>',
        encoding="utf-8",
    )

    before = artifacts_module._pinned_dependency_identity(root, trusted)
    mesh.write_bytes(b"mesh-v2")
    after = artifacts_module._pinned_dependency_identity(root, trusted)

    assert before != after


def test_pinned_dependency_identity_uses_compiler_order_and_explicit_meshdir(
    tmp_path: Path,
) -> None:
    trusted = tmp_path / "trusted"
    final_meshes = trusted / "final-meshes"
    final_meshes.mkdir(parents=True)
    mesh = final_meshes / "mesh.obj"
    mesh.write_bytes(b"mesh-v1")
    (trusted / "settings.xml").write_text(
        '<mujoco><compiler assetdir="unused" meshdir="final-meshes"/></mujoco>',
        encoding="utf-8",
    )
    root = trusted / "root.xml"
    root.write_text(
        """<mujoco>
  <compiler assetdir="also-unused" meshdir="old-meshes"/>
  <include file="settings.xml"/>
  <compiler assetdir="last-unused"/>
  <asset><mesh file="mesh.obj"/></asset>
</mujoco>""",
        encoding="utf-8",
    )

    before = artifacts_module._pinned_dependency_identity(root, trusted)
    mesh.write_bytes(b"mesh-v2")
    after = artifacts_module._pinned_dependency_identity(root, trusted)

    assert before != after


def test_pinned_dependency_identity_rejects_unmodeled_file_asset_tag(
    tmp_path: Path,
) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    (trusted / "texture.png").write_bytes(b"texture")
    root = trusted / "root.xml"
    root.write_text(
        '<mujoco><asset><texture file="texture.png"/></asset></mujoco>',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported pinned file-bearing tag"):
        artifacts_module._pinned_dependency_identity(root, trusted)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra", "array set"),
        ("dtype", "dtype"),
        ("length", "leading dimension"),
        ("nonfinite", "finite"),
        ("timestamps", "timestamps_s must be strictly increasing"),
    ],
)
def test_writer_rejects_invalid_array_payloads(
    tmp_path: Path, mutation: str, message: str
) -> None:
    arrays = _trajectory_arrays()
    if mutation == "extra":
        arrays["unexpected"] = np.ones(2, dtype=np.float64)
    elif mutation == "dtype":
        arrays["confidence"] = arrays["confidence"].astype(np.float32)
    elif mutation == "length":
        arrays["confidence"] = np.ones(3, dtype=np.float64)
    elif mutation == "nonfinite":
        arrays["centers_px"][1, 0] = np.nan
    else:
        arrays["timestamps_s"] = np.array([0.1, 0.1], dtype=np.float64)

    with pytest.raises(ValueError, match=message):
        write_npz_artifact(
            tmp_path / "trajectory_2d.npz",
            arrays,
            TRAJECTORY_2D_V1,
            PROVENANCE,
        )


def test_writer_rejects_pickle_backed_phase_array(tmp_path: Path) -> None:
    arrays = {
        "timestamps_s": np.array([0.0, 0.1], dtype=np.float64),
        "ee_positions": np.zeros((2, 3), dtype=np.float64),
        "quaternion_wxyz": np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (2, 1)
        ),
        "gripper_width": np.zeros(2, dtype=np.float64),
        "phase": np.array(["close", "lift"], dtype=object),
    }

    with pytest.raises(ValueError, match="object arrays are forbidden"):
        write_npz_artifact(
            tmp_path / "robot_reference.npz",
            arrays,
            ROBOT_REFERENCE_V1,
            PROVENANCE,
        )


def test_robot_reference_uses_fixed_unicode_and_wxyz_contract(tmp_path: Path) -> None:
    arrays = {
        "timestamps_s": np.array([0.0, 0.1], dtype=np.float64),
        "ee_positions": np.zeros((2, 3), dtype=np.float64),
        "quaternion_wxyz": np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (2, 1)
        ),
        "gripper_width": np.zeros(2, dtype=np.float64),
        "phase": np.array(["close", "lift"], dtype="<U16"),
    }
    path = tmp_path / "robot_reference.npz"

    write_npz_artifact(path, arrays, ROBOT_REFERENCE_V1, PROVENANCE)
    loaded = load_npz_artifact(path, ROBOT_REFERENCE_V1)

    assert loaded["phase"].dtype == np.dtype("<U16")
    sidecar = json.loads(path.with_suffix(".schema.json").read_text(encoding="utf-8"))
    assert sidecar["quaternion_order"] == "wxyz"
    _rewrite_sidecar(path, lambda payload: payload.update(quaternion_order="xyzw"))
    with pytest.raises(ValueError, match="artifact contract mismatch"):
        load_npz_artifact(path, ROBOT_REFERENCE_V1)


def test_loader_rejects_meter_unit_rewritten_as_pixel(tmp_path: Path) -> None:
    arrays = {
        "timestamps_s": np.array([0.0, 0.1], dtype=np.float64),
        "ee_positions": np.zeros((2, 3), dtype=np.float64),
        "quaternion_wxyz": np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (2, 1)
        ),
        "gripper_width": np.zeros(2, dtype=np.float64),
        "phase": np.array(["close", "lift"], dtype="<U16"),
    }
    path = tmp_path / "robot_reference.npz"
    write_npz_artifact(path, arrays, ROBOT_REFERENCE_V1, PROVENANCE)

    def replace_unit(payload: dict[str, object]) -> None:
        declared = payload["arrays"]
        assert isinstance(declared, dict)
        positions = declared["ee_positions"]
        assert isinstance(positions, dict)
        positions["unit"] = "pixel"

    _rewrite_sidecar(path, replace_unit)

    with pytest.raises(ValueError, match="artifact contract mismatch"):
        load_npz_artifact(path, ROBOT_REFERENCE_V1)


def test_dynamic_contracts_pin_all_model_widths_and_exact_array_sets() -> None:
    baseline = baseline_control_contract(nu=8)
    simulation = simulation_contract(nq=16, nv=15, nu=8)

    assert set(baseline.arrays) == {"timestamps_s", "control", "phase"}
    assert baseline.arrays["control"].trailing_shape == (8,)
    assert set(simulation.arrays) == {
        "timestamps_s",
        "control",
        "qpos",
        "qvel",
        "can_pose",
        "tcp_position",
        "tcp_quaternion_wxyz",
        "phase",
        "contact_count",
        "bilateral_contact",
        "box_support_contact",
        "forbidden_contact",
        "maximum_penetration_m",
        "tcp_position_within_tolerance",
        "tcp_orientation_within_tolerance",
        "joint_position_violation",
        "joint_velocity_violation",
        "joint_acceleration_violation",
        "valid_numerical_state",
    }
    assert simulation.arrays["control"].trailing_shape == (8,)
    assert simulation.arrays["qpos"].trailing_shape == (16,)
    assert simulation.arrays["qvel"].trailing_shape == (15,)
    for array in simulation.arrays.values():
        assert array.semantic
        assert array.dtype
        assert array.unit
        assert array.coordinate_frame
        assert array.timebase


def test_sidecar_cannot_self_declare_a_different_dynamic_width(tmp_path: Path) -> None:
    contract = baseline_control_contract(nu=8)
    path = tmp_path / "baseline_control_trace.npz"
    arrays = {
        "timestamps_s": np.array([0.0, 0.1], dtype=np.float64),
        "control": np.zeros((2, 8), dtype=np.float64),
        "phase": np.array(["close", "lift"], dtype="<U16"),
    }
    write_npz_artifact(path, arrays, contract, PROVENANCE)

    def widen(payload: dict[str, object]) -> None:
        declared = payload["arrays"]
        assert isinstance(declared, dict)
        control = declared["control"]
        assert isinstance(control, dict)
        control["shape"] = ["T", 9]

    _rewrite_sidecar(path, widen)

    with pytest.raises(ValueError, match="artifact contract mismatch"):
        load_npz_artifact(path, baseline_control_contract(nu=8))


def test_dynamic_npz_width_cannot_override_pinned_model_contract(tmp_path: Path) -> None:
    path = tmp_path / "baseline_control_trace.npz"
    arrays = {
        "timestamps_s": np.array([0.0, 0.1], dtype=np.float64),
        "control": np.zeros((2, 9), dtype=np.float64),
        "phase": np.array(["close", "lift"], dtype="<U16"),
    }

    with pytest.raises(ValueError, match="shape"):
        write_npz_artifact(
            path, arrays, baseline_control_contract(nu=8), PROVENANCE
        )


def test_directory_capture_precharges_memory_caps_before_file_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oversized = tmp_path / "oversized.bin"
    oversized.write_bytes(b"x" * 257)
    monkeypatch.setattr(
        artifacts_module, "_RUN_CAPTURE_MAX_FILE_BYTES", 256, raising=False
    )
    reads: list[Path] = []
    original_read = artifacts_module._stable_file_bytes

    def record_read(path: Path, *args: object, **kwargs: object) -> object:
        reads.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(artifacts_module, "_stable_file_bytes", record_read)

    with pytest.raises(ValueError, match="capture.*cap"):
        artifacts_module._capture_directory_bytes(tmp_path)

    assert oversized not in reads


def test_directory_capture_enumerates_at_most_one_entry_past_file_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(5):
        (tmp_path / f"entry-{index}.bin").write_bytes(b"x")
    monkeypatch.setattr(
        artifacts_module, "_RUN_CAPTURE_MAX_FILES", 2, raising=False
    )
    original_iterdir = Path.iterdir
    yielded: list[Path] = []

    def counted_iterdir(directory: Path):
        children = original_iterdir(directory)
        if directory != tmp_path:
            return children

        def counted_children():
            for child in children:
                yielded.append(child)
                yield child

        return counted_children()

    monkeypatch.setattr(Path, "iterdir", counted_iterdir)

    with pytest.raises(ValueError, match="file cap exceeded"):
        artifacts_module._capture_directory_bytes(tmp_path)

    assert len(yielded) == 3


def test_directory_capture_still_rejects_link_entries(
    tmp_path: Path,
) -> None:
    target = tmp_path.parent / f"{tmp_path.name}-link-target"
    target.mkdir()
    link = tmp_path / "linked-entry"
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout
    else:
        link.symlink_to(target, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="regular file"):
            artifacts_module._capture_directory_bytes(tmp_path)
    finally:
        if os.name == "nt":
            link.rmdir()
        else:
            link.unlink()


def test_directory_capture_preserves_enumeration_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "entry.bin"
    candidate.write_bytes(b"x")

    def interrupted_iterdir(directory: Path):
        assert directory == tmp_path
        yield candidate
        raise OSError("enumeration interrupted")

    monkeypatch.setattr(Path, "iterdir", interrupted_iterdir)

    with pytest.raises(OSError, match="enumeration interrupted"):
        artifacts_module._capture_directory_bytes(tmp_path)


def test_bounded_file_capture_stops_if_open_file_grows_past_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "growing.bin"
    candidate.write_bytes(b"x")
    requested: list[int] = []

    def growing_read(file_descriptor: int, size: int) -> bytes:
        requested.append(size)
        return b"x" * min(size, 2)

    monkeypatch.setattr(artifacts_module.os, "read", growing_read)

    with pytest.raises(ValueError, match="memory cap"):
        artifacts_module._stable_file_bytes(candidate, max_bytes=2)

    assert requested == [3, 1]

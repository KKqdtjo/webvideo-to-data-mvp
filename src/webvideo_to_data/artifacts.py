"""Versioned, self-describing NumPy artifacts for trusted experiment runs."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import gc
from hashlib import sha256
from io import BytesIO
from itertools import islice
import json
import os
from pathlib import Path
import stat as stat_module
import tempfile
import threading
import time
from typing import Any
from uuid import uuid4
from xml.etree import ElementTree

import mujoco
import numpy as np
from numpy.typing import NDArray

from .scene import DEFAULT_SCENE_PATH
from .simulation import _DEFAULT_MODEL_PATH


_ARTIFACT_FORMAT = "webvideo_to_data.npz"
_RUN_MANIFEST_PRODUCER = "webvideo_to_data.experiment"
_V3_MANIFEST_FIELDS = {"producer", "format_version", "variant", "status", "files"}
_V4_MANIFEST_FIELDS = {
    "producer",
    "format_version",
    "variant",
    "status",
    "reason",
    "action_export_eligible",
    "action_export_reason",
    "action_exported",
    "config_sha256",
    "source_sha256",
    "model_sha256",
    "model_nq",
    "model_nv",
    "model_nu",
    "files",
}
_MEDIA_SUFFIXES = {".gif", ".mp4", ".png"}
_EXPECTED_MEDIA_CLASSIFICATION: dict[str, tuple[str, bool]] = {
    "contact_sheet.png": ("source_contact_sheet", True),
    "mujoco_replay.mp4": ("simulation_only", False),
    "side_by_side.mp4": ("source_simulation_comparison", True),
    "tracking_overlay.mp4": ("source_tracking_overlay", True),
    "trajectory_2d.png": ("derived_trajectory_plot", False),
}
_RUN_CAPTURE_MAX_FILES = 256
_RUN_CAPTURE_MAX_FILE_BYTES = 128 * 1024 * 1024
_RUN_CAPTURE_MAX_AGGREGATE_BYTES = 256 * 1024 * 1024
_PROVENANCE_FIELDS = {
    "producer",
    "git_commit",
    "source_sha256",
    "config_sha256",
    "model_sha256",
    "terminal_status",
    "terminal_reason",
    "action_export_eligible",
}


@dataclass(frozen=True)
class ArrayContract:
    dtype: str
    trailing_shape: tuple[int, ...]
    semantic: str
    unit: str
    coordinate_frame: str
    timebase: str | None = None


@dataclass(frozen=True)
class NPZContract:
    name: str
    schema_version: int
    arrays: Mapping[str, ArrayContract]


@dataclass(frozen=True)
class VerifiedRun:
    path: Path
    metrics: Mapping[str, Any]
    manifest: Mapping[str, Any]
    directory_identity: tuple[int, int]
    snapshot: Mapping[str, tuple[int, str, int, int, int]]


@dataclass(frozen=True)
class _FileBytes:
    size: int
    mtime_ns: int
    device: int
    inode: int
    content: bytes


@dataclass(frozen=True)
class _DirectoryBytes:
    directory_identity: tuple[int, int]
    files: Mapping[str, _FileBytes]


@dataclass(frozen=True)
class _ArtifactPairBytes:
    artifact: _FileBytes
    sidecar: _FileBytes


_PAIR_LOCKS_GUARD = threading.Lock()
_PAIR_LOCKS: dict[str, threading.Lock] = {}


def _array(
    dtype: str,
    trailing_shape: tuple[int, ...],
    semantic: str,
    unit: str,
    coordinate_frame: str,
    timebase: str,
) -> ArrayContract:
    return ArrayContract(
        dtype=dtype,
        trailing_shape=trailing_shape,
        semantic=semantic,
        unit=unit,
        coordinate_frame=coordinate_frame,
        timebase=timebase,
    )


TRAJECTORY_2D_V1 = NPZContract(
    name="trajectory_2d",
    schema_version=1,
    arrays={
        "timestamps_s": _array(
            "float64", (), "source_frame_timestamp", "s", "source_image_xy", "source_seconds"
        ),
        "centers_px": _array(
            "float64", (2,), "tracked_object_center", "pixel", "source_image_xy", "source_seconds"
        ),
        "confidence": _array(
            "float64", (), "tracking_point_availability", "ratio", "source_image_xy", "source_seconds"
        ),
    },
)


ROBOT_REFERENCE_V1 = NPZContract(
    name="robot_reference",
    schema_version=1,
    arrays={
        "timestamps_s": _array(
            "float64", (), "reference_timestamp", "s", "robot_base", "source_seconds"
        ),
        "ee_positions": _array(
            "float64", (3,), "commanded_tcp_position", "m", "robot_base", "source_seconds"
        ),
        "quaternion_wxyz": _array(
            "float64", (4,), "commanded_tcp_orientation", "dimensionless", "robot_base", "source_seconds"
        ),
        "gripper_width": _array(
            "float64", (), "commanded_gripper_width", "m", "robot_base", "source_seconds"
        ),
        "phase": _array(
            "<U16", (), "commanded_motion_phase", "category", "robot_base", "source_seconds"
        ),
    },
)


def baseline_control_contract(nu: int) -> NPZContract:
    if not isinstance(nu, int) or isinstance(nu, bool) or nu <= 0:
        raise ValueError("nu must be a positive integer")
    return NPZContract(
        name="baseline_control_trace",
        schema_version=1,
        arrays={
            "timestamps_s": _array(
                "float64", (), "control_timestamp", "s", "robot_base", "simulation_seconds"
            ),
            "control": _array(
                "float64", (nu,), "commanded_actuator_control", "actuator_native", "robot_base", "simulation_seconds"
            ),
            "phase": _array(
                "<U16", (), "commanded_motion_phase", "category", "robot_base", "simulation_seconds"
            ),
        },
    )


def simulation_contract(nq: int, nv: int, nu: int) -> NPZContract:
    dimensions = {"nq": nq, "nv": nv, "nu": nu}
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in dimensions.values()
    ):
        raise ValueError("nq, nv, and nu must be positive integers")
    observed = "simulation_seconds"
    arrays = {
        "timestamps_s": _array(
            "float64", (), "simulation_timestamp", "s", "world", observed
        ),
        "control": _array(
            "float64", (nu,), "commanded_actuator_control", "actuator_native", "robot_base", observed
        ),
        "qpos": _array(
            "float64", (nq,), "measured_joint_position", "mixed_si", "robot_base", observed
        ),
        "qvel": _array(
            "float64", (nv,), "measured_joint_velocity", "mixed_si", "robot_base", observed
        ),
        "can_pose": _array(
            "float64", (7,), "measured_can_pose", "mixed_si", "world", observed
        ),
        "tcp_position": _array(
            "float64", (3,), "measured_tcp_position", "m", "world", observed
        ),
        "tcp_quaternion_wxyz": _array(
            "float64", (4,), "measured_tcp_orientation", "dimensionless", "world", observed
        ),
        "phase": _array(
            "<U16", (), "executed_motion_phase", "category", "robot_base", observed
        ),
        "contact_count": _array(
            "int64", (), "measured_contact_count", "count", "world", observed
        ),
        "bilateral_contact": _array(
            "bool", (), "bilateral_gripper_contact_gate", "boolean", "world", observed
        ),
        "box_support_contact": _array(
            "bool", (), "box_support_contact_gate", "boolean", "world", observed
        ),
        "forbidden_contact": _array(
            "bool", (), "contact_policy_violation", "boolean", "world", observed
        ),
        "maximum_penetration_m": _array(
            "float64", (), "maximum_forbidden_contact_penetration", "m", "world", observed
        ),
        "tcp_position_within_tolerance": _array(
            "bool", (), "tcp_position_tracking_gate", "boolean", "robot_base", observed
        ),
        "tcp_orientation_within_tolerance": _array(
            "bool", (), "tcp_orientation_tracking_gate", "boolean", "robot_base", observed
        ),
        "joint_position_violation": _array(
            "bool", (), "joint_position_limit_violation", "boolean", "robot_base", observed
        ),
        "joint_velocity_violation": _array(
            "bool", (), "joint_velocity_limit_violation", "boolean", "robot_base", observed
        ),
        "joint_acceleration_violation": _array(
            "bool", (), "joint_acceleration_limit_violation", "boolean", "robot_base", observed
        ),
        "valid_numerical_state": _array(
            "bool", (), "finite_simulation_state_gate", "boolean", "world", observed
        ),
    }
    return NPZContract(name="simulation", schema_version=1, arrays=arrays)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _contract_arrays(contract: NPZContract) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": array.dtype,
            "shape": ["T", *array.trailing_shape],
            "semantic": array.semantic,
            "unit": array.unit,
            "coordinate_frame": array.coordinate_frame,
            "timebase": array.timebase,
        }
        for name, array in contract.arrays.items()
    }


def _exact_json_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _has_quaternions(contract: NPZContract) -> bool:
    return any("quaternion" in name for name in contract.arrays)


def _validate_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    if set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("artifact provenance fields mismatch")
    copied = dict(provenance)
    string_fields = _PROVENANCE_FIELDS - {"action_export_eligible"}
    if any(not isinstance(copied[name], str) or not copied[name] for name in string_fields):
        raise ValueError("artifact provenance string fields must be nonempty")
    if not isinstance(copied["action_export_eligible"], bool):
        raise ValueError("artifact provenance action eligibility must be boolean")
    if not _is_sha256(copied["source_sha256"]) and copied["source_sha256"] != "not_used":
        raise ValueError("artifact provenance source hash is invalid")
    if not _is_sha256(copied["config_sha256"]) or not _is_sha256(
        copied["model_sha256"]
    ):
        raise ValueError("artifact provenance hash is invalid")
    commit = copied["git_commit"]
    if not _is_git_commit(commit):
        raise ValueError("artifact provenance git commit is invalid")
    if copied["terminal_status"] not in {"completed", "rejected", "failed", "not_run"}:
        raise ValueError("artifact provenance terminal status is invalid")
    reason = copied["terminal_reason"]
    if reason != reason.strip():
        raise ValueError("artifact provenance terminal reason is invalid")
    return copied


def _validate_arrays(
    arrays: Mapping[str, NDArray[Any]], contract: NPZContract
) -> dict[str, NDArray[Any]]:
    if set(arrays) != set(contract.arrays):
        raise ValueError("artifact array set mismatch")
    validated: dict[str, NDArray[Any]] = {}
    frame_count: int | None = None
    for name, expected in contract.arrays.items():
        value = np.asarray(arrays[name])
        if value.dtype.hasobject:
            raise ValueError("object arrays are forbidden")
        if value.dtype != np.dtype(expected.dtype):
            raise ValueError(f"artifact dtype mismatch for {name}")
        expected_shape = (value.shape[0], *expected.trailing_shape) if value.ndim else None
        if value.ndim == 0 or value.shape != expected_shape:
            raise ValueError(f"artifact shape mismatch for {name}")
        if frame_count is None:
            frame_count = value.shape[0]
            if frame_count == 0:
                raise ValueError("artifact leading dimension T must be positive")
        elif value.shape[0] != frame_count:
            raise ValueError("artifact leading dimension T mismatch")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"artifact values must be finite for {name}")
        validated[name] = value
    timestamps = validated.get("timestamps_s")
    if timestamps is not None and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps_s must be strictly increasing")
    return validated


@contextmanager
def _serialized_pair(path: Path) -> Any:
    parent = path.parent.resolve(strict=True)
    parent_identity = _directory_identity(parent)
    lock_key = ":".join(
        (str(parent_identity[0]), str(parent_identity[1]), os.path.normcase(path.name))
    )
    with _PAIR_LOCKS_GUARD:
        thread_lock = _PAIR_LOCKS.setdefault(lock_key, threading.Lock())
    lock_root = Path(tempfile.gettempdir()) / "webvideo_to_data_artifact_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / (sha256(lock_key.encode("utf-8")).hexdigest() + ".lock")
    with thread_lock:
        with lock_path.open("a+b") as lock_file:
            if lock_file.seek(0, os.SEEK_END) == 0:
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
                        time.sleep(0.01)
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


def _temporary_npz(parent: Path, arrays: Mapping[str, NDArray[Any]]) -> Path:
    with tempfile.NamedTemporaryFile(mode="w+b", dir=parent, delete=False, suffix=".npz.tmp") as temporary:
        np.savez(temporary, **arrays)
        temporary.flush()
        os.fsync(temporary.fileno())
        return Path(temporary.name)


def _temporary_json(parent: Path, payload: Mapping[str, Any]) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=parent, delete=False, suffix=".json.tmp") as temporary:
        json.dump(payload, temporary, ensure_ascii=False, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        return Path(temporary.name)


def _backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("artifact pair target must be a regular file")
    backup = path.parent / f".{path.name}.backup-{uuid4().hex}"
    with path.open("rb") as source, backup.open("xb") as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
    return backup


def _restore_pair(
    artifact_path: Path,
    sidecar_path: Path,
    artifact_backup: Path | None,
    sidecar_backup: Path | None,
) -> None:
    errors: list[Exception] = []
    for target, backup in ((artifact_path, artifact_backup), (sidecar_path, sidecar_backup)):
        try:
            if backup is None:
                if target.exists():
                    if target.is_symlink() or not target.is_file():
                        raise ValueError("artifact pair rollback target changed")
                    target.unlink()
            else:
                backup.replace(target)
        except Exception as error:
            errors.append(error)
    if errors:
        raise ValueError("artifact pair rollback failed; backups preserved") from errors[0]


def write_npz_artifact(
    path: str | Path,
    arrays: Mapping[str, NDArray[Any]],
    contract: NPZContract,
    provenance: Mapping[str, Any],
) -> tuple[Path, Path]:
    artifact_path = Path(path)
    if artifact_path.suffix != ".npz":
        raise ValueError("artifact path must end in .npz")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    validated = _validate_arrays(arrays, contract)
    copied_provenance = _validate_provenance(provenance)
    sidecar_path = artifact_path.with_suffix(".schema.json")
    npz_temporary = _temporary_npz(artifact_path.parent, validated)
    sidecar: dict[str, Any] = {
        "artifact_format": _ARTIFACT_FORMAT,
        "contract_name": contract.name,
        "schema_version": contract.schema_version,
        "arrays": _contract_arrays(contract),
        "npz_sha256": _sha256_file(npz_temporary),
        "provenance": copied_provenance,
    }
    if _has_quaternions(contract):
        sidecar["quaternion_order"] = "wxyz"
    sidecar_temporary = _temporary_json(artifact_path.parent, sidecar)
    artifact_backup: Path | None = None
    sidecar_backup: Path | None = None
    try:
        with _serialized_pair(artifact_path):
            if artifact_path.exists() != sidecar_path.exists():
                raise ValueError("existing artifact pair is incomplete")
            artifact_backup = _backup_file(artifact_path)
            sidecar_backup = _backup_file(sidecar_path)
            try:
                npz_temporary.replace(artifact_path)
                sidecar_temporary.replace(sidecar_path)
            except Exception:
                try:
                    _restore_pair(
                        artifact_path,
                        sidecar_path,
                        artifact_backup,
                        sidecar_backup,
                    )
                except Exception:
                    artifact_backup = sidecar_backup = None
                    raise
                artifact_backup = sidecar_backup = None
                raise
    finally:
        for temporary in (
            npz_temporary,
            sidecar_temporary,
            artifact_backup,
            sidecar_backup,
        ):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return artifact_path, sidecar_path


def _capture_artifact_pair(
    artifact_path: Path, sidecar_path: Path
) -> _ArtifactPairBytes:
    return _ArtifactPairBytes(
        artifact=_stable_file_bytes(artifact_path),
        sidecar=_stable_file_bytes(sidecar_path),
    )


def _load_npz_artifact_bytes(
    artifact_bytes: bytes,
    sidecar_bytes: bytes,
    expected_contract: NPZContract,
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    try:
        sidecar = json.loads(sidecar_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact contract mismatch: unreadable sidecar") from error
    if not isinstance(sidecar, dict):
        raise ValueError("artifact contract mismatch: sidecar root")
    expected_sidecar: dict[str, Any] = {
        "artifact_format": _ARTIFACT_FORMAT,
        "contract_name": expected_contract.name,
        "schema_version": expected_contract.schema_version,
        "arrays": _contract_arrays(expected_contract),
    }
    for key, value in expected_sidecar.items():
        if not _exact_json_equal(sidecar.get(key), value):
            raise ValueError(f"artifact contract mismatch: {key}")
    expected_keys = set(expected_sidecar) | {"npz_sha256", "provenance"}
    if _has_quaternions(expected_contract):
        expected_keys.add("quaternion_order")
        if sidecar.get("quaternion_order") != "wxyz":
            raise ValueError("artifact contract mismatch: quaternion order")
    if set(sidecar) != expected_keys:
        raise ValueError("artifact contract mismatch: sidecar fields")
    try:
        _validate_provenance(sidecar["provenance"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("artifact contract mismatch: provenance") from error
    if not _is_sha256(sidecar.get("npz_sha256")):
        raise ValueError("artifact contract mismatch: npz hash")
    if sidecar.get("npz_sha256") != sha256(artifact_bytes).hexdigest():
        raise ValueError("artifact hash mismatch")
    try:
        with np.load(BytesIO(artifact_bytes), allow_pickle=False) as archive:
            loaded = {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError("artifact contract mismatch: unreadable NPZ") from error
    return _validate_arrays(loaded, expected_contract), sidecar


def load_npz_artifact(
    path: str | Path, expected_contract: NPZContract
) -> dict[str, NDArray[Any]]:
    artifact_path = Path(path)
    sidecar_path = artifact_path.with_suffix(".schema.json")
    try:
        with _serialized_pair(artifact_path):
            before = _capture_artifact_pair(artifact_path, sidecar_path)
            loaded, _ = _load_npz_artifact_bytes(
                before.artifact.content,
                before.sidecar.content,
                expected_contract,
            )
            after = _capture_artifact_pair(artifact_path, sidecar_path)
    except OSError as error:
        raise ValueError("artifact contract mismatch: unreadable pair") from error
    if after != before:
        raise ValueError("artifact pair changed during validation")
    return loaded


def _windows_handle_identity(handle: int) -> tuple[int, int]:
    import ctypes
    from ctypes import wintypes

    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("last_access_time", wintypes.FILETIME),
            ("last_write_time", wintypes.FILETIME),
            ("volume_serial_number", wintypes.DWORD),
            ("file_size_high", wintypes.DWORD),
            ("file_size_low", wintypes.DWORD),
            ("number_of_links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        ]

    information = _ByHandleFileInformation()
    get_information = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandle
    get_information.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ByHandleFileInformation)]
    get_information.restype = wintypes.BOOL
    if not get_information(wintypes.HANDLE(handle), ctypes.byref(information)):
        error = ctypes.get_last_error()
        raise OSError(error, "GetFileInformationByHandle failed")
    file_index = (int(information.file_index_high) << 32) | int(
        information.file_index_low
    )
    return int(information.volume_serial_number), file_index


def _windows_file_identity(file_descriptor: int) -> tuple[int, int]:
    import msvcrt

    return _windows_handle_identity(msvcrt.get_osfhandle(file_descriptor))


def _file_identity_from_handle(file_descriptor: int, value: Any) -> tuple[int, int]:
    if os.name == "nt" and int(value.st_ino) == 0:
        return _windows_file_identity(file_descriptor)
    return int(value.st_dev), int(value.st_ino)


def _stable_file_bytes(path: Path, *, max_bytes: int | None = None) -> _FileBytes:
    if path.is_symlink():
        raise ValueError("captured file must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    file_descriptor = os.open(path, flags)
    try:
        before = os.fstat(file_descriptor)
        if not stat_module.S_ISREG(before.st_mode):
            raise ValueError("captured file must be a regular file")
        if max_bytes is not None and int(before.st_size) > max_bytes:
            raise ValueError("run byte capture memory cap exceeded")
        before_identity = _file_identity_from_handle(file_descriptor, before)
        chunks: list[bytes] = []
        captured_size = 0
        while True:
            read_size = 1024 * 1024
            if max_bytes is not None:
                read_size = min(read_size, max_bytes - captured_size + 1)
            chunk = os.read(file_descriptor, read_size)
            if not chunk:
                break
            captured_size += len(chunk)
            if max_bytes is not None and captured_size > max_bytes:
                raise ValueError("run byte capture memory cap exceeded")
            chunks.append(chunk)
        content = b"".join(chunks)
        after = os.fstat(file_descriptor)
        after_identity = _file_identity_from_handle(file_descriptor, after)
    finally:
        os.close(file_descriptor)
    before_signature = (
        int(before.st_size),
        int(before.st_mtime_ns),
        before_identity,
    )
    after_signature = (
        int(after.st_size),
        int(after.st_mtime_ns),
        after_identity,
    )
    if (
        path.is_symlink()
        or before_signature != after_signature
        or len(content) != after_signature[0]
    ):
        raise ValueError("captured file changed during byte capture")
    return _FileBytes(
        size=after_signature[0],
        mtime_ns=after_signature[1],
        device=after_identity[0],
        inode=after_identity[1],
        content=content,
    )


def _windows_directory_identity(directory: Path) -> tuple[int, int]:
    import ctypes
    from ctypes import wintypes

    create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(directory),
        0,
        0x1 | 0x2 | 0x4,
        None,
        3,
        0x02000000,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        error = ctypes.get_last_error()
        raise OSError(error, "CreateFileW failed for directory")
    try:
        return _windows_handle_identity(handle)
    finally:
        close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        close_handle(handle)


def _directory_identity(directory: Path) -> tuple[int, int]:
    if os.name == "nt":
        return _windows_directory_identity(directory)
    file_descriptor = os.open(directory, os.O_RDONLY)
    try:
        value = os.fstat(file_descriptor)
        if not stat_module.S_ISDIR(value.st_mode):
            raise ValueError("run path must be a real directory")
        return int(value.st_dev), int(value.st_ino)
    finally:
        os.close(file_descriptor)


def _capture_directory_bytes(directory: Path) -> _DirectoryBytes:
    identity = _directory_identity(directory)
    children = tuple(
        islice(directory.iterdir(), _RUN_CAPTURE_MAX_FILES + 1)
    )
    if len(children) > _RUN_CAPTURE_MAX_FILES:
        raise ValueError("run byte capture file cap exceeded")
    children = tuple(sorted(children, key=lambda item: item.name))
    declared_aggregate = 0
    for child in children:
        child_status = child.stat(follow_symlinks=False)
        if not stat_module.S_ISREG(child_status.st_mode):
            raise ValueError("captured file must be a regular file")
        size = int(child_status.st_size)
        if size > _RUN_CAPTURE_MAX_FILE_BYTES:
            raise ValueError("run byte capture per-file cap exceeded")
        declared_aggregate += size
        if declared_aggregate > _RUN_CAPTURE_MAX_AGGREGATE_BYTES:
            raise ValueError("run byte capture aggregate cap exceeded")
    files: dict[str, _FileBytes] = {}
    captured_aggregate = 0
    for child in children:
        remaining = _RUN_CAPTURE_MAX_AGGREGATE_BYTES - captured_aggregate
        captured = _stable_file_bytes(
            child,
            max_bytes=min(_RUN_CAPTURE_MAX_FILE_BYTES, remaining),
        )
        captured_aggregate += captured.size
        files[child.name] = captured
    if _directory_identity(directory) != identity:
        raise ValueError("run directory changed during byte capture")
    return _DirectoryBytes(directory_identity=identity, files=files)


def _failed_action_gate(variant: str) -> dict[str, Any]:
    if variant == "B0":
        collision, physics, reason = "failed", "failed", "physics_validation_failed"
    elif variant == "B1":
        collision = physics = "not_applicable_kinematic"
        reason = "kinematic_replay_not_action"
    else:
        collision = physics = "not_run"
        reason = "metric_depth_not_available"
    return {
        "collision_validation": collision,
        "physics_validation": physics,
        "action_export_eligible": False,
        "action_export_reason": reason,
        "action_exported": False,
    }


def terminal_metrics_error(metrics: Mapping[str, Any], file_names: set[str]) -> str | None:
    status = metrics.get("status")
    if status not in {"completed", "rejected", "failed", "not_run"}:
        return "unknown terminal status"
    variant = metrics.get("variant")
    if variant not in ("B0", "B1", "B2", "B3", "B4"):
        return "unknown experiment variant"
    if "actions.npz" in file_names:
        return "terminal status contains an action"
    if status == "completed":
        return "completed action-data status is not supported"
    reason = metrics.get("reason")
    if not isinstance(reason, str) or not reason.strip() or reason != reason.strip():
        return "non-completed status lacks a reason"
    if metrics.get("action_export_eligible") is not False:
        return "non-completed status lacks explicit action ineligibility"
    if metrics.get("action_exported") is not False:
        return "non-completed status must record action_exported=false"

    if status == "failed":
        if reason not in {"publication_exception", "stage_exception"}:
            return "failed status has an unknown infrastructure reason"
        failure_stage = metrics.get("failure_stage")
        if (
            type(failure_stage) is not str
            or not failure_stage
            or failure_stage != failure_stage.strip()
        ):
            return "failed status lacks failure diagnostics"
        if "rejection.json" not in file_names:
            return "failed status lacks failure diagnostics"
        error_type = metrics.get("error_type")
        error_message = metrics.get("error_message")
        if any(
            type(value) is not str or not value or value != value.strip()
            for value in (error_type, error_message)
        ):
            return "failed status lacks string error diagnostics"
        if metrics.get("placed_successfully", False) is not False:
            return "failed status cannot claim successful placement"
        if any(
            metrics.get(key) != value
            for key, value in _failed_action_gate(variant).items()
        ):
            return "failed status action gate disagrees with its variant"
        return None

    if variant == "B0":
        if status != "rejected":
            return "B0 must terminate rejected"
        physics = metrics.get("physics_validation")
        collision = metrics.get("collision_validation")
        if (physics, collision) not in {("passed", "passed"), ("failed", "failed")}:
            return "B0 physics and collision validation disagree"
        passed = physics == "passed"
        expected_reason = (
            "manual_baseline_not_video_grounded" if passed else "physics_validation_failed"
        )
        if metrics.get("placed_successfully") is not passed:
            return "B0 placement and physics validation disagree"
        if reason != expected_reason or metrics.get("action_export_reason") != expected_reason:
            return "B0 terminal reason disagrees with physics validation"
    elif variant == "B1":
        if status != "rejected":
            return "B1 must terminate rejected"
        if (
            reason != "kinematic_replay_not_action"
            or metrics.get("action_export_reason") != "kinematic_replay_not_action"
            or metrics.get("collision_validation") != "not_applicable_kinematic"
            or metrics.get("physics_validation") != "not_applicable_kinematic"
            or metrics.get("placed_successfully") is not False
        ):
            return "B1 terminal metrics are not kinematic-only"
    else:
        if status != "not_run":
            return "metric-depth variants must terminate not_run"
        if (
            reason != "metric_depth_not_available"
            or metrics.get("action_export_reason") != "metric_depth_not_available"
            or metrics.get("collision_validation") != "not_run"
            or metrics.get("physics_validation") != "not_run"
            or metrics.get("placed_successfully", False) is not False
        ):
            return "metric-depth variant terminal metrics disagree"
    if status == "rejected":
        rejection_stage = metrics.get("rejection_stage")
        if (
            "rejection.json" not in file_names
            or type(rejection_stage) is not str
            or not rejection_stage
            or rejection_stage != rejection_stage.strip()
        ):
            return "rejected status lacks rejection diagnostics"
    if status == "not_run" and "rejection.json" in file_names:
        return "not_run status must not contain rejection diagnostics"
    return None


def _json_object_bytes(file: _FileBytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(file.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {name}") from error
    if type(value) is not dict:
        raise ValueError(f"expected JSON object in {name}")
    return value


def _verify_captured_run(
    captured: _DirectoryBytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    captured_files = captured.files
    manifest = _json_object_bytes(captured_files["run_manifest.json"], "run_manifest.json")
    metrics = _json_object_bytes(captured_files["metrics.json"], "metrics.json")
    actual_names = set(captured_files) - {"run_manifest.json"}
    files = manifest.get("files")
    if (
        manifest.get("producer") != _RUN_MANIFEST_PRODUCER
        or type(files) is not dict
        or set(files) != actual_names
        or manifest.get("status") != metrics.get("status")
        or manifest.get("variant") != metrics.get("variant")
    ):
        raise ValueError("manifest root mismatch")
    version = manifest.get("format_version")
    dimensions: tuple[int, int, int] | None = None
    if type(version) is not int:
        raise ValueError("manifest version must be an integer")
    if version == 3:
        if set(manifest) != _V3_MANIFEST_FIELDS:
            raise ValueError("manifest v3 fields mismatch")
    elif version == 4:
        dimensions = _verify_v4_manifest(captured_files, manifest, metrics)
    else:
        raise ValueError("unknown manifest version")
    if terminal_metrics_error(metrics, actual_names) is not None:
        raise ValueError("invalid terminal metrics")
    for name, recorded in files.items():
        child = captured_files[name]
        if type(recorded) is not dict:
            raise ValueError("manifest file entry must be an object")
        expected_entry_fields = {"size", "sha256"}
        if version == 4 and Path(name).suffix.lower() in _MEDIA_SUFFIXES:
            expected_entry_fields |= {
                "media_role",
                "contains_private_source_frames",
            }
            role = recorded.get("media_role")
            private = recorded.get("contains_private_source_frames")
            if type(role) is not str or not role or type(private) is not bool:
                raise ValueError("invalid media classification")
            if (role, private) != _EXPECTED_MEDIA_CLASSIFICATION.get(name):
                raise ValueError("untrusted media classification")
            if role == "simulation_only" and private:
                raise ValueError("simulation-only media cannot be private")
        if set(recorded) != expected_entry_fields:
            raise ValueError("manifest file entry fields mismatch")
        digest = sha256(child.content).hexdigest()
        if (
            type(recorded.get("size")) is not int
            or recorded["size"] < 0
            or not _is_sha256(recorded.get("sha256"))
            or recorded["size"] != child.size
            or recorded["sha256"] != digest
        ):
            raise ValueError("manifest file signature mismatch")
    if version == 4:
        if dimensions is None:
            raise ValueError("missing model dimensions")
        _verify_v4_npz_artifacts(captured_files, manifest, dimensions)
    return manifest, metrics


def verify_run_directory(path: str | Path) -> VerifiedRun:
    directory = Path(path)
    before: _DirectoryBytes | None = None
    after: _DirectoryBytes | None = None
    result: tuple[dict[str, Any], dict[str, Any]] | None = None
    validation_error: Exception | None = None
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("run path must be a real directory")
        before = _capture_directory_bytes(directory)
        result = _verify_captured_run(before)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        validation_error = error
    if before is not None:
        try:
            after = _capture_directory_bytes(directory)
        except (OSError, TypeError, ValueError) as error:
            validation_error = validation_error or error
    if before is None or after is None or after != before:
        validation_error = validation_error or ValueError(
            "run directory bytes changed during verification"
        )
    if validation_error is not None or result is None:
        raise ValueError("trusted run directory verification failed") from validation_error
    manifest, metrics = result
    snapshot = {
        name: (
            file.size,
            sha256(file.content).hexdigest(),
            file.mtime_ns,
            file.device,
            file.inode,
        )
        for name, file in after.files.items()
    }
    return VerifiedRun(
        path=directory,
        metrics=dict(metrics),
        manifest=dict(manifest),
        directory_identity=after.directory_identity,
        snapshot=snapshot,
    )


def _pinned_dependency_identity(
    model_path: str | Path, trusted_root: str | Path
) -> str:
    lexical_root = Path(trusted_root).absolute()
    if lexical_root.is_symlink() or not lexical_root.is_dir():
        raise ValueError("trusted model root must be a real directory")
    trusted = lexical_root.resolve(strict=True)
    root_candidate = Path(model_path).absolute()
    main_directory = root_candidate.parent

    def validate_relative_reference(reference: str) -> Path:
        reference_path = Path(reference)
        if reference_path.is_absolute():
            raise ValueError("model dependency escapes trusted model root")
        if any(part == os.pardir for part in reference_path.parts):
            raise ValueError("model dependency contains a parent path component")
        return reference_path

    def resolve_dependency(reference: str) -> Path:
        reference_path = validate_relative_reference(reference)
        candidate = main_directory / reference_path
        try:
            relative_candidate = candidate.absolute().relative_to(lexical_root)
        except ValueError as error:
            raise ValueError("model dependency escapes trusted model root") from error
        inspected = lexical_root
        for part in relative_candidate.parts:
            inspected = inspected / part
            if inspected.is_symlink():
                raise ValueError("model dependency symlinks are forbidden")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(trusted)
        except ValueError as error:
            raise ValueError("model dependency escapes trusted model root") from error
        if not resolved.is_file():
            raise ValueError("model dependency must be a regular file")
        return resolved

    if root_candidate.is_symlink():
        raise ValueError("model dependency symlinks are forbidden")
    root_reference = os.path.relpath(root_candidate, lexical_root)
    root = resolve_dependency(root_reference)
    dependencies: set[Path] = set()
    visiting_xml: set[Path] = set()
    expanded_elements: list[ElementTree.Element] = []

    def expand_element(element: ElementTree.Element) -> None:
        if element.tag == "include":
            reference = element.get("file")
            if not reference:
                raise ValueError("model include lacks a file")
            dependency = resolve_dependency(reference)
            if dependency.suffix.lower() != ".xml":
                raise ValueError("included model dependency must be XML")
            expand_xml(dependency)
            return
        expanded_elements.append(element)
        for child in element:
            expand_element(child)

    def expand_xml(current: Path) -> None:
        if current in visiting_xml:
            raise ValueError("model include cycle")
        visiting_xml.add(current)
        dependencies.add(current)
        try:
            document = ElementTree.fromstring(current.read_bytes())
        except ElementTree.ParseError as error:
            raise ValueError("invalid pinned model XML") from error
        for element in document:
            expand_element(element)
        visiting_xml.remove(current)

    expand_xml(root)
    compiler_settings: dict[str, str] = {}
    asset_references: list[tuple[str, str]] = []
    pinned_file_tags = {"mesh": "meshdir"}
    for element in expanded_elements:
        if element.tag == "compiler":
            for name in ("assetdir", "meshdir", "texturedir"):
                if name in element.attrib:
                    value = element.attrib[name]
                    validate_relative_reference(value)
                    compiler_settings[name] = value
        reference = element.get("file")
        if reference is None:
            continue
        directory_setting = pinned_file_tags.get(element.tag)
        if directory_setting is None:
            raise ValueError("unsupported pinned file-bearing tag")
        asset_references.append((directory_setting, reference))
    assetdir = compiler_settings.get("assetdir", "")
    effective_directories = {
        "meshdir": compiler_settings.get("meshdir", assetdir),
        "texturedir": compiler_settings.get("texturedir", assetdir),
    }
    for directory_setting, reference in asset_references:
        resource_dir = effective_directories[directory_setting]
        validate_relative_reference(resource_dir)
        validate_relative_reference(reference)
        dependency = resolve_dependency(str(Path(resource_dir) / reference))
        dependencies.add(dependency)
    digest = sha256()
    for dependency in sorted(
        dependencies,
        key=lambda item: Path(os.path.relpath(item, trusted)).as_posix(),
    ):
        relative = Path(os.path.relpath(dependency, trusted)).as_posix().encode("utf-8")
        content = dependency.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _pinned_model_path(variant: str) -> Path:
    if variant == "B1":
        return Path(_DEFAULT_MODEL_PATH)
    if variant in {"B0", "B2", "B3", "B4"}:
        return Path(DEFAULT_SCENE_PATH)
    raise ValueError("model variant is not pinned")


def pinned_model_identity(variant: str) -> str:
    model_path = _pinned_model_path(variant)
    return _pinned_dependency_identity(model_path, model_path.parent)


def pinned_model_logical_identity(variant: str) -> str:
    """Return the stable logical identity paired with a pinned variant model."""

    if variant == "B1":
        return "primitive_7dof_panda_like_diagnostic"
    if variant in {"B0", "B2", "B3", "B4"}:
        return "mujoco_menagerie_franka_emika_panda_exp001"
    raise ValueError("model variant is not pinned")


@lru_cache(maxsize=2)
def _cached_model_dimensions(
    model_path_text: str, model_sha256: str
) -> tuple[int, int, int]:
    model_path = Path(model_path_text)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    dimensions = (model.nq, model.nv, model.nu)
    del model
    gc.collect()
    return dimensions


def pinned_model_dimensions(
    variant: str, model_sha256: str | None = None
) -> tuple[int, int, int]:
    current_identity = pinned_model_identity(variant)
    if model_sha256 is not None and (
        not _is_sha256(model_sha256) or model_sha256.lower() != current_identity
    ):
        raise ValueError("model hash is not pinned for variant")
    model_path = _pinned_model_path(variant).resolve(strict=True)
    return _cached_model_dimensions(str(model_path), current_identity)


def _verified_manifest_dimensions(manifest: Mapping[str, Any]) -> tuple[int, int, int]:
    model_sha256 = manifest.get("model_sha256")
    if not isinstance(model_sha256, str):
        raise ValueError("invalid model hash")
    variant = manifest.get("variant")
    if not isinstance(variant, str):
        raise ValueError("invalid model variant")
    pinned_dimensions = pinned_model_dimensions(variant, model_sha256)
    dimensions = (manifest.get("model_nq"), manifest.get("model_nv"), manifest.get("model_nu"))
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError("invalid model dimensions")
    if dimensions != pinned_dimensions:
        raise ValueError("model dimensions disagree with pinned model")
    return pinned_dimensions


def _verify_v4_manifest(
    captured_files: Mapping[str, _FileBytes],
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> tuple[int, int, int]:
    if set(manifest) != _V4_MANIFEST_FIELDS:
        raise ValueError("manifest v4 fields mismatch")
    if type(manifest.get("variant")) is not str or manifest.get("variant") not in {"B0", "B1", "B2", "B3", "B4"}:
        raise ValueError("invalid variant")
    if type(manifest.get("status")) is not str or manifest.get("status") not in {"completed", "rejected", "failed", "not_run"}:
        raise ValueError("invalid terminal status")
    if (
        type(manifest.get("reason")) is not str
        or not manifest["reason"]
        or manifest["reason"] != manifest["reason"].strip()
    ):
        raise ValueError("invalid terminal reason")
    for name in ("action_export_eligible", "action_exported"):
        if type(manifest.get(name)) is not bool:
            raise ValueError("invalid action semantics")
    if (
        type(manifest.get("action_export_reason")) is not str
        or manifest["action_export_reason"] != manifest["action_export_reason"].strip()
    ):
        raise ValueError("invalid action reason")
    terminal_fields = {
        "reason",
        "action_export_eligible",
        "action_export_reason",
        "action_exported",
    }
    if any(manifest.get(name) != metrics.get(name) for name in terminal_fields):
        raise ValueError("manifest terminal semantics disagree with metrics")
    if not _is_sha256(manifest.get("config_sha256")):
        raise ValueError("invalid config hash")
    source_sha256 = manifest.get("source_sha256")
    if source_sha256 != "not_used" and not _is_sha256(source_sha256):
        raise ValueError("invalid source hash")
    dimensions = _verified_manifest_dimensions(manifest)
    provenance_file = captured_files.get("provenance.json")
    if provenance_file is not None:
        provenance = _json_object_bytes(provenance_file, "provenance.json")
        config = provenance.get("config")
        model_record = provenance.get("model")
        resolved_config = config.get("resolved") if type(config) is dict else None
        experiment_id = provenance.get("experiment_id")
        if (
            type(config) is not dict
            or config.get("sha256") != manifest["config_sha256"]
            or type(resolved_config) is not dict
            or type(experiment_id) is not str
            or not experiment_id
            or experiment_id != experiment_id.strip()
            or resolved_config.get("experiment_id") != experiment_id
            or type(model_record) is not dict
            or model_record.get("sha256") != manifest["model_sha256"]
            or provenance.get("variant") != manifest["variant"]
        ):
            raise ValueError("manifest hashes disagree with provenance")
        source = provenance.get("source")
        if source_sha256 != "not_used" and (
            type(source) is not dict or source.get("sha256") != source_sha256
        ):
            raise ValueError("manifest source hash disagrees with provenance")
    return dimensions


def _verify_v4_npz_artifacts(
    captured_files: Mapping[str, _FileBytes],
    manifest: Mapping[str, Any],
    dimensions: tuple[int, int, int],
) -> None:
    files = manifest["files"]
    npz_names = {name for name in files if Path(name).suffix == ".npz"}
    sidecar_names = {name for name in files if name.endswith(".schema.json")}
    expected_sidecars = {str(Path(name).with_suffix(".schema.json")) for name in npz_names}
    if sidecar_names != expected_sidecars:
        raise ValueError("NPZ sidecar set mismatch")
    nq, nv, nu = dimensions
    contracts: dict[str, NPZContract] = {
        "trajectory_2d.npz": TRAJECTORY_2D_V1,
        "robot_reference.npz": ROBOT_REFERENCE_V1,
        "baseline_control_trace.npz": baseline_control_contract(nu),
        "simulation.npz": simulation_contract(nq, nv, nu),
    }
    if not npz_names <= set(contracts):
        raise ValueError("unknown NPZ contract")
    provenance_file = captured_files.get("provenance.json")
    if npz_names and provenance_file is None:
        raise ValueError("NPZ artifacts require provenance")
    provenance = (
        _json_object_bytes(provenance_file, "provenance.json")
        if provenance_file is not None
        else {}
    )
    generator = provenance.get("generator")
    if npz_names and (
        type(generator) is not dict or not _is_git_commit(generator.get("git_commit"))
    ):
        raise ValueError("invalid generator provenance")
    for name in npz_names:
        artifact_file = captured_files[name]
        sidecar_name = str(Path(name).with_suffix(".schema.json"))
        sidecar_file = captured_files[sidecar_name]
        loaded, sidecar = _load_npz_artifact_bytes(
            artifact_file.content,
            sidecar_file.content,
            contracts[name],
        )
        del loaded
        artifact_provenance = sidecar["provenance"]
        if artifact_provenance.get("producer") != manifest.get("producer"):
            raise ValueError("artifact producer disagrees with manifest")
        if artifact_provenance.get("git_commit") != generator.get("git_commit"):
            raise ValueError("artifact commit disagrees with provenance")
        expected_source = manifest["source_sha256"]
        for manifest_name, sidecar_name in (
            ("config_sha256", "config_sha256"),
            ("model_sha256", "model_sha256"),
            ("status", "terminal_status"),
            ("reason", "terminal_reason"),
            ("action_export_eligible", "action_export_eligible"),
        ):
            if artifact_provenance.get(sidecar_name) != manifest.get(manifest_name):
                raise ValueError("artifact provenance disagrees with manifest")
        if artifact_provenance.get("source_sha256") != expected_source:
            raise ValueError("artifact source provenance disagrees with manifest")

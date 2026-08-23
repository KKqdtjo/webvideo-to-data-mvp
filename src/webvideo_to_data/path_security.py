"""Lexical validation and comparison for supported Windows filesystem paths."""

from __future__ import annotations

import ntpath
import os
from pathlib import Path, PureWindowsPath
import re


_NON_FILESYSTEM_UNC_SHARES = {"ipc$", "mailslot", "pipe"}
_RESERVED_DOS_DEVICE_COMPONENTS = frozenset(
    {
        "nul",
        "con",
        "prn",
        "aux",
        "clock$",
        "conin$",
        "conout$",
        "com1",
        "com2",
        "com3",
        "com4",
        "com5",
        "com6",
        "com7",
        "com8",
        "com9",
        "lpt1",
        "lpt2",
        "lpt3",
        "lpt4",
        "lpt5",
        "lpt6",
        "lpt7",
        "lpt8",
        "lpt9",
        "com¹",
        "com²",
        "com³",
        "lpt¹",
        "lpt²",
        "lpt³",
    }
)


def _validate_windows_object_components(
    lexical_tail: str, description: str
) -> None:
    """Reject object components that Win32 resolves as DOS devices."""

    for component in lexical_tail.replace("/", "\\").split("\\"):
        file_part = component.split(":", 1)[0].rstrip(" .")
        device_stem = file_part.split(".", 1)[0].rstrip(" .").casefold()
        if device_stem in _RESERVED_DOS_DEVICE_COMPONENTS:
            raise ValueError(f"unsafe {description} path device alias")


def windows_path_for_containment(path: Path) -> PureWindowsPath:
    """Return one comparison spelling for equivalent DOS/UNC path aliases."""

    normalized = ntpath.normpath(str(path))
    if normalized.startswith("\\\\?\\"):
        remainder = normalized[4:]
        if remainder[:4].casefold() == "unc\\":
            unc_path = remainder[4:]
            server, separator, share_and_tail = unc_path.partition("\\")
            share, _, _ = share_and_tail.partition("\\")
            if server and separator and share:
                normalized = "\\\\" + unc_path
        else:
            drive, tail = ntpath.splitdrive(remainder)
            if len(drive) == 2 and drive[1] == ":" and tail.startswith("\\"):
                normalized = remainder
    return PureWindowsPath(ntpath.normcase(normalized))


def validate_windows_path_namespace(path: Path, description: str) -> None:
    """Allow only absolute DOS/UNC filesystem namespaces, including aliases."""

    lexical = str(path).replace("/", "\\")
    folded = lexical.casefold()
    if folded.startswith(("\\\\.\\", "\\??\\", "\\\\??\\")):
        raise ValueError(f"unsafe {description} path namespace")
    if folded.startswith("\\\\?\\"):
        remainder = lexical[4:]
        if remainder[:4].casefold() == "unc\\":
            unc_path = remainder[4:]
            server, separator, share_and_tail = unc_path.partition("\\")
            share, _, object_tail = share_and_tail.partition("\\")
            if (
                server
                and separator
                and share
                and server not in {".", ".."}
                and share not in {".", ".."}
                and share.casefold() not in _NON_FILESYSTEM_UNC_SHARES
            ):
                _validate_windows_object_components(object_tail, description)
                return
        else:
            drive, tail = ntpath.splitdrive(remainder)
            if re.fullmatch(r"[A-Za-z]:", drive) is not None and tail.startswith(
                "\\"
            ):
                _validate_windows_object_components(tail, description)
                return
        raise ValueError(f"unsafe {description} path namespace")
    drive, tail = ntpath.splitdrive(lexical)
    if re.fullmatch(r"[A-Za-z]:", drive) is not None and tail.startswith("\\"):
        _validate_windows_object_components(tail, description)
        return
    if lexical.startswith("\\\\"):
        unc_path = lexical[2:]
        server, separator, share_and_tail = unc_path.partition("\\")
        share, _, object_tail = share_and_tail.partition("\\")
        if (
            server
            and separator
            and share
            and server not in {".", ".."}
            and share not in {".", ".."}
            and share.casefold() not in _NON_FILESYSTEM_UNC_SHARES
        ):
            _validate_windows_object_components(object_tail, description)
            return
    raise ValueError(f"unsafe {description} path namespace")


def absolute_windows_filesystem_path(path: Path, description: str) -> Path:
    """Validate a Windows path lexically, anchoring a safe relative spelling."""

    lexical = str(path).replace("/", "\\")
    drive, _ = ntpath.splitdrive(lexical)
    if drive or lexical.startswith("\\"):
        validate_windows_path_namespace(path, description)
        return path
    _validate_windows_object_components(lexical, description)
    if ":" in lexical:
        raise ValueError(f"unsafe {description} path namespace")
    anchored = Path.cwd() / path
    validate_windows_path_namespace(anchored, description)
    return anchored


def absolute_filesystem_path(path: str | Path, description: str) -> Path:
    """Lexically validate and cwd-anchor one caller-provided filesystem path."""

    candidate = Path(path)
    if os.name == "nt":
        return absolute_windows_filesystem_path(candidate, description)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate

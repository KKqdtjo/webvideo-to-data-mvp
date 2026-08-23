from __future__ import annotations

import os
from pathlib import Path
import stat

import pytest

from webvideo_to_data.path_security import absolute_filesystem_path


_RESERVED_DOS_DEVICE_NAMES = (
    "NUL",
    "CON",
    "PRN",
    "AUX",
    "CLOCK$",
    "CONIN$",
    "CONOUT$",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
    "COM¹",
    "COM²",
    "COM³",
    "LPT¹",
    "LPT²",
    "LPT³",
)


def _windows_path_with_component(
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


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize(
    "suffix",
    ("", ".txt", ".tar.gz", " ", ".", ":stream", ".txt:stream", " .txt", " :stream"),
)
@pytest.mark.parametrize("reserved", _RESERVED_DOS_DEVICE_NAMES)
@pytest.mark.parametrize("letter_case", ("upper", "lower"))
def test_windows_reserved_dos_device_component_is_rejected(
    reserved: str, suffix: str, letter_case: str
) -> None:
    """Catch a legacy device alias being treated as an ordinary relative component."""

    base = reserved if letter_case == "upper" else reserved.lower()
    with pytest.raises(ValueError, match="unsafe verified suite path device alias"):
        absolute_filesystem_path(
            Path("ordinary") / f"{base}{suffix}" / "suite", "verified suite"
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize(
    "component",
    (
        "COM10",
        "LPT0",
        "console",
        "null",
        "data",
        "COM0",
        "LPT10",
        "xNUL",
        "NULx",
        "COM10.txt",
        "data.",
        "data ",
        ".data",
        "data .txt",
    ),
)
@pytest.mark.parametrize(
    "spelling", ("relative", "dos", "extended-dos", "unc", "extended-unc")
)
def test_windows_non_device_component_remains_accepted(
    tmp_path: Path, component: str, spelling: str
) -> None:
    """Catch a bounded device inventory becoming a broad prefix blacklist."""

    candidate = _windows_path_with_component(tmp_path, spelling, component)
    result = absolute_filesystem_path(candidate, "verified suite")

    expected = Path.cwd() / candidate if spelling == "relative" else candidate
    assert result == expected


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize(
    "candidate",
    (
        Path(r"\\con.example\Share\ordinary\suite"),
        Path(r"\\Server\NUL\ordinary\suite"),
        Path(r"\\Server\COM1\ordinary\suite"),
        Path(r"\\?\UNC\con.example\Share\ordinary\suite"),
        Path(r"\\?\UNC\Server\NUL\ordinary\suite"),
        Path(r"\\?\UNC\Server\COM1\ordinary\suite"),
    ),
)
def test_windows_unc_volume_components_may_resemble_dos_devices(
    candidate: Path,
) -> None:
    """Catch host/share volume roots being scanned as filesystem objects."""

    assert absolute_filesystem_path(candidate, "verified suite") == candidate


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize(
    "candidate",
    (
        Path(r"\\Server\NUL\ordinary\NUL.txt\suite"),
        Path(r"\\Server\COM1\ordinary\COM1:stream\suite"),
        Path(r"\\?\UNC\Server\NUL\ordinary\NUL \suite"),
        Path(r"\\?\UNC\Server\COM1\ordinary\COM1.\suite"),
    ),
)
def test_windows_unc_object_after_reserved_looking_share_is_rejected(
    candidate: Path,
) -> None:
    """Catch skipping the share accidentally skipping later object components."""

    with pytest.raises(ValueError, match="unsafe verified suite path device alias"):
        absolute_filesystem_path(candidate, "verified suite")


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespace regression")
@pytest.mark.parametrize(
    "candidate",
    (
        Path(r"\\Server\PIPE\ordinary\suite"),
        Path(r"\\Server\mailslot\ordinary\suite"),
        Path(r"\\Server\IPC$\ordinary\suite"),
        Path(r"\\?\UNC\Server\PIPE\ordinary\suite"),
        Path(r"\\?\UNC\Server\mailslot\ordinary\suite"),
        Path(r"\\?\UNC\Server\IPC$\ordinary\suite"),
    ),
)
def test_windows_nonfilesystem_unc_shares_remain_rejected(candidate: Path) -> None:
    """Catch root-component skipping bypassing the explicit IPC share policy."""

    with pytest.raises(ValueError, match="unsafe verified suite path namespace"):
        absolute_filesystem_path(candidate, "verified suite")


@pytest.mark.skipif(os.name != "nt", reason="Windows device regression")
def test_windows_nul_component_is_a_real_character_device_before_preflight(
    tmp_path: Path,
) -> None:
    """Prove the rejected spelling reaches a device through a real metadata call."""

    assert stat.S_ISCHR(os.lstat(tmp_path / "NUL").st_mode)
    with pytest.raises(ValueError, match="unsafe verified suite path device alias"):
        absolute_filesystem_path(tmp_path / "NUL", "verified suite")

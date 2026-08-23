"""Central redaction and bounded publication privacy auditing."""

from __future__ import annotations

import ast
import codecs
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, field
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import struct
import subprocess
import tempfile
from types import MappingProxyType
from typing import BinaryIO, Callable, Iterable
import zipfile
import zlib

import numpy as np

if os.name == "nt":
    import msvcrt
    from ctypes import wintypes


_WINDOW_BYTES = 1024 * 1024
_TEXT_CHUNK_BYTES = 64 * 1024
_TEXT_OVERLAP = 1024
_CONTAINER_SCAN_LIMIT = 64 * 1024 * 1024
_NPZ_MAX_ENTRIES = 128
_NPZ_MAX_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
_NPZ_MAX_COMPRESSION_RATIO = 100.0
_MEDIA_SUFFIXES = {".gif", ".mp4", ".m4v", ".mov", ".avi", ".webm"}
_FORMAT_AWARE_SUFFIXES = {".gif", ".mp4", ".m4v", ".mov", ".npz", ".png"}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_MAX_CHUNKS = 4_096
_PNG_MAX_METADATA_BYTES = 4 * 1024 * 1024
_PNG_MAX_DECOMPRESSED_BYTES = 32 * 1024 * 1024
_PNG_MAX_WORK_UNITS = 100_000
_GIF_MAX_BLOCKS = 131_072
_GIF_MAX_METADATA_BYTES = 4 * 1024 * 1024
_GIF_MAX_LOGICAL_PIXELS = 16 * 1024 * 1024
_GIF_MAX_IMAGE_PIXELS = 8 * 1024 * 1024
_GIF_MAX_CUMULATIVE_PIXELS = 128 * 1024 * 1024
_GIF_MAX_FRAMES = 512
_GIF_MAX_COMPRESSED_BYTES = 32 * 1024 * 1024
_GIF_MAX_SUB_BLOCKS = 131_072
_GIF_MAX_DECODE_WORK = 384 * 1024 * 1024
_STABLE_SNAPSHOT_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_STABLE_SNAPSHOT_MAX_FILES = 4_096
_STABLE_SNAPSHOT_MAX_FILE_BYTES = 512 * 1024 * 1024
_STABLE_SNAPSHOT_MAX_AGGREGATE_BYTES = 2 * 1024 * 1024 * 1024
_STABLE_SNAPSHOT_MAX_WORK_UNITS = 100_000
_STABLE_SNAPSHOT_MAX_RELATIVE_CHARS = 4_096
_STABLE_SNAPSHOT_MAX_PATH_DEPTH = 64
_STABLE_CAPTURED_DASHBOARD_BYTES = 8 * 1024 * 1024
_STABLE_CAPTURED_PREVIEW_BYTES = 64 * 1024 * 1024
_PNG_KNOWN_ANCILLARY = {
    b"bKGD",
    b"cHRM",
    b"cICP",
    b"cLLI",
    b"eXIf",
    b"gAMA",
    b"hIST",
    b"iCCP",
    b"iTXt",
    b"mDCV",
    b"pHYs",
    b"sBIT",
    b"sPLT",
    b"sRGB",
    b"tEXt",
    b"tIME",
    b"tRNS",
    b"zTXt",
}
_PNG_SINGLETON_ANCILLARY = _PNG_KNOWN_ANCILLARY - {
    b"iTXt",
    b"sPLT",
    b"tEXt",
    b"zTXt",
}
_PNG_BEFORE_PLTE_AND_IDAT = {
    b"cHRM",
    b"cICP",
    b"cLLI",
    b"gAMA",
    b"iCCP",
    b"mDCV",
    b"sBIT",
    b"sRGB",
}
_PNG_BEFORE_IDAT = _PNG_BEFORE_PLTE_AND_IDAT | {
    b"bKGD",
    b"eXIf",
    b"hIST",
    b"pHYs",
    b"sPLT",
    b"tRNS",
}
_PNG_AFTER_PLTE_IF_PRESENT = {b"bKGD", b"hIST", b"tRNS"}
_TIFF_TYPE_SIZES = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 1,
    7: 1,
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
    13: 4,
}
_TIFF_IFD_POINTER_TAGS = {0x014A, 0x8769, 0x8825, 0xA005}
_TIFF_XP_STRING_TAGS = {0x9C9B, 0x9C9C, 0x9C9D, 0x9C9E, 0x9C9F}
_TIFF_USER_COMMENT_TAG = 0x9286
_ISO_BMFF_TEXT_SAMPLE_TYPES = {
    b"text",
    b"tx3g",
    b"wvtt",
    b"stpp",
    b"mett",
    b"metx",
}
_ISO_BMFF_TEXT_HANDLER_TYPES = {b"text", b"sbtl", b"subt", b"clcp", b"meta"}
_ISO_BMFF_NON_TEXT_HANDLER_TYPES = {b"vide", b"soun", b"hint", b"auxv", b"pict"}
_ISO_MAX_BOXES = 1_024
_ISO_MAX_TRACKS = 32
_ISO_MAX_TABLE_ENTRIES = 16_384
_ISO_MAX_SAMPLES = 16_384
_ISO_MAX_CHUNKS = 4_096
_ISO_MAX_TEXT_RANGES = 8_192
_ISO_MAX_WORK = 25_000
_AUTHORIZATION = re.compile(
    r"(?i)\bauthorization\s*:\s*(?:bearer|basic)?\s*[^\s,;]+"
)
_COOKIE = re.compile(r"(?im)\b(?:set-cookie|cookie)\s*:\s*[^\r\n]+")
_CREDENTIAL_QUERY = re.compile(
    r"(?i)(?P<prefix>[?&](?:token|key|signature|credential)=)[^&#\s]+"
)
_SSH_KEY_BLOCK = re.compile(
    r"(?is)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
)
_SSH_PRIVATE_HEADER = re.compile(r"(?i)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_SSH_PUBLIC = re.compile(r"(?i)\bssh-(?:rsa|ed25519)\s+[A-Za-z0-9+/=]{32,}")
_PROVIDER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"gh[pousr]_[A-Za-z0-9]{20,255}|"
    r"xox[baprs]-[A-Za-z0-9-]{20,255}|"
    r"(?:sk|rk|pk)-[A-Za-z0-9_-]{20,255}|"
    r"hf_[A-Za-z0-9]{20,255}|"
    r"glpat-[A-Za-z0-9_-]{20,255}|"
    r"AIza[A-Za-z0-9_-]{20,255}|"
    r"npm_[A-Za-z0-9]{20,255}|"
    r"AKIA[A-Z0-9]{16}"
    r")(?![A-Za-z0-9])"
)
_WINDOWS_HOME = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:(?:\\{1,2}|/)"
    r"(?:Users|Documents and Settings)(?:\\{1,2}|/))"
    r"[^\\/\s\"'<>]+(?:(?:\\{1,2}|/)[^\s\"'<>]*)?"
)
_POSIX_HOME = re.compile(
    r"(?<![A-Za-z0-9])/(?:home|Users)/[^/\s\"'<>]+(?:/[^\s\"'<>]*)?"
)
_WINDOWS_ABSOLUTE = re.compile(
    r"(?i)(?<![A-Za-z0-9])"
    r"(?<![\x00-\x08\x0b\x0c\x0e-\x1f\x7f])(?:"
    r"\\\\[^\\/:*?\"<>|\x00-\x20\x7f,;()\[\]{}]+"
    r"\\[^\\/:*?\"<>|\x00-\x20\x7f,;()\[\]{}]+"
    r"(?:\\[^\\/:*?\"<>|\x00-\x20\x7f,;()\[\]{}]+)*|"
    r"[A-Z]:(?:"
    r"[\\/][^\\/:*?\"<>|\x00-\x20\x7f,;()\[\]{}]+"
    r"(?:[\\/][^\\/:*?\"<>|\x00-\x20\x7f,;()\[\]{}]+)*|"
    r"[^\\/:*?\"<>|\x00-\x20\x7f,;()\[\]{}]+"
    r"(?:[\\/][^\\/:*?\"<>|\x00-\x20\x7f,;()\[\]{}]+)*"
    r")|"
    r"\\(?!\\)[^\\/:*?\"<>|\x00-\x20\x7f,;()\[\]{}]+"
    r"(?:\\[^\\/:*?\"<>|\x00-\x20\x7f,;()\[\]{}]+)*"
    r")"
)
_POSIX_ABSOLUTE = re.compile(
    r"(?<![:/A-Za-z0-9])"
    r"(?<![\x00-\x08\x0b\x0c\x0e-\x1f\x7f])"
    r"(?<!<external-output>)"
    r"(?<!<)"
    r"/(?!/)[\w.@%+=~-]+(?:/[\w.@%+=~-]+)*"
)
_FILE_ATTRIBUTE_REPARSE_POINT = getattr(
    stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
)
_TRAILING_PATH_PUNCTUATION = ".,;:!?)]}"

if os.name == "nt":
    class _FileTime(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]


    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("attributes", wintypes.DWORD),
            ("creation", _FileTime),
            ("access", _FileTime),
            ("write", _FileTime),
            ("volume", wintypes.DWORD),
            ("size_high", wintypes.DWORD),
            ("size_low", wintypes.DWORD),
            ("links", wintypes.DWORD),
            ("index_high", wintypes.DWORD),
            ("index_low", wintypes.DWORD),
        ]


    class _UnicodeString(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.USHORT),
            ("maximum_length", wintypes.USHORT),
            ("buffer", wintypes.LPWSTR),
        ]


    class _ObjectAttributes(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.ULONG),
            ("root_directory", wintypes.HANDLE),
            ("object_name", ctypes.POINTER(_UnicodeString)),
            ("attributes", wintypes.ULONG),
            ("security_descriptor", wintypes.LPVOID),
            ("security_quality_of_service", wintypes.LPVOID),
        ]


    class _IoStatusBlock(ctypes.Structure):
        _fields_ = [
            ("status", ctypes.c_ssize_t),
            ("information", ctypes.c_size_t),
        ]


    class _FileRenameInformation(ctypes.Structure):
        _fields_ = [
            ("replace_if_exists", wintypes.BOOLEAN),
            ("root_directory", wintypes.HANDLE),
            ("file_name_length", wintypes.ULONG),
            ("file_name", wintypes.WCHAR * 1),
        ]


    class _FileDispositionInformation(ctypes.Structure):
        _fields_ = [("delete_file", wintypes.BOOLEAN)]


    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _GET_FILE_INFORMATION = _KERNEL32.GetFileInformationByHandle
    _GET_FILE_INFORMATION.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ByHandleFileInformation),
    ]
    _GET_FILE_INFORMATION.restype = wintypes.BOOL
    _CLOSE_HANDLE = _KERNEL32.CloseHandle
    _CLOSE_HANDLE.argtypes = [wintypes.HANDLE]
    _CLOSE_HANDLE.restype = wintypes.BOOL
    _SET_FILE_POINTER = _KERNEL32.SetFilePointerEx
    _SET_FILE_POINTER.argtypes = [
        wintypes.HANDLE,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.DWORD,
    ]
    _SET_FILE_POINTER.restype = wintypes.BOOL
    _READ_FILE = _KERNEL32.ReadFile
    _READ_FILE.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    _READ_FILE.restype = wintypes.BOOL
    _GET_CURRENT_PROCESS = _KERNEL32.GetCurrentProcess
    _GET_CURRENT_PROCESS.argtypes = []
    _GET_CURRENT_PROCESS.restype = wintypes.HANDLE
    _DUPLICATE_HANDLE = _KERNEL32.DuplicateHandle
    _DUPLICATE_HANDLE.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    _DUPLICATE_HANDLE.restype = wintypes.BOOL

    _NTDLL = ctypes.WinDLL("ntdll")
    _NT_CREATE_FILE = _NTDLL.NtCreateFile
    _NT_CREATE_FILE.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(_ObjectAttributes),
        ctypes.POINTER(_IoStatusBlock),
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    _NT_CREATE_FILE.restype = ctypes.c_long
    _NT_SET_INFORMATION_FILE = _NTDLL.NtSetInformationFile
    _NT_SET_INFORMATION_FILE.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_IoStatusBlock),
        ctypes.c_void_p,
        wintypes.ULONG,
        wintypes.ULONG,
    ]
    _NT_SET_INFORMATION_FILE.restype = ctypes.c_long
    _NT_QUERY_DIRECTORY_FILE = _NTDLL.NtQueryDirectoryFile
    _NT_QUERY_DIRECTORY_FILE.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(_IoStatusBlock),
        ctypes.c_void_p,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.BOOLEAN,
        ctypes.POINTER(_UnicodeString),
        wintypes.BOOLEAN,
    ]
    _NT_QUERY_DIRECTORY_FILE.restype = ctypes.c_long
    _RTL_NT_STATUS_TO_DOS_ERROR = _NTDLL.RtlNtStatusToDosError
    _RTL_NT_STATUS_TO_DOS_ERROR.argtypes = [ctypes.c_long]
    _RTL_NT_STATUS_TO_DOS_ERROR.restype = wintypes.ULONG


@dataclass(frozen=True)
class PublicationFinding:
    path: str
    kind: str
    detail: str


@dataclass(frozen=True)
class _StablePublicationSnapshot:
    root_name: str
    kind: str
    ancestry_identities: tuple[tuple[int, int], ...]
    directory_identity: tuple[int, int]
    directory_identities: Mapping[str, tuple[int, int]]
    file_identities: Mapping[str, tuple[int, int, int]]
    file_signatures: Mapping[str, tuple[int, str]]
    captured_files: Mapping[str, bytes]
    materialized_root: Path | None = field(repr=False)


def _redacted_path_match(match: re.Match[str]) -> str:
    value = match.group(0)
    punctuation = ""
    while value and value[-1] in _TRAILING_PATH_PUNCTUATION:
        punctuation = value[-1] + punctuation
        value = value[:-1]
    return "<redacted-path>" + punctuation


def _path_spellings(value: str | Path) -> set[str]:
    raw = str(value)
    spellings = (
        set()
        if raw in {"", ".", ".."}
        else {raw, raw.replace("\\", "/"), raw.replace("/", "\\")}
    )
    try:
        resolved = str(Path(value).resolve())
    except (OSError, ValueError):
        resolved = ""
    if resolved:
        spellings.update(
            {resolved, resolved.replace("\\", "/"), resolved.replace("/", "\\")}
        )
    spellings.update(item.replace("\\", "\\\\") for item in tuple(spellings))
    return {item for item in spellings if item}


def redact_text(
    value: str,
    workspace: Path | None = None,
    *,
    sensitive_paths: Iterable[str | Path] = (),
) -> str:
    """Return text with credentials and identifying local paths removed."""

    redacted = str(value)
    path_values: list[str | Path] = list(sensitive_paths)
    if workspace is not None:
        path_values.append(workspace)
    candidates: set[str] = set()
    for path in path_values:
        candidates.update(_path_spellings(path))
    for candidate in sorted(candidates, key=len, reverse=True):
        escaped = re.escape(candidate)
        if "/" not in candidate and "\\" not in candidate:
            escaped = rf"(?<![A-Za-z0-9_.-]){escaped}(?![A-Za-z0-9_.-])"
        redacted = re.sub(escaped, "<redacted-path>", redacted, flags=re.I)
    redacted = _SSH_KEY_BLOCK.sub("<redacted>", redacted)
    redacted = _AUTHORIZATION.sub("Authorization: <redacted>", redacted)
    redacted = _COOKIE.sub("Cookie: <redacted>", redacted)
    redacted = _CREDENTIAL_QUERY.sub(lambda match: match.group("prefix") + "<redacted>", redacted)
    redacted = _SSH_PUBLIC.sub("ssh-key <redacted>", redacted)
    redacted = _PROVIDER_TOKEN.sub("<redacted>", redacted)
    redacted = _WINDOWS_HOME.sub(_redacted_path_match, redacted)
    redacted = _POSIX_HOME.sub(_redacted_path_match, redacted)
    redacted = _WINDOWS_ABSOLUTE.sub(_redacted_path_match, redacted)
    redacted = _POSIX_ABSOLUTE.sub(_redacted_path_match, redacted)
    return redacted


def _sensitive_kinds(value: str) -> set[str]:
    kinds: set[str] = set()
    if _AUTHORIZATION.search(value):
        kinds.add("authorization")
    if _CREDENTIAL_QUERY.search(value):
        kinds.add("credential_query")
    if (
        _WINDOWS_HOME.search(value)
        or _POSIX_HOME.search(value)
        or _WINDOWS_ABSOLUTE.search(value)
        or _POSIX_ABSOLUTE.search(value)
    ):
        kinds.add("local_path")
    if (
        _COOKIE.search(value)
        or _SSH_KEY_BLOCK.search(value)
        or _SSH_PRIVATE_HEADER.search(value)
        or _SSH_PUBLIC.search(value)
        or _PROVIDER_TOKEN.search(value)
    ):
        kinds.add("secret_pattern")
    return kinds


def _reader_chunks(
    read_at: Callable[[int, int], bytes], size: int
) -> Iterable[bytes]:
    offset = 0
    while offset < size:
        chunk = read_at(offset, min(_TEXT_CHUNK_BYTES, size - offset))
        if not chunk:
            raise OSError("stable file read ended early")
        offset += len(chunk)
        yield chunk


def _stable_reader_kinds(
    read_at: Callable[[int, int], bytes], size: int
) -> set[str]:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    is_text = True
    try:
        for chunk in _reader_chunks(read_at, size):
            if b"\x00" in chunk:
                is_text = False
                break
            decoder.decode(chunk, final=False)
        if is_text:
            decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        is_text = False
    if not is_text:
        if size <= 2 * _WINDOW_BYTES:
            sample = read_at(0, size)
        else:
            sample = read_at(0, _WINDOW_BYTES) + read_at(
                size - _WINDOW_BYTES, _WINDOW_BYTES
            )
        return _sensitive_kinds(sample.decode("utf-8", errors="ignore"))
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    overlap = ""
    kinds: set[str] = set()
    for chunk in _reader_chunks(read_at, size):
        text = decoder.decode(chunk, final=False)
        combined = overlap + text
        kinds.update(_sensitive_kinds(combined))
        overlap = combined[-_TEXT_OVERLAP:]
    tail = decoder.decode(b"", final=True)
    kinds.update(_sensitive_kinds(overlap + tail))
    return kinds


def _byte_text_kinds(value: bytes) -> set[str]:
    return _sensitive_kinds(value.decode("utf-8", errors="ignore"))


class _PngParseError(Exception):
    """The PNG layout is malformed or exceeds the audit's bounded subset."""


@dataclass
class _PngBudget:
    chunks: int = 0
    metadata_bytes: int = 0
    referenced_bytes: int = 0
    decompressed_bytes: int = 0
    work_units: int = 0

    @staticmethod
    def _reserve(current: int, count: int, limit: int) -> int:
        if (
            type(count) is not int
            or count < 0
            or type(limit) is not int
            or limit < 0
            or count > limit - current
        ):
            raise _PngParseError
        return current + count

    def reserve_chunk(self, payload_length: int) -> None:
        self.chunks = self._reserve(self.chunks, 1, _PNG_MAX_CHUNKS)
        self.reserve_work(1 + ((payload_length + 1_023) // 1_024))

    def reserve_metadata(self, count: int) -> None:
        self.metadata_bytes = self._reserve(
            self.metadata_bytes, count, _PNG_MAX_METADATA_BYTES
        )

    def reserve_referenced(self, count: int) -> None:
        self.referenced_bytes = self._reserve(
            self.referenced_bytes, count, _PNG_MAX_METADATA_BYTES
        )
        self.reserve_work((count + 1_023) // 1_024)

    def reserve_decompressed(self, count: int) -> None:
        self.decompressed_bytes = self._reserve(
            self.decompressed_bytes, count, _PNG_MAX_DECOMPRESSED_BYTES
        )
        self.reserve_work((count + 1_023) // 1_024)

    def reserve_work(self, count: int) -> None:
        self.work_units = self._reserve(
            self.work_units, count, _PNG_MAX_WORK_UNITS
        )


def _png_decompress(value: bytes, remaining: int) -> bytes:
    if type(remaining) is not int or remaining < 0:
        raise _PngParseError
    try:
        decompressor = zlib.decompressobj()
        result = decompressor.decompress(value, remaining + 1)
        if (
            len(result) > remaining
            or decompressor.unconsumed_tail
            or not decompressor.eof
            or decompressor.unused_data
        ):
            raise _PngParseError
        tail = decompressor.flush()
    except zlib.error as error:
        raise _PngParseError from error
    if len(tail) > remaining - len(result):
        raise _PngParseError
    return result + tail


def _png_keyword(payload: bytes) -> tuple[bytes, bytes]:
    try:
        keyword, remainder = payload.split(b"\0", 1)
    except ValueError as error:
        raise _PngParseError from error
    if not 1 <= len(keyword) <= 79 or b"\0" in keyword:
        raise _PngParseError
    return keyword, remainder


def _png_exif_comment_text(
    value: bytes, *, little_endian: bool
) -> str:
    if len(value) < 8:
        raise _PngParseError
    marker, encoded = value[:8], value[8:]
    try:
        if marker == b"ASCII\0\0\0":
            return encoded.rstrip(b"\0").decode("ascii", errors="strict")
        if marker == b"JIS\0\0\0\0\0":
            return encoded.rstrip(b"\0").decode("shift_jis", errors="strict")
        if marker == b"UNICODE\0":
            if len(encoded) % 2:
                raise _PngParseError
            if encoded.startswith((b"\xff\xfe", b"\xfe\xff")):
                return encoded.decode("utf-16", errors="strict").rstrip("\0")
            codec = "utf-16-le" if little_endian else "utf-16-be"
            return encoded.decode(codec, errors="strict").rstrip("\0")
    except UnicodeDecodeError as error:
        raise _PngParseError from error
    raise _PngParseError


def _png_exif_kinds(payload: bytes, budget: _PngBudget) -> set[str]:
    if len(payload) < 8 or payload[:2] not in {b"II", b"MM"}:
        raise _PngParseError
    little_endian = payload[:2] == b"II"
    order = "<" if little_endian else ">"
    if struct.unpack_from(f"{order}H", payload, 2)[0] != 42:
        raise _PngParseError
    first_ifd = struct.unpack_from(f"{order}I", payload, 4)[0]
    if first_ifd == 0:
        raise _PngParseError
    pending = [first_ifd]
    visited: set[int] = set()
    referenced_values: dict[tuple[int, int], bytes] = {}
    scanned_spans: set[tuple[str, tuple[int, int]]] = set()
    kinds: set[str] = set()
    while pending:
        ifd_offset = pending.pop()
        if (
            ifd_offset in visited
            or ifd_offset < 8
            or ifd_offset % 2
            or ifd_offset > len(payload) - 2
        ):
            raise _PngParseError
        visited.add(ifd_offset)
        entry_count = struct.unpack_from(f"{order}H", payload, ifd_offset)[0]
        entries_start = ifd_offset + 2
        entries_end = entries_start + entry_count * 12
        if entries_end > len(payload) - 4:
            raise _PngParseError
        budget.reserve_work(1 + entry_count)
        for index in range(entry_count):
            entry_offset = entries_start + index * 12
            tag, field_type, count = struct.unpack_from(
                f"{order}HHI", payload, entry_offset
            )
            type_size = _TIFF_TYPE_SIZES.get(field_type)
            if type_size is None:
                raise _PngParseError
            value_size = count * type_size
            value_field = payload[entry_offset + 8 : entry_offset + 12]
            value_span: tuple[int, int] | None = None
            if value_size <= 4:
                value = value_field[:value_size]
            else:
                value_offset = struct.unpack(f"{order}I", value_field)[0]
                if (
                    value_offset < 8
                    or value_offset % 2
                    or value_offset > len(payload) - value_size
                ):
                    raise _PngParseError
                value_span = (value_offset, value_size)
                value = referenced_values.get(value_span)
                if value is None:
                    budget.reserve_referenced(value_size)
                    value = payload[value_offset : value_offset + value_size]
                    referenced_values[value_span] = value
            if field_type == 2:
                scan_key = ("ascii", value_span)
                if value_span is None or scan_key not in scanned_spans:
                    if not value or value[-1] != 0:
                        raise _PngParseError
                    try:
                        text = value[:-1].replace(b"\0", b" ").decode(
                            "ascii", errors="strict"
                        )
                    except UnicodeDecodeError as error:
                        raise _PngParseError from error
                    kinds.update(_sensitive_kinds(text))
                    if value_span is not None:
                        scanned_spans.add(scan_key)
            if tag in _TIFF_XP_STRING_TAGS:
                scan_key = ("xp", value_span)
                if field_type not in {1, 7} or not value or len(value) % 2:
                    raise _PngParseError
                if value_span is None or scan_key not in scanned_spans:
                    try:
                        text = value.decode("utf-16-le", errors="strict")
                    except UnicodeDecodeError as error:
                        raise _PngParseError from error
                    if not text.endswith("\0"):
                        raise _PngParseError
                    kinds.update(_sensitive_kinds(text.rstrip("\0")))
                    if value_span is not None:
                        scanned_spans.add(scan_key)
            if tag == _TIFF_USER_COMMENT_TAG:
                scan_key = ("user-comment", value_span)
                if field_type != 7:
                    raise _PngParseError
                if value_span is None or scan_key not in scanned_spans:
                    kinds.update(
                        _sensitive_kinds(
                            _png_exif_comment_text(
                                value, little_endian=little_endian
                            )
                        )
                    )
                    if value_span is not None:
                        scanned_spans.add(scan_key)
            if tag in _TIFF_IFD_POINTER_TAGS:
                scan_key = ("ifd-pointer", value_span)
                if field_type not in {4, 13} or count == 0:
                    raise _PngParseError
                if value_span is None or scan_key not in scanned_spans:
                    budget.reserve_work(count)
                    pending.extend(
                        struct.unpack(f"{order}{count}I", value)
                    )
                    if value_span is not None:
                        scanned_spans.add(scan_key)
        next_ifd = struct.unpack_from(f"{order}I", payload, entries_end)[0]
        if next_ifd:
            pending.append(next_ifd)
    return kinds


def _png_metadata_kinds(
    chunk_type: bytes, payload: bytes, budget: _PngBudget
) -> set[str]:
    kinds = (
        _byte_text_kinds(payload)
        if chunk_type in {b"eXIf", b"iCCP", b"iTXt", b"tEXt", b"zTXt"}
        else set()
    )
    if chunk_type == b"eXIf":
        kinds.update(_png_exif_kinds(payload, budget))
        return kinds
    compressed: bytes | None = None
    if chunk_type == b"tEXt":
        _, text = _png_keyword(payload)
        if b"\0" in text:
            raise _PngParseError
    elif chunk_type == b"zTXt":
        _, remainder = _png_keyword(payload)
        if not remainder or remainder[0] != 0:
            raise _PngParseError
        compressed = remainder[1:]
    elif chunk_type == b"iTXt":
        _, remainder = _png_keyword(payload)
        if len(remainder) < 2 or remainder[0] not in {0, 1} or remainder[1] != 0:
            raise _PngParseError
        compressed_flag = remainder[0]
        try:
            _, remainder = remainder[2:].split(b"\0", 1)
            _, text = remainder.split(b"\0", 1)
        except ValueError as error:
            raise _PngParseError from error
        if compressed_flag:
            compressed = text
        elif b"\0" in text:
            raise _PngParseError
    elif chunk_type == b"iCCP":
        _, remainder = _png_keyword(payload)
        if not remainder or remainder[0] != 0:
            raise _PngParseError
        compressed = remainder[1:]
    elif chunk_type == b"sPLT":
        keyword, _ = _png_keyword(payload)
        kinds.update(_byte_text_kinds(keyword))
    if compressed is None:
        return kinds
    decoded = _png_decompress(
        compressed, _PNG_MAX_DECOMPRESSED_BYTES - budget.decompressed_bytes
    )
    budget.reserve_decompressed(len(decoded))
    kinds.update(_byte_text_kinds(decoded))
    return kinds


def _png_validate_ancillary(
    chunk_type: bytes,
    payload: bytes,
    *,
    color_type: int,
    bit_depth: int,
    palette_entries: int | None,
) -> None:
    exact_lengths = {
        b"cHRM": 32,
        b"cICP": 4,
        b"gAMA": 4,
        b"mDCV": 24,
        b"cLLI": 8,
        b"pHYs": 9,
        b"sRGB": 1,
        b"tIME": 7,
    }
    if chunk_type in exact_lengths and len(payload) != exact_lengths[chunk_type]:
        raise _PngParseError
    if chunk_type == b"sRGB" and payload[0] > 3:
        raise _PngParseError
    if chunk_type == b"pHYs" and payload[8] > 1:
        raise _PngParseError
    if chunk_type == b"sBIT":
        expected = {0: 1, 2: 3, 3: 3, 4: 2, 6: 4}[color_type]
        if len(payload) != expected or any(value == 0 or value > bit_depth for value in payload):
            raise _PngParseError
    if chunk_type == b"bKGD":
        expected = {0: 2, 2: 6, 3: 1, 4: 2, 6: 6}[color_type]
        if len(payload) != expected:
            raise _PngParseError
    if chunk_type == b"hIST":
        if palette_entries is None or len(payload) != 2 * palette_entries:
            raise _PngParseError
    if chunk_type == b"tRNS":
        if color_type in {4, 6}:
            raise _PngParseError
        if color_type == 0 and len(payload) != 2:
            raise _PngParseError
        if color_type == 2 and len(payload) != 6:
            raise _PngParseError
        if color_type == 3 and (
            palette_entries is None
            or not 1 <= len(payload) <= palette_entries
        ):
            raise _PngParseError
    if chunk_type == b"sPLT":
        _, remainder = _png_keyword(payload)
        if not remainder or remainder[0] not in {8, 16}:
            raise _PngParseError
        entry_size = 6 if remainder[0] == 8 else 10
        if len(remainder) == 1 or (len(remainder) - 1) % entry_size:
            raise _PngParseError
    if chunk_type == b"eXIf" and not payload:
        raise _PngParseError


def _png_scanline_passes(
    width: int, height: int, bits_per_pixel: int, interlace: int
) -> tuple[tuple[int, int], ...]:
    if interlace == 0:
        return (((width * bits_per_pixel + 7) // 8, height),)
    starts = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8),
              (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))
    passes: list[tuple[int, int]] = []
    for start_x, start_y, step_x, step_y in starts:
        if width <= start_x or height <= start_y:
            continue
        pass_width = (width - start_x + step_x - 1) // step_x
        pass_height = (height - start_y + step_y - 1) // step_y
        passes.append(((pass_width * bits_per_pixel + 7) // 8, pass_height))
    return tuple(passes)


def _png_paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def _png_unfilter(
    decoded: bytes,
    scanline_passes: tuple[tuple[int, int], ...],
    bytes_per_pixel: int,
) -> bytes:
    pixels = bytearray()
    offset = 0
    for row_length, row_count in scanline_passes:
        previous = b""
        for _ in range(row_count):
            filter_type = decoded[offset]
            if filter_type > 4:
                raise _PngParseError
            offset += 1
            filtered = decoded[offset : offset + row_length]
            offset += row_length
            row = bytearray(row_length)
            for index, value in enumerate(filtered):
                left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
                above = previous[index] if previous else 0
                upper_left = (
                    previous[index - bytes_per_pixel]
                    if previous and index >= bytes_per_pixel
                    else 0
                )
                if filter_type == 0:
                    predictor = 0
                elif filter_type == 1:
                    predictor = left
                elif filter_type == 2:
                    predictor = above
                elif filter_type == 3:
                    predictor = (left + above) // 2
                else:
                    predictor = _png_paeth(left, above, upper_left)
                row[index] = (value + predictor) & 0xFF
            pixels.extend(row)
            previous = row
    if offset != len(decoded):
        raise _PngParseError
    return bytes(pixels)


def _png_text_kinds(data: bytes) -> set[str] | None:
    if len(data) < len(_PNG_SIGNATURE) or data[:8] != _PNG_SIGNATURE:
        return None
    budget = _PngBudget()
    try:
        budget.reserve_work(1)
        offset = 8
        ihdr: tuple[int, int, int, int, int] | None = None
        palette_entries: int | None = None
        seen: set[bytes] = set()
        idat_payloads: list[bytes] = []
        idat_finished = False
        kinds: set[str] = set()
        while offset < len(data):
            if len(data) - offset < 12:
                raise _PngParseError
            length = struct.unpack_from(">I", data, offset)[0]
            chunk_type = data[offset + 4 : offset + 8]
            payload_start = offset + 8
            payload_end = payload_start + length
            chunk_end = payload_end + 4
            if (
                chunk_end > len(data)
                or len(chunk_type) != 4
                or any(not (65 <= value <= 90 or 97 <= value <= 122) for value in chunk_type)
                or not 65 <= chunk_type[2] <= 90
            ):
                raise _PngParseError
            stored_crc = struct.unpack_from(">I", data, payload_end)[0]
            if zlib.crc32(data[offset + 4 : payload_end]) & 0xFFFFFFFF != stored_crc:
                raise _PngParseError
            budget.reserve_chunk(length)
            payload = data[payload_start:payload_end]
            if chunk_type == b"IHDR":
                if offset != 8 or chunk_type in seen or length != 13:
                    raise _PngParseError
                (
                    width,
                    height,
                    bit_depth,
                    color_type,
                    compression,
                    filtering,
                    interlace,
                ) = struct.unpack(">IIBBBBB", payload)
                valid_depths = {
                    0: {1, 2, 4, 8, 16},
                    2: {8, 16},
                    3: {1, 2, 4, 8},
                    4: {8, 16},
                    6: {8, 16},
                }
                if (
                    width == 0
                    or height == 0
                    or color_type not in valid_depths
                    or bit_depth not in valid_depths[color_type]
                    or compression != 0
                    or filtering != 0
                    or interlace not in {0, 1}
                ):
                    raise _PngParseError
                ihdr = (width, height, bit_depth, color_type, interlace)
            elif ihdr is None:
                raise _PngParseError
            elif chunk_type == b"PLTE":
                if (
                    b"PLTE" in seen
                    or idat_payloads
                    or color_type in {0, 4}
                    or seen & _PNG_AFTER_PLTE_IF_PRESENT
                ):
                    raise _PngParseError
                if not 3 <= length <= 768 or length % 3:
                    raise _PngParseError
                palette_entries = length // 3
                if color_type == 3 and palette_entries > 2 ** bit_depth:
                    raise _PngParseError
            elif chunk_type == b"IDAT":
                if idat_finished or (color_type == 3 and palette_entries is None):
                    raise _PngParseError
                idat_payloads.append(payload)
            elif chunk_type == b"IEND":
                if chunk_type in seen or length != 0 or not idat_payloads:
                    raise _PngParseError
                if chunk_end != len(data):
                    raise _PngParseError
                seen.add(chunk_type)
                offset = chunk_end
                break
            else:
                if idat_payloads:
                    idat_finished = True
                if chunk_type[0] & 0x20 == 0 or chunk_type not in _PNG_KNOWN_ANCILLARY:
                    raise _PngParseError
                if chunk_type in _PNG_SINGLETON_ANCILLARY and chunk_type in seen:
                    raise _PngParseError
                if (
                    chunk_type in _PNG_BEFORE_PLTE_AND_IDAT
                    and b"PLTE" in seen
                ):
                    raise _PngParseError
                if chunk_type in _PNG_BEFORE_IDAT and idat_payloads:
                    raise _PngParseError
                if (
                    chunk_type in {b"bKGD", b"hIST", b"tRNS"}
                    and color_type == 3
                    and palette_entries is None
                ):
                    raise _PngParseError
                budget.reserve_metadata(length)
                _png_validate_ancillary(
                    chunk_type,
                    payload,
                    color_type=color_type,
                    bit_depth=bit_depth,
                    palette_entries=palette_entries,
                )
                kinds.update(_png_metadata_kinds(chunk_type, payload, budget))
            seen.add(chunk_type)
            offset = chunk_end
        if offset != len(data) or b"IEND" not in seen or ihdr is None:
            raise _PngParseError
        width, height, bit_depth, color_type, interlace = ihdr
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
        scanline_passes = _png_scanline_passes(
            width, height, bit_depth * channels, interlace
        )
        expected_size = sum(
            (row_length + 1) * row_count
            for row_length, row_count in scanline_passes
        )
        if expected_size > _PNG_MAX_DECOMPRESSED_BYTES - budget.decompressed_bytes:
            raise _PngParseError
        total_rows = sum(row_count for _, row_count in scanline_passes)
        budget.reserve_work(
            total_rows + ((expected_size + 1_023) // 1_024)
        )
        decoded = _png_decompress(
            b"".join(idat_payloads),
            _PNG_MAX_DECOMPRESSED_BYTES - budget.decompressed_bytes,
        )
        budget.reserve_decompressed(len(decoded))
        if len(decoded) != expected_size:
            raise _PngParseError
        _png_unfilter(
            decoded,
            scanline_passes,
            max(1, (bit_depth * channels + 7) // 8),
        )
        return kinds
    except (OverflowError, struct.error, _PngParseError):
        return None


class _GifParseError(Exception):
    """The GIF layout is malformed or exceeds the bounded parser subset."""


@dataclass
class _GifBudget:
    blocks: int = 0
    metadata_bytes: int = 0
    logical_pixels: int = 0
    image_pixels: int = 0
    frames: int = 0
    compressed_bytes: int = 0
    sub_blocks: int = 0
    decode_work: int = 0

    def _reserve(self, field: str, count: int, limit: int) -> None:
        if type(count) is not int or count < 0:
            raise _GifParseError
        current = getattr(self, field)
        if count > limit - current:
            raise _GifParseError
        setattr(self, field, current + count)

    def reserve_block(self) -> None:
        self._reserve("blocks", 1, _GIF_MAX_BLOCKS)

    def reserve_metadata(self, count: int) -> None:
        self._reserve("metadata_bytes", count, _GIF_MAX_METADATA_BYTES)

    def reserve_logical_pixels(self, count: int) -> None:
        self._reserve("logical_pixels", count, _GIF_MAX_LOGICAL_PIXELS)

    def reserve_image(self, pixels: int) -> None:
        if pixels > _GIF_MAX_IMAGE_PIXELS:
            raise _GifParseError
        self._reserve("image_pixels", pixels, _GIF_MAX_CUMULATIVE_PIXELS)
        self._reserve("frames", 1, _GIF_MAX_FRAMES)

    def reserve_compressed(self, count: int) -> None:
        self._reserve("compressed_bytes", count, _GIF_MAX_COMPRESSED_BYTES)

    def reserve_sub_block(self) -> None:
        self._reserve("sub_blocks", 1, _GIF_MAX_SUB_BLOCKS)

    def reserve_decode_work(self, count: int) -> None:
        self._reserve("decode_work", count, _GIF_MAX_DECODE_WORK)


@dataclass(frozen=True)
class _GifImageData:
    sub_blocks_start: int
    sub_blocks_end: int
    minimum_code_size: int
    palette_entries: int
    pixel_count: int
    compressed_bytes: int


def _gif_sub_blocks(
    data: bytes,
    offset: int,
    budget: _GifBudget,
    *,
    collect: bool,
    compressed: bool = False,
) -> tuple[bytes, int, int, int]:
    blocks: list[bytes] = []
    payload_bytes = 0
    sub_block_count = 0
    while offset < len(data):
        budget.reserve_block()
        budget.reserve_sub_block()
        sub_block_count += 1
        length = data[offset]
        offset += 1
        if length == 0:
            return b"".join(blocks), offset, payload_bytes, sub_block_count
        end = offset + length
        if end > len(data):
            raise _GifParseError
        payload_bytes += length
        if compressed:
            budget.reserve_compressed(length)
        if collect:
            budget.reserve_metadata(length)
            blocks.append(data[offset:end])
        offset = end
    raise _GifParseError


def _gif_fixed_extension(
    data: bytes,
    offset: int,
    size: int,
    budget: _GifBudget,
) -> tuple[bytes, int]:
    if offset >= len(data) or data[offset] != size:
        raise _GifParseError
    start = offset + 1
    end = start + size
    if end > len(data):
        raise _GifParseError
    budget.reserve_block()
    budget.reserve_metadata(size)
    return data[start:end], end


def _gif_image_bytes(data: bytes, start: int, end: int) -> Iterable[int]:
    offset = start
    while offset < end:
        length = data[offset]
        offset += 1
        if length == 0:
            if offset != end:
                raise _GifParseError
            return
        payload_end = offset + length
        if payload_end > end:
            raise _GifParseError
        yield from data[offset:payload_end]
        offset = payload_end
    raise _GifParseError


def _gif_lzw_decoded_pixels(data: bytes, image: _GifImageData) -> int:
    minimum_size = image.minimum_code_size
    clear_code = 1 << minimum_size
    end_code = clear_code + 1
    code_size = minimum_size + 1
    next_code = end_code + 1
    lengths = [0] * 4_096
    first_values = [0] * 4_096
    for value in range(clear_code):
        lengths[value] = 1
        first_values[value] = value

    byte_stream = iter(
        _gif_image_bytes(data, image.sub_blocks_start, image.sub_blocks_end)
    )
    bit_buffer = 0
    bit_count = 0
    bits_read = 0

    def read_code(width: int) -> int:
        nonlocal bit_buffer, bit_count, bits_read
        while bit_count < width:
            try:
                value = next(byte_stream)
            except StopIteration as error:
                raise _GifParseError from error
            bit_buffer |= value << bit_count
            bit_count += 8
        code = bit_buffer & ((1 << width) - 1)
        bit_buffer >>= width
        bit_count -= width
        bits_read += width
        return code

    decoded_pixels = 0
    previous_length: int | None = None
    previous_first = 0
    require_clear = True
    while True:
        code = read_code(code_size)
        if require_clear:
            if code != clear_code:
                raise _GifParseError
            require_clear = False
            continue
        if code == clear_code:
            code_size = minimum_size + 1
            next_code = end_code + 1
            previous_length = None
            continue
        if code == end_code:
            remaining_bits = image.compressed_bytes * 8 - bits_read
            if decoded_pixels != image.pixel_count or not 0 <= remaining_bits <= 7:
                raise _GifParseError
            return decoded_pixels

        if code < clear_code:
            if code >= image.palette_entries:
                raise _GifParseError
            current_length = 1
            current_first = code
        elif code < next_code and lengths[code] > 0:
            current_length = lengths[code]
            current_first = first_values[code]
        elif code == next_code and previous_length is not None:
            current_length = previous_length + 1
            current_first = previous_first
        else:
            raise _GifParseError
        decoded_pixels += current_length
        if decoded_pixels > image.pixel_count:
            raise _GifParseError

        if previous_length is not None and next_code < 4_096:
            lengths[next_code] = previous_length + 1
            first_values[next_code] = previous_first
            next_code += 1
            if next_code == 1 << code_size and code_size < 12:
                code_size += 1
        previous_length = current_length
        previous_first = current_first


def _gif_text_kinds(data: bytes) -> set[str] | None:
    if (
        len(data) < 14
        or len(data) > _CONTAINER_SCAN_LIMIT
        or data[:6] not in {b"GIF87a", b"GIF89a"}
    ):
        return None
    budget = _GifBudget()
    try:
        logical_width, logical_height = struct.unpack_from("<HH", data, 6)
        if logical_width == 0 or logical_height == 0:
            raise _GifParseError
        logical_pixels = logical_width * logical_height
        budget.reserve_logical_pixels(logical_pixels)
        packed = data[10]
        global_palette_bits = (packed & 0x07) + 1 if packed & 0x80 else None
        global_entries = 1 << global_palette_bits if global_palette_bits else 0
        if global_entries and data[11] >= global_entries:
            raise _GifParseError
        offset = 13 + 3 * global_entries
        if offset > len(data):
            raise _GifParseError

        images: list[_GifImageData] = []
        kinds: set[str] = set()
        trailer_seen = False
        while offset < len(data):
            marker = data[offset]
            if marker == 0x3B:
                if not images or offset + 1 != len(data):
                    raise _GifParseError
                trailer_seen = True
                break
            if marker == 0x2C:
                if offset + 10 > len(data):
                    raise _GifParseError
                left, top, width, height = struct.unpack_from("<HHHH", data, offset + 1)
                image_packed = data[offset + 9]
                if (
                    width == 0
                    or height == 0
                    or left + width > logical_width
                    or top + height > logical_height
                    or image_packed & 0x18
                ):
                    raise _GifParseError
                image_pixels = width * height
                budget.reserve_image(image_pixels)
                offset += 10
                if image_packed & 0x80:
                    palette_bits = (image_packed & 0x07) + 1
                    palette_entries = 1 << palette_bits
                    offset += 3 * palette_entries
                else:
                    palette_bits = global_palette_bits
                    palette_entries = global_entries
                if offset > len(data) or palette_bits is None or offset >= len(data):
                    raise _GifParseError
                minimum_code_size = data[offset]
                if not max(2, palette_bits) <= minimum_code_size <= 8:
                    raise _GifParseError
                sub_blocks_start = offset + 1
                _, offset, compressed_bytes, sub_block_count = _gif_sub_blocks(
                    data,
                    sub_blocks_start,
                    budget,
                    collect=False,
                    compressed=True,
                )
                if compressed_bytes == 0:
                    raise _GifParseError
                budget.reserve_decode_work(
                    logical_pixels
                    + image_pixels
                    + 8 * compressed_bytes
                    + sub_block_count
                )
                images.append(
                    _GifImageData(
                        sub_blocks_start=sub_blocks_start,
                        sub_blocks_end=offset,
                        minimum_code_size=minimum_code_size,
                        palette_entries=palette_entries,
                        pixel_count=image_pixels,
                        compressed_bytes=compressed_bytes,
                    )
                )
                continue
            if marker != 0x21 or offset + 2 > len(data):
                raise _GifParseError
            label = data[offset + 1]
            offset += 2
            if label == 0xF9:
                control, offset = _gif_fixed_extension(data, offset, 4, budget)
                if control[0] & 0xE0 or offset >= len(data) or data[offset] != 0:
                    raise _GifParseError
                budget.reserve_block()
                offset += 1
                continue
            if label == 0xFE:
                payload, offset, _, _ = _gif_sub_blocks(
                    data, offset, budget, collect=True
                )
                kinds.update(_byte_text_kinds(payload))
                continue
            if label not in {0x01, 0xFF}:
                raise _GifParseError
            fixed_size = 12 if label == 0x01 else 11
            header, offset = _gif_fixed_extension(
                data, offset, fixed_size, budget
            )
            if label == 0x01:
                left, top, width, height = struct.unpack_from("<HHHH", header)
                if (
                    global_entries == 0
                    or width == 0
                    or height == 0
                    or left + width > logical_width
                    or top + height > logical_height
                    or header[8] == 0
                    or header[9] == 0
                    or header[10] >= global_entries
                    or header[11] >= global_entries
                ):
                    raise _GifParseError
                text_pixels = width * height
                budget.reserve_image(text_pixels)
            payload, offset, payload_bytes, sub_block_count = _gif_sub_blocks(
                data, offset, budget, collect=True
            )
            if label == 0x01:
                budget.reserve_decode_work(
                    logical_pixels + text_pixels + payload_bytes + sub_block_count
                )
            kinds.update(_byte_text_kinds(header + payload))
        if not trailer_seen:
            raise _GifParseError
        for image in images:
            _gif_lzw_decoded_pixels(data, image)
        return kinds
    except (OverflowError, struct.error, _GifParseError):
        return None


@dataclass(frozen=True)
class _IsoBox:
    box_type: bytes
    start: int
    payload_start: int
    end: int


class _IsoParseError(Exception):
    """The ISO-BMFF layout is malformed or outside the supported subset."""


@dataclass
class _IsoBudget:
    boxes: int = 0
    tracks: int = 0
    table_entries: int = 0
    samples: int = 0
    chunks: int = 0
    ranges: int = 0
    work: int = 0

    def _reserve(self, field: str, count: int, limit: int) -> None:
        if type(count) is not int or count < 0:
            raise _IsoParseError
        current = getattr(self, field)
        if count > limit - current:
            raise _IsoParseError
        setattr(self, field, current + count)

    def reserve_boxes(self, count: int) -> None:
        self._reserve("boxes", count, _ISO_MAX_BOXES)

    def reserve_tracks(self, count: int) -> None:
        self._reserve("tracks", count, _ISO_MAX_TRACKS)

    def reserve_table_entries(self, count: int) -> None:
        self._reserve("table_entries", count, _ISO_MAX_TABLE_ENTRIES)

    def reserve_samples(self, count: int) -> None:
        self._reserve("samples", count, _ISO_MAX_SAMPLES)

    def reserve_chunks(self, count: int) -> None:
        self._reserve("chunks", count, _ISO_MAX_CHUNKS)

    def reserve_ranges(self, count: int) -> None:
        self._reserve("ranges", count, _ISO_MAX_TEXT_RANGES)

    def reserve_work(self, count: int) -> None:
        self._reserve("work", count, _ISO_MAX_WORK)


@dataclass(frozen=True)
class _IsoSampleSizes:
    count: int
    default_size: int | None
    values: tuple[int, ...]

    def at(self, index: int) -> int:
        if not 0 <= index < self.count:
            raise _IsoParseError
        if self.default_size is not None:
            return self.default_size
        return self.values[index]


@dataclass(frozen=True)
class _IsoNoText:
    pass


@dataclass(frozen=True)
class _IsoTextRanges:
    ranges: tuple[tuple[int, int], ...]


_IsoTrackResult = _IsoNoText | _IsoTextRanges


class _IsoParser:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.budget = _IsoBudget()

    def boxes(self, start: int, end: int) -> tuple[_IsoBox, ...]:
        if not 0 <= start <= end <= len(self.data):
            raise _IsoParseError
        boxes: list[_IsoBox] = []
        offset = start
        while offset < end:
            remaining = end - offset
            if remaining < 8:
                raise _IsoParseError
            size = struct.unpack_from(">I", self.data, offset)[0]
            box_type = self.data[offset + 4 : offset + 8]
            header_size = 8
            if size == 1:
                if remaining < 16:
                    raise _IsoParseError
                size = struct.unpack_from(">Q", self.data, offset + 8)[0]
                header_size = 16
            elif size == 0:
                size = remaining
            if (
                size < header_size
                or size > remaining
                or any(byte < 0x20 or byte > 0x7E for byte in box_type)
            ):
                raise _IsoParseError
            self.budget.reserve_boxes(1)
            self.budget.reserve_work(1)
            boxes.append(
                _IsoBox(
                    box_type=box_type,
                    start=offset,
                    payload_start=offset + header_size,
                    end=offset + size,
                )
            )
            offset += size
        return tuple(boxes)

    @staticmethod
    def one(boxes: tuple[_IsoBox, ...], box_type: bytes) -> _IsoBox:
        match: _IsoBox | None = None
        for box in boxes:
            if box.box_type != box_type:
                continue
            if match is not None:
                raise _IsoParseError
            match = box
        if match is None:
            raise _IsoParseError
        return match

    @staticmethod
    def one_of(boxes: tuple[_IsoBox, ...], box_types: set[bytes]) -> _IsoBox:
        match: _IsoBox | None = None
        for box in boxes:
            if box.box_type not in box_types:
                continue
            if match is not None:
                raise _IsoParseError
            match = box
        if match is None:
            raise _IsoParseError
        return match

    def full_box_payload(self, box: _IsoBox) -> int:
        if box.payload_start + 4 > box.end or self.data[box.payload_start] != 0:
            raise _IsoParseError
        return box.payload_start + 4

    def sample_descriptions(self, box: _IsoBox) -> tuple[tuple[bytes, int], ...]:
        payload = self.full_box_payload(box)
        if payload + 4 > box.end:
            raise _IsoParseError
        entry_count = struct.unpack_from(">I", self.data, payload)[0]
        self.budget.reserve_table_entries(entry_count)
        self.budget.reserve_work(entry_count)
        entries = self.boxes(payload + 4, box.end)
        if len(entries) != entry_count:
            raise _IsoParseError
        descriptions: list[tuple[bytes, int]] = []
        for entry in entries:
            if entry.payload_start + 8 > entry.end:
                raise _IsoParseError
            reference_index = struct.unpack_from(">H", self.data, entry.payload_start + 6)[0]
            if reference_index == 0:
                raise _IsoParseError
            descriptions.append((entry.box_type, reference_index))
        return tuple(descriptions)

    def data_references(self, minf_boxes: tuple[_IsoBox, ...]) -> tuple[bool, ...]:
        dinf = self.one(minf_boxes, b"dinf")
        dinf_boxes = self.boxes(dinf.payload_start, dinf.end)
        dref = self.one(dinf_boxes, b"dref")
        payload = self.full_box_payload(dref)
        if payload + 4 > dref.end:
            raise _IsoParseError
        entry_count = struct.unpack_from(">I", self.data, payload)[0]
        self.budget.reserve_table_entries(entry_count)
        self.budget.reserve_work(entry_count)
        entries = self.boxes(payload + 4, dref.end)
        if not entries or len(entries) != entry_count:
            raise _IsoParseError
        references: list[bool] = []
        for entry in entries:
            if entry.box_type not in {b"url ", b"urn "}:
                raise _IsoParseError
            self.full_box_payload(entry)
            version_and_flags = struct.unpack_from(">I", self.data, entry.payload_start)[0]
            references.append(bool(version_and_flags & 0x000001))
        return tuple(references)

    def sample_sizes(self, box: _IsoBox) -> _IsoSampleSizes:
        payload = self.full_box_payload(box)
        if payload + 8 > box.end:
            raise _IsoParseError
        if box.box_type == b"stsz":
            sample_size, count = struct.unpack_from(">II", self.data, payload)
            if count == 0:
                raise _IsoParseError
            self.budget.reserve_samples(count)
            if sample_size:
                if payload + 8 != box.end:
                    raise _IsoParseError
                return _IsoSampleSizes(count, sample_size, ())
            self.budget.reserve_table_entries(count)
            self.budget.reserve_work(count)
            table_start = payload + 8
            if table_start + 4 * count != box.end:
                raise _IsoParseError
            values = tuple(
                struct.unpack_from(">I", self.data, table_start + 4 * index)[0]
                for index in range(count)
            )
        elif box.box_type == b"stz2":
            field_size = self.data[payload + 3]
            count = struct.unpack_from(">I", self.data, payload + 4)[0]
            if count == 0:
                raise _IsoParseError
            self.budget.reserve_samples(count)
            self.budget.reserve_table_entries(count)
            self.budget.reserve_work(count)
            packed = self.data[payload + 8 : box.end]
            if field_size == 4:
                if len(packed) != (count + 1) // 2:
                    raise _IsoParseError
                values = tuple(
                    (packed[index // 2] >> 4)
                    if index % 2 == 0
                    else (packed[index // 2] & 0x0F)
                    for index in range(count)
                )
            elif field_size == 8:
                if len(packed) != count:
                    raise _IsoParseError
                values = tuple(packed)
            elif field_size == 16:
                if len(packed) != 2 * count:
                    raise _IsoParseError
                values = tuple(
                    struct.unpack_from(">H", packed, 2 * index)[0]
                    for index in range(count)
                )
            else:
                raise _IsoParseError
        else:
            raise _IsoParseError
        if any(size == 0 for size in values):
            raise _IsoParseError
        return _IsoSampleSizes(count, None, values)

    def chunk_offsets(self, box: _IsoBox) -> tuple[int, ...]:
        payload = self.full_box_payload(box)
        if payload + 4 > box.end:
            raise _IsoParseError
        count = struct.unpack_from(">I", self.data, payload)[0]
        if count == 0:
            raise _IsoParseError
        self.budget.reserve_chunks(count)
        self.budget.reserve_table_entries(count)
        self.budget.reserve_work(count)
        width = 4 if box.box_type == b"stco" else 8 if box.box_type == b"co64" else 0
        table_start = payload + 4
        if width == 0 or table_start + width * count != box.end:
            raise _IsoParseError
        code = ">I" if width == 4 else ">Q"
        offsets = tuple(
            struct.unpack_from(code, self.data, table_start + width * index)[0]
            for index in range(count)
        )
        if any(left >= right for left, right in zip(offsets, offsets[1:])):
            raise _IsoParseError
        return offsets

    def sample_to_chunk(self, box: _IsoBox) -> tuple[tuple[int, int, int], ...]:
        payload = self.full_box_payload(box)
        if payload + 4 > box.end:
            raise _IsoParseError
        count = struct.unpack_from(">I", self.data, payload)[0]
        if count == 0:
            raise _IsoParseError
        self.budget.reserve_table_entries(count)
        self.budget.reserve_work(count)
        table_start = payload + 4
        if table_start + 12 * count != box.end:
            raise _IsoParseError
        entries = tuple(
            struct.unpack_from(">III", self.data, table_start + 12 * index)
            for index in range(count)
        )
        if (
            entries[0][0] != 1
            or any(
                samples == 0 or description == 0
                for _, samples, description in entries
            )
            or any(left[0] >= right[0] for left, right in zip(entries, entries[1:]))
        ):
            raise _IsoParseError
        return entries

    def sample_ranges(
        self,
        stbl_boxes: tuple[_IsoBox, ...],
        media_ranges: tuple[tuple[int, int], ...],
        description_count: int,
        *,
        collect: bool,
    ) -> tuple[tuple[int, int], ...]:
        sizes = self.sample_sizes(
            self.one_of(stbl_boxes, {b"stsz", b"stz2"})
        )
        offsets = self.chunk_offsets(
            self.one_of(stbl_boxes, {b"stco", b"co64"})
        )
        mappings = self.sample_to_chunk(self.one(stbl_boxes, b"stsc"))
        if (
            mappings[-1][0] > len(offsets)
            or any(description > description_count for _, _, description in mappings)
        ):
            raise _IsoParseError
        covered_samples = 0
        for index, (first_chunk, samples_per_chunk, _) in enumerate(mappings):
            next_chunk = (
                mappings[index + 1][0]
                if index + 1 < len(mappings)
                else len(offsets) + 1
            )
            covered_samples += (next_chunk - first_chunk) * samples_per_chunk
        if covered_samples != sizes.count:
            raise _IsoParseError
        self.budget.reserve_work(len(offsets))
        self.budget.reserve_work(sizes.count)
        if collect:
            self.budget.reserve_ranges(sizes.count)
            self.budget.reserve_work(sizes.count)

        ranges: list[tuple[int, int]] = []
        sample_index = 0
        mapping_index = 0
        media_index = 0
        previous_end = -1
        for chunk_index, chunk_offset in enumerate(offsets, start=1):
            while (
                mapping_index + 1 < len(mappings)
                and mappings[mapping_index + 1][0] <= chunk_index
            ):
                mapping_index += 1
            samples_per_chunk = mappings[mapping_index][1]
            if chunk_offset < previous_end:
                raise _IsoParseError
            offset = chunk_offset
            for _ in range(samples_per_chunk):
                size = sizes.at(sample_index)
                sample_end = offset + size
                while media_index < len(media_ranges) and offset >= media_ranges[media_index][1]:
                    media_index += 1
                if (
                    size == 0
                    or media_index >= len(media_ranges)
                    or offset < media_ranges[media_index][0]
                    or sample_end > media_ranges[media_index][1]
                ):
                    raise _IsoParseError
                if collect:
                    ranges.append((offset, sample_end))
                offset = sample_end
                previous_end = sample_end
                sample_index += 1
        if sample_index != sizes.count or (collect and len(ranges) != sizes.count):
            raise _IsoParseError
        return tuple(ranges)

    def track(
        self,
        track: _IsoBox,
        media_ranges: tuple[tuple[int, int], ...],
    ) -> _IsoTrackResult:
        track_boxes = self.boxes(track.payload_start, track.end)
        mdia = self.one(track_boxes, b"mdia")
        mdia_boxes = self.boxes(mdia.payload_start, mdia.end)
        minf = self.one(mdia_boxes, b"minf")
        hdlr = self.one(mdia_boxes, b"hdlr")
        handler_payload = self.full_box_payload(hdlr)
        if handler_payload + 8 > hdlr.end:
            raise _IsoParseError
        handler_type = self.data[handler_payload + 4 : handler_payload + 8]
        minf_boxes = self.boxes(minf.payload_start, minf.end)
        stbl = self.one(minf_boxes, b"stbl")
        stbl_boxes = self.boxes(stbl.payload_start, stbl.end)
        descriptions = self.sample_descriptions(self.one(stbl_boxes, b"stsd"))
        if not descriptions:
            raise _IsoParseError
        references = self.data_references(minf_boxes)
        if any(
            reference_index > len(references) or not references[reference_index - 1]
            for _, reference_index in descriptions
        ):
            raise _IsoParseError

        recognized = tuple(
            sample_type in _ISO_BMFF_TEXT_SAMPLE_TYPES
            for sample_type, _ in descriptions
        )
        is_text = handler_type in _ISO_BMFF_TEXT_HANDLER_TYPES
        if is_text:
            if len(descriptions) != 1 or not recognized[0]:
                raise _IsoParseError
        elif (
            handler_type not in _ISO_BMFF_NON_TEXT_HANDLER_TYPES
            or any(recognized)
        ):
            raise _IsoParseError
        sample_ranges = self.sample_ranges(
            stbl_boxes,
            media_ranges,
            len(descriptions),
            collect=is_text,
        )
        if is_text:
            return _IsoTextRanges(sample_ranges)
        return _IsoNoText()

    def parse(self) -> set[str]:
        top_boxes = self.boxes(0, len(self.data))
        if not top_boxes or any(
            box.box_type in {b"moof", b"mfra"} for box in top_boxes
        ):
            raise _IsoParseError
        moov = self.one(top_boxes, b"moov")
        media_ranges = tuple(
            (box.payload_start, box.end)
            for box in top_boxes
            if box.box_type == b"mdat"
        )
        if not media_ranges:
            raise _IsoParseError
        moov_children = self.boxes(moov.payload_start, moov.end)
        if any(box.box_type == b"mvex" for box in moov_children):
            raise _IsoParseError
        track_count = sum(box.box_type == b"trak" for box in moov_children)
        if track_count == 0:
            raise _IsoParseError
        self.budget.reserve_tracks(track_count)
        self.budget.reserve_work(track_count)

        text_ranges: list[tuple[int, int]] = []
        for track in moov_children:
            if track.box_type != b"trak":
                continue
            result = self.track(track, media_ranges)
            if isinstance(result, _IsoTextRanges):
                text_ranges.extend(result.ranges)
        self.budget.reserve_work(len(text_ranges))
        text_ranges.sort()
        if any(
            left[1] > right[0]
            for left, right in zip(text_ranges, text_ranges[1:])
        ):
            raise _IsoParseError

        kinds: set[str] = set()
        for box in top_boxes:
            if box.box_type != b"mdat":
                kinds.update(_byte_text_kinds(self.data[box.start : box.end]))
        for start, end in text_ranges:
            kinds.update(_byte_text_kinds(self.data[start:end]))
        return kinds


def _iso_bmff_text_kinds(data: bytes) -> set[str] | None:
    try:
        return _IsoParser(data).parse()
    except (_IsoParseError, struct.error):
        return None


def _npy_text_kinds(data: bytes) -> set[str] | None:
    if len(data) < 10 or not data.startswith(b"\x93NUMPY"):
        return None
    major = data[6]
    if major == 1:
        header_start = 10
        header_length = struct.unpack_from("<H", data, 8)[0]
        encoding = "latin1"
    elif major in {2, 3} and len(data) >= 12:
        header_start = 12
        header_length = struct.unpack_from("<I", data, 8)[0]
        encoding = "utf-8" if major == 3 else "latin1"
    else:
        return None
    header_end = header_start + header_length
    if header_end > len(data):
        return None
    try:
        header_text = data[header_start:header_end].decode(encoding)
        header = ast.literal_eval(header_text)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(header, dict):
        return None
    if (
        type(header.get("fortran_order")) is not bool
        or type(header.get("shape")) is not tuple
        or any(type(item) is not int or item < 0 for item in header["shape"])
    ):
        return None
    kinds = _sensitive_kinds(header_text)
    descriptor = header.get("descr")
    payload = data[header_end:]
    try:
        dtype = np.lib.format.descr_to_dtype(descriptor)
    except (TypeError, ValueError):
        return None
    if dtype.hasobject:
        return None
    count = 1
    for dimension in header["shape"]:
        count *= dimension
        if count * max(1, dtype.itemsize) > _CONTAINER_SCAN_LIMIT:
            return None
    expected_size = count * dtype.itemsize
    if expected_size != len(payload):
        return None
    try:
        values = np.frombuffer(payload, dtype=dtype, count=count)
    except (TypeError, ValueError):
        return None

    def scan_fields(array: np.ndarray) -> None:
        value_dtype = array.dtype
        if value_dtype.fields is not None:
            for name in value_dtype.names or ():
                scan_fields(array[name])
            return
        if value_dtype.kind not in {"S", "U"}:
            return
        for value in np.nditer(
            array,
            flags=["refs_ok", "zerosize_ok"],
            op_flags=["readonly"],
        ):
            item = value.item()
            text = (
                item.decode("latin1", errors="ignore")
                if isinstance(item, bytes)
                else str(item)
            )
            kinds.update(_sensitive_kinds(text))

    scan_fields(values)
    return kinds


def _npz_text_kinds(data: bytes) -> set[str] | None:
    eocd = data.rfind(b"PK\x05\x06")
    if eocd < 0 or eocd + 22 > len(data):
        return None
    comment_length = struct.unpack_from("<H", data, eocd + 20)[0]
    archive_end = eocd + 22 + comment_length
    if archive_end > len(data):
        return None
    kinds = _byte_text_kinds(data[eocd + 22 : archive_end])
    kinds.update(_byte_text_kinds(data[archive_end:]))
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if (
                not infos
                or len(infos) > _NPZ_MAX_ENTRIES
                or any(not info.filename.endswith(".npy") for info in infos)
            ):
                return None
            aggregate_size = 0
            for info in infos:
                aggregate_size += info.file_size
                if (
                    info.file_size > _NPZ_MAX_UNCOMPRESSED_BYTES
                    or aggregate_size > _NPZ_MAX_UNCOMPRESSED_BYTES
                    or (info.file_size > 0 and info.compress_size <= 0)
                    or (
                        info.file_size
                        / max(1, info.compress_size)
                        > _NPZ_MAX_COMPRESSION_RATIO
                    )
                ):
                    return None
            for info in infos:
                kinds.update(_sensitive_kinds(info.filename))
                kinds.update(_byte_text_kinds(info.comment))
                npy_kinds = _npy_text_kinds(archive.read(info))
                if npy_kinds is None:
                    return None
                kinds.update(npy_kinds)
    except (OSError, ValueError, zipfile.BadZipFile):
        return None
    return kinds


def _format_aware_text_kinds(suffix: str, data: bytes) -> set[str] | None:
    if suffix == ".gif":
        return _gif_text_kinds(data)
    if suffix == ".png":
        return _png_text_kinds(data)
    if suffix in {".mp4", ".m4v", ".mov"}:
        return _iso_bmff_text_kinds(data)
    if suffix == ".npz":
        return _npz_text_kinds(data)
    return None


def validate_media_container_bytes(suffix: str, content: bytes) -> bool:
    """Validate one bounded complete media container through its exact EOF."""

    if type(suffix) is not str or type(content) is not bytes:
        raise ValueError("media container validation input is invalid")
    normalized = suffix.lower()
    if normalized not in {".png", ".gif", ".mp4", ".m4v", ".mov"}:
        raise ValueError("media container type is unsupported")
    if len(content) > _CONTAINER_SCAN_LIMIT:
        return False
    return _format_aware_text_kinds(normalized, content) is not None


def _strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _media_metadata_status(
    path: Path,
    *,
    stable_fd: int | None = None,
    stable_stream: BinaryIO | None = None,
) -> tuple[bool, bool]:
    executable = shutil.which("ffprobe")
    if executable is None:
        return False, False
    probe_path = str(path)
    run_options: dict[str, object] = {}
    if stable_stream is not None:
        try:
            stable_stream.seek(0)
        except OSError:
            return False, False
        probe_path = "pipe:0"
        run_options["stdin"] = stable_stream
    elif stable_fd is not None and os.name == "posix":
        for prefix in ("/proc/self/fd", "/dev/fd"):
            candidate = f"{prefix}/{stable_fd}"
            if os.path.exists(candidate):
                probe_path = candidate
                run_options["pass_fds"] = (stable_fd,)
                break
        else:
            return False, False
    try:
        completed = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-show_entries",
                "format_tags:stream_tags",
                "-of",
                "json",
                probe_path,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
            **run_options,
        )
    except (OSError, subprocess.SubprocessError):
        return False, False
    if completed.returncode != 0:
        return False, False
    try:
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise ValueError
        streams = payload.get("streams")
        format_record = payload.get("format")
        if not isinstance(streams, list) or not streams or not isinstance(format_record, dict):
            raise ValueError
        tag_records: list[dict[str, object]] = []
        for record in [*streams, format_record]:
            if not isinstance(record, dict):
                raise ValueError
            tags = record.get("tags", {})
            if not isinstance(tags, dict) or any(
                not isinstance(key, str) or not isinstance(item, str)
                for key, item in tags.items()
            ):
                raise ValueError
            tag_records.append(tags)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, False
    sensitive = any(
        _sensitive_kinds(value)
        for record in tag_records
        for value in _strings(record)
    )
    return True, sensitive


def _is_link_or_reparse(file_stat: object) -> bool:
    mode = getattr(file_stat, "st_mode", 0)
    attributes = getattr(file_stat, "st_file_attributes", 0)
    return stat.S_ISLNK(mode) or bool(
        attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _stat_signature(file_stat: object) -> tuple[int, int, int, int, int]:
    return (
        int(getattr(file_stat, "st_dev", 0)),
        int(getattr(file_stat, "st_ino", 0)),
        int(getattr(file_stat, "st_mode", 0)),
        int(getattr(file_stat, "st_size", 0)),
        int(getattr(file_stat, "st_mtime_ns", 0)),
    )


def _lexical_root(path: str | Path) -> tuple[Path | None, bool]:
    raw = os.fspath(path)
    if not isinstance(raw, str) or "\x00" in raw:
        raise ValueError("publication path is invalid")
    lexical = Path(raw)
    if any(part == os.pardir for part in lexical.parts):
        return None, True
    if os.name == "nt" and lexical.drive and not lexical.root:
        raise ValueError("drive-relative publication paths are not supported")
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    return Path(os.path.abspath(os.fspath(lexical))), False


def _lexical_chain(root: Path) -> list[tuple[Path, object]]:
    parts = root.parts
    if not parts:
        raise ValueError("publication path must be absolute")
    current = Path(parts[0])
    chain: list[tuple[Path, object]] = []
    for index, part in enumerate(parts):
        if index:
            current /= part
        try:
            current_stat = current.lstat()
        except OSError as error:
            raise ValueError("publication path could not be inspected") from error
        chain.append((current, current_stat))
        if _is_link_or_reparse(current_stat):
            break
    return chain


def _lexical_ancestry_identities(root: Path) -> tuple[tuple[int, int], ...]:
    chain = _lexical_chain(root)
    if (
        not chain
        or chain[-1][0] != root
        or any(_is_link_or_reparse(item) for _, item in chain)
    ):
        raise ValueError("stable snapshot ancestry contains a link")
    return tuple(
        (
            int(getattr(item, "st_dev", 0)),
            int(getattr(item, "st_ino", 0)),
        )
        for _, item in chain
    )


@dataclass(frozen=True)
class _WinDirectoryEntry:
    name: str
    attributes: int
    size: int
    write_time: int
    file_id: int


class _WinRootReparseError(OSError):
    pass


class _WinAncestorReparseError(OSError):
    pass


def _win_handle_information(handle: object) -> tuple[int, int, int, int, int]:
    information = _ByHandleFileInformation()
    if not _GET_FILE_INFORMATION(handle, ctypes.byref(information)):
        raise OSError(ctypes.get_last_error(), "stable handle inspection failed")
    return (
        int(information.attributes),
        int(information.volume),
        (int(information.index_high) << 32) | int(information.index_low),
        (int(information.size_high) << 32) | int(information.size_low),
        (int(information.write.high) << 32) | int(information.write.low),
    )


def _win_nt_path(path: Path) -> str:
    value = str(path)
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    if value.startswith("\\\\"):
        return "\\??\\UNC\\" + value.lstrip("\\")
    return "\\??\\" + value


def _win_native_open(
    name: str, *, directory: bool, root_handle: object | None = None
) -> tuple[object, tuple[int, ...]]:
    if not name or len(name.encode("utf-16-le")) > 0xFFFC:
        raise OSError("stable no-follow name is invalid")
    if root_handle is not None and ("\\" in name or "/" in name or name in {".", ".."}):
        raise OSError("handle-relative child name is invalid")
    name_buffer = ctypes.create_unicode_buffer(name)
    unicode_name = _UnicodeString(
        len(name.encode("utf-16-le")),
        len(name_buffer) * ctypes.sizeof(ctypes.c_wchar),
        ctypes.cast(name_buffer, wintypes.LPWSTR),
    )
    attributes = _ObjectAttributes(
        ctypes.sizeof(_ObjectAttributes),
        root_handle,
        ctypes.pointer(unicode_name),
        0x00000040 | 0x00001000,
        None,
        None,
    )
    io_status = _IoStatusBlock()
    handle = wintypes.HANDLE()
    create_options = (
        (0x00000001 if directory else 0x00000040)
        | 0x00000020
        | 0x00200000
    )
    status = _NT_CREATE_FILE(
        ctypes.byref(handle),
        0x00000001 | 0x00000080 | 0x00100000,
        ctypes.byref(attributes),
        ctypes.byref(io_status),
        None,
        0,
        0x00000001 | 0x00000002 | 0x00000004,
        0x00000001,
        create_options,
        None,
        0,
    )
    if status != 0:
        error = int(_RTL_NT_STATUS_TO_DOS_ERROR(status))
        if error == 4395:
            raise _WinAncestorReparseError(
                error, "ancestor reparse point rejected by stable open"
            )
        raise OSError(error, "stable no-follow open failed")
    try:
        information = _win_handle_information(handle)
        if information[0] & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise _WinRootReparseError(
                "root reparse point rejected by stable open"
            )
        is_directory = bool(information[0] & 0x00000010)
        if is_directory != directory:
            raise OSError("stable open returned the wrong entry type")
        return handle, information
    except Exception:
        _CLOSE_HANDLE(handle)
        raise


def _win_create_child(
    directory_handle: object, name: str, *, directory: bool
) -> tuple[object, tuple[int, ...]]:
    if os.name != "nt":
        raise OSError("Windows stable creation is unavailable")
    if (
        not name
        or "\\" in name
        or "/" in name
        or name in {".", ".."}
        or len(name.encode("utf-16-le")) > 0xFFFC
    ):
        raise OSError("handle-relative child name is invalid")
    name_buffer = ctypes.create_unicode_buffer(name)
    unicode_name = _UnicodeString(
        len(name.encode("utf-16-le")),
        len(name_buffer) * ctypes.sizeof(ctypes.c_wchar),
        ctypes.cast(name_buffer, wintypes.LPWSTR),
    )
    attributes = _ObjectAttributes(
        ctypes.sizeof(_ObjectAttributes),
        directory_handle,
        ctypes.pointer(unicode_name),
        0x00000040 | 0x00001000,
        None,
        None,
    )
    io_status = _IoStatusBlock()
    handle = wintypes.HANDLE()
    create_options = (
        (0x00000001 if directory else 0x00000040)
        | 0x00000020
        | 0x00200000
    )
    desired_access = (
        (0x00000001 | 0x00000002 | 0x00000004 if directory else 0x00000002)
        | 0x00000080
        | 0x00010000
        | 0x00100000
    )
    share_access = (
        0x00000001 | 0x00000002 | 0x00000004
        if directory
        else 0x00000001 | 0x00000004
    )
    status = _NT_CREATE_FILE(
        ctypes.byref(handle),
        desired_access,
        ctypes.byref(attributes),
        ctypes.byref(io_status),
        None,
        0x00000100 if not directory else 0,
        share_access,
        0x00000002,
        create_options,
        None,
        0,
    )
    if status != 0:
        error = int(_RTL_NT_STATUS_TO_DOS_ERROR(status))
        raise OSError(error, "stable handle-relative create failed")
    try:
        information = _win_handle_information(handle)
        if information[0] & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise OSError("stable creation returned a reparse point")
        is_directory = bool(information[0] & 0x00000010)
        if is_directory != directory:
            raise OSError("stable creation returned the wrong entry type")
        return handle, information
    except Exception:
        _CLOSE_HANDLE(handle)
        raise


def _win_rename_child(
    file_handle: object, directory_handle: object, name: str
) -> None:
    if os.name != "nt":
        raise OSError("Windows stable rename is unavailable")
    if (
        not name
        or "\\" in name
        or "/" in name
        or name in {".", ".."}
        or len(name.encode("utf-16-le")) > 0xFFFC
    ):
        raise OSError("handle-relative child name is invalid")
    encoded_name = name.encode("utf-16-le")
    buffer_size = ctypes.sizeof(_FileRenameInformation) + len(encoded_name)
    rename_buffer = ctypes.create_string_buffer(buffer_size)
    rename = _FileRenameInformation.from_buffer(rename_buffer)
    rename.replace_if_exists = True
    rename.root_directory = directory_handle
    rename.file_name_length = len(encoded_name)
    ctypes.memmove(
        ctypes.addressof(rename_buffer) + _FileRenameInformation.file_name.offset,
        encoded_name,
        len(encoded_name),
    )
    io_status = _IoStatusBlock()
    status = _NT_SET_INFORMATION_FILE(
        file_handle,
        ctypes.byref(io_status),
        rename_buffer,
        buffer_size,
        10,  # FileRenameInformation
    )
    if status != 0:
        error = int(_RTL_NT_STATUS_TO_DOS_ERROR(status))
        raise OSError(error, "stable handle-relative rename failed")


def _win_delete_handle(file_handle: object) -> None:
    """Mark an uninstalled temporary file for deletion without reopening its path."""

    if os.name != "nt":
        raise OSError("Windows stable handle deletion is unavailable")
    disposition = _FileDispositionInformation(True)
    io_status = _IoStatusBlock()
    status = _NT_SET_INFORMATION_FILE(
        file_handle,
        ctypes.byref(io_status),
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
        13,  # FileDispositionInformation
    )
    if status != 0:
        error = int(_RTL_NT_STATUS_TO_DOS_ERROR(status))
        raise OSError(error, "stable handle-relative delete failed")


def _win_open_root(path: Path) -> tuple[object, tuple[int, ...]]:
    return _win_native_open(_win_nt_path(path), directory=True)


def _win_open_child(
    directory_handle: object, name: str, *, directory: bool
) -> tuple[object, tuple[int, ...]]:
    return _win_native_open(
        name, directory=directory, root_handle=directory_handle
    )


def _stable_directory_identity_no_follow(path: str | Path) -> tuple[int, int]:
    """Open a complete lexical ancestry without following links or reparses."""

    root, has_parent_traversal = _lexical_root(path)
    if has_parent_traversal or root is None:
        raise ValueError("stable directory path contains parent traversal")
    if os.name == "nt":
        root_handle: object | None = None
        try:
            root_handle, information = _win_open_root(root)
            return information[1], information[2]
        finally:
            if root_handle is not None:
                _CLOSE_HANDLE(root_handle)
    if os.name != "posix":
        raise ValueError("stable no-follow directory API is unavailable")
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise ValueError("stable no-follow directory API is unavailable")
    chain = _lexical_chain(root)
    if (
        not chain
        or chain[-1][0] != root
        or any(_is_link_or_reparse(item) for _, item in chain)
    ):
        raise ValueError("stable directory ancestry contains a link")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd: int | None = None
    try:
        for component, expected in chain:
            if directory_fd is None:
                opened_fd = os.open(component, flags)
            else:
                opened_fd = os.open(
                    component.name, flags, dir_fd=directory_fd
                )
                os.close(directory_fd)
            directory_fd = opened_fd
            opened = os.fstat(directory_fd)
            if _stat_signature(opened) != _stat_signature(expected):
                raise ValueError("stable directory ancestry changed during open")
        if directory_fd is None:
            raise ValueError("stable directory could not be opened")
        opened = os.fstat(directory_fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise ValueError("stable directory root is not a directory")
        return int(opened.st_dev), int(opened.st_ino)
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def _win_directory_entries(
    directory_handle: object, *, max_entries: int | None = None
) -> tuple[_WinDirectoryEntry, ...]:
    buffer = ctypes.create_string_buffer(64 * 1024)
    entries: list[_WinDirectoryEntry] = []
    restart = True
    while True:
        io_status = _IoStatusBlock()
        status = _NT_QUERY_DIRECTORY_FILE(
            directory_handle,
            None,
            None,
            None,
            ctypes.byref(io_status),
            buffer,
            len(buffer),
            37,
            False,
            None,
            restart,
        )
        restart = False
        unsigned_status = ctypes.c_ulong(status).value
        if unsigned_status == 0x80000006:
            break
        if unsigned_status not in {0, 0x80000005}:
            error = int(_RTL_NT_STATUS_TO_DOS_ERROR(status))
            raise OSError(error, "stable directory enumeration failed")
        used = int(io_status.information)
        offset = 0
        while offset < used:
            if used - offset < 104:
                raise OSError("stable directory enumeration was malformed")
            next_offset = struct.unpack_from("<I", buffer.raw, offset)[0]
            attributes = struct.unpack_from("<I", buffer.raw, offset + 56)[0]
            name_length = struct.unpack_from("<I", buffer.raw, offset + 60)[0]
            if name_length % 2 or offset + 104 + name_length > used:
                raise OSError("stable directory enumeration was malformed")
            name = buffer.raw[
                offset + 104 : offset + 104 + name_length
            ].decode("utf-16-le", errors="strict")
            if name not in {".", ".."}:
                entries.append(
                    _WinDirectoryEntry(
                        name=name,
                        attributes=attributes,
                        size=struct.unpack_from("<Q", buffer.raw, offset + 40)[0],
                        write_time=struct.unpack_from("<Q", buffer.raw, offset + 24)[0],
                        file_id=struct.unpack_from("<Q", buffer.raw, offset + 96)[0],
                    )
                )
                if max_entries is not None and len(entries) > max_entries:
                    raise OSError("stable directory entry cap exceeded")
            if next_offset == 0:
                break
            if next_offset < 104 or offset + next_offset > used:
                raise OSError("stable directory enumeration was malformed")
            offset += next_offset
        if status == 0 and used == 0:
            break
    return tuple(sorted(entries, key=lambda item: item.name))


def _win_entry_matches(
    entry: _WinDirectoryEntry, information: tuple[int, ...]
) -> bool:
    is_directory = bool(entry.attributes & 0x00000010)
    return (
        entry.attributes == information[0]
        and entry.file_id == information[2]
        and (
            is_directory
            or (
                entry.size == information[3]
                and entry.write_time == information[4]
            )
        )
    )


def _win_duplicate_stream(handle: object) -> BinaryIO:
    process = _GET_CURRENT_PROCESS()
    duplicate = wintypes.HANDLE()
    if not _DUPLICATE_HANDLE(
        process,
        handle,
        process,
        ctypes.byref(duplicate),
        0,
        False,
        0x00000002,
    ):
        raise OSError(ctypes.get_last_error(), "stable handle duplication failed")
    try:
        descriptor = msvcrt.open_osfhandle(
            int(duplicate.value), os.O_RDONLY | os.O_BINARY
        )
    except OSError:
        _CLOSE_HANDLE(duplicate)
        raise
    return os.fdopen(descriptor, "rb")


def _win_read_at(handle: object, offset: int, length: int) -> bytes:
    if not _SET_FILE_POINTER(handle, offset, None, 0):
        raise OSError(ctypes.get_last_error(), "stable file seek failed")
    buffer = ctypes.create_string_buffer(length)
    read = wintypes.DWORD()
    if not _READ_FILE(handle, buffer, length, ctypes.byref(read), None):
        raise OSError(ctypes.get_last_error(), "stable file read failed")
    return buffer.raw[: read.value]


def _snapshot_json_object(content: bytes) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("stable root manifest contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            content.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("stable root manifest is invalid JSON") from error
    if type(value) is not dict:
        raise ValueError("stable root manifest must be an object")
    return value


def _snapshot_relative_name(name: object) -> str:
    if (
        type(name) is not str
        or not name
        or len(name) > _STABLE_SNAPSHOT_MAX_RELATIVE_CHARS
        or "\\" in name
    ):
        raise ValueError("stable manifest file name is invalid")
    relative = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        relative.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or relative.as_posix() != name
        or any(part in {"", ".", ".."} for part in relative.parts)
        or len(relative.parts) > _STABLE_SNAPSHOT_MAX_PATH_DEPTH
    ):
        raise ValueError("stable manifest file name is invalid")
    return name


def _snapshot_declared_sizes(
    manifest_name: str, content: bytes
) -> tuple[str, dict[str, int]]:
    manifest = _snapshot_json_object(content)
    files = manifest.get("files")
    if type(files) is not dict:
        raise ValueError("stable root manifest lacks a file mapping")
    if len(files) + 1 > _STABLE_SNAPSHOT_MAX_FILES:
        raise ValueError("stable root manifest file cap exceeded")
    sizes = {manifest_name: len(content)}
    aggregate = len(content)
    work = 1 + ((len(content) + 1_048_575) // 1_048_576)
    for raw_name, entry in files.items():
        name = _snapshot_relative_name(raw_name)
        work += len(PurePosixPath(name).parts)
        if work > _STABLE_SNAPSHOT_MAX_WORK_UNITS:
            raise ValueError("stable root manifest work cap exceeded")
        if name in sizes or type(entry) is not dict:
            raise ValueError("stable root manifest file entry is invalid")
        size = entry.get("size")
        if type(size) is not int or size < 0:
            raise ValueError("stable root manifest file size is invalid")
        if size > _STABLE_SNAPSHOT_MAX_FILE_BYTES:
            raise ValueError("stable root manifest per-file cap exceeded")
        aggregate += size
        work += 1 + ((size + 1_048_575) // 1_048_576)
        if (
            aggregate > _STABLE_SNAPSHOT_MAX_AGGREGATE_BYTES
            or work > _STABLE_SNAPSHOT_MAX_WORK_UNITS
        ):
            raise ValueError("stable root manifest aggregate cap exceeded")
        sizes[name] = size
    kind = "suite" if manifest_name == "suite-manifest.json" else "variant"
    return kind, sizes


def _snapshot_expected_directories(names: Mapping[str, int]) -> set[str]:
    directories: set[str] = set()
    for name in names:
        parts = PurePosixPath(name).parts[:-1]
        for count in range(1, len(parts) + 1):
            directories.add(PurePosixPath(*parts[:count]).as_posix())
    return directories


def _snapshot_read_windows(handle: object, size: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = _win_read_at(handle, offset, min(1024 * 1024, size - offset))
        if not chunk:
            raise OSError("stable file ended before its declared size")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _win_filetime_to_unix_ns(value: int) -> int:
    return (value - 116_444_736_000_000_000) * 100


def _stable_capture_limit(name: str, manifest_name: str) -> int | None:
    if name == manifest_name:
        return _STABLE_SNAPSHOT_MAX_MANIFEST_BYTES
    if name == "dashboard/index.html":
        return _STABLE_CAPTURED_DASHBOARD_BYTES
    if name in {
        "dashboard/media/B0-preview.gif",
        "dashboard/media/B1-preview.gif",
    }:
        return _STABLE_CAPTURED_PREVIEW_BYTES
    return None


def _stable_materialized_path(root: Path, relative: str) -> Path:
    return root.joinpath(*PurePosixPath(relative).parts)


def _stream_windows_file(
    handle: object,
    size: int,
    destination: Path | None,
    capture_limit: int | None,
    *,
    retain_captured: bool,
) -> tuple[str, bytes | None]:
    if capture_limit is not None and size > capture_limit:
        raise ValueError("stable captured file exceeds its explicit cap")
    output = None
    captured = bytearray() if retain_captured and capture_limit is not None else None
    digest = sha256()
    try:
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            output = destination.open("xb")
        offset = 0
        while offset < size:
            chunk = _win_read_at(handle, offset, min(1024 * 1024, size - offset))
            if not chunk:
                raise OSError("stable file ended before its declared size")
            digest.update(chunk)
            if output is not None and output.write(chunk) != len(chunk):
                raise OSError("stable snapshot materialization was incomplete")
            if captured is not None:
                captured.extend(chunk)
            offset += len(chunk)
    finally:
        if output is not None:
            output.close()
    return digest.hexdigest(), bytes(captured) if captured is not None else None


def _snapshot_windows(
    root_handle: object,
    root_signature: tuple[int, ...],
    root_entries: tuple[_WinDirectoryEntry, ...],
    *,
    materialized_root: Path | None,
    retain_captured: bool,
) -> _StablePublicationSnapshot:
    manifest_entries = {
        entry.name: entry
        for entry in root_entries
        if entry.name in {"suite-manifest.json", "run_manifest.json"}
    }
    if len(manifest_entries) != 1:
        raise ValueError("stable root has ambiguous manifest identity")
    manifest_name, manifest_entry = next(iter(manifest_entries.items()))
    if (
        manifest_entry.attributes & (_FILE_ATTRIBUTE_REPARSE_POINT | 0x10)
        or manifest_entry.size > _STABLE_SNAPSHOT_MAX_MANIFEST_BYTES
    ):
        raise ValueError("stable root manifest is not a bounded regular file")
    manifest_handle: object | None = None
    try:
        manifest_handle, manifest_signature = _win_open_child(
            root_handle, manifest_name, directory=False
        )
        if not _win_entry_matches(manifest_entry, manifest_signature):
            raise ValueError("stable root manifest changed before read")
        manifest_content = _snapshot_read_windows(
            manifest_handle, manifest_entry.size
        )
        if _win_handle_information(manifest_handle) != manifest_signature:
            raise ValueError("stable root manifest changed during read")
    finally:
        if manifest_handle is not None:
            _CLOSE_HANDLE(manifest_handle)
    kind, expected_sizes = _snapshot_declared_sizes(
        manifest_name, manifest_content
    )
    expected_directories = _snapshot_expected_directories(expected_sizes)
    directory_handles: list[tuple[object, tuple[int, ...], str]] = []
    file_handles: list[tuple[object, tuple[int, ...], str, int]] = []
    directory_identities = {"": (root_signature[1], root_signature[2])}
    file_identities: dict[str, tuple[int, int, int]] = {}
    seen: set[str] = set()
    entry_work = 0

    def walk(
        directory_handle: object,
        directory_signature: tuple[int, ...],
        relative_directory: str,
        known_entries: tuple[_WinDirectoryEntry, ...] | None = None,
    ) -> None:
        nonlocal entry_work
        if _win_handle_information(directory_handle) != directory_signature:
            raise ValueError("stable directory changed before enumeration")
        remaining = _STABLE_SNAPSHOT_MAX_WORK_UNITS - entry_work
        if remaining < 0:
            raise ValueError("stable directory work cap exceeded")
        entries = (
            known_entries
            if known_entries is not None
            else _win_directory_entries(directory_handle, max_entries=remaining)
        )
        entry_work += len(entries)
        if entry_work > _STABLE_SNAPSHOT_MAX_WORK_UNITS:
            raise ValueError("stable directory work cap exceeded")
        for entry in entries:
            if any(mark in entry.name for mark in ("/", "\\")) or entry.name in {
                "",
                ".",
                "..",
            }:
                raise ValueError("stable directory entry name is invalid")
            relative = (
                entry.name
                if not relative_directory
                else f"{relative_directory}/{entry.name}"
            )
            if entry.attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError("stable directory contains a reparse entry")
            is_directory = bool(entry.attributes & 0x10)
            if is_directory:
                if relative not in expected_directories:
                    raise ValueError("stable directory contains an extra directory")
                if materialized_root is not None:
                    _stable_materialized_path(materialized_root, relative).mkdir()
                child_handle, child_signature = _win_open_child(
                    directory_handle, entry.name, directory=True
                )
                if not _win_entry_matches(entry, child_signature):
                    _CLOSE_HANDLE(child_handle)
                    raise ValueError("stable directory changed before child open")
                directory_handles.append((child_handle, child_signature, relative))
                directory_identities[relative] = (
                    child_signature[1],
                    child_signature[2],
                )
                walk(child_handle, child_signature, relative)
                continue
            expected_size = expected_sizes.get(relative)
            if expected_size is None:
                raise ValueError("stable directory contains an extra file")
            if entry.size != expected_size:
                raise ValueError("stable file size disagrees with its manifest")
            file_handle, file_signature = _win_open_child(
                directory_handle, entry.name, directory=False
            )
            if not _win_entry_matches(entry, file_signature):
                _CLOSE_HANDLE(file_handle)
                raise ValueError("stable file changed before open")
            file_handles.append(
                (file_handle, file_signature, relative, expected_size)
            )
            file_identities[relative] = (
                _win_filetime_to_unix_ns(file_signature[4]),
                file_signature[1],
                file_signature[2],
            )
            seen.add(relative)
        if _win_handle_information(directory_handle) != directory_signature:
            raise ValueError("stable directory changed during enumeration")

    try:
        walk(root_handle, root_signature, "", root_entries)
        if seen != set(expected_sizes):
            raise ValueError("stable directory file set disagrees with its manifest")
        captured: dict[str, bytes] = {}
        file_signatures: dict[str, tuple[int, str]] = {}
        for handle, signature, relative, size in file_handles:
            digest, retained = _stream_windows_file(
                handle,
                size,
                (
                    _stable_materialized_path(materialized_root, relative)
                    if materialized_root is not None
                    else None
                ),
                _stable_capture_limit(relative, manifest_name),
                retain_captured=retain_captured,
            )
            file_signatures[relative] = (size, digest)
            if retained is not None:
                captured[relative] = retained
            if _win_handle_information(handle) != signature:
                raise ValueError("stable file changed during capture")
        for handle, signature, _ in directory_handles:
            if _win_handle_information(handle) != signature:
                raise ValueError("stable directory changed during capture")
        if _win_handle_information(root_handle) != root_signature:
            raise ValueError("stable root changed during capture")
        if file_signatures.get(manifest_name) != (
            len(manifest_content),
            sha256(manifest_content).hexdigest(),
        ):
            raise ValueError("stable root manifest changed after preflight")
        return _StablePublicationSnapshot(
            root_name="",
            kind=kind,
            ancestry_identities=(),
            directory_identity=(root_signature[1], root_signature[2]),
            directory_identities=MappingProxyType(directory_identities),
            file_identities=MappingProxyType(file_identities),
            file_signatures=MappingProxyType(file_signatures),
            captured_files=MappingProxyType(captured),
            materialized_root=materialized_root,
        )
    finally:
        for handle, _, _, _ in file_handles:
            _CLOSE_HANDLE(handle)
        for handle, _, _ in reversed(directory_handles):
            _CLOSE_HANDLE(handle)


def _snapshot_posix_entries(
    directory_fd: int, remaining: int
) -> tuple[tuple[str, object], ...]:
    entries: list[tuple[str, object]] = []
    with os.scandir(directory_fd) as iterator:
        for entry in iterator:
            if len(entries) >= remaining:
                raise OSError("stable directory entry cap exceeded")
            entries.append(
                (
                    entry.name,
                    os.stat(entry.name, dir_fd=directory_fd, follow_symlinks=False),
                )
            )
    return tuple(sorted(entries, key=lambda item: item[0]))


def _snapshot_read_posix(file_fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(file_fd, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise OSError("stable file ended before its declared size")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _stream_posix_file(
    file_fd: int,
    size: int,
    destination: Path | None,
    capture_limit: int | None,
    *,
    retain_captured: bool,
) -> tuple[str, bytes | None]:
    if capture_limit is not None and size > capture_limit:
        raise ValueError("stable captured file exceeds its explicit cap")
    output = None
    captured = bytearray() if retain_captured and capture_limit is not None else None
    digest = sha256()
    try:
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            output = destination.open("xb")
        offset = 0
        while offset < size:
            chunk = os.pread(file_fd, min(1024 * 1024, size - offset), offset)
            if not chunk:
                raise OSError("stable file ended before its declared size")
            digest.update(chunk)
            if output is not None and output.write(chunk) != len(chunk):
                raise OSError("stable snapshot materialization was incomplete")
            if captured is not None:
                captured.extend(chunk)
            offset += len(chunk)
    finally:
        if output is not None:
            output.close()
    return digest.hexdigest(), bytes(captured) if captured is not None else None


def _snapshot_posix(
    root_fd: int,
    root_signature: tuple[int, int, int, int, int],
    root_entries: tuple[tuple[str, object], ...],
    *,
    materialized_root: Path | None,
    retain_captured: bool,
) -> _StablePublicationSnapshot:
    manifest_entries = {
        name: item
        for name, item in root_entries
        if name in {"suite-manifest.json", "run_manifest.json"}
    }
    if len(manifest_entries) != 1:
        raise ValueError("stable root has ambiguous manifest identity")
    manifest_name, manifest_stat = next(iter(manifest_entries.items()))
    if (
        not stat.S_ISREG(manifest_stat.st_mode)
        or _is_link_or_reparse(manifest_stat)
        or manifest_stat.st_size > _STABLE_SNAPSHOT_MAX_MANIFEST_BYTES
    ):
        raise ValueError("stable root manifest is not a bounded regular file")
    manifest_fd = os.open(
        manifest_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd
    )
    try:
        opened_manifest = os.fstat(manifest_fd)
        if _stat_signature(opened_manifest) != _stat_signature(manifest_stat):
            raise ValueError("stable root manifest changed before read")
        manifest_content = _snapshot_read_posix(
            manifest_fd, manifest_stat.st_size
        )
        if _stat_signature(os.fstat(manifest_fd)) != _stat_signature(manifest_stat):
            raise ValueError("stable root manifest changed during read")
    finally:
        os.close(manifest_fd)
    kind, expected_sizes = _snapshot_declared_sizes(
        manifest_name, manifest_content
    )
    expected_directories = _snapshot_expected_directories(expected_sizes)
    directory_fds: list[tuple[int, tuple[int, int, int, int, int], str]] = []
    file_fds: list[tuple[int, tuple[int, int, int, int, int], str, int]] = []
    directory_identities = {"": (root_signature[0], root_signature[1])}
    file_identities: dict[str, tuple[int, int, int]] = {}
    seen: set[str] = set()
    entry_work = 0

    def walk(
        directory_fd: int,
        directory_signature: tuple[int, int, int, int, int],
        relative_directory: str,
        known_entries: tuple[tuple[str, object], ...] | None = None,
    ) -> None:
        nonlocal entry_work
        if _stat_signature(os.fstat(directory_fd)) != directory_signature:
            raise ValueError("stable directory changed before enumeration")
        remaining = _STABLE_SNAPSHOT_MAX_WORK_UNITS - entry_work
        if remaining < 0:
            raise ValueError("stable directory work cap exceeded")
        entries = (
            known_entries
            if known_entries is not None
            else _snapshot_posix_entries(directory_fd, remaining)
        )
        entry_work += len(entries)
        if entry_work > _STABLE_SNAPSHOT_MAX_WORK_UNITS:
            raise ValueError("stable directory work cap exceeded")
        for name, entry_stat in entries:
            if any(mark in name for mark in ("/", "\\")) or name in {"", ".", ".."}:
                raise ValueError("stable directory entry name is invalid")
            relative = name if not relative_directory else f"{relative_directory}/{name}"
            if _is_link_or_reparse(entry_stat):
                raise ValueError("stable directory contains a linked entry")
            if stat.S_ISDIR(entry_stat.st_mode):
                if relative not in expected_directories:
                    raise ValueError("stable directory contains an extra directory")
                if materialized_root is not None:
                    _stable_materialized_path(materialized_root, relative).mkdir()
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                child_signature = _stat_signature(os.fstat(child_fd))
                if child_signature != _stat_signature(entry_stat):
                    os.close(child_fd)
                    raise ValueError("stable directory changed before child open")
                directory_fds.append((child_fd, child_signature, relative))
                directory_identities[relative] = (
                    child_signature[0],
                    child_signature[1],
                )
                walk(child_fd, child_signature, relative)
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise ValueError("stable directory contains a special file")
            expected_size = expected_sizes.get(relative)
            if expected_size is None:
                raise ValueError("stable directory contains an extra file")
            if entry_stat.st_size != expected_size:
                raise ValueError("stable file size disagrees with its manifest")
            file_fd = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd
            )
            file_signature = _stat_signature(os.fstat(file_fd))
            if file_signature != _stat_signature(entry_stat):
                os.close(file_fd)
                raise ValueError("stable file changed before open")
            file_fds.append((file_fd, file_signature, relative, expected_size))
            file_identities[relative] = (
                file_signature[4],
                file_signature[0],
                file_signature[1],
            )
            seen.add(relative)
        if _stat_signature(os.fstat(directory_fd)) != directory_signature:
            raise ValueError("stable directory changed during enumeration")

    try:
        walk(root_fd, root_signature, "", root_entries)
        if seen != set(expected_sizes):
            raise ValueError("stable directory file set disagrees with its manifest")
        captured: dict[str, bytes] = {}
        file_signatures: dict[str, tuple[int, str]] = {}
        for file_fd, signature, relative, size in file_fds:
            digest, retained = _stream_posix_file(
                file_fd,
                size,
                (
                    _stable_materialized_path(materialized_root, relative)
                    if materialized_root is not None
                    else None
                ),
                _stable_capture_limit(relative, manifest_name),
                retain_captured=retain_captured,
            )
            file_signatures[relative] = (size, digest)
            if retained is not None:
                captured[relative] = retained
            if _stat_signature(os.fstat(file_fd)) != signature:
                raise ValueError("stable file changed during capture")
        for directory_fd, signature, _ in directory_fds:
            if _stat_signature(os.fstat(directory_fd)) != signature:
                raise ValueError("stable directory changed during capture")
        if _stat_signature(os.fstat(root_fd)) != root_signature:
            raise ValueError("stable root changed during capture")
        if file_signatures.get(manifest_name) != (
            len(manifest_content),
            sha256(manifest_content).hexdigest(),
        ):
            raise ValueError("stable root manifest changed after preflight")
        return _StablePublicationSnapshot(
            root_name="",
            kind=kind,
            ancestry_identities=(),
            directory_identity=(root_signature[0], root_signature[1]),
            directory_identities=MappingProxyType(directory_identities),
            file_identities=MappingProxyType(file_identities),
            file_signatures=MappingProxyType(file_signatures),
            captured_files=MappingProxyType(captured),
            materialized_root=materialized_root,
        )
    finally:
        for file_fd, _, _, _ in file_fds:
            os.close(file_fd)
        for directory_fd, _, _ in reversed(directory_fds):
            os.close(directory_fd)


def _same_stable_tree(
    before: _StablePublicationSnapshot,
    after: _StablePublicationSnapshot,
) -> bool:
    return (
        before.kind == after.kind
        and before.directory_identity == after.directory_identity
        and before.directory_identities == after.directory_identities
        and before.file_identities == after.file_identities
        and before.file_signatures == after.file_signatures
    )


@contextmanager
def _stable_publication_snapshot(
    path: str | Path,
) -> Iterator[_StablePublicationSnapshot]:
    root, has_parent_traversal = _lexical_root(path)
    if has_parent_traversal or root is None or not root.name:
        raise ValueError("stable snapshot root is invalid")
    ancestry_identities = _lexical_ancestry_identities(root)
    temporary = tempfile.TemporaryDirectory(
        prefix="webvideo-to-data-stable-snapshot-"
    )
    materialized_root = Path(temporary.name) / root.name
    materialized_root.mkdir()
    if os.name == "nt":
        root_handle: object | None = None
        try:
            root_handle, root_signature = _win_open_root(root)
            root_entries = _win_directory_entries(
                root_handle, max_entries=_STABLE_SNAPSHOT_MAX_WORK_UNITS
            )
            snapshot = _snapshot_windows(
                root_handle,
                root_signature,
                root_entries,
                materialized_root=materialized_root,
                retain_captured=True,
            )
            snapshot = _StablePublicationSnapshot(
                root_name=root.name,
                kind=snapshot.kind,
                ancestry_identities=ancestry_identities,
                directory_identity=snapshot.directory_identity,
                directory_identities=snapshot.directory_identities,
                file_identities=snapshot.file_identities,
                file_signatures=snapshot.file_signatures,
                captured_files=snapshot.captured_files,
                materialized_root=materialized_root,
            )
            if snapshot.directory_identity != ancestry_identities[-1]:
                raise ValueError("stable snapshot ancestry changed during open")
            yield snapshot
            if _win_handle_information(root_handle) != root_signature:
                raise ValueError("stable snapshot root changed during use")
            after = _snapshot_windows(
                root_handle,
                root_signature,
                _win_directory_entries(
                    root_handle, max_entries=_STABLE_SNAPSHOT_MAX_WORK_UNITS
                ),
                materialized_root=None,
                retain_captured=False,
            )
            if not _same_stable_tree(snapshot, after):
                raise ValueError("stable snapshot tree changed during use")
            if _lexical_ancestry_identities(root) != ancestry_identities:
                raise ValueError("stable snapshot ancestry changed during use")
        finally:
            if root_handle is not None:
                _CLOSE_HANDLE(root_handle)
            temporary.cleanup()
        return
    if os.name != "posix" or any(
        not hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "pread")
    ):
        temporary.cleanup()
        raise ValueError("stable snapshot API is unavailable")
    chain = _lexical_chain(root)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd: int | None = None
    try:
        for component, expected in chain:
            if root_fd is None:
                opened_fd = os.open(component, flags)
            else:
                opened_fd = os.open(component.name, flags, dir_fd=root_fd)
                os.close(root_fd)
            root_fd = opened_fd
            if _stat_signature(os.fstat(root_fd)) != _stat_signature(expected):
                raise ValueError("stable snapshot ancestry changed during open")
        if root_fd is None:
            raise ValueError("stable snapshot root could not be opened")
        root_signature = _stat_signature(os.fstat(root_fd))
        root_entries = _snapshot_posix_entries(
            root_fd, _STABLE_SNAPSHOT_MAX_WORK_UNITS
        )
        snapshot = _snapshot_posix(
            root_fd,
            root_signature,
            root_entries,
            materialized_root=materialized_root,
            retain_captured=True,
        )
        snapshot = _StablePublicationSnapshot(
            root_name=root.name,
            kind=snapshot.kind,
            ancestry_identities=ancestry_identities,
            directory_identity=snapshot.directory_identity,
            directory_identities=snapshot.directory_identities,
            file_identities=snapshot.file_identities,
            file_signatures=snapshot.file_signatures,
            captured_files=snapshot.captured_files,
            materialized_root=materialized_root,
        )
        yield snapshot
        if _stat_signature(os.fstat(root_fd)) != root_signature:
            raise ValueError("stable snapshot root changed during use")
        after = _snapshot_posix(
            root_fd,
            root_signature,
            _snapshot_posix_entries(root_fd, _STABLE_SNAPSHOT_MAX_WORK_UNITS),
            materialized_root=None,
            retain_captured=False,
        )
        if not _same_stable_tree(snapshot, after):
            raise ValueError("stable snapshot tree changed during use")
        if _lexical_ancestry_identities(root) != ancestry_identities:
            raise ValueError("stable snapshot ancestry changed during use")
    finally:
        if root_fd is not None:
            os.close(root_fd)
        temporary.cleanup()


def _audit_publication_reader(
    candidate: Path,
    relative: str,
    read_at: Callable[[int, int], bytes],
    size: int,
    *,
    stable_fd: int | None = None,
    stable_stream: BinaryIO | None = None,
) -> tuple[PublicationFinding, ...]:
    try:
        kinds = _stable_reader_kinds(read_at, size)
    except OSError:
        kinds = {"secret_pattern"}
    suffix = candidate.suffix.lower()
    if suffix in _FORMAT_AWARE_SUFFIXES:
        if size <= _CONTAINER_SCAN_LIMIT:
            try:
                format_kinds = _format_aware_text_kinds(
                    suffix, read_at(0, size)
                )
            except OSError:
                format_kinds = None
        else:
            format_kinds = None
        if format_kinds is None:
            kinds.add("secret_pattern")
        elif suffix == ".png":
            # A verified PNG adapter scans defined text/profile/EXIF metadata,
            # including bounded decompression where the format defines it.
            # IDAT raster values remain numerical pixels after their structure,
            # filters, decompressed size, and work bounds have been validated.
            kinds = format_kinds
        else:
            # Only a successfully parsed container can suppress path-shaped
            # coincidences in compressed/binary payloads. Its declared text
            # samples, metadata, arrays, and trailing bytes remain audited.
            kinds.discard("local_path")
            kinds.update(format_kinds)
    findings = [
        PublicationFinding(relative, kind, "sensitive value detected")
        for kind in sorted(kinds)
    ]
    if suffix in _MEDIA_SUFFIXES:
        if os.name == "nt" and stable_stream is None:
            verifiable, sensitive = False, False
        else:
            verifiable, sensitive = _media_metadata_status(
                candidate,
                stable_fd=stable_fd,
                stable_stream=stable_stream,
            )
        if not verifiable or sensitive:
            detail = (
                "sensitive media tag detected"
                if sensitive
                else "media metadata could not be verified"
            )
            findings.append(
                PublicationFinding(relative, "media_metadata", detail)
            )
    return tuple(findings)


def audit_publication_bytes(
    relative_path: str | Path, content: bytes
) -> tuple[PublicationFinding, ...]:
    """Audit exact would-be publication bytes without writing or reopening them."""

    candidate = Path(relative_path)
    if (
        candidate.is_absolute()
        or not candidate.name
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or type(content) is not bytes
    ):
        raise ValueError("publication byte audit input is invalid")
    relative = candidate.as_posix()
    stream = io.BytesIO(content)
    return _audit_publication_reader(
        candidate,
        relative,
        lambda offset, length: content[offset : offset + length],
        len(content),
        stable_stream=stream if candidate.suffix.lower() in _MEDIA_SUFFIXES else None,
    )


def audit_publication_tree(path: str | Path) -> tuple[PublicationFinding, ...]:
    """Scan a publication tree without returning matched private values."""

    root, has_parent_traversal = _lexical_root(path)
    if has_parent_traversal:
        return (
            PublicationFinding(
                ".", "secret_pattern", "parent path traversal is not publishable"
            ),
        )
    if root is None:
        raise ValueError("publication path is invalid")
    chain: list[tuple[Path, object]] = []
    root_stat: object | None = None
    if os.name != "nt":
        chain = _lexical_chain(root)
        linked_component = next(
            (component for component, item in chain if _is_link_or_reparse(item)),
            None,
        )
        if linked_component is not None:
            detail = (
                "root symlink or reparse point is not publishable"
                if linked_component == root
                else "ancestor symlink or reparse point is not publishable"
            )
            return (PublicationFinding(".", "secret_pattern", detail),)
        if not chain or chain[-1][0] != root:
            return (
                PublicationFinding(
                    ".", "secret_pattern", "publication path ancestry is incomplete"
                ),
            )
        root_stat = chain[-1][1]
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError("publication path must be a directory")
    findings: list[PublicationFinding] = []
    seen: set[tuple[str, str]] = set()

    def add(relative: str, kind: str, detail: str) -> None:
        key = (relative, kind)
        if key not in seen:
            seen.add(key)
            findings.append(PublicationFinding(relative, kind, detail))

    def record_file(
        candidate: Path,
        relative: str,
        read_at: Callable[[int, int], bytes],
        size: int,
        *,
        stable_fd: int | None = None,
        stable_stream: BinaryIO | None = None,
    ) -> None:
        for finding in _audit_publication_reader(
            candidate,
            relative,
            read_at,
            size,
            stable_fd=stable_fd,
            stable_stream=stable_stream,
        ):
            add(finding.path, finding.kind, finding.detail)

    def unchanged(candidate: Path, before: object) -> bool:
        try:
            after = candidate.lstat()
        except OSError:
            return False
        return (
            not _is_link_or_reparse(after)
            and _stat_signature(after) == _stat_signature(before)
        )

    def walk_posix(directory_fd: int, directory: Path, before: object) -> None:
        relative_directory = directory.relative_to(root).as_posix() or "."
        opened_signature = _stat_signature(os.fstat(directory_fd))
        if opened_signature != _stat_signature(before) or not unchanged(directory, before):
            add(relative_directory, "secret_pattern", "directory changed before scan")
            return
        try:
            with os.scandir(directory_fd) as entries:
                names = sorted(entry.name for entry in entries)
        except OSError:
            add(relative_directory, "secret_pattern", "directory could not be inspected")
            return
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        for name in names:
            candidate = directory / name
            relative = candidate.relative_to(root).as_posix()
            try:
                candidate_stat = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
            except OSError:
                add(relative, "secret_pattern", "entry could not be inspected")
                continue
            if _is_link_or_reparse(candidate_stat):
                add(relative, "secret_pattern", "symlink or reparse point is not publishable")
                continue
            if stat.S_ISDIR(candidate_stat.st_mode):
                try:
                    child_fd = os.open(name, flags, dir_fd=directory_fd)
                except OSError:
                    add(relative, "secret_pattern", "directory stable open failed")
                    continue
                try:
                    if (
                        _stat_signature(os.fstat(child_fd)) != _stat_signature(candidate_stat)
                        or not unchanged(candidate, candidate_stat)
                    ):
                        add(relative, "secret_pattern", "directory changed before scan")
                        continue
                    walk_posix(child_fd, candidate, candidate_stat)
                    if (
                        _stat_signature(os.fstat(child_fd)) != _stat_signature(candidate_stat)
                        or not unchanged(candidate, candidate_stat)
                    ):
                        add(relative, "secret_pattern", "directory changed during scan")
                finally:
                    os.close(child_fd)
                continue
            if stat.S_ISREG(candidate_stat.st_mode):
                try:
                    file_fd = os.open(
                        name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd
                    )
                except OSError:
                    add(relative, "secret_pattern", "file stable open failed")
                    continue
                try:
                    opened = os.fstat(file_fd)
                    if (
                        _stat_signature(opened) != _stat_signature(candidate_stat)
                        or not unchanged(candidate, candidate_stat)
                    ):
                        add(relative, "secret_pattern", "file changed before scan")
                        continue
                    record_file(
                        candidate,
                        relative,
                        lambda offset, length: os.pread(file_fd, length, offset),
                        candidate_stat.st_size,
                        stable_fd=file_fd,
                    )
                    if (
                        _stat_signature(os.fstat(file_fd)) != _stat_signature(candidate_stat)
                        or not unchanged(candidate, candidate_stat)
                    ):
                        add(relative, "secret_pattern", "file changed during scan")
                finally:
                    os.close(file_fd)
                continue
            add(relative, "secret_pattern", "special file is not publishable")
        if (
            _stat_signature(os.fstat(directory_fd)) != opened_signature
            or not unchanged(directory, before)
        ):
            add(relative_directory, "secret_pattern", "directory changed during scan")

    def walk_windows(
        directory_handle: object,
        handle_signature: tuple[int, ...],
        relative_directory: str,
    ) -> None:
        if _win_handle_information(directory_handle) != handle_signature:
            add(relative_directory, "secret_pattern", "directory changed before scan")
            return
        try:
            entries = _win_directory_entries(directory_handle)
        except OSError:
            add(relative_directory, "secret_pattern", "directory could not be inspected")
            return
        for entry in entries:
            relative = (
                entry.name
                if relative_directory == "."
                else f"{relative_directory}/{entry.name}"
            )
            if entry.attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
                add(relative, "secret_pattern", "symlink or reparse point is not publishable")
                continue
            is_directory = bool(entry.attributes & 0x00000010)
            if is_directory:
                try:
                    child_handle, child_signature = _win_open_child(
                        directory_handle, entry.name, directory=True
                    )
                except OSError:
                    add(relative, "secret_pattern", "directory stable open failed")
                    continue
                try:
                    if not _win_entry_matches(entry, child_signature):
                        add(relative, "secret_pattern", "directory changed before scan")
                        continue
                    walk_windows(
                        child_handle,
                        child_signature,
                        relative,
                    )
                    if _win_handle_information(child_handle) != child_signature:
                        add(relative, "secret_pattern", "directory changed during scan")
                finally:
                    _CLOSE_HANDLE(child_handle)
                continue
            try:
                file_handle, file_signature = _win_open_child(
                    directory_handle, entry.name, directory=False
                )
            except OSError:
                add(relative, "secret_pattern", "file stable open failed")
                continue
            try:
                if not _win_entry_matches(entry, file_signature):
                    add(relative, "secret_pattern", "file changed before scan")
                    continue
                stable_stream: BinaryIO | None = None
                if Path(entry.name).suffix.lower() in _MEDIA_SUFFIXES:
                    try:
                        stable_stream = _win_duplicate_stream(file_handle)
                    except OSError:
                        stable_stream = None
                try:
                    record_file(
                        Path(entry.name),
                        relative,
                        lambda offset, length: _win_read_at(
                            file_handle, offset, length
                        ),
                        file_signature[3],
                        stable_stream=stable_stream,
                    )
                finally:
                    if stable_stream is not None:
                        stable_stream.close()
                if _win_handle_information(file_handle) != file_signature:
                    add(relative, "secret_pattern", "file changed during scan")
            finally:
                _CLOSE_HANDLE(file_handle)
        if _win_handle_information(directory_handle) != handle_signature:
            add(relative_directory, "secret_pattern", "directory changed during scan")

    snapshots = {component: item for component, item in chain}
    if os.name == "posix":
        required = ("O_DIRECTORY", "O_NOFOLLOW", "pread")
        if any(not hasattr(os, name) for name in required):
            return (
                PublicationFinding(
                    ".", "secret_pattern", "stable no-follow API is unavailable"
                ),
            )
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory_fd: int | None = None
        try:
            for component, component_stat in chain:
                if directory_fd is None:
                    opened_fd = os.open(component, flags)
                else:
                    opened_fd = os.open(
                        component.name, flags, dir_fd=directory_fd
                    )
                    os.close(directory_fd)
                directory_fd = opened_fd
                if _stat_signature(os.fstat(directory_fd)) != _stat_signature(component_stat):
                    raise OSError("directory identity changed while opening root")
            if directory_fd is None:
                raise OSError("publication root was not opened")
            if root_stat is None:
                raise OSError("publication root was not inspected")
            walk_posix(directory_fd, root, root_stat)
        except OSError:
            add(".", "secret_pattern", "stable root open failed")
        finally:
            if directory_fd is not None:
                os.close(directory_fd)
    elif os.name == "nt":
        root_handle: object | None = None
        try:
            root_handle, root_signature = _win_open_root(root)
            walk_windows(root_handle, root_signature, ".")
        except _WinRootReparseError:
            add(
                ".",
                "secret_pattern",
                "root symlink or reparse point is not publishable",
            )
        except _WinAncestorReparseError:
            add(
                ".",
                "secret_pattern",
                "ancestor symlink or reparse point is not publishable",
            )
        except OSError:
            add(".", "secret_pattern", "stable root open failed")
        finally:
            if root_handle is not None:
                _CLOSE_HANDLE(root_handle)
    else:
        add(".", "secret_pattern", "stable no-follow API is unavailable")

    if os.name == "posix":
        for component, before in snapshots.items():
            if not unchanged(component, before):
                add(".", "secret_pattern", "publication path ancestry changed")

    return tuple(findings)

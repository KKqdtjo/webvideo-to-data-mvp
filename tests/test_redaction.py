from __future__ import annotations

import base64
import json
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
import zipfile
import zlib

import numpy as np
import pytest

import webvideo_to_data.redaction as redaction_module
from webvideo_to_data.redaction import audit_publication_tree, redact_text


def test_redaction_removes_secret_url_query_authorization_and_workspace() -> None:
    workspace = Path.cwd().resolve()
    bearer = "TEST_" + "BEARER_SECRET"
    signed_value = "TEST_" + "SIGNED_QUERY_SECRET"
    raw = (
        f"Authorization: Bearer {bearer} "
        f"https://example.invalid/x?token={signed_value} "
        f"{workspace / 'video' / 'private.mp4'}"
    )
    clean = redact_text(raw, workspace=workspace)
    assert bearer not in clean
    assert signed_value not in clean
    assert str(workspace) not in clean
    assert "<redacted>" in clean


def test_redaction_covers_cookie_ssh_provider_tokens_and_home_paths() -> None:
    cookie = "TEST_" + "COOKIE_VALUE"
    ssh_material = "AA" + ("A" * 48)
    provider = "gh" + "p_" + ("Q" * 40)
    posix_user = "sample" + "-person"
    windows_user = "Sample" + "Person"
    raw = (
        f"Cookie: session={cookie}\n"
        f"ssh-rsa {ssh_material} fixture@example.invalid\n"
        f"provider={provider}\n"
        f"/home/{posix_user}/project/secret.txt\n"
        f"C:\\Users\\{windows_user}\\project\\secret.txt"
    )
    clean = redact_text(raw)
    for secret in (cookie, ssh_material, provider, posix_user, windows_user):
        assert secret not in clean


@pytest.mark.parametrize(
    "build_token",
    [
        lambda: "s" + "k-" + ("A" * 32),
        lambda: "h" + "f_" + ("B" * 32),
        lambda: "gl" + "pat-" + ("C" * 32),
        lambda: "AI" + "za" + ("D" * 32),
        lambda: "np" + "m_" + ("E" * 36),
    ],
)
def test_redaction_covers_common_provider_token_shapes(build_token: object) -> None:
    token = build_token()  # type: ignore[operator]
    assert token not in redact_text(f"provider={token}")


@pytest.mark.parametrize("key", ["token", "key", "signature", "credential"])
def test_redaction_covers_every_credential_query_key(key: str) -> None:
    secret = "TEST_" + "QUERY_VALUE"
    clean = redact_text(f"https://example.invalid/x?{key}={secret}&safe=1")
    assert secret not in clean
    assert "safe=1" in clean


def test_redaction_removes_external_absolute_and_known_sensitive_paths_without_urls() -> None:
    windows_path = "D:" + "\\private-area\\source.mp4"
    posix_path = "/mnt/" + "private-area/source.mp4"
    known_path = "relative-" + "private/source.mp4"
    url = "https://example.invalid/public/path"
    clean = redact_text(
        f"{windows_path} {posix_path} {known_path} {url}",
        sensitive_paths=(known_path,),
    )
    assert windows_path not in clean
    assert posix_path not in clean
    assert known_path not in clean
    assert url in clean


def test_redaction_covers_single_posix_rooted_and_drive_relative_paths() -> None:
    private_values = (
        "/" + "tmp",
        "\\" + "private\\fixture.bin",
        "D:" + "private\\fixture.bin",
        "D:" + "\\private\\fixture.bin",
        "\\\\" + "server\\share\\fixture.bin",
    )
    public = (
        "https://example.invalid https://example.invalid/public/item "
        "version=1.2/3 ordinary=left/right output"
    )
    clean = redact_text(" ".join((*private_values, public)))
    for private_value in private_values:
        assert private_value not in clean
    assert public in clean


@pytest.mark.parametrize("punctuation", [".", ",", ";", ":", "!", "?", ")", "]", "}"])
def test_redaction_preserves_trailing_non_path_punctuation(
    punctuation: str,
) -> None:
    assert redact_text("error at /tmp" + punctuation) == (
        "error at <redacted-path>" + punctuation
    )
    assert redact_text("error at D:" + "\\private\\clip.mp4" + punctuation) == (
        "error at <redacted-path>" + punctuation
    )


def test_known_short_relative_path_does_not_corrupt_surrounding_words() -> None:
    clean = redact_text("output failed at out", sensitive_paths=(Path("out"),))
    assert clean == "output failed at <redacted-path>"


def _write_tagged_mp4(path: Path, metadata: str) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("ffmpeg is required for publication metadata auditing")
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:r=5:d=0.4",
            "-metadata",
            f"comment={metadata}",
            "-c:v",
            "mpeg4",
            "-y",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _write_mov_text_mp4(path: Path, text_sample: str) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("ffmpeg is required for publication media auditing")
    subtitle = f"1\n00:00:00,000 --> 00:00:00,400\n{text_sample}\n"
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:r=5:d=0.4",
            "-f",
            "srt",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-map",
            "1:s:0",
            "-c:v",
            "mpeg4",
            "-c:s",
            "mov_text",
            "-movflags",
            "+faststart",
            "-shortest",
            "-y",
            str(path),
        ],
        input=subtitle,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _write_real_text_container(
    path: Path,
    text_sample: str,
    *,
    include_video: bool,
    sample_entry: str,
) -> None:
    executable = shutil.which("ffmpeg")
    probe = shutil.which("ffprobe")
    if executable is None or probe is None:
        pytest.skip("ffmpeg and ffprobe are required for text-track auditing")
    subtitle = f"1\n00:00:00,000 --> 00:00:00,400\n{text_sample}\n"
    command = [executable, "-v", "error"]
    if include_video:
        command.extend(
            ("-f", "lavfi", "-i", "color=c=black:s=32x24:r=5:d=0.8")
        )
    command.extend(("-f", "srt", "-i", "pipe:0"))
    if include_video:
        command.extend(("-map", "0:v:0", "-map", "1:s:0", "-c:v", "mjpeg"))
    command.extend(("-c:s", "mov_text"))
    if sample_entry == "text":
        command.extend(("-tag:s", "text"))
    if include_video:
        command.append("-shortest")
    command.extend(("-f", path.suffix.removeprefix("."), "-y", str(path)))
    completed = subprocess.run(
        command,
        input=subtitle,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    probed = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,codec_tag_string",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probed.returncode == 0, probed.stderr
    streams = json.loads(probed.stdout)["streams"]
    assert any(
        stream.get("codec_type") == "subtitle"
        and stream.get("codec_name") == "mov_text"
        and stream.get("codec_tag_string") == sample_entry
        for stream in streams
    )


def _write_real_classic_container(path: Path, stream_kind: str) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("ffmpeg is required for classic ISO-BMFF controls")
    command = [executable, "-v", "error"]
    if stream_kind in {"video", "mixed"}:
        command.extend(("-f", "lavfi", "-i", "color=c=black:s=32x24:r=5:d=0.4"))
    if stream_kind in {"audio", "mixed"}:
        command.extend(("-f", "lavfi", "-i", "sine=f=440:r=8000:d=0.4"))
    if stream_kind == "video":
        command.extend(("-c:v", "mpeg4"))
    elif stream_kind == "audio":
        command.extend(("-c:a", "aac"))
    elif stream_kind == "mixed":
        command.extend(("-c:v", "mjpeg", "-c:a", "pcm_s16le", "-shortest"))
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(f"unsupported classic stream kind: {stream_kind}")
    command.extend(("-y", str(path)))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def _iso_test_box(box_type: bytes, payload: bytes) -> bytes:
    assert len(box_type) == 4
    return struct.pack(">I4s", len(payload) + 8, box_type) + payload


def _iso_test_full_box(box_type: bytes, payload: bytes) -> bytes:
    return _iso_test_box(box_type, b"\0\0\0\0" + payload)


def _iso_test_fixture(
    *,
    handler_type: bytes = b"text",
    descriptions: tuple[bytes, ...] = (b"tx3g",),
    hierarchy_counts: dict[bytes, int] | None = None,
    sample_table: str = "stsz-fixed",
    sample_count: int = 1,
    sample_size: int = 1,
    size_entries_count: int | None = None,
    sample_payload: bytes | None = None,
    chunk_count: int = 1,
    offset_box_type: bytes = b"stco",
    stsc_entry_count: int = 1,
    stsc_samples_per_chunk: int | None = None,
    stsc_description_index: int = 1,
    track_count: int = 1,
    extra_free_boxes: int = 0,
) -> bytes:
    counts = hierarchy_counts or {}

    def repeated(box_type: bytes, content: bytes) -> bytes:
        return content * counts.get(box_type, 1)

    if sample_payload is None:
        sample_payload = b"x" * (sample_count * sample_size)
    materialized_sizes = (
        sample_count if size_entries_count is None else size_entries_count
    )
    free = _iso_test_box(b"free", b"") * extra_free_boxes
    ftyp = _iso_test_box(b"ftyp", b"isom\0\0\0\0isom")
    mdat_payload_start = len(free) + len(ftyp) + 8
    mdat = _iso_test_box(b"mdat", sample_payload)

    entries = b"".join(
        _iso_test_box(
            description,
            b"\0" * 6 + struct.pack(">H", 1),
        )
        for description in descriptions
    )
    stsd = _iso_test_full_box(
        b"stsd", struct.pack(">I", len(descriptions)) + entries
    )
    if sample_table == "stsz-fixed":
        size_box = _iso_test_full_box(
            b"stsz", struct.pack(">II", sample_size, sample_count)
        )
    elif sample_table == "stsz-variable":
        size_box = _iso_test_full_box(
            b"stsz",
            struct.pack(">II", 0, sample_count)
            + struct.pack(">I", sample_size) * materialized_sizes,
        )
    elif sample_table == "stz2":
        assert 0 <= sample_size <= 15
        packed = bytearray((materialized_sizes + 1) // 2)
        for index in range(materialized_sizes):
            if index % 2:
                packed[index // 2] |= sample_size
            else:
                packed[index // 2] = sample_size << 4
        size_box = _iso_test_full_box(
            b"stz2",
            b"\0\0\0\x04" + struct.pack(">I", sample_count) + bytes(packed),
        )
    else:
        raise AssertionError(f"unsupported test sample table: {sample_table}")

    assert chunk_count > 0
    assert sample_count % chunk_count == 0
    natural_samples_per_chunk = sample_count // chunk_count
    samples_per_chunk = (
        natural_samples_per_chunk
        if stsc_samples_per_chunk is None
        else stsc_samples_per_chunk
    )
    offsets = tuple(
        mdat_payload_start + index * natural_samples_per_chunk * sample_size
        for index in range(chunk_count)
    )
    if offset_box_type == b"stco":
        offset_payload = b"".join(struct.pack(">I", offset) for offset in offsets)
    elif offset_box_type == b"co64":
        offset_payload = b"".join(struct.pack(">Q", offset) for offset in offsets)
    else:
        raise AssertionError("unsupported test offset table")
    offset_box = _iso_test_full_box(
        offset_box_type, struct.pack(">I", len(offsets)) + offset_payload
    )
    assert 1 <= stsc_entry_count <= chunk_count
    if stsc_entry_count == 1:
        stsc_entries = ((1, samples_per_chunk, stsc_description_index),)
    else:
        assert stsc_entry_count == chunk_count
        stsc_entries = tuple(
            (index, samples_per_chunk, stsc_description_index)
            for index in range(1, chunk_count + 1)
        )
    stsc = _iso_test_full_box(
        b"stsc",
        struct.pack(">I", len(stsc_entries))
        + b"".join(struct.pack(">III", *entry) for entry in stsc_entries),
    )
    stbl = _iso_test_box(
        b"stbl",
        repeated(b"stsd", stsd)
        + repeated(size_box[4:8], size_box)
        + repeated(b"stsc", stsc)
        + repeated(offset_box_type, offset_box),
    )
    local_url = _iso_test_full_box(b"url ", b"")[:-4] + b"\0\0\0\x01"
    dref = _iso_test_full_box(b"dref", struct.pack(">I", 1) + local_url)
    dinf = _iso_test_box(b"dinf", dref)
    minf = _iso_test_box(b"minf", dinf + repeated(b"stbl", stbl))
    hdlr = _iso_test_full_box(b"hdlr", b"\0\0\0\0" + handler_type)
    mdia = _iso_test_box(
        b"mdia", repeated(b"hdlr", hdlr) + repeated(b"minf", minf)
    )
    trak = _iso_test_box(b"trak", repeated(b"mdia", mdia))
    moov = _iso_test_box(b"moov", trak * track_count)
    return free + ftyp + mdat + moov


def _iso_with_moov_child(
    content: bytes, child: bytes, *, before_tracks: bool
) -> bytes:
    moov_type = content.find(b"moov")
    assert moov_type >= 4
    moov_start = moov_type - 4
    moov_size = struct.unpack_from(">I", content, moov_start)[0]
    moov_end = moov_start + moov_size
    assert moov_size >= 8 and moov_end == len(content)
    insertion = moov_type + 4 if before_tracks else moov_end
    mutated = bytearray(content)
    struct.pack_into(">I", mutated, moov_start, moov_size + len(child))
    mutated[insertion:insertion] = child
    return bytes(mutated)


def _assert_iso_parser_rejects_quickly(content: bytes) -> None:
    started = time.perf_counter()
    assert redaction_module._iso_bmff_text_kinds(content) is None
    assert time.perf_counter() - started < 2.0


def _gif_test_fixture(
    *,
    logical_width: int = 1,
    logical_height: int = 1,
    image_width: int | None = None,
    image_height: int | None = None,
    image_count: int = 1,
    palette_bits: int = 1,
    local_palette_bits: int | None = None,
    lzw_minimum_code_size: int = 2,
    image_sub_blocks: tuple[bytes, ...] = (b"\x44\x01",),
) -> bytes:
    width = logical_width if image_width is None else image_width
    height = logical_height if image_height is None else image_height
    assert all(1 <= value <= 0xFFFF for value in (logical_width, logical_height))
    assert all(1 <= value <= 0xFFFF for value in (width, height))
    assert image_count > 0
    assert 1 <= palette_bits <= 8
    assert local_palette_bits is None or 1 <= local_palette_bits <= 8
    assert all(0 < len(block) <= 255 for block in image_sub_blocks)
    header = (
        b"GIF89a"
        + struct.pack("<HH", logical_width, logical_height)
        + bytes((0x80 | (palette_bits - 1), 0, 0))
        + b"\x00\x00\x00" * (1 << palette_bits)
    )
    image_data = b"".join(bytes((len(block),)) + block for block in image_sub_blocks)
    local_palette = (
        b"\x00\x00\x00" * (1 << local_palette_bits)
        if local_palette_bits is not None
        else b""
    )
    image_packed = (
        0x80 | (local_palette_bits - 1)
        if local_palette_bits is not None
        else 0
    )
    image = (
        b"\x2c"
        + struct.pack("<HHHH", 0, 0, width, height)
        + bytes((image_packed,))
        + local_palette
        + bytes((lzw_minimum_code_size,))
        + image_data
        + b"\x00"
    )
    return header + image * image_count + b"\x3b"


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


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_SAFE_SCANLINE = b"\0\0"


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    assert len(chunk_type) == 4
    crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I4s", len(payload), chunk_type) + payload + struct.pack(">I", crc)


def _png_ihdr(
    *,
    width: int = 1,
    height: int = 1,
    bit_depth: int = 8,
    color_type: int = 0,
    interlace: int = 0,
) -> bytes:
    return _png_chunk(
        b"IHDR",
        struct.pack(
            ">IIBBBBB",
            width,
            height,
            bit_depth,
            color_type,
            0,
            0,
            interlace,
        ),
    )


def _png_fixture(
    *,
    idat: bytes | None = None,
    before_idat: tuple[bytes, ...] = (),
    after_idat: tuple[bytes, ...] = (),
    width: int = 1,
    height: int = 1,
    bit_depth: int = 8,
    color_type: int = 0,
    interlace: int = 0,
) -> bytes:
    compressed = zlib.compress(_PNG_SAFE_SCANLINE) if idat is None else idat
    return b"".join(
        (
            _PNG_SIGNATURE,
            _png_ihdr(
                width=width,
                height=height,
                bit_depth=bit_depth,
                color_type=color_type,
                interlace=interlace,
            ),
            *before_idat,
            _png_chunk(b"IDAT", compressed),
            *after_idat,
            _png_chunk(b"IEND", b""),
        )
    )


def test_publication_audit_ignores_path_shaped_bytes_only_in_valid_png_idat() -> None:
    raw_scanline = base64.b64decode(
        "AKWuuMLM1+Lt+AQQHCk2Q1BebHqJmKe2xtbm9wgZKjxOYHOGmazA1Oj9Eic8Umh+lazD2vIKIjtUbYagutTvCg=="
    )
    compressed = base64.b64decode(
        "eAEBQAC//wClrrjCzNfi7fgEEBwpNkNQXmx6iZintsbW5vcIGSo8TmBzhpmswNTo/RInPFJofpWsw9ryCiI7VG2GoLrU7wovUyDT"
    )
    assert len(raw_scanline) == 64
    assert zlib.decompress(compressed) == raw_scanline

    findings = redaction_module.audit_publication_bytes(
        "simulation-only.png",
        _png_fixture(idat=compressed, width=63),
    )

    assert findings == ()


def test_publication_audit_ignores_path_shaped_decompressed_png_raster() -> None:
    private_path = ("C:" + "\\Users\\Alice\\private-source.mp4").encode("ascii")
    raw_scanline = base64.b64decode(
        "AAAAAAAAAAAAAAAAAAAAAABDOlxVc2Vyc1xBbGljZVxwcml2YXRlLXNvdXJjZS5tcDQAAAAAAAAAAAAAAAAAAAAA"
    )
    compressed = base64.b64decode(
        "eNpjYEADzlYxocWpRcUxjjmZyakxBUWZZYklqbrF+aVFyal6uQUm6BoAmpwMeg=="
    )
    assert len(raw_scanline) == 66
    assert private_path in raw_scanline
    assert private_path not in compressed
    assert zlib.decompress(compressed) == raw_scanline

    findings = redaction_module.audit_publication_bytes(
        "decoded-pixels.png",
        _png_fixture(idat=compressed, width=65),
    )

    assert findings == ()


def test_publication_audit_ignores_path_shaped_unfiltered_png_pixels() -> None:
    private_path = ("C:" + "\\Users\\Alice\\private-source.mp4").encode("ascii")
    filtered = bytes(
        (value - (private_path[index - 1] if index else 0)) & 0xFF
        for index, value in enumerate(private_path)
    )
    raw_scanline = b"\1" + filtered
    assert private_path not in raw_scanline

    findings = redaction_module.audit_publication_bytes(
        "sub-filtered-pixels.png",
        _png_fixture(idat=zlib.compress(raw_scanline), width=len(private_path)),
    )

    assert findings == ()


def test_publication_audit_ignores_short_path_shaped_pixel_bytes() -> None:
    raw_scanline = b"\0.\\U"

    findings = redaction_module.audit_publication_bytes(
        "random-pixels.png",
        _png_fixture(idat=zlib.compress(raw_scanline), width=3),
    )

    assert findings == ()


def test_publication_audit_does_not_scan_png_chunk_crc_as_text() -> None:
    text_chunk = _png_chunk(b"tEXt", b"Key\0public.aez")
    assert redaction_module._byte_text_kinds(text_chunk[-4:]) == {"local_path"}

    findings = redaction_module.audit_publication_bytes(
        "crc-coincidence.png",
        _png_fixture(before_idat=(text_chunk,)),
    )

    assert findings == ()


@pytest.mark.parametrize(
    ("chunk_type", "payload"),
    (
        (b"tEXt", b"Comment\0C:\\Users\\Alice\\private-source.mp4"),
        (
            b"zTXt",
            b"Comment\0\0" + zlib.compress(b"C:\\Users\\Alice\\private-source.mp4"),
        ),
        (
            b"iTXt",
            b"Comment\0\1\0en\0Comment\0"
            + zlib.compress(b"C:\\Users\\Alice\\private-source.mp4"),
        ),
        (
            b"iCCP",
            b"Profile\0\0" + zlib.compress(b"C:\\Users\\Alice\\private-source.mp4"),
        ),
        (b"eXIf", b"C:\\Users\\Alice\\private-source.mp4"),
        (
            b"sPLT",
            b"C:\\Users\\Alice\\private-source.mp4\0\x08\0\0\0\0\0\0",
        ),
    ),
)
def test_publication_audit_scans_png_text_profile_and_metadata_chunks(
    chunk_type: bytes, payload: bytes
) -> None:
    findings = redaction_module.audit_publication_bytes(
        "metadata.png",
        _png_fixture(before_idat=(_png_chunk(chunk_type, payload),)),
    )

    assert any(finding.kind == "local_path" for finding in findings)


@pytest.mark.parametrize(
    "corruption",
    (
        "length",
        "crc",
        "critical-order",
        "duplicate-ihdr",
        "duplicate-iend",
        "nonconsecutive-idat",
        "trailer",
        "unknown-chunk",
    ),
)
def test_publication_audit_fails_closed_on_malformed_png(
    corruption: str,
) -> None:
    idat = _png_chunk(b"IDAT", zlib.compress(_PNG_SAFE_SCANLINE))
    ihdr = _png_ihdr()
    iend = _png_chunk(b"IEND", b"")
    if corruption == "length":
        content = bytearray(_png_fixture())
        idat_type = content.find(b"IDAT")
        struct.pack_into(">I", content, idat_type - 4, len(idat) + len(content))
        content = bytes(content)
    elif corruption == "crc":
        content = bytearray(_png_fixture())
        idat_type = content.find(b"IDAT")
        idat_length = struct.unpack_from(">I", content, idat_type - 4)[0]
        content[idat_type + 4 + idat_length] ^= 1
        content = bytes(content)
    elif corruption == "critical-order":
        content = _PNG_SIGNATURE + idat + ihdr + iend
    elif corruption == "duplicate-ihdr":
        content = _PNG_SIGNATURE + ihdr + ihdr + idat + iend
    elif corruption == "duplicate-iend":
        content = _png_fixture() + iend
    elif corruption == "nonconsecutive-idat":
        content = (
            _PNG_SIGNATURE
            + ihdr
            + _png_chunk(b"IDAT", zlib.compress(_PNG_SAFE_SCANLINE)[:4])
            + _png_chunk(b"tEXt", b"Comment\0public")
            + _png_chunk(b"IDAT", zlib.compress(_PNG_SAFE_SCANLINE)[4:])
            + iend
        )
    elif corruption == "trailer":
        content = _png_fixture() + b"public trailer"
    else:
        content = _png_fixture(before_idat=(_png_chunk(b"ruSt", b"public"),))

    findings = redaction_module.audit_publication_bytes("malformed.png", content)

    assert any(finding.kind == "secret_pattern" for finding in findings)


def test_publication_audit_enforces_png_chunk_count_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(redaction_module, "_PNG_MAX_CHUNKS", 3, raising=False)
    content = _png_fixture(before_idat=(_png_chunk(b"tEXt", b"Key\0public"),))

    findings = redaction_module.audit_publication_bytes("too-many-chunks.png", content)

    assert any(finding.kind == "secret_pattern" for finding in findings)


def test_publication_audit_enforces_png_metadata_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        redaction_module, "_PNG_MAX_METADATA_BYTES", 8, raising=False
    )
    content = _png_fixture(before_idat=(_png_chunk(b"eXIf", b"public metadata"),))

    findings = redaction_module.audit_publication_bytes("metadata-cap.png", content)

    assert any(finding.kind == "secret_pattern" for finding in findings)


def test_publication_audit_enforces_png_work_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(redaction_module, "_PNG_MAX_WORK_UNITS", 2, raising=False)

    findings = redaction_module.audit_publication_bytes(
        "work-cap.png", _png_fixture()
    )

    assert any(finding.kind == "secret_pattern" for finding in findings)


def test_publication_audit_enforces_png_decompression_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        redaction_module, "_PNG_MAX_DECOMPRESSED_BYTES", 1, raising=False
    )

    findings = redaction_module.audit_publication_bytes(
        "decompression-cap.png", _png_fixture()
    )

    assert any(finding.kind == "secret_pattern" for finding in findings)


def test_publication_audit_precharges_png_row_work_before_decompression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(redaction_module, "_PNG_MAX_WORK_UNITS", 10, raising=False)

    def forbidden_decompression(*_: object) -> bytes:
        raise AssertionError("row-work rejection must precede decompression")

    monkeypatch.setattr(redaction_module, "_png_decompress", forbidden_decompression)

    findings = redaction_module.audit_publication_bytes(
        "too-many-rows.png",
        _png_fixture(height=11),
    )

    assert any(finding.kind == "secret_pattern" for finding in findings)


def _little_endian_tiff_xp_comment(text: str) -> bytes:
    encoded = (text + "\0").encode("utf-16-le")
    value_offset = 8 + 2 + 12 + 4
    return b"".join(
        (
            b"II\x2a\x00",
            struct.pack("<I", 8),
            struct.pack("<H", 1),
            struct.pack("<HHII", 0x9C9C, 1, len(encoded), value_offset),
            struct.pack("<I", 0),
            encoded,
        )
    )


def _big_endian_tiff_xp_comment(text: str) -> bytes:
    encoded = (text + "\0").encode("utf-16-le")
    value_offset = 8 + 2 + 12 + 4
    return b"".join(
        (
            b"MM\x00\x2a",
            struct.pack(">I", 8),
            struct.pack(">H", 1),
            struct.pack(">HHII", 0x9C9C, 1, len(encoded), value_offset),
            struct.pack(">I", 0),
            encoded,
        )
    )


def test_publication_audit_decodes_utf16le_xpcomment_from_valid_png_exif() -> None:
    tiff = _little_endian_tiff_xp_comment(
        "C:" + "\\Users\\Alice\\private-source.mp4"
    )

    findings = redaction_module.audit_publication_bytes(
        "exif-xpcomment.png",
        _png_fixture(before_idat=(_png_chunk(b"eXIf", tiff),)),
    )

    assert any(finding.kind == "local_path" for finding in findings)


def test_publication_audit_decodes_xpcomment_as_utf16le_in_big_endian_tiff() -> None:
    tiff = _big_endian_tiff_xp_comment(
        "C:" + "\\Users\\Alice\\private-source.mp4"
    )

    findings = redaction_module.audit_publication_bytes(
        "big-endian-exif-xpcomment.png",
        _png_fixture(before_idat=(_png_chunk(b"eXIf", tiff),)),
    )

    assert any(finding.kind == "local_path" for finding in findings)


def _little_endian_tiff_ascii_references(values: tuple[bytes, ...]) -> bytes:
    entries_end = 8 + 2 + len(values) * 12 + 4
    payloads: list[bytes] = []
    entries: list[bytes] = []
    offset = entries_end
    for index, value in enumerate(values):
        entries.append(
            struct.pack("<HHII", 0x010E + index, 2, len(value), offset)
        )
        payloads.append(value)
        offset += len(value)
        if offset % 2:
            payloads.append(b"\0")
            offset += 1
    return b"".join(
        (
            b"II\x2a\x00",
            struct.pack("<I", 8),
            struct.pack("<H", len(values)),
            *entries,
            struct.pack("<I", 0),
            *payloads,
        )
    )


def _little_endian_tiff_repeated_ascii_reference(
    value: bytes, count: int
) -> bytes:
    value_offset = 8 + 2 + count * 12 + 4
    return b"".join(
        (
            b"II\x2a\x00",
            struct.pack("<I", 8),
            struct.pack("<H", count),
            *(struct.pack("<HHII", 0x010E, 2, len(value), value_offset) for _ in range(count)),
            struct.pack("<I", 0),
            value,
        )
    )


def test_png_exif_deduplicates_identical_referenced_value_scan_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = (b"public " * 9_000) + b"C:\\Users\\Alice\\private.mp4\0"
    payload = _little_endian_tiff_repeated_ascii_reference(marker, 500)
    original_sensitive_kinds = redaction_module._sensitive_kinds
    long_scans = 0

    def count_long_scans(value: str) -> set[str]:
        nonlocal long_scans
        if len(value) > 60_000:
            long_scans += 1
        return original_sensitive_kinds(value)

    monkeypatch.setattr(redaction_module, "_sensitive_kinds", count_long_scans)

    kinds = redaction_module._png_exif_kinds(
        payload, redaction_module._PngBudget()
    )

    assert "local_path" in kinds
    assert long_scans == 1


def test_png_exif_charges_distinct_referenced_values_before_second_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = (b"a" * 8_192) + b"\0"
    second = (b"b" * 8_192) + b"\0"
    payload = _little_endian_tiff_ascii_references((first, second))
    monkeypatch.setattr(redaction_module, "_PNG_MAX_WORK_UNITS", 20, raising=False)
    original_sensitive_kinds = redaction_module._sensitive_kinds
    long_scans = 0

    def count_long_scans(value: str) -> set[str]:
        nonlocal long_scans
        if len(value) > 8_000:
            long_scans += 1
        return original_sensitive_kinds(value)

    monkeypatch.setattr(redaction_module, "_sensitive_kinds", count_long_scans)

    with pytest.raises(redaction_module._PngParseError):
        redaction_module._png_exif_kinds(
            payload, redaction_module._PngBudget()
        )

    assert long_scans == 1


@pytest.mark.parametrize("invalid_tag", (0x9C9C, 0x9286, 0x014A))
def test_png_exif_dedup_does_not_skip_tag_specific_type_validation(
    invalid_tag: int,
) -> None:
    value = b"public metadata\0"
    value_offset = 8 + 2 + 2 * 12 + 4
    payload = b"".join(
        (
            b"II\x2a\x00",
            struct.pack("<I", 8),
            struct.pack("<H", 2),
            struct.pack("<HHII", 0x010E, 2, len(value), value_offset),
            struct.pack("<HHII", invalid_tag, 2, len(value), value_offset),
            struct.pack("<I", 0),
            value,
        )
    )

    with pytest.raises(redaction_module._PngParseError):
        redaction_module._png_exif_kinds(
            payload, redaction_module._PngBudget()
        )


@pytest.mark.parametrize(
    "exif",
    (
        b"ZZ\x2a\x00\x08\x00\x00\x00",
        b"II\x2b\x00\x08\x00\x00\x00",
        b"II\x2a\x00\xff\xff\xff\x7f",
    ),
)
def test_publication_audit_fails_closed_on_unsupported_or_malformed_png_exif(
    exif: bytes,
) -> None:
    findings = redaction_module.audit_publication_bytes(
        "unsupported-exif.png",
        _png_fixture(before_idat=(_png_chunk(b"eXIf", exif),)),
    )

    assert any(finding.kind == "secret_pattern" for finding in findings)


def test_publication_audit_precharges_exif_ifd_pointer_array_before_unpack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer_values_offset = 8 + 2 + 12 + 4
    tiff = b"".join(
        (
            b"II\x2a\x00",
            struct.pack("<I", 8),
            struct.pack("<H", 1),
            struct.pack("<HHII", 0x014A, 4, 3, pointer_values_offset),
            struct.pack("<I", 0),
            struct.pack("<III", 0, 0, 0),
        )
    )
    monkeypatch.setattr(redaction_module, "_PNG_MAX_WORK_UNITS", 9, raising=False)
    original_unpack = redaction_module.struct.unpack

    def monitored_unpack(format_string: str, value: bytes) -> tuple[object, ...]:
        if format_string == "<3I":
            raise AssertionError("pointer-array rejection must precede unpack")
        return original_unpack(format_string, value)

    monkeypatch.setattr(redaction_module.struct, "unpack", monitored_unpack)

    findings = redaction_module.audit_publication_bytes(
        "exif-pointer-work-cap.png",
        _png_fixture(before_idat=(_png_chunk(b"eXIf", tiff),)),
    )

    assert any(finding.kind == "secret_pattern" for finding in findings)


@pytest.mark.parametrize(
    ("chunk_type", "payload"),
    (
        (b"cHRM", b"\0" * 32),
        (b"cICP", b"\0" * 4),
        (b"cLLI", b"\0" * 8),
        (b"gAMA", struct.pack(">I", 45_455)),
        (b"iCCP", b"Profile\0\0" + zlib.compress(b"public profile")),
        (b"mDCV", b"\0" * 24),
        (b"sBIT", b"\x08\x08\x08"),
        (b"sRGB", b"\0"),
    ),
)
def test_publication_audit_rejects_pre_palette_chunks_after_plte(
    chunk_type: bytes,
    payload: bytes,
) -> None:
    findings = redaction_module.audit_publication_bytes(
        "bad-pre-palette-order.png",
        _png_fixture(
            before_idat=(
                _png_chunk(b"PLTE", b"\0\0\0"),
                _png_chunk(chunk_type, payload),
            ),
            idat=zlib.compress(b"\0\0\0\0"),
            color_type=2,
        ),
    )

    assert any(finding.kind == "secret_pattern" for finding in findings)


def test_publication_audit_rejects_clli_after_idat() -> None:
    findings = redaction_module.audit_publication_bytes(
        "late-clli.png",
        _png_fixture(after_idat=(_png_chunk(b"cLLI", b"\0" * 8),)),
    )

    assert any(finding.kind == "secret_pattern" for finding in findings)


@pytest.mark.parametrize(
    ("chunk_type", "payload"),
    (
        (b"bKGD", b"\0" * 6),
        (b"tRNS", b"\0" * 6),
    ),
)
def test_publication_audit_rejects_optional_plte_after_dependent_chunk(
    chunk_type: bytes,
    payload: bytes,
) -> None:
    findings = redaction_module.audit_publication_bytes(
        "late-palette.png",
        _png_fixture(
            before_idat=(
                _png_chunk(chunk_type, payload),
                _png_chunk(b"PLTE", b"\0\0\0"),
            ),
            idat=zlib.compress(b"\0\0\0\0"),
            color_type=2,
        ),
    )

    assert any(finding.kind == "secret_pattern" for finding in findings)


def test_gif_parser_rejects_huge_logical_screen_from_small_header_quickly() -> None:
    content = _gif_test_fixture(logical_width=0xFFFF, logical_height=0xFFFF)

    started = time.perf_counter()
    assert not redaction_module.validate_media_container_bytes(".gif", content)
    assert time.perf_counter() - started < 2.0


def test_gif_parser_rejects_huge_image_rectangle_from_small_header_quickly() -> None:
    content = _gif_test_fixture(logical_width=4_097, logical_height=2_048)

    started = time.perf_counter()
    assert not redaction_module.validate_media_container_bytes(".gif", content)
    assert time.perf_counter() - started < 2.0


def test_gif_parser_rejects_frame_count_cap_from_small_blocks() -> None:
    content = _gif_test_fixture(image_count=513)

    assert not redaction_module.validate_media_container_bytes(".gif", content)


def test_gif_parser_precharges_cumulative_pixels_before_lzw_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _gif_test_fixture(
        logical_width=1_024,
        logical_height=1_024,
        image_count=129,
    )

    def lzw_tripwire(*args: object, **kwargs: object) -> int:
        raise AssertionError("cumulative pixel rejection must precede LZW work")

    monkeypatch.setattr(
        redaction_module,
        "_gif_lzw_decoded_pixels",
        lzw_tripwire,
        raising=False,
    )

    assert not redaction_module.validate_media_container_bytes(".gif", content)


@pytest.mark.parametrize(
    ("limit_name", "sub_blocks"),
    (
        ("_GIF_MAX_COMPRESSED_BYTES", (b"\x44\x01",)),
        ("_GIF_MAX_SUB_BLOCKS", (b"\x44", b"\x01")),
        ("_GIF_MAX_DECODE_WORK", (b"\x44\x01",)),
    ),
)
def test_gif_parser_precharges_compressed_subblock_and_decode_work_caps(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    sub_blocks: tuple[bytes, ...],
) -> None:
    monkeypatch.setattr(redaction_module, limit_name, 1, raising=False)
    content = _gif_test_fixture(image_sub_blocks=sub_blocks)

    def lzw_tripwire(*args: object, **kwargs: object) -> int:
        raise AssertionError("GIF resource rejection must precede LZW work")

    monkeypatch.setattr(
        redaction_module, "_gif_lzw_decoded_pixels", lzw_tripwire
    )

    assert not redaction_module.validate_media_container_bytes(".gif", content)


@pytest.mark.parametrize(
    "content",
    (
        _gif_test_fixture(palette_bits=8, lzw_minimum_code_size=2),
        _gif_test_fixture(
            logical_width=2,
            logical_height=1,
            image_width=2,
            image_height=1,
        ),
        _gif_test_fixture(image_sub_blocks=(b"\x04",)),
        _gif_test_fixture(image_sub_blocks=(b"\x44\x01\x00",)),
    ),
    ids=("palette-code-size", "decoded-pixel-count", "missing-eoi", "after-eoi"),
)
def test_gif_parser_rejects_invalid_lzw_and_subblock_relationships(
    content: bytes,
) -> None:
    assert not redaction_module.validate_media_container_bytes(".gif", content)


def test_gif_parser_accepts_lzw_codes_split_across_subblocks() -> None:
    content = _gif_test_fixture(image_sub_blocks=(b"\x44", b"\x01"))

    assert redaction_module.validate_media_container_bytes(".gif", content)


@pytest.mark.parametrize(
    ("palette_bits", "local_palette_bits"),
    ((1, None), (2, 1)),
    ids=("global", "local-overrides-global"),
)
def test_gif_parser_rejects_root_literal_outside_active_palette(
    palette_bits: int, local_palette_bits: int | None
) -> None:
    # With a two-entry active palette, root literal 3 is not a color index.
    content = _gif_test_fixture(
        palette_bits=palette_bits,
        local_palette_bits=local_palette_bits,
        image_sub_blocks=(b"\x5c\x01",),  # clear(4), literal(3), EOI(5)
    )

    assert not redaction_module.validate_media_container_bytes(".gif", content)


def test_gif_parser_rejects_out_of_palette_root_literal_after_clear_reset() -> None:
    content = _gif_test_fixture(
        image_width=2,
        logical_width=2,
        image_sub_blocks=(
            b"\x04\x57",  # clear, literal(0), clear, literal(3), EOI
        ),
    )

    assert not redaction_module.validate_media_container_bytes(".gif", content)


@pytest.mark.parametrize(
    ("local_palette_bits", "image_sub_blocks"),
    ((None, (b"\x4c\x01",)), (2, (b"\x5c\x01",))),
    ids=("global-root", "local-overrides-global-root"),
)
def test_gif_parser_accepts_root_literals_inside_active_palette(
    local_palette_bits: int | None,
    image_sub_blocks: tuple[bytes, ...],
) -> None:
    content = _gif_test_fixture(
        local_palette_bits=local_palette_bits,
        image_sub_blocks=image_sub_blocks,
    )

    assert redaction_module.validate_media_container_bytes(".gif", content)


def test_gif_parser_keeps_valid_dictionary_codes_legal_across_clear_reset() -> None:
    content = _gif_test_fixture(
        logical_width=5,
        image_width=5,
        image_sub_blocks=(
            b"\x44\x4c\x29",  # clear, 0, 1, dictionary(6), clear, 1, EOI
        ),
    )

    assert redaction_module.validate_media_container_bytes(".gif", content)


@pytest.mark.parametrize(
    ("name", "expected_frames"),
    (
        ("exp001-b0-side-by-side.gif", 78),
        ("exp001-b1-side-by-side.gif", 96),
    ),
)
def test_gif_parser_accepts_frozen_960x720_real_preview_controls(
    name: str,
    expected_frames: int,
) -> None:
    media = Path(__file__).parents[1] / "docs" / "media" / name
    assert redaction_module.validate_media_container_bytes(".gif", media.read_bytes())
    executable = shutil.which("ffprobe")
    if executable is None:
        pytest.skip("ffprobe is required for real GIF compatibility controls")
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            str(media),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    stream = json.loads(completed.stdout)["streams"][0]
    assert (stream["width"], stream["height"], int(stream["nb_read_frames"])) == (
        960,
        720,
        expected_frames,
    )


def test_media_metadata_probe_reads_real_gif_from_stable_stream(
    tmp_path: Path,
) -> None:
    source = Path(__file__).parents[1] / "docs" / "media" / "exp001-b0-side-by-side.gif"
    stable_copy = tmp_path / "stable.gif"
    shutil.copyfile(source, stable_copy)
    with stable_copy.open("rb") as stream:
        assert redaction_module._media_metadata_status(
            tmp_path / "must-not-be-opened.gif", stable_stream=stream
        ) == (True, False)


def test_publication_audit_does_not_invent_paths_from_html_or_gif_compression(
    tmp_path: Path,
) -> None:
    """Closing tags and compressed bytes are not local paths; media metadata is audited."""

    (tmp_path / "index.html").write_text(
        "<!doctype html><html><body><p>public diagnostic</p></body></html>",
        encoding="utf-8",
    )
    source = Path(__file__).parents[1] / "docs" / "media" / "exp001-b0-side-by-side.gif"
    shutil.copyfile(source, tmp_path / "preview.gif")

    assert audit_publication_tree(tmp_path) == ()


def test_publication_audit_does_not_invent_paths_from_verified_npz_compression(
    tmp_path: Path,
) -> None:
    np.savez(
        tmp_path / "simulation.npz",
        qpos=np.linspace(-1.0, 1.0, 100_000, dtype=np.float64),
    )

    assert audit_publication_tree(tmp_path) == ()


def _forbid_npz_decompression(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_read(*args: object, **kwargs: object) -> bytes:
        pytest.fail("resource-bound NPZ was decompressed")

    monkeypatch.setattr(zipfile.ZipFile, "read", fail_read)


def test_publication_audit_rejects_excessive_npz_entry_count_before_decompression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "too-many-entries.npz"
    np.savez_compressed(
        archive,
        **{f"value_{index:03d}": np.asarray([index]) for index in range(129)},
    )
    _forbid_npz_decompression(monkeypatch)

    findings = audit_publication_tree(tmp_path)

    assert any(finding.path == archive.name for finding in findings)


def test_publication_audit_rejects_excessive_npz_aggregate_before_decompression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "aggregate-too-large.npz"
    size = 17 * 1024 * 1024
    np.savez(
        archive,
        first=np.arange(size, dtype=np.uint8),
        second=np.arange(size, dtype=np.uint8),
    )
    _forbid_npz_decompression(monkeypatch)

    findings = audit_publication_tree(tmp_path)

    assert any(finding.path == archive.name for finding in findings)


def test_publication_audit_rejects_extreme_npz_ratio_before_decompression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "ratio-too-large.npz"
    np.savez_compressed(
        archive,
        zeros=np.zeros(4 * 1024 * 1024, dtype=np.uint8),
    )
    _forbid_npz_decompression(monkeypatch)

    findings = audit_publication_tree(tmp_path)

    assert any(finding.path == archive.name for finding in findings)


@pytest.mark.parametrize("suffix", (".gif", ".mp4"))
def test_publication_audit_detects_literal_path_appended_to_valid_media(
    tmp_path: Path, suffix: str
) -> None:
    private_path = "C:" + "\\Users\\Alice\\private-source.mp4"
    media = tmp_path / f"preview{suffix}"
    if suffix == ".gif":
        shutil.copyfile(
            Path(__file__).parents[1]
            / "docs"
            / "media"
            / "exp001-b0-side-by-side.gif",
            media,
        )
    else:
        _write_tagged_mp4(media, "public diagnostic")
    with media.open("ab") as stream:
        stream.write(b"\n" + private_path.encode("ascii") + b"\n")

    findings = audit_publication_tree(tmp_path)

    assert any(finding.path == media.name and finding.kind == "local_path" for finding in findings)


def test_publication_audit_detects_literal_path_in_valid_mp4_metadata(
    tmp_path: Path,
) -> None:
    private_path = "C:" + "\\Users\\Alice\\private-source.mp4"
    media = tmp_path / "tagged.mp4"
    _write_tagged_mp4(media, private_path)

    findings = audit_publication_tree(tmp_path)

    assert any(
        finding.path == media.name and finding.kind == "local_path"
        for finding in findings
    )


def test_publication_audit_detects_literal_path_in_valid_mov_text_sample(
    tmp_path: Path,
) -> None:
    private_path = "C:" + "\\Users\\Alice\\private-source.mp4"
    media = tmp_path / "text-sample.mp4"
    _write_mov_text_mp4(media, private_path)

    findings = audit_publication_tree(tmp_path)

    assert any(
        finding.path == media.name and finding.kind == "local_path"
        for finding in findings
    )


@pytest.mark.parametrize(
    ("suffix", "sample_entry", "include_video"),
    (
        (".mov", "text", False),
        (".mov", "text", True),
        (".mp4", "tx3g", False),
        (".mp4", "tx3g", True),
    ),
)
def test_publication_audit_detects_path_in_real_text_track_sample_tables(
    tmp_path: Path,
    suffix: str,
    sample_entry: str,
    include_video: bool,
) -> None:
    private_path = "C:" + "\\Users\\Alice\\private-source.mp4"
    media = tmp_path / f"text-track{suffix}"
    _write_real_text_container(
        media,
        private_path,
        include_video=include_video,
        sample_entry=sample_entry,
    )

    findings = audit_publication_tree(tmp_path)

    assert any(
        finding.path == media.name and finding.kind == "local_path"
        for finding in findings
    )


def test_publication_audit_ignores_path_shaped_compressed_video_sample(
    tmp_path: Path,
) -> None:
    private_path = ("C:" + "\\Users\\Alice\\private-source.mp4").encode("ascii")
    media = tmp_path / "mixed-public-text.mp4"
    _write_real_text_container(
        media,
        "public subtitle",
        include_video=True,
        sample_entry="tx3g",
    )
    content = bytearray(media.read_bytes())
    quantization_table = content.find(b"\xff\xdb\x00C\x00")
    assert quantization_table >= 0
    payload_start = quantization_table + 5
    content[payload_start : payload_start + len(private_path)] = private_path
    media.write_bytes(content)
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(media), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert decoded.returncode == 0, decoded.stderr

    findings = audit_publication_tree(tmp_path)

    assert findings == ()


@pytest.mark.parametrize("corruption", ("outside-mdat-offset", "fragment-layout"))
def test_publication_audit_fails_closed_on_ambiguous_text_sample_layout(
    tmp_path: Path, corruption: str
) -> None:
    private_path = "C:" + "\\Users\\Alice\\private-source.mp4"
    media = tmp_path / "ambiguous-text.mov"
    _write_real_text_container(
        media,
        private_path,
        include_video=False,
        sample_entry="text",
    )
    content = bytearray(media.read_bytes())
    if corruption == "outside-mdat-offset":
        stco_type = content.find(b"stco")
        assert stco_type >= 4
        entry_count = struct.unpack_from(">I", content, stco_type + 8)[0]
        assert entry_count > 0
        struct.pack_into(">I", content, stco_type + 12, len(content) + 64)
    else:
        content.extend(struct.pack(">I4s", 8, b"moof"))
    media.write_bytes(content)

    findings = audit_publication_tree(tmp_path)

    assert any(
        finding.path == media.name and finding.kind == "secret_pattern"
        for finding in findings
    )


@pytest.mark.parametrize(
    ("box_type", "multiplicity"),
    (
        (b"mdia", 0),
        (b"mdia", 2),
        (b"minf", 0),
        (b"minf", 2),
        (b"hdlr", 0),
        (b"hdlr", 2),
        (b"stbl", 0),
        (b"stbl", 2),
        (b"stsd", 0),
        (b"stsd", 2),
    ),
)
def test_publication_audit_rejects_missing_or_duplicate_required_iso_box(
    box_type: bytes, multiplicity: int
) -> None:
    private_path = ("C:" + "\\Users\\Alice\\private-source.mp4").encode("ascii")
    content = _iso_test_fixture(
        hierarchy_counts={box_type: multiplicity},
        sample_size=len(private_path),
        sample_payload=private_path,
    )

    findings = redaction_module.audit_publication_bytes("ambiguous.mp4", content)

    assert any(finding.kind == "secret_pattern" for finding in findings)


@pytest.mark.parametrize(
    "descriptions",
    (
        (),
        (b"zzzz",),
        (b"tx3g", b"tx3g"),
        (b"tx3g", b"zzzz"),
    ),
)
def test_publication_audit_rejects_ambiguous_text_handler_description(
    descriptions: tuple[bytes, ...]
) -> None:
    private_path = ("C:" + "\\Users\\Alice\\private-source.mp4").encode("ascii")
    content = _iso_test_fixture(
        descriptions=descriptions,
        sample_size=len(private_path),
        sample_payload=private_path,
    )

    findings = redaction_module.audit_publication_bytes("ambiguous.mp4", content)

    assert any(finding.kind == "secret_pattern" for finding in findings)


@pytest.mark.parametrize(
    ("suffix", "stream_kind"),
    ((".mp4", "video"), (".mp4", "audio"), (".mov", "mixed")),
)
def test_iso_parser_accepts_real_classic_video_audio_and_mov_controls(
    tmp_path: Path, suffix: str, stream_kind: str
) -> None:
    media = tmp_path / f"classic-{stream_kind}{suffix}"
    _write_real_classic_container(media, stream_kind)

    assert redaction_module.validate_media_container_bytes(
        suffix, media.read_bytes()
    )


@pytest.mark.parametrize("before_tracks", (True, False), ids=("before", "after"))
def test_iso_parser_rejects_any_mvex_child_inside_moov(
    before_tracks: bool,
) -> None:
    content = _iso_with_moov_child(
        _iso_test_fixture(handler_type=b"vide", descriptions=(b"mp4v",)),
        _iso_test_box(b"mvex", b""),
        before_tracks=before_tracks,
    )

    assert not redaction_module.validate_media_container_bytes(".mp4", content)


@pytest.mark.parametrize(
    ("handler_type", "description"),
    (
        (b"vide", b"mp4v"),
        (b"soun", b"mp4a"),
        (b"hint", b"rtp "),
        (b"auxv", b"auxv"),
        (b"pict", b"jpeg"),
    ),
)
def test_iso_parser_requires_sample_table_for_every_classic_non_text_handler(
    handler_type: bytes, description: bytes
) -> None:
    content = _iso_test_fixture(
        handler_type=handler_type,
        descriptions=(description,),
        hierarchy_counts={b"stsz": 0},
    )

    assert not redaction_module.validate_media_container_bytes(".mp4", content)


@pytest.mark.parametrize(
    ("sample_table", "offset_box_type", "box_type"),
    (
        ("stsz-fixed", b"stco", b"stsz"),
        ("stz2", b"stco", b"stz2"),
        ("stsz-fixed", b"stco", b"stsc"),
        ("stsz-fixed", b"stco", b"stco"),
        ("stsz-fixed", b"co64", b"co64"),
    ),
)
@pytest.mark.parametrize("multiplicity", (0, 2))
def test_iso_parser_requires_singleton_classic_non_text_sample_tables(
    sample_table: str,
    offset_box_type: bytes,
    box_type: bytes,
    multiplicity: int,
) -> None:
    content = _iso_test_fixture(
        handler_type=b"vide",
        descriptions=(b"mp4v",),
        sample_table=sample_table,
        offset_box_type=offset_box_type,
        hierarchy_counts={box_type: multiplicity},
    )

    assert not redaction_module.validate_media_container_bytes(".mp4", content)


@pytest.mark.parametrize("description_index", (0, 2))
def test_iso_parser_rejects_non_text_stsc_description_index_outside_stsd(
    description_index: int,
) -> None:
    content = _iso_test_fixture(
        handler_type=b"vide",
        descriptions=(b"mp4v",),
        stsc_description_index=description_index,
    )

    assert not redaction_module.validate_media_container_bytes(".mp4", content)


def test_iso_parser_accepts_non_text_stsc_reference_to_second_description() -> None:
    content = _iso_test_fixture(
        handler_type=b"vide",
        descriptions=(b"mp4v", b"avc1"),
        stsc_description_index=2,
    )

    assert redaction_module.validate_media_container_bytes(".mp4", content)


def test_iso_parser_rejects_non_text_chunk_mapping_without_exact_sample_coverage() -> None:
    content = _iso_test_fixture(
        handler_type=b"soun",
        descriptions=(b"mp4a",),
        sample_count=2,
        stsc_samples_per_chunk=1,
    )

    assert not redaction_module.validate_media_container_bytes(".mp4", content)


def test_iso_parser_rejects_non_text_sample_range_outside_local_mdat() -> None:
    content = bytearray(
        _iso_test_fixture(handler_type=b"vide", descriptions=(b"mp4v",))
    )
    stco_type = content.find(b"stco")
    assert stco_type >= 4
    struct.pack_into(">I", content, stco_type + 12, len(content) + 64)

    assert not redaction_module.validate_media_container_bytes(
        ".mp4", bytes(content)
    )


@pytest.mark.parametrize(
    "fixture_kwargs",
    (
        {"sample_count": 1_025, "track_count": 16},
        {
            "sample_table": "stsz-variable",
            "sample_count": 512,
            "track_count": 32,
        },
        {"sample_count": 129, "chunk_count": 129, "track_count": 32},
        {
            "sample_table": "stsz-variable",
            "sample_count": 800,
            "track_count": 16,
        },
    ),
    ids=("samples", "table-entries", "chunks", "work"),
)
def test_iso_parser_applies_file_wide_caps_across_non_text_tracks(
    fixture_kwargs: dict[str, object],
) -> None:
    content = _iso_test_fixture(
        handler_type=b"vide", descriptions=(b"mp4v",), **fixture_kwargs
    )

    _assert_iso_parser_rejects_quickly(content)


def test_iso_parser_rejects_huge_fixed_stsz_count_before_legacy_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def allocation_tripwire(*args: object, **kwargs: object) -> object:
        raise AssertionError("legacy fixed-size stsz materializer was reached")

    monkeypatch.setattr(
        redaction_module, "_iso_sample_sizes", allocation_tripwire, raising=False
    )
    content = _iso_test_fixture(
        sample_count=0xFFFFFFFF,
        sample_payload=b"x",
    )

    _assert_iso_parser_rejects_quickly(content)


@pytest.mark.parametrize("sample_table", ("stsz-variable", "stz2"))
def test_iso_parser_rejects_huge_compact_sample_count_before_iteration(
    sample_table: str,
) -> None:
    content = _iso_test_fixture(
        sample_table=sample_table,
        sample_count=0xFFFFFFFF,
        size_entries_count=0,
        sample_payload=b"x",
    )

    _assert_iso_parser_rejects_quickly(content)


@pytest.mark.parametrize("sample_table", ("stsz-variable", "stz2"))
def test_iso_parser_rejects_materializable_sample_table_above_limit(
    sample_table: str,
) -> None:
    content = _iso_test_fixture(
        sample_table=sample_table,
        sample_count=16_385,
    )

    _assert_iso_parser_rejects_quickly(content)


def test_iso_parser_rejects_excessive_stsc_entries() -> None:
    content = _iso_test_fixture(
        sample_count=4_097,
        chunk_count=4_097,
        stsc_entry_count=4_097,
    )

    _assert_iso_parser_rejects_quickly(content)


@pytest.mark.parametrize("offset_box_type", (b"stco", b"co64"))
def test_iso_parser_rejects_excessive_chunk_offset_entries(
    offset_box_type: bytes,
) -> None:
    content = _iso_test_fixture(
        sample_count=4_097,
        chunk_count=4_097,
        offset_box_type=offset_box_type,
    )

    _assert_iso_parser_rejects_quickly(content)


def test_iso_parser_rejects_excessive_total_boxes() -> None:
    content = _iso_test_fixture(extra_free_boxes=1_025)

    _assert_iso_parser_rejects_quickly(content)


def test_iso_parser_rejects_excessive_tracks() -> None:
    content = _iso_test_fixture(
        handler_type=b"vide",
        descriptions=(b"mp4v",),
        track_count=33,
    )

    _assert_iso_parser_rejects_quickly(content)


def test_iso_parser_rejects_excessive_text_ranges() -> None:
    content = _iso_test_fixture(sample_count=8_193)

    _assert_iso_parser_rejects_quickly(content)


def test_iso_parser_rejects_excessive_total_table_entries() -> None:
    content = _iso_test_fixture(
        sample_table="stsz-variable",
        sample_count=8_192,
        chunk_count=4_096,
        stsc_entry_count=4_096,
    )

    _assert_iso_parser_rejects_quickly(content)


def test_iso_parser_rejects_excessive_cumulative_work() -> None:
    content = _iso_test_fixture(
        sample_table="stsz-variable",
        sample_count=8_000,
        chunk_count=4_000,
    )

    _assert_iso_parser_rejects_quickly(content)


@pytest.mark.parametrize("placement", ("archive-trailer", "unicode-array"))
def test_publication_audit_detects_literal_path_in_valid_npz_text(
    tmp_path: Path, placement: str
) -> None:
    private_path = "C:" + "\\Users\\Alice\\private-source.mp4"
    archive = tmp_path / "simulation.npz"
    if placement == "archive-trailer":
        np.savez(archive, qpos=np.arange(8, dtype=np.float64))
        with archive.open("ab") as stream:
            stream.write(b"\n" + private_path.encode("ascii") + b"\n")
    else:
        np.savez(archive, source=np.asarray([private_path], dtype="U64"))

    findings = audit_publication_tree(tmp_path)

    assert any(
        finding.path == archive.name and finding.kind == "local_path"
        for finding in findings
    )


@pytest.mark.parametrize(
    "placement",
    (
        "structured-unicode",
        "structured-bytes",
        "unicode-subarray",
        "nested-unicode",
    ),
)
def test_publication_audit_detects_paths_in_structured_npz_fields(
    tmp_path: Path, placement: str
) -> None:
    private_path = "C:" + "\\Users\\Alice\\private-source.mp4"
    if placement == "structured-unicode":
        values = np.zeros(1, dtype=[("private", "U96"), ("count", "<i4")])
        values["private"][0] = private_path
    elif placement == "structured-bytes":
        values = np.zeros(1, dtype=[("private", "S96"), ("count", "<i4")])
        values["private"][0] = private_path.encode("ascii")
    elif placement == "unicode-subarray":
        values = np.zeros(1, dtype=[("private", "U96", (2,))])
        values["private"][0, 1] = private_path
    else:
        values = np.zeros(1, dtype=[("outer", [("private", "U96")])])
        values["outer"]["private"][0] = private_path
    archive = tmp_path / "structured.npz"
    np.savez_compressed(archive, records=values)

    findings = audit_publication_tree(tmp_path)

    assert any(
        finding.path == archive.name and finding.kind == "local_path"
        for finding in findings
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows directory metadata caching")
def test_publication_audit_ignores_unreliable_directory_metadata(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "public.txt").write_text("public", encoding="utf-8")

    assert audit_publication_tree(tmp_path) == ()


def test_media_metadata_probe_reads_tagged_mp4_from_stable_stream(
    tmp_path: Path,
) -> None:
    metadata = "Authorization: Bearer " + "TEST_" + "STABLE_MEDIA_VALUE"
    stable_copy = tmp_path / "stable.mp4"
    _write_tagged_mp4(stable_copy, metadata)
    with stable_copy.open("rb") as stream:
        assert redaction_module._media_metadata_status(
            tmp_path / "must-not-be-opened.mp4", stable_stream=stream
        ) == (True, True)


def write_publication_audit_fixtures(root: Path) -> tuple[str, ...]:
    authorization = "TEST_" + "AUTHORIZATION_VALUE"
    signed_query = "TEST_" + "SIGNED_VALUE"
    provider = "gh" + "p_" + ("P" * 40)
    user = "fixture" + "-person"
    metadata = "Authorization: Bearer " + "TEST_" + "MEDIA_VALUE"
    (root / "diagnostic.txt").write_text(
        f"Authorization: Bearer {authorization}\n"
        f"https://example.invalid/a?signature={signed_query}\n"
        f"/home/{user}/workspace/private.mp4\n",
        encoding="utf-8",
    )
    (root / "bounded.bin").write_bytes(b"\x00\xff" + provider.encode("ascii") + b"\x00")
    _write_tagged_mp4(root / "tagged.mp4", metadata)
    return authorization, signed_query, provider, user, metadata


def test_publication_audit_detects_text_binary_and_media_metadata(tmp_path: Path) -> None:
    secrets = write_publication_audit_fixtures(tmp_path)
    findings = audit_publication_tree(tmp_path)
    assert {finding.kind for finding in findings} == {
        "authorization",
        "credential_query",
        "local_path",
        "secret_pattern",
        "media_metadata",
    }
    rendered = repr(findings)
    assert str(tmp_path.resolve()) not in rendered
    for secret in secrets:
        assert secret not in rendered


def test_publication_audit_samples_bounded_binary_windows(tmp_path: Path) -> None:
    provider = "gh" + "p_" + ("Z" * 40)
    (tmp_path / "large.bin").write_bytes(
        provider.encode("ascii")
        + b"\x00"
        + b"x" * (3 * 1024 * 1024)
        + b"\x00"
        + provider.encode("ascii")
    )
    findings = audit_publication_tree(tmp_path)
    assert [finding.kind for finding in findings] == ["secret_pattern"]


def test_publication_audit_streams_complete_utf8_text_including_middle(
    tmp_path: Path,
) -> None:
    provider = "gh" + "p_" + ("M" * 40)
    path = tmp_path / "large.txt"
    path.write_text(
        "x" * (1536 * 1024) + "\n" + provider + "\n" + "y" * (1536 * 1024),
        encoding="utf-8",
    )
    findings = audit_publication_tree(tmp_path)
    assert [(finding.path, finding.kind) for finding in findings] == [
        ("large.txt", "secret_pattern")
    ]


def test_publication_audit_reports_symlinks_without_following_them(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    link = tmp_path / "linked.txt"
    if os.name == "nt":
        target.mkdir()
        (target / "must-not-read.txt").write_text("safe", encoding="utf-8")
        _make_directory_link(link, target)
    else:
        target.write_text("safe", encoding="utf-8")
        link.symlink_to(target)
    try:
        findings = audit_publication_tree(tmp_path)
    finally:
        _remove_directory_link(link)
    assert any(
        finding.path == "linked.txt"
        and finding.kind == "secret_pattern"
        and "symlink" in finding.detail
        for finding in findings
    )


def test_publication_audit_rejects_root_link_without_following_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "root-target"
    target.mkdir()
    (target / "must-not-scan.txt").write_text("safe", encoding="utf-8")
    root = tmp_path / "root-link"
    if os.name == "nt":
        _make_directory_link(root, target)
    else:
        root.symlink_to(target, target_is_directory=True)
    try:
        findings = audit_publication_tree(root)
    finally:
        _remove_directory_link(root)
    assert [(item.path, item.kind) for item in findings] == [
        (".", "secret_pattern")
    ]
    assert "link" in findings[0].detail or "reparse" in findings[0].detail


def test_publication_audit_rejects_link_ancestor_of_requested_subdirectory(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ancestor-target"
    requested = target / "requested"
    requested.mkdir(parents=True)
    (requested / "must-not-scan.txt").write_text("safe", encoding="utf-8")
    ancestor = tmp_path / "ancestor-link"
    if os.name == "nt":
        _make_directory_link(ancestor, target)
    else:
        ancestor.symlink_to(target, target_is_directory=True)
    try:
        findings = audit_publication_tree(ancestor / "requested")
    finally:
        _remove_directory_link(ancestor)
    assert [(item.path, item.kind) for item in findings] == [
        (".", "secret_pattern")
    ]
    assert "ancestor" in findings[0].detail


def test_publication_audit_rejects_lexical_link_parent_traversal_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted = tmp_path / "trusted"
    collapsed = trusted / "publication"
    collapsed.mkdir(parents=True)
    (collapsed / "must-not-read.txt").write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside"
    linked_target = outside / "nested"
    linked_target.mkdir(parents=True)
    (outside / "publication").mkdir()
    link = trusted / "ancestor-link"
    _make_directory_link(link, linked_target)
    reads = 0
    original_reader = redaction_module._stable_reader_kinds

    def record_read(*args: object, **kwargs: object) -> set[str]:
        nonlocal reads
        reads += 1
        return original_reader(*args, **kwargs)

    monkeypatch.setattr(redaction_module, "_stable_reader_kinds", record_read)
    try:
        findings = audit_publication_tree(link / ".." / "publication")
    finally:
        _remove_directory_link(link)
    assert reads == 0
    assert findings == (
        redaction_module.PublicationFinding(
            ".", "secret_pattern", "parent path traversal is not publishable"
        ),
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows ancestor ABA contract")
def test_publication_audit_rejects_windows_ancestor_junction_aba_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ancestor = tmp_path / "ancestor"
    root = ancestor / "publication"
    root.mkdir(parents=True)
    (root / "safe.txt").write_text("safe", encoding="utf-8")
    outside_ancestor = tmp_path / "outside-ancestor"
    outside_root = outside_ancestor / "publication"
    outside_root.mkdir(parents=True)
    provider = "gh" + "p_" + ("W" * 40)
    (outside_root / "outside-only.txt").write_text(provider, encoding="utf-8")
    backup = tmp_path / "ancestor-backup"
    original_open_root = getattr(redaction_module, "_win_open_root", None)
    triggered = False

    def swap_around_root_open(candidate: Path) -> tuple[object, tuple[int, ...]]:
        nonlocal triggered
        triggered = True
        ancestor.replace(backup)
        _make_directory_link(ancestor, outside_ancestor)
        try:
            assert original_open_root is not None
            return original_open_root(candidate)
        finally:
            _remove_directory_link(ancestor)
            backup.replace(ancestor)

    monkeypatch.setattr(
        redaction_module, "_win_open_root", swap_around_root_open, raising=False
    )
    findings = audit_publication_tree(root)
    assert triggered
    assert [(item.path, item.detail) for item in findings] == [
        (".", "ancestor symlink or reparse point is not publishable")
    ]
    assert not any(item.path == "outside-only.txt" for item in findings)
    assert provider not in repr(findings)


def _restore_race_fixture(candidate: Path, backup: Path) -> None:
    if getattr(os.path, "isjunction", lambda _: False)(candidate):
        candidate.rmdir()
    elif candidate.is_symlink() or candidate.is_file():
        candidate.unlink()
    elif candidate.is_dir():
        for child in candidate.iterdir():
            child.unlink()
        candidate.rmdir()
    backup.replace(candidate)


def _swap_before_stable_child_open(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    swap: object,
) -> None:
    if os.name == "nt":
        original_open_child = redaction_module._win_open_child

        def controlled_open_child(
            directory_handle: object, child_name: str, *, directory: bool
        ) -> tuple[object, tuple[int, ...]]:
            if child_name == name:
                swap()  # type: ignore[operator]
            return original_open_child(
                directory_handle, child_name, directory=directory
            )

        monkeypatch.setattr(
            redaction_module, "_win_open_child", controlled_open_child
        )
        return
    original_open = os.open

    def controlled_open(
        path: object, flags: int, *args: object, **kwargs: object
    ) -> int:
        if path == name and kwargs.get("dir_fd") is not None:
            swap()  # type: ignore[operator]
        return original_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(redaction_module.os, "open", controlled_open)


def test_publication_audit_rejects_file_swap_after_lstat_without_reading_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.txt"
    candidate.write_text("safe", encoding="utf-8")
    external = tmp_path / "external.txt"
    provider = "gh" + "p_" + ("T" * 40)
    external.write_text(provider, encoding="utf-8")
    backup = tmp_path / "candidate-backup.txt"
    triggered = False
    swapped = False

    def swap() -> None:
        nonlocal triggered, swapped
        if triggered:
            return
        triggered = True
        try:
            candidate.replace(backup)
        except OSError:
            return
        swapped = True
        try:
            candidate.symlink_to(external)
        except (NotImplementedError, OSError):
            candidate.write_text(provider, encoding="utf-8")

    _swap_before_stable_child_open(monkeypatch, candidate.name, swap)
    try:
        findings = audit_publication_tree(tmp_path)
    finally:
        if swapped:
            _restore_race_fixture(candidate, backup)
    assert triggered
    assert not swapped or any(item.path == "candidate.txt" for item in findings)
    assert provider not in repr(findings)


def test_publication_audit_rejects_directory_swap_after_lstat_without_descent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate-directory"
    candidate.mkdir()
    (candidate / "safe.txt").write_text("safe", encoding="utf-8")
    external = tmp_path / "external-directory"
    external.mkdir()
    provider = "gh" + "p_" + ("U" * 40)
    (external / "private.txt").write_text(provider, encoding="utf-8")
    backup = tmp_path / "candidate-directory-backup"
    triggered = False
    swapped = False

    def swap() -> None:
        nonlocal triggered, swapped
        if triggered:
            return
        triggered = True
        try:
            candidate.replace(backup)
        except OSError:
            return
        swapped = True
        if os.name == "nt":
            _make_directory_link(candidate, external)
        else:
            candidate.symlink_to(external, target_is_directory=True)

    _swap_before_stable_child_open(monkeypatch, candidate.name, swap)
    try:
        findings = audit_publication_tree(tmp_path)
    finally:
        if swapped:
            _restore_race_fixture(candidate, backup)
    assert triggered
    assert not swapped or any(
        item.path == "candidate-directory" for item in findings
    )
    assert not any(item.path.startswith("candidate-directory/") for item in findings)
    assert provider not in repr(findings)


def test_publication_audit_skips_directory_reparse_children(
    tmp_path: Path,
) -> None:
    target = tmp_path / "directory-target"
    target.mkdir()
    provider = "gh" + "p_" + ("J" * 40)
    (target / "must-not-scan.txt").write_text(provider, encoding="utf-8")
    linked = tmp_path / "linked-directory"
    if os.name == "nt":
        _make_directory_link(linked, target)
    else:
        linked.symlink_to(target, target_is_directory=True)
    try:
        findings = audit_publication_tree(tmp_path)
    finally:
        _remove_directory_link(linked)
    assert any(item.path == "linked-directory" for item in findings)
    assert not any(item.path.startswith("linked-directory/") for item in findings)


@pytest.mark.parametrize(
    "probe_result",
    [
        None,
        SimpleNamespace(returncode=1, stdout="", stderr="probe failed"),
        SimpleNamespace(returncode=0, stdout="not-json", stderr=""),
        SimpleNamespace(
            returncode=0,
            stdout='{"streams":[{"tags":[]}],"format":{}}',
            stderr="",
        ),
    ],
)
def test_publication_audit_fails_closed_when_media_metadata_cannot_be_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe_result: object,
) -> None:
    (tmp_path / "fixture.mp4").write_bytes(b"synthetic-media")
    if probe_result is None:
        monkeypatch.setattr("webvideo_to_data.redaction.shutil.which", lambda _: None)
    else:
        monkeypatch.setattr(
            "webvideo_to_data.redaction.shutil.which", lambda _: "fixture-ffprobe"
        )
        monkeypatch.setattr(
            "webvideo_to_data.redaction.subprocess.run",
            lambda *args, **kwargs: probe_result,
        )
    findings = audit_publication_tree(tmp_path)
    assert any(finding.kind == "media_metadata" for finding in findings)


def test_publication_audit_fails_closed_when_ffprobe_execution_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "fixture.mp4").write_bytes(b"synthetic-media")
    monkeypatch.setattr(
        "webvideo_to_data.redaction.shutil.which", lambda _: "fixture-ffprobe"
    )

    def unavailable(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("ffprobe disappeared")

    monkeypatch.setattr("webvideo_to_data.redaction.subprocess.run", unavailable)
    findings = audit_publication_tree(tmp_path)
    assert any(finding.kind == "media_metadata" for finding in findings)

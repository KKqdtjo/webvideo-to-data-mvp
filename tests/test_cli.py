from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import cv2
import imageio.v2 as imageio
import numpy as np
import pytest
import zlib

from helpers import write_complete_config
import webvideo_to_data.cli as cli_module
import webvideo_to_data.experiment as experiment_module
import webvideo_to_data.redaction as redaction_module
import webvideo_to_data.suite as suite_module
from webvideo_to_data.cli import main
from webvideo_to_data.experiment import run_experiment
from webvideo_to_data.suite import run_suite


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (("preflight", "--config", "missing.yaml"), 2),
        (("run", "--config", "bad.yaml", "--variant", "B0"), 2),
        (("verify", "--run", "missing-run"), 4),
        (("dashboard", "--run", "missing-run"), 4),
    ],
)
def test_cli_exit_codes(argv: tuple[str, ...], expected: int) -> None:
    assert main(argv) == expected


@pytest.mark.parametrize(
    "extra",
    [
        (),
        ("--variant", "B0"),
        ("--output-dir", "out"),
        ("--variant", "B0", "--output-dir", "out", "--all"),
        ("--variant", "B0", "--output-dir", "out", "--artifacts-root", "artifacts"),
        ("--variant", "B0", "--output-dir", "out", "--run-id", "run-1"),
    ],
)
def test_run_invalid_or_missing_options_exit_two_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: tuple[str, ...]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(("run", "--config", "missing.yaml", *extra)) == 2
    assert not (tmp_path / "out").exists()


def test_run_writes_exactly_the_requested_single_variant_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_complete_config(
        tmp_path, {"source.sha256": "0" * 64}, with_source=False
    )
    output = tmp_path / "requested-B2"
    assert (
        main(
            (
                "run",
                "--config",
                str(config),
                "--variant",
                "B2",
                "--output-dir",
                str(output),
                "--no-render",
                "--json",
            )
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert output.is_dir()
    assert not (output.parent / "B0").exists()
    assert payload["variant"] == "B2"


def test_legacy_script_maps_default_output_to_explicit_cli_form(tmp_path: Path) -> None:
    config = write_complete_config(
        tmp_path, {"source.sha256": "0" * 64}, with_source=False
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_exp001.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--variant",
            "B2",
            "--no-render",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["command"] == "run"
    assert payload["variant"] == "B2"
    assert (tmp_path / "artifacts" / "EXP-001" / "B2").is_dir()
    assert str(tmp_path.resolve()) not in completed.stdout + completed.stderr


def test_json_output_is_one_object_and_never_exposes_external_absolute_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_complete_config(
        tmp_path, {"source.sha256": "0" * 64}, with_source=False
    )
    output = tmp_path / "external-result"
    assert main(("run", "--config", str(config), "--variant", "B2", "--output-dir", str(output), "--no-render", "--json")) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["artifact_path"] == "<external-output>/external-result"
    assert captured.out.count("\n") == 1
    assert str(tmp_path.resolve()) not in captured.out + captured.err


def test_successful_relative_artifact_path_remains_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    config = write_complete_config(
        tmp_path, {"source.sha256": "0" * 64}, with_source=False
    )
    assert main(("run", "--config", str(config), "--variant", "B2", "--output-dir", "relative-out", "--no-render", "--json")) == 0
    assert json.loads(capsys.readouterr().out)["artifact_path"] == "relative-out"


def test_require_completed_preserves_b0_exit_five(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_complete_config(
        tmp_path, {"source.sha256": "0" * 64}, with_source=False
    )
    output = tmp_path / "manual-B0"
    code = main(("run", "--config", str(config), "--variant", "B0", "--output-dir", str(output), "--no-render", "--require-completed", "--json"))
    payload = json.loads(capsys.readouterr().out)
    assert code == 5
    assert payload["status"] != "completed"
    assert output.is_dir()


@pytest.mark.parametrize(
    "build_sensitive",
    [
        lambda root: "Authorization: Bearer " + "TEST_" + "CLI_SECRET" + " at " + str(root),
        lambda root: "https://example.invalid/x?signature=" + "TEST_" + "SIGNED_SECRET" + " at " + str(root),
        lambda root: "provider=" + "s" + "k-" + ("K" * 32) + " at " + str(root),
    ],
)
def test_cli_redacts_config_and_runner_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    build_sensitive: object,
) -> None:
    secret = "TEST_" + "CLI_SECRET"
    missing = tmp_path / secret / "missing.yaml"
    assert main(("preflight", "--config", str(missing), "--variant", "B0", "--json")) == 2
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert str(tmp_path.resolve()) not in captured.out + captured.err

    config = write_complete_config(
        tmp_path, {"source.sha256": "0" * 64}, with_source=False
    )
    raw = build_sensitive(tmp_path.resolve())  # type: ignore[operator]

    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError(raw)

    monkeypatch.setattr("webvideo_to_data.cli.run_experiment", fail)
    assert main(("run", "--config", str(config), "--variant", "B2", "--output-dir", str(tmp_path / "out"), "--no-render", "--json")) == 10
    captured = capsys.readouterr()
    assert raw not in captured.out + captured.err
    assert str(tmp_path.resolve()) not in captured.out + captured.err


@pytest.mark.parametrize(
    "build_sensitive",
    [
        lambda root: "Authorization: Bearer " + "TEST_" + "VERIFY_SECRET" + " at " + str(root),
        lambda root: "https://example.invalid/x?credential=" + "TEST_" + "VERIFY_QUERY" + " at " + str(root),
        lambda root: "provider=" + "h" + "f_" + ("V" * 32) + " at " + str(root),
    ],
)
def test_cli_redacts_verify_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    build_sensitive: object,
) -> None:
    raw = build_sensitive(tmp_path.resolve())  # type: ignore[operator]

    def fail(*_: object, **__: object) -> object:
        raise ValueError(raw)

    run = _trusted_b2(tmp_path)
    monkeypatch.setattr("webvideo_to_data.cli.verify_run_directory", fail)
    assert main(("verify", "--run", str(run), "--json")) == 4
    captured = capsys.readouterr()
    assert raw not in captured.out + captured.err
    assert str(tmp_path.resolve()) not in captured.out + captured.err


def _trusted_b2(tmp_path: Path) -> Path:
    config = write_complete_config(
        tmp_path, {"source.sha256": "0" * 64}, with_source=False
    )
    output = tmp_path / "trusted-B2"
    metrics = run_experiment(config, output, variant="B2", no_render=True)
    assert metrics["status"] == "not_run"
    return output


def _trusted_suite(tmp_path: Path) -> Path:
    config = write_complete_config(
        tmp_path, {"source.sha256": "0" * 64}, with_source=False
    )
    return run_suite(
        config,
        tmp_path / "artifacts",
        ("B2",),
        True,
        run_id="20260822T120102123456Z-a1b2c3d4-7f29",
    ).run_dir


def test_verify_auto_detects_valid_suite_and_forwards_strict_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite = _trusted_suite(tmp_path)
    decoded: list[Path] = []
    audited: list[Path] = []
    original_decode = cli_module._decode_manifested_media
    original_audit = cli_module.audit_publication_tree

    def record_decode(path: Path, verified: object) -> tuple[str, ...]:
        decoded.append(path)
        return original_decode(path, verified)  # type: ignore[arg-type]

    def record_audit(path: Path) -> object:
        audited.append(path)
        return original_audit(path)

    monkeypatch.setattr(cli_module, "_decode_manifested_media", record_decode)
    monkeypatch.setattr(cli_module, "audit_publication_tree", record_audit)

    assert main(
        (
            "verify",
            "--run",
            str(suite),
            "--decode-media",
            "--privacy-audit",
            "--json",
        )
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "artifact_path": f"<external-output>/{suite.name}",
        "command": "verify",
        "decoded_media": [],
        "privacy_audit": "passed",
        "status": "recorded",
        "variant": None,
        "verified": True,
    }
    assert len(decoded) == 1
    assert decoded == audited
    assert decoded[0] != suite
    assert not decoded[0].exists()


def test_suite_manifest_decoder_covers_root_and_nested_png_gif_mp4_once(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    root_preview = suite / "dashboard" / "media" / "B0-preview.gif"
    root_plot = suite / "dashboard" / "media" / "trajectory.png"
    child_preview = suite / "variants" / "B0" / "mujoco_replay.mp4"
    child_plot = suite / "variants" / "B0" / "trajectory_2d.png"
    root_preview.parent.mkdir(parents=True)
    child_preview.parent.mkdir(parents=True)
    source = (
        Path(__file__).parents[1]
        / "docs"
        / "media"
        / "exp001-b0-side-by-side.gif"
    )
    shutil.copyfile(source, root_preview)
    assert cv2.imwrite(str(root_plot), np.full((12, 18, 3), 120, dtype=np.uint8))
    _write_mp4(child_preview)
    assert cv2.imwrite(str(child_plot), np.full((8, 14, 3), 90, dtype=np.uint8))
    verified = SimpleNamespace(
        manifest={
            "files": {
                "dashboard/media/B0-preview.gif": {},
                "dashboard/media/trajectory.png": {},
                "variants/B0/mujoco_replay.mp4": {},
                "variants/B0/trajectory_2d.png": {},
            }
        }
    )

    decoded = cli_module._decode_manifested_media(suite, verified)

    assert decoded == (
        "dashboard/media/B0-preview.gif",
        "dashboard/media/trajectory.png",
        "variants/B0/mujoco_replay.mp4",
        "variants/B0/trajectory_2d.png",
    )


def test_manifest_decoder_rejects_malformed_png(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    malformed = suite / "dashboard" / "media" / "bad.png"
    malformed.parent.mkdir(parents=True)
    malformed.write_bytes(b"not a png")
    verified = SimpleNamespace(
        manifest={"files": {"dashboard/media/bad.png": {}}}
    )

    with pytest.raises(ValueError, match="PNG"):
        cli_module._decode_manifested_media(suite, verified)


def test_manifest_decoder_rejects_png_pixel_cap_before_opencv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = tmp_path / "suite"
    media = suite / "dashboard" / "media" / "huge.png"
    media.parent.mkdir(parents=True)
    media.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + (100_000).to_bytes(4, "big")
        + (100_000).to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )
    verified = SimpleNamespace(
        manifest={"files": {"dashboard/media/huge.png": {}}}
    )

    def forbidden_decode(*_: object, **__: object) -> object:
        raise AssertionError("PNG cap rejection must precede OpenCV")

    monkeypatch.setattr(cli_module.cv2, "imread", forbidden_decode)

    with pytest.raises(ValueError, match="PNG"):
        cli_module._decode_manifested_media(suite, verified)


def test_manifest_decoder_rejects_parent_traversal_before_media_open(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    source = (
        Path(__file__).parents[1]
        / "docs"
        / "media"
        / "exp001-b0-side-by-side.gif"
    )
    shutil.copyfile(source, tmp_path / "outside.gif")
    verified = SimpleNamespace(manifest={"files": {"../outside.gif": {}}})

    with pytest.raises(ValueError, match="manifested media path"):
        cli_module._decode_manifested_media(suite, verified)


def test_verify_auto_detects_valid_child_and_preserves_json_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _trusted_b2(tmp_path)

    assert main(("verify", "--run", str(run), "--json")) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["variant"] == "B2"
    assert payload["status"] == "not_run"
    assert payload["decoded_media"] == []
    assert payload["privacy_audit"] == "not_requested"


def test_verify_rejects_ambiguous_manifest_identity_without_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _trusted_b2(tmp_path)
    (run / "suite-manifest.json").write_text("{}\n", encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        cli_module,
        "verify_run_directory",
        lambda *_: calls.append("child"),
    )
    monkeypatch.setattr(
        cli_module,
        "verify_suite_directory",
        lambda *_: calls.append("suite"),
    )

    assert main(("verify", "--run", str(run), "--json")) == 4
    assert calls == []


def test_verify_rejects_missing_manifest_identity_without_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "neither-manifest"
    run.mkdir()
    calls: list[str] = []
    monkeypatch.setattr(
        cli_module,
        "verify_run_directory",
        lambda *_: calls.append("child"),
    )
    monkeypatch.setattr(
        cli_module,
        "verify_suite_directory",
        lambda *_: calls.append("suite"),
    )

    assert main(("verify", "--run", str(run), "--json")) == 4
    assert calls == []


def test_verify_does_not_fallback_after_tampered_suite_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _trusted_suite(tmp_path)
    (suite / "suite-manifest.json").write_text("{}\n", encoding="utf-8")
    child_calls = 0

    def record_child(*_: object) -> object:
        nonlocal child_calls
        child_calls += 1
        raise AssertionError("suite failure must not fall back to child verification")

    monkeypatch.setattr(cli_module, "verify_run_directory", record_child)

    assert main(("verify", "--run", str(suite), "--json")) == 4
    assert child_calls == 0


def test_verify_does_not_fallback_after_tampered_child_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _trusted_b2(tmp_path)
    (run / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    suite_calls = 0

    def record_suite(*_: object) -> object:
        nonlocal suite_calls
        suite_calls += 1
        raise AssertionError("child failure must not fall back to suite verification")

    monkeypatch.setattr(cli_module, "verify_suite_directory", record_suite)

    assert main(("verify", "--run", str(run), "--json")) == 4
    assert suite_calls == 0


def test_verify_fails_closed_if_suite_manifest_identity_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _trusted_suite(tmp_path)
    original_verify = cli_module._verify_suite_snapshot
    calls = 0

    def mutate_after_verify(
        materialized: Path,
        snapshot: object,
        *,
        display_path: Path,
    ) -> object:
        nonlocal calls
        calls += 1
        verified = original_verify(
            materialized,
            snapshot,  # type: ignore[arg-type]
            display_path=display_path,
        )
        if calls == 1:
            manifest_path = materialized / "suite-manifest.json"
            manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
        return verified

    monkeypatch.setattr(cli_module, "_verify_suite_snapshot", mutate_after_verify)

    assert main(("verify", "--run", str(suite), "--json")) == 4
    assert calls == 2


def _write_mp4(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (24, 16)
    )
    assert writer.isOpened()
    try:
        for index in range(4):
            writer.write(np.full((16, 24, 3), index * 30, dtype=np.uint8))
    finally:
        writer.release()


def _write_strict_media(path: Path) -> None:
    if path.suffix == ".png":
        assert cv2.imwrite(
            str(path), np.full((16, 24, 3), 90, dtype=np.uint8)
        )
    elif path.suffix == ".gif":
        imageio.mimsave(
            path,
            [
                np.full((16, 24, 3), 30, dtype=np.uint8),
                np.full((16, 24, 3), 150, dtype=np.uint8),
            ],
            format="GIF",
            duration=125,
            loop=0,
        )
    elif path.suffix == ".mp4":
        _write_mp4(path)
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(f"unsupported strict-media fixture: {path.suffix}")


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return (
        len(payload).to_bytes(4, "big")
        + chunk_type
        + payload
        + checksum.to_bytes(4, "big")
    )


def _manifest_file(
    run: Path,
    name: str,
    *,
    media: bool = False,
    media_role: str = "simulation_only",
    contains_private_source_frames: bool = False,
) -> None:
    path = run / name
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry: dict[str, object] = {
        "size": path.stat().st_size,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }
    if media:
        entry.update(
            media_role=media_role,
            contains_private_source_frames=contains_private_source_frames,
        )
    manifest["files"][name] = entry
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _accept_test_media_manifest(
    run: Path,
    verified: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    accepted = replace(verified, manifest=manifest)
    monkeypatch.setattr(cli_module, "verify_run_directory", lambda _: accepted)


def test_verify_decodes_every_manifested_media_frame(tmp_path: Path) -> None:
    run = _trusted_b2(tmp_path)
    _write_mp4(run / "mujoco_replay.mp4")
    _manifest_file(run, "mujoco_replay.mp4", media=True)
    assert main(("verify", "--run", str(run), "--decode-media", "--privacy-audit", "--json")) == 0


@pytest.mark.parametrize("suffix", (".png", ".gif", ".mp4"))
def test_verify_strict_decode_accepts_complete_media_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str,
) -> None:
    run = _trusted_b2(tmp_path)
    verified = cli_module.verify_run_directory(run)
    media = run / f"complete{suffix}"
    _write_strict_media(media)
    _manifest_file(run, media.name, media=True)
    _accept_test_media_manifest(run, verified, monkeypatch)

    assert main(("verify", "--run", str(run), "--decode-media", "--json")) == 0


def test_verify_strict_decode_precharges_gif_frame_cap_before_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _trusted_b2(tmp_path)
    verified = cli_module.verify_run_directory(run)
    media = run / "precharged.gif"
    _write_strict_media(media)
    _manifest_file(run, media.name, media=True)
    _accept_test_media_manifest(run, verified, monkeypatch)
    monkeypatch.setattr(redaction_module, "_GIF_MAX_FRAMES", 1, raising=False)
    probed: list[Path] = []
    original_ffprobe = cli_module._ffprobe_facts

    def monitored_ffprobe(path: Path) -> tuple[int, int, float, float, int]:
        probed.append(path)
        return original_ffprobe(path)

    monkeypatch.setattr(cli_module, "_ffprobe_facts", monitored_ffprobe)

    assert main(("verify", "--run", str(run), "--decode-media", "--json")) == 4
    assert probed == []


def test_verify_strict_decode_rejects_out_of_palette_gif_root_before_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _trusted_b2(tmp_path)
    verified = cli_module.verify_run_directory(run)
    media = run / "out-of-palette-root.gif"
    media.write_bytes(
        b"GIF89a\x01\x00\x01\x00\x80\x00\x00"
        + b"\x00\x00\x00" * 2
        + b"\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00"
        + b"\x02\x02\x5c\x01\x00\x3b"  # clear(4), root(3), EOI(5)
    )
    _manifest_file(run, media.name, media=True)
    _accept_test_media_manifest(run, verified, monkeypatch)
    probed: list[Path] = []
    original_ffprobe = cli_module._ffprobe_facts

    def monitored_ffprobe(path: Path) -> tuple[int, int, float, float, int]:
        probed.append(path)
        return original_ffprobe(path)

    monkeypatch.setattr(cli_module, "_ffprobe_facts", monitored_ffprobe)

    assert main(("verify", "--run", str(run), "--decode-media", "--json")) == 4
    assert probed == []


@pytest.mark.parametrize("table_type", (b"stsz", b"stsc", b"stco"))
def test_verify_strict_decode_rejects_invalid_classic_iso_table_before_ffprobe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    table_type: bytes,
) -> None:
    run = _trusted_b2(tmp_path)
    verified = cli_module.verify_run_directory(run)
    media = run / "invalid-classic-table.mp4"
    _write_mp4(media)
    content = bytearray(media.read_bytes())
    table_offset = content.find(table_type)
    assert table_offset >= 4
    content[table_offset : table_offset + 4] = b"free"
    media.write_bytes(bytes(content))
    _manifest_file(run, media.name, media=True)
    _accept_test_media_manifest(run, verified, monkeypatch)
    probed: list[Path] = []
    original_ffprobe = cli_module._ffprobe_facts

    def monitored_ffprobe(path: Path) -> tuple[int, int, float, float, int]:
        probed.append(path)
        return original_ffprobe(path)

    monkeypatch.setattr(cli_module, "_ffprobe_facts", monitored_ffprobe)

    assert main(("verify", "--run", str(run), "--decode-media", "--json")) == 4
    assert probed == []


def test_verify_strict_decode_rejects_moov_mvex_before_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _trusted_b2(tmp_path)
    verified = cli_module.verify_run_directory(run)
    media = run / "hybrid-mvex.mp4"
    _write_mp4(media)
    content = bytearray(media.read_bytes())
    moov_type = content.find(b"moov")
    assert moov_type >= 4
    moov_start = moov_type - 4
    moov_size = int.from_bytes(content[moov_start:moov_type], "big")
    moov_end = moov_start + moov_size
    assert moov_size >= 8 and moov_end <= len(content)
    content[moov_start:moov_type] = (moov_size + 8).to_bytes(4, "big")
    content[moov_end:moov_end] = b"\x00\x00\x00\x08mvex"
    media.write_bytes(bytes(content))
    _manifest_file(run, media.name, media=True)
    _accept_test_media_manifest(run, verified, monkeypatch)
    probed: list[Path] = []
    original_ffprobe = cli_module._ffprobe_facts

    def monitored_ffprobe(path: Path) -> tuple[int, int, float, float, int]:
        probed.append(path)
        return original_ffprobe(path)

    monkeypatch.setattr(cli_module, "_ffprobe_facts", monitored_ffprobe)

    assert main(("verify", "--run", str(run), "--decode-media", "--json")) == 4
    assert probed == []


@pytest.mark.parametrize("suffix", (".png", ".gif", ".mp4"))
def test_verify_strict_decode_rejects_bytes_after_container_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str,
) -> None:
    run = _trusted_b2(tmp_path)
    verified = cli_module.verify_run_directory(run)
    media = run / f"trailer{suffix}"
    _write_strict_media(media)
    media.write_bytes(media.read_bytes() + b"invalid trailer")
    _manifest_file(run, media.name, media=True)
    _accept_test_media_manifest(run, verified, monkeypatch)

    assert main(("verify", "--run", str(run), "--decode-media", "--json")) == 4


@pytest.mark.parametrize("suffix", (".png", ".gif", ".mp4"))
def test_verify_strict_decode_rejects_truncated_container_without_privacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str,
) -> None:
    run = _trusted_b2(tmp_path)
    verified = cli_module.verify_run_directory(run)
    media = run / f"truncated{suffix}"
    _write_strict_media(media)
    content = media.read_bytes()
    trim = 12 if suffix == ".png" else 1 if suffix == ".gif" else 4
    media.write_bytes(content[:-trim])
    _manifest_file(run, media.name, media=True)
    _accept_test_media_manifest(run, verified, monkeypatch)

    assert main(("verify", "--run", str(run), "--decode-media", "--json")) == 4


@pytest.mark.parametrize("suffix", (".png", ".gif", ".mp4"))
def test_verify_strict_decode_rejects_malformed_container_without_privacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str,
) -> None:
    run = _trusted_b2(tmp_path)
    verified = cli_module.verify_run_directory(run)
    media = run / f"malformed{suffix}"
    _write_strict_media(media)
    content = bytearray(media.read_bytes())
    if suffix == ".png":
        content[-1] ^= 0x01  # IEND CRC.
    elif suffix == ".gif":
        content[:6] = b"GIF89x"
    else:
        content[:4] = (7).to_bytes(4, "big")
    media.write_bytes(bytes(content))
    _manifest_file(run, media.name, media=True)
    _accept_test_media_manifest(run, verified, monkeypatch)

    assert main(("verify", "--run", str(run), "--decode-media", "--json")) == 4


@pytest.mark.parametrize("suffix", (".png", ".gif", ".mp4"))
def test_verify_strict_decode_rejects_container_resource_caps_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str,
) -> None:
    run = _trusted_b2(tmp_path)
    verified = cli_module.verify_run_directory(run)
    media = run / f"capped{suffix}"
    _write_strict_media(media)
    content = media.read_bytes()
    if suffix == ".png":
        iend = content.rfind(b"\x00\x00\x00\x00IEND")
        assert iend > 0
        content = (
            content[:iend]
            + _png_chunk(b"tEXt", b"k\x00") * 4_097
            + content[iend:]
        )
    elif suffix == ".gif":
        assert content.endswith(b"\x3b")
        content = (
            content[:-1]
            + b"\x21\xfe"
            + (b"\x01A" * 131_073)
            + b"\x00\x3b"
        )
    else:
        content += b"\x00\x00\x00\x08free" * 1_025
    media.write_bytes(content)
    _manifest_file(run, media.name, media=True)
    _accept_test_media_manifest(run, verified, monkeypatch)

    assert main(("verify", "--run", str(run), "--decode-media", "--json")) == 4


def test_verify_rejects_manifested_truncated_media(tmp_path: Path) -> None:
    run = _trusted_b2(tmp_path)
    media = run / "mujoco_replay.mp4"
    _write_mp4(media)
    content = media.read_bytes()
    media.write_bytes(content[: len(content) // 2])
    _manifest_file(run, media.name, media=True)
    assert main(("verify", "--run", str(run), "--decode-media", "--json")) == 4


def test_verify_rejects_manifested_malformed_png_in_strict_decode(
    tmp_path: Path,
) -> None:
    run = _trusted_b2(tmp_path)
    media = run / "trajectory_2d.png"
    media.write_bytes(b"not a png")
    _manifest_file(
        run,
        media.name,
        media=True,
        media_role="derived_trajectory_plot",
    )

    assert main(("verify", "--run", str(run), "--decode-media", "--json")) == 4


@pytest.mark.parametrize(
    "stream_update,format_update",
    [
        ({}, {}),
        ({"nb_read_frames": "N/A"}, {}),
        ({"nb_read_frames": "0"}, {}),
        ({"nb_read_frames": "5"}, {}),
        ({"nb_read_frames": "4", "avg_frame_rate": "NaN/1"}, {}),
        ({"nb_read_frames": "4"}, {"duration": "NaN"}),
        ({"nb_read_frames": "4"}, {"duration": "Inf"}),
        ({"nb_read_frames": "4"}, {"duration": "1.05"}),
    ],
)
def test_verify_fails_closed_on_untrusted_ffprobe_frame_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream_update: dict[str, str],
    format_update: dict[str, str],
) -> None:
    run = _trusted_b2(tmp_path)
    _write_mp4(run / "mujoco_replay.mp4")
    _manifest_file(run, "mujoco_replay.mp4", media=True)
    stream = {"width": 24, "height": 16, "avg_frame_rate": "5/1"}
    stream.update(stream_update)
    format_facts = {"duration": "0.8"}
    format_facts.update(format_update)
    monkeypatch.setattr(
        "webvideo_to_data.cli.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"streams": [stream], "format": format_facts}),
            stderr="",
        ),
    )
    assert main(("verify", "--run", str(run), "--decode-media", "--json")) == 4


def test_cli_privacy_audit_returns_four_when_ffprobe_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _trusted_b2(tmp_path)
    _write_mp4(run / "mujoco_replay.mp4")
    _manifest_file(run, "mujoco_replay.mp4", media=True)
    monkeypatch.setattr("webvideo_to_data.redaction.shutil.which", lambda _: None)
    assert main(("verify", "--run", str(run), "--privacy-audit", "--json")) == 4


@pytest.mark.parametrize(
    "argv",
    [
        ("--json", "--help"),
        ("--help", "--json"),
        ("run", "--json", "--help"),
        ("run", "--help", "--json"),
    ],
)
def test_json_help_is_one_object_without_system_exit(
    argv: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["command"] == "help"
    assert captured.out.count("\n") == 1
    assert captured.err == ""


def test_cli_suite_verify_owns_only_one_stable_snapshot_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite = _trusted_suite(tmp_path)
    original_snapshot = redaction_module._stable_publication_snapshot
    calls = 0

    @contextmanager
    def count_snapshot(path: str | Path):
        nonlocal calls
        calls += 1
        with original_snapshot(path) as snapshot:
            yield snapshot

    monkeypatch.setattr(cli_module, "_stable_publication_snapshot", count_snapshot)
    monkeypatch.setattr(suite_module, "_stable_publication_snapshot", count_snapshot)

    assert main(("verify", "--run", str(suite), "--json")) == 0
    assert calls == 1
    assert json.loads(capsys.readouterr().out)["verified"] is True


def test_stable_snapshot_streams_large_unretained_file_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _trusted_b2(tmp_path)
    payload = (b"streamed-not-retained-" * 160_000)[:3_000_000]
    (run / "large.bin").write_bytes(payload)
    _manifest_file(run, "large.bin")
    read_sizes: list[int] = []
    helper_name = (
        "_snapshot_read_windows" if sys.platform == "win32" else "_snapshot_read_posix"
    )
    original_read = getattr(redaction_module, helper_name)

    def bounded_manifest_read(handle: object, size: int) -> bytes:
        read_sizes.append(size)
        return original_read(handle, size)

    monkeypatch.setattr(redaction_module, helper_name, bounded_manifest_read)

    with redaction_module._stable_publication_snapshot(run) as snapshot:
        assert snapshot.materialized_root is not None
        assert (snapshot.materialized_root / "large.bin").read_bytes() == payload
        assert "large.bin" not in snapshot.captured_files
        assert snapshot.file_signatures["large.bin"] == (
            len(payload),
            sha256(payload).hexdigest(),
        )

    assert read_sizes
    assert max(read_sizes) < len(payload)


def test_stable_snapshot_precharges_captured_file_cap_before_payload_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _trusted_b2(tmp_path)
    dashboard = run / "dashboard" / "index.html"
    dashboard.parent.mkdir()
    dashboard.write_bytes(b"x" * 257)
    _manifest_file(run, "dashboard/index.html")
    monkeypatch.setattr(
        redaction_module, "_STABLE_CAPTURED_DASHBOARD_BYTES", 256
    )
    payload_reads = 0
    if sys.platform == "win32":
        original_read_at = redaction_module._win_read_at

        def reject_payload_read(handle: object, offset: int, size: int) -> bytes:
            nonlocal payload_reads
            if size == 257:
                payload_reads += 1
            return original_read_at(handle, offset, size)

        monkeypatch.setattr(redaction_module, "_win_read_at", reject_payload_read)
    else:
        original_pread = redaction_module.os.pread

        def reject_payload_read(fd: int, size: int, offset: int) -> bytes:
            nonlocal payload_reads
            if size == 257:
                payload_reads += 1
            return original_pread(fd, size, offset)

        monkeypatch.setattr(redaction_module.os, "pread", reject_payload_read)

    with pytest.raises(ValueError, match="explicit cap"):
        with redaction_module._stable_publication_snapshot(run):
            pass

    assert payload_reads == 0


@pytest.mark.parametrize("argv", [("--help",), ("run", "--help")])
def test_human_help_returns_zero_and_prints_once(
    argv: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert captured.out.count("usage:") == 1
    assert captured.err == ""


@pytest.mark.parametrize(
    "sensitive",
    [
        lambda: "Authorization: Bearer " + "TEST_" + "PRIVATE_VALUE",
        lambda: "https://example.invalid/x?credential=" + "TEST_" + "SIGNED_VALUE",
        lambda: "gh" + "p_" + ("R" * 40),
        lambda: "/home/" + "fixture-person" + "/private/video.mp4",
    ],
)
def test_verify_privacy_audit_rejects_each_sensitive_fixture(
    tmp_path: Path, sensitive: object
) -> None:
    run = _trusted_b2(tmp_path)
    value = sensitive()  # type: ignore[operator]
    (run / "notes.txt").write_text(value, encoding="utf-8")
    _manifest_file(run, "notes.txt")
    assert main(("verify", "--run", str(run), "--privacy-audit", "--json")) == 4


def test_pipeline_failure_is_redacted_before_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_complete_config(tmp_path)
    output = tmp_path / "failed-B1"
    secret = "TEST_" + "PERSISTED_SECRET"
    generic_paths = (
        "/" + "tmp",
        "\\" + "private\\stage.bin",
        "D:" + "private\\stage.bin",
    )
    raw = (
        "Authorization: Bearer " + secret + " at " + str(tmp_path.resolve())
        + " " + " ".join(generic_paths)
    )

    def fail_probe(*_: object) -> object:
        raise RuntimeError(raw)

    monkeypatch.setattr("webvideo_to_data.experiment.probe_video", fail_probe)
    metrics = run_experiment(config, output, variant="B1", no_render=True)
    assert metrics["status"] == "failed"
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.iterdir()
        if path.suffix == ".json"
    )
    assert secret not in persisted
    assert str(tmp_path.resolve()) not in persisted
    for private_value in generic_paths:
        assert private_value not in persisted


def test_publication_failure_json_redacts_transaction_and_external_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _trusted_b2(tmp_path)
    config = tmp_path / "config.yaml"
    windows_path = "D:" + "\\private-area\\publication.bin"
    posix_path = "/mnt/" + "private-area/publication.bin"
    generic_paths = (
        "/" + "tmp",
        "\\" + "private\\publication.bin",
        "D:" + "private\\publication.bin",
    )

    def fail_cleanup(backup: Path, expected: object) -> None:
        del expected
        raise OSError(
            f"cleanup {backup} {windows_path} {posix_path} "
            + " ".join(generic_paths)
        )

    monkeypatch.setattr(experiment_module, "_remove_trusted_backup", fail_cleanup)
    metrics = run_experiment(config, output, variant="B2", no_render=True)
    assert metrics["status"] == "failed"
    persisted = "\n".join(
        (output / name).read_text(encoding="utf-8")
        for name in ("metrics.json", "rejection.json")
    )
    for private_value in (
        str(output.resolve()), windows_path, posix_path, *generic_paths
    ):
        assert private_value not in persisted


def test_developer_traceback_is_formatted_then_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_complete_config(
        tmp_path, {"source.sha256": "0" * 64}, with_source=False
    )
    output = tmp_path / "trace-output"
    windows_path = "D:" + "\\private-area\\trace.bin"
    posix_path = "/mnt/" + "private-area/trace.bin"
    generic_paths = (
        "/" + "tmp",
        "\\" + "private\\trace.bin",
        "D:" + "private\\trace.bin",
    )
    raw = (
        f"runner failed at {windows_path} and {posix_path} "
        + " ".join(generic_paths)
    )

    def fail(*_: object, **__: object) -> object:
        raise RuntimeError(raw)

    monkeypatch.setattr("webvideo_to_data.cli.run_experiment", fail)
    monkeypatch.setenv("WEBVIDEO_TO_DATA_DEVELOPER_TRACEBACK", "1")
    code = main(
        (
            "run",
            "--config",
            str(config),
            "--variant",
            "B2",
            "--output-dir",
            str(output),
            "--no-render",
            "--json",
        )
    )
    captured = capsys.readouterr()
    assert code == 10
    assert isinstance(json.loads(captured.out), dict)
    assert captured.out.count("\n") == 1
    assert "Traceback" in captured.err
    for private_value in (
        windows_path, posix_path, str(config), str(output), *generic_paths
    ):
        assert private_value not in captured.out + captured.err


def test_cli_verify_identity_rejects_root_reparse_point_without_optional_flags(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = _trusted_b2(tmp_path)
    run = tmp_path / "trusted-B2-link"
    if sys.platform == "win32":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(run), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout
    else:
        run.symlink_to(target, target_is_directory=True)
    try:
        code = main(("verify", "--run", str(run), "--json"))
    finally:
        if sys.platform == "win32":
            run.rmdir()
        else:
            run.unlink()
    assert code == 4
    assert isinstance(json.loads(capsys.readouterr().out), dict)


def test_cli_verify_identity_rejects_ancestor_reparse_point_without_optional_flags(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target_parent = tmp_path / "target-parent"
    target_parent.mkdir()
    target = _trusted_b2(target_parent)
    linked_parent = tmp_path / "linked-parent"
    if sys.platform == "win32":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(linked_parent), str(target_parent)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout
    else:
        linked_parent.symlink_to(target_parent, target_is_directory=True)
    try:
        code = main(("verify", "--run", str(linked_parent / target.name), "--json"))
    finally:
        if sys.platform == "win32":
            linked_parent.rmdir()
        else:
            linked_parent.unlink()
    assert code == 4
    assert isinstance(json.loads(capsys.readouterr().out), dict)


def test_cli_rejects_huge_declared_file_before_verifier_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _trusted_b2(tmp_path)
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["metrics.json"]["size"] = 8_193
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        redaction_module, "_STABLE_SNAPSHOT_MAX_FILE_BYTES", 8_192, raising=False
    )
    dispatched: list[Path] = []

    def forbidden_dispatch(path: Path) -> object:
        dispatched.append(path)
        raise AssertionError("declared cap rejection must precede verifier dispatch")

    monkeypatch.setattr(cli_module, "verify_run_directory", forbidden_dispatch)

    assert main(("verify", "--run", str(run), "--json")) == 4
    assert dispatched == []
    assert isinstance(json.loads(capsys.readouterr().out), dict)


def test_cli_rejects_huge_extra_file_before_verifier_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _trusted_b2(tmp_path)
    (run / "undeclared.bin").write_bytes(b"x" * 8_193)
    monkeypatch.setattr(
        redaction_module, "_STABLE_SNAPSHOT_MAX_FILE_BYTES", 8_192, raising=False
    )
    dispatched: list[Path] = []

    def forbidden_dispatch(path: Path) -> object:
        dispatched.append(path)
        raise AssertionError("extra entry rejection must precede verifier dispatch")

    monkeypatch.setattr(cli_module, "verify_run_directory", forbidden_dispatch)

    assert main(("verify", "--run", str(run), "--json")) == 4
    assert dispatched == []


def test_cli_rejects_declared_aggregate_and_work_caps_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _trusted_b2(tmp_path)
    monkeypatch.setattr(
        redaction_module, "_STABLE_SNAPSHOT_MAX_AGGREGATE_BYTES", 1, raising=False
    )
    monkeypatch.setattr(
        redaction_module, "_STABLE_SNAPSHOT_MAX_FILES", 1, raising=False
    )
    dispatched: list[Path] = []

    def forbidden_dispatch(path: Path) -> object:
        dispatched.append(path)
        raise AssertionError("aggregate rejection must precede verifier dispatch")

    monkeypatch.setattr(cli_module, "verify_run_directory", forbidden_dispatch)

    assert main(("verify", "--run", str(run), "--json")) == 4
    assert dispatched == []


def test_cli_rejects_excessive_declared_path_depth_before_prefix_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _trusted_b2(tmp_path)
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    deep_name = "/".join(["d"] * 65 + ["extra.bin"])
    manifest["files"][deep_name] = {"size": 0, "sha256": "0" * 64}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    prefix_calls = 0

    def forbidden_prefix_allocation(*_: object) -> object:
        nonlocal prefix_calls
        prefix_calls += 1
        raise AssertionError("path-depth cap must precede prefix allocation")

    monkeypatch.setattr(
        redaction_module,
        "_snapshot_expected_directories",
        forbidden_prefix_allocation,
    )

    assert main(("verify", "--run", str(run), "--json")) == 4
    assert prefix_calls == 0


@pytest.mark.skipif(not hasattr(__import__("os"), "mkfifo"), reason="FIFO unavailable")
def test_cli_rejects_special_file_before_verifier_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    run = _trusted_b2(tmp_path)
    os.mkfifo(run / "undeclared.fifo")
    dispatched: list[Path] = []

    def forbidden_dispatch(path: Path) -> object:
        dispatched.append(path)
        raise AssertionError("special entry rejection must precede verifier dispatch")

    monkeypatch.setattr(cli_module, "verify_run_directory", forbidden_dispatch)

    assert main(("verify", "--run", str(run), "--json")) == 4
    assert dispatched == []


def test_cli_cleans_private_materialization_after_verifier_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _trusted_b2(tmp_path)
    dispatched: list[Path] = []

    def fail_after_materialization(path: Path) -> object:
        dispatched.append(path)
        raise ValueError("expected verifier failure")

    monkeypatch.setattr(cli_module, "verify_run_directory", fail_after_materialization)

    assert main(("verify", "--run", str(run), "--json")) == 4
    assert len(dispatched) == 1
    assert dispatched[0] != run
    assert not dispatched[0].exists()


def test_cli_rejects_root_manifest_change_between_preflight_and_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _trusted_b2(tmp_path)
    manifest_path = run / "run_manifest.json"
    original_stat = manifest_path.stat()
    original_content = manifest_path.read_bytes()
    assert original_content.endswith(b"\n")
    mutated = False
    helper_name = (
        "_snapshot_read_windows" if sys.platform == "win32" else "_snapshot_read_posix"
    )
    original_read = getattr(redaction_module, helper_name)

    def mutate_after_preflight(handle: object, size: int) -> bytes:
        nonlocal mutated
        content = original_read(handle, size)
        if not mutated and size == len(original_content):
            manifest_path.write_bytes(original_content[:-1] + b" ")
            __import__("os").utime(
                manifest_path,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            mutated = True
        return content

    monkeypatch.setattr(redaction_module, helper_name, mutate_after_preflight)
    dispatched: list[Path] = []

    def forbidden_dispatch(path: Path) -> object:
        dispatched.append(path)
        raise AssertionError("manifest mutation must precede verifier dispatch")

    monkeypatch.setattr(cli_module, "verify_run_directory", forbidden_dispatch)

    assert main(("verify", "--run", str(run), "--json")) == 4
    assert mutated
    assert dispatched == []


def test_cli_uses_pinned_root_when_original_ancestor_is_swapped_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trusted_parent = tmp_path / "trusted-parent"
    trusted_parent.mkdir()
    trusted = _trusted_b2(trusted_parent)
    alternate_parent = tmp_path / "alternate-parent"
    alternate_parent.mkdir()
    config = write_complete_config(
        alternate_parent, {"source.sha256": "0" * 64}, with_source=False
    )
    alternate = alternate_parent / trusted.name
    alternate_metrics = run_experiment(
        config, alternate, variant="B0", no_render=True
    )
    assert alternate_metrics["status"] == "rejected"
    backup = tmp_path / "trusted-parent-backup"
    original_identity = cli_module._verify_identity
    swapped = False

    def install_alternate() -> None:
        trusted_parent.replace(backup)
        if sys.platform == "win32":
            completed = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(trusted_parent), str(alternate_parent)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert completed.returncode == 0, completed.stderr or completed.stdout
        else:
            trusted_parent.symlink_to(alternate_parent, target_is_directory=True)

    def restore_trusted() -> None:
        if sys.platform == "win32":
            trusted_parent.rmdir()
        else:
            trusted_parent.unlink()
        backup.replace(trusted_parent)

    def swap_around_identity(path: Path) -> object:
        nonlocal swapped
        if not swapped:
            identity = original_identity(path)
            install_alternate()
            swapped = True
            return identity
        restore_trusted()
        return original_identity(path)

    monkeypatch.setattr(cli_module, "_verify_identity", swap_around_identity)
    try:
        code = main(("verify", "--run", str(trusted), "--json"))
    finally:
        if backup.exists():
            if trusted_parent.exists():
                if sys.platform == "win32":
                    trusted_parent.rmdir()
                else:
                    trusted_parent.unlink()
            backup.replace(trusted_parent)
    payload = json.loads(capsys.readouterr().out)

    if code == 0:
        assert swapped
        assert payload["variant"] == "B2"
        assert payload["status"] == "not_run"
    else:
        assert code == 4
        assert not swapped
        assert payload["error"] == "verification_failed"

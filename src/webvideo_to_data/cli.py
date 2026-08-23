"""Stable command line interface for one trustworthy experiment variant."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from math import isfinite
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import subprocess
import sys
import traceback
from typing import Mapping, Sequence

import cv2

from .artifacts import (
    VerifiedRun,
    _stable_file_bytes,
    verify_run_directory,
)
from .config import load_experiment_config
from .dashboard import generate_dashboard, write_dashboard_copy
from .experiment import run_experiment
from .path_security import absolute_filesystem_path
from .preflight import PreflightReport, run_preflight
from .redaction import (
    _is_link_or_reparse,
    _StablePublicationSnapshot,
    _stable_directory_identity_no_follow,
    _stable_publication_snapshot,
    audit_publication_tree,
    redact_text,
    validate_media_container_bytes,
)
from .suite import (
    VerifiedSuite,
    SuiteResult,
    _verify_suite_snapshot,
    run_suite,
    verify_suite_directory,
)


EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_PREFLIGHT = 3
EXIT_VERIFY = 4
EXIT_NOT_COMPLETED = 5
EXIT_RUNNER = 10
_VARIANTS = ("B0", "B1", "B2", "B3", "B4")
_STRICT_MEDIA_MAX_BYTES = 64 * 1024 * 1024
_STRICT_PNG_MAX_PIXELS = 16 * 1024 * 1024


class _CliFailure(Exception):
    def __init__(self, code: int, kind: str, detail: object, remediation: str = "") -> None:
        super().__init__(str(detail))
        self.code = code
        self.kind = kind
        self.detail = str(detail)
        self.remediation = remediation


class _HelpRequested(Exception):
    def __init__(self, help_text: str) -> None:
        super().__init__(help_text)
        self.help_text = help_text


class _ArgumentParser(argparse.ArgumentParser):
    def print_help(self, file: object = None) -> None:
        del file
        raise _HelpRequested(self.format_help())

    def error(self, message: str) -> None:
        raise _CliFailure(
            EXIT_CONFIG,
            "usage_error",
            message,
            "Use --help and provide every required option.",
        )


def _parser() -> _ArgumentParser:
    parser = _ArgumentParser(prog="webvideo-to-data", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="run read-only checks")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--variant", choices=_VARIANTS, action="append", required=True)
    preflight.add_argument("--no-render", action="store_true")
    preflight.add_argument("--json", action="store_true")

    run = commands.add_parser("run", help="run one append-only suite or one compatibility variant")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--variant", choices=_VARIANTS, action="append")
    run.add_argument("--all", action="store_true")
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--artifacts-root", type=Path)
    run.add_argument("--run-id")
    run.add_argument("--no-render", action="store_true")
    run.add_argument("--require-completed", action="store_true")
    run.add_argument("--json", action="store_true")

    verify = commands.add_parser("verify", help="verify one suite or v4 variant directory")
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--decode-media", action="store_true")
    verify.add_argument("--privacy-audit", action="store_true")
    verify.add_argument("--require-completed", action="store_true")
    verify.add_argument("--json", action="store_true")

    dashboard = commands.add_parser(
        "dashboard", help="verify and locate or externally copy a static suite dashboard"
    )
    dashboard.add_argument("--run", type=Path, required=True)
    dashboard.add_argument("--output", type=Path)
    dashboard.add_argument("--json", action="store_true")
    return parser


def _display_path(path: str | Path) -> str:
    candidate = Path(path).resolve()
    workspace = Path.cwd().resolve()
    try:
        relative = candidate.relative_to(workspace)
    except ValueError:
        return f"<external-output>/{candidate.name}"
    rendered = relative.as_posix()
    return rendered if rendered else "."


def _sanitize(value: object, *, sensitive_paths: Sequence[Path] = ()) -> object:
    if isinstance(value, str):
        return redact_text(
            value,
            workspace=Path.cwd().resolve(),
            sensitive_paths=sensitive_paths,
        )
    if isinstance(value, Path):
        return _display_path(value)
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize(item, sensitive_paths=sensitive_paths)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, sensitive_paths=sensitive_paths) for item in value]
    return value


def _report_payload(report: PreflightReport) -> dict[str, object]:
    return {"passed": report.passed, "checks": [asdict(check) for check in report.checks]}


def _preflight_lines(report: PreflightReport) -> list[str]:
    lines: list[str] = []
    for check in report.checks:
        label = "PASS" if check.passed else "FAIL"
        line = f"[{label}] {check.name}: {check.status} ({check.code}) {check.detail}"
        if check.remediation:
            line += f"; remediation: {check.remediation}"
        lines.append(line)
    return lines


def _preflight_exit(report: PreflightReport) -> int:
    if report.passed:
        return EXIT_OK
    return EXIT_CONFIG if not report.by_name("config").passed else EXIT_PREFLIGHT


def _command_preflight(arguments: argparse.Namespace) -> tuple[int, dict[str, object], list[str]]:
    report = run_preflight(
        arguments.config, tuple(arguments.variant), arguments.no_render
    )
    return (
        _preflight_exit(report),
        {"command": "preflight", **_report_payload(report)},
        [*_preflight_lines(report), f"preflight: {'passed' if report.passed else 'failed'}"],
    )


def _load_cli_config(path: Path) -> None:
    try:
        load_experiment_config(path)
    except Exception as error:
        raise _CliFailure(
            EXIT_CONFIG,
            "invalid_config",
            error,
            "Provide a complete schema-v2 experiment config.",
        ) from error


def _command_run(arguments: argparse.Namespace) -> tuple[int, dict[str, object], list[str]]:
    variants = tuple(arguments.variant or ())
    if arguments.output_dir is not None:
        if arguments.all or arguments.run_id is not None or arguments.artifacts_root is not None:
            raise _CliFailure(
                EXIT_CONFIG,
                "usage_error",
                "--output-dir conflicts with append-only suite options",
                "Use --output-dir only with exactly one --variant.",
            )
        if len(variants) != 1:
            raise _CliFailure(
                EXIT_CONFIG,
                "usage_error",
                "--output-dir requires exactly one --variant",
                "Provide one --variant with the compatibility output form.",
            )
    else:
        if arguments.all and variants:
            raise _CliFailure(
                EXIT_CONFIG,
                "usage_error",
                "--all conflicts with --variant",
                "Choose --all or one or more --variant options.",
            )
        if not arguments.all and not variants:
            raise _CliFailure(
                EXIT_CONFIG,
                "usage_error",
                "run requires --variant or --all",
                "Choose --all or provide at least one --variant.",
            )
        if arguments.all:
            variants = _VARIANTS
    _load_cli_config(arguments.config)
    report = run_preflight(
        arguments.config, variants, arguments.no_render
    )
    if not report.passed:
        return (
            _preflight_exit(report),
            {"command": "run", "preflight": _report_payload(report)},
            [*_preflight_lines(report), "preflight: failed"],
        )
    if arguments.output_dir is not None:
        variant = variants[0]
        metrics = run_experiment(
            arguments.config,
            arguments.output_dir,
            variant=variant,
            no_render=arguments.no_render,
        )
        code = EXIT_NOT_COMPLETED if arguments.require_completed and metrics.get("status") != "completed" else EXIT_OK
        artifact_path = _display_path(arguments.output_dir)
        payload = {
            "command": "run",
            "variant": variant,
            "status": metrics.get("status"),
            "artifact_path": artifact_path,
            "metrics": metrics,
            "preflight": _report_payload(report),
        }
        lines = [*_preflight_lines(report), f"status: {metrics.get('status')}", f"artifact: {artifact_path}"]
        return code, payload, lines

    artifacts_root = arguments.artifacts_root or Path("artifacts")
    result = run_suite(
        arguments.config,
        artifacts_root,
        variants=variants,
        no_render=arguments.no_render,
        run_id=arguments.run_id,
    )
    return _suite_cli_result(
        result,
        artifacts_root=artifacts_root,
        variants=variants,
        require_completed=arguments.require_completed,
        preflight_lines=_preflight_lines(report),
    )


def _suite_cli_result(
    result: SuiteResult,
    *,
    artifacts_root: Path,
    variants: Sequence[str],
    require_completed: bool,
    preflight_lines: Sequence[str],
) -> tuple[int, dict[str, object], list[str]]:
    root = artifacts_root.resolve(strict=False)
    try:
        run_path = result.run_dir.resolve(strict=True).relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("suite result escaped artifacts root") from error
    variant_metrics = result.metrics.get("variants")
    if not isinstance(variant_metrics, Mapping):
        raise ValueError("suite result lacks variant metrics")
    statuses = [
        variant_metrics[variant].get("status")
        for variant in variants
        if isinstance(variant_metrics.get(variant), Mapping)
    ]
    if len(variants) == 1:
        terminal = variant_metrics[variants[0]]
        payload: dict[str, object] = {
            "run_id": result.run_id,
            "run_path": run_path,
            "requested_variants": list(variants),
            "status": terminal.get("status"),
            "reason": terminal.get("reason"),
        }
        status = terminal.get("status")
    else:
        payload = {
            "run_id": result.run_id,
            "run_path": run_path,
            "requested_variants": list(variants),
            "status": "recorded",
            "variants": dict(variant_metrics),
        }
        status = "recorded"
    code = (
        EXIT_NOT_COMPLETED
        if require_completed and any(item != "completed" for item in statuses)
        else EXIT_OK
    )
    lines = [*preflight_lines, f"status: {status}", f"artifact: {run_path}"]
    return code, payload, lines


def _ffprobe_facts(path: Path) -> tuple[int, int, float, float, int]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("ffprobe rejected manifested media")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        dimensions: list[int] = []
        for raw_dimension in (stream["width"], stream["height"]):
            if isinstance(raw_dimension, bool):
                raise ValueError
            if isinstance(raw_dimension, int):
                dimension = raw_dimension
            elif isinstance(raw_dimension, str) and raw_dimension.isdecimal():
                dimension = int(raw_dimension)
            else:
                raise ValueError
            dimensions.append(dimension)
        width, height = dimensions
        numerator, denominator = stream["avg_frame_rate"].split("/", maxsplit=1)
        fps = float(numerator) / float(denominator)
        duration = float(payload["format"]["duration"])
        raw_count = stream["nb_read_frames"]
        if not isinstance(raw_count, str) or not raw_count.isdecimal():
            raise ValueError
        ffprobe_count = int(raw_count)
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("ffprobe returned invalid manifested media facts") from error
    if (
        min(width, height, ffprobe_count) <= 0
        or not isfinite(fps)
        or not isfinite(duration)
        or fps <= 0.0
        or duration <= 0.0
    ):
        raise ValueError("ffprobe returned non-positive or non-finite media facts")
    return width, height, fps, duration, ffprobe_count


def _manifested_media_path(root: Path, name: object) -> Path:
    if type(name) is not str or not name or "\\" in name:
        raise ValueError("manifested media path is invalid")
    relative = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        relative.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or relative.as_posix() != name
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("manifested media path is invalid")
    return root.joinpath(*relative.parts)


def _decode_manifested_media(
    run_path: Path, verified: VerifiedRun | VerifiedSuite
) -> tuple[str, ...]:
    files = verified.manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("verified manifest lacks files")
    decoded: list[str] = []
    for name in sorted(files):
        media_path = _manifested_media_path(run_path, name)
        suffix = media_path.suffix.lower()
        if suffix not in {".png", ".gif", ".mp4"}:
            continue
        size = media_path.stat().st_size
        if size > _STRICT_MEDIA_MAX_BYTES:
            raise ValueError("manifested media exceeds strict decode byte cap")
        content = media_path.read_bytes()
        if len(content) != size:
            raise ValueError("manifested media changed during strict decode")
        if suffix == ".png":
            header = content[:24]
            if (
                len(header) != 24
                or header[:8] != b"\x89PNG\r\n\x1a\n"
                or header[8:16] != b"\x00\x00\x00\rIHDR"
            ):
                raise ValueError("manifested PNG header is invalid")
            width = int.from_bytes(header[16:20], "big")
            height = int.from_bytes(header[20:24], "big")
            if (
                width <= 0
                or height <= 0
                or width > _STRICT_PNG_MAX_PIXELS // height
            ):
                raise ValueError("manifested PNG exceeds strict decode pixel cap")
        if not validate_media_container_bytes(suffix, content):
            label = {".png": "PNG", ".gif": "GIF", ".mp4": "ISO-BMFF"}[suffix]
            raise ValueError(f"manifested {label} container is invalid")
        if suffix == ".png":
            image = cv2.imread(str(media_path), cv2.IMREAD_UNCHANGED)
            if (
                image is None
                or image.size == 0
                or image.ndim not in {2, 3}
                or min(image.shape[:2]) <= 0
            ):
                raise ValueError("OpenCV could not decode manifested PNG")
            decoded.append(name)
            continue
        width, height, fps, duration, ffprobe_count = _ffprobe_facts(media_path)
        capture = cv2.VideoCapture(str(media_path))
        frame_count = 0
        try:
            if not capture.isOpened():
                raise ValueError("OpenCV could not open manifested media")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame is None or frame.size == 0:
                    raise ValueError("OpenCV returned an empty manifested media frame")
                if frame.shape[1] != width or frame.shape[0] != height:
                    raise ValueError("decoded media dimensions disagree with ffprobe")
                frame_count += 1
        finally:
            capture.release()
        if frame_count == 0:
            raise ValueError("manifested media contains no decodable frames")
        if frame_count != ffprobe_count:
            raise ValueError("decoded frame count disagrees with ffprobe")
        last_timestamp = (frame_count - 1) / fps
        expected_duration = last_timestamp + (1.0 / fps)
        if abs(expected_duration - duration) > max(0.001, 0.5 / fps):
            raise ValueError("decoded media duration disagrees with ffprobe")
        decoded.append(name)
    return tuple(decoded)


def _same_snapshot(before: VerifiedRun, after: VerifiedRun) -> bool:
    return (
        before.directory_identity == after.directory_identity
        and before.snapshot == after.snapshot
        and before.manifest == after.manifest
        and before.metrics == after.metrics
    )


def _same_suite_snapshot(before: VerifiedSuite, after: VerifiedSuite) -> bool:
    return (
        before.manifest == after.manifest
        and before.metrics == after.metrics
        and before.environment == after.environment
        and before.variant_provenance == after.variant_provenance
        and set(before.variant_runs) == set(after.variant_runs)
        and all(
            _same_snapshot(before.variant_runs[name], after.variant_runs[name])
            for name in before.variant_runs
        )
    )


@dataclass(frozen=True)
class _VerifyIdentity:
    kind: str
    directory_identity: tuple[int, int]
    suite_manifest: object | None
    variant_manifest: object | None


def _stable_optional_manifest(path: Path) -> object | None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return None
    if _is_link_or_reparse(status) or not stat.S_ISREG(status.st_mode):
        raise ValueError("verification manifest must be a stable regular file")
    return _stable_file_bytes(path)


def _verify_identity(path: Path) -> _VerifyIdentity:
    before_identity = _stable_directory_identity_no_follow(path)
    before = (
        _stable_optional_manifest(path / "suite-manifest.json"),
        _stable_optional_manifest(path / "run_manifest.json"),
    )
    after = (
        _stable_optional_manifest(path / "suite-manifest.json"),
        _stable_optional_manifest(path / "run_manifest.json"),
    )
    if (
        before != after
        or _stable_directory_identity_no_follow(path) != before_identity
    ):
        raise ValueError("verification manifest identity changed")
    suite_manifest, variant_manifest = after
    if (suite_manifest is None) == (variant_manifest is None):
        raise ValueError("verification directory has ambiguous manifest identity")
    return _VerifyIdentity(
        kind="suite" if suite_manifest is not None else "variant",
        directory_identity=before_identity,
        suite_manifest=suite_manifest,
        variant_manifest=variant_manifest,
    )


def _command_verify(arguments: argparse.Namespace) -> tuple[int, dict[str, object], list[str]]:
    display = _display_path(arguments.run)
    try:
        with _stable_publication_snapshot(arguments.run) as snapshot:
            stable_run = snapshot.materialized_root
            if stable_run is None:
                raise ValueError("stable verification snapshot was not materialized")
            identity = _verify_identity(stable_run)
            if identity.kind != snapshot.kind:
                raise ValueError("stable manifest identity changed")
            if identity.kind == "suite":
                verified_suite = _verify_suite_snapshot(
                    stable_run,
                    snapshot,
                    display_path=Path(arguments.run),
                )
                decoded = (
                    _decode_manifested_media(stable_run, verified_suite)
                    if arguments.decode_media
                    else ()
                )
                variant: str | None = None
                status = verified_suite.metrics.get("status")
            else:
                verified_run = verify_run_directory(stable_run)
                if verified_run.manifest.get("format_version") != 4:
                    raise ValueError("verify accepts only v4 variant directories")
                decoded = (
                    _decode_manifested_media(stable_run, verified_run)
                    if arguments.decode_media
                    else ()
                )
                variant = verified_run.manifest.get("variant")
                status = verified_run.metrics.get("status")
            findings = (
                audit_publication_tree(stable_run)
                if arguments.privacy_audit
                else ()
            )
            if findings:
                raise ValueError("publication privacy audit found sensitive material")
            if identity.kind == "suite":
                after_suite = _verify_suite_snapshot(
                    stable_run,
                    snapshot,
                    display_path=Path(arguments.run),
                )
                if not _same_suite_snapshot(verified_suite, after_suite):
                    raise ValueError(
                        "suite directory changed during extended verification"
                    )
            else:
                after_run = verify_run_directory(stable_run)
                if not _same_snapshot(verified_run, after_run):
                    raise ValueError(
                        "run directory changed during extended verification"
                    )
            if _verify_identity(stable_run) != identity:
                raise ValueError("verification manifest identity changed")
            code = (
                EXIT_NOT_COMPLETED
                if arguments.require_completed and status != "completed"
                else EXIT_OK
            )
            payload = {
                "command": "verify",
                "verified": True,
                "variant": variant,
                "status": status,
                "artifact_path": display,
                "decoded_media": list(decoded),
                "privacy_audit": (
                    "passed" if arguments.privacy_audit else "not_requested"
                ),
            }
            label = variant if variant is not None else "suite"
            lines = [
                f"verified: {label} {status}",
                f"artifact: {display}",
            ]
    except Exception as error:
        raise _CliFailure(
            EXIT_VERIFY,
            "verification_failed",
            error,
            "Regenerate the variant and rerun verification.",
        ) from error
    return code, payload, lines


def _command_dashboard(
    arguments: argparse.Namespace,
) -> tuple[int, dict[str, object], list[str]]:
    try:
        run = absolute_filesystem_path(arguments.run, "verified suite")
        if arguments.output is None:
            generate_dashboard(run)
            dashboard_path = "dashboard/index.html"
        else:
            lexical_output = absolute_filesystem_path(
                arguments.output, "dashboard copy output"
            )
            output = write_dashboard_copy(run, lexical_output)
            dashboard_path = _display_path(output)
    except Exception as error:
        raise _CliFailure(
            EXIT_VERIFY,
            "verification_failed",
            error,
            "Regenerate the enhanced suite and rerun dashboard verification.",
        ) from error
    payload = {
        "command": "dashboard",
        "verified": True,
        "dashboard_path": dashboard_path,
    }
    return EXIT_OK, payload, [f"dashboard: {dashboard_path}"]


def _dispatch(arguments: argparse.Namespace) -> tuple[int, dict[str, object], list[str]]:
    if arguments.command == "preflight":
        return _command_preflight(arguments)
    if arguments.command == "run":
        return _command_run(arguments)
    if arguments.command == "verify":
        return _command_verify(arguments)
    if arguments.command == "dashboard":
        return _command_dashboard(arguments)
    raise _CliFailure(EXIT_CONFIG, "usage_error", "unknown command")


def _argument_sensitive_paths(arguments: argparse.Namespace | None) -> tuple[Path, ...]:
    if arguments is None:
        return ()
    paths = tuple(
        value
        for name in ("config", "output_dir", "artifacts_root", "run", "output")
        if isinstance((value := getattr(arguments, name, None)), Path)
    )
    config_path = getattr(arguments, "config", None)
    if isinstance(config_path, Path):
        try:
            paths += (load_experiment_config(config_path).source.path,)
        except Exception:
            pass
    return paths


def _emit(
    payload: object,
    lines: Sequence[str],
    *,
    json_mode: bool,
    error: bool = False,
    sensitive_paths: Sequence[Path] = (),
) -> None:
    if json_mode:
        print(
            json.dumps(
                _sanitize(payload, sensitive_paths=sensitive_paths),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    target = sys.stderr if error else sys.stdout
    for line in lines:
        print(_sanitize(line, sensitive_paths=sensitive_paths), file=target)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one command with a single redacting exception boundary."""

    raw = tuple(sys.argv[1:] if argv is None else argv)
    json_mode = "--json" in raw
    command: str | None = raw[0] if raw else None
    arguments: argparse.Namespace | None = None
    try:
        arguments = _parser().parse_args(raw)
        sensitive_paths = _argument_sensitive_paths(arguments)
        code, payload, lines = _dispatch(arguments)
        _emit(
            payload,
            lines,
            json_mode=bool(arguments.json),
            sensitive_paths=(
                sensitive_paths
                if code in {EXIT_CONFIG, EXIT_PREFLIGHT, EXIT_VERIFY, EXIT_RUNNER}
                else ()
            ),
        )
        return code
    except _HelpRequested as requested:
        if json_mode:
            _emit(
                {"command": "help", "help": requested.help_text},
                (),
                json_mode=True,
            )
        else:
            print(requested.help_text, end="")
        return EXIT_OK
    except _CliFailure as failure:
        sensitive_paths = _argument_sensitive_paths(arguments)
        payload = {"ok": False, "error": failure.kind, "detail": failure.detail, "remediation": failure.remediation}
        _emit(
            payload,
            [f"error: {failure.kind}: {failure.detail}", failure.remediation],
            json_mode=json_mode,
            error=True,
            sensitive_paths=sensitive_paths,
        )
        return failure.code
    except SystemExit as error:
        payload = {
            "ok": False,
            "error": "usage_error",
            "detail": "argument parsing terminated unexpectedly",
            "remediation": "Use --help and provide every required option.",
        }
        _emit(
            payload,
            ["error: usage_error: argument parsing terminated unexpectedly"],
            json_mode=json_mode,
            error=True,
        )
        return EXIT_CONFIG
    except Exception as error:
        sensitive_paths = _argument_sensitive_paths(arguments)
        if os.environ.get("WEBVIDEO_TO_DATA_DEVELOPER_TRACEBACK") == "1":
            formatted = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            redacted = redact_text(
                formatted,
                workspace=Path.cwd().resolve(),
                sensitive_paths=sensitive_paths,
            )
            print(redacted, file=sys.stderr, end="" if redacted.endswith("\n") else "\n")
        code = EXIT_VERIFY if command in {"verify", "dashboard"} else EXIT_RUNNER
        kind = "verification_failed" if code == EXIT_VERIFY else "runner_failed"
        payload = {"ok": False, "error": kind, "detail": str(error), "remediation": "Inspect the redacted diagnostic and retry."}
        _emit(
            payload,
            [f"error: {kind}: {error}", str(payload["remediation"])],
            json_mode=json_mode,
            error=True,
            sensitive_paths=sensitive_paths,
        )
        return code


if __name__ == "__main__":
    raise SystemExit(main())

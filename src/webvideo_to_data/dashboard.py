"""Dependency-free, public-safe rendering for verified experiment suites."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Literal, Mapping
from uuid import uuid4

if os.name == "nt":
    from ctypes import wintypes

from .artifacts import VerifiedRun, verify_run_directory
from .path_security import (
    absolute_filesystem_path,
    windows_path_for_containment,
)


_VARIANTS = ("B0", "B1", "B2", "B3", "B4")
_TITLES = {
    "B0": "B0 manual physics baseline",
    "B1": "B1 kinematic diagnostic",
    "B2": "B2 metric-depth diagnostic",
    "B3": "B3 metric-depth diagnostic",
    "B4": "B4 metric-depth diagnostic",
}


@dataclass(frozen=True)
class _DashboardInput:
    run_dir: Path
    metrics: Mapping[str, Any]
    environment: Mapping[str, Any]
    variants: Mapping[str, VerifiedRun]
    variant_provenance: Mapping[str, Mapping[str, Any]]
    variant_prefix: str
    suite_manifest: Mapping[str, Any] | None
    trusted_preview_variants: frozenset[str]
    detached_copy: bool


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_dashboard_input(
    run_dir: Path,
    *,
    verified_suite: object | None = None,
    trusted_preview_variants: frozenset[str] = frozenset(),
    detached_copy: bool = False,
) -> _DashboardInput:
    metrics = (
        dict(getattr(verified_suite, "metrics"))
        if verified_suite is not None
        else _json_object(run_dir / "suite-metrics.json")
    )
    environment = (
        dict(getattr(verified_suite, "environment"))
        if verified_suite is not None
        else _json_object(run_dir / "environment.json")
    )
    requested = metrics.get("requested_variants")
    summaries = metrics.get("variants")
    if (
        type(requested) is not list
        or any(type(item) is not str or item not in _VARIANTS for item in requested)
        or len(set(requested)) != len(requested)
        or type(summaries) is not dict
        or metrics.get("actions_exported") != 0
    ):
        raise ValueError("dashboard requires actionless suite metrics")
    suite_variants = getattr(verified_suite, "variant_runs", None)
    if isinstance(suite_variants, Mapping):
        variants = dict(suite_variants)
        variant_provenance = dict(
            getattr(verified_suite, "variant_provenance")
        )
        prefix = "variants"
    else:
        variants_root = run_dir / "variants"
        prefix = "variants" if variants_root.is_dir() else ""
        variants = {}
        variant_provenance = {}
        for variant in requested:
            directory = (variants_root / variant) if prefix else (run_dir / variant)
            verified = verify_run_directory(directory)
            variant_provenance[variant] = _json_object(
                directory / "provenance.json"
            )
            summary = summaries.get(variant)
            digest = sha256((directory / "run_manifest.json").read_bytes()).hexdigest()
            if (
                type(summary) is not dict
                or summary.get("status") != verified.metrics.get("status")
                or summary.get("reason") != verified.metrics.get("reason")
                or summary.get("run_manifest_sha256") != digest
            ):
                raise ValueError("dashboard variant summary mismatch")
            variants[variant] = verified
    manifest = (
        dict(getattr(verified_suite, "manifest"))
        if verified_suite is not None
        else None
    )
    return _DashboardInput(
        run_dir=run_dir,
        metrics=metrics,
        environment=environment,
        variants=variants,
        variant_provenance=variant_provenance,
        variant_prefix=prefix,
        suite_manifest=manifest,
        trusted_preview_variants=trusted_preview_variants,
        detached_copy=detached_copy,
    )


def _short(value: object) -> str:
    return str(value)[:12] if type(value) is str and value else "N/A"


def _value(value: object) -> str:
    if value is None:
        return "N/A"
    if type(value) is bool:
        return "YES" if value else "NO"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (str, int)):
        rendered = str(value).strip()
        return rendered if rendered else "N/A"
    return "N/A"


def _relative_media(input_data: _DashboardInput, variant: str) -> tuple[str | None, bool]:
    verified = input_data.variants.get(variant)
    if verified is None:
        return None, False
    private_present = False
    safe_name: str | None = None
    for name, entry in sorted(verified.manifest.get("files", {}).items()):
        if not isinstance(entry, Mapping) or Path(name).suffix.lower() not in {".gif", ".mp4", ".png"}:
            continue
        if entry.get("contains_private_source_frames") is True:
            private_present = True
            continue
        if entry.get("media_role") == "simulation_only" and Path(name).suffix.lower() == ".mp4":
            safe_name = name
            break
    if input_data.detached_copy:
        return None, private_present
    if input_data.suite_manifest is not None:
        suite_files = input_data.suite_manifest.get("files")
        preview_name = f"dashboard/media/{variant}-preview.gif"
        preview_entry = suite_files.get(preview_name) if isinstance(suite_files, Mapping) else None
        if isinstance(preview_entry, Mapping):
            if preview_entry.get("contains_private_source_frames") is True:
                private_present = True
            elif preview_entry.get("contains_private_source_frames") is False:
                return f"media/{variant}-preview.gif", private_present
    elif (
        variant in input_data.trusted_preview_variants
        and (input_data.run_dir / "dashboard" / "media" / f"{variant}-preview.gif").is_file()
    ):
        return f"media/{variant}-preview.gif", private_present
    if safe_name is None:
        return None, private_present
    pieces = [".."]
    if input_data.variant_prefix:
        pieces.append(input_data.variant_prefix)
    pieces.extend((variant, *Path(safe_name).parts))
    return "/".join(pieces), private_present


def _variant_card(input_data: _DashboardInput, variant: str) -> str:
    verified = input_data.variants.get(variant)
    summary = input_data.metrics.get("variants", {}).get(variant, {})
    generator_commit = _short(input_data.environment.get("generator_commit"))
    if verified is None:
        return f"""
        <article class="card muted">
          <h2>{escape(_TITLES[variant])}</h2>
          <div class="status">NOT REQUESTED — NOT ACTION DATA</div>
          <dl><dt>REASON</dt><dd>N/A</dd><dt>PHYSICS</dt><dd>N/A</dd>
          <dt>ACTION ELIGIBLE</dt><dd>NO</dd><dt>INPUT COMMIT</dt><dd>N/A</dd>
          <dt>GENERATOR COMMIT</dt><dd>{escape(generator_commit)}</dd>
          <dt>ARTIFACTS</dt><dd>N/A</dd><dt>HASHES</dt><dd>N/A</dd></dl>
          <p class="media-note">media not generated (variant not requested)</p>
        </article>"""
    metrics = verified.metrics
    manifest = verified.manifest
    provenance = input_data.variant_provenance.get(variant, {})
    input_commit = _short(
        provenance.get("generator", {}).get("git_commit")
        if isinstance(provenance.get("generator"), Mapping)
        else None
    )
    physics = (
        summary.get("physics_validation")
        if isinstance(summary, Mapping) and summary.get("physics_validation") is not None
        else metrics.get("physics_validation")
    )
    if variant == "B0" and input_data.metrics.get("b0_physics_baseline") in {"passed", "failed"}:
        physics = input_data.metrics["b0_physics_baseline"]
    action_eligible = metrics.get("action_export_eligible") is True
    status = f"{str(metrics.get('status', 'unknown')).upper()} — NOT ACTION DATA"
    hash_text = " · ".join(
        (
            f"config {_short(manifest.get('config_sha256'))}",
            f"source {_short(manifest.get('source_sha256'))}",
            f"model {_short(manifest.get('model_sha256'))}",
        )
    )
    media_path, private_present = _relative_media(input_data, variant)
    media_markup = ""
    if input_data.detached_copy:
        media_markup = (
            '<p class="media-note">media omitted from detached dashboard copy</p>'
        )
    elif media_path is not None:
        safe_path = escape(media_path, quote=True)
        if media_path.endswith(".gif"):
            media_markup = f'<img src="{safe_path}" alt="{escape(status)} simulation-only preview">'
        else:
            media_markup = f'<video controls muted loop preload="metadata" src="{safe_path}"></video>'
    else:
        media_markup = '<p class="media-note">media not generated (--no-render)</p>'
    if private_present:
        media_markup += '<p class="privacy">private local media omitted</p>'
    verdict = ""
    if variant == "B0" and str(physics).lower() == "passed" and not action_eligible:
        verdict = (
            '<p class="verdict"><strong>PHYSICS BASELINE PASSED</strong><br>'
            'REJECTED AS ACTION DATA — manual baseline is not video-grounded</p>'
        )
    warning = (
        '<p class="warning">availability != semantic accuracy</p>'
        if variant == "B1"
        else ""
    )
    return f"""
      <article class="card">
        <h2>{escape(_TITLES[variant])}</h2>
        <div class="status">{escape(status)}</div>
        {verdict}{warning}
        <p class="provenance">INPUT COMMIT · {escape(input_commit)}</p>
        <p class="provenance">GENERATOR COMMIT · {escape(generator_commit)}</p>
        <dl>
          <dt>REASON</dt><dd>{escape(_value(metrics.get('reason')))}</dd>
          <dt>PHYSICS</dt><dd>{escape(_value(physics).upper())}</dd>
          <dt>ACTION ELIGIBLE</dt><dd>{'YES' if action_eligible else 'NO'}</dd>
          <dt>ARTIFACTS</dt><dd>ARTIFACTS VERIFIED</dd>
          <dt>HASHES</dt><dd>{escape(hash_text)}</dd>
        </dl>
        {media_markup}
      </article>"""


def _dashboard_html(input_data: _DashboardInput) -> str:
    eligible = sum(
        int(run.metrics.get("action_export_eligible") is True)
        for run in input_data.variants.values()
    )
    successes = input_data.metrics.get("b0_successes")
    rollouts = input_data.metrics.get("b0_rollouts")
    physical = ""
    if isinstance(successes, int) and isinstance(rollouts, int):
        ruling = "PASSED" if input_data.metrics.get("b0_physics_baseline") == "passed" else "REJECTED"
        physical = f'<p class="physical">PHYSICAL BASELINE {ruling} · {successes} / {rollouts} rollouts</p>'
    cards = "\n".join(_variant_card(input_data, variant) for variant in _VARIANTS)
    verified_note = (
        "ARTIFACTS VERIFIED · detached copy · media intentionally omitted · no tracking scripts"
        if input_data.detached_copy
        else "ARTIFACTS VERIFIED · static local dashboard · no tracking scripts"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Experiment diagnostics — no action data</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b1016; --panel:#151d27; --line:#2b3948; --text:#eef4fa; --muted:#aab8c5; --danger:#ff6b6b; --amber:#ffc857; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; }}
    main {{ width:min(1180px,calc(100% - 32px)); margin:0 auto; padding:36px 0 60px; }}
    header {{ border:2px solid var(--danger); background:#241316; padding:24px; margin-bottom:22px; }}
    h1 {{ margin:0 0 8px; color:#fff; font-size:clamp(25px,4vw,46px); letter-spacing:-.04em; }}
    .lead {{ color:var(--danger); font-weight:800; font-size:clamp(17px,2.5vw,25px); margin:0; }}
    .physical {{ color:var(--amber); font-weight:700; margin:10px 0 0; }}
    .verified {{ color:#82e6ad; margin:10px 0 0; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }}
    .card {{ border:1px solid var(--line); background:var(--panel); padding:18px; min-width:0; }}
    .card h2 {{ margin:0 0 10px; font:700 18px/1.25 system-ui,sans-serif; }} .muted {{ opacity:.72; }}
    .status {{ display:inline-block; background:#7d1d28; color:#fff; font-weight:800; padding:6px 9px; margin-bottom:12px; }}
    dl {{ display:grid; grid-template-columns:max-content minmax(0,1fr); gap:5px 12px; margin:8px 0 14px; }}
    dt {{ color:var(--muted); }} dd {{ margin:0; overflow-wrap:anywhere; }}
    img,video {{ display:block; width:100%; height:auto; background:#101010; border:1px solid var(--line); }}
    .warning,.privacy,.media-note {{ color:var(--amber); }} .verdict {{ border-left:3px solid var(--amber); padding-left:10px; }}
  </style>
</head>
<body><main>
  <header>
    <h1>NO ACTION EXPORTED · {eligible} / 5 eligible</h1>
    <p class="lead">REJECTED — NOT ACTION DATA</p>
    {physical}
    <p class="verified">{verified_note}</p>
  </header>
  <section class="grid">{cards}</section>
</main></body>
</html>
"""


def _write_html(path: Path, html: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as temporary:
        temporary.write(html)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)
    return path


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError:
        return False
    return bool(attributes & 0x0400)  # FILE_ATTRIBUTE_REPARSE_POINT


def _external_output_path(
    suite_directory: Path, output_path: str | Path, description: str
) -> Path:
    suite_input = Path(suite_directory)
    output_input = Path(output_path)
    suite_lexical = absolute_filesystem_path(suite_input, "verified suite")
    lexical = absolute_filesystem_path(output_input, f"{description} output")
    for candidate in (lexical, *lexical.parents):
        if _is_link_or_junction(candidate):
            raise ValueError(f"{description} output links and junctions are forbidden")
    suite = suite_lexical.resolve(strict=False)
    output = lexical.resolve(strict=False)
    try:
        if os.name == "nt":
            windows_path_for_containment(output).relative_to(
                windows_path_for_containment(suite)
            )
        else:
            output.relative_to(suite)
    except ValueError:
        pass
    else:
        raise ValueError(f"{description} output must be outside the verified suite")
    if output.exists() and not output.is_file():
        raise ValueError(f"{description} output must be a regular file")
    return output


def _write_bytes_atomic(path: Path, content: bytes) -> Path:
    if os.name == "nt":
        return _write_bytes_atomic_windows(path, content)
    return _write_bytes_atomic_posix(path, content)


@contextmanager
def _stable_posix_parent(path: Path) -> Any:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise ValueError("stable no-follow output APIs are unavailable")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(path.anchor, flags)
    try:
        for part in path.parent.parts[1:]:
            try:
                child_fd = os.open(part, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                os.mkdir(part, mode=0o700, dir_fd=directory_fd)
                child_fd = os.open(part, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        yield directory_fd
    finally:
        os.close(directory_fd)


def _write_bytes_atomic_posix(path: Path, content: bytes) -> Path:
    temporary_name = f".{path.name}.{uuid4().hex}.tmp"
    with _stable_posix_parent(path) as parent_fd:
        descriptor: int | None = None
        installed = False
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("stable output write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            try:
                existing = os.stat(
                    path.name, dir_fd=parent_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                existing = None
            if existing is not None and not stat.S_ISREG(existing.st_mode):
                raise ValueError("output destination must be a regular file")
            os.replace(
                temporary_name,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            installed = True
            os.fsync(parent_fd)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if not installed:
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
    return path


if os.name == "nt":
    _OUTPUT_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _OUTPUT_WRITE_FILE = _OUTPUT_KERNEL32.WriteFile
    _OUTPUT_WRITE_FILE.argtypes = [
        wintypes.HANDLE,
        wintypes.LPCVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    _OUTPUT_WRITE_FILE.restype = wintypes.BOOL
    _OUTPUT_FLUSH_FILE = _OUTPUT_KERNEL32.FlushFileBuffers
    _OUTPUT_FLUSH_FILE.argtypes = [wintypes.HANDLE]
    _OUTPUT_FLUSH_FILE.restype = wintypes.BOOL
    _OUTPUT_CLOSE_HANDLE = _OUTPUT_KERNEL32.CloseHandle
    _OUTPUT_CLOSE_HANDLE.argtypes = [wintypes.HANDLE]
    _OUTPUT_CLOSE_HANDLE.restype = wintypes.BOOL


@contextmanager
def _stable_windows_parent(path: Path) -> Any:
    from . import redaction as redaction_module

    handles: list[Any] = []
    try:
        existing = path.parent
        missing: list[str] = []
        while not existing.exists():
            if existing.is_symlink() or existing == existing.parent:
                raise ValueError(
                    "output parent link, junction, or reparse point is forbidden"
                )
            missing.append(existing.name)
            existing = existing.parent
        try:
            root_handle, _ = redaction_module._win_open_root(existing)
        except OSError as error:
            raise ValueError(
                "output parent link, junction, or reparse point is forbidden"
            ) from error
        handles.append(root_handle)
        current = existing
        for part in reversed(missing):
            current /= part
            try:
                child_handle, _ = redaction_module._win_create_child(
                    handles[-1], part, directory=True
                )
            except OSError as open_error:
                raise ValueError(
                    "output parent could not be created handle-relatively"
                ) from open_error
            handles.append(child_handle)
        yield handles[-1]
    finally:
        for handle in reversed(handles):
            redaction_module._CLOSE_HANDLE(handle)


def _write_bytes_atomic_windows(path: Path, content: bytes) -> Path:
    if os.name != "nt":
        raise ValueError("Windows stable output APIs are unavailable")
    temporary_name = f".{path.name}.{uuid4().hex}.tmp"
    with _stable_windows_parent(path) as parent_handle:
        from . import redaction as redaction_module

        handle, _ = redaction_module._win_create_child(
            parent_handle, temporary_name, directory=False
        )
        installed = False
        try:
            offset = 0
            while offset < len(content):
                chunk = content[offset : offset + 1024 * 1024]
                buffer = ctypes.create_string_buffer(chunk)
                written = wintypes.DWORD()
                if not _OUTPUT_WRITE_FILE(
                    handle,
                    buffer,
                    len(chunk),
                    ctypes.byref(written),
                    None,
                ):
                    raise OSError(ctypes.get_last_error(), "stable output write failed")
                if written.value <= 0:
                    raise OSError("stable output write made no progress")
                offset += written.value
            if not _OUTPUT_FLUSH_FILE(handle):
                raise OSError(ctypes.get_last_error(), "stable output flush failed")
            redaction_module._win_rename_child(
                handle, parent_handle, path.name
            )
            installed = True
        finally:
            try:
                if not installed:
                    redaction_module._win_delete_handle(handle)
            finally:
                _OUTPUT_CLOSE_HANDLE(handle)
    return path


def _build_dashboard(
    run_dir: Path,
    *,
    trusted_preview_variants: frozenset[str] = frozenset(),
) -> Path:
    input_data = _load_dashboard_input(
        run_dir,
        trusted_preview_variants=trusted_preview_variants,
    )
    return _write_html(run_dir / "dashboard" / "index.html", _dashboard_html(input_data))


def generate_dashboard(run_dir: str | Path) -> Path:
    """Validate a finalized suite dashboard and return its location.

    The returned path is a display/location value, not a trusted content
    capability. Use ``write_dashboard_copy`` to consume verified bytes.
    """

    directory = absolute_filesystem_path(run_dir, "verified suite")
    from .suite import _verified_suite_capability

    with _verified_suite_capability(directory) as verified:
        output = directory / "dashboard" / "index.html"
        entry = verified.manifest.get("files", {}).get("dashboard/index.html")
        content = verified.captured_files.get("dashboard/index.html")
        if not isinstance(entry, Mapping) or not isinstance(content, bytes):
            raise ValueError("verified suite does not contain a dashboard")
    return output


def write_dashboard_copy(run_dir: str | Path, output_path: str | Path) -> Path:
    """Write a separate dashboard copy after verifying the immutable source suite."""

    directory = absolute_filesystem_path(run_dir, "verified suite")
    lexical_output = absolute_filesystem_path(
        output_path, "dashboard copy output"
    )
    from .suite import _verified_suite_capability

    with _verified_suite_capability(directory) as verified:
        output = _external_output_path(directory, lexical_output, "dashboard copy")
        input_data = _load_dashboard_input(
            directory, verified_suite=verified, detached_copy=True
        )
        content = _dashboard_html(input_data).encode("utf-8")
        from . import redaction as redaction_module

        findings = redaction_module.audit_publication_bytes(
            "dashboard.html", content
        )
        if findings:
            raise ValueError("detached dashboard privacy audit failed")
    return _write_bytes_atomic(output, content)


def copy_public_preview(
    run_dir: str | Path,
    variant: Literal["B0", "B1"],
    output_path: str | Path,
) -> Path:
    """Copy one suite-verified simulation-only GIF to an external public path."""

    if variant not in {"B0", "B1"}:
        raise ValueError("public preview variant must be B0 or B1")
    directory = absolute_filesystem_path(run_dir, "verified suite")
    lexical_output = absolute_filesystem_path(
        output_path, "public preview output"
    )
    from .suite import _verified_suite_capability

    with _verified_suite_capability(directory) as verified:
        name = f"dashboard/media/{variant}-preview.gif"
        requested = verified.manifest.get("requested_variants")
        entry = verified.manifest.get("files", {}).get(name)
        if (
            not isinstance(requested, list)
            or variant not in requested
            or not isinstance(entry, Mapping)
        ):
            raise ValueError(f"{variant} preview is not present in verified suite")
        if entry.get("contains_private_source_frames") is not False:
            raise ValueError(f"{variant} preview contains private source frames")
        if entry.get("media_role") != "public_simulation_preview":
            raise ValueError(f"{variant} preview is not a public simulation preview")
        content = verified.captured_files.get(name)
        if not isinstance(content, bytes):
            raise ValueError("public preview is absent from the captured suite")
        output = _external_output_path(directory, lexical_output, "public preview")
    return _write_bytes_atomic(output, content)

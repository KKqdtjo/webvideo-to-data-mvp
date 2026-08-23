"""Strict, immutable registry of locally available experiment sources."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from types import MappingProxyType
from typing import Mapping

import yaml
from yaml.nodes import MappingNode


_SOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORD_FIELDS = {
    "id",
    "path",
    "sha256",
    "origin",
    "captured_on",
    "captured_on_status",
    "license",
    "publishable",
    "privacy_review",
    "access",
}


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys in every mapping."""

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[object, object]:
        if not isinstance(node, MappingNode):
            raise ValueError("YAML mapping node is invalid")
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as error:
                raise ValueError("YAML mapping keys must be hashable") from error
            if duplicate:
                raise ValueError(f"duplicate YAML key: {key}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


@dataclass(frozen=True)
class SourceRecord:
    id: str
    path: Path
    sha256: str
    origin: str
    captured_on: str | None
    captured_on_status: str
    license: str
    publishable: bool
    privacy_review: str
    access: str


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return dict(value)


def _exact_fields(mapping: Mapping[str, object], fields: set[str], name: str) -> None:
    if set(mapping) != fields:
        raise ValueError(f"{name} fields must be exactly: {', '.join(sorted(fields))}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _parse_record(
    value: object, registry_dir: Path, trusted_root: Path
) -> SourceRecord:
    raw = _mapping(value, "source record")
    _exact_fields(raw, _RECORD_FIELDS, "source record")
    source_id = _string(raw["id"], "source.id")
    if _SOURCE_ID.fullmatch(source_id) is None:
        raise ValueError("source.id is invalid")
    relative_text = _string(raw["path"], "source.path")
    relative = Path(relative_text)
    windows_path = PureWindowsPath(relative_text)
    if (
        relative.is_absolute()
        or PurePosixPath(relative_text).is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
    ):
        raise ValueError("source.path must be relative to the registry")
    if relative == Path("."):
        raise ValueError("source.path must name a file")
    digest = _string(raw["sha256"], "source.sha256")
    if _SHA256.fullmatch(digest) is None:
        raise ValueError("source.sha256 must be lowercase SHA-256 hex")
    origin = _string(raw["origin"], "source.origin")
    if re.fullmatch(r"[a-z][a-z0-9_]*", origin) is None:
        raise ValueError("source.origin is invalid")
    captured_on = raw["captured_on"]
    if captured_on is not None:
        captured_on = _string(captured_on, "source.captured_on")
        try:
            if date.fromisoformat(captured_on).isoformat() != captured_on:
                raise ValueError
        except ValueError as error:
            raise ValueError("source.captured_on must be an ISO date or null") from error
    captured_status = _string(raw["captured_on_status"], "source.captured_on_status")
    if captured_status not in {"recorded", "not_recorded"}:
        raise ValueError("source.captured_on_status is invalid")
    if (captured_on is None) != (captured_status == "not_recorded"):
        raise ValueError("source capture date and status disagree")
    publishable = raw["publishable"]
    if type(publishable) is not bool:
        raise ValueError("source.publishable must be a boolean")
    resolved_path = (registry_dir / relative).resolve()
    try:
        resolved_path.relative_to(trusted_root)
    except ValueError as error:
        raise ValueError("source.path escapes the trusted project root") from error
    return SourceRecord(
        id=source_id,
        path=resolved_path,
        sha256=digest,
        origin=origin,
        captured_on=captured_on,
        captured_on_status=captured_status,
        license=_string(raw["license"], "source.license"),
        publishable=publishable,
        privacy_review=_string(raw["privacy_review"], "source.privacy_review"),
        access=_string(raw["access"], "source.access"),
    )


def load_source_registry(
    path: str | Path, *, trusted_root: str | Path | None = None
) -> Mapping[str, SourceRecord]:
    """Load an exact schema-v1 registry and return a read-only ID mapping."""

    registry_path = Path(path).resolve()
    root = (
        Path(trusted_root).resolve()
        if trusted_root is not None
        else (
            registry_path.parent.parent.resolve()
            if registry_path.parent.name.lower() == "configs"
            else registry_path.parent.resolve()
        )
    )
    if not root.is_dir():
        raise ValueError("trusted source registry root must be a directory")
    try:
        document = _mapping(
            yaml.load(
                registry_path.read_text(encoding="utf-8"),
                Loader=_UniqueKeySafeLoader,
            ),
            "source registry",
        )
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError("source registry could not be read") from error
    _exact_fields(document, {"schema_version", "sources"}, "source registry")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("source registry schema_version must be 1")
    sources = document["sources"]
    if not isinstance(sources, list):
        raise ValueError("source registry sources must be a list")
    records: dict[str, SourceRecord] = {}
    for value in sources:
        record = _parse_record(value, registry_path.parent, root)
        if record.id in records:
            raise ValueError("source registry IDs must be unique")
        records[record.id] = record
    return MappingProxyType(records)

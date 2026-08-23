from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from webvideo_to_data.source_registry import load_source_registry


def _write_registry(path: Path, sources: list[dict[str, object]]) -> Path:
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "sources": sources}, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "fixture-source-v0",
        "path": "media/fixture.mp4",
        "sha256": "a" * 64,
        "origin": "user_recorded",
        "captured_on": None,
        "captured_on_status": "not_recorded",
        "license": "private_not_redistributable",
        "publishable": False,
        "privacy_review": "local_only",
        "access": "local file required; not distributed",
    }
    record.update(overrides)
    return record


def test_private_source_record_is_complete_without_inventing_capture_date() -> None:
    records = load_source_registry(Path("configs/sources.yaml"))
    record = records["exp001-phone-can-private"]
    assert record.id == "exp001-phone-can-private"
    assert record.path == (Path("configs") / "source-placeholder.mp4").resolve()
    assert record.sha256 == "0" * 64
    assert record.origin == "user_recorded"
    assert record.captured_on is None
    assert record.captured_on_status == "not_recorded"
    assert record.license == "private_not_redistributable"
    assert record.publishable is False
    assert record.privacy_review == "local_only"
    assert record.access == "local file required; not distributed"


def test_registry_mapping_and_records_are_deeply_immutable(tmp_path: Path) -> None:
    records = load_source_registry(_write_registry(tmp_path / "sources.yaml", [_record()]))
    with pytest.raises(TypeError):
        records["new"] = records["fixture-source-v0"]  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        records["fixture-source-v0"].origin = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "document",
    [
        {"schema_version": 1, "sources": [], "extra": True},
        {"schema_version": 2, "sources": []},
        {"schema_version": 1},
        {"schema_version": 1, "sources": [_record(extra=True)]},
        {"schema_version": 1, "sources": [_record(path="C:/private/source.mp4")]},
        {"schema_version": 1, "sources": [_record(sha256="A" * 64)]},
        {
            "schema_version": 1,
            "sources": [_record(), _record(path="media/other.mp4")],
        },
        {
            "schema_version": 1,
            "sources": [_record(captured_on="2026-01-02", captured_on_status="not_recorded")],
        },
    ],
)
def test_registry_rejects_non_exact_or_ambiguous_documents(
    tmp_path: Path, document: dict[str, object]
) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_source_registry(path)


@pytest.mark.parametrize(
    "document",
    [
        "schema_version: 1\nschema_version: 1\nsources: []\n",
        """schema_version: 1
sources:
  - id: fixture-source-v0
    path: media/fixture.mp4
    sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    origin: user_recorded
    origin: user_recorded
    captured_on: null
    captured_on_status: not_recorded
    license: private_not_redistributable
    publishable: false
    privacy_review: local_only
    access: local file required; not distributed
""",
    ],
)
def test_registry_rejects_duplicate_keys_at_every_mapping_level(
    tmp_path: Path, document: str
) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_source_registry(path)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "\\private\\source.mp4",
        "D:private\\source.mp4",
        "D:\\private\\source.mp4",
        "\\\\server\\share\\source.mp4",
        "/mnt/private/source.mp4",
        "../../outside-project.mp4",
    ],
)
def test_registry_rejects_rooted_or_project_escaping_source_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    project = tmp_path / "project"
    configs = project / "configs"
    configs.mkdir(parents=True)
    path = _write_registry(configs / "sources.yaml", [_record(path=unsafe_path)])
    with pytest.raises(ValueError, match="source.path"):
        load_source_registry(path)


def test_registry_allows_parent_path_that_stays_inside_project_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    configs = project / "configs"
    configs.mkdir(parents=True)
    records = load_source_registry(
        _write_registry(configs / "sources.yaml", [_record(path="../video/source.mp4")])
    )
    assert records["fixture-source-v0"].path == (project / "video/source.mp4").resolve()


def test_registry_keeps_yaml_safe_tag_restrictions(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(
        "schema_version: 1\nsources: !!python/object/apply:builtins.list []\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_source_registry(path)

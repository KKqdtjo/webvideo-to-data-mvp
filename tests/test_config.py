from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pytest
import yaml

from tests.helpers import write_complete_config
from webvideo_to_data.config import load_experiment_config, to_public_resolved_mapping


def test_loads_complete_schema_v2_config(tmp_path: Path) -> None:
    """Catch a loader that fails to preserve configured physics parameters."""
    path = write_complete_config(tmp_path)
    config = load_experiment_config(path)
    assert config.schema_version == 2
    assert config.ik.position_tolerance_m == pytest.approx(0.005)
    assert config.ik.orientation_tolerance_rad == pytest.approx(np.deg2rad(5.0))
    assert config.ik.maximum_iterations == 200
    assert config.collision.maximum_penetration_m == pytest.approx(0.002)
    assert config.perturbation.rollout_count == 30


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"unexpected": 1}, "unknown experiment config fields: unexpected"),
        ({"experiment_id": "../escape"}, "experiment_id must match"),
        ({"source.roi_xywh": [1, 2, -3, 4]}, "roi width and height must be positive"),
        ({"ik.maximum_iterations": 0}, "maximum_iterations must be positive"),
        ({"perturbation.rollout_count": 29}, "rollout_count must be 30"),
    ],
)
def test_rejects_invalid_or_unknown_values(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    """Catch accepting an ambiguous experiment condition."""
    path = write_complete_config(tmp_path, mutation)
    with pytest.raises(ValueError, match=re.escape(message)):
        load_experiment_config(path)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_numbers(tmp_path: Path, invalid: float) -> None:
    """Catch non-finite physics parameters reaching resolved configuration."""
    path = write_complete_config(tmp_path, {"ik.damping": invalid})
    with pytest.raises(ValueError, match="finite"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"tracking.forward_backward_threshold_px": -0.1}, "must be nonnegative"),
        ({"tracking.minimum_valid_ratio": -0.1}, "must be between 0 and 1"),
        ({"tracking.minimum_valid_ratio": 1.01}, "must be between 0 and 1"),
        ({"perturbation.mass_fraction": 1.01}, "must be between 0 and 1"),
        ({"perturbation.friction_fraction": -0.1}, "must be between 0 and 1"),
    ],
)
def test_rejects_out_of_range_tracking_and_perturbation_values(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    """Catch impossible tracking thresholds or physical perturbation fractions."""
    with pytest.raises(ValueError, match=message):
        load_experiment_config(write_complete_config(tmp_path, mutation))


@pytest.mark.parametrize(
    "source_id",
    ["../escape", "C:\\source", "https://user:pass@example.test", "source?token=x", "CON"],
)
def test_rejects_unsafe_source_ids(tmp_path: Path, source_id: str) -> None:
    """Catch a public source identifier becoming a path or credential carrier."""
    with pytest.raises(ValueError, match="source.id must match"):
        load_experiment_config(write_complete_config(tmp_path, {"source.id": source_id}))


def test_rejects_missing_required_group(tmp_path: Path) -> None:
    """Catch a config silently defaulting an omitted collision policy."""
    path = write_complete_config(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    del raw["collision"]
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="missing experiment config fields: collision"):
        load_experiment_config(path)


def test_resolves_source_relative_to_config_without_changing_public_id(tmp_path: Path) -> None:
    """Catch a relative source path being resolved from the process directory."""
    path = write_complete_config(tmp_path)
    config = load_experiment_config(path)
    assert config.source.path == (path.parent / "moving.avi").resolve()
    assert config.source.id == "synthetic-moving-object"


def test_public_config_mapping_contains_no_absolute_paths(tmp_path: Path) -> None:
    """Catch resolved local paths leaking into public provenance."""
    config = load_experiment_config(write_complete_config(tmp_path))
    public = to_public_resolved_mapping(config)
    encoded = json.dumps(public, sort_keys=True)
    assert str(tmp_path.resolve()) not in encoded
    assert public["source"]["path"] == "registry:synthetic-moving-object"


def test_public_config_mapping_uses_only_safe_source_id(tmp_path: Path) -> None:
    """Catch public registry serialization preserving a raw local source path."""
    config = load_experiment_config(
        write_complete_config(tmp_path, {"source.id": "lowercase-source-id"})
    )
    public = to_public_resolved_mapping(config)
    assert public["source"]["id"] == "lowercase-source-id"
    assert public["source"]["path"] == "registry:lowercase-source-id"

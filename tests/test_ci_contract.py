from __future__ import annotations

from pathlib import Path
import re
import tomllib


REPOSITORY_ROOT = Path(__file__).parents[1]


def test_pytest_pins_repository_root_for_console_script_collection() -> None:
    """Would catch ``uv run pytest`` losing the repository-root test package."""
    pyproject = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert pyproject["tool"]["pytest"]["ini_options"].get("pythonpath") == ["."]


def test_ci_pins_actions_and_verifies_built_wheel_assets() -> None:
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    uses = re.findall(r"^\s*- uses: ([^\s#]+)(?:\s+#\s*(v[^\s]+))?\s*$", workflow, re.MULTILINE)

    assert uses
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", action) for action, _ in uses)
    assert all(version for _, version in uses)
    assert "uv build --wheel --out-dir dist" in workflow
    assert (
        "uv run python tests/verify_wheel_assets.py "
        "dist/webvideo_to_data-0.1.0-py3-none-any.whl"
    ) in workflow

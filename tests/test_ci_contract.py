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


def test_ci_installs_runtime_dependencies_and_uses_osmesa_on_ubuntu() -> None:
    """Would catch CI runners missing FFmpeg or a usable MuJoCo GL backend."""
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert re.search(
        r"- name: Install runtime dependencies \(Windows\)\n"
        r"\s+if: runner\.os == 'Windows'\n"
        r"\s+run: choco install ffmpeg --yes --no-progress",
        workflow,
    )
    assert re.search(
        r"- name: Install runtime dependencies \(Ubuntu\)\n"
        r"\s+if: runner\.os == 'Linux'\n"
        r"\s+run: \|\n"
        r"\s+sudo apt-get update\n"
        r"\s+sudo apt-get install --yes --no-install-recommends ffmpeg libosmesa6",
        workflow,
    )
    assert re.search(
        r"- name: Run public tests \(Ubuntu\).*?\n"
        r"\s+if: runner\.os == 'Linux'\n"
        r"\s+env:\n"
        r"\s+MUJOCO_GL: osmesa",
        workflow,
        re.DOTALL,
    )


def test_ci_runs_renderer_tests_only_on_the_renderer_capable_runner() -> None:
    """Keep Windows coverage broad without claiming hosted-runner OpenGL support."""
    pyproject = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    markers = pyproject["tool"]["pytest"]["ini_options"]["markers"]
    assert any(marker.startswith("requires_renderer:") for marker in markers)
    assert re.search(
        r"- name: Run public tests \(Windows\).*?\n"
        r"\s+if: runner\.os == 'Windows'\n"
        r'\s+run: uv run pytest -m "not acceptance and not private_video '
        r'and not requires_renderer" -q -p no:cacheprovider',
        workflow,
        re.DOTALL,
    )
    ubuntu_step = re.search(
        r"- name: Run public tests \(Ubuntu\)(.*?)\n\s+- run: git diff --check",
        workflow,
        re.DOTALL,
    )
    assert ubuntu_step
    assert 'pytest -m "not acceptance and not private_video"' in ubuntu_step.group(1)
    assert "not requires_renderer" not in ubuntu_step.group(1)

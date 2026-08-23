# Phase 2A Trustworthy Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independently valid Franka/MuJoCo manual pick-and-place baseline, append-only experiment entry point, versioned artifacts, and an honest static dashboard without exporting video action data.

**Architecture:** Keep `run_experiment()` as the safe single-variant publication primitive, but split configuration, scene loading, IK, validation, artifact serialization, suite orchestration, preflight, and reporting into focused modules. B0 bypasses video tracking and evaluates a manual reference on the pinned Menagerie Panda; B1-B4 keep their current diagnostic semantics. A new append-only suite runner composes variant runs, a 30-seed B0 benchmark, verification, and dashboard generation.

**Tech Stack:** Python 3.11, NumPy, SciPy, MuJoCo 3.x, OpenCV, PyYAML, stdlib HTML/JSON, pytest, uv, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-19-phase2a-trustworthy-baseline-design.md`

## Global Constraints

- Python is exactly 3.11; public tests run on Windows and Ubuntu.
- The runtime dependency set stays lightweight. Phase 2A does not add SAM2, CoTracker, TAP, Depth Pro, Metric3D, FoundationPose, Isaac Lab, V2D, or CHORD.
- Vendor MuJoCo Menagerie `franka_emika_panda` at upstream commit `da76818e269b82289eba39808e2fb91d679d6994`, preserving its Apache-2.0 LICENSE and recording every adapted file.
- B0 uses 5 mm position tolerance, 5 degree orientation tolerance, at most 200 IK iterations, and 95% execution tracking as the minimum.
- The 30-seed evaluation perturbs initial XY by ±10 mm, yaw by ±5 degrees, mass by ±10%, and friction by ±10%; at least 24 of 30 rollouts must pass.
- A rollout needs at least 50 mm lift, final target-center error at most 40 mm, settle time 1 second, final tilt at most 15 degrees, and final linear speed at most 0.02 m/s.
- Any forbidden contact with distance below `-0.002 m` is an illegal penetration.
- A successful manual B0 remains action-ineligible: terminal `status="rejected"`, `reason="manual_baseline_not_video_grounded"`, `physics_validation="passed"`, no `actions.npz`, and an explicitly named `baseline_control_trace.npz`.
- B1 remains `kinematic_replay_not_action`; B2-B4 remain `not_run/metric_depth_not_available`.
- Existing publication locking, trusted snapshots, rollback, quarantine, and no-recursive-delete behavior must stay covered by regression tests.
- Private source video, absolute local paths, `github_token.txt`, Seedance credentials, Authorization values, signed URL queries, cookies, and SSH details never enter Git, artifacts, logs, HTML, or test fixtures.
- Media keep the source aspect ratio. Rejected media show `REJECTED — NOT ACTION DATA`; B1 also shows `KINEMATIC OBJECT-POSE OVERRIDE`; time-warped comparisons show both source and simulation clocks.
- Public CI excludes private-video tests and the full 30-seed acceptance run; both are run and recorded locally before completion.

## File structure

| Path | Responsibility |
| --- | --- |
| `src/webvideo_to_data/config.py` | Versioned strict YAML contracts and resolved configuration |
| `src/webvideo_to_data/scene.py` | Pinned Panda scene loading, stable model-name mapping, perturbation application |
| `src/webvideo_to_data/ik.py` | Offline pose IK and joint-control program generation |
| `src/webvideo_to_data/physics_validation.py` | Allowed-contact policy, collision/penetration classification, rollout verdict |
| `src/webvideo_to_data/artifacts.py` | NPZ sidecar schema v1, manifest v4, verified loaders |
| `src/webvideo_to_data/preflight.py` | Read-only environment, source, ffprobe, model, and renderer checks |
| `src/webvideo_to_data/redaction.py` | Secret/path/query redaction for user-visible and persisted errors |
| `src/webvideo_to_data/source_registry.py` | Private/public source metadata and hash validation |
| `src/webvideo_to_data/suite.py` | Append-only run IDs, variant orchestration, latest pointer, 30-seed B0 summary |
| `src/webvideo_to_data/dashboard.py` | Sanitized static HTML report generation |
| `src/webvideo_to_data/cli.py` | `preflight`, `run`, `verify`, and `dashboard` commands and exit codes |
| `src/webvideo_to_data/simulation.py` | MuJoCo execution and measured state history only |
| `src/webvideo_to_data/retargeting.py` | Manual B0 Cartesian phase reference and legacy B1 reference |
| `src/webvideo_to_data/experiment.py` | Single-variant stage orchestration and transactional publication |
| `src/webvideo_to_data/visualization.py` | Aspect-preserving frames, status banners, clocks, tracking overlay |
| `src/webvideo_to_data/assets/mujoco_menagerie/franka_emika_panda/` | Pinned upstream Panda assets, license, metadata, adapted model, EXP-001 scene |
| `configs/exp001.yaml` | Schema-v2 experiment parameters |
| `configs/sources.yaml` | Source registry; private video remains local-only |
| `tests/` | Unit, integration, publication-safety, CLI, dashboard, and acceptance tests |

---

### Task 1: Strict schema-v2 configuration and locked environment

**Files:**
- Create: `src/webvideo_to_data/config.py`
- Create: `tests/test_config.py`
- Create: `tests/helpers.py`
- Create: `uv.lock`
- Modify: `configs/exp001.yaml`
- Modify: `src/webvideo_to_data/experiment.py:93-171`
- Modify: `tests/test_experiment.py:40-104`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `load_experiment_config(path: str | Path) -> ExperimentConfig`
- Produces: `to_public_resolved_mapping(config: ExperimentConfig) -> Mapping[str, object]`
- Produces: immutable `SourceConfig`, `TrackingConfig`, `SceneConfig`, `IKConfig`, `ControlConfig`, `CollisionConfig`, `PerturbationConfig`, `MediaConfig`, `SimulationConfig`, and `ExperimentConfig`
- Preserves: `webvideo_to_data.experiment.load_experiment_config` as a compatibility import

- [ ] **Step 1: Write failing strict-config tests**

Add a single shared complete config writer to `tests/helpers.py`. It deep-copies this literal mapping, applies dotted-key overrides by walking nested dictionaries, writes a five-frame 96×72 MJPG fixture when `with_source=True`, replaces `source.sha256` with `sha256_file(source)`, and writes YAML to `tmp_path / "config.yaml"`:

```python
VALID_CONFIG = {
    "schema_version": 2,
    "experiment_id": "SYNTHETIC",
    "source": {
        "id": "synthetic-moving-object",
        "path": "moving.avi",
        "sha256": "set-by-writer",
        "fps": 10.0,
        "roi_xywh": [20, 20, 16, 20],
    },
    "tracking": {
        "forward_backward_threshold_px": 1.5,
        "minimum_live_points": 4,
        "minimum_valid_ratio": 0.5,
    },
    "scene": {
        "x_bounds_m": [-0.15, 0.15],
        "y_bounds_m": [0.35, 0.65],
        "b0_start_m": [0.12, 0.45, 0.045],
        "b0_goal_m": [-0.05, 0.55, 0.115],
        "grasp_quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
    },
    "ik": {
        "position_tolerance_m": 0.005,
        "orientation_tolerance_deg": 5.0,
        "maximum_iterations": 200,
        "damping": 0.05,
        "step_size": 0.2,
        "orientation_weight": 0.25,
        "joint_limit_weight": 0.05,
    },
    "control": {
        "control_hz": 100.0,
        "maximum_joint_velocity_rad_s": 2.0,
        "maximum_joint_acceleration_rad_s2": 8.0,
        "gripper_open_width_m": 0.08,
        "gripper_closed_width_m": 0.0,
        "phase_duration_s": {
            "home": 0.5, "pregrasp": 0.8, "approach": 0.6, "close": 0.8,
            "lift": 0.8, "transport": 1.2, "lower": 0.6, "open": 0.6,
            "retreat": 0.6, "settle": 1.0,
        },
    },
    "collision": {
        "maximum_penetration_m": 0.002,
        "minimum_lift_m": 0.05,
        "maximum_target_error_m": 0.04,
        "settle_duration_s": 1.0,
        "maximum_final_tilt_deg": 15.0,
        "maximum_final_linear_speed_m_s": 0.02,
        "minimum_bilateral_contact_duration_s": 0.2,
        "minimum_lift_contact_duration_s": 0.1,
        "allowed_contact_pairs": {
            "home": [["can", "table"]],
            "pregrasp": [["can", "table"]],
            "approach": [["can", "table"]],
            "close": [["can", "table"], ["left_finger", "can"], ["right_finger", "can"]],
            "lift": [["left_finger", "can"], ["right_finger", "can"]],
            "transport": [["left_finger", "can"], ["right_finger", "can"]],
            "lower": [["left_finger", "can"], ["right_finger", "can"]],
            "open": [["can", "box"]],
            "retreat": [["can", "box"]],
            "settle": [["can", "box"]],
        },
    },
    "perturbation": {
        "rollout_count": 30,
        "xy_half_range_m": 0.01,
        "yaw_half_range_deg": 5.0,
        "mass_fraction": 0.10,
        "friction_fraction": 0.10,
    },
    "simulation": {
        "b0_mode": "physics_grasp",
        "b1_mode": "kinematic_replay",
        "render_size": [320, 320],
        "render_every": 50,
    },
    "media": {
        "canvas_size": [960, 320],
        "panel_size": [320, 320],
        "letterbox_bgr": [16, 16, 16],
        "output_fps": 30.0,
        "comparison_alignment": "time_warped",
    },
    "random_seed": 19,
}
```

Name the helper `write_complete_config(tmp_path: Path, overrides: Mapping[str, object] | None = None, with_source: bool = True) -> Path`. Add literal fixtures to `tests/test_config.py`. The break each test catches is accepting an ambiguous experiment condition.

```python
def test_loads_complete_schema_v2_config(tmp_path: Path) -> None:
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
    path = write_complete_config(tmp_path, mutation)
    with pytest.raises(ValueError, match=re.escape(message)):
        load_experiment_config(path)


def test_rejects_missing_required_group(tmp_path: Path) -> None:
    path = write_complete_config(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    del raw["collision"]
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="missing experiment config fields: collision"):
        load_experiment_config(path)


def test_resolves_source_relative_to_config_without_changing_public_id(tmp_path: Path) -> None:
    path = write_complete_config(tmp_path)
    config = load_experiment_config(path)
    assert config.source.path == (path.parent / "moving.avi").resolve()
    assert config.source.id == "synthetic-moving-object"


def test_public_config_mapping_contains_no_absolute_paths(tmp_path: Path) -> None:
    config = load_experiment_config(write_complete_config(tmp_path))
    public = to_public_resolved_mapping(config)
    encoded = json.dumps(public, sort_keys=True)
    assert str(tmp_path.resolve()) not in encoded
    assert public["source"]["path"] == "registry:synthetic-moving-object"
```

Update the synthetic config writer in `tests/test_experiment.py` to emit the same complete schema. Do not add permissive defaults to production merely to keep the old fixture short.

- [ ] **Step 2: Verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_experiment.py -q -p no:cacheprovider
```

Expected: collection fails because `webvideo_to_data.config` does not exist, then the strict validation assertions fail until the loader replaces the permissive parser.

- [ ] **Step 3: Implement typed configuration**

Use a small explicit mapping helper; do not silently ignore keys.

```python
def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _take(mapping: dict[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class IKConfig:
    position_tolerance_m: float
    orientation_tolerance_rad: float
    maximum_iterations: int
    damping: float
    step_size: float
    orientation_weight: float
    joint_limit_weight: float
```

The remaining immutable dataclasses have these exact fields and types; derived degree-to-radian values are computed once while parsing:

```python
@dataclass(frozen=True)
class SourceConfig:
    id: str
    path: Path
    sha256: str
    fps: float
    roi_xywh: tuple[int, int, int, int]

@dataclass(frozen=True)
class TrackingConfig:
    forward_backward_threshold_px: float
    minimum_live_points: int
    minimum_valid_ratio: float

@dataclass(frozen=True)
class SceneConfig:
    x_bounds_m: tuple[float, float]
    y_bounds_m: tuple[float, float]
    b0_start_m: tuple[float, float, float]
    b0_goal_m: tuple[float, float, float]
    grasp_quaternion_wxyz: tuple[float, float, float, float]

@dataclass(frozen=True)
class ControlConfig:
    control_hz: float
    maximum_joint_velocity_rad_s: float
    maximum_joint_acceleration_rad_s2: float
    gripper_open_width_m: float
    gripper_closed_width_m: float
    phase_duration_s: Mapping[str, float]

@dataclass(frozen=True)
class CollisionConfig:
    maximum_penetration_m: float
    minimum_lift_m: float
    maximum_target_error_m: float
    settle_duration_s: float
    maximum_final_tilt_rad: float
    maximum_final_linear_speed_m_s: float
    minimum_bilateral_contact_duration_s: float
    minimum_lift_contact_duration_s: float
    allowed_contact_pairs: Mapping[str, tuple[tuple[str, str], ...]]

@dataclass(frozen=True)
class PerturbationConfig:
    rollout_count: int
    xy_half_range_m: float
    yaw_half_range_rad: float
    mass_fraction: float
    friction_fraction: float

@dataclass(frozen=True)
class SimulationConfig:
    b0_mode: str
    b1_mode: str
    render_size: tuple[int, int]
    render_every: int

@dataclass(frozen=True)
class MediaConfig:
    canvas_size: tuple[int, int]
    panel_size: tuple[int, int]
    letterbox_bgr: tuple[int, int, int]
    output_fps: float
    comparison_alignment: str

@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_id: str
    source: SourceConfig
    tracking: TrackingConfig
    scene: SceneConfig
    ik: IKConfig
    control: ControlConfig
    collision: CollisionConfig
    perturbation: PerturbationConfig
    simulation: SimulationConfig
    media: MediaConfig
    random_seed: int
    config_path: Path
```

The committed `configs/exp001.yaml` must contain these exact groups and values:

```yaml
schema_version: 2
ik:
  position_tolerance_m: 0.005
  orientation_tolerance_deg: 5.0
  maximum_iterations: 200
  damping: 0.05
  step_size: 0.2
  orientation_weight: 0.25
  joint_limit_weight: 0.05
control:
  control_hz: 100.0
  maximum_joint_velocity_rad_s: 2.0
  maximum_joint_acceleration_rad_s2: 8.0
  gripper_open_width_m: 0.08
  gripper_closed_width_m: 0.0
  phase_duration_s:
    home: 0.5
    pregrasp: 0.8
    approach: 0.6
    close: 0.8
    lift: 0.8
    transport: 1.2
    lower: 0.6
    open: 0.6
    retreat: 0.6
    settle: 1.0
collision:
  maximum_penetration_m: 0.002
  minimum_lift_m: 0.05
  maximum_target_error_m: 0.04
  settle_duration_s: 1.0
  maximum_final_tilt_deg: 15.0
  maximum_final_linear_speed_m_s: 0.02
  minimum_bilateral_contact_duration_s: 0.2
  minimum_lift_contact_duration_s: 0.1
  allowed_contact_pairs:
    home: [[can, table]]
    pregrasp: [[can, table]]
    approach: [[can, table]]
    close: [[can, table], [left_finger, can], [right_finger, can]]
    lift: [[left_finger, can], [right_finger, can]]
    transport: [[left_finger, can], [right_finger, can]]
    lower: [[left_finger, can], [right_finger, can]]
    open: [[can, box]]
    retreat: [[can, box]]
    settle: [[can, box]]
perturbation:
  rollout_count: 30
  xy_half_range_m: 0.01
  yaw_half_range_deg: 5.0
  mass_fraction: 0.10
  friction_fraction: 0.10
media:
  canvas_size: [960, 320]
  panel_size: [320, 320]
  letterbox_bgr: [16, 16, 16]
  output_fps: 30.0
  comparison_alignment: time_warped
```

Wrap every mapping-valued field with `MappingProxyType` after validation so frozen config objects cannot be mutated through nested dictionaries. `to_public_resolved_mapping` converts tuples/proxies to YAML/JSON-compatible values, omits `config_path`, and publishes source path only as `registry:<source.id>`. Move config dataclasses and parsing out of `experiment.py`, then re-export the loader there:

```python
from .config import ExperimentConfig, load_experiment_config
```

Replace every `asdict(config)`/raw path provenance write in `experiment.py` with `to_public_resolved_mapping(config)`. Update existing provenance tests that currently expect an absolute config/source/model path: they must instead assert the logical source ID, hashes, model identifier/hash, and absence of the resolved workspace string.

Validate `experiment_id` against `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$` and reject case-insensitive Windows device basenames `CON`, `PRN`, `AUX`, `NUL`, `COM1`–`COM9`, and `LPT1`–`LPT9`. This excludes separators, drive syntax, dot segments, trailing dots/spaces, and absolute paths before any output path is built.

- [ ] **Step 4: Lock the Python 3.11 environment**

Add the console-script declaration now so the lock contains its final project metadata; `cli.py` arrives in Task 6.

```toml
[project.scripts]
webvideo-to-data = "webvideo_to_data.cli:main"

[tool.pytest.ini_options]
markers = [
  "acceptance: slow local physical acceptance tests",
  "private_video: requires the local registered EXP-001 source",
]
```

Generate and check the lock:

```powershell
py -3.11 -m pip install uv
py -3.11 -m uv lock --python 3.11
py -3.11 -m uv lock --check
py -3.11 -m uv sync --python 3.11 --frozen --extra dev
py -3.11 -m uv pip check --python .venv\Scripts\python.exe
```

Expected: all commands return 0 and `uv.lock` records a Python 3.11-compatible dependency graph.

- [ ] **Step 5: Verify GREEN and full compatibility**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_experiment.py -q -p no:cacheprovider
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe -m compileall -q src tests
git diff --check
```

Expected: all current tests plus the strict-config tests pass; compileall and diff-check return 0.

- [ ] **Step 6: Commit**

```powershell
git add pyproject.toml uv.lock configs/exp001.yaml src/webvideo_to_data/config.py src/webvideo_to_data/experiment.py tests/helpers.py tests/test_config.py tests/test_experiment.py
git commit -m "feat: validate and lock experiment configuration"
```

---

### Task 2: Pinned Menagerie Panda scene and model contract

**Files:**
- Create: `src/webvideo_to_data/assets/mujoco_menagerie/franka_emika_panda/` upstream snapshot
- Create: `src/webvideo_to_data/assets/mujoco_menagerie/franka_emika_panda/UPSTREAM.json`
- Create: `src/webvideo_to_data/assets/mujoco_menagerie/franka_emika_panda/panda_exp001.xml`
- Create: `src/webvideo_to_data/assets/mujoco_menagerie/franka_emika_panda/exp001_scene.xml`
- Create: `src/webvideo_to_data/scene.py`
- Create: `tests/test_scene.py`
- Create: `tests/verify_wheel_assets.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `DEFAULT_SCENE_PATH: Path`
- Produces: `PandaSceneIds`
- Produces: `load_panda_scene(path: str | Path = DEFAULT_SCENE_PATH) -> tuple[mujoco.MjModel, mujoco.MjData, PandaSceneIds]`
- Produces: `apply_scene_perturbation(model, data, ids, perturbation: ScenePerturbation) -> None`

```python
@dataclass(frozen=True)
class ScenePerturbation:
    can_dx_m: float
    can_dy_m: float
    can_yaw_rad: float
    mass_scale: float
    friction_scale: float
```

- [ ] **Step 1: Write failing model-contract tests**

```python
ASSET_ROOT = (
    Path(__file__).parents[1] / "src" / "webvideo_to_data" / "assets"
    / "mujoco_menagerie" / "franka_emika_panda"
)
UPSTREAM_PATH = ASSET_ROOT / "UPSTREAM.json"
LICENSE_PATH = ASSET_ROOT / "LICENSE"


def test_pinned_panda_scene_has_required_dynamics_and_collision_geometry() -> None:
    model, data, ids = load_panda_scene()
    assert len(ids.arm_joint_ids) == 7
    assert len(ids.arm_actuator_ids) == 7
    assert len(ids.finger_body_ids) == 2
    assert model.site(ids.tcp_site_id).name == "panda_tcp"
    robot_geoms = [g for g in ids.robot_geom_ids if model.geom_group[g] == 3]
    assert robot_geoms
    assert all(model.geom_contype[g] != 0 for g in robot_geoms)
    assert model.body("can").id == ids.can_body_id
    assert model.body("box").id == ids.box_body_id
    assert model.geom("table_geom").id == ids.table_geom_id
    assert np.isfinite(data.qpos).all()


def test_upstream_metadata_pins_commit_and_license() -> None:
    metadata = json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))
    assert metadata["commit"] == "da76818e269b82289eba39808e2fb91d679d6994"
    assert metadata["source"] == "https://github.com/google-deepmind/mujoco_menagerie/tree/da76818e269b82289eba39808e2fb91d679d6994/franka_emika_panda"
    assert metadata["license"] == "Apache-2.0"
    assert LICENSE_PATH.read_text(encoding="utf-8").startswith("Apache License")
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_scene.py -q -p no:cacheprovider
```

Expected: import fails because `scene.py` and the pinned model do not exist.

- [ ] **Step 3: Vendor the exact upstream model and record adaptation**

Fetch only `franka_emika_panda/` at the pinned commit. Preserve `panda.xml`, `hand.xml`, all referenced assets, `README.md`, `CHANGELOG.md`, and `LICENSE`. `UPSTREAM.json` records commit, source, license, SHA-256 for upstream files, and this adaptation:

```json
{
  "adapted_file": "panda_exp001.xml",
  "base_file": "panda.xml",
  "change": "add site panda_tcp to body hand; no inertia, joint, actuator, visual, or collision geometry changes"
}
```

`panda_exp001.xml` differs from `panda.xml` only by this child of body `hand`:

```xml
<site name="panda_tcp" pos="0 0 0.1034" size="0.005" rgba="0 1 0 1"/>
```

`exp001_scene.xml` includes the adapted Panda and defines names consumed by `scene.py`: `table_geom`, `can`, `can_free`, `can_geom`, `box`, `box_floor_geom`, four box walls, and camera `overview`. Use the existing EXP-001 start `[0.12, 0.45, 0.045]` and box center `[-0.05, 0.55]`; keep the cylinder radius 0.025 m and half-height 0.045 m.

Add setuptools package data for `assets/**/*.xml`, `assets/**/*.obj`, `assets/**/*.stl`, `assets/**/*.png`, `assets/**/LICENSE`, `assets/**/README.md`, `assets/**/CHANGELOG.md`, and `assets/**/UPSTREAM.json`. `DEFAULT_SCENE_PATH` must resolve relative to the installed `webvideo_to_data` package, not the repository root.

- [ ] **Step 4: Implement stable model-name mapping**

```python
@dataclass(frozen=True)
class PandaSceneIds:
    arm_joint_ids: tuple[int, ...]
    arm_actuator_ids: tuple[int, ...]
    finger_joint_ids: tuple[int, int]
    finger_body_ids: tuple[int, int]
    robot_body_ids: frozenset[int]
    robot_geom_ids: frozenset[int]
    tcp_site_id: int
    gripper_actuator_id: int
    can_body_id: int
    can_joint_id: int
    can_geom_id: int
    box_body_id: int
    box_floor_geom_id: int
    box_geom_ids: frozenset[int]
    table_geom_id: int


def load_panda_scene(
    path: str | Path = DEFAULT_SCENE_PATH,
) -> tuple[mujoco.MjModel, mujoco.MjData, PandaSceneIds]:
    model = mujoco.MjModel.from_xml_path(str(Path(path)))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ids = _resolve_scene_ids(model)
    return model, data, ids
```

Resolve official names `joint1..joint7`, `actuator1..actuator7`, `finger_joint1`, `finger_joint2`, `actuator8`, `left_finger`, and `right_finger`. Use `raise ValueError(f"Panda scene is missing required {object_type}: {name}")` on the first missing name.

- [ ] **Step 5: Verify GREEN and license packaging**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_scene.py -q -p no:cacheprovider
.venv\Scripts\python.exe -c "from webvideo_to_data.scene import load_panda_scene; m,d,i=load_panda_scene(); print(m.nq,m.nv,m.nu,len(i.robot_geom_ids))"
$packageCheck = Join-Path $env:TEMP ("webvideo-to-data-wheel-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $packageCheck | Out-Null
py -3.11 -m uv build --wheel --out-dir $packageCheck
$wheel = Get-ChildItem $packageCheck -Filter *.whl | Select-Object -First 1
.venv\Scripts\python.exe tests\verify_wheel_assets.py $wheel.FullName
py -3.11 -m venv "$packageCheck\venv"
py -3.11 -m uv pip install --python "$packageCheck\venv\Scripts\python.exe" $wheel.FullName
Push-Location $packageCheck
try {
  & "$packageCheck\venv\Scripts\python.exe" -c "from webvideo_to_data.scene import load_panda_scene; m,d,i=load_panda_scene(); print(m.nq,m.nv,m.nu,len(i.robot_geom_ids))"
} finally {
  Pop-Location
}
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
git diff --check
```

`verify_wheel_assets.py` locates `UPSTREAM.json` inside the wheel, requires LICENSE/README/CHANGELOG/adapted XML, then verifies every upstream relative filename and SHA-256 listed by `UPSTREAM.json` against the corresponding wheel member. Expected: scene compiles from both the source tree and an installed wheel outside the repository; seven arm joints and actuators resolve, every pinned upstream asset/license/metadata hash matches, collision geoms are enabled, and all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add pyproject.toml src/webvideo_to_data/assets/mujoco_menagerie src/webvideo_to_data/scene.py tests/test_scene.py tests/verify_wheel_assets.py
git commit -m "feat: pin official Panda simulation scene"
```

---

### Task 3: Offline IK and manual B0 control program

**Files:**
- Create: `src/webvideo_to_data/ik.py`
- Create: `tests/test_ik.py`
- Modify: `src/webvideo_to_data/retargeting.py`
- Modify: `tests/test_retargeting.py`
- Modify: `src/webvideo_to_data/simulation.py`
- Modify: `tests/test_simulation.py`

**Interfaces:**
- Consumes: `PandaSceneIds`, `IKConfig`, `ControlConfig`, `RobotReference`
- Produces: `IKPlanningError(phase: str)`, `ControlLimitError(phase: str, limit: str)`
- Produces: `IKResult`
- Produces: `JointControlProgram`
- Produces: `solve_pose_ik(model, data, ids, position_m, quaternion_wxyz, options, initial_arm_qpos) -> IKResult`
- Produces: `plan_joint_control(model, data, ids, reference, ik, control) -> JointControlProgram`
- Produces: `build_manual_b0_reference(start_m, goal_m, control, grasp_quaternion_wxyz) -> RobotReference`

Both planning exceptions inherit `RuntimeError`, store the listed fields, and construct stable redacted messages (`IK failed for phase <phase>` and `control limit <limit> failed for phase <phase>`). They represent an expected rollout rejection, not runner infrastructure failure.

- [ ] **Step 1: Write failing IK tests against real Panda kinematics**

```python
def test_pose_ik_converges_to_reachable_target_and_respects_limits(tmp_path: Path) -> None:
    model, data, ids = load_panda_scene()
    config = load_experiment_config(write_complete_config(tmp_path, with_source=False))
    target = np.array([0.12, 0.45, 0.20])
    quat = np.array([0.0, 1.0, 0.0, 0.0])
    result = solve_pose_ik(model, data, ids, target, quat, config.ik, None)
    assert result.converged
    assert result.position_error_m <= 0.005
    assert result.orientation_error_rad <= np.deg2rad(5.0)
    for value, joint_id in zip(result.arm_qpos, ids.arm_joint_ids):
        assert model.jnt_range[joint_id, 0] <= value <= model.jnt_range[joint_id, 1]


def test_pose_ik_rejects_unreachable_target_after_exact_iteration_cap(tmp_path: Path) -> None:
    model, data, ids = load_panda_scene()
    config = load_experiment_config(write_complete_config(tmp_path, with_source=False))
    result = solve_pose_ik(
        model, data, ids, np.array([5.0, 5.0, 5.0]),
        np.array([1.0, 0.0, 0.0, 0.0]), config.ik, None,
    )
    assert not result.converged
    assert result.iterations == 200
    assert result.position_error_m > 1.0
```

Add a test that mutates a target quaternion while holding position fixed; the returned joint target and orientation residual must change. This catches a position-only solver. Add a second test with two redundant initial arm postures reaching the same pose; both must converge within task tolerances, and the solution with `joint_limit_weight > 0` must have a smaller normalized joint-center cost than the same solve with weight 0. This catches a joint-center term that is not projected into the task null space.

- [ ] **Step 2: Write failing manual-reference and control-program tests**

```python
def test_manual_b0_reference_has_exact_phase_order_and_duration(tmp_path: Path) -> None:
    config = load_experiment_config(write_complete_config(tmp_path, with_source=False))
    reference = build_manual_b0_reference(
        config.scene.b0_start_m, config.scene.b0_goal_m, config.control,
        config.scene.grasp_quaternion_wxyz,
    )
    assert tuple(dict.fromkeys(reference.phase)) == (
        "home", "pregrasp", "approach", "close", "lift",
        "transport", "lower", "open", "retreat", "settle",
    )
    assert reference.timestamps_s[-1] == pytest.approx(7.5, abs=0.011)
    close = reference.phase.index("close")
    open_ = reference.phase.index("open")
    np.testing.assert_allclose(reference.ee_positions[close], [0.12, 0.45, 0.045], atol=1e-9)
    np.testing.assert_allclose(reference.ee_positions[open_], [-0.05, 0.55, 0.115], atol=1e-9)


def test_control_program_refuses_any_unconverged_key_pose(tmp_path: Path) -> None:
    model, data, ids = load_panda_scene()
    config = load_experiment_config(write_complete_config(tmp_path, with_source=False))
    reference = build_manual_b0_reference(
        config.scene.b0_start_m, config.scene.b0_goal_m, config.control,
        config.scene.grasp_quaternion_wxyz,
    )
    reference = replace(reference, ee_positions=np.full_like(reference.ee_positions, 5.0))
    with pytest.raises(IKPlanningError, match="IK failed for phase home"):
        plan_joint_control(model, data, ids, reference, config.ik, config.control)
```

- [ ] **Step 3: Verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_ik.py tests/test_retargeting.py tests/test_simulation.py -q -p no:cacheprovider
```

Expected: the new imports and phase-order assertions fail; the old one-step online IK cannot satisfy the convergence contract.

- [ ] **Step 4: Implement iterative damped least-squares IK**

```python
@dataclass(frozen=True)
class IKResult:
    arm_qpos: NDArray[np.float64]
    converged: bool
    iterations: int
    position_error_m: float
    orientation_error_rad: float


def solve_pose_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: PandaSceneIds,
    position_m: NDArray[np.float64],
    quaternion_wxyz: NDArray[np.float64],
    options: IKConfig,
    initial_arm_qpos: NDArray[np.float64] | None,
) -> IKResult:
    working = mujoco.MjData(model)
    working.qpos[:] = data.qpos
    qpos_addresses = np.array(
        [model.jnt_qposadr[joint_id] for joint_id in ids.arm_joint_ids]
    )
    dof_addresses = np.array(
        [model.jnt_dofadr[joint_id] for joint_id in ids.arm_joint_ids]
    )
    if initial_arm_qpos is not None:
        working.qpos[qpos_addresses] = initial_arm_qpos
    desired_quaternion = quaternion_wxyz / np.linalg.norm(quaternion_wxyz)
    position_norm = np.inf
    orientation_norm = np.inf
    for iteration in range(1, options.maximum_iterations + 1):
        mujoco.mj_forward(model, working)
        current_quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(current_quaternion, working.site_xmat[ids.tcp_site_id])
        current_inverse = current_quaternion * np.array([1.0, -1.0, -1.0, -1.0])
        quaternion_error = np.empty(4, dtype=float)
        mujoco.mju_mulQuat(quaternion_error, desired_quaternion, current_inverse)
        if quaternion_error[0] < 0.0:
            quaternion_error *= -1.0
        rotation_error = np.empty(3, dtype=float)
        mujoco.mju_quat2Vel(rotation_error, quaternion_error, 1.0)
        position_error = position_m - working.site_xpos[ids.tcp_site_id]
        position_norm = float(np.linalg.norm(position_error))
        orientation_norm = float(np.linalg.norm(rotation_error))
        if (
            position_norm <= options.position_tolerance_m
            and orientation_norm <= options.orientation_tolerance_rad
        ):
            return IKResult(
                working.qpos[qpos_addresses].copy(), True, iteration,
                position_norm, orientation_norm,
            )
        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacSite(model, working, jacp, jacr, ids.tcp_site_id)
        jacobian = np.vstack(
            [jacp[:, dof_addresses], options.orientation_weight * jacr[:, dof_addresses]]
        )
        error = np.r_[position_error, options.orientation_weight * rotation_error]
        normal = jacobian @ jacobian.T + options.damping**2 * np.eye(6)
        damped_pseudoinverse = jacobian.T @ np.linalg.solve(normal, np.eye(6))
        task_delta = damped_pseudoinverse @ error
        lower = model.jnt_range[np.array(ids.arm_joint_ids), 0]
        upper = model.jnt_range[np.array(ids.arm_joint_ids), 1]
        center = 0.5 * (lower + upper)
        span = np.maximum(upper - lower, 1e-9)
        joint_limit_gradient = (center - working.qpos[qpos_addresses]) / span
        nullspace = np.eye(len(ids.arm_joint_ids)) - damped_pseudoinverse @ jacobian
        delta = task_delta + nullspace @ (
            options.joint_limit_weight * joint_limit_gradient
        )
        next_qpos = working.qpos[qpos_addresses] + options.step_size * delta
        working.qpos[qpos_addresses] = np.clip(next_qpos, lower, upper)
    return IKResult(
        working.qpos[qpos_addresses].copy(), False, options.maximum_iterations,
        position_norm, orientation_norm,
    )
```

The expectation values in tests are literals from the spec, never computed using solver helpers. Add a manual-reference assertion that `home`, `pregrasp`, and `close` do not all share the same quaternion. Construct the grasp frame from the cylinder axis and TCP offset; use `grasp_quaternion_wxyz` only to select the approach convention, then derive phase orientations from geometry instead of copying one quaternion to every sample.

- [ ] **Step 5: Build phase keyframes and interpolate joint targets**

Solve the first sample and every phase boundary, using the previous solution as the next initial guess. Reject the whole program with `IKPlanningError(phase)` if a boundary does not converge. Interpolate arm targets at `control_hz=100`; limit velocity to 2 rad/s and acceleration to 8 rad/s² by stretching the affected segment, then update the timestamps. Raise `ControlLimitError(phase, limit)` if the configured stretching cannot satisfy a position/velocity/acceleration limit. Gripper targets use `control.gripper_open_width_m` and `control.gripper_closed_width_m`, mapped respectively to the official `actuator8` limits 255 and 0.

```python
@dataclass(frozen=True)
class JointControlProgram:
    timestamps_s: NDArray[np.float64]
    arm_qpos_targets: NDArray[np.float64]  # [T, 7]
    gripper_ctrl: NDArray[np.float64]      # [T]
    ee_positions: NDArray[np.float64]      # [T, 3]
    quaternion_wxyz: NDArray[np.float64]   # [T, 4]
    phase: tuple[str, ...]
    keyframe_ik: tuple[IKResult, ...]
```

Keep the legacy `_apply_damped_ik` only for B1 until its behavior is migrated; B0 must execute the precomputed `JointControlProgram`.

- [ ] **Step 6: Verify GREEN**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_ik.py tests/test_retargeting.py tests/test_simulation.py -q -p no:cacheprovider
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe -m compileall -q src tests
git diff --check
```

Expected: reachable IK converges within the exact tolerances, unreachable IK stops at iteration 200, phase/control tests and the unchanged suite pass.

- [ ] **Step 7: Commit**

```powershell
git add src/webvideo_to_data/ik.py src/webvideo_to_data/retargeting.py src/webvideo_to_data/simulation.py tests/test_ik.py tests/test_retargeting.py tests/test_simulation.py
git commit -m "feat: plan converged Panda baseline controls"
```

---

### Task 4: Collision-aware physical execution and conservative verdict

**Files:**
- Create: `src/webvideo_to_data/physics_validation.py`
- Create: `tests/test_physics_validation.py`
- Modify: `src/webvideo_to_data/simulation.py`
- Modify: `tests/test_simulation.py`
- Modify: `src/webvideo_to_data/experiment.py`
- Modify: `tests/test_experiment.py`

**Interfaces:**
- Consumes: `PandaSceneIds`, `JointControlProgram`, `CollisionConfig`
- Produces: `PhysicsRolloutFailure(failed_checks: tuple[str, ...])`, `InvalidNumericalStateError(timestamp_s: float)`
- Produces: `ContactObservation`, `PhysicsValidationResult`
- Produces: `observe_contacts(model, data, ids, phase, collision_config) -> ContactObservation`
- Produces: `validate_rollout(simulation, collision_config) -> PhysicsValidationResult`
- Extends: `SimulationResult` with control trace and measured safety fields

Both rollout exceptions inherit `RuntimeError` and store the listed evidence. `InvalidNumericalStateError` is raised immediately on the first non-finite measured state; `PhysicsRolloutFailure` is used by the benchmark executor to turn an otherwise valid but failed rollout verdict into a classified record.

The extension is explicit: add `timestamps_s: NDArray[np.float64]`, `control: NDArray[np.float64]`, `qvel: NDArray[np.float64]`, `tcp_position: NDArray[np.float64]`, `tcp_quaternion_wxyz: NDArray[np.float64]`, `phase: tuple[str, ...]`, `contact_count: NDArray[np.int64]`, `forbidden_contact: NDArray[np.bool_]`, `maximum_penetration_m: NDArray[np.float64]`, `bilateral_contact: NDArray[np.bool_]`, `box_support_contact: NDArray[np.bool_]`, `tcp_position_within_tolerance: NDArray[np.bool_]`, `tcp_orientation_within_tolerance: NDArray[np.bool_]`, `joint_position_violation: NDArray[np.bool_]`, `joint_velocity_violation: NDArray[np.bool_]`, `joint_acceleration_violation: NDArray[np.bool_]`, and `valid_numerical_state: NDArray[np.bool_]`. Existing `qpos`, `can_pose`, and rendered-frame fields keep their current types.

- [ ] **Step 1: Write failing phase-aware contact tests**

Use tiny real MuJoCo XML fixtures with named bodies; do not mock `data.contact`.

Define `contact_fixture(pair: tuple[str, str], distance: float)` in the test file. It compiles a zero-gravity XML with two 20 mm spheres. The first sphere is at the origin and the second is at x=`0.04 + distance`, so MuJoCo reports the requested separation or penetration after `mj_forward`. Assign the requested body names and build a `PandaSceneIds` whose `robot_body_ids`, finger bodies, can/box bodies, and table geom correspond to those real model IDs; unused arm tuples are empty. This keeps contact generation real while making each policy branch deterministic.

```python
@pytest.mark.parametrize(
    ("phase", "pair", "allowed"),
    [
        ("close", ("left_finger", "can"), True),
        ("transport", ("right_finger", "can"), True),
        ("settle", ("can", "box"), True),
        ("home", ("can", "table"), True),
        ("transport", ("can", "table"), False),
        ("transport", ("link4", "table"), False),
        ("transport", ("hand", "box"), False),
        ("transport", ("link5", "can"), False),
        ("transport", ("link4", "link6"), False),
    ],
)
def test_contact_policy_is_phase_and_body_specific(
    phase: str, pair: tuple[str, str], allowed: bool
) -> None:
    model, data, ids = contact_fixture(pair, distance=-0.001)
    observation = observe_contacts(model, data, ids, phase, collision_config())
    assert (not observation.has_forbidden_contact) is allowed


def test_penetration_beyond_two_millimeters_is_forbidden_even_for_allowed_pair() -> None:
    model, data, ids = contact_fixture(("left_finger", "can"), distance=-0.0021)
    observation = observe_contacts(model, data, ids, "close", collision_config())
    assert observation.has_forbidden_contact
    assert observation.maximum_forbidden_penetration_m == pytest.approx(0.0021)
```

- [ ] **Step 2: Write failing rollout-verdict tests**

```python
def valid_evidence() -> dict[str, object]:
    return {
        "execution_tracking_ratio": 0.95,
        "bilateral_close_contact_duration_s": 0.2,
        "bilateral_lift_contact_duration_s": 0.1,
        "maximum_lift_m": 0.05,
        "target_error_m": 0.04,
        "settle_duration_s": 1.0,
        "final_tilt_rad": np.deg2rad(15.0),
        "final_linear_speed_m_s": 0.02,
        "forbidden_contact_count": 0,
        "maximum_forbidden_penetration_m": 0.0,
        "joint_position_violation_count": 0,
        "joint_velocity_violation_count": 0,
        "joint_acceleration_violation_count": 0,
        "invalid_numerical_state": False,
    }


def test_exact_boundary_evidence_passes() -> None:
    assert verdict_from_evidence(**valid_evidence()).passed


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("execution_tracking_ratio", 0.949),
        ("bilateral_close_contact_duration_s", 0.199),
        ("bilateral_lift_contact_duration_s", 0.099),
        ("maximum_lift_m", 0.0499),
        ("target_error_m", 0.0401),
        ("settle_duration_s", 0.999),
        ("final_tilt_rad", np.deg2rad(15.01)),
        ("final_linear_speed_m_s", 0.0201),
        ("forbidden_contact_count", 1),
        ("joint_velocity_violation_count", 1),
        ("joint_position_violation_count", 1),
        ("joint_acceleration_violation_count", 1),
        ("maximum_forbidden_penetration_m", 0.0021),
        ("invalid_numerical_state", True),
    ],
)
def test_each_failed_gate_has_a_specific_reason(field: str, bad_value: object) -> None:
    evidence = valid_evidence()
    evidence[field] = bad_value
    verdict = verdict_from_evidence(**evidence)
    assert not verdict.passed
    assert field in verdict.failed_checks
```

- [ ] **Step 3: Verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_physics_validation.py tests/test_simulation.py tests/test_experiment.py -q -p no:cacheprovider
```

Expected: `physics_validation` imports fail and the old global `collision_validation_not_implemented` assertions fail.

- [ ] **Step 4: Implement contact observation and physical verdict**

```python
@dataclass(frozen=True)
class ContactObservation:
    bilateral_fingertip_can_contact: bool
    has_box_support_contact: bool
    has_forbidden_contact: bool
    forbidden_pairs: tuple[tuple[str, str], ...]
    maximum_forbidden_penetration_m: float


@dataclass(frozen=True)
class PhysicsValidationResult:
    passed: bool
    failed_checks: tuple[str, ...]
    forbidden_contact_count: int
    maximum_forbidden_penetration_m: float
```

Classify contacts using body ancestry, not fragile anonymous geom names, and consult only `collision_config.allowed_contact_pairs[phase]`. Deduplicate pair names within a step. Record both `contact.dist` and the phase that made a pair allowed or forbidden. Track the longest contiguous bilateral-contact interval separately in `close` and `lift`; isolated frames never satisfy the configured 0.2 s/0.1 s gates.

- [ ] **Step 5: Execute the joint-control program and record limits**

For each MuJoCo step, sample `JointControlProgram` by time, write official actuators 1–8, step physics, and record:

```python
control: [T, model.nu]
timestamps_s: [T]
qpos: [T, model.nq]
qvel: [T, model.nv]
can_pose: [T, 7]
tcp_position: [T, 3]
tcp_quaternion_wxyz: [T, 4]
phase: [T]
contact_count: [T]
bilateral_contact: [T]
box_support_contact: [T]
forbidden_contact: [T]
maximum_penetration_m: [T]
tcp_position_within_tolerance: [T]
tcp_orientation_within_tolerance: [T]
joint_position_violation: [T]
joint_velocity_violation: [T]
joint_acceleration_violation: [T]
valid_numerical_state: [T]
```

Compute acceleration from measured qvel and simulation timestep. Compute can tilt as the angle between the can local z-axis and world z. `placed_successfully` becomes exactly the result of `validate_rollout`; it is never inferred from final XY alone.

Define `execution_tracking_ratio` as the fraction of control samples for which measured TCP position error is at most `ik.position_tolerance_m` and measured quaternion geodesic error is at most `ik.orientation_tolerance_rad`. Record both Boolean per-sample gates in `simulation.npz`; do not reuse the legacy planning-time `reachability_ratio` name for B0 execution.

- [ ] **Step 6: Update terminal metrics without weakening action safety**

Refactor `_execute_run` so variant dispatch happens immediately after config/model validation. B0 constructs the manual reference and Panda control program without opening, probing, hashing, decoding, or tracking `config.source.path`; its rendered artifact is simulation-only. B1 alone enters the source probe/tracking path and may create local source/overlay comparisons. B2-B4 return their auditable `not_run` record without touching the source. Add sentinels around every source helper and assert they are never called for B0/B2-B4 in both render and no-render tests.

Replace the static action block with a status-specific helper:

```python
def _action_gate(variant: Variant, physics: PhysicsValidationResult | None) -> dict[str, Any]:
    if variant == "B0" and physics is not None and physics.passed:
        return {
            "collision_validation": "passed",
            "physics_validation": "passed",
            "action_export_eligible": False,
            "action_export_reason": "manual_baseline_not_video_grounded",
            "action_exported": False,
        }
    if variant == "B0":
        return {
            "collision_validation": "failed",
            "physics_validation": "failed",
            "action_export_eligible": False,
            "action_export_reason": "physics_validation_failed",
            "action_exported": False,
        }
    if variant == "B1":
        return {
            "collision_validation": "not_applicable_kinematic",
            "physics_validation": "not_applicable_kinematic",
            "action_export_eligible": False,
            "action_export_reason": "kinematic_replay_not_action",
            "action_exported": False,
        }
    return {
        "collision_validation": "not_run",
        "physics_validation": "not_run",
        "action_export_eligible": False,
        "action_export_reason": "metric_depth_not_available",
        "action_exported": False,
    }
```

The literal terminal outcomes are:

```text
B0 physical pass -> rejected / manual_baseline_not_video_grounded
B0 physical fail -> rejected / physics_validation_failed
B1               -> rejected / kinematic_replay_not_action
B2-B4            -> not_run / metric_depth_not_available
```

This intentionally separates two states: B0's machine-readable `physics_validation`/30-seed baseline can pass, while the variant terminal status remains `rejected` because the produced control is manual rather than video-grounded. `--require-completed` therefore returns 5 for B0; the dashboard displays `PHYSICS BASELINE PASSED` and `REJECTED AS ACTION DATA` independently. Do not collapse these into a misleading `completed` action-data status.

Write `baseline_control_trace.npz` for B0 and update `_GENERATED_RUN_FILES`, `_validate_required_run_files`, trusted manifest validation, and regression fixtures. Never create `actions.npz`.

- [ ] **Step 7: Verify GREEN and publication safety**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_physics_validation.py tests/test_simulation.py tests/test_experiment.py -q -p no:cacheprovider
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe -m compileall -q src tests
git diff --check
```

Expected: collision boundaries and exact failure reasons pass; all publication race/rollback/quarantine tests remain green; no test output contains `actions.npz` for B0.

- [ ] **Step 8: Commit**

```powershell
git add src/webvideo_to_data/physics_validation.py src/webvideo_to_data/simulation.py src/webvideo_to_data/experiment.py tests/test_physics_validation.py tests/test_simulation.py tests/test_experiment.py
git commit -m "feat: validate Panda contact and collision safety"
```

---

### Task 5: Versioned NPZ schemas and manifest v4

**Files:**
- Create: `src/webvideo_to_data/artifacts.py`
- Create: `tests/test_artifacts.py`
- Modify: `src/webvideo_to_data/experiment.py`
- Modify: `tests/test_experiment.py`

**Interfaces:**
- Produces: `ArrayContract`, `NPZContract`
- Produces: `write_npz_artifact(path, arrays, contract, provenance) -> tuple[Path, Path]`
- Produces: `load_npz_artifact(path, expected_contract) -> dict[str, NDArray]`
- Produces: `verify_run_directory(path: str | Path) -> VerifiedRun`
- Upgrades: run manifest producer format from v3 to v4 while retaining v3 verification

- [ ] **Step 1: Write failing round-trip and rejection tests**

```python
PROVENANCE = {
    "producer": "tests",
    "git_commit": "0" * 40,
    "source_sha256": "1" * 64,
    "config_sha256": "2" * 64,
    "model_sha256": "3" * 64,
    "terminal_status": "rejected",
    "terminal_reason": "kinematic_replay_not_action",
    "action_export_eligible": False,
}


def test_npz_round_trip_checks_shape_dtype_units_frame_and_timebase(tmp_path: Path) -> None:
    path = tmp_path / "trajectory_2d.npz"
    arrays = {
        "timestamps_s": np.array([0.0, 0.1], dtype=np.float64),
        "centers_px": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "confidence": np.array([1.0, 0.8], dtype=np.float64),
    }
    write_npz_artifact(path, arrays, TRAJECTORY_2D_V1, PROVENANCE)
    loaded = load_npz_artifact(path, TRAJECTORY_2D_V1)
    np.testing.assert_array_equal(loaded["centers_px"], arrays["centers_px"])
    sidecar = json.loads(path.with_suffix(".schema.json").read_text())
    assert sidecar["schema_version"] == 1
    assert sidecar["arrays"]["centers_px"]["unit"] == "pixel"
    assert sidecar["arrays"]["centers_px"]["coordinate_frame"] == "source_image_xy"
    assert sidecar["arrays"]["timestamps_s"]["timebase"] == "source_seconds"


@pytest.mark.parametrize("mutation", ["dtype", "shape", "semantic", "unit", "frame", "schema"])
def test_loader_rejects_semantic_contract_mismatch(tmp_path: Path, mutation: str) -> None:
    path = write_valid_fixture(tmp_path)
    mutate_npz_or_sidecar(path, mutation)
    with pytest.raises(ValueError, match="artifact contract mismatch"):
        load_npz_artifact(path, TRAJECTORY_2D_V1)
```

In the test file, `write_valid_fixture` calls `write_npz_artifact` with the three literal arrays from the round-trip test. `mutate_npz_or_sidecar` handles each literal case: rewrite `schema_version` to 999; rewrite the `centers_px` semantic to `robot_joint_target`; rewrite its unit to `m`; rewrite its coordinate frame to `robot_base`; rewrite its declared dtype to `float32`; or replace the NPZ `centers_px` array with shape `(2, 3)` while leaving the sidecar unchanged. It then writes the modified JSON with sorted keys and a trailing newline.

Add a non-monotonic timestamps fixture and require `ValueError("timestamps_s must be strictly increasing")`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_artifacts.py tests/test_experiment.py -q -p no:cacheprovider
```

Expected: artifact imports fail and manifests still report format 3.

- [ ] **Step 3: Implement sidecar contracts and verified loader**

```python
@dataclass(frozen=True)
class ArrayContract:
    dtype: str
    trailing_shape: tuple[int, ...]
    semantic: str
    unit: str
    coordinate_frame: str
    timebase: str | None = None


@dataclass(frozen=True)
class NPZContract:
    name: str
    schema_version: int
    arrays: Mapping[str, ArrayContract]


@dataclass(frozen=True)
class VerifiedRun:
    path: Path
    metrics: Mapping[str, Any]
    manifest: Mapping[str, Any]
    snapshot: Mapping[str, tuple[int, str]]
```

Dynamic MuJoCo widths are supplied by trusted model metadata, never copied from the sidecar under validation. Implement `baseline_control_contract(nu: int) -> NPZContract` with `control.trailing_shape == (nu,)`, and `simulation_contract(nq: int, nv: int, nu: int) -> NPZContract` with the three dynamic trailing shapes `(nq,)`, `(nv,)`, and `(nu,)` for qpos, qvel, and control respectively.

The v4 manifest records `model_nq`, `model_nv`, `model_nu`, and `model_sha256`; `verify_run_directory` loads the pinned model identified by that hash, compares its dimensions to the manifest, then constructs these expected contracts. A sidecar cannot change its own accepted dynamic shape.

`write_npz_artifact` writes the NPZ to a temporary sibling, fsyncs and replaces it, then does the same for `path.with_suffix(".schema.json")`. The sidecar records producer, git commit, source/config/model hashes, terminal status/reason/action eligibility, quaternion order `wxyz` where relevant, and the SHA-256 of the NPZ.

- [ ] **Step 4: Define exact contracts for every Phase 2A NPZ**

Define literals for:

```text
trajectory_2d: timestamps_s [T], centers_px [T,2], confidence [T]
robot_reference: timestamps_s [T], ee_positions [T,3], quaternion_wxyz [T,4], gripper_width [T], phase [T]
baseline_control_trace: timestamps_s [T], control [T,nu], phase [T]
simulation: timestamps_s [T], control [T,nu], qpos [T,nq], qvel [T,nv], can_pose [T,7], tcp_position [T,3], tcp_quaternion_wxyz [T,4], phase [T], contact_count [T], bilateral_contact [T], box_support_contact [T], forbidden_contact [T], maximum_penetration_m [T], tcp_position_within_tolerance [T], tcp_orientation_within_tolerance [T], joint_position_violation [T], joint_velocity_violation [T], joint_acceleration_violation [T], valid_numerical_state [T]
```

For every listed array, define a literal semantic string (for example `measured_joint_position`, `measured_can_pose`, `commanded_actuator_control`, or `contact_policy_violation`) in addition to dtype, trailing shape, unit, coordinate frame, and timebase. Use SI units, `source_image_xy`, `robot_base`, `world`, and `simulation_seconds` exactly as coordinate/timebase strings; phase uses unit `category`, Boolean gates use unit `boolean`, and actuator controls use unit `actuator_native`. Unit tests instantiate dynamic contracts at `(nq=16, nv=15, nu=8)`, then reject a sidecar or NPZ that substitutes any other dynamic width.

- [ ] **Step 5: Upgrade manifests and integrate all writers**

Manifest v4 records each file's size and SHA-256 plus terminal status, reason, action semantics, config hash, source hash or `not_used`, and model hash. Every GIF/MP4 file entry additionally requires `media_role` and `contains_private_source_frames: bool`; non-media entries omit both keys. The writer sets the Boolean from the rendering API's input classification, and the verifier rejects a missing/ill-typed field or a `simulation_only` role marked true. `_trusted_run_snapshot` accepts v3 with the existing exact contract and v4 with the new contract. It does not rewrite a v3 directory.

Refactor `_execute_run` to hold validated arrays in memory/staging until the variant terminal status and reason are known. Only then write every NPZ and sidecar in one terminal-serialization step. A tracking-stage failure writes no partial NPZ; it writes only the terminal metrics/rejection/manifest contract. Replace each raw `np.savez` in `experiment.py` with `write_npz_artifact`. Add sidecars and `baseline_control_trace.npz` to the generated-file allowlist and required-file validator.

- [ ] **Step 6: Verify GREEN and mutation coverage**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_artifacts.py tests/test_experiment.py -q -p no:cacheprovider
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe -m compileall -q src tests
git diff --check
```

Then temporarily alter one fixture sidecar unit from `m` to `pixel` and rerun its test; it must fail with `artifact contract mismatch`. Restore the fixture and rerun GREEN.

- [ ] **Step 7: Commit**

```powershell
git add src/webvideo_to_data/artifacts.py src/webvideo_to_data/experiment.py tests/test_artifacts.py tests/test_experiment.py
git commit -m "feat: version experiment artifact contracts"
```

---

### Task 6: Source registry, redacted preflight, and stable CLI

**Files:**
- Create: `src/webvideo_to_data/source_registry.py`
- Create: `src/webvideo_to_data/redaction.py`
- Create: `src/webvideo_to_data/preflight.py`
- Create: `src/webvideo_to_data/cli.py`
- Create: `configs/sources.yaml`
- Create: `tests/test_source_registry.py`
- Create: `tests/test_redaction.py`
- Create: `tests/test_preflight.py`
- Create: `tests/test_cli.py`
- Modify: `src/webvideo_to_data/experiment.py`
- Modify: `scripts/run_exp001.py`

**Interfaces:**
- Produces: `SourceRecord`, `load_source_registry(path) -> Mapping[str, SourceRecord]`
- Produces: `redact_text(value: str, workspace: Path | None = None) -> str`
- Produces: `PublicationFinding`, `audit_publication_tree(path) -> tuple[PublicationFinding, ...]`
- Produces: `PreflightDeps`, `PreflightCheck`, `PreflightReport`, `run_preflight(config_path, variants, no_render, deps=None, registry_path=Path("configs/sources.yaml")) -> PreflightReport`
- Produces: `main(argv: Sequence[str] | None = None) -> int`

```python
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
```

- [ ] **Step 1: Write failing registry and redaction tests**

```python
def test_private_source_record_is_complete_without_inventing_capture_date() -> None:
    records = load_source_registry(Path("configs/sources.yaml"))
    record = records["exp001-phone-can-v0"]
    assert record.origin == "user_recorded"
    assert record.captured_on is None
    assert record.captured_on_status == "not_recorded"
    assert record.license == "private_not_redistributable"
    assert record.publishable is False
    assert record.privacy_review == "local_only"


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


def test_publication_audit_detects_text_binary_and_media_metadata(tmp_path: Path) -> None:
    write_publication_audit_fixtures(tmp_path)
    findings = audit_publication_tree(tmp_path)
    assert {finding.kind for finding in findings} == {
        "authorization", "credential_query", "local_path", "secret_pattern",
        "media_metadata",
    }
```

`write_publication_audit_fixtures` constructs test values from fragments at runtime, never committing a live-looking provider token or real username. It writes one UTF-8 file, one binary file, and an MP4 with a synthetic metadata tag. The auditor scans text, bounded binary byte windows, URL query keys (`token`, `key`, `signature`, `credential`), Authorization/cookie/SSH patterns, generic provider-token shapes, absolute Windows/POSIX home paths, and ffprobe format/stream tags.

- [ ] **Step 2: Write failing preflight behavior tests**

Use a frozen `PreflightDeps` dataclass containing executable lookup, version runner, renderer probe, source hashing, scene loader, and writable-parent probe callables. Unit tests pass deterministic real local fixtures/functions through this object; one integration smoke uses the default system dependencies. Assert behavior, not call mocks.

```python
def test_b0_preflight_does_not_require_private_video(tmp_path: Path) -> None:
    config = write_complete_config(
        tmp_path, {"source.path": "absent-private-video.mp4"}, with_source=False
    )
    report = run_preflight(
        config, variants=("B0",), no_render=False, deps=passing_preflight_deps()
    )
    assert report.passed
    assert report.by_name("source_video").status == "not_required_for_manual_b0"


def test_preflight_is_read_only_even_when_checks_fail(tmp_path: Path) -> None:
    config = write_complete_config(tmp_path)
    registry_path = write_source_registry(tmp_path)
    before = snapshot_tree(tmp_path)
    report = run_preflight(
        config, variants=("B1",), no_render=True,
        deps=failing_preflight_deps(), registry_path=registry_path,
    )
    assert not report.passed
    assert snapshot_tree(tmp_path) == before


def test_b1_preflight_reports_hash_mismatch_with_fix_hint(tmp_path: Path) -> None:
    config = write_complete_config(tmp_path, {"source.sha256": "0" * 64})
    report = run_preflight(
        config, variants=("B1",), no_render=True,
        deps=passing_preflight_deps(), registry_path=write_source_registry(tmp_path),
    )
    assert not report.passed
    check = report.by_name("source_video")
    assert check.code == "source_sha256_mismatch"
    assert "configs/sources.yaml" in check.remediation
```

- [ ] **Step 3: Write failing CLI exit-code tests**

```python
@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (("preflight", "--config", "missing.yaml"), 2),
        (("run", "--config", "bad.yaml", "--variant", "B0"), 2),
        (("verify", "--run", "missing-run"), 4),
    ],
)
def test_cli_exit_codes(argv: tuple[str, ...], expected: int) -> None:
    assert main(argv) == expected

```

- [ ] **Step 4: Verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_source_registry.py tests/test_redaction.py tests/test_preflight.py tests/test_cli.py -q -p no:cacheprovider
```

Expected: imports fail because the four modules do not exist.

- [ ] **Step 5: Implement the registry and preflight report**

`configs/sources.yaml` contains the source ID, a replaceable relative placeholder path, an all-zero 64-hex placeholder, and access text `local file required; not distributed`. A B1 operator replaces the path and hash only in the local checkout; the public repository never stores the private source identity.

```python
@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    status: str
    code: str
    detail: str
    remediation: str


@dataclass(frozen=True)
class PreflightReport:
    passed: bool
    checks: tuple[PreflightCheck, ...]
```

Run checks in stable order: Python, config, source registry, source file/hash when needed, ffmpeg, ffprobe, MuJoCo scene, renderer or no-render, output parent. Source registry/file/hash are required only if B1 is among the requested variants; B0 is manual/source-independent and B2-B4 stop before source use, so those checks return `not_required_for_requested_variants`. Catch each exception, redact it, and continue so the user sees all failures at once.

- [ ] **Step 6: Implement argparse subcommands and compatibility wrapper**

```python
EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_PREFLIGHT = 3
EXIT_VERIFY = 4
EXIT_NOT_COMPLETED = 5
EXIT_RUNNER = 10
```

`cli.py` prints exactly one JSON object to stdout for `--json`; human mode prints one line per preflight check and ends with the artifact path. In Task 6, `run` requires exactly one `--variant` plus explicit `--output-dir` and invokes only the lower-level `run_experiment`; `--all`, `--artifacts-root`, and `--run-id` are not registered until Task 7. `scripts/run_exp001.py` maps its legacy output behavior to this explicit form. Add tests that it writes exactly the requested single-variant directory and invalid/missing option combinations exit 2 without creating it.

CLI responses never echo absolute local paths. For explicit `--output-dir`, return the path relative to the current workspace when it is inside the workspace; otherwise return `"<external-output>/<basename>"`. Tests pass an absolute temporary directory and assert the real prefix appears in neither stdout nor stderr.

In Task 6, `verify --run <path>` verifies a v4 variant; Task 7 extends the same command to auto-detect a v1 suite. `--decode-media` decodes every frame of every manifest-listed GIF/MP4 with OpenCV and compares dimensions/duration to ffprobe facts; `--privacy-audit` runs `audit_publication_tree`. Either failure exits 4. Task 6 tests include a truncated MP4 and each privacy fixture and assert exit 4, then a clean generated variant and assert exit 0; Task 7 adds the equivalent verified-suite cases.

Place one exception boundary around the entire `main()` dispatch. Map typed config/preflight/verify/not-completed/runner failures to the documented exit code, run every exception message and remediation through `redact_text` before either JSON or human output, and never print a traceback unless an explicit developer-only environment flag is set. Parameterize config, run, verify, and dashboard failures with runtime-assembled workspace paths, Authorization values, provider-token shapes, and signed queries; assert neither stdout nor stderr contains the original values.

Before persisting `_PipelineFailure` messages in `experiment.py`, pass them through `redact_text`. Redaction never logs the original value.

- [ ] **Step 7: Verify GREEN**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_source_registry.py tests/test_redaction.py tests/test_preflight.py tests/test_cli.py tests/test_experiment.py -q -p no:cacheprovider
.venv\Scripts\webvideo-to-data.exe preflight --config configs/exp001.yaml --variant B0 --no-render --json
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
git diff --check
```

Expected: B0 no-render preflight passes without reading the private video; all tests pass; stdout/artifacts contain no absolute workspace path or test secret.

- [ ] **Step 8: Commit**

```powershell
git add configs/sources.yaml scripts/run_exp001.py src/webvideo_to_data/source_registry.py src/webvideo_to_data/redaction.py src/webvideo_to_data/preflight.py src/webvideo_to_data/cli.py src/webvideo_to_data/experiment.py tests/test_source_registry.py tests/test_redaction.py tests/test_preflight.py tests/test_cli.py
git commit -m "feat: add safe preflight and command line interface"
```

---

### Task 7: Append-only suite runner and fixed 30-seed B0 benchmark

**Files:**
- Create: `src/webvideo_to_data/suite.py`
- Create: `tests/test_suite.py`
- Create: `tests/test_b0_acceptance.py`
- Modify: `src/webvideo_to_data/cli.py`
- Modify: `src/webvideo_to_data/experiment.py`
- Modify: `tests/test_experiment.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `run_experiment`, `run_mujoco_replay`, `apply_scene_perturbation`
- Produces: `RunIdentity`, `RolloutRecord`, `B0BenchmarkSummary`, `SuiteResult`
- Produces: `VerifiedSuite`, `verify_suite_directory(path: str | Path) -> VerifiedSuite`
- Produces: `make_run_id(now_utc, config_sha256, random_suffix) -> str`
- Produces: `validate_run_id(value: str) -> str`
- Produces: `RolloutExecutor = Callable[[ExperimentConfig, int, ScenePerturbation], RolloutRecord]`
- Produces: `evaluate_b0_robustness(config, seeds: Sequence[int], executor: RolloutExecutor | None = None) -> B0BenchmarkSummary`
- Produces: `SuiteDeps`, `run_suite(config_path: str | Path, artifacts_root: str | Path, variants: Sequence[Variant], no_render: bool, run_id: str | None = None, deps: SuiteDeps | None = None) -> SuiteResult`

Task 7 extends `run` with append-only `--artifacts-root` (default `artifacts`), `--variant` or `--all`, optional `--run-id`, and `--require-completed`; explicit `--output-dir` remains the mutually exclusive Task 6 compatibility form. The deterministic fixture's single-variant JSON response is `{"run_id": "20260819T120102123456Z-a1b2c3d4-7f29", "run_path": "SYNTHETIC/runs/20260819T120102123456Z-a1b2c3d4-7f29", "requested_variants": ["B2"], "status": "not_run", "reason": "metric_depth_not_available"}`. An `--all` response uses terminal `status: "recorded"` plus a `variants` mapping. Paths are logical and relative to the supplied artifacts root, never absolute.

```python
@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    run_dir: Path

@dataclass(frozen=True)
class B0BenchmarkSummary:
    rollouts: int
    successes: int
    passed: bool
    reason: str
    wilson_95_low: float
    wilson_95_high: float
    total_forbidden_contacts: int
    maximum_forbidden_penetration_m: float
    records: tuple[RolloutRecord, ...]

@dataclass(frozen=True)
class SuiteResult:
    run_id: str
    run_dir: Path
    metrics: Mapping[str, Any]
    dashboard_path: Path | None

@dataclass(frozen=True)
class VerifiedSuite:
    path: Path
    metrics: Mapping[str, Any]
    manifest: Mapping[str, Any]
    variant_runs: Mapping[str, VerifiedRun]

@dataclass(frozen=True)
class SuiteDeps:
    now_utc: Callable[[], datetime]
    random_suffix: Callable[[], str]
    run_variant: Callable[[ExperimentConfig, Path, Variant, bool], Mapping[str, Any]]
    evaluate_b0: Callable[[ExperimentConfig, Sequence[int]], B0BenchmarkSummary]
```

Production uses real functions. Test helpers return deterministic timestamps/suffixes and write fully verified variant fixtures; they do not bypass suite manifest or latest-pointer code.

- [ ] **Step 1: Write failing append-only tests**

```python
def test_run_id_is_stable_format_with_config_hash() -> None:
    value = make_run_id(
        datetime(2026, 8, 19, 12, 1, 2, 123456, tzinfo=timezone.utc),
        "a1b2c3d4" + "0" * 56,
        "7f29",
    )
    assert value == "20260819T120102123456Z-a1b2c3d4-7f29"


def test_suite_refuses_existing_run_and_never_replaces_it(tmp_path: Path) -> None:
    config_path = write_complete_config(tmp_path)
    config = load_experiment_config(config_path)
    occupied = tmp_path / config.experiment_id / "runs" / "20260819T120102123456Z-a1b2c3d4-7f29"
    occupied.mkdir(parents=True)
    personal = occupied / "personal.txt"
    personal.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="run already exists"):
        run_suite(
            config_path=config_path,
            artifacts_root=tmp_path,
            variants=("B0",),
            no_render=True,
            run_id=occupied.name,
        )
    assert personal.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "unsafe",
    ("..", ".", "../escape", "a/b", r"a\b", r"C:\escape", "/escape", "CON"),
)
def test_suite_rejects_path_escape_before_any_output_write(
    tmp_path: Path, unsafe: str,
) -> None:
    config_path = write_complete_config(tmp_path)
    artifacts_root = tmp_path / "artifacts"
    before = snapshot_tree(tmp_path)
    with pytest.raises(ValueError, match="invalid run_id"):
        run_suite(
            config_path, artifacts_root, variants=("B2",), no_render=True,
            run_id=unsafe,
        )
    assert snapshot_tree(tmp_path) == before
```

Define test-record helpers with literal safe/failed fields:

```python
def passing_record(seed: int) -> RolloutRecord:
    return RolloutRecord(
        seed=seed, perturbation={}, passed=True, failed_checks=(),
        execution_tracking_ratio=0.96, maximum_lift_m=0.051,
        target_error_m=0.039, final_tilt_rad=np.deg2rad(14.0),
        final_linear_speed_m_s=0.019, forbidden_contact_count=0,
        maximum_forbidden_penetration_m=0.0,
    )


def failed_record(seed: int, field: str) -> RolloutRecord:
    return replace(
        passing_record(seed), passed=False, failed_checks=(field,),
        target_error_m=0.041,
    )
```

Test that `latest.json` changes only after every requested variant and suite manifest verify successfully; inject a B1 failure and assert the prior pointer bytes are unchanged.

Add a deterministic benchmark-executor seam and test that seed 23 raising `IKPlanningError("transport")` produces one failed `RolloutRecord` with `failed_checks=("ik_key_pose_transport",)`, while seeds 19–22 and 24–48 still execute and the JSONL contains exactly 30 records in seed order. Catch only typed expected rollout failures (`IKPlanningError`, `ControlLimitError`, `PhysicsRolloutFailure`, `InvalidNumericalStateError`) and convert them to records; model-load, file-I/O, manifest, and programmer exceptions remain infrastructure failures that stop the suite and preserve the old `latest.json`.

Add two real-process tests sharing the same experiment root. A deterministic barrier pauses both just before `latest.json`; the per-experiment pointer lock must serialize final manifest verification plus pointer replacement. Assert the final pointer is the later UTC run ID, never a partially written run, and both immutable run directories still verify. Alias the parent path in the second process using the same physical-path strategy as the existing publication-lock regression.

Add Task 7 CLI tests: normal append-only run creates `<artifacts-root>/<experiment_id>/runs/<run-id>`; `--output-dir` conflicts with `--all`, `--run-id`, and an explicitly supplied `--artifacts-root`; single B2 `--require-completed --json` returns exit 5 and the exact single-variant response above; `--all --json` returns the suite response; neither stdout nor stderr contains the absolute temporary root.

- [ ] **Step 2: Write failing benchmark aggregation tests**

```python
def test_benchmark_passes_only_at_twenty_four_of_thirty_with_no_illegal_contact() -> None:
    records = [passing_record(seed) for seed in range(19, 43)]
    records += [failed_record(seed, "target_error_m") for seed in range(43, 49)]
    summary = summarize_b0(records)
    assert summary.successes == 24
    assert summary.rollouts == 30
    assert summary.passed
    assert summary.wilson_95_low == pytest.approx(0.6269, abs=1e-4)


def test_any_illegal_contact_fails_benchmark_even_with_twenty_four_successes() -> None:
    records = [passing_record(seed) for seed in range(19, 49)]
    records[0] = replace(records[0], forbidden_contact_count=1)
    summary = summarize_b0(records)
    assert not summary.passed
    assert summary.reason == "illegal_contact_observed"
```

- [ ] **Step 3: Write the slow physical acceptance test**

```python
@pytest.mark.acceptance
def test_fixed_thirty_seed_panda_b0_acceptance() -> None:
    config = load_experiment_config("configs/exp001.yaml")
    summary = evaluate_b0_robustness(config, seeds=tuple(range(19, 49)))
    assert summary.rollouts == 30
    assert summary.successes >= 24
    assert summary.total_forbidden_contacts == 0
    assert summary.maximum_forbidden_penetration_m <= 0.002


@pytest.mark.acceptance
def test_real_b0_suite_never_exports_actions(tmp_path: Path) -> None:
    result = run_suite(
        "configs/exp001.yaml", tmp_path, variants=("B0",), no_render=True,
    )
    assert verify_suite_directory(result.run_dir).metrics["actions_exported"] == 0
    assert list(result.run_dir.rglob("actions.npz")) == []
```

- [ ] **Step 4: Verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_suite.py tests/test_b0_acceptance.py -q -p no:cacheprovider
```

Expected: suite imports fail. Once aggregation exists, the acceptance test remains RED until the physical baseline meets the actual threshold.

- [ ] **Step 5: Implement deterministic perturbations and benchmark records**

For seed `s`, use only `np.random.default_rng(s)`. Sample each scalar with `uniform(-half_range, half_range)` and record the literal draw. Apply mass/friction from immutable nominal values, never cumulatively from the prior rollout.

Because the canonical can is axisymmetric, record `yaw_perturbation_observability="geometrically_unobservable_for_axisymmetric_can"` in benchmark metadata and add a scene test that each sampled yaw is actually applied to the free-joint quaternion. Do not claim yaw robustness as an independently observable success dimension until a non-axisymmetric object is introduced.

```python
@dataclass(frozen=True)
class RolloutRecord:
    seed: int
    perturbation: Mapping[str, float]
    passed: bool
    failed_checks: tuple[str, ...]
    execution_tracking_ratio: float
    maximum_lift_m: float
    target_error_m: float
    final_tilt_rad: float
    final_linear_speed_m_s: float
    forbidden_contact_count: int
    maximum_forbidden_penetration_m: float


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    denominator = 1.0 + z * z / trials
    center = (successes / trials + z * z / (2.0 * trials)) / denominator
    radius = z * np.sqrt(successes * (trials - successes) / trials**3 + z*z/(4*trials**2)) / denominator
    return center - radius, center + radius
```

- [ ] **Step 6: Implement append-only suite publication**

Create the unique run directory with `Path.mkdir(exist_ok=False)`. Write `resolved-config.yaml` through `to_public_resolved_mapping(config)`: omit `config_path`, replace the resolved local `source.path` with `registry:<source.id>`, and retain source/config hashes. `environment.json` contains only OS name/version, architecture, Python/MuJoCo/NumPy/OpenCV/ffmpeg/ffprobe versions, generator commit/dirty flag, model hash, and renderer backend—never executable, virtualenv, workspace, or user paths. Then call `run_experiment` into each child variant. For B0, do not probe or track the source video; use the nominal `run_experiment` output for diagnostic media, then build the same manual reference for the 30-seed benchmark. Write each rollout as one JSON line and the aggregate to `B0/benchmark-summary.json`.

Accept a run ID only if it matches `^\d{8}T\d{12}Z-[0-9a-f]{8}-[0-9a-f]{4}$`; generated IDs pass through the same validator. Before creating the experiment parent or run directory, resolve `artifacts_root`, resolve the candidate through any existing symlink/junction parents with `strict=False`, and require `candidate.relative_to(resolved_artifacts_root)` to succeed. Apply the same containment check to `latest.json` and the lock path. No `mkdir`, lock, temp file, or error artifact occurs before all identifier and containment checks pass.

Write schema-v1 `suite-metrics.json` with `experiment_id`, `run_id`, requested variants, per-variant terminal status/reason/manifest hash, and action count. If B0 was requested, add the exact Python payload `{"b0_physics_baseline": "passed" if summary.passed else "failed", "b0_rollouts": summary.rollouts, "b0_successes": summary.successes}`; otherwise those three keys are `null` and B0 is `not_requested`. Always add `"actions_exported": 0`. Define schema-v1 JSONL records for `B0/benchmark-rollouts.jsonl`, schema-v1 `B0/benchmark-summary.json`, schema-v1 `suite-manifest.json` with `feature_set: ["core"]` and SHA-256/size for every run file except `suite-manifest.json` itself, and schema-v1 `latest.json` outside the run directory containing only relative `run_path`, `run_id`, and `suite_manifest_sha256`. Suite media entries copy the v4 `media_role` and `contains_private_source_frames` fields; the suite verifier enforces them independently. `verify_suite_directory` rejects unknown versions, missing/extra files, hash/size mismatch, inconsistent run IDs, an unverified variant manifest, or any `actions.npz`.

Task 7 publishes a core suite containing resolved config, environment, benchmark records, and variant runs—no dashboard/GIF dependency yet. Verify that final core manifest before atomically replacing `latest.json`. Hold a physical-parent-derived per-experiment file lock across final verification and `latest.json` replacement, and compare the full microsecond UTC run key so an older run finishing later cannot replace a newer pointer. On a failure, leave the unique run directory with explicit `failed` suite metrics and do not change `latest.json`. Task 8 extends the builder for newly created suites only; it never mutates a Task 7 finalized run.

- [ ] **Step 7: Make the physical acceptance GREEN**

Run the acceptance test after each controller/scene adjustment:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_b0_acceptance.py -m acceptance -vv -p no:cacheprovider
```

Changes are limited to configured phase durations, grasp/TCP transform, interpolation, actuator gains exposed in `panda_exp001.xml`, and cylinder contact parameters. Every adjustment must be recorded in `configs/exp001.yaml` or upstream adaptation metadata. Do not weaken the fixed success thresholds, reduce the perturbation ranges, discard failed seeds, or introduce kinematic object pose overrides.

Expected: 30 rollouts, at least 24 successful, zero forbidden contacts, maximum illegal penetration no more than 0.002 m.

- [ ] **Step 8: Verify GREEN and append-only behavior**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_suite.py -q -p no:cacheprovider
.venv\Scripts\python.exe -m pytest tests/test_b0_acceptance.py -m acceptance -vv -p no:cacheprovider
.venv\Scripts\python.exe -m pytest -m "not acceptance and not private_video" -q -p no:cacheprovider
.venv\Scripts\python.exe -m compileall -q src tests
git diff --check
```

Expected: append-only tests, real 30-seed acceptance, public suite, compileall, and diff-check pass.

- [ ] **Step 9: Commit**

```powershell
git add src/webvideo_to_data/suite.py src/webvideo_to_data/cli.py src/webvideo_to_data/experiment.py src/webvideo_to_data/ik.py src/webvideo_to_data/retargeting.py src/webvideo_to_data/simulation.py tests/test_suite.py tests/test_b0_acceptance.py tests/test_experiment.py configs/exp001.yaml src/webvideo_to_data/assets/mujoco_menagerie/franka_emika_panda/panda_exp001.xml src/webvideo_to_data/assets/mujoco_menagerie/franka_emika_panda/exp001_scene.xml src/webvideo_to_data/assets/mujoco_menagerie/franka_emika_panda/UPSTREAM.json
git commit -m "feat: add append-only Panda baseline benchmark"
```

---

### Task 8: Honest diagnostic media and static dashboard

**Files:**
- Create: `src/webvideo_to_data/dashboard.py`
- Create: `tests/test_dashboard.py`
- Modify: `src/webvideo_to_data/visualization.py`
- Modify: `src/webvideo_to_data/experiment.py`
- Modify: `src/webvideo_to_data/suite.py`
- Modify: `src/webvideo_to_data/cli.py`
- Modify: `tests/test_visualization.py`
- Modify: `tests/test_experiment.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `letterbox_frame(frame, panel_size, color) -> NDArray[np.uint8]`
- Produces: `MediaLabels`, `labels_for_metrics(metrics) -> MediaLabels`
- Produces: `render_comparison_video(source_path, overlay_path, simulation_frames, output_path, media_config, metrics, source_duration_s, simulation_duration_s) -> Path`
- Produces: `render_preview_gif(comparison_path, output_path, width_px=960, fps=8, maximum_duration_s=12.0) -> Path`
- Produces: `render_public_simulation_preview(simulation_path, output_path, labels, media_config) -> Path`
- Produces: `copy_public_preview(run_dir: str | Path, variant: Literal["B0", "B1"], output_path: str | Path) -> Path`
- Produces: `generate_dashboard(run_dir: str | Path) -> Path`

```python
@dataclass(frozen=True)
class MediaLabels:
    status: str
    mode: str
    metric_warning: str
    time_alignment: str
```

- [ ] **Step 1: Write failing aspect-ratio and label tests**

```python
def test_letterbox_preserves_vertical_source_aspect_ratio() -> None:
    frame = np.zeros((120, 60, 3), dtype=np.uint8)
    frame[:, :, 1] = 255
    output = letterbox_frame(frame, (120, 120), (16, 16, 16))
    assert output.shape == (120, 120, 3)
    assert np.all(output[:, :30] == 16)
    assert np.all(output[:, 90:] == 16)
    assert np.all(output[:, 30:90, 1] == 255)


def test_rejected_b1_media_labels_cannot_look_like_action_success() -> None:
    labels = labels_for_metrics({"status": "rejected", "variant": "B1"})
    assert labels.status == "REJECTED — NOT ACTION DATA"
    assert labels.mode == "KINEMATIC OBJECT-POSE OVERRIDE"
    assert labels.metric_warning == "availability != semantic accuracy"


def test_preview_gif_preserves_comparison_aspect_and_frame_count(tmp_path: Path) -> None:
    comparison = write_labeled_comparison_fixture(
        tmp_path, size=(960, 320), frames=24, fps=8
    )
    output = render_preview_gif(comparison, tmp_path / "preview.gif", fps=8)
    frames = decode_all_frames(output)
    assert len(frames) == 24
    assert frames[0].shape[:2] == (320, 960)
    assert np.mean(np.abs(frames[0].astype(float) - frames[-1].astype(float))) > 1.0
```

`write_labeled_comparison_fixture` uses the production status-banner compositor to create 24 changing frames containing `REJECTED — NOT ACTION DATA`; `decode_all_frames` uses OpenCV and fails if any frame cannot be decoded. The GIF is derived from already labeled comparison frames, so it cannot silently omit the rejection watermark.

- [ ] **Step 2: Write failing dashboard behavior tests**

```python
def test_dashboard_leads_with_action_outcome_and_uses_relative_assets(tmp_path: Path) -> None:
    run = write_dashboard_input_fixture(tmp_path)
    output = generate_dashboard(run)
    html = output.read_text(encoding="utf-8")
    assert "NO ACTION EXPORTED · 0 / 5 eligible" in html
    assert "B0 manual physics baseline" in html
    assert "B1 kinematic diagnostic" in html
    assert "availability != semantic accuracy" in html
    assert "INPUT COMMIT · " + "1" * 12 in html
    assert "GENERATOR COMMIT · " + "2" * 12 in html
    assert "ARTIFACTS VERIFIED" in html
    assert "C:\\Users" not in html
    assert "file://" not in html
    assert "../B0/simulation.mp4" in html
    assert "../B1/side_by_side.mp4" not in html
    assert "private local media omitted" in html


def test_dashboard_cli_verifies_run_and_returns_output_path(tmp_path: Path, capsys) -> None:
    run = write_verified_suite_fixture(tmp_path)
    before = snapshot_tree(run)
    assert main(("dashboard", "--run", str(run), "--json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dashboard_path"] == "dashboard/index.html"
    assert snapshot_tree(run) == before


def test_dashboard_cli_rejects_unverified_run(tmp_path: Path) -> None:
    run = write_verified_suite_fixture(tmp_path)
    (run / "suite-metrics.json").write_text("{}", encoding="utf-8")
    assert main(("dashboard", "--run", str(run))) == 4


def test_task8_suite_includes_dashboard_without_mutating_prior_core_run(tmp_path: Path) -> None:
    core = write_verified_core_suite_fixture(tmp_path)
    core_before = snapshot_tree(core)
    enhanced = run_suite(
        write_complete_config(tmp_path), tmp_path / "enhanced",
        variants=("B0", "B1"), no_render=False, deps=passing_suite_deps(),
    )
    verified = verify_suite_directory(enhanced.run_dir)
    assert verified.manifest["feature_set"] == ["core", "dashboard"]
    assert "dashboard/index.html" in verified.manifest["files"]
    assert snapshot_tree(core) == core_before


def test_public_preview_copy_refuses_private_source_frames(tmp_path: Path) -> None:
    run = write_verified_suite_fixture(tmp_path, b1_private=True)
    with pytest.raises(ValueError, match="contains private source frames"):
        copy_public_preview(run, "B1", tmp_path / "public.gif")


def test_public_preview_copy_reports_unrequested_variant(tmp_path: Path) -> None:
    run = write_verified_suite_fixture(tmp_path, variants=("B0",))
    with pytest.raises(ValueError, match="B1 preview is not present in verified suite"):
        copy_public_preview(run, "B1", tmp_path / "public.gif")
```

`write_dashboard_input_fixture` creates `suite-metrics.json` with five literal variants (B0/B1 rejected, B2-B4 not_run), `actions_exported: 0`, a B0 `physics_validation: passed`, input commit `"1" * 40`, generator commit `"2" * 40`, a public simulation-only MP4 at `B0/simulation.mp4`, and a private-source comparison at `B1/side_by_side.mp4`. It writes the corresponding media privacy fields and verified variant manifests using the Task 5 artifact helper so the dashboard exercises the real loader rather than bypassing validation. `write_verified_suite_fixture` calls it, generates the dashboard through the same internal builder used by `run_suite`, writes the final suite manifest, and verifies it; public `generate_dashboard` itself never publishes or mutates a finalized run.

- [ ] **Step 3: Verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_visualization.py tests/test_dashboard.py -q -p no:cacheprovider
```

Expected: letterbox, label, and dashboard imports fail.

- [ ] **Step 4: Implement aspect-preserving comparison frames**

```python
def letterbox_frame(
    frame: NDArray[np.uint8],
    panel_size: tuple[int, int],
    color: tuple[int, int, int],
) -> NDArray[np.uint8]:
    panel_width, panel_height = panel_size
    scale = min(panel_width / frame.shape[1], panel_height / frame.shape[0])
    width = max(1, round(frame.shape[1] * scale))
    height = max(1, round(frame.shape[0] * scale))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.full((panel_height, panel_width, 3), color, dtype=np.uint8)
    x = (panel_width - width) // 2
    y = (panel_height - height) // 2
    canvas[y:y+height, x:x+width] = resized
    return canvas
```

Every comparison frame contains source clock formatted as `VIDEO t={source_time_s:.2f}s`, simulation clock formatted as `SIM t={simulation_time_s:.2f}s`, and `TIME-WARPED FOR COMPARISON` when durations differ. Draw status banners after letterboxing so they are never cropped.

- [ ] **Step 5: Implement dependency-free static HTML**

Parse only verified suite/variant JSON. Escape every dynamic string with `html.escape`. Use embedded CSS and local relative paths; no CDN, JavaScript framework, tracking, or absolute path. Variant cards order B0-B4 and show status, reason, physics verdict, action eligibility, input commit (or `N/A`), generator commit, artifact verification status, hashes, and missing measurements as `N/A`. These values come from the verified manifest/provenance, never from query parameters or filenames. Never emit a `<video>`, `<img>`, or hyperlink for a media entry whose verified `contains_private_source_frames` is true; show `private local media omitted` instead.

When B0 passes physics but is action-ineligible, render both facts:

```text
PHYSICS BASELINE PASSED
REJECTED AS ACTION DATA — manual baseline is not video-grounded
```

- [ ] **Step 6: Integrate media generation and dashboard into the suite**

Replace the stretching helpers in `experiment.py` with visualization-module calls. B0 renders simulation-only because it is source-independent. Local B1 may render a source/overlay/simulation comparison, but a source marked `publishable: false` can never feed a committed or suite-level public preview. Extend `run_suite` for newly created runs with `feature_set=["core", "dashboard"]`: after variant verification but before suite-manifest creation, generate suite-level `dashboard/media/B0-preview.gif` and `B1-preview.gif` using only verified simulation MP4s plus terminal labels, then generate the dashboard and finalize the suite manifest. Core Task 7 runs remain byte-for-byte immutable. The GIF preserves the simulation aspect ratio and already-rendered status/time labels; every media entry records `media_role` and `contains_private_source_frames`, and `copy_public_preview` verifies the suite plus a false privacy flag before copying. Never point README at an MP4 as though GitHub would embed it inline. If media are disabled, dashboard cards show `media not generated (--no-render)` and do not emit broken `<video>` elements.

- [ ] **Step 7: Verify GREEN and inspect real renders**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_visualization.py tests/test_dashboard.py tests/test_experiment.py -q -p no:cacheprovider
.venv\Scripts\python.exe -m pytest -m "not acceptance and not private_video" -q -p no:cacheprovider
git diff --check
```

Generate one rendered B0 suite and inspect `dashboard/index.html`, a contact sheet, and a frame from each MP4. Confirm the phone video is not stretched in the local B1 comparison, the two clocks differ when time-warped, public previews contain no source frames, and status banners are legible at GitHub README size. `webvideo-to-data dashboard --run <run> --json` verifies the immutable suite and returns its existing dashboard path without changing any byte; `--output <external-path>` may render a separate copy outside the run. It returns exit 4 for a mutated run.

- [ ] **Step 8: Commit**

```powershell
git add src/webvideo_to_data/dashboard.py src/webvideo_to_data/visualization.py src/webvideo_to_data/experiment.py src/webvideo_to_data/suite.py src/webvideo_to_data/cli.py tests/test_dashboard.py tests/test_visualization.py tests/test_experiment.py tests/test_cli.py
git commit -m "feat: generate honest experiment dashboard"
```

---

### Task 9: CI, clean-room quickstart, real EXP-001 run, and documentation

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `docs/media/exp001-b0-side-by-side.gif`
- Modify: `docs/media/exp001-b1-side-by-side.gif`
- Modify: `README.md`
- Modify: `docs/method.md`
- Modify: `experiments/README.md`
- Modify: `experiments/EXP-001-phone-can-mujoco/README.md`
- Modify: `experiments/EXP-001-phone-can-mujoco/report.md`
- Modify: `experiments/EXP-001-phone-can-mujoco/metrics.json`
- Modify: `summary.md`

**Interfaces:**
- Consumes: all prior Phase 2A commands and artifacts
- Produces: public CI contract, verified quickstart, versioned sanitized experiment conclusion

- [ ] **Step 1: Write CI workflow with exact public command**

Use a Windows/Ubuntu matrix with Python 3.11 and uv:

```yaml
strategy:
  matrix:
    os: [windows-latest, ubuntu-latest]
steps:
  - uses: actions/checkout@v4
  - uses: astral-sh/setup-uv@v6
    with:
      enable-cache: true
  - run: uv python install 3.11
  - run: uv sync --python 3.11 --frozen --extra dev
  - run: uv run python -m compileall -q src tests
  - run: uv run pytest -m "not acceptance and not private_video" -q -p no:cacheprovider
  - run: git diff --check
```

Set `MUJOCO_GL=egl` on Ubuntu only for tests that construct a renderer. Do not add repository secrets.

- [ ] **Step 2: Verify a clean Python 3.11 installation locally**

Create a task-specific temporary venv outside tracked paths, then run the public contract:

```powershell
$phase2aEnv = Join-Path $env:TEMP ("webvideo-to-data-phase2a-" + [guid]::NewGuid())
$phase2aPreviousVirtualEnv = $env:VIRTUAL_ENV
py -3.11 -m venv $phase2aEnv
& "$phase2aEnv\Scripts\python.exe" -m pip install uv
try {
  $env:VIRTUAL_ENV = $phase2aEnv
  & "$phase2aEnv\Scripts\python.exe" -m uv sync --active --frozen --extra dev
  & "$phase2aEnv\Scripts\python.exe" -m pytest -m "not acceptance and not private_video" -q -p no:cacheprovider
} finally {
  if ([string]::IsNullOrEmpty($phase2aPreviousVirtualEnv)) {
    Remove-Item Env:\VIRTUAL_ENV -ErrorAction SilentlyContinue
  } else {
    $env:VIRTUAL_ENV = $phase2aPreviousVirtualEnv
  }
}
```

Expected: installation and public tests return 0 without the private phone video.

- [ ] **Step 3: Run the real local Phase 2A experiment**

First verify the source registry, then run a fresh append-only suite:

```powershell
$phase2aPreflight = .venv\Scripts\webvideo-to-data.exe preflight --config configs/exp001.yaml --all --json | ConvertFrom-Json
if (-not $phase2aPreflight.passed) { throw "Phase 2A preflight failed" }
$phase2aRun = .venv\Scripts\webvideo-to-data.exe run --config configs/exp001.yaml --all --json | ConvertFrom-Json
$phase2aRunPath = Resolve-Path (Join-Path "artifacts" $phase2aRun.run_path)
.venv\Scripts\webvideo-to-data.exe verify --run $phase2aRunPath --decode-media --privacy-audit
.venv\Scripts\python.exe -m pytest tests/test_b0_acceptance.py -m acceptance -vv -p no:cacheprovider
```

Expected: preflight passes; a new unique run is created; verify returns 0; the acceptance test reports 30 rollouts, at least 24 successes, zero forbidden contacts, and no `actions.npz` anywhere in the run.

- [ ] **Step 4: Audit generated media and manifests**

For every generated MP4, run ffprobe for codec, width, height, frame rate, and duration, then decode all frames with OpenCV. Verify each file against manifest v4. Check that no file contains the workspace path or known local secret filenames.

```powershell
Get-ChildItem $phase2aRunPath -Recurse -Filter *.mp4 | ForEach-Object {
  ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,avg_frame_rate:format=duration -of json -- $_.FullName
}
.venv\Scripts\webvideo-to-data.exe verify --run $phase2aRunPath --decode-media --privacy-audit
$actionFiles = @(Get-ChildItem $phase2aRunPath -Recurse -Filter actions.npz)
if ($actionFiles.Count -ne 0) { throw "unexpected actions.npz in verified run" }
```

Expected: ffprobe returns valid streams; the verifier hashes every manifest, decodes all media frames, finds no privacy patterns/unsafe metadata, and exits 0; the explicit action assertion finds no files.

- [ ] **Step 5: Rewrite documentation from measured outputs**

README quickstart begins with:

```powershell
py -3.11 -m venv .venv
py -3.11 -m pip install uv
py -3.11 -m uv sync --python 3.11 --frozen --extra dev
.venv\Scripts\webvideo-to-data.exe preflight --config configs/exp001.yaml --variant B0 --no-render
.venv\Scripts\webvideo-to-data.exe run --config configs/exp001.yaml --variant B0
```

Update all result numbers only from the verified suite JSON. State separately:

- B0 manual physics result and 30-seed success count;
- B0 terminal rejection as action data;
- B1 kinematic rejection and known LK semantic drift;
- B2-B4 `metric_depth_not_available`;
- zero exported actions;
- Phase 2B requires annotation and mask-gated tracker comparison;
- metric claims require measured geometry or new calibrated video.

Copy the verified simulation-only B0/B1 suite previews to `docs/media/exp001-b0-side-by-side.gif` and `docs/media/exp001-b1-side-by-side.gif`. The copy step must refuse any preview whose manifest records `contains_private_source_frames=true`; assert the committed previews record `false`. Embed those low-bitrate files with relative Markdown image paths so they animate directly on the GitHub README page. Caption both as diagnostic rejected results, keep the rejection watermark visible in every frame, and link the tracked experiment report for the full audit. The run-local dashboard is not copied or linked as a public artifact; although it omits private media links, its verified run remains local-only. Do not use an MP4 link as the primary visual because GitHub renders it as a download/navigation target rather than an inline README video.

Move publication-race implementation history to a reproducibility appendix. Do not claim contact inference from hand evidence or metric 3D reconstruction.

- [ ] **Step 6: Run final verification**

Run fresh commands after the documentation edits:

```powershell
.venv\Scripts\python.exe -m pytest -m "not private_video" -q -p no:cacheprovider
.venv\Scripts\python.exe -m compileall -q src tests
py -3.11 -m uv lock --check
git diff --check
git status --short
```

Expected: public plus acceptance tests pass, compileall/lock/diff checks return 0, and status lists only Phase 2A files intended for the final documentation commit.

- [ ] **Step 7: Commit**

```powershell
git add .github/workflows/ci.yml README.md docs/media/exp001-b0-side-by-side.gif docs/media/exp001-b1-side-by-side.gif docs/method.md experiments/README.md experiments/EXP-001-phone-can-mujoco/README.md experiments/EXP-001-phone-can-mujoco/report.md experiments/EXP-001-phone-can-mujoco/metrics.json summary.md
git commit -m "docs: publish Phase 2A baseline evidence"
```

---

## Plan self-review checklist

- Every spec section maps to a task: configuration (1), official model (2), IK/control (3), collision/physics (4), schema (5), preflight/registry/CLI (6), append-only benchmark (7), dashboard/media (8), CI/docs/real run (9).
- The terminal-status interpretation is explicit: manual B0 can pass physics while remaining rejected as action data.
- All new public signatures are defined before a later task consumes them.
- The existing publication-safety regression suite runs after every task that changes `experiment.py`.
- The only slow acceptance gate is the real fixed 30-seed test; it cannot be satisfied by mocks, threshold changes, discarded seeds, or kinematic object motion.
- Phase 2B and Phase 2C remain outside this implementation plan.

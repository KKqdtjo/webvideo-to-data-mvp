# WebVideo to Data MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal (original plan):** Convert a local private demonstration video into an auditable 2D motion/contact record and a MuJoCo Panda pick-and-place replay with metrics and visualizations. The public plan uses `source-placeholder.mp4` instead of the private filename.

**Actual MVP note (2026-08-19):** The implementation uses a primitive 7-DoF Panda-like diagnostic model, not the official Franka Panda. B0/B1 were rejected and B2-B4 were not run. Pixel/contact annotations, measured geometry, perturbation evaluation, and complete collision validation were not performed. Action export is hard-disabled with `collision_validation_not_implemented`; actual evidence is in `experiments/EXP-001-phone-can-mujoco/`.

**Architecture:** A Python 3.11 package separates video/media contracts, lightweight OpenCV perception, object-centric retargeting, and MuJoCo execution. Every stage writes file artifacts; unvalidated trajectories are never exported as action-bearing episodes. B0 proves the simulator/control path, while B1-B4 progressively substitute video-derived motion, automatic phases, and constrained smoothing.

**Tech Stack:** Python 3.11, NumPy, SciPy, OpenCV, MuJoCo, imageio/ffmpeg, Matplotlib, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-webvideo-to-data-mvp-design.md`

## Global Constraints

- Preserve all files under `video/`; compute and record hashes but never rewrite source media.
- Use a Python 3.11 virtual environment at `.venv`; do not use the system Python 3.13 for experiment execution.
- Read no API credential during the MVP; ignore `video-generation/seedance_api.txt` and `www.youtube.com_cookies.txt` in Git and logs.
- Use test-first red-green-refactor for production Python behavior.
- Store generated binary artifacts under `artifacts/` and ignore them in Git; commit human-readable reports and machine-readable summary metrics under `experiments/`.
- Use SI units, right-handed frames, `quaternion_wxyz`, and transform names `T_target_source`.
- B3/B4 may become action-bearing only when IK/reachability is at least 95% and simulation validation succeeds.
- If a full Panda grasp controller is unstable, report the measured result and keep B0/B1 as trajectory-replay feasibility; do not relabel a kinematic visualization as physics success.
- Do not install or alter software on the team-provided SSH worker in this plan. Its endpoint is intentionally not stored in the public repository.

---

### Task 1: Secure project foundation, media contracts, and provenance

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/webvideo_to_data/__init__.py`
- Create: `src/webvideo_to_data/schema.py`
- Create: `src/webvideo_to_data/media.py`
- Create: `tests/test_schema.py`
- Create: `tests/test_media.py`

**Interfaces:**
- Consumes: local source video path and ffprobe JSON.
- Produces: `RunStatus`, `VideoMetadata`, `PhaseInterval`, `Trajectory2D`, `sha256_file(path)`, and `probe_video(path)` for all later tasks.

- [ ] **Step 1: Create the Python 3.11 environment and install declared dependencies**

Create `pyproject.toml` with project name `webvideo-to-data`, Python requirement `>=3.11,<3.12`, and dependencies `numpy>=1.26,<3`, `scipy>=1.11,<2`, `opencv-python>=4.9,<5`, `mujoco>=3.2,<4`, `imageio>=2.34,<3`, `imageio-ffmpeg>=0.5,<1`, `matplotlib>=3.8,<4`, `PyYAML>=6,<7`. Add pytest under `[project.optional-dependencies].dev`.

Run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Expected: dependency installation exits 0 and `.venv\Scripts\python.exe -c "import mujoco, cv2, numpy"` exits 0.

- [ ] **Step 2: Write failing schema tests**

```python
import numpy as np
import pytest
from webvideo_to_data.schema import PhaseInterval, Trajectory2D

def test_phase_interval_rejects_reversed_frames():
    with pytest.raises(ValueError, match="end_frame"):
        PhaseInterval("hold", 8, 4, 0.8, ("motion",))

def test_trajectory_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        Trajectory2D(
            timestamps_s=np.array([0.0, 0.1]),
            centers_px=np.array([[10.0, 20.0]]),
            confidence=np.array([0.9, 0.8]),
        )
```

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_schema.py -v`
Expected: FAIL because `webvideo_to_data.schema` does not exist.

- [ ] **Step 3: Implement validated data contracts**

Implement frozen dataclasses. `PhaseInterval` validates frame order and confidence in `[0,1]`. `Trajectory2D` converts values to NumPy arrays, validates `[T]`, `[T,2]`, `[T]`, finite values, matching lengths, monotonic timestamps, and confidence range.

```python
class RunStatus(str, Enum):
    COMPLETED = "completed"
    NOT_RUN = "not_run"
    REJECTED = "rejected"
    FAILED = "failed"
```

Run the schema tests again. Expected: 2 passed.

- [ ] **Step 4: Write failing media tests against a generated fixture video**

The test uses OpenCV `VideoWriter` to create ten 64×48 MJPG frames in `tmp_path`, then asserts:

```python
metadata = probe_video(fixture_path)
assert metadata.width == 64
assert metadata.height == 48
assert metadata.frame_count == 10
assert metadata.fps == pytest.approx(10.0, rel=0.1)
assert len(sha256_file(fixture_path)) == 64
```

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_media.py -v`
Expected: FAIL because `probe_video` and `sha256_file` are missing.

- [ ] **Step 5: Implement media probing and hashing**

`probe_video` reads width, height, fps, frame count, and duration using OpenCV, raises `FileNotFoundError` for a missing path and `ValueError("video cannot be opened")` for an unreadable file. `sha256_file` streams 1 MiB chunks.

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_schema.py tests/test_media.py -v`
Expected: all tests pass.

- [ ] **Step 6: Secure generated and sensitive files**

Add `.gitignore` entries for `.venv/`, `.superpowers/`, `artifacts/`, `.pytest_cache/`, `__pycache__/`, `*.pyc`, `video-generation/seedance_api.txt`, `www.youtube.com_cookies.txt`, and generated `*.egg-info/`. Verify:

```powershell
git check-ignore video-generation/seedance_api.txt www.youtube.com_cookies.txt
```

Expected: both paths print and exit 0.

- [ ] **Step 7: Commit Task 1**

```powershell
git add .gitignore pyproject.toml src tests docs README.md summary.md experiments Doubao-Seedance-2.0-mini调用示例.md video-generation/具身数据文献整理与思考.md
git commit -m "feat: establish auditable video data contracts"
```

Do not add `video/`, credential files, cookies, `.venv/`, or `artifacts/`.

---

### Task 2: Lightweight video tracking, contact phases, and diagnostic overlay

**Files:**
- Create: `src/webvideo_to_data/tracking.py`
- Create: `src/webvideo_to_data/contact.py`
- Create: `src/webvideo_to_data/visualization.py`
- Create: `tests/test_tracking.py`
- Create: `tests/test_contact.py`
- Create: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `VideoMetadata`, source video, and configured initial object ROI `[x,y,w,h]`.
- Produces: `track_roi_lk(video_path, initial_roi) -> Trajectory2D`, `infer_motion_phases(trajectory, fps) -> tuple[PhaseInterval, ...]`, and `render_tracking_overlay(...) -> Path`.

- [ ] **Step 1: Write a failing deterministic tracking test**

Generate a 20-frame 96×72 video containing a textured 16×20 blue rectangle that moves exactly two pixels right per frame. Call `track_roi_lk` with its first-frame ROI and assert final displacement is `38±3` pixels, valid confidence exists for at least 18 frames, and all timestamps are monotonic.

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_tracking.py -v`
Expected: FAIL because `tracking.py` does not exist.

- [ ] **Step 2: Implement Lucas-Kanade ROI tracking**

Use `cv2.goodFeaturesToTrack` in the ROI, `cv2.calcOpticalFlowPyrLK`, forward-backward error below 1.5 px, and median displacement. Re-detect features within the current ROI when fewer than eight survive. Confidence is `min(1, valid_points/24)` multiplied by the valid forward-backward fraction. Reject an ROI outside the frame or with fewer than four initial features.

Run the tracking test. Expected: pass.

- [ ] **Step 3: Write failing contact-state tests**

Use a literal center trajectory with 10 stationary frames, 20 moving frames, and 10 stationary frames. With `fps=10`, `speed_on_px_s=8`, `speed_off_px_s=3`, `min_phase_s=0.3`, assert phase names are exactly `("approach", "hold", "release", "settle")`, `hold.start_frame` is within one frame of 10, and `release.start_frame` is within two frames of 30.

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_contact.py -v`
Expected: FAIL because `infer_motion_phases` is missing.

- [ ] **Step 4: Implement motion-based phase inference**

Smooth centers with Savitzky-Golay when at least seven frames exist; compute speed in px/s. Use hysteresis and minimum duration to identify first sustained motion as hold onset and first sustained post-motion stillness as release. Emit four non-overlapping intervals with evidence strings `object_still`, `object_motion`, and `object_settled`. Mark confidence from the separation between observed speed and thresholds.

Run tracking and contact tests. Expected: pass.

- [ ] **Step 5: Write and implement the overlay test**

The failing test renders five frames and checks the output video exists, has five readable frames, and the first output frame differs from the input in a 5-pixel neighborhood around the tracked center. Implement colored ROI, point center, trail, phase label, confidence bar, and timestamp. No network or model weight is used.

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_visualization.py -v`
Expected after implementation: pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add src/webvideo_to_data/tracking.py src/webvideo_to_data/contact.py src/webvideo_to_data/visualization.py tests/test_tracking.py tests/test_contact.py tests/test_visualization.py
git commit -m "feat: track object motion and visualize contact phases"
```

---

### Task 3: Object-centric trajectory mapping and MuJoCo B0/B1 replay

**Files:**
- Create: `src/webvideo_to_data/retargeting.py`
- Create: `src/webvideo_to_data/simulation.py`
- Create: `src/webvideo_to_data/assets/panda_pick_place.xml`
- Create: `tests/test_retargeting.py`
- Create: `tests/test_simulation.py`

**Interfaces:**
- Consumes: `Trajectory2D`, phase intervals, scene calibration values, and a trajectory variant (`B0` or `B1`).
- Produces: `RobotReference`, `map_pixels_to_scene(...)`, `build_pick_place_reference(...)`, and `run_mujoco_replay(...) -> SimulationResult`.

- [ ] **Step 1: Write failing pixel-to-scene mapping tests**

With image size 540×960, scene bounds x `[-0.15,0.15]` m and y `[0.35,0.65]` m, assert center pixel `(270,480)` maps to `(0.0,0.5)`, left/top `(0,0)` maps to `(-0.15,0.35)`, and all points are clipped to bounds. This is an explicit canonical mapping, not a claim of metric reconstruction.

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_retargeting.py -v`
Expected: FAIL because `retargeting.py` does not exist.

- [ ] **Step 2: Implement canonical scene mapping and reference phases**

Define `RobotReference` with timestamps, end-effector positions `[T,3]`, end-effector `quaternion_wxyz`, gripper width, phase per frame, and source variant. B0 uses fixed start `[0.12,0.45,0.04]` and goal `[-0.05,0.55,0.13]`. B1 maps the first and last stable tracked centers to those canonical bounds while preserving normalized path shape. Generate approach, close, lift, transport, lower, open, and retreat segments with finite-difference velocity below configured `0.35 m/s` after time scaling.

Run retargeting tests. Expected: pass.

- [ ] **Step 3: Write a failing MuJoCo smoke test**

Load the XML in headless mode, reset, run 100 simulation steps, and assert finite `qpos/qvel`, the model contains body names `can`, `box`, and `panda_hand`, and `SimulationResult` reports a nonnegative minimum distance and no invalid numerical state.

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_simulation.py -v`
Expected: FAIL because the scene and runner do not exist.

- [ ] **Step 4: Implement the Panda scene and replay runner**

Original implementation intent: use MuJoCo's packaged Menagerie Panda model if available, otherwise vendor an attributed model. Actual MVP: a hard-coded primitive 7-DoF Panda-like diagnostic XML was used; it is not the official Franka Panda and complete collision validation is not implemented. The runner distinguishes `kinematic_replay` from `physics_grasp` in `SimulationResult.mode`, but neither mode is action-export eligible in the current implementation.

Run simulation and retargeting tests. Expected: pass.

- [ ] **Step 5: Implement physics success checks**

`placed_successfully` is true only if the can was lifted at least 3 cm, ends inside the box top x/y margin, has support contact for 1 s after gripper opening, and has no numerical instability. A pure mocap/kinematic object replay must return `placed_successfully=False` with `mode="kinematic_replay"`.

Add one test where a kinematic replay cannot be counted as physics success and one stationary-scene negative test. Run all Task 3 tests. Expected: pass.

- [ ] **Step 6: Commit Task 3**

```powershell
git add src/webvideo_to_data/retargeting.py src/webvideo_to_data/simulation.py src/webvideo_to_data/assets tests/test_retargeting.py tests/test_simulation.py
git commit -m "feat: replay object-centric references in MuJoCo"
```

---

### Task 4: End-to-end experiment runner, B0-B4 results, and visual report

**Files:**
- Create: `configs/exp001.yaml`
- Create: `src/webvideo_to_data/experiment.py`
- Create: `scripts/run_exp001.py`
- Create: `tests/test_experiment.py`
- Modify: `README.md`
- Modify: `experiments/EXP-001-phone-can-mujoco/README.md`
- Create: `experiments/EXP-001-phone-can-mujoco/metrics.json`
- Create: `experiments/EXP-001-phone-can-mujoco/report.md`
- Generated, ignored: `artifacts/EXP-001/**`

**Interfaces:**
- Consumes: all Task 1-3 interfaces and `configs/exp001.yaml`.
- Produces: one command that runs source probing, tracking, phase inference, retargeting, simulation, metrics, tracking overlay, trajectory plot, MuJoCo replay, and side-by-side comparison.

- [ ] **Step 1: Write a failing orchestration test**

Use a synthetic moving-object video and a small config. Assert `run_experiment(config_path, output_dir)` creates `provenance.json`, `trajectory_2d.npz`, `phases.json`, `metrics.json`, and `tracking_overlay.mp4`; assert metrics contain `source_sha256`, `lk_point_availability_ratio`, `phase_count`, `variant`, `simulation_mode`, and `placed_successfully`. The LK metric is point confidence/forward-backward availability, not semantic tracking accuracy.

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment.py -v`
Expected: FAIL because `experiment.py` does not exist.

- [ ] **Step 2: Implement orchestration and atomic artifact writes**

Parse YAML into explicit dataclasses, write JSON through a temporary file followed by `Path.replace`, and save NumPy arrays with named keys. A rejected stage writes `rejection.json` and does not write `actions.npz`. The runner accepts `--variant B0|B1|B2|B3|B4` and `--no-render`.

Run the orchestration test. Expected: pass.

- [ ] **Step 3: Lock the real-video configuration**

Set source path `source-placeholder.mp4` and replace the all-zero SHA-256 placeholder locally before B1. Keep the first-frame ROI `[374, 423, 104, 155]` in `[x,y,w,h]` pixels, 30 FPS, forward-backward threshold 1.5 px, minimum 8 live points, canonical scene bounds x `[-0.15,0.15]` m and y `[0.35,0.65]` m, B0 start `[0.12,0.45,0.04]` m, B0 goal `[-0.05,0.55,0.13]` m, and random seed 19. The ROI was fixed from frame 0 before any run and must not be tuned after reading success metrics.

- [ ] **Step 4: Run B0 and B1, then B2-B4 as supported**

```powershell
.\.venv\Scripts\python.exe scripts/run_exp001.py --config configs/exp001.yaml --variant B0
.\.venv\Scripts\python.exe scripts/run_exp001.py --config configs/exp001.yaml --variant B1
.\.venv\Scripts\python.exe scripts/run_exp001.py --config configs/exp001.yaml --variant B2
.\.venv\Scripts\python.exe scripts/run_exp001.py --config configs/exp001.yaml --variant B3
.\.venv\Scripts\python.exe scripts/run_exp001.py --config configs/exp001.yaml --variant B4
```

If depth is not implemented in this MVP, B2-B4 must be recorded as `not_run` with exact reason `metric_depth_not_available`; B1 still records canonical 2D-to-scene replay. Never copy B1 numbers into B2-B4.

- [ ] **Step 5: Produce visualizations**

Generate:

- `tracking_overlay.mp4` with ROI, trail, contact phase and confidence;
- `trajectory_2d.png` with x/y and speed over time plus contact boundaries;
- `mujoco_replay.mp4` showing Panda, can and target;
- `side_by_side.mp4` with source video, perception overlay and MuJoCo replay synchronized to a common duration;
- a contact sheet PNG containing source, overlay and simulation checkpoints.

Use ffprobe to verify every MP4 is readable and has nonzero duration.

- [ ] **Step 6: Write measured results, not expected results**

Populate `metrics.json` from generated metrics. In `report.md`, state which variants ran, exact commands, runtime, track validity, inferred phase times, simulation mode, lift/placement result, target error, failures, and artifact paths. If B0/B1 fail, keep the failure artifacts and explain the measured cause.

- [ ] **Step 7: Run full verification**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts
ffprobe -v error -show_entries format=duration -of default=nw=1 artifacts/EXP-001/B1/tracking_overlay.mp4
ffprobe -v error -show_entries format=duration -of default=nw=1 artifacts/EXP-001/B1/mujoco_replay.mp4
```

Expected: tests exit 0; compileall exits 0; each ffprobe duration is greater than zero. Physics success is reported only if the recorded metrics meet its checks.

- [ ] **Step 8: Update documentation and commit Task 4**

Add the verified artifact index and actual result summary to README and the EXP-001 report. Commit source, tests, config, metrics, and reports, but not generated binary artifacts or source videos.

```powershell
git add configs src scripts tests README.md experiments/EXP-001-phone-can-mujoco
git commit -m "exp: run phone video to MuJoCo feasibility study"
```

# EXP-001 实测报告：手机罐体视频到 MuJoCo

## 结论

真实视频的读取、LK 跟踪、阶段推断、B0/B1 参考轨迹、MuJoCo 回放和可视化链路均实际运行。实验没有产生任何可用 robot action：B0 的真实 `physics_grasp` 因 reachability 仅 0.0235655737704918、没有双指接触、lift=0 而拒绝；B1 是对象位姿覆盖的 `kinematic_replay`，只作为 canonical 2D-to-scene 诊断回放，明确标记为拒绝且 `placed_successfully=false`。B2–B4 未实现公制深度，独立记录为 `not_run`，原因均为 `metric_depth_not_available`。

## 输入与锁定配置

- 源视频：`video/手机录制.mp4`（未修改）
- SHA-256：`55E98463B8E270F4E7D87BDA5D0EE73329880A96CF753FC2C78E1849F1C4AB17`
- ffprobe：HEVC，540×960，30 FPS，210 帧，7.019083 秒
- 首帧 ROI：`[374,423,104,155]`，格式 `[x,y,w,h]`
- forward-backward 阈值：1.5 px；最少 live points：8
- canonical bounds：x `[-0.15,0.15]` m，y `[0.35,0.65]` m
- B0 start `[0.12,0.45,0.04]` m；goal `[-0.05,0.55,0.13]` m
- 随机种子：19

ROI 在任何结果读取前已固定，本实验没有据结果调参。

## TDD 证据

1. 首个 orchestration RED：`python -m pytest tests/test_experiment.py -v` 因 `ModuleNotFoundError: webvideo_to_data.experiment` 失败；最小实现后 GREEN 为 1 passed。
2. 可视化、unsupported variant 与 tracking rejection RED：分别缺少四种媒体、B2 被错误当成 completed、tracking 未拒绝；实现后 3 passed。
3. CLI RED：`scripts/run_exp001.py` 不存在；实现 `--variant`、`--output-dir`、`--no-render` 后 GREEN。
4. B0 稳定基线 RED：长视频把 reference 从 40 个采样错误扩成 241 个；固定五点手工 transport 后，基线独立于视频帧数，并复现 Task 3 实测。
5. action gate RED：B0/B1 初次错误保留 `actions.npz` 且状态为 completed；改为先写 `robot_reference.npz`，仅物理成功后才允许 `actions.npz`。B0/B1 现均写 simulation rejection，回归测试 2 passed。
6. 配置透传 RED：forward-backward 阈值只解析未使用；透传至 LK tracker 后 GREEN。
7. B2–B4 runtime RED：缺失 `runtime_s`；加入实测 probe runtime 后 GREEN。

## 真实运行命令与结果

```powershell
.\.venv\Scripts\python.exe scripts/run_exp001.py --config configs/exp001.yaml --variant B0
.\.venv\Scripts\python.exe scripts/run_exp001.py --config configs/exp001.yaml --variant B1
.\.venv\Scripts\python.exe scripts/run_exp001.py --config configs/exp001.yaml --variant B2
.\.venv\Scripts\python.exe scripts/run_exp001.py --config configs/exp001.yaml --variant B3
.\.venv\Scripts\python.exe scripts/run_exp001.py --config configs/exp001.yaml --variant B4
```

| 变体 | 状态 / 模式 | runtime | 实测结果 |
| --- | --- | ---: | --- |
| B0 | rejected / physics_grasp | 3.756149 s | reachability 0.0235656；双指接触 0 帧；lift 0 m；target error 0.197230 m；无最终支撑接触；not placed；无 action 导出 |
| B1 | rejected / kinematic_replay | 10.294673 s | canonical 2D-to-scene diagnostic；reachability 0.0969086；位姿覆盖产生的 height gain 0.185 m 不算物理 lift；target error 0.059916 m；not placed；无 action 导出 |
| B2 | not_run | 0.019968 s | `metric_depth_not_available` |
| B3 | not_run | 0.019399 s | `metric_depth_not_available` |
| B4 | not_run | 0.022026 s | `metric_depth_not_available` |

B1 的 9761 个 `grasp_contact` 仿真帧来自 kinematic object pose override 下的几何接触诊断，不是合格物理抓取证据；`maximum_lift_m` 仍为 0，且成功判定被 simulation mode gate 否决。

## 跟踪与阶段

valid track ratio 为 1.0，共推断 4 个阶段：approach 0.000–0.033 s（confidence 0.0156961）、hold 0.067–5.200 s（1.0）、release 5.233–5.500 s（0.0）、settle 5.533–6.967 s（0.0）。B0/B1 metrics 均明确记录 `perception_status=degraded` 和两个 `zero_confidence_phase` warnings；没有人工把它们改成更好看的时间段。

## 可视化与 ffprobe

每个实际运行的 B0/B1 都生成 `tracking_overlay.mp4`、`trajectory_2d.png`、`mujoco_replay.mp4`、`side_by_side.mp4`、`contact_sheet.png`。contact sheet 已目视检查，包含 source、overlay 和 MuJoCo 四个同步检查点；PNG 尺寸为 1280×720，轨迹图为 1120×840。

| 文件 | codec / 分辨率 / 帧 | ffprobe duration |
| --- | --- | ---: |
| B0 tracking_overlay.mp4 | MPEG-4 / 540×960 / 210 | 7.000 s |
| B0 mujoco_replay.mp4 | MPEG-4 / 320×240 / 39 | 3.900 s |
| B0 side_by_side.mp4 | MPEG-4 / 960×240 / 210 | 7.000 s |
| B1 tracking_overlay.mp4 | MPEG-4 / 540×960 / 210 | 7.000 s |
| B1 mujoco_replay.mp4 | MPEG-4 / 320×240 / 240 | 24.000 s |
| B1 side_by_side.mp4 | MPEG-4 / 960×240 / 210 | 7.000 s |

所有 MP4 均可读且 duration > 0。side-by-side 将 source、perception overlay 和 MuJoCo replay 重采样到源视频的共同 7.0 秒时间轴；独立 MuJoCo replay 保留自身完整时长。

## Artifact 清单

- `artifacts/EXP-001/B0/` 与 `B1/`：`run_manifest.json`、`provenance.json`、`trajectory_2d.npz`、`phases.json`、`robot_reference.npz`、`simulation.npz`、`metrics.json`、`rejection.json` 及五个可视化文件。
- `artifacts/EXP-001/B2/`、`B3/`、`B4/`：各自独立的 `run_manifest.json`、`provenance.json` 和 `metrics.json`。
- 没有任何 B0/B1 `actions.npz`；binary artifacts 和源视频均由 `.gitignore` 排除。
- 版本化汇总：`experiments/EXP-001-phone-can-mujoco/metrics.json`。

## 自审与 concerns

- B0 严格复现已知失败口径，没有把首次因帧数耦合得到的不同 replay 当成锁定基线。
- B1 没有被描述为 physics success；对象位姿覆盖、漂亮回放、height gain 和几何接触均未转换成 action 或 placement success。
- B2–B4 没有 B1 数值，只有各自的 not-run 原因和 runtime。
- 主要 concern 是控制/IK：B0 reachability 远低于 0.95，无法验证抓放。
- 感知 concern 是自动阶段边界：release/settle confidence 为 0，且长 hold 很可能混合了手、罐体和相机运动。
- 公制深度尚不可用，因此不能评价 B2–B4 的尺度、三维重投影或物理约束收益。

## Review fix round 1/5

runner 已改为同级 fresh staging + validated directory swap。parse、SHA、tracking、simulation 或 visualization exception 都会发布可审计的 failed metrics/rejection；复用 output 不会保留旧 action/media。发布前每个 MP4 均通过 ffprobe duration 与首帧解码；`--no-render` 不创建 MuJoCo Renderer。B0 配置起终点已实际透传，相对 source 改为相对 config 目录解析。

修复后真实 B0–B4 重跑的 runtime 为 3.7561494、10.2946731、0.0199679、0.0193988、0.0220259 秒。完整测试为 54 passed，compileall exit 0，六个 MP4 duration>0；B0/B1 无 action，B2–B4 各自只有 manifest/provenance/metrics。实验结论与失败口径不变。

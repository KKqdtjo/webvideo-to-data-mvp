# EXP-001 实测报告：手机罐体视频到 MuJoCo

## 结论

真实视频的读取、LK 跟踪、阶段推断、B0/B1 参考轨迹、MuJoCo 回放和可视化链路均实际运行。实验没有产生任何可用 robot action：B0 的真实 `physics_grasp` 因 reachability 仅 0.0235655737704918、没有双指接触、lift=0 而拒绝；B1 是对象位姿覆盖的 `kinematic_replay`，只作为 canonical 2D-to-scene 诊断回放，明确标记为拒绝且 `placed_successfully=false`。B2–B4 未实现公制深度，独立记录为 `not_run`，原因均为 `metric_depth_not_available`。当前硬编码几何的 primitive 7-DoF Panda-like diagnostic model 不是官方 Franka Panda，且未实现完整 collision validation；所有变体均记录 `action_export_eligible=false` / `collision_validation_not_implemented`。

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
| B0 | rejected / physics_grasp | 6.886575 s | reachability 0.0235656；双指接触 0 帧；lift 0 m；target error 0.197230 m；无最终支撑接触；not placed；collision validation 未实现；无 action 导出 |
| B1 | rejected / kinematic_replay | 10.316316 s | canonical 2D-to-scene diagnostic；reachability 0.0969086；位姿覆盖产生的 height gain 0.185 m 不算物理 lift；target error 0.059916 m；语义 endpoint/path 不可靠；not placed；无 action 导出 |
| B2 | not_run | 0.216154 s | `metric_depth_not_available`；未复制 B1 指标 |
| B3 | not_run | 0.219323 s | `metric_depth_not_available`；未复制 B1 指标 |
| B4 | not_run | 0.194170 s | `metric_depth_not_available`；未复制 B1 指标 |

B1 的 9761 个 `grasp_contact` 仿真帧来自 kinematic object pose override 下的几何接触诊断，不是合格物理抓取证据；`maximum_lift_m` 仍为 0，且成功判定被 simulation mode gate 否决。

## 跟踪与阶段

`lk_point_availability_ratio` 为 1.0，共推断 4 个阶段：approach 0.000–0.033 s（confidence 0.0156961）、hold 0.067–5.200 s（1.0）、release 5.233–5.500 s（0.0）、settle 5.533–6.967 s（0.0）。这个比例只表示 LK 点存在非零 confidence/forward-backward availability，不是 semantic accuracy。人工视觉 QA 明确观察到放置后 LK 点从罐体漂到手上，因此 B1 endpoint/path 均不可靠；本轮没有像素级罐体中心 ground truth，can-center checkpoint error 为 `not_measured`。B0/B1 metrics 均明确记录 `perception_status=degraded` 和两个 `zero_confidence_phase` warnings；没有人工把它们改成更好看的时间段。机器可读视觉 QA 见 `semantic-assessment.json`。

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

## Fix round 2/5：可信所有权与并发发布

review 指出白名单文件名不是目录所有权证明，而且首次验证与约 10 秒后的 swap 之间存在 TOCTOU。修复采用 v2 `run_manifest.json`：marker 固定 producer，并记录每个生成文件的 byte size 与 SHA-256；无 producer/digest 的 v1 marker 无法证明归属，安全拒绝且不做替换。

RED 覆盖个人 `metrics.json`、伪 v1 marker、运行中 content mutation、同路径并发运行、staging swap/rollback、backup cleanup、backup 中途加入用户文件，以及 rejected marker 携带 stale action。round 2 当时的锁以 lexical absolute path 为 key，只证明相同路径字符串的串行化；物理 alias 与 staging rename post-success error 尚未覆盖，见 round 3。发布瞬间会再次核对目录 identity 与完整内容 snapshot。backup 仅在再次验证后逐个 unlink 已签名普通文件，绝不递归删除；任何变异 backup 被隔离保留。

真实 B0–B4 重新运行并发布 v2 marker。runtime 分别为 B0 7.1053832 s、B1 12.8797578 s、B2 0.0335064 s、B3 0.0560038 s、B4 0.0456489 s。物理结论不变：B0 reachability=0.0235655737704918、双指接触 0、lift=0、not placed；B1 仍为 rejected kinematic diagnostic；B2–B4 仍分别为 `metric_depth_not_available`。B0/B1 无 action，B2–B4 目录隔离，无 staging/backup 残留；六个 MP4 ffprobe duration 仍为 B0 3.9/7.0/7.0 s 与 B1 24.0/7.0/7.0 s。

最终验证：experiment regression 28 passed；完整 suite 64 passed in 41.45 s；compileall exit 0。

## Fix round 3/5：staging post-success reconciliation 与物理 alias lock

新增 fresh/existing 两个真实 RED：`staging.replace(destination)` 已完成物理 rename 后抛异常时，前者误报 changed-during-run 且没有 publication rejection，后者错误尝试把旧 backup 覆盖到已存在的新 canonical，最终同样没有 rejection。修复在 swap 前保存可信 staging identity/file snapshot；异常后只可能按 snapshot 把 canonical 判定为已发布 staging、原 run、runner 新建空目录或外部替换。前三种写明确 failure，外部替换不写不删。

round 3 的 `_mark_canonical_publication_failure` 接收并重新验证 runner-owned expected snapshot，能拒绝 helper 调用前已发生的 canonical replacement；该 precheck 与随后 pathname mutation 之间仍有竞态，见 round 4。lock key 改为 resolved parent 的 `st_dev/st_ino` 加 normalized basename，不再依赖 lexical path。Windows 无 symlink 权限时用 `\\?\` 物理别名进行真实双进程测试；第二进程到达 lock 前置 ready barrier 后，直到第一进程释放才可进入。

本轮不改变 tracking/simulation/rendering 或 artifact schema，因此未重跑真实视频。现有 B0–B4 均通过当前 v2 manifest/hash validator：B0/B1 仍为 rejected 且无 action，B2–B4 仍为独立 `metric_depth_not_available`，无 staging/backup。六个 MP4 ffprobe duration 保持 B0 3.9/7.0/7.0 s、B1 24.0/7.0/7.0 s。最终 experiment regression 32 passed，完整 suite 68 passed in 44.76 s，compileall exit 0。

## Fix round 4/5：failure marker 的 check/use 隔离

新增确定性 RED 在 expected snapshot 返回成功后、首次 action unlink 前替换 canonical，并放入个人 action/metrics/rejection；旧 helper 未报错且删除/覆盖个人文件。第二个 RED 在 runner-owned working 隔离后由外部目录占据 canonical，约束 restore 不得覆盖个人内容，并要求 runner failure 留在 actionless sibling working。

failure reporting 现在先把 canonical 原子移动到唯一 `.failure-working-<uuid>` sibling，再对移动后目录重新核对原 expected identity/content。若移动的是外部 replacement，候选只会在 canonical 仍空时原样恢复；否则保留在唯一 working 位置并停止，不 unlink 或覆盖其中任何文件。只有移动后仍精确属于 runner 的目录才会在 working pathname 下移除 action、写 failed metrics/rejection/manifest；最终仅在 canonical 未被占据时移回。若 canonical 已被外部占据，个人目录保持原样，runner-owned failed working 被保留且不含 action。任何这些分支都不递归删除目录。

本轮仍未改变真实数据链路或 artifact schema，因此未重跑 B0–B4。当前 validator/hash audit 仍确认 B0/B1 rejected 且无 action，B2–B4 为 `metric_depth_not_available`，无 staging/backup/failure-working 残留；六个 MP4 duration 均 >0。最终 experiment regression 34 passed，完整 suite 70 passed in 36.83 s，compileall exit 0。

## Final review fix：语义、碰撞门禁、provenance 与状态契约

本轮按 TDD 增加并先运行 RED，覆盖：旧 `valid_track_ratio` 命名、缺少 generator/config/model/runtime/ffprobe provenance、一个 otherwise-passing physics verdict 错误导出 action、未知或缺少 reason 的 terminal status 被 trusted validator 接受，以及 completed run 缺少 collision validation。GREEN 后 runner 统一使用 `completed|not_run|rejected|failed` terminal vocabulary；感知降级另写 `perception_status=degraded`。manifest 升级到 format v3，并把 terminal status/reason/action absence 约束纳入 trusted-run 验证。

当前 primitive XML 的 arm/hand collisions 与 self/table/box/penetration validation 不完整。因此 runner 硬性设置 `collision_validation=not_implemented`、`action_export_eligible=false`、`action_export_reason=collision_validation_not_implemented`；即使构造 otherwise-passing physics metrics 也不能导出 `actions.npz`。B0 的 primary rejection 仍如实保留 `physics_validation_failed`，碰撞门禁作为独立机器字段记录。

生成器先以 commit `74e80827294ec156fb210377e2ef33ba74ec6e0d` 提交并确认 clean，随后从真实源视频重跑 B0–B4。每个 `provenance.json` 均记录该 commit、`git_dirty=false`、解析后的配置及 SHA-256、source/model SHA-256、Python/依赖/MuJoCo/OpenCV 版本，以及 source/generated media 的 ffprobe stream facts。最终 runtime 为 B0 6.8865752 s、B1 10.3163155 s、B2 0.2161542 s、B3 0.2193229 s、B4 0.1941702 s。

发布审计确认：B0/B1 manifest format v3，状态均 rejected 且无 action；B2–B4 分别为 `not_run/metric_depth_not_available`；五个目录合计 `actions.npz` 为 0。六个 MP4 均由 ffprobe 验证可解码且 duration>0：B0 replay/side-by-side/overlay 为 3.9/7.0/7.0 s，B1 为 24.0/7.0/7.0 s。

视觉 QA 结论单独版本化到 `semantic-assessment.json`：这是人工目视结论，不是像素级标注。放置后 LK 点从罐体漂到手上；`lk_point_availability_ratio=1.0` 仅表示非零 point confidence/forward-backward availability，不表示 semantic accuracy。B1 endpoint/path 均不可靠；由于没有像素级 can-center ground truth，checkpoint error 明确为 `not_measured`。

# WebVideo to Data

把短视频转换为可检查的机器人参考与仿真证据。当前 Phase 2A 的结论是“可信拒绝”，不是可用 action data：固定物理基线 30 次全部失败，B1 也只是运动学诊断，整个 fresh suite 导出 action 数为 0。

## Python 3.11 快速开始

B0 不读取私有视频，因此下面的公开流程可在没有源素材的 clean checkout 中运行。

仓库里的 `configs/exp001.yaml` 和 `configs/sources.yaml` 只放了合法的占位路径与 64 位零哈希，不包含团队私有素材的真实文件名或内容哈希。运行 B1 前，需要在本地同时替换两个文件中的 `source.path`/`sources[].path` 和 `sha256`，并保持 source ID 一致。B0 不读取这些字段指向的文件，也不需要为它填入私有值。

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
& ".venv\Scripts\Activate.ps1"
python -m pip install uv
uv sync --active --frozen --extra dev
webvideo-to-data preflight --config configs/exp001.yaml --variant B0 --no-render
webvideo-to-data run --config configs/exp001.yaml --variant B0 --no-render
```

POSIX shell：

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install uv
uv sync --active --frozen --extra dev
webvideo-to-data preflight --config configs/exp001.yaml --variant B0 --no-render
webvideo-to-data run --config configs/exp001.yaml --variant B0 --no-render
```

命令输出会给出新建的 append-only run path。对 suite 或单个 variant 都使用同一个严格验证入口：

```text
webvideo-to-data verify --run <run-path> --decode-media --privacy-audit
```

公开 CI 在 Windows 与 Ubuntu 上固定 Python 3.11，执行 compileall、排除 acceptance/private-video 的完整公开测试、lockfile 安装和 diff 检查。Ubuntu 渲染测试使用 EGL。

## Phase 2A 实测结论

事实源是 fresh verified enhanced suite `20260822T091027937019Z-93982cd7-b106`，逻辑相对路径为 `EXP-001-phone-can-mujoco/runs/20260822T091027937019Z-93982cd7-b106`。

| 证据 | 已验证结果 |
| --- | --- |
| B0 固定物理基线 | seeds 19–48，`0/30` success，Wilson 95% 区间 `[0, 0.11351339317396876]` |
| B0 非法接触 | 共 `18,660` 个 forbidden-contact frames；最大 penetration `0.01252896243320321 m` |
| B0 terminal | `rejected / physics_validation_failed`；不是物理成功，也不是 action data |
| B1 terminal | `rejected / kinematic_replay_not_action`；对象位姿覆盖只用于诊断 |
| B2–B4 | 本次 B0+B1 narrower run 未请求；没有用 B1 指标填充 |
| action gate | `actions_exported=0`，suite 内没有 `actions.npz` |

B0 的单次 fresh child 同样失败：execution tracking ratio 为 `0.13316892725030827`，forbidden contacts 为 `161`，最大 penetration 为 `0.0045891964780619895 m`，并有速度与加速度越界。即使它产生过高度变化，也没有通过完整物理门禁。

B1 的 `lk_point_availability_ratio=1.0` 只表示点可用性，不表示语义正确。人工视觉 QA 已观察到后段 hand takeover drift；semantic accuracy 仍是 `not_measured`。它的 height gain `0.185 m` 来自 kinematic object-pose override，`maximum_lift_m=0.0`，不能称为物理抓取。

## 公开的 simulation-only 诊断

两张动画只含 MuJoCo 场景，由 fresh suite 中满足 `media_role=public_simulation_preview` 且 `contains_private_source_frames=false` 的 manifest entry 通过 guarded copy API 发布。每帧保留 rejection、warning 和 simulation clock；原始视频、tracking/source panel、run-local dashboard 与 release MP4 均未发布。

### B0：rejected manual physics baseline

![B0 simulation-only rejected diagnostic](docs/media/exp001-b0-side-by-side.gif)

### B1：rejected kinematic replay

![B1 simulation-only rejected diagnostic](docs/media/exp001-b1-side-by-side.gif)

动画便于检查失败模式，不改变 terminal verdict，也不证明数据可用于机器人训练。

## 数据边界

流水线把产物分为四层：

```text
video evidence
  -> reconstructed motion
  -> robot reference
  -> simulation validation or explicit rejection
```

二维 tracking availability、IK reachability、运动学回放和漂亮动画都不是物理 action 的替代品。只有通过配置锁定、来源校验、动力学、碰撞与接触门禁、manifest 验证及隐私审计的 episode 才可能进入 action-bearing 数据集。失败样本保留为诊断证据，但不会静默降级门槛。

详细方法见 [`docs/method.md`](docs/method.md)，机器可读结果见 [`experiments/EXP-001-phone-can-mujoco/metrics.json`](experiments/EXP-001-phone-can-mujoco/metrics.json)，实验叙述见 [`report.md`](experiments/EXP-001-phone-can-mujoco/report.md)。

## Phase 2B 建议

先使用现有数据完成两项有边界的 A/B：

1. 制作罐体 mask/检查点标注，比较 mask-gated SAM2/CoTracker 与当前 LK，明确 hand takeover drift 是否下降；
2. 在 Panda 物理基线中加入 grasp-feasibility 与 contact-aware control，保持当前 30-seed 门禁不变。

普通单目视频不支持可靠的公制三维结论。若要报告 metric 3D error，应新增经过标定的 RGB-D/双目数据，或重新录制带内外参与实物几何测量的受控视频；不能用类别先验或 B1 replay 数值冒充测量。

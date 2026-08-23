# EXP-001：phone-can 到 MuJoCo 的可信基线

状态：fresh enhanced suite 已记录，但没有 action。B0 `rejected / physics_validation_failed`，B1 `rejected / kinematic_replay_not_action`，B2–B4 在这次 B0+B1 narrower run 中未请求。

## 可复核身份

- logical run ID：`20260822T091027937019Z-93982cd7-b106`
- relative run path：`EXP-001-phone-can-mujoco/runs/20260822T091027937019Z-93982cd7-b106`
- requested variants：`B0`, `B1`
- suite status：`recorded / suite_recorded`
- action contract：`actions_exported=0`，无 `actions.npz`

本页不公开私有源的文件名、本地位置或画面。B0 的 provenance 明确记录 `source_accessed=false`；B1 只在 registry ID 与 SHA strict preflight 通过后运行。

## 结论

| 分支 | 结果 |
| --- | --- |
| B0 fixed benchmark | seeds 19–48，0/30；18,660 forbidden-contact frames；最大 penetration 0.01252896243320321 m |
| B0 child | tracking 0.13316892725030827；161 forbidden contacts；penetration 0.0045891964780619895 m；rejected |
| B1 child | LK availability 1.0 但 semantic accuracy not measured；hand takeover drift；kinematic rejected |
| B2–B4 | not requested；没有 B1 指标或 metric 3D claim |

固定 B0 门槛没有调低，failed seeds 没有丢弃。`maximum_lift_m`、height gain、IK 或动画单独好看都不能覆盖 contact/collision/dynamics/placement gate。

## 公开媒体

仓库 `docs/media` 中的 `exp001-b0-side-by-side.gif` 与 `exp001-b1-side-by-side.gif` 是 guarded-copy 的 simulation-only GIF。两者 manifest 均为 `public_simulation_preview`、`contains_private_source_frames=false`；逐帧解码、ffprobe、privacy audit 与视觉检查通过。

完整 fresh suite、child、private diagnostics 与 dashboard 均保留为本地 ignored evidence，不是公开 artifact。详细数值见 [`metrics.json`](metrics.json)，叙述与验证见 [`report.md`](report.md)，语义 QA 边界见 [`semantic-assessment.json`](semantic-assessment.json)。

## 下一步

- 为罐体制作 mask/checkpoint annotation，比较 mask-gated SAM2/CoTracker 与 LK；
- 在 pinned Panda 场景研究 grasp-feasibility/contact-aware control，不改现有 30-seed 门禁；
- metric 3D 使用校准 RGB-D/双目或新的受控标定录制，不用单目先验代替测量。

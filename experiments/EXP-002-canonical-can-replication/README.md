# EXP-002：different-source canonical replication

状态：suite 已记录，没有 action。B0 固定物理基线为 `0/30`，B0 与 B1 都被拒绝。

## 实验身份

- logical run ID：`20260823T083932722546Z-aac0791e-aa8b`
- relative run path：`EXP-002-canonical-can-replication/runs/20260823T083932722546Z-aac0791e-aa8b`
- generator commit：`bf6615c1bf15f476ca6c51e9f20bbda69d6549a7`，dirty `false`
- requested variants：`B0`, `B1`
- suite status：`recorded`
- action contract：`actions_exported=0`，没有 `actions.npz`

这是换用另一段私有录制素材的 replication，不是 EXP-001 的同源重跑。公开记录不含源素材的标识、文件名、本地路径、内容哈希或画面。

## 证据来源

`different_source` 是运行前确定的实验设计，不是从结果 JSON 反推的标签。run ID、terminal、B0/B1 数值和公共 GIF 字段来自公开 safe summary。suite、B0 和 B1 的 verify、媒体解码与隐私审计结论来自 reviewer/controller 对 immutable suite 的独立 strict CLI 复跑。这些来源在 [`metrics.json`](metrics.json) 的 `evidence_basis` 中分开记录。

## 结果

| 分支 | 已验证结果 |
| --- | --- |
| B0 fixed benchmark | `0/30`，status `failed` |
| B0 child | `rejected / physics_validation_failed`；161 个 forbidden contacts；最大 penetration `0.0045891964780619895 m` |
| B1 child | `rejected / kinematic_replay_not_action`；reachability `0.0969085909507541`；release/settle confidence 为 0 |
| B2–B4 | `not_requested` |

B1 的 `lk_point_availability_ratio=1.0` 只表示点在跟踪流程中可用。semantic accuracy 没有测量，不能据此认定轨迹仍落在目标物体上。

EXP-002 和 EXP-001 使用了不同源素材，因此不能把 runtime 或其他数值的升降归因于算法变化。本次 generator 就是 C1，没有算法改动。

## 公开媒体

本次 suite 从 C1 重新生成了两张 simulation-only GIF。它们与仓库已有的 EXP-001 公共 GIF 逐字节一致，因此直接复用原链接，不再提交约 22 MB 的重复文件：

![B0 rejected simulation preview](../../docs/media/exp001-b0-side-by-side.gif)

![B1 rejected simulation preview](../../docs/media/exp001-b1-side-by-side.gif)

两张 GIF 的 manifest 分类都是 `public_simulation_preview`，`contains_private_source_frames=false`。逐帧解码与隐私审计通过。它们只显示仿真诊断，不含私有源帧，也不改变 rejected verdict。

完整数值见 [`metrics.json`](metrics.json)，语义边界见 [`semantic-assessment.json`](semantic-assessment.json)，执行记录见 [`report.md`](report.md)。

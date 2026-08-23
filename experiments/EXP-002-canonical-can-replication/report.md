# EXP-002 different-source replication 报告

## 摘要

实验用 C1 `bf6615c1bf15f476ca6c51e9f20bbda69d6549a7` 和另一段私有录制素材重新执行严格 B0+B1 流程。suite terminal 为 `recorded`。B0 固定评测 `0/30`，B0/B1 terminal 均为 rejected，action 数为 0。

这次实验回答的是“同一实现换源后能否产出通过门禁的数据”。它不是 EXP-001 同一素材上的回归测试，也没有包含算法修改。跨实验的数值差异混合了输入变化，不能解释成算法进步或退步。

## 证据来源

`different_source` 是执行前写入的实验设计。公开 safe summary 提供 run ID、suite status、B0/B1 指标、action 数和 GIF 字段。verify、媒体解码与隐私审计结论由 reviewer/controller 使用 immutable suite 独立复跑 strict CLI 得到，不属于 safe-summary projection。

## Suite aggregate

| 字段 | 值 |
| --- | --- |
| logical run ID | `20260823T083932722546Z-aac0791e-aa8b` |
| status | `recorded` |
| requested variants | `B0`, `B1` |
| B0 fixed benchmark | `failed`, `0/30` |
| B0 terminal | `rejected / physics_validation_failed` |
| B1 terminal | `rejected / kinematic_replay_not_action` |
| actions exported | `0` |

`recorded` 表示 suite 已完整记录并通过验证，不表示抓取成功。

## B0 physics baseline

| 指标 | 值 |
| --- | ---: |
| runtime | 3.266457600002468 s |
| execution tracking ratio | 0.13316892725030827 |
| bilateral close contact | 0.052 s |
| bilateral lift contact | 0.14 s |
| maximum lift | 0.22670993268105838 m |
| target error | 0.004783042194876456 m |
| forbidden contacts | 161 |
| maximum penetration | 0.0045891964780619895 m |
| joint velocity violations | 37 |
| joint acceleration violations | 1201 |
| collision / physics validation | failed / failed |
| placed successfully | false |

正的 lift 与较小的 target error 没有覆盖接触、碰撞和动态门禁。B0 未获得 action export 资格。

## B1 kinematic diagnostic

| 指标 | 值 |
| --- | ---: |
| runtime | 17.642936800002644 s |
| reachability ratio | 0.0969085909507541 |
| LK point availability | 1.0 |
| maximum can height gain | 0.185 m |
| maximum physical lift | 0.0 m |
| target error | 0.0599163907205696 m |
| target height error | 0.193 m |
| support contact duration | 0.0 s |
| placed successfully | false |

LK 指标的 scope 是 `point_availability_not_semantic_accuracy`。semantic accuracy 为 `not_measured`，release 与 settle 两个 phase 的 confidence 为 0，perception status 为 `degraded`。对象位姿来自 kinematic replay，physics/collision validation 都是 `not_applicable_kinematic`，因此该分支不能导出 action。

## 验证与媒体

以下结论来自 reviewer/controller 对 immutable suite 的 strict CLI 复跑，不是 safe-summary projection：suite、B0 和 B1 分别通过 `--decode-media --privacy-audit` 验证，解码媒体数为 8、1、5。verifier 捕获字节与 manifest 的 hash、size 和分类一致；递归检查确认 `actions.npz` 数量为 0。

| Public preview | SHA-256 | Probe/decode |
| --- | --- | --- |
| [B0 GIF](../../docs/media/exp001-b0-side-by-side.gif) | `9730164ea6ae93904d352592d0209bad50d01b0cf86fe93ce88db65fd2efb395` | 960×720, 25/3 fps, 78 frames, 9.36 s |
| [B1 GIF](../../docs/media/exp001-b1-side-by-side.gif) | `918f236e6268ba811efcef7a65499ae170ab457cc38e990daa95101d8577fa11` | 960×720, 25/3 fps, 96 frames, 11.52 s |

这两张图由本次 C1 suite 重新生成，捕获字节与仓库现有 EXP-001 文件一致。仓库复用已有文件，避免重复提交 22,134,240 bytes。两张图只含仿真画面，manifest 标记 `contains_private_source_frames=false`，隐私审计通过。

## 结论

换源后，当前实现仍未生成可进入 action-bearing dataset 的 episode。B0 的固定物理基线没有通过，B1 仍停留在语义准确率未测的运动学诊断层。EXP-002 为下一轮同源 A/B 提供另一条失败记录，但不能用于声称算法改进。

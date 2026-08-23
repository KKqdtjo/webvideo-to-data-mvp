# EXP-001 Phase 2A 实测报告

## 摘要

fresh enhanced suite `20260822T091027937019Z-93982cd7-b106` 完成 B0+B1 并通过递归 manifest、逐帧媒体 decode 与 privacy audit。结果不是成功数据集：B0 固定物理基线为 `0/30`，B0/B1 terminal 均为 rejected，`actions_exported=0`，递归没有 `actions.npz`。

逻辑相对路径：

```text
EXP-001-phone-can-mujoco/runs/20260822T091027937019Z-93982cd7-b106
```

本报告只引用 fresh verified JSON 和公开 simulation-only copy。它不包含私有源文件名、本地路径、dashboard 地址或 source/tracking frames。

## Suite aggregate

`suite-metrics.json` 记录：

| 字段 | 值 |
| --- | --- |
| status / reason | `recorded / suite_recorded` |
| requested variants | `B0`, `B1` |
| B0 physics baseline | `failed` |
| B0 rollouts / successes | `30 / 0` |
| B0 terminal | `rejected / physics_validation_failed` |
| B1 terminal | `rejected / kinematic_replay_not_action` |
| actions exported | `0` |

`recorded` 只表示 suite 已完整落盘并验证，不表示 action 或物理成功。

## B0 fixed 30-seed benchmark

`benchmark-summary.json` 是 seeds 19–48 的固定 aggregate：

- successes：`0/30`；
- Wilson 95% interval：`[0.0, 0.11351339317396876]`；
- reason：`illegal_contact_observed`；
- forbidden-contact frames：`18,660`；
- maximum forbidden penetration：`0.01252896243320321 m`；
- yaw observability：`geometrically_unobservable_for_axisymmetric_can`。

原物理 acceptance threshold 没有调整。30 个 seed 都保留；该 acceptance 作为明确的 expected xfail，独立 no-actions acceptance 通过。

fresh B0 child 为 source-independent `physics_grasp`，provenance 记录 `source_accessed=false`。其 terminal metrics：

| 指标 | 值 |
| --- | ---: |
| runtime | 3.4659598000007463 s |
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

正的 lift 或较小 target error 不能单独构成成功；tracking、close contact、collision、penetration 与动态门禁均失败，所以 action 不合格。

## B1 kinematic diagnostic

B1 通过 strict registry/hash preflight 后运行。它的 `simulation_mode=kinematic_replay`，对象位姿由 override 驱动；collision/physics 均为 `not_applicable_kinematic`。

| 指标 | 值 |
| --- | ---: |
| runtime | 17.69941590000053 s |
| reachability ratio | 0.0969085909507541 |
| LK point availability | 1.0 |
| maximum can height gain | 0.185 m |
| maximum physical lift | 0.0 m |
| target error | 0.0599163907205696 m |
| target height error | 0.193 m |
| support contact duration | 0.0 s |
| placed successfully | false |

availability 的 scope 是 `point_availability_not_semantic_accuracy`。人工视觉 QA 观察到后段 hand takeover drift；semantic accuracy 为 `not_measured`。release 与 settle 都有 zero-confidence warning。B1 的轨迹可用于暴露感知与映射问题，不能用于 action 或公制三维结论。

## B2–B4

本次 fresh suite 是明确的 B0+B1 narrower run，因此 B2、B3、B4 都是 `not_requested`。现有数据没有 calibrated metric depth；没有把 B1 的 reachability、height 或 target error 复制为 B2–B4 结果。

## Media audit

所有 suite-manifest media 先验证 hash、size、role 与 private-frame flag。公开 GIF 只能由 `copy_public_preview()` 复制：

| Public copy | Manifest classification | SHA-256 | Probe/decode |
| --- | --- | --- | --- |
| B0 GIF | `public_simulation_preview`, private frames `false` | `9730164ea6ae93904d352592d0209bad50d01b0cf86fe93ce88db65fd2efb395` | 960×720, 25/3 fps, 78/78 frames, 9.36 s |
| B1 GIF | `public_simulation_preview`, private frames `false` | `918f236e6268ba811efcef7a65499ae170ab457cc38e990daa95101d8577fa11` | 960×720, 25/3 fps, 96/96 frames, 11.52 s |

fresh manifested MP4 facts：B0 replay 为 MPEG-4 320×240、10 fps、97/97 frames、9.7 s；B1 overlay 为 540×960、30 fps、210/210、7 s，replay 为 320×240、10 fps、240/240、24 s，private comparison 为 960×320、30 fps、210/210、7 s。所有 MP4/GIF 的 full decode exit 为 0，但只有上述两张 simulation-only GIF 被公开。

resolved suite 与 B1 child 分别执行：

```text
webvideo-to-data verify --run <resolved-run> --decode-media --privacy-audit
```

两次均 exit 0。public-copy directory 的 privacy findings 也为 0。视觉检查覆盖 midpoint 与多帧 contact sheet：每帧只有 MuJoCo robot/table/can/box，保留 `REJECTED — NOT ACTION DATA`、完整 warning 和 simulation clock；没有源视频或 tracking panel。

## 结论与 Phase 2B

Phase 2A 建立了可信的拒绝链路，但未建立可用 action。下一轮先做 mask/checkpoint annotation，A/B mask-gated SAM2/CoTracker 与 LK；同时在 pinned Panda 场景研究 grasp-feasibility 与 contact-aware control，不修改当前 30-seed gate。

任何 metric 3D claim 都需要 calibrated RGB-D/双目，或新的带相机标定与实物几何测量的受控录制。现有单目素材不足以支持这种结论。

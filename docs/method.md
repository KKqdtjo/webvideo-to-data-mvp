# Phase 2A 方法与换源复现

## 研究边界

EXP-001 检查“短时单物体视频证据能否成为 Panda 场景中的可信机器人数据”。Phase 2A 不训练策略，也不把二维轨迹或运动学 replay 当作 action。输出只有两种：通过全部物理与安全门禁的候选 episode，或带机器可读原因的拒绝。本轮只有后者。

fresh evidence 来自 logical run `20260822T091027937019Z-93982cd7-b106`，相对路径 `EXP-001-phone-can-mujoco/runs/20260822T091027937019Z-93982cd7-b106`。suite 只请求 B0 与 B1；B2–B4 未请求。

EXP-002 在 C1 `bf6615c1bf15f476ca6c51e9f20bbda69d6549a7` 上换用另一段私有录制素材，logical run 为 `20260823T083932722546Z-aac0791e-aa8b`。它仍只请求 B0 与 B1。该实验是 different-source replication，不是 EXP-001 同源重跑；数值变化同时受输入变化影响，不能用于评价算法改进。

## 锁定输入与分支

私有源由 registry logical ID、SHA-256 和 strict preflight 绑定；公开文档不记录本地文件名或路径。B0 是 source-independent：它不得 probe、hash、decode 或 render 私有源。B1 只有在 registry ID 与 hash 都通过时才可读取源。

| Variant | 输入与用途 | 本轮 terminal |
| --- | --- | --- |
| B0 | 固定 canonical start/goal、合成 phases、pinned MuJoCo Menagerie Panda 物理控制 | `rejected / physics_validation_failed` |
| B1 | LK ROI tracking、自动 phases、canonical 2D-to-scene object-pose override | `rejected / kinematic_replay_not_action` |
| B2 | metric depth 与尺度 | `not_requested`；metric depth 不可用 |
| B3 | B2 + 自动 contact | `not_requested` |
| B4 | B3 + 几何与物理约束 | `not_requested` |

## B0 固定物理评测

B0 使用不可调的 seeds 19–48，共 30 次 rollout。每个 seed 独立扰动初始 xy、质量、摩擦和 yaw；轴对称罐体的 yaw 扰动被记录为几何不可观测。门禁同时检查：

- execution tracking；
- close/lift 阶段双指接触持续时间；
- lift、target error、settle、最终 tilt/speed；
- phase-aware body-ancestry collision 与 penetration；
- joint position/velocity/acceleration；
- numerical validity 与 placement。

所有 30 次均失败，`successes=0`，Wilson 95% 区间 `[0, 0.11351339317396876]`。聚合共有 `18,660` 个 forbidden-contact frames，最大 penetration `0.01252896243320321 m`，reason 为 `illegal_contact_observed`。阈值、seed 和失败记录均未为结果调节或丢弃。

fresh B0 child 的主要值为：tracking `0.13316892725030827`、close contact `0.052 s`、lift contact `0.14 s`、forbidden contacts `161`、最大 penetration `0.0045891964780619895 m`、velocity violations `37`、acceleration violations `1201`。`maximum_lift_m=0.22670993268105838` 单独看为正，但完整 gate 失败，所以 terminal 仍是 rejection。

## B1 感知与运动学诊断

B1 使用 Lucas–Kanade forward/backward availability。`lk_point_availability_ratio=1.0` 的 metric scope 明确是 `point_availability_not_semantic_accuracy`。人工视觉 QA 观察到后段点从罐体漂到手部；罐体中心 checkpoint ground truth 未制作，因此 semantic accuracy 是 `not_measured`，不是 1.0。

B1 的 reachability 为 `0.0969085909507541`，height gain 为 `0.185 m`，target error 为 `0.0599163907205696 m`。对象位姿由 kinematic override 驱动，physics/collision validation 均为 `not_applicable_kinematic`，`maximum_lift_m=0.0`，release 与 settle phase confidence 为 0。该分支只能诊断时序与映射，不能证明抓取、接触语义或可执行性。

EXP-002 的 B1 也得到 `lk_point_availability_ratio=1.0`，scope 仍为 `point_availability_not_semantic_accuracy`。semantic accuracy 是 `not_measured`，release 与 settle confidence 为 0，terminal 为 `rejected / kinematic_replay_not_action`。这次没有发布可支持跨源语义误差比较的 checkpoint 或 mask 真值。

## Action 与 artifact 合同

suite terminal 是 `recorded`，只表示证据记录完成。B0/B1 都是 `rejected`；`actions_exported=0`，递归检查没有 `actions.npz`。

每个 child 用 v4 manifest 绑定 config、model、逻辑 source identity、schema、hash、size 与 terminal semantics。suite v1 manifest 再递归绑定 child、固定 B0 JSONL/aggregate、environment、public resolved config、dashboard 与 media classification。`verify --run` 根据稳定 manifest identity 在 suite verifier 与 variant verifier之间唯一分派；both/neither、reparse、identity mutation 或选中 verifier 失败都 fail closed，且不会 fallback。

`--decode-media` 对 manifest 中的 GIF/MP4 做 ffprobe，并逐帧验证数量、尺寸和 duration；`--privacy-audit` 审计整个所选目录。PNG audit 只在完整验证 signature、chunk order、length、CRC、filter/interlace、精确解压尺寸、IEND/no-trailer 与资源预算后排除 IDAT 压缩载荷；所有定义的 text/profile/EXIF metadata 仍按格式扫描和有界解压，malformed/unknown layout 回到 raw fail-closed。decoded raster 是数值像素，不套用 generic ASCII path regex；可见文字由 simulation-only 生成约束和 `view_image` 检查，若未来自动化则应使用 bounded OCR。

## 公开媒体

公开 GIF 只能由 `copy_public_preview()` 从 `public_simulation_preview` entry 复制，且 manifest 必须声明 `contains_private_source_frames=false`。本轮 B0 与 B1 公共 GIF 都是 `960×720`、`25/3 fps`：B0 是 78 帧，时长 9.36 秒；B1 是 96 帧，时长 11.52 秒。逐帧解码与 privacy audit 均通过；画面只含 MuJoCo robot/table/can/box，并永久显示 `REJECTED — NOT ACTION DATA`、variant warning 与 simulation clock。

EXP-002 从 C1 suite 重新生成的 B0/B1 GIF 与这两张文件逐字节一致，hash、size、帧数和时长均相同。公开记录直接复用现有链接，避免再提交 22,134,240 bytes。这个字节一致性只说明发布预览相同，不表示不同源视频的感知结果或语义准确率相同。

私有 source/tracking/contact-sheet 媒体、run-local dashboard 和 release MP4 不复制、不链接。公开 GIF 不能被解释为 action success。

## 复现与安全附录

runner 使用 append-only run ID、per-experiment lock、不可变 manifest 和 monotonic latest pointer。variant publication 与 public preview copy 使用稳定 identity/no-reparse 边界；隐私审计拒绝 symlink/reparse、特殊文件、路径 race、未验证容器和敏感 metadata。CLI 的错误文本与 JSON 在输出前统一 redaction，不持久化 token、Authorization、signed query 或本地绝对路径。

CI 使用 Windows/Ubuntu、Python 3.11 与 frozen uv lock。公开 selection 不需要私有源：

```text
uv run python -m compileall -q src tests
uv run pytest -m "not acceptance and not private_video" -q -p no:cacheprovider
git diff --check
```

物理 acceptance 保留原阈值并以已裁决的 expected xfail 报告；独立 no-actions acceptance 必须通过。

## Phase 2B

第一步制作罐体 mask 与 checkpoint annotation，在同一素材上比较 mask-gated SAM2/CoTracker 和 LK，直接测 hand takeover drift。第二步在 pinned Panda 场景中研究 grasp-feasibility 与 contact-aware control，同时保持 seeds 19–48 和现有门禁冻结。算法 A/B 必须使用同源输入；EXP-001 与 EXP-002 的横向数值不承担这个用途。

若要进入 B2–B4 并报告 metric 3D，必须提供校准 RGB-D/双目，或新录制带相机标定与实物几何测量的受控视频。单目类别先验只能是探索假设，不能作为公制真值。

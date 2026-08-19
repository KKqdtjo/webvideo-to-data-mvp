# EXP-001：手机罐体视频到 MuJoCo 诊断回放

状态：已完成单次真实视频可行性实验；B0/B1 均被验证门拒绝，B2–B4 未运行，没有导出 action。这里的模型是硬编码几何的 primitive 7-DoF Panda-like diagnostic model，不是官方 Franka Panda 模型，也不是经过完整碰撞验证的机器人。

## 输入

- `video/手机录制.mp4`
- 锁定首帧 ROI `[374,423,104,155]`
- 固定 canonical scene bounds、B0 start/goal 与合成 reference phases

本实验没有制作像素级罐体中心标注、contact ground truth、双标注者数据或实物几何测量。罐体、桌面和纸盒几何来自 primitive XML 的硬编码诊断形状，不是从视频测得。

## 变体

- B0：固定 canonical 起终点 + 合成 reference phases；
- B1：LK point tracks + canonical 2D-to-scene kinematic replay；
- B2：point tracks + 单目深度和尺度；
- B3：B2 + 自动 contact phases；
- B4：B3 + 轨迹和物理约束。

## 实测摘要

- B0 `physics_grasp`：reachability 0.0235656，无双指接触，lift 0，not placed，`physics_validation_failed`。
- B1 `kinematic_replay`：仅 canonical 2D-to-scene 诊断，not placed，`kinematic_replay_not_action`。
- B2–B4：`not_run`，精确原因 `metric_depth_not_available`。
- `lk_point_availability_ratio=1.0` 只表示 LK 点有非零 confidence/forward-backward availability，不表示语义跟踪正确。人工视觉 QA 观察到放置后跟踪点从罐体漂到手上，因此 B1 endpoint/path 均不可靠；罐体中心误差未测量。机器可读结论见 [`semantic-assessment.json`](semantic-assessment.json)。
- 碰撞验证尚未实现；即使其他物理指标将来通过，当前 runner 仍以 `collision_validation_not_implemented` 硬性禁止 action 导出。
- 完整实测、ffprobe 与 concerns 见 [`report.md`](report.md)，机器可读汇总见 [`metrics.json`](metrics.json)。

## Artifact 索引

可再生成的 binary 位于 ignored `artifacts/EXP-001/{B0,B1}/`：tracking overlay、2D trajectory plot、MuJoCo replay、side-by-side、contact sheet，以及 manifest、provenance、reference、simulation 和 rejection diagnostics。B2–B4 目录仅含 manifest/provenance/metrics。B0/B1 均无 `actions.npz`。

## 原设计交付状态

- 已完成：分层 artifacts、primitive 7-DoF Panda-like reference、MuJoCo 诊断回放、`metrics.json` 与失败分类报告。
- 未执行：像素级/双标注者 annotation、实测场景几何、metric-depth B2–B4、20 次扰动评测、完整 collision/self-collision/table/box/penetration validation。
- 不支持：auditable physics-action output。

验收阈值和拒绝规则见 `docs/superpowers/specs/2026-08-19-webvideo-to-data-mvp-design.md`。

# EXP-001：手机罐体视频到 Panda/MuJoCo

状态：已完成单次真实视频可行性实验；B0/B1 均被验证门拒绝，未导出 action。

## 输入

- `video/手机录制.mp4`
- 首轮人工 contact/检查帧标注
- 罐体和纸盒尺寸配置

## 变体

- B0：手工起终点和接触阶段；
- B1：point tracks + 固定深度；
- B2：point tracks + 单目深度和尺度；
- B3：B2 + 自动 contact phases；
- B4：B3 + 轨迹和物理约束。

## 实测摘要

- B0 `physics_grasp`：reachability 0.0235656，无双指接触，lift 0，not placed，`physics_validation_failed`。
- B1 `kinematic_replay`：仅 canonical 2D-to-scene 诊断，not placed，`kinematic_replay_not_action`。
- B2–B4：`not_run`，精确原因 `metric_depth_not_available`。
- 完整实测、ffprobe 与 concerns 见 [`report.md`](report.md)，机器可读汇总见 [`metrics.json`](metrics.json)。

## Artifact 索引

可再生成的 binary 位于 ignored `artifacts/EXP-001/{B0,B1}/`：tracking overlay、2D trajectory plot、MuJoCo replay、side-by-side、contact sheet，以及 provenance、reference、simulation 和 rejection diagnostics。B2–B4 目录仅含 provenance/metrics。B0/B1 均无 `actions.npz`。

## 原设计交付

- 分层 artifacts；
- Panda 参考轨迹；
- MuJoCo 回放视频；
- 20 次扰动评测；
- `metrics.json` 与失败分类报告。

验收阈值和拒绝规则见 `docs/superpowers/specs/2026-08-19-webvideo-to-data-mvp-design.md`。

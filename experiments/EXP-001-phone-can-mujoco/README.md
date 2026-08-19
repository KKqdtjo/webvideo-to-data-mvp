# EXP-001：手机罐体视频到 Panda/MuJoCo

状态：设计完成，待技术设计审阅后实施。

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

## 交付

- 分层 artifacts；
- Panda 参考轨迹；
- MuJoCo 回放视频；
- 20 次扰动评测；
- `metrics.json` 与失败分类报告。

验收阈值和拒绝规则见 `docs/superpowers/specs/2026-08-19-webvideo-to-data-mvp-design.md`。

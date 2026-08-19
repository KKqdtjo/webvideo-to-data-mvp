# 实验记录约定

实验目录只记录事实：运行了什么、产出了什么、为什么失败。方法讨论放在 `docs/method.md`，领域综述放在 `summary.md`。

## 命名

- 实验：`EXP-NNN-short-name`
- 运行：`run-YYYYMMDD-HHMMSS-variant`
- 配置：每次运行保存解析后的完整配置，不依赖 shell 中未记录的默认值

## 每次运行应保存

```text
run.yaml
environment.json
metrics.json
report.md
stdout.log
stderr.log
artifacts/
```

日志在落盘前移除 API key、Authorization header、私有 URL 和 SSH 凭据。

## 状态

| 实验 | 状态 | 目的 |
| --- | --- | --- |
| EXP-000 | completed | 本地素材与远端环境只读盘点 |
| EXP-001 | completed | 手机罐体视频到 primitive 7-DoF Panda-like/MuJoCo 诊断回放；B0/B1 rejected，B2–B4 not_run |

Runner terminal status 统一为 `completed`、`not_run`、`rejected`、`failed`；实验汇总的 `completed` 只表示记录已完成，不表示变体成功或可导出 action。

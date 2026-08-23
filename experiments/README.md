# 实验索引

每个实验目录只保存可公开、可版本化的结论与机器可读汇总。私有源、run-local dashboard、逐帧产物和可再生成的完整 artifact tree 不进入 Git。

建议结构：

```text
EXP-XXX-name/
├── README.md
├── report.md
├── metrics.json
└── annotation-or-assessment.json
```

Terminal vocabulary：

- `recorded`：suite 证据已完整记录，不代表 variant 成功；
- `rejected`：运行完成但未通过 action gate；
- `not_run` / `not_requested`：该方法深度没有执行；
- `failed`：基础设施或 publication 失败。

| ID | 状态 | 结论 |
| --- | --- | --- |
| EXP-000 | recorded | 本地素材与执行环境的只读盘点 |
| [EXP-001](EXP-001-phone-can-mujoco/) | recorded, no actions | fresh B0 固定物理基线 `0/30`，B0/B1 均 rejected，B2–B4 在本次 narrower suite 未请求，action 数为 0 |
| [EXP-002](EXP-002-canonical-can-replication/) | recorded, no actions | different-source replication；B0 `0/30`，B0/B1 均 rejected，action 数为 0 |

EXP-001 的公开动画是 simulation-only rejected diagnostics。`recorded`、可解码媒体或 kinematic replay 都不能替代物理成功。

EXP-002 换用了另一段私有录制素材，generator 仍是 C1。它不是 EXP-001 的同源重跑，跨实验数值变化不能归因于算法。EXP-002 重生成的公共 GIF 与现有 EXP-001 文件逐字节一致，因此只复用链接，没有复制媒体文件。

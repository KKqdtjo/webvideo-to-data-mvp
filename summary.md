# Web Video to Embodied Data：文献与项目整理

更新时间：2026-08-19  
检索范围：论文原文、作者项目页、官方 GitHub 与 NVIDIA 文档。

## 1. 先把问题说清楚

“把视频转成具身数据”至少包含四件不同的事：

1. 从视频中找出操作片段，识别手、物体、工具和接触阶段；
2. 恢复物体、手和相机的二维或三维运动；
3. 把人的运动映射到某一种机器人，并在仿真中检查可达性和接触；
4. 用通过检查的数据训练闭环策略。

这四层不能混为一谈。视频中有一条稳定的物体轨迹，不代表机器人一定抓得住；机器人轨迹能通过逆运动学，也不代表物理回放时物体不会掉落。第一版项目应该保留这些边界，并为每一层记录置信度和失败原因。

建议把产物分成四类：

| 资产类型 | 内容 | 能否称为机器人训练数据 |
| --- | --- | --- |
| `video_evidence` | 原视频、片段边界、关键帧、来源、语义标签 | 否 |
| `reconstructed_motion` | mask、点轨迹、深度、手物接触、物体位姿 | 仍不能直接称为 action data |
| `robot_reference` | 目标机器人末端或关节参考、夹爪状态、IK 结果 | 只能称为伪动作或参考轨迹 |
| `sim_validated_episode` | 在仿真中通过碰撞、接触和任务检查的 observation/action episode | 可以作为仿真训练数据，但仍要标记其来源和 sim-to-real 风险 |

## 2. NVIDIA Video to Data

项目：NVIDIA Isaac Video to Data（V2D）  
仓库：https://github.com/nvidia-isaac/video_to_data  
文档：https://nvidia-isaac.github.io/video_to_data/  
CHORD 论文：https://arxiv.org/abs/2607.00033  
数据集合：https://huggingface.co/collections/nvidia/video-to-data

V2D 是三段式工程流水线：

```text
Video Ingestion Agent
    -> Reconstruction
    -> Robotic Grounding
```

### 2.1 Video Ingestion Agent

这一段把长视频变成可检索的操作片段。LangGraph 工作流先分段，再做可选的 verify/refine，随后抽取实体关系图和 SigLIP-2 帧向量。结果写入 `graph.db`、`vector.db` 和裁切后的视频片段。

它解决的是“长视频中哪一段值得继续处理”，不是三维重建。默认方案需要本地 VLM 服务，仓库文档以 Qwen3-VL-8B 为例，约需 24 GB 显存。系统论文尚在准备中，因此更适合把它当作开源工程参考，而非已有同行评审结论。

文档：https://github.com/nvidia-isaac/video_to_data/blob/main/video_ingestion_agent/README.md

### 2.2 Reconstruction

重建部分集成了深度、内参、检测、分割、物体网格、6D 位姿和人体恢复模块，包括 MoGe、Grounding DINO、SAM2、FoundationPose、BundleSDF 和 SAM3D-Body 等。各模块运行在独立容器中，通过强类型数据契约和磁盘文件衔接。

最小示例可以做 `video -> depth + intrinsics`，但完整 HOI 示例并不是任意单目视频一键运行。官方多视角流程需要 rosbag、相机外参和物体 mesh；FoundationPose 也依赖 mask、depth、intrinsics 和物体几何。

文档：https://github.com/nvidia-isaac/video_to_data/blob/main/reconstruction/README.md

### 2.3 Robotic Grounding 与 CHORD

Grounding 部分把 MANO/SOMA 等人类运动转换为 Sharpa、Dex3 或 G1 等机器人参考，生成 Parquet 轨迹、URDF/USD 资产和可视化结果，再进入 Isaac Lab 与 RSL-RL PPO。

CHORD 的贡献是 object-centric contact wrench guidance。它用机器人接触可能对物体产生的力和力矩效果来比较人类与机器人的操作，而不是要求机器人逐点复制人手轨迹。作者构建了 4,739 个双手仿真任务，在 1,831 个任务上报告 82.12% 平均成功率。该结果来自 arXiv v2（2026-08-14），不是已发表会议论文。

contact wrench 是仿真和 RL 中的物理指导信号，不能从普通 RGB 视频直接可靠读出。它位于流水线后段，不应被包装成一个低成本视频标签。

架构文档：https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/docs/ARCHITECTURE.md

### 2.4 V2D 对本项目最有用的部分

- 模块以文件为边界，任何阶段都可以停止、检查和重跑；
- 中间产物有明确类型，而不是在模块间传裸数组；
- 先筛选片段，再把昂贵模型用在少量候选上；
- 重建、retargeting、仿真验证和策略训练分别报告结果。

当前仓库的完整 HOI 流程验证于 RTX A6000 与 L40S。Blackwell `sm_120` 受 TensorRT/cuVSLAM 版本限制。Grounding 还需要 Git LFS、MANO 许可、外部数据集和 Isaac Lab 环境。这决定了它适合做中期基线，不适合作为当前服务器上的第一项实验。

## 3. 代表性工作

### 3.1 从视频提取可迁移的运动表示

| 工作 | 发表状态 | 输入与输出 | 解决的问题和创新 | 开源情况与限制 |
| --- | --- | --- | --- | --- |
| [Track2Act](https://arxiv.org/abs/2405.01527) / [项目页](https://homangab.github.io/track2act/) / [代码](https://github.com/homangab/Track-2-Act) | ECCV 2024 | 当前图像、目标图像、选定像素点 → 未来 2D point tracks → 物体刚体运动和末端轨迹 | 从互联网视频学习“物体应如何运动”，再用少量机器人数据训练 residual policy | 代码公开。更适合短时刚体操作；点轨迹本身不表达抓取稳定性和力控 |
| [HOWTransfer](https://arxiv.org/abs/2606.10743) | arXiv v1，2026 | 校准双目/多视角人类演示 → 3D wrist、contact interval、平行夹爪 grasp hypotheses | 把 contact onset 当作迁移锚点，并结合手闭合、可见性、邻近性和手物共同运动定位接触 | 当前公开信息以论文为主。输入条件比普通互联网单目视频严格，目标是二指夹爪 |
| [VideoManip](https://arxiv.org/abs/2602.09013) / [项目页](https://videomanip.github.io/) / [代码](https://github.com/hychen-naza/VideoManip) | arXiv v2，2026 | 静态 egocentric 单目 RGB → 手物三维轨迹、机器人手轨迹和策略 | 估计深度、手、mesh 与物体位姿，做 contact optimization 和轨迹合成 | 已发布重建代码；grasp model 与完整 policy 训练代码的公开范围需按仓库最新状态核验。视觉误差会逐级传播 |
| [Do as I Do](https://arxiv.org/abs/2606.19333) / [代码](https://github.com/malik-group/do-as-i-do) | arXiv，2026 | egocentric/exocentric 单目 RGB → 手物重建 → 多指手动作 | 面向日常、网络视频改进遮挡下的 4D HOI 重建，并以采样式 MPC 做 retargeting | 代码按 reconstruction、retargeting、deployment 分层发布。系统较新，依赖多种重建模型与机器人资产 |
| [EasyMimic](https://arxiv.org/abs/2602.11464) / [项目页](https://zt375356.github.io/EasyMimic-Project/) | arXiv，2026 | 普通 RGB 人类视频 + 少量机器人数据 → 低成本 LeRobot 策略 | 提取 3D 手轨迹，映射到夹爪空间，用视觉增强和人机数据共训练缩小域差异 | 适合低成本二指夹爪。仍需要少量机器人数据，不能把人类轨迹原样当 action |

Track2Act 是第一版最合适的思想来源。它允许我们先验证二维物体运动是否稳定，再决定是否值得进入三维重建。HOWTransfer 则提醒我们：contact onset、hold、release 应是独立数据字段，不能只靠一句视频 caption 代替。

### 3.2 手物、相机与物体重建基础设施

| 工作 | 解决的问题 | 与本项目的关系 | URL |
| --- | --- | --- | --- |
| ARCTIC | 双手操纵铰接物体的数据集，提供手、物体和接触标注 | 可用于验证 retargeting 和仿真阶段，避免把所有误差都归给视频重建 | https://arctic.is.tue.mpg.de/ |
| HOT3D | 头戴多相机采集的第一视角手物 3D 跟踪数据 | 高质量多视角对照数据，适合校验普通手机视频的误差 | https://facebookresearch.github.io/hot3d/ |
| TACO | 工具、动作、物体组合的双手 HOI 数据 | 可测试工具使用和组合泛化 | https://taco2024.github.io/ |
| HOLD | 无预扫描模板的单目手物联合重建 | 可作为重建模块或对照，仍需处理尺度和遮挡 | https://github.com/zc-alexfan/hold |
| FoundationPose | 新物体 6D pose estimation/tracking | V2D 已集成；常见用法需要 RGB-D、mask 和 mesh | https://nvlabs.github.io/FoundationPose/ |
| BundleSDF | 未知刚体的 RGB-D 位姿跟踪与神经重建 | 适合有深度输入的物体数字孪生，不适合直接处理非刚体 | https://github.com/NVlabs/BundleSDF |
| SAM 2 | 提示驱动的视频分割 | 可生成跨帧物体 mask，但不提供深度、接触或动作 | https://github.com/facebookresearch/sam2 |

### 3.3 从人类运动到机器人和仿真

| 工作 | 发表状态 | 方法与创新 | 对本项目的价值 | URL |
| --- | --- | --- | --- | --- |
| Human2Sim2Robot | CoRL 2025 | 从一段 RGB-D 人类演示提取物体位姿轨迹和操作前手姿，在仿真中用 object-centric reward 训练灵巧手 | 最直接的“单视频 → 仿真 RL”对照，但需要 RGB-D 和任务数字孪生 | https://human2sim2robot.github.io/ 、https://github.com/tylerlum/human2sim2robot |
| DexTrack | ICLR 2025 | RL+IL 跟踪已重定向的人物交互参考 | 适合测试“已有参考轨迹怎样变成闭环策略”，不负责视频前端 | https://meowuu7.github.io/DexTrack/ 、https://github.com/Meowuu7/DexTrack |
| ManipTrans | CVPR 2025 | 通用轨迹模仿器加交互约束 residual，处理双手灵巧迁移 | 可作为物理 retargeting 基线，依赖 IsaacGym 和高质量上游轨迹 | https://arxiv.org/abs/2503.21860 、https://github.com/ManipTrans/ManipTrans |
| DexMachina | ICML 2026 | 以逐渐衰减的虚拟物体控制器构造课程，让策略接管物体状态跟踪 | 提供与 CHORD 不同的 functional retargeting 方案 | https://arxiv.org/abs/2505.24853 、https://project-dexmachina.github.io/ |
| MimicGen | CoRL 2023 | 把少量机器人 source demos 分解并重组为大量仿真示范 | 适合在轨迹已经通过仿真后做数据扩增，不负责视频转 action | https://mimicgen.github.io/ 、https://github.com/NVlabs/mimicgen |

### 3.4 不显式重建完整三维场景的策略路线

| 工作 | 发表状态 | 方法 | 需要注意的边界 | URL |
| --- | --- | --- | --- | --- |
| EgoMimic | ICRA 2025 | Project Aria 第一视角视频和 3D 手跟踪，与 teleop robot data 共同训练统一策略 | 使用专门眼镜、SLAM 和机器人数据，并非任意网页视频直接训练 | https://arxiv.org/abs/2410.24221 、https://egomimic.github.io/ 、https://github.com/SimarKareer/EgoMimic |
| ZeroMimic | ICRA 2025 | 从 EPIC-KITCHENS 蒸馏开、关、倒、抓放、切和搅拌等 goal-conditioned skills | 规模化能力强，但物理可审计性弱于显式重建路线 | https://arxiv.org/abs/2503.23877 、https://zeromimic.github.io/ |
| EgoDex | ICLR 2026 | 829 小时第一视角视频和录制时 3D 手指跟踪，提供手轨迹预测与 H-RDT 训练路径 | 数据比普通网页视频干净；许可为非商业且禁止演绎的条款需要单独审查 | https://github.com/apple/ml-egodex |
| Diffusion Policy | RSS 2023 | 条件动作扩散和 receding-horizon 行为克隆 | 可消费已经 grounding 的 observation/action，不解决 human-to-robot gap | https://arxiv.org/abs/2303.04137 、https://github.com/real-stanford/diffusion_policy |

## 4. 方法之间真正的差别

```text
低成本、弱物理约束                                      高成本、强物理约束

片段/语义 → 2D tracks → hand/contact → 3D HOI → robot IK → sim replay → RL
```

这不是简单的性能排行榜。越靠右，信息越接近机器人执行，但对视角、标定、mesh、算力和物理参数的要求也越高。

- 2D point tracks 适合大规模筛选和短时物体运动；
- contact interval 能提供夹爪开合和动作分段依据；
- 3D HOI 支持 retargeting，但容易受到深度、尺度和遮挡误差影响；
- object-centric reward 或 contact wrench 能跨 embodiment，却依赖仿真和接触建模；
- 直接策略路线能吸收大规模视频表征，但更难解释每条视频具体贡献了什么动作监督。

## 5. 本项目的第一版定位

第一版不承诺把任意互联网视频变成可部署策略。它回答一个更可测的问题：

> 对一个清晰、短时、单物体刚体操作视频，能否自动得到带置信度的物体轨迹、接触阶段和机器人参考，并在仿真中完成同一任务？

目标层级是 `sim_validated_episode`。首轮使用本地手机拍摄的“拿起饮料罐并放到纸盒上”，目标机器人为 Franka Panda，仿真器为 MuJoCo。实验会保留手工起终点基线，用来区分“仿真和机器人映射失败”与“视频感知失败”。

## 6. 建议的数据治理规则

每个视频和派生产物至少记录：

- 来源 URL、下载/拍摄时间、内容哈希和许可状态；
- 原始编码、时长、分辨率、帧率与转码参数；
- 模型名称、权重版本、配置、随机种子和运行环境；
- 相机内参的来源、尺度假设、坐标系定义和置信度；
- contact onset/hold/release、遮挡状态及证据；
- IK、碰撞、接触、物体漂移和任务成功结果；
- 人工修改过的字段及修改原因。

只有通过几何和仿真检查的轨迹才进入 action-bearing 数据集。未通过的片段仍可作为检索、视频表征或失败案例数据，不应被静默丢弃，也不应冒充机器人动作。

## 7. 当前判断

这个方向值得做，但“数量问题”不会因为抓取大量视频而自动解决。真正能积累规模的是一条会拒绝坏样本、保留不确定性并能逐层复跑的流水线。第一轮实验的价值不在于成功播放一个漂亮动画，而在于测出误差在哪一层开始失控，以及哪些视频条件能稳定地产生可执行伪数据。

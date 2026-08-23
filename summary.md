# Web Video to Embodied Data：文献、项目与实验判断

资料校验日期：2026-08-23。下文优先链接论文项目页和官方仓库，不用二手综述代替原始资料。

## 先把“data”说清楚

从视频到机器人训练数据，中间至少有四层：

| 层级 | 典型产物 | 是否已是 robot action data |
| --- | --- | --- |
| video evidence | clip、帧、来源和语义标签 | 否 |
| reconstructed motion | mask、point tracks、深度、手和物体位姿、接触候选 | 否 |
| robot reference | 末端和关节参考、夹爪状态、IK 结果 | 否，只是候选参考 |
| sim-validated episode | 通过动力学、碰撞、接触、任务和安全门禁的 observation/action | 才可能是，仍要标注 sim-to-real 风险 |

所以，点轨迹连续不等于语义跟踪正确，IK 可解不等于抓取可行，物体位姿强制回放也不是 action。这是本项目对所有方法的共同判定标准。

## 一、视频导入、切分与可检索数据库

### 1. [NVIDIA Video to Data](https://github.com/nvidia-isaac/video_to_data)

它要解决的是端到端工程问题：把长视频切为可检索的动作片段，再串起深度、mask、mesh、6-DoF pose、人体参数和机器人 grounding。其工程创新是容器化模块、强类型中间契约和文件化数据流；ingestion 还建立 scene graph 与 SigLIP-2 帧特征，允许按自然语言检索 clip。

截至 2026-08-23，旧版“robotic grounding 尚未发布”的描述已经失效。[2026-08-14 的 v0.2.0 release](https://github.com/nvidia-isaac/video_to_data/releases/tag/v0.2.0) 已发布 robotic grounding，[main 于 2026-08-21 的官方提交](https://github.com/nvidia-isaac/video_to_data/commit/87b40bd98bb54a0b783886d381259bb4ab48769a) 也包含该目录。但它还不是通用的“网页视频转 action”解决方案：whole-body retargeting 延后到 0.3，示例 HOI 序列仍在法务审查中，kinematic replay 是 teleport state 而非 physics-valid rollout，自动 URDF 的质量与惯量是占位值。ingestion 中途后端失败仍可能 exit 0；reconstruction 存在物体跟踪非确定性和手轨迹抖动或过度平滑的已知限制。[官方 reconstruction 说明](https://github.com/nvidia-isaac/video_to_data/blob/main/reconstruction/README.md) 还写明 BundleSDF wrapper 只覆盖 SDF 学习和纹理烘焙，不支持 BundleTrack pose tracking；HOI pipeline 的验证 GPU 为 A6000 和 L40S，当前 TensorRT 与 cuVSLAM 组合不支持 Blackwell。它适合做本项目的模块边界参考，不能取代我们的 fail-closed 验证。

### 2. [Ego4D](https://github.com/facebookresearch/Ego4d)

Ego4D 用数千小时第一视角视频覆盖日常活动，并把 episodic memory、forecasting、hands-and-objects 等任务放在统一 benchmark 下。它解决视频规模与语义多样性，但主体仍是感知数据，没有机器人本体的关节命令、接触力或动力学验证。我们可以借用它的 clip 索引和标注体系，不应把其 action label 当作 robot action。

### 3. [Ego-Exo4D](https://ego-exo4d-data.org/)

Ego-Exo4D 将 Aria 第一视角与多个第三视角同步，还提供标定和 3D 相关任务。多视角同步是它对单目尺度歧义的直接改进，对我们重录受控素材很有参考价值。它仍是人类活动数据，机器人重定向、接触参数和安全门禁需要另做。

### 4. [EPIC-KITCHENS-100](https://github.com/epic-kitchens/epic-kitchens-100-annotations)

该数据集为厨房第一视角视频提供动作、动词与物体等标注，主要解决长尾动作理解与跨环境泛化。“动词+名词”的标注对 ingestion 和 clip retrieval 实用，但没有物体公制位姿、力或机器人控制轨迹。它更适合作为候选视频来源和语义评测，不是 action 真值。

### 5. [Assembly101](https://assembly-101.github.io/)

Assembly101 包含拆装玩具的 12 路同步视角、细粒度动作段和大量 3D 手姿，还保留错误与纠正过程。它的价值是长程序、多视角和失败样本同时存在，很适合测 temporal segmentation 和 hand tracking。局限是对象域窄，手姿也不等于工具座标系下的机器人动作。

## 二、分割、稠密跟踪与运动表示

### 6. [SAM 2](https://github.com/facebookresearch/sam2)

SAM 2 把可提示分割扩展到视频，用 streaming memory 传播物体 mask，并允许交互式修正。它很适合给 point tracker 加一层“点必须仍在物体 mask 内”的语义门禁。mask 传播仍会在遮挡、相似外观和手物粘连时漂移；它不产生深度、接触力或 action。

### 7. [Cutie](https://github.com/hkchengrex/Cutie)

Cutie 是长时视频物体分割方法，通过 object-level memory 提高目标一致性，同时提供交互式 GUI。它可作为 SAM 2 之外的 mask propagation 对照，也便于人工修正检查点。它只保护像素语义归属，不解决尺度、物体位姿或机器人可执行性。

### 8. [CoTracker 3](https://github.com/facebookresearch/co-tracker)

CoTracker 联合跟踪大量点，让轨迹之间共享上下文；CoTracker 3 还用真实视频伪标签减少对大规模人工真值的依赖。对我们而言，它是 LK 的有力 A/B，特别适合遮挡与非刚体手部周边。它输出的是像素坐标与可见性，点仍可能从罐体转到手上，因此必须用 mask 和人工 checkpoint 评估语义漂移。

### 9. [TAPIR / BootsTAPIR](https://github.com/google-deepmind/tapnet)

TAPIR 先在全帧匹配查找候选点，再用局部相关性反复精化轨迹；BootsTAP 通过真实无标注视频的变换一致性继续训练。官方仓库同时提供 TAP-Vid、RoboTAP 和 TAPVid-3D 的评测资源，因此它适合做统一 tracker benchmark。限制与 CoTracker 相同：二维对应关系本身不含物体身份、公制尺度和接触状态。

## 三、三维重建、手物体姿态与接触

### 10. [FoundationPose](https://nvlabs.github.io/FoundationPose/)

FoundationPose 统一了新物体的 6D pose estimation 和 tracking，既支持给定 CAD，也支持用少量参考图像建立隐式物体表示。它可以替换我们目前过于简化的二维到位姿映射，但前提是有可用的物体几何或参考视图和合适的 RGB-D 输入。它不会自动给出材料、摩擦、接触力或可抓取区域。

### 11. [BundleSDF](https://bundlesdf.github.io/)

BundleSDF 从单路 RGB-D 视频同时跟踪未知刚体的 6-DoF pose 并重建几何，核心是在线 Neural Object Field、pose graph 和动态 memory frame pool。它能处理遮挡、弱纹理和反光，很接近“视频到可使用物体资产”这一层。局限是需要 RGB-D 且假设首帧有物体分割；重建 mesh 仍需要质量、惯量、碰撞简化和接触验证才能进仿真。

### 12. [DUSt3R](https://github.com/naver/dust3r)

DUSt3R 把图像对直接映射为 pointmaps，降低了传统 SfM 中特征匹配、内参和 pose 估计的工程门槛。对普通 RGB 网页视频，它适合生成场景结构候选或相对深度。它并不自动消除单目公制尺度歧义，也不保证重建能直接用于碰撞或接触动力学。

### 13. [HaMeR](https://github.com/geopavlakos/hamer)

HaMeR 用 transformer 从单帧 RGB 恢复 3D 手部 mesh，解决了手姿在遮挡、多样视角下的泛化问题。它可以为手部轨迹和 grasp phase 提供比稀疏 LK 点更强的先验。输出是 MANO 类手模型，不包含真实接触力；单帧误差、尺度和时序抖动也需要另外处理。

### 14. [ContactPose](https://contactpose.cc.gatech.edu/)

ContactPose 将手部 pose、物体 pose、RGB-D 和手物体接触配对，用 25 个日常物体的 2306 个抓持补上了接触标注缺口。它适合用来学习接触区域先验，或检查网络是否把“手靠近物体”误当成“稳定抓持”。它是有限物体上的准静态抓持数据，不提供机器人夹爪控制、动态力或任务成功真值。

## 四、从人类视频到机器人 grounding 与模仿学习

### 15. [Track2Act](https://github.com/homangab/Track-2-Act)

Track2Act 从互联网视频学习 point-track prediction，将初始图像与目标图像之间的运动表达为可转移的像素轨迹，再用于机器人操作。创新点是弱化了人手与机器人外观差异，使运动成为中介。它仍需要机器人端的控制学习与环境反馈；point track 不直接给出 3D 接触、关节力矩或碰撞安全性。

### 16. [RoboTAP](https://robotap.github.io/)

RoboTAP 用 TAPIR 找到示范中与任务相关的点，再用低层控制器让这些点追踪目标轨迹，以少量示范完成插入、堆叠和路径跟随。它证明 point track 可以是高效的任务表示，并提供真实机器人点标注 benchmark。这条路线需要现场示范和闭环视觉控制，不是从任意网页 clip 直接导出离线 action。

### 17. [VideoDex](https://video-dex.github.io/)

VideoDex 从互联网人手视频中提取视觉、动作和物理先验，把 MANO 手重定向到 Allegro Hand，并用移动相机估计手腕运动。它没有把 embodiment gap 假装成已经消失：方法还采集少量 teleoperation demonstration 来连接人类先验与机器人。对本项目的启示是把 web video 定位为 prior，而非未验证的 action 真值。

### 18. [Vid2Robot](https://vid2robot.github.io/)

Vid2Robot 用 cross-attention transformer 让机器人策略以人类或机器人 prompt video 为条件，从视频中理解任务并在新场景执行。它绕过了“先导出明确关节标签”这个中间目标，直接学习 video-conditioned policy。但训练仍依赖配对的机器人数据，prompt video 是任务条件而非可审计的 action label。

### 19. [RT-Trajectory](https://rt-trajectory.github.io/)

RT-Trajectory 把粗糙 trajectory sketch 作为策略条件，用几何路径补足语言指令对新任务结构表达不足的问题。轨迹草图是一个与本项目 point-track/reference 层很相近的中介，也能由不同方法生成。它不负责从原始 web video 恢复可靠的草图，训练策略和安全验证仍需要机器人数据。

## 五、数据扩增、统一格式与 benchmark

### 20. [MimicGen](https://github.com/NVlabs/mimicgen)

MimicGen 从少量人类示范中切出 object-centric 子任务，改变对象初始状态并重组轨迹，自动生成大量仿真示范。它直接解决通过物理仿真放大已有示范的数量问题。前提是种子示范、子任务切分和仿真场景已经可用；它不能修复错误的物体几何或本来就不合法的抓取。

### 21. [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment)

Open X-Embodiment 把多机构、多机器人的数据统一到 RLDS episode 格式，并提供 RT-1-X 等跨 embodiment 训练资源。它解决了数据 schema 破碎和数据集混合的工程问题，值得我们参考 episode/provenance 层设计。统一格式不会消除 action space、标定、频率和数据质量的差异；它收录的是机器人数据，不是普通视频自动转换器。

### 22. [DROID](https://github.com/droid-dataset/droid)

DROID 通过可复制的硬件与遥操平台，在多地真实环境中采集 76k 条操作示范，解决真机数据场景单一的问题。它把硬件规范、标定流程和 policy-learning code 一起开放，数据质量边界比网页视频清楚得多。代价也正是我们要降低的部分：需要真实硬件、人工遥操、标定和质控。它适合用作少量高质量 anchor data，而不是被 web video 完全替代。

### 23. [RoboCasa365](https://github.com/robocasa/robocasa)

RoboCasa365 在厨房仿真中扩展到 365 个任务、2,500 多个场景和大量人类与机器人示范，并配套策略 benchmark 和 leaderboard。它给我们的直接价值是：当一条视频参考已经通过物理门禁，可以在更大的场景和物体分布上测试泛化。仿真资产、接触模型和自动轨迹的偏差不会自动消失，评测结果也不能直接等同于真机成功率。

### 24. [robomimic](https://github.com/ARISE-Initiative/robomimic)

robomimic 把机器人模仿学习数据、视觉与低维 observation encoder、BC 与 offline RL 等方法放到可复现框架中。它适合用来测试“加入经过门禁的 video-derived episodes 后，策略是否真的受益”。它不负责从视频恢复 action，也不会因为训练 loss 下降就证明新 episode 的接触与动力学正确。

## 对本项目的判断

现有工作已经把局部问题做得很强：切 clip、建索引、传播 mask、跟踪点、恢复手和物体 pose、用轨迹条件控制策略，或在仿真里放大示范。真正没有被一个通用模型解决的，是这些层之间的可审计转换，尤其是公制几何、接触语义、embodiment gap 和物理可行性。

这也解释了 Phase 2A 的结果为什么有用，即使它是失败结果。fresh verified EXP-001 suite `20260822T091027937019Z-93982cd7-b106` 记录了：

- B0 fixed seeds 19–48 为 `0/30`；
- 共 `18,660` 个 forbidden-contact frames，最大 penetration `0.01252896243320321 m`；
- B0 是 `rejected / physics_validation_failed`；
- B1 是 `rejected / kinematic_replay_not_action`，且人工 QA 发现后段 hand takeover drift；
- B2–B4 在这次 narrower suite 中未请求，不存在 metric-depth 结论；
- `actions_exported=0`，没有 `actions.npz`。

simulation-only GIF 是失败诊断，不是“已经把视频转成可训练 action”的演示。这条边界应继续保留。

## EXP-002 换源复现

EXP-002 使用另一段私有录制素材，在 C1 `bf6615c1bf15f476ca6c51e9f20bbda69d6549a7` 上重新执行 B0+B1。fresh verified suite `20260823T083932722546Z-aac0791e-aa8b` 为 `recorded`，结果如下：

- B0 fixed benchmark 为 `0/30`，child 是 `rejected / physics_validation_failed`；
- B1 是 `rejected / kinematic_replay_not_action`；
- B1 的 LK availability 为 `1.0`，只表示 point availability，semantic accuracy 没有测量；
- release 与 settle phase confidence 为 0；
- action 数为 0，没有 `actions.npz`。

这是 different-source replication，不是 EXP-001 同源重跑。输入变了，runtime、reachability、误差等数值的变化不能归因于算法；本次 generator 也没有算法改动。

EXP-002 从 C1 重生成的两张公共 GIF 与现有 `docs/media/exp001-*` 文件逐字节一致。它们只含仿真画面，`contains_private_source_frames=false`。仓库复用现有媒体链接，没有再复制约 22 MB 文件。机器可读记录见 [`experiments/EXP-002-canonical-can-replication/metrics.json`](experiments/EXP-002-canonical-can-replication/metrics.json)。

## 建议的 Phase 2B

1. 用现有素材制作罐体 mask 和固定 checkpoint annotation，在同一帧集上比较 LK、SAM 2/Cutie mask-gated CoTracker 与 TAPIR。主指标改为 semantic checkpoint error、occlusion recovery 和 hand takeover rate，不再只看 point availability。
2. 在 pinned Panda 场景里单独修复 B0：先做 gripper-object 几何可行性检查，再加 contact-aware close/lift control。seeds 19–48、扰动范围和所有碰撞门禁保持不变。
3. 只有 B0 通过后，才比较视频驱动分支带来的额外失败。这能把“机器人本来就抓不住”和“视频感知传错了”分开。
4. 如果要报告 metric 3D error，再重录带内外参、实物尺寸和 AprilTag/棋盘格的 RGB-D 或双目视频。现有普通单目视频足够做 2D 语义跟踪 A/B，但不足以验证公制几何和接触。

## 数据治理底线

每个可发布 run 要绑定 config/source/model 的逻辑身份和内部哈希，同时记录环境、seed、schema、terminal reason、media role、private-frame flag 和 action absence/presence。真实私有文件名、本地路径、内容哈希和源帧不进公开仓库，公开配置只保留明确的可替换占位值。

失败样本可以用于检索、表征学习或失败分类，但不能因为数量多就进入 action-bearing dataset。进入前仍要通过完整的物理与安全门禁。

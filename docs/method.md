# EXP-001 方法说明：手机视频到 MuJoCo 诊断回放

## 研究问题

一段 7 秒的单目手机视频能否提供足够的物体运动和接触信息，供 primitive 7-DoF Panda-like model 在 MuJoCo 中诊断“拿起罐体并放到盒子上”的参考？该 primitive model 不是官方 Franka Panda。

我们不比较视频生成质量，也不训练通用策略。实验只测一条伪数据链路是否成立，以及误差从哪一步进入。

## 原设计假设

以下是实验提出的假设，不是 EXP-001 已经验证的结论：

- H1：物体 mask 和 point tracks 足以恢复稳定的二维运动与阶段时序；
- H2：加入单目深度、相机补偿和已知物体尺寸后，轨迹可以落到可用的公制场景；
- H3：物体中心轨迹加 contact phases，比直接复制人手腕轨迹更适合二指夹爪；
- H4：桌面、速度、预抓取距离等简单约束能显著提高仿真成功率；
- H5：一部分失败无法由平滑修复，应由 readiness/rejection 机制拦截。

## 样本

| ID | 文件 | 用途 |
| --- | --- | --- |
| V0 | `video/手机录制.mp4` | 主实验：短时第一视角抓放 |
| V1 | `video/拋瓶甩瓶遊戲…成功.webm` | 负对照：快速旋转、抛掷、液体 |
| V2 | `video/周興哲 - 如果雨之後 (acoustic guitar solo).webm` | 负对照：双手、长时、密集接触 |
| V3 | `video/guitar.webm` | 长视频切片和低质量阶段测试 |

V1 至 V3 首轮只做 readiness 评估，不要求生成机器人 action。

## 人工标注与几何：本轮未执行

EXP-001 没有两名标注者、像素级罐体框/中心标注、contact ground truth，也没有实测罐体或纸盒尺寸。版本化的 [`semantic-assessment.json`](../experiments/EXP-001-phone-can-mujoco/semantic-assessment.json) 只是人工视觉 QA：观察到放置后 LK 点从罐体漂到手上；它不是像素级 annotation，罐体中心 checkpoint error 为 `not_measured`。

B0 实际使用配置中的固定 canonical start/goal 与程序合成的 reference phases，不由人工 contact annotation 构造。B1 使用自动 LK/phase 输出，但其 endpoint/path 因上述语义漂移而不可靠。

## 感知流程

### 视频与帧

以原视频时间戳为准，实验副本固定为 30 FPS。每一帧保存 `frame_index`、`timestamp_s` 和原始 PTS 对应关系。

### 物体 mask（原设计，未执行）

在首个完整可见帧给出饮料罐提示框。分割模型向前传播 mask。逐帧计算面积、质心、外接框和 mask 连通性。突然超过 30% 的面积跳变或置信度下降被标记为异常，不直接删除。

### point tracks（实际为 LK ROI tracking）

实际 runner 在锁定 ROI `[374,423,104,155]` 内使用 Lucas–Kanade 与 forward-backward check。`lk_point_availability_ratio` 只衡量非零 point confidence/FB availability，不衡量点是否仍在罐体上；本次值为 1.0，但人工视觉 QA 已确认后段漂到手上。mask、RANSAC 语义剔除和 can-center error evaluation 均未执行。

### 手部证据（原设计，未执行）

轻量基线输出手部关键点与 palm center。手被物体遮挡时保留 `visibility=0`，不插入假关键点。后续如使用 HaMeR/HaWoR，应单独记录模型与权重。

## contact phases

估计器使用三条主证据：手物接近、物体由静止转为随手移动、手物速度耦合。释放由距离增加、共同运动结束和物体重新稳定共同判断。

状态机：

```text
free -> approaching -> contact_candidate -> holding -> releasing -> free
```

`contact_candidate` 至少持续 3 帧才进入 `holding`。短时缺失最多桥接 5 帧，超过后标记为遮挡区间。输出 onset/release 的帧和秒级时间，以及每条证据的分数。

## 三维恢复（原设计，本轮未执行）

### 深度与内参

从模型获得逐帧相对深度和内参估计。使用 mask 内深度中位数代表罐体中心深度。桌面区域用于检查时序漂移。

### 尺度

优先级：

1. 实测罐体尺寸；
2. 清晰可见的纸盒尺寸；
3. 类别先验。

类别先验只用于探索性 run，不能作为最终公制真值。尺度来源写进 `provenance.json`。

### 相机补偿

在手和目标物体以外的背景区域跟踪特征，估计相机运动。若背景内点不足或镜头发生切换，则三维结果降级或拒绝。当前 V0 允许用静态相机近似作为对照，但 B2/B3 仍应报告相机运动估计。

### 平滑

对位置使用置信度加权样条或 Savitzky-Golay 滤波，窗口不跨越 contact onset/release。约束版本 B4 再加入桌面平面、最大速度、最小抬升高度和目标表面约束。

## 机器人参考

罐体、桌面、纸盒和机器人几何均硬编码在 primitive XML 中，没有从视频或实物测量。primitive 7-DoF Panda-like diagnostic model 使用二指侧抓参考；这不是官方 Franka Panda 几何，也不构成经过验证的 action。

人手动作中的犹豫、指尖调整和不必要摆动不直接复制。运输段保留物体运动的主要路径形状，抓取前后使用机器人自己的 approach/retreat 段。

IK 逐帧求解，并用上一帧关节作为初值。B0 实测 reachability 仅 0.0235656。除此之外，当前 collision validation 为 `not_implemented`；runner 对所有变体硬设 `action_export_eligible=false`，即使未来其他 physics 指标通过也不能导出 action。

## MuJoCo 回放

### 场景

- primitive 7-DoF Panda-like model 固定基座；
- 桌面高度和机器人基座变换写入配置；
- 罐体使用圆柱碰撞体；
- 纸盒使用长方体碰撞体，顶面是放置目标；
- 初始物体位置依据视频场景规范化到机器人可达区域。

这些形状和放置是硬编码诊断几何，不是测量所得。arm/hand collision geometry 未完整启用，也没有实现 self/table/box/penetration validation。

视频并不提供机器人基座在真实世界的位置，因此 scene-to-robot 放置是一项显式设计选择。我们保持相对物体运动和任务拓扑，不声称恢复了原场景的绝对机器人坐标。

### 控制

B0 使用固定 canonical start `[0.12,0.45,0.04]` m、goal `[-0.05,0.55,0.13]` m 和合成 reference phases，再用逆运动学生成诊断 reference。仿真按固定控制频率跟踪。本轮结果是 physics rejection，不是控制器验证成功。

### 扰动（原设计，未执行）

原设计提出 20 次评测对以下量做小范围随机化；EXP-001 没有执行：

- 罐体初始 x/y 位置；
- 摩擦系数；
- 罐体质量；
- 轨迹时间缩放。

随机范围在第一次正式评测前锁定。训练和评测不共享为了结果调参的随机种子。

## 指标和分析

下表是拟议指标集合；EXP-001 只报告实际计算的子集，未测字段不能用代理数值替代。

| 层 | 主要指标 |
| --- | --- |
| 视频 | 解码完整率、时间戳一致性 |
| mask/track | mask 连续性、语义 track accuracy、重投影误差（未测）；LK point availability（已测但非语义准确率） |
| contact | onset/release 时间误差、区间 IoU（无 ground truth，未测） |
| 3D | 尺度一致性、平滑度、背景补偿残差 |
| retarget | IK 可解率、最大关节速度；碰撞次数（未实现） |
| sim | grasp/place 成功率、最终位置/姿态误差、掉落/穿透 |

原设计的主要比较如下；本轮只有被拒绝的 B0/B1 单次诊断，B2–B4 因 `metric_depth_not_available` 未运行，不能进行这些消融比较：

- B1 对 B0：仅二维信息损失多少；
- B2 对 B1：单目三维是否真的有帮助；
- B3 对 B2：自动 contact 对结果的影响；
- B4 对 B3：物理和几何约束能修复多少失败。

## 何时停止或拒绝

以下任一条件满足时不生成 action-bearing episode：

- 主物体未被稳定分割；
- 有效 tracks 少于 8 条并持续超过 0.5 秒；
- 缺少完整的 onset、hold、release；
- 公制尺度来源未知；
- IK 可解率低于 95%；
- 仿真中存在严重穿透或控制发散；
- provenance 或模型版本不完整。
- collision validation 未实现。

拒绝结果仍保存已有 artifacts 和原因，供后续 readiness 模型使用。

## Seedance 对照

只有当 V0 无法满足感知条件，或需要受控改变视角、遮挡和运动速度时才调用 Seedance。生成提示应要求固定相机、单手、单圆柱物体、完整抓放、无遮挡和纯色背景。生成结果标记为 synthetic，不作为真实物理真值。

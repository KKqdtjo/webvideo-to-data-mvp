# WebVideo to Data MVP 技术设计

日期：2026-08-19
状态：已审阅的原始设计；下列“应满足”是拟议验收目标，不是实际完成声明。
首轮任务（原设计）：从手机拍摄的饮料罐抓放视频生成 Panda/MuJoCo 仿真伪数据。

实现注记（2026-08-19）：EXP-001 实际使用硬编码几何的 primitive 7-DoF Panda-like diagnostic model，不是官方 Franka Panda。没有执行像素级/双标注者 annotation、实物几何测量、metric-depth B2–B4 或 20 次扰动；arm/hand collision geometry 与 self/table/box/penetration validation 也未完整实现。因此所有 action export 均被 `collision_validation_not_implemented` 硬性禁用。实际结果以 `experiments/EXP-001-phone-can-mujoco/` 为准。

## 1. 目标与验收范围

MVP 验证一条完整但窄的链路：

```text
单目短视频
  -> 可审计的二维/三维运动证据
  -> 机器人参考轨迹
  -> MuJoCo 回放
  -> 通过验证的仿真 episode 或明确的拒绝原因
```

输入是一段本地私有手机视频，公开仓库不记录它的精确文件名。视频中一只手拿起圆柱形饮料罐，将其放到纸盒顶面。原设计目标 embodiment 是 Franka Panda 二指夹爪；本轮实现仅为 primitive 7-DoF Panda-like diagnostic model。

MVP 完成时应满足：

- 原始视频保持不变，所有派生产物可重新生成；
- 自动生成物体轨迹和 contact phases，并保存置信度；
- 机器人参考通过 IK、关节限制和碰撞检查；
- 手工基线 B0 能稳定完成抓放；
- 自动路线 B3 或 B4 在 20 次轻量扰动中达到至少 70% 放置成功率；
- 任一失败能归到感知、尺度、映射、控制或物理参数中的一类；
- 不把未通过仿真的轨迹导出为正式 action-bearing episode。

## 2. 不做的事情

首轮不处理：

- 灵巧手、双手、工具使用、非刚体和液体；
- 快速抛掷、瓶子翻转和接触力精确恢复；
- 从网络视频直接训练通用闭环策略；
- 完整 NVIDIA V2D、CHORD 或 Isaac Lab 复现；
- 自动生成物体高精度 mesh；
- 把 Seedance 视频当作真实物理数据；
- LeRobot/RLDS 全量导出。只有 B3/B4 通过后才评估 action-bearing 格式。

这部分范围是硬约束。后续若扩展到吉他或瓶子翻转，应另写实验设计，因为它们引入双手、多接触、快速动力学或液体等新问题。

## 3. 设计原则

### 3.1 工件优先

每个阶段读写明确的文件，不通过隐藏的进程内状态连接。任何模型都可以被替换，下游只依赖 schema。

### 3.2 观测、推断和修正分开

原始帧与 mask 属于观测；深度、三维位姿和 contact 属于推断；平滑、尺度校准和物理修正属于后处理。三类字段分别记录，不能覆盖原值。

### 3.3 允许拒绝

视频不满足条件时输出 `rejected` 和原因。拒绝是正常结果，不以零轨迹、默认深度或伪高置信度继续运行。

### 3.4 分层验证

先验证仿真场景和控制，再替换视频感知结果。B0、B1、B2、B3、B4 的差分用来定位误差来源。

## 4. 系统组成

### 4.1 `media`：视频探测与规范化

职责：

- 用 ffprobe 读取编码、时长、分辨率、帧率和音视频流；
- 计算 SHA-256；
- 生成固定 30 FPS、可稳定解码的实验副本；
- 保存帧序号与原视频时间戳的映射。

接口：

```python
probe_video(path: Path) -> VideoMetadata
normalize_video(source: Path, output: Path, fps: int = 30) -> NormalizationRecord
```

如果 OpenCV 能稳定读取 HEVC，则规范化副本只做无损抽帧；否则生成 H.264 CRF 18 副本。任何情况下都保留原文件哈希。

### 4.2 `annotations`：人工真值与阶段定义

首轮需要一份轻量人工标注：

- 罐体初始区域和目标纸盒顶面；
- contact onset、稳定抓持、release 和 settle 时间；
- 5 至 10 个检查帧上的罐体中心或框；
- 已知或测量的罐体尺寸、纸盒尺寸。

它用于评测，不作为 B3 的隐藏输入。B0 可以直接使用人工起终点和阶段。

### 4.3 `perception`：mask、点轨迹与手部证据

物体分割以首帧框或点提示初始化，输出逐帧二值 mask。点跟踪器在 mask 内均匀采样点，并保留可见性分数。手部模块输出关键点或 hand mask，作为 contact 判断的辅助证据。

候选实现：

- SAM 2：物体 mask；
- CoTracker 或 TAPIR：point tracks；
- MediaPipe Hands：低成本手部基线；
- MoGe 系列：深度和相机内参估计。

实现阶段先做模型探针，选择在实际硬件上能运行且许可证可接受的组合。若重模型不可用，B1 允许使用 OpenCV 特征点跟踪和人工首帧 mask，但结果必须标成 baseline。

### 4.4 `contact`：接触阶段估计

接触不是由单一距离阈值决定。估计器组合以下证据：

- 手与物体 mask/关键点距离；
- 手部闭合或抓持姿态；
- 手和物体运动速度的相关性；
- 物体是否从静止转为随手移动；
- 释放后手物距离增加，物体进入稳定状态；
- 遮挡和可见性。

输出为区间和证据列表：

```json
{
  "phase": "hold",
  "start_frame": 54,
  "end_frame": 132,
  "confidence": 0.83,
  "evidence": ["hand_object_proximity", "motion_coupling"],
  "occluded": false
}
```

若接触阶段不完整或平均置信度低于 0.5，B3 停止并输出 `contact_unreliable`。阈值由 EXP-001 固定，不在看完结果后修改。

### 4.5 `reconstruction`：场景坐标与物体轨迹

三维估计分四步：

1. 由内参或模型估计把像素反投影到相机坐标；
2. 用已知罐体高度/直径确定公制尺度；
3. 从背景特征估计相机运动，并从表观运动中扣除；
4. 对罐体位置和姿态做置信度加权平滑。

输出保留三套轨迹：

- `raw_observation`：来自 mask、tracks 和 depth 的逐帧值；
- `estimated_state`：完成相机补偿和尺度恢复后的三维状态；
- `corrected_state`：经过平滑、桌面约束和物理可行性修正的状态。

罐体采用轴对称几何。绕自身竖直轴的角度在视觉上不可观时，不强行给出精确 yaw；字段设为低置信度，机器人抓取使用预定义侧抓姿态。

### 4.6 `retargeting`：物体轨迹到 Panda 参考

映射以任务结果为中心，不逐帧复制人手。根据 contact phases 构造八段参考：

```text
pre-grasp -> approach -> close -> lift -> transport -> lower -> open -> retreat
```

抓取姿态由罐体轴、夹爪开口和桌面法向确定。运输阶段跟踪物体平移，姿态使用最小变化插值。轨迹生成后依次检查：

- Panda 工作空间；
- IK 可解性；
- 关节位置/速度/加速度；
- 自碰撞、桌面、纸盒和物体碰撞；
- 夹爪宽度与罐体直径。

若 IK 可解帧比例低于 95%，结果标记为 `retargeting_failed`，不进入仿真 episode 导出。

### 4.7 `simulation`：MuJoCo 场景、控制与任务判定

场景包含 Panda、平面桌、圆柱罐和纸盒。几何尺寸来自人工测量或配置；质量、摩擦和阻尼有明确默认值，并在 B3/B4 做轻量随机化。

B0 先验证控制器和抓取参数。自动路线沿同一控制接口发送末端目标与夹爪命令。任务成功需要同时满足：

- 罐体在运输阶段离开桌面；
- 最终罐体中心投影落在纸盒顶面安全区域；
- 罐体与纸盒保持接触并稳定 1 秒；
- 机器人松爪后罐体未掉落；
- 场景无严重穿透或非法关节状态。

每次回放保存结果 JSON、状态轨迹、控制输入和渲染视频。无显示环境使用 MuJoCo headless EGL 或 OSMesa。

### 4.8 `export`：数据包与拒绝记录

通过验证的数据包包含：

```text
episode.json
observations.npz
actions.npz
object_states.npz
contacts.json
provenance.json
metrics.json
replay.mp4
```

未通过的数据包至少包含 `provenance.json`、已有中间产物和 `rejection.json`。拒绝样本不写 `actions.npz`，避免下游误用。

## 5. 核心 schema

### 5.1 `provenance.json`

```json
{
  "source": {
    "path": "source-placeholder.mp4",
    "sha256": "<computed-at-runtime>",
    "origin": "team_phone_capture",
    "license": "internal_research"
  },
  "media": {
    "duration_s": 7.019083,
    "width": 540,
    "height": 960,
    "fps": 30.0,
    "codec": "hevc"
  },
  "pipeline": {
    "config": "configs/exp001.yaml",
    "git_commit": "<computed-at-runtime>",
    "models": []
  }
}
```

运行时字段由程序计算，不手填假值。如果工作区没有 Git 仓库，`git_commit` 为 `null` 并记录 `workspace_not_versioned`。

### 5.2 `trajectory.npz`

数组统一使用秒、米、弧度和右手坐标系：

| 字段 | shape | 含义 |
| --- | --- | --- |
| `timestamp_s` | `[T]` | 与规范化视频对齐的时间 |
| `position_m` | `[T, 3]` | 罐体在 scene frame 中的位置 |
| `quaternion_wxyz` | `[T, 4]` | 罐体姿态 |
| `visibility` | `[T]` | 0 到 1 |
| `confidence` | `[T]` | 综合置信度 |
| `covariance` | `[T, 6, 6]` | 位姿不确定性；无法完整估计时使用对角近似并标注方法 |

### 5.3 坐标系

- `image_px`：左上角为原点，x 向右，y 向下；
- `camera`：x 向右，y 向下，z 向前；
- `scene`：x 沿桌面向右，y 沿桌面向前，z 向上；
- `robot_base`：由 MuJoCo 模型定义，通过固定 `T_robot_scene` 与 scene frame 相连。

所有变换保存为 4×4 齐次矩阵，命名采用 `T_target_source`，表示把 source 中的点变换到 target。

## 6. 对照实验

### B0：手工场景和轨迹

人工给定物体起点、终点、抓取姿态和阶段。若 B0 失败，说明问题在仿真、控制或物理参数，与视频感知无关。

### B1：二维运动

使用 point tracks，深度固定。它测试二维方向、相对位移和时序是否足以生成可用参考。

### B2：三维恢复

加入单目深度、相机补偿和尺度校准。与 B1 的差异反映三维估计的收益或损害。

### B3：自动接触阶段

用自动 contact onset/hold/release 替代人工阶段，形成完整视频转换链。

### B4：物理修正

在 B3 上加入平滑、桌面平面、预抓取安全距离、最大速度和目标放置约束。它衡量可解释约束是否比直接回放更可靠。

## 7. 误差处理

当前 runner 的 terminal status 采用严格、互斥的词汇：

```text
completed | not_run | rejected | failed
```

- `completed`：该 run 完成，且仅在全部 action gate 通过时才允许 action；
- `not_run`：前置能力缺失而未执行，必须给出原因；
- `rejected`：输入不满足视频或几何条件，属于预期数据结果；
- `failed`：代码、依赖或运行环境错误。

感知质量下降使用独立的 `perception_status=degraded`，不作为 terminal status。所有非 `completed` 状态不得包含 `actions.npz`；当前 collision validation 未实现，所以实际 EXP-001 没有任何 eligible action。

错误记录包含阶段、稳定错误码、人类可读说明、输入 artifact、配置和异常摘要。日志不包含 API key、Authorization header、私有下载 URL 或原视频像素内容的无控制转储。

## 8. 测试策略

### 单元测试

- 时间戳和帧映射；
- 坐标变换方向；
- 四元数归一化与插值；
- contact phase 状态机；
- 轨迹平滑和速度限制；
- schema 序列化与拒绝规则。

### 集成测试

- 3 至 5 秒合成点轨迹经过重建和 retargeting；
- Panda IK 对已知可达位姿返回合法关节；
- B0 完成单次 headless 抓放；
- 同一配置重复运行产生一致的指标和 artifact 清单。

### 数据测试

- 原视频哈希未改变；
- mask、track、depth 和轨迹帧数一致；
- 时间戳单调；
- 数值无 NaN/Inf；
- action-bearing 数据包必须存在通过的 sim metrics。

## 9. 实验记录

实验 ID 使用 `EXP-NNN-短名称`。每次运行保存：

- `run.yaml`：解析后的完整配置；
- `environment.json`：系统、Python、依赖和设备；
- `stdout.log` / `stderr.log`：脱敏日志；
- `metrics.json`：机器可读指标；
- `report.md`：结论、失败分析和下一步；
- `artifacts/`：中间数据和视频。

任何阈值修改都新建 run，不覆盖旧结果。

## 10. 安全、许可与数据治理

- `video-generation/seedance_api.txt` 视为敏感凭据，不读取到日志，不复制到远端，不纳入版本控制；
- 若凭据曾被提交或分享，应在调用 API 前轮换；
- YouTube 素材保留来源 URL、下载时间、平台条款和研究使用状态；
- 人脸、声音和可识别环境在发布数据前单独审核；
- MANO、ARCTIC、HOT3D、EgoDex 等数据和模型按各自许可使用；
- 生成视频只能作为合成对照，必须标记模型、提示词和生成时间。

## 11. 环境策略

当前远端服务器没有标准 NVIDIA GPU 运行时、Docker 或 Isaac Lab，PyTorch 还缺厂商共享库。MVP 不以修复该环境为前置条件。

执行顺序：

1. 检查本机 Python、GPU 和可安装依赖；
2. 若本机可运行视觉模型，在本机生成感知 artifact；
3. MuJoCo 优先在本机或远端 CPU headless 运行；
4. 只有在获得标准 NVIDIA 环境后，才建立 V2D/Isaac Lab 对照；
5. 不在未知 PPU 环境中强装 CUDA/NVIDIA 容器栈。

## 12. 文件边界

计划创建：

```text
configs/exp001.yaml
src/webvideo_to_data/media.py
src/webvideo_to_data/schema.py
src/webvideo_to_data/tracking.py
src/webvideo_to_data/contact.py
src/webvideo_to_data/reconstruction.py
src/webvideo_to_data/retargeting.py
src/webvideo_to_data/simulation.py
src/webvideo_to_data/export.py
scripts/run_exp001.py
tests/
```

每个模块只承担一种职责。模型适配器与核心几何逻辑分开，使没有重型权重的测试环境也能运行单元测试。

## 13. 决策记录

- 选择 MuJoCo 而非 Isaac Lab：当前机器无法稳定运行 NVIDIA 容器栈，首轮只需要刚体抓放和可重复接触；
- 选择 Franka Panda：公开模型成熟，二指夹爪与首轮任务匹配；
- 选择手机罐体视频：接触清楚、任务短、物体近似刚体，能隔离最基本链路；
- 暂不调用 Seedance：现有素材已经满足实验条件，生成视频不会提供真实接触物理真值；
- 保留手工 B0：没有它就无法判断失败来自感知还是仿真。

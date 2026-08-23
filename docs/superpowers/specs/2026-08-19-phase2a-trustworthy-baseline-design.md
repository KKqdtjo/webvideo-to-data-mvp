# Phase 2A：可信物理基线与实验入口设计

## 1. 背景

EXP-001 已经跑通视频读取、LK 跟踪、阶段推断、机器人参考、MuJoCo 回放、结果发布和拒绝门禁。它没有产生可用 action。现有证据只能说明系统能够发现并记录失败：

- B0 绕过视频感知后仍无法抓取，IK reachability 为 0.0235656，双指接触为 0，lift 为 0；
- B1 是对象位姿覆盖的运动学回放，且 LK 点在尾段从罐体漂到手上；
- B2 至 B4 因缺少公制深度而没有运行；
- primitive Panda-like 模型没有完整碰撞几何，action export 被硬性关闭。

在这个状态下直接接入 SAM2、CoTracker 或 CHORD，会把物理后端的错误和视觉误差混在一起。Phase 2A 先建立一个能独立通过的手工物理基线，再补齐实验入口、结果契约和展示层。它不实现视频公制三维重建，也不输出视频生成的机器人 action。

## 2. 目标

Phase 2A 交付六项结果：

1. 使用 MuJoCo Menagerie 的 Franka Emika Panda 模型，不再把自制 primitive 模型称为目标机器人；
2. 用收敛 IK 和分阶段控制完成手工 canonical pick-and-place；
3. 对碰撞、穿透、关节限制、抓取、抬升、放置和稳定状态进行机器验证；
4. 用 30 个固定随机种子运行轻量扰动，形成可重复的 B0 基线；
5. 增加严格配置、追加式 run、版本化 artifact、preflight、批量 CLI 和 CI；
6. 生成不会把 rejected/kinematic 结果误写成成功的静态 dashboard 和媒体。

完成 Phase 2A 后，才能进入 Phase 2B 的标注、mask-gated tracking 和多 tracker 对比。Phase 2C 的 metric depth、相机补偿与 6D pose 仍是独立子项目。

## 3. 不在本阶段处理的内容

- 不安装或运行 NVIDIA V2D、Isaac Lab、CHORD；
- 不把 Depth Pro、Metric3D、SAM2、CoTracker 或 FoundationPose 加入主依赖；
- 不训练策略，不做真机部署；
- 不将 B0 的手工控制轨迹命名为视频 action；
- 不把现有手机视频上传到仓库或 Release；
- 不以 B1 的对象位姿覆盖接触数作为物理抓取证据。

## 4. 总体结构

```text
config YAML
  -> strict config loader
  -> preflight
  -> append-only suite runner
       -> B0 manual physics baseline
            -> official Panda model
            -> converged IK + phase controller
            -> collision/contact validator
            -> 30-seed evaluation
       -> legacy B1-B4 diagnostics
  -> versioned artifacts + manifest
  -> sanitized experiment summary
  -> static dashboard + diagnostic media
```

现有 `run_experiment(config, output_dir, variant, no_render)` 保留为底层单变体接口，以免破坏已有调用和发布安全回归。新的 suite runner 负责 run ID、批量变体、汇总和 dashboard。物理计算、artifact 发布和 HTML 生成保持独立，任何一层都不能通过读取另一个模块的私有实现来工作。

## 5. 模块与接口

### 5.1 严格配置

新增 `src/webvideo_to_data/config.py`，用 dataclass 表达配置，不引入 Pydantic。加载器必须：

- 拒绝未知字段、缺少字段和错误类型；
- 验证 ROI、边界、模式、随机种子、阈值和扰动范围；
- 将所有影响结果的 IK、控制、碰撞、媒体和随机化参数写入 resolved config；
- 为配置指定 `schema_version: 2`；
- 对源路径做相对配置文件的解析，但 provenance 只发布经过脱敏的逻辑标识和内容哈希。

需要显式配置的参数包括：

- IK tolerance、最大迭代次数、阻尼、步长和姿态权重；
- 控制频率、每个阶段的持续时间和夹爪目标；
- 允许的 contact pair、最大允许穿透深度和 settle 时间；
- B0 扰动的初始位置、yaw、质量和摩擦范围；
- 媒体画布大小、letterbox 颜色、输出帧率和时间对齐模式。

### 5.2 环境预检

新增 `webvideo-to-data preflight --config <path>`。预检只读，不创建 artifact。它检查：

- Python 版本是否为 3.11；
- 配置是否通过 schema；
- 源视频是否存在且 SHA-256 匹配；
- ffmpeg/ffprobe 是否存在并能解析版本；
- MuJoCo 能否载入官方 Panda 场景；
- renderer 是否可用；传入 `--no-render` 时只检查 headless simulation；
- 输出父目录是否可写，但不覆盖或删除其中内容。

文本输出面向人，JSON 输出面向 CI。缺失项需要给出具体修复命令或配置字段，不能只返回 Python traceback。

### 5.3 官方 Panda 场景

仓库保存 MuJoCo Menagerie `franka_emika_panda` 的固定版本快照、原 LICENSE、来源 URL 和上游 commit。只保留场景实际需要的 XML 与 mesh，避免运行时静默下载。

场景包含 Panda、桌面、圆柱体和目标盒。视觉 geom 与 collision geom 分开；机器人 link、手掌、指尖、桌面、盒子和罐体都参与碰撞检测。允许的接触为：

- 初始和失败状态下的罐体与桌面；
- 抓取及运输阶段的左右指尖与罐体；
- 释放及 settle 阶段的罐体与目标盒。

机器人与桌面、机器人与盒子、非指尖机器人 link 与罐体、非法 self-collision 都是失败。任何未经允许的接触距离小于 `-0.002 m` 记为穿透失败。

### 5.4 IK 与分阶段控制

IK 在执行前离线求解每个关键 pose。每个 pose 最多迭代 200 次；位置误差阈值为 5 mm，姿态误差阈值为 5°。求解器包含 joint-limit null-space 项，并以上一帧解为初值。没有收敛的关键 pose 会在进入物理仿真前拒绝。

B0 使用如下阶段：

```text
home -> pregrasp -> approach -> close -> lift
     -> transport -> lower -> open -> retreat -> settle
```

抓取 frame 由罐体几何和夹爪 TCP 定义，不再把固定四元数直接套用到所有阶段。关键 pose 之间使用限速插值；关节位置、速度和加速度必须在配置限制内。控制器输出每一步实际发送给 MuJoCo actuator 的 control trace，但该文件命名为 `baseline_control_trace.npz`，不是 `actions.npz`。

### 5.5 物理验证与 B0 验收

单次 rollout 的成功条件全部满足才记为成功：

- 所有关键 IK pose 收敛，执行期目标跟踪帧比例不低于 95%；
- 左右指尖在 close/lift 阶段形成持续接触；
- 罐体相对初始位置至少抬升 50 mm；
- 放置后罐体中心距目标中心不超过 40 mm；
- settle 1 秒后罐体倾角不超过 15°，线速度不超过 0.02 m/s；
- 没有非法 self/environment contact，也没有超过 2 mm 的非法穿透；
- 所有关节位置、速度和加速度没有越界。

固定评测使用 30 个种子。每个种子扰动罐体初始 XY ±10 mm、yaw ±5°、质量 ±10%、接触摩擦 ±10%。通过线为至少 24/30 成功，并报告 Wilson 95% 置信区间。随机种子、扰动值和每次失败分类全部写入 artifact。

即使 B0 通过，它仍是手工 baseline。`action_export_eligible` 保持 `false`，原因改为 `manual_baseline_not_video_grounded`。只有后续视频重建 reference 通过同一验证器后，才有资格讨论 action export。

### 5.6 追加式运行记录

新的默认布局为：

```text
artifacts/<experiment-id>/runs/<run-id>/
  resolved-config.yaml
  environment.json
  suite-metrics.json
  dashboard/index.html
  B0/
  B1/
  B2/
  B3/
  B4/
```

`run-id` 使用 UTC 时间、配置哈希前八位和一个随机后缀，例如 `20260819T120102Z-a1b2c3d4-7f29`。如果目录已存在，命令失败，不覆盖旧 run。`artifacts/<experiment-id>/latest.json` 只保存最近一次完整 suite 的相对路径和 manifest hash；更新采用原子替换。

底层单变体接口继续接受显式 `output_dir`，保留现有可信 marker、snapshot、锁和发布失败隔离机制。suite runner 不绕过这些机制。

### 5.7 Artifact schema

manifest 升级到 format v4。每个 NPZ 同目录增加 schema JSON，至少包含：

- `schema_version`、producer 和生成 commit；
- array 名、dtype、shape 和语义；
- 单位、坐标系、四元数顺序和时间基准；
- source/config/model hash；
- terminal status、reason 和 action eligibility。

加载器强校验 schema。未知版本、错误 shape/dtype、缺少单位、非单调时间或坐标系不匹配都拒绝读取。旧 v3 artifact 保持只读兼容，不原地升级。

### 5.8 CLI 和退出码

安装后提供 `webvideo-to-data` console script：

```text
webvideo-to-data preflight --config configs/exp001.yaml
webvideo-to-data run --config configs/exp001.yaml --variant B0
webvideo-to-data run --config configs/exp001.yaml --all
webvideo-to-data dashboard --run artifacts/EXP-001/runs/<run-id>
webvideo-to-data verify --run artifacts/EXP-001/runs/<run-id>
```

退出码约定：

- `0`：命令完成，包括可审计的 `rejected` 或 `not_run`；
- `2`：CLI 或配置错误；
- `3`：环境/输入预检失败；
- `4`：artifact 验证失败；
- `10`：runner 基础设施失败。

`run --require-completed` 在任一请求变体不是 `completed` 时返回 `5`，供 CI 或批处理使用。任何错误文本写入 JSON 前都经过路径、URL query、Authorization、token 和常见 secret pattern 脱敏。

### 5.9 静态 dashboard 与媒体

dashboard 是无服务端的单页 HTML，只读取已发布的 JSON 和相对媒体路径。首屏显示：

```text
EXP-001
NO ACTION EXPORTED · 0 / 5 eligible
B0 manual physics baseline: completed/rejected
B1 kinematic diagnostic: rejected
B2-B4: not run
```

每个 variant 卡片展示 status、reason、可解释指标、输入/生成 commit 和 artifact 验证状态。工程修复历史不放在首屏，移到 reproducibility 附录。

所有媒体必须：

- 保持源视频宽高比，用 letterbox/pillarbox 填充，不做非等比 resize；
- 左上角显示 source time，仿真画面显示 sim time；
- 时间经过重采样时固定显示 `TIME-WARPED FOR COMPARISON`；
- rejected 媒体固定显示红色 `REJECTED — NOT ACTION DATA`；
- B1 固定显示 `KINEMATIC OBJECT-POSE OVERRIDE`；
- availability 指标旁写明 `availability != semantic accuracy`；
- 没有人工真值的字段显示 `N/A`，不能用 0 代替未测量。

GitHub README 继续嵌入低码率 GIF，但 GIF 必须包含相同水印。高清 MP4 仍作为 Release asset，dashboard 和 README 不能暗示浏览器中的动画是物理成功结果。

## 6. 数据与隐私

增加 source registry，记录 source ID、内容哈希、来源类型、采集日期、许可、是否可公开、隐私审查状态和访问说明。发布 provenance 不包含本地绝对路径。当前手机视频标记为 private/local-only，不进入 Git 或 Release。

`github_token.txt`、Seedance API 文件、Authorization header、带签名 URL 和 SSH 信息不能出现在 artifact、日志、dashboard、测试 fixture 或错误消息中。

Phase 2A 不要求公开 demo video。新 clone 用户可以完成安装、preflight 的环境部分、运行测试和读取版本化结果；运行 EXP-001 B0/B1 前仍需按 source registry 准备匹配哈希的本地视频。

## 7. 测试策略

所有行为变更采用测试先行。测试分四层：

1. 单元测试：严格配置、IK 收敛/拒绝、contact 分类、schema loader、脱敏、时间轴和媒体布局；
2. 集成测试：官方 Panda 场景加载、允许/非法 contact、单次 B0、追加式 run、manifest v4、dashboard 生成；
3. 回归测试：保留现有发布目录 TOCTOU、backup、rollback、action removal 和 no-render 测试；
4. 实验验收：固定 30-seed B0，检查至少 24 次成功、零非法碰撞和无 `actions.npz`。

CI 使用 Python 3.11，在 Windows 和 Ubuntu 上运行不依赖私有视频的测试。渲染测试使用明确的 headless backend；需要私有视频或完整 30-seed 的实验标为本地验收，不伪装成公共 CI 结果。

## 8. 文档与复现

README 增加从新 clone 开始的安装、preflight、测试、单变体、全 suite 和 dashboard 命令。命令块必须在一个干净 Python 3.11 环境中验证。依赖使用锁文件固定，CI 检查锁文件与 `pyproject.toml` 一致。

EXP-001 报告拆成结果正文和工程附录。正文只回答实验是否成功、失败发生在哪一层、是否导出 action。发布安全修复和实现历史保留在附录，不覆盖研究结论。

## 9. 完成条件

Phase 2A 完成需要同时满足：

- 正式 Panda 模型来源、commit 和许可证可审计；
- B0 固定 30-seed 至少 24 次成功，且没有非法碰撞或穿透；
- B0 不生成 `actions.npz`，只生成明确标注的 baseline control trace；
- 新 CLI、preflight、追加式 run、manifest v4、loader 和 dashboard 均有回归测试；
- Windows 和 Ubuntu 公共测试通过；
- dashboard、GIF 和 MP4 保持比例并显示状态、双时间轴和必要水印；
- 干净 Python 3.11 环境按 README 可以安装并运行公共测试；
- 私有视频和凭据没有进入 Git、Release、日志或 HTML；
- 文档将已完成结果、未运行项和下一阶段计划分开表述。

若 30-seed B0 没有达到 24 次成功，本阶段仍可交付失败报告，但不能被标记为物理基线完成；实现继续保持 action gate，并按 IK、控制、接触、场景或物理参数分类失败。

## 10. Phase 2B/2C 的接口边界

Phase 2B 只能向本阶段提供版本化的 `ObjectTrack2D`、`HandEvidence2D` 和人工标注引用。它不能直接操作 MuJoCo 或写 action。Phase 2C 提供带坐标系、尺度来源和不确定度的 `ObjectTrajectory3D`。两者都通过同一个 retarget、collision 和 physics validator，避免为某个视觉模型单独放宽验收门。

计划中的第一组视觉方法是 mask-gated LK、SAM2 + CoTracker3，以及资源允许时的 TAPNext/TAPIR。第一组深度方法是 Depth Pro 与 Metric3D；没有实测尺寸或标定证据时，它们的输出只标记为 exploratory。

参考实现：

- MuJoCo Menagerie Panda：https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda
- NVIDIA Video to Data：https://github.com/nvidia-isaac/video_to_data
- SAM2：https://github.com/facebookresearch/sam2
- CoTracker：https://github.com/facebookresearch/co-tracker
- TAP：https://github.com/google-deepmind/tapnet
- Depth Pro：https://github.com/apple/ml-depth-pro
- FoundationPose：https://github.com/NVlabs/FoundationPose

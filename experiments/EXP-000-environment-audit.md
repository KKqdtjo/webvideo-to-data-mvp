# EXP-000：素材与环境盘点

日期：2026-08-19  
状态：完成  
操作范围：只读；未安装依赖，未调用付费 API，未修改远端状态。

## 本地素材

| 文件 | 时长 | 分辨率 | FPS | 编码 | 初步判断 |
| --- | ---: | ---: | ---: | --- | --- |
| `guitar.webm` | 286.148 s | 1920×1080 | 29.97 | AV1 | 长时、双手、密集接触，不适合首轮 |
| `周興哲 - 如果雨之後 (acoustic guitar solo).webm` | 274.548 s | 1280×720 | 25 | AV1 | 长时、双手，作为负对照 |
| `手机录制.mp4` | 7.019 s | 540×960 | 30 | HEVC | 第一视角罐体抓放，选为 EXP-001 |
| `拋瓶甩瓶遊戲…成功.webm` | 6.228 s | 720×1280 | 30 | VP9 | 抛掷、旋转、液体，首轮只做拒绝测试 |

抽样帧显示，手机视频中的手、饮料罐、纸盒和接触阶段清楚。瓶子翻转视频有字幕遮挡，动作快，包含首轮不建模的液体和弹道阶段。

所有本地源视频均以 SHA-256 绑定：

| 文件 | SHA-256 |
| --- | --- |
| `guitar.webm` | `CEDD48C42311B9FB0B269FA5EFB6FBB25A8DEF4BAE746CE2A1B9EA8FD33E2CE5` |
| `周興哲 - 如果雨之後 (acoustic guitar solo).webm` | `14065DD6FC0775CBBB0B0CE792380CF051C42F6165658B2E60DA1D9B925A0BED` |
| `手机录制.mp4` | `55E98463B8E270F4E7D87BDA5D0EE73329880A96CF753FC2C78E1849F1C4AB17` |
| `拋瓶甩瓶遊戲｜慣性定律｜3分之一瓶身的水量｜絕對能成功.webm` | `B77431B42743319181EFD54866766ABC275A26A6C6F114FBBA9804034C94ED40` |

## 远端环境

连接：`root@8.160.161.137:1007`，BatchMode 非交互登录成功。

| 项目 | 结果 |
| --- | --- |
| OS | Ubuntu 24.04.2 LTS |
| CPU/RAM | 24 vCPU / 256 GiB |
| 磁盘 | 容器层约 718 GiB 可用；共享挂载约 6.3 TiB 可用 |
| Python | 3.12.3 |
| CUDA toolkit | 12.8 |
| GPU 形态 | `/dev/alixpu*`，没有 `nvidia-smi` |
| PyTorch | 2.6.0，但导入时报缺少 `libhggcrt1.so` |
| Docker/Isaac | 未安装 Docker、Isaac Sim、Isaac Lab |
| 视频工具 | 远端没有 ffprobe |

结论：当前远端不能直接运行 NVIDIA V2D 或 Isaac Lab。它可以在后续承担 CPU 任务或存储；视觉推理要等厂商运行库修复后再评估。

## 安全记录

`video-generation/seedance_api.txt` 已由 `.gitignore` 忽略。MVP 实施未读取该路径的内容；正式实现只允许通过环境变量读取凭据。若凭据曾出现在仓库、聊天或日志中，应先轮换。

## 决策

EXP-001 使用本地手机视频、Franka Panda 和 MuJoCo。当前不调用 Seedance，不在远端部署 NVIDIA 容器栈。

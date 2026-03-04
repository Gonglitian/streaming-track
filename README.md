# streaming-track

USB 摄像头实时捕捉人体动作，通过 GVHMR 提取 SMPL 参数，经 GMR 重定向到 Unitree G1 机器人，在 MuJoCo 中实时驱动。

支持两种模式：**离线管线**（录制视频 → 批量处理 → 回放）和 **实时管线**（摄像头 → 流式推理 → 实时驱动）。

## 系统要求

| 项目 | 要求 |
|------|------|
| OS | Linux (Ubuntu 20.04+) |
| GPU | NVIDIA GPU, CUDA 12.1 (测试于 RTX 4060 8GB) |
| Python | 3.10 |
| 摄像头 | USB 摄像头（实时模式需要） |
streaming-track
## 快速开始

### 1. 克隆项目

```bash
git clone <repo-url> streaming-track
cd streaming-track
```

### 2. 克隆第三方依赖

```bash
git clone https://github.com/YanjieZe/GMR.git third_party/GMR
git clone https://github.com/zju3dv/GVHMR.git third_party/GVHMR
```

### 3. 创建 conda 环境

**方式 A：一键脚本**

```bash
bash setup_env.sh
```

**方式 B：手动安装**

```bash
conda create -n streaming-track python=3.10 -y
conda activate streaming-track

# 安装所有依赖
pip install -r requirements.txt

# 以 editable 模式安装 GMR 和 GVHMR
pip install -e third_party/GMR
pip install -e third_party/GVHMR

# 修复 Linux 渲染所需的 libstdc++
conda install -c conda-forge libstdcxx-ng -y
```

### 4. 下载模型权重

```bash
# 下载 GVHMR 预训练权重 (YOLOv8, ViTPose, HMR2, GVHMR)
bash scripts/download_gvhmr_weights.sh
```

### 5. 下载 SMPL-X Body Models

GMR 的 IK 重定向需要 SMPL-X body models：

```bash
# 方式 A：如果有下载好的 zip
bash scripts/setup_body_models.sh /path/to/models_smplx_v1_1.zip

# 方式 B：手动下载
# 1. 注册 https://smpl-x.is.tue.mpg.de/
# 2. 下载 SMPL-X v1.1
# 3. 解压到 third_party/GMR/assets/body_models/smplx/
```

最终目录结构：

```
third_party/GMR/assets/body_models/smplx/
├── SMPLX_FEMALE.npz
├── SMPLX_MALE.npz
└── SMPLX_NEUTRAL.npz
```

### 6. 验证安装

```bash
conda activate streaming-track

# 验证所有 Python 包正确安装
python verify_env.py

# 验证 GMR + G1 机器人部署 (6 项检查)
python scripts/verify_gmr.py

# (可选) 性能基准测试
python scripts/benchmark_retarget.py
```

---

## 使用方法

### 离线管线

录制视频（或使用已有视频），批量运行 GVHMR + GMR，最后在 MuJoCo 中回放。

```bash
conda activate streaming-track

# 用摄像头录制 10 秒视频，然后运行全管线（-s 表示静态摄像头）
python scripts/offline_pipeline.py --record --duration 10 -s

# 使用已有视频
python scripts/offline_pipeline.py --video videos/my_motion.mp4 -s

# 跳过 GVHMR 推理，直接从 .pt 结果文件开始
python scripts/offline_pipeline.py --gvhmr_pt outputs/result.pt

# 无 GUI 模式，导出 MuJoCo 回放视频
python scripts/offline_pipeline.py --video input.mp4 -s --headless --export_video playback.mp4

# 只做重定向，不回放
python scripts/offline_pipeline.py --video input.mp4 -s --no_playback
```

离线管线共 5 步：

```
[1/5] 录制视频 (--record) 或加载已有视频 (--video)
[2/5] GVHMR 推理 → smpl_params_global (.pt)
[3/5] SMPL-X 前向运动学 → {joint_name: (pos, quat)} per frame
[4/5] GMR IK 重定向 → qpos [36-dim] per frame
[5/5] MuJoCo G1 回放 (交互式查看器 / 无头渲染 / 视频导出)
```

### 实时管线

摄像头实时捕捉，流式 GVHMR 推理，GMR 重定向到 G1 机器人。

```bash
conda activate streaming-track

# 基础实时模式（静态摄像头，显示性能仪表盘）
python scripts/realtime_pipeline.py

# 开启所有可视化
python scripts/realtime_pipeline.py --vis-all

# 显示 2D 骨架叠加
python scripts/realtime_pipeline.py --vis-pose

# 显示人体 vs 机器人对比视图
python scripts/realtime_pipeline.py --vis-compare

# 自定义 GVHMR 滑窗参数
python scripts/realtime_pipeline.py --window 90 --stride 15

# 无 GUI 模式（仅终端输出性能指标）
python scripts/realtime_pipeline.py --headless

# 指定摄像头和分辨率
python scripts/realtime_pipeline.py -c 0 --width 1280 --height 720 --fps 30
```

**实时参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--window` | 90 | GVHMR 滑动窗口大小（帧数） |
| `--stride` | 15 | 每隔多少帧触发一次 GVHMR 推理 |
| `--ema-alpha` | 0.3 | EMA 平滑系数（减少帧间抖动） |
| `--vis-pose` | off | 2D 骨架关键点叠加 |
| `--vis-dashboard` | on | 性能指标仪表盘 |
| `--vis-compare` | off | 人体 vs 机器人并排对比 |
| `--vis-all` | off | 开启全部可视化 |

### 单独使用 GVHMR 推理

```bash
# 对视频运行 GVHMR，输出 .pt 文件
python scripts/gvhmr_infer.py --video input.mp4 --output results.pt

# 静态摄像头模式（跳过视觉里程计，更快更稳定）
python scripts/gvhmr_infer.py --video input.mp4 --output results.pt -s
```

### 摄像头工具

```bash
# 检测可用摄像头
python -m src.camera.main detect

# 实时预览
python -m src.camera.main stream

# 录制到 MP4
python -m src.camera.main record -o output.mp4
```

---

## 项目结构

```
streaming-track/
├── third_party/
│   ├── GMR/                        # General Motion Retargeting (IK-based)
│   └── GVHMR/                      # Gravity-View Human Motion Recovery
├── src/
│   ├── camera/                     # USB 摄像头采集模块
│   │   ├── detector.py             #   摄像头检测
│   │   ├── recorder.py             #   MP4 录制
│   │   ├── streamer.py             #   实时帧流 (线程安全)
│   │   ├── fps_counter.py          #   FPS 计数器
│   │   └── main.py                 #   CLI 入口
│   ├── bridge/                     # GVHMR → GMR 数据桥接
│   │   ├── gvhmr_to_gmr.py         #   SMPL 参数 → 关节字典
│   │   └── mujoco_playback.py      #   MuJoCo G1 回放/渲染
│   ├── realtime/                   # 实时管线
│   │   ├── preprocessor.py         #   单帧特征提取 (YOLO+ViTPose+HMR2)
│   │   ├── sliding_window.py       #   滑动窗口 + GVHMR 推理
│   │   └── pipeline.py             #   4 线程编排器
│   └── vis/                        # 可视化工具
│       ├── pose_overlay.py         #   2D 骨架叠加 (COCO-17)
│       ├── dashboard.py            #   性能指标 HUD
│       ├── skeleton_viewer.py      #   3D 关节查看器 (MuJoCo)
│       ├── smpl_renderer.py        #   SMPL mesh 叠加 (pytorch3d)
│       └── comparison_view.py      #   人体 vs 机器人对比
├── scripts/
│   ├── offline_pipeline.py         # 离线管线 CLI
│   ├── realtime_pipeline.py        # 实时管线 CLI
│   ├── gvhmr_infer.py              # GVHMR 推理脚本
│   ├── verify_gmr.py               # GMR 验证 (6 项检查)
│   ├── benchmark_retarget.py       # 性能基准测试
│   ├── download_gvhmr_weights.sh   # GVHMR 权重下载
│   └── setup_body_models.sh        # SMPL-X body model 安装
├── requirements.txt                # Python 依赖
├── setup_env.sh                    # 一键环境搭建
├── verify_env.py                   # 环境验证
└── CLAUDE.md                       # 开发者参考文档
```

## 技术架构

### 数据流

```
USB 摄像头 (30fps)
    │
    ▼
┌─────────────────────────────────────────┐
│  GVHMR (Gravity-View HMR)              │
│  YOLOv8 → ViTPose → HMR2 → Transformer │
│  输出: smpl_params_global               │
│    - global_orient (F, 3)  轴角        │
│    - body_pose (F, 63)     21关节轴角   │
│    - betas (F, 10)         体型参数     │
│    - transl (F, 3)         全局位移     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Bridge (src/bridge/)                   │
│  SMPLX 前向运动学: 轴角 → 全局旋转+位置 │
│  坐标变换: Y-up (OpenGL) → Z-up (MuJoCo)│
│  输出: {joint_name: (pos_3d, quat_wxyz)} │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  GMR (General Motion Retargeting)       │
│  两阶段 IK (mink 库):                   │
│    Phase 1: 粗对齐 (低权重)              │
│    Phase 2: 精调 (高权重)               │
│  输出: qpos [36-dim]                    │
│    - root_pos (3) + root_quat (4)       │
│    - 29 DOF joint angles                │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  MuJoCo Simulation                      │
│  Unitree G1 (29 DOF)                    │
│  交互式查看器 / 无头渲染 / 视频导出       │
└─────────────────────────────────────────┘
```

### 实时管线线程架构

```
Thread 1 (Camera):      摄像头 → FrameBuffer (只保留最新帧)
Thread 2 (Preprocess):  单帧 YOLO+ViTPose+HMR2 → SlidingWindowBuffer
Thread 3 (Inference):   90 帧窗口积满 → GVHMR 推理 (每 15 帧触发一次)
Main Thread (Retarget): Bridge 转换 → GMR IK → EMA 平滑 → 显示/渲染
```

- **窗口**: 90 帧 (3 秒 @ 30fps)
- **步进**: 15 帧 → 每 0.5 秒触发一次推理
- **延迟**: ~3 秒（窗口积累时间）
- **平滑**: EMA (alpha=0.3) 消除窗口间跳变

### GPU 显存预算 (RTX 4060 8GB)

| 模块 | 显存 |
|------|------|
| YOLOv8s | ~300 MB |
| ViTPose-H | ~300 MB |
| HMR2 | ~500 MB |
| GVHMR model | ~1.5 GB |
| 工作显存 | ~1.5 GB |
| **合计** | **~4.1 GB** (余量 ~3.9 GB) |

## 关键配置

| 配置项 | 路径 |
|--------|------|
| 机器人 | Unitree G1 (29 DOF) |
| IK 配置 | `third_party/GMR/general_motion_retargeting/ik_configs/smplx_to_g1.json` |
| 机器人 MJCF | `third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml` |
| SMPLX 模型 | `third_party/GMR/assets/body_models/smplx/` |
| GVHMR 权重 | `third_party/GVHMR/inputs/checkpoints/` |

## 常见问题

### Q: GVHMR 推理报错 hydra 配置问题？

GVHMR 需要在其目录下运行。脚本已通过 context manager 处理了 `os.chdir`，如果直接调用 `gvhmr_infer.py` 出错，确认 `third_party/GVHMR/` 目录完整。

### Q: 实时模式延迟太高？

- 减小 `--window` 参数（如 45 帧），但精度会下降
- 增大 `--stride` 参数（如 30 帧），减少推理频率
- 使用 `-s`（静态摄像头）跳过视觉里程计

### Q: 显存不够？

实时模式约需 4.1GB 显存。如果 OOM：
- 确保没有其他 GPU 进程在运行
- 减小摄像头分辨率: `--width 320 --height 240`

### Q: SMPL-X body models 格式不对？

GMR 默认读取 `.npz` 格式。如果下载的是 `.pkl`，参考 `scripts/setup_body_models.sh` 中的说明。

## 性能参考 (RTX 4060)

| 指标 | 数值 |
|------|------|
| GMR IK 重定向 (无渲染) | ~100 FPS |
| GMR IK + MuJoCo 渲染 | ~85 FPS |
| GVHMR 离线推理 (45s 视频) | ~280 ms |

## 致谢

- [GVHMR](https://github.com/zju3dv/GVHMR) - Gravity-View Human Motion Recovery (SIGGRAPH Asia 2024)
- [GMR](https://github.com/YanjieZe/GMR) - General Motion Retargeting
- [SMPL-X](https://smpl-x.is.tue.mpg.de/) - Expressive Body Model
- [MuJoCo](https://mujoco.org/) - Physics Simulation

# streaming-track

Real-time human motion capture → robot motion retargeting pipeline using GVHMR + GMR on Unitree G1.

## Setup

```bash
# Create unified conda environment
conda create -n streaming-track python=3.10 -y
conda activate streaming-track
pip install -r requirements.txt

# Install repos in editable mode
pip install -e third_party/GMR
pip install -e third_party/GVHMR
conda install -c conda-forge libstdcxx-ng -y

# Download GVHMR model weights
bash scripts/download_gvhmr_weights.sh

# (Optional) Download SMPL-X body models for real data testing
bash scripts/setup_body_models.sh [path_to_models.zip]
```

## Architecture

- `third_party/GMR/` - General Motion Retargeting library (IK-based)
- `third_party/GVHMR/` - GVHMR inference (cloned separately)
- `src/camera/` - USB camera capture (detect, record, stream)
- `src/bridge/` - GVHMR→GMR data bridge and MuJoCo playback
  - `gvhmr_to_gmr.py` - Converts GVHMR .pt → per-joint dicts via SMPL-X forward pass
  - `mujoco_playback.py` - Headless/interactive G1 playback with video export
- `src/realtime/` - Real-time GVHMR+GMR pipeline (4-thread architecture)
  - `preprocessor.py` - Per-frame YOLO + ViTPose + HMR2 feature extraction
  - `sliding_window.py` - Thread-safe buffer with GVHMR inference trigger
  - `pipeline.py` - Orchestrator with EMA smoothing
- `src/vis/` - Visualization tools
  - `pose_overlay.py` - 2D skeleton overlay (COCO-17, OpenCV)
  - `dashboard.py` - Pipeline metrics HUD
  - `skeleton_viewer.py` - 3D SMPLX joint viewer (MuJoCo offscreen)
  - `smpl_renderer.py` - SMPL mesh overlay (pytorch3d)
  - `comparison_view.py` - Side-by-side human vs robot
- `scripts/offline_pipeline.py` - End-to-end offline pipeline (record→GVHMR→GMR→MuJoCo)
- `scripts/realtime_pipeline.py` - Real-time pipeline CLI
- `scripts/gvhmr_infer.py` - GVHMR inference on video files
- `scripts/verify_gmr.py` - Deployment verification (6-check suite)
- `scripts/benchmark_retarget.py` - Performance benchmarks
- `scripts/setup_body_models.sh` - SMPL-X body model setup helper
- `scripts/download_gvhmr_weights.sh` - GVHMR model weight downloader

## Key Technical Details

- **Robot**: Unitree G1 (29 DOF, pelvis root)
- **IK Config**: `third_party/GMR/general_motion_retargeting/ik_configs/smplx_to_g1.json`
- **Robot Model**: `third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml`
- **Performance**: ~100 FPS IK-only, ~85 FPS IK+render on RTX 4060

## GMR API

```python
from general_motion_retargeting import GeneralMotionRetargeting as GMR

retarget = GMR(actual_human_height=1.8, src_human="smplx", tgt_robot="unitree_g1")
qpos = retarget.retarget(human_data)  # dict of {body_name: (pos, quat_wxyz)}
# qpos: [7 (root pos+quat) + 29 (joint DOFs)] = 36 dims
```

## Offline Pipeline

Each run creates a timestamped output folder: `outputs/<source>_<YYYYMMDD_HHMMSS>/`

Internal file naming is consistent regardless of input mode:
- `input.mp4` — source video (recorded or symlinked)
- `gvhmr.pt` — GVHMR SMPL parameters
- `incam.mp4` — SMPL mesh overlay (optional)
- `retarget.pkl` — retargeted robot motion
- `playback.mp4` — MuJoCo playback video (optional)

```bash
# Full pipeline: record 10s video → GVHMR → GMR → MuJoCo viewer
python scripts/offline_pipeline.py --record --duration 10 -s

# From existing video (static camera)
python scripts/offline_pipeline.py --video videos/my_motion.mp4 -s

# From existing GVHMR .pt result (skip recording + inference)
python scripts/offline_pipeline.py --gvhmr_pt outputs/result.pt

# Headless video export (no GUI)
python scripts/offline_pipeline.py --video input.mp4 -s --headless --export_video playback.mp4

# Retarget only (no playback)
python scripts/offline_pipeline.py --video input.mp4 -s --no_playback
```

### Data Flow

```
USB Camera / MP4 → GVHMR .pt (smpl_params_global)
    → SMPL-X forward pass (src/bridge/gvhmr_to_gmr.py)
    → {joint_name: (pos, quat_wxyz)} per frame
    → GMR.retarget() → qpos [36-dim]
    → MuJoCo viewer / video export
```

## Real-time Pipeline

```bash
# Basic real-time (static camera, dashboard HUD)
python scripts/realtime_pipeline.py

# With all visualization
python scripts/realtime_pipeline.py --vis-all

# Custom GVHMR window (90 frames, 15 stride = ~3s latency)
python scripts/realtime_pipeline.py --window 90 --stride 15

# Headless (metrics only, no GUI)
python scripts/realtime_pipeline.py --headless
```

### Real-time Architecture

```
Thread 1: CameraStreamer → FrameBuffer
Thread 2: Preprocessor (YOLO+ViTPose+HMR2) → SlidingWindowBuffer
Thread 3: GVHMR inference (90-frame window, 15-frame stride)
Main:     GMR retarget → display/render
```

## Running

```bash
conda activate streaming-track
python scripts/verify_gmr.py      # Full verification
python scripts/benchmark_retarget.py  # Performance benchmark
```

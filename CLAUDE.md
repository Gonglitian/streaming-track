# streaming-track

Real-time human motion capture → robot motion retargeting pipeline using GVHMR + GMR on Unitree G1.

## Setup

```bash
# Create conda environment
conda create -n gmr python=3.10 -y
conda activate gmr

# Clone and install GMR
git clone https://github.com/YanjieZe/GMR.git third_party/GMR
pip install -e third_party/GMR
conda install -c conda-forge libstdcxx-ng -y

# (Optional) Download SMPL-X body models for real data testing
bash scripts/setup_body_models.sh [path_to_models.zip]
```

## Architecture

- `third_party/GMR/` - General Motion Retargeting library (IK-based)
- `gvhmr_repo/` - GVHMR inference (cloned by `scripts/gvhmr_setup.sh`)
- `src/camera/` - USB camera capture (detect, record, stream)
- `src/bridge/` - GVHMR→GMR data bridge and MuJoCo playback
  - `gvhmr_to_gmr.py` - Converts GVHMR .pt → per-joint dicts via SMPL-X forward pass
  - `mujoco_playback.py` - Headless/interactive G1 playback with video export
- `scripts/offline_pipeline.py` - End-to-end offline pipeline (record→GVHMR→GMR→MuJoCo)
- `scripts/gvhmr_infer.py` - GVHMR inference on video files
- `scripts/verify_gmr.py` - Deployment verification (6-check suite)
- `scripts/benchmark_retarget.py` - Performance benchmarks
- `scripts/setup_body_models.sh` - SMPL-X body model setup helper

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

## Running

```bash
conda activate gmr
python scripts/verify_gmr.py      # Full verification
python scripts/benchmark_retarget.py  # Performance benchmark
```

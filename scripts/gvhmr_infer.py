#!/usr/bin/env python3
"""
GVHMR Inference Script
Extracts world-grounded SMPL parameters from a monocular video.

Usage:
    python scripts/gvhmr_infer.py --video input.mp4 --output results.pt
    python scripts/gvhmr_infer.py --video input.mp4 --output results.pt --static_cam
    python scripts/gvhmr_infer.py --video input.mp4 --output results.pt --skip_render
"""

import sys
import os
import argparse
from pathlib import Path

# Resolve CLI args before changing directory
_ORIG_CWD = os.getcwd()

# Add gvhmr_repo to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GVHMR_DIR = PROJECT_ROOT / "gvhmr_repo"
sys.path.insert(0, str(GVHMR_DIR))

os.chdir(GVHMR_DIR)

import torch
import numpy as np
from hydra import initialize_config_module, compose

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.pylogger import Log
from hmr4d.utils.video_io_utils import get_video_lwh, get_writer, get_video_reader
from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from tqdm import tqdm
import hydra


def run_inference(video_path: str, output_path: str, static_cam: bool = False,
                  skip_render: bool = False, f_mm: int = None):
    """Run GVHMR inference on a video file.

    Args:
        video_path: Path to input video
        output_path: Path to save .pt results
        static_cam: If True, assume static camera (skip visual odometry)
        skip_render: If True, skip mesh rendering (faster, output only .pt)
        f_mm: Focal length of fullframe camera in mm (None for auto-estimate)

    Returns:
        dict with smpl_params_global and smpl_params_incam
    """
    video_path = Path(video_path)
    assert video_path.exists(), f"Video not found: {video_path}"

    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input] {video_path}: {length} frames, {width}x{height}")

    # Setup hydra config
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={static_cam}",
            f"verbose=False",
            f"use_dpvo=False",
        ]
        if f_mm is not None:
            overrides.append(f"f_mm={f_mm}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy video to working dir
    if not Path(cfg.video_path).exists():
        reader = get_video_reader(video_path)
        writer = get_writer(cfg.video_path, fps=30, crf=23)
        for img in tqdm(reader, total=length, desc="Copy video"):
            writer.write_frame(img)
        writer.close()
        reader.close()

    paths = cfg.paths

    # === Preprocessing === #
    Log.info("[Preprocess] Starting...")
    tic = Log.time()

    # 1. YOLOv8 tracking
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(cfg.video_path).float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]

    # 2. ViTPose
    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(cfg.video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose)

    # 3. HMR2 features
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(cfg.video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor

    # 4. Visual odometry (if not static cam)
    if not static_cam:
        if not Path(paths.slam).exists():
            simple_vo = SimpleVO(cfg.video_path, scale=0.5, step=8, method="sift", f_mm=f_mm)
            vo_results = simple_vo.compute()
            torch.save(vo_results, paths.slam)

    Log.info(f"[Preprocess] Done in {Log.time()-tic:.2f}s")

    # === Load preprocessed data === #
    from pytorch3d.transforms import quaternion_to_matrix

    if static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(paths.slam)
        R_w2c = torch.from_numpy(traj[:, :3, :3])

    if f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features),
    }

    # === GVHMR Inference === #
    Log.info("[GVHMR] Running model inference...")
    model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()

    tic = Log.sync_time()
    with torch.no_grad():
        pred = model.predict(data, static_cam=static_cam)
    pred = detach_to_cpu(pred)
    Log.info(f"[GVHMR] Inference: {Log.sync_time()-tic:.2f}s for {length/30:.1f}s video")

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pred, output_path)
    Log.info(f"[Output] Saved to {output_path}")

    # Print summary
    params = pred["smpl_params_global"]
    Log.info(f"[Summary] smpl_params_global:")
    for k, v in params.items():
        Log.info(f"  {k}: {v.shape}")

    return pred


def main():
    parser = argparse.ArgumentParser(description="GVHMR Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default=None, help="Path to save .pt output")
    parser.add_argument("--static_cam", "-s", action="store_true", help="Assume static camera")
    parser.add_argument("--skip_render", action="store_true", help="Skip rendering visualization")
    parser.add_argument("--f_mm", type=int, default=None, help="Focal length in mm")
    args = parser.parse_args()

    # Resolve paths relative to original CWD (before os.chdir to gvhmr_repo)
    args.video = str((Path(_ORIG_CWD) / args.video).resolve())
    if args.output is None:
        args.output = str((Path(_ORIG_CWD) / f"outputs/{Path(args.video).stem}_gvhmr.pt").resolve())
    else:
        args.output = str((Path(_ORIG_CWD) / args.output).resolve())

    Log.info(f"[GPU] {torch.cuda.get_device_name()}")
    pred = run_inference(args.video, args.output, args.static_cam, args.skip_render, args.f_mm)


if __name__ == "__main__":
    main()

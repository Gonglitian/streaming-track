#!/usr/bin/env python3
"""
Offline Pipeline: Record → GVHMR → GMR → MuJoCo Playback

End-to-end script that chains:
  1. Camera recording (or accept existing MP4)
  2. GVHMR inference (monocular video → SMPL params)
  3. Data bridge (SMPL params → per-joint positions/orientations)
  4. GMR retargeting (human joints → Unitree G1 qpos)
  5. MuJoCo playback (interactive viewer or headless video export)

Output directory: outputs/<source>_<YYYYMMDD_HHMMSS>/
  Each run creates a timestamped folder with consistent internal naming:
    input.mp4       - Source video (recorded or symlinked)
    gvhmr.pt        - GVHMR SMPL parameters
    incam.mp4       - SMPL mesh overlay (optional, --vis_gvhmr)
    retarget.pkl    - Retargeted robot motion
    playback.mp4    - MuJoCo playback (optional, --export_video)

Usage:
    # Full pipeline from recording
    python scripts/offline_pipeline.py --record --duration 10

    # From existing video
    python scripts/offline_pipeline.py --video videos/my_motion.mp4

    # From existing GVHMR .pt result (skip steps 1-2)
    python scripts/offline_pipeline.py --gvhmr_pt outputs/result.pt

    # Headless mode with video export
    python scripts/offline_pipeline.py --video input.mp4 --headless --export_video playback.mp4
"""

import sys
import os
import argparse
import time
import pathlib
import pickle
from datetime import datetime

import numpy as np

# Project paths
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GMR_ROOT = PROJECT_ROOT / "third_party" / "GMR"
GVHMR_DIR = PROJECT_ROOT / "third_party" / "GVHMR"
BODY_MODELS_DIR = GMR_ROOT / "assets" / "body_models"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(GMR_ROOT))


def step_record(args) -> str:
    """Step 1: Record video from USB camera."""
    print()
    print("=" * 60)
    print("[1/5] Recording video from camera")
    print("=" * 60)

    from src.camera.recorder import CameraRecorder

    output_path = args.output_dir / "input.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Camera index: {args.camera}")
    print(f"  Duration: {args.duration}s")
    print(f"  Resolution: {args.width}x{args.height} @ {args.record_fps} FPS")
    print(f"  Output: {output_path}")
    print()

    recorder = CameraRecorder(
        camera_index=args.camera,
        output_path=str(output_path),
        width=args.width,
        height=args.height,
        fps=args.record_fps,
        preview=not args.headless,
    )

    print("  Recording... (press Ctrl+C to stop early)")
    recorder.start()
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\n  Recording stopped by user.")
    recorder.stop()

    print(f"  -> Recorded to {output_path} ({recorder.current_fps:.1f} FPS)")
    return str(output_path)


def step_gvhmr(video_path: str, args) -> str:
    """Step 2: Run GVHMR inference on video."""
    print()
    print("=" * 60)
    print("[2/5] GVHMR inference")
    print("=" * 60)

    pt_path = args.output_dir / "gvhmr.pt"

    if pt_path.exists() and not args.force:
        print(f"  GVHMR result already exists: {pt_path}")
        print(f"  (use --force to re-run)")
        return str(pt_path)

    print(f"  Video: {video_path}")
    print(f"  Output: {pt_path}")
    print(f"  Static camera: {args.static_cam}")

    # Import and call directly (avoid sys.executable which may be Cursor AppImage)
    sys.path.insert(0, str(SCRIPT_DIR))
    from gvhmr_infer import run_inference

    try:
        run_inference(
            video_path=str(video_path),
            output_path=str(pt_path),
            static_cam=args.static_cam,
            skip_render=True,
        )
    except Exception as e:
        print(f"  -> FAIL: GVHMR inference failed: {e}")
        sys.exit(1)

    print(f"  -> GVHMR result saved to {pt_path}")
    return str(pt_path)


def step_bridge(pt_path: str, args) -> list:
    """Step 3: Convert GVHMR .pt → per-frame SMPL-X joint dicts."""
    print()
    print("=" * 60)
    print("[3/5] GVHMR → SMPL-X data bridge")
    print("=" * 60)

    from src.bridge.gvhmr_to_gmr import load_gvhmr_result, gvhmr_to_smplx_frames

    print(f"  Loading: {pt_path}")
    gvhmr_result = load_gvhmr_result(pt_path)

    num_frames = gvhmr_result["body_pose"].shape[0]
    print(f"  Frames: {num_frames}")
    print(f"  Betas shape: {gvhmr_result['betas'].shape}")
    print(f"  Body models: {BODY_MODELS_DIR}")

    tic = time.perf_counter()
    frames, aligned_fps, human_height = gvhmr_to_smplx_frames(
        gvhmr_result,
        smplx_model_path=str(BODY_MODELS_DIR),
        tgt_fps=30,
    )
    elapsed = time.perf_counter() - tic

    print(f"  Human height estimate: {human_height:.2f}m")
    print(f"  Output frames: {len(frames)} @ {aligned_fps} FPS")
    print(f"  Joint names: {list(frames[0].keys())[:5]}...")
    print(f"  Conversion time: {elapsed:.2f}s")
    print(f"  -> Bridge conversion complete")

    return frames, aligned_fps, human_height


def step_retarget(frames: list, human_height: float, fps: float, args) -> np.ndarray:
    """Step 4: GMR retargeting → robot qpos."""
    print()
    print("=" * 60)
    print("[4/5] GMR retargeting (SMPL-X → Unitree G1)")
    print("=" * 60)

    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    retarget = GMR(
        actual_human_height=human_height,
        src_human="smplx",
        tgt_robot="unitree_g1",
        verbose=False,
    )
    print(f"  GMR initialized (human={human_height:.2f}m → G1)")

    num_frames = len(frames)
    qpos_list = []

    tic = time.perf_counter()
    for i, frame in enumerate(frames):
        qpos = retarget.retarget(frame)
        qpos_list.append(qpos.copy())
        if (i + 1) % 100 == 0 or (i + 1) == num_frames:
            print(f"  Retargeted {i+1}/{num_frames} frames...", end="\r")
    elapsed = time.perf_counter() - tic
    print()

    qpos_array = np.array(qpos_list)
    ik_fps = num_frames / elapsed

    print(f"  Output shape: {qpos_array.shape}")
    print(f"  root_pos range: {qpos_array[:, :3].min(axis=0).round(3)} ~ {qpos_array[:, :3].max(axis=0).round(3)}")
    print(f"  dof_pos range: [{qpos_array[:, 7:].min():.3f}, {qpos_array[:, 7:].max():.3f}]")
    print(f"  IK speed: {ik_fps:.1f} FPS ({elapsed:.2f}s)")

    # Save retargeted motion as pickle (GMR-compatible format)
    pkl_path = args.output_dir / "retarget.pkl"
    root_pos = qpos_array[:, :3]
    root_rot = qpos_array[:, 3:7][:, [1, 2, 3, 0]]  # wxyz → xyzw for storage
    dof_pos = qpos_array[:, 7:]
    motion_data = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"  Saved motion pickle: {pkl_path}")

    print(f"  -> Retargeting complete")
    return qpos_array


def step_playback(qpos_array: np.ndarray, fps: float, args) -> None:
    """Step 5: MuJoCo playback."""
    print()
    print("=" * 60)
    print("[5/5] MuJoCo G1 playback")
    print("=" * 60)

    from src.bridge.mujoco_playback import MujocoPlayback

    video_path = None
    if args.export_video:
        export_name = args.export_video if args.export_video != "auto" else "playback.mp4"
        video_path = str(args.output_dir / export_name)
        print(f"  Export video: {video_path}")

    playback = MujocoPlayback(headless=args.headless)

    print(f"  Frames: {len(qpos_array)}")
    print(f"  FPS: {fps}")
    print(f"  Mode: {'headless' if args.headless else 'interactive viewer'}")

    if args.headless and video_path:
        print(f"  Rendering video...")
        tic = time.perf_counter()
        playback.play(qpos_array, fps=fps, video_path=video_path)
        elapsed = time.perf_counter() - tic
        render_fps = len(qpos_array) / elapsed
        print(f"  Render speed: {render_fps:.1f} FPS ({elapsed:.2f}s)")
        print(f"  -> Video saved to {video_path}")
    elif args.headless:
        print(f"  Headless playback (no video export)...")
        playback.play(qpos_array, fps=fps)
        print(f"  -> Playback complete")
    else:
        print(f"  Launching interactive viewer (close window to exit)...")
        playback.play(qpos_array, fps=fps, loop=True)
        print(f"  -> Viewer closed")


def main():
    parser = argparse.ArgumentParser(
        description="Offline Pipeline: Record → GVHMR → GMR → MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--record", action="store_true",
                             help="Record from USB camera first")
    input_group.add_argument("--video", type=str,
                             help="Path to existing MP4 video")
    input_group.add_argument("--gvhmr_pt", type=str,
                             help="Path to existing GVHMR .pt result (skip steps 1-2)")

    # Recording options
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Recording duration in seconds (default: 10)")
    parser.add_argument("--width", type=int, default=640, help="Recording width")
    parser.add_argument("--height", type=int, default=480, help="Recording height")
    parser.add_argument("--record_fps", type=int, default=30, help="Recording FPS")

    # GVHMR options
    parser.add_argument("--static_cam", "-s", action="store_true",
                        help="Assume static camera (skip visual odometry)")

    # Playback options
    parser.add_argument("--headless", action="store_true",
                        help="Headless mode (no viewer window)")
    parser.add_argument("--export_video", type=str, default=None,
                        help="Export MuJoCo playback video (filename, saved to output_dir)")
    parser.add_argument("--no_playback", action="store_true",
                        help="Skip playback step (only generate retargeted motion)")

    # Visualization
    parser.add_argument("--vis_gvhmr", action="store_true",
                        help="Render SMPL mesh overlay on original video (incam)")

    # General
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: outputs/<source>_<YYYYMMDD_HHMMSS>)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run of cached steps")

    args = parser.parse_args()

    # Default: if nothing specified, show help
    if not args.record and not args.video and not args.gvhmr_pt:
        parser.print_help()
        print("\nError: Specify one of --record, --video, or --gvhmr_pt")
        sys.exit(1)

    # Resolve output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        args.output_dir = pathlib.Path(args.output_dir)
    elif args.video:
        source_name = pathlib.Path(args.video).stem
        args.output_dir = PROJECT_ROOT / "outputs" / f"{source_name}_{timestamp}"
    elif args.gvhmr_pt:
        source_name = pathlib.Path(args.gvhmr_pt).stem
        # Strip _gvhmr suffix if present (e.g., tennis_gvhmr → tennis)
        if source_name.endswith("_gvhmr"):
            source_name = source_name[:-6]
        args.output_dir = PROJECT_ROOT / "outputs" / f"{source_name}_{timestamp}"
    else:
        args.output_dir = PROJECT_ROOT / "outputs" / f"recorded_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Offline Pipeline: Record → GVHMR → GMR → MuJoCo     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Output dir: {args.output_dir}")

    pipeline_start = time.perf_counter()

    # === Step 1: Record (optional) ===
    if args.gvhmr_pt:
        src_pt = pathlib.Path(args.gvhmr_pt).resolve()
        # Symlink into output dir for consistency
        gvhmr_link = args.output_dir / "gvhmr.pt"
        if not gvhmr_link.exists():
            gvhmr_link.symlink_to(src_pt)
        pt_path = str(src_pt)
        print(f"\n  Skipping steps 1-2, using existing GVHMR result: {pt_path}")
    else:
        if args.record:
            video_path = step_record(args)
        else:
            src_video = pathlib.Path(args.video).resolve()
            if not src_video.exists():
                print(f"\nError: Video not found: {src_video}")
                sys.exit(1)
            # Symlink source video into output dir as input.mp4
            input_link = args.output_dir / "input.mp4"
            if not input_link.exists():
                input_link.symlink_to(src_video)
            video_path = str(src_video)
            print(f"\n  Using existing video: {video_path}")

        # === Step 2: GVHMR ===
        pt_path = step_gvhmr(video_path, args)

        # === Step 2b: GVHMR incam visualization (optional) ===
        if args.vis_gvhmr:
            print()
            print("=" * 60)
            print("[2b] Rendering GVHMR incam overlay")
            print("=" * 60)
            from gvhmr_infer import render_incam
            incam_video = str(args.output_dir / "incam.mp4")
            try:
                render_incam(video_path, pt_path, incam_video)
            except Exception as e:
                print(f"  [WARN] Incam render failed: {e}")

    # === Step 3: Bridge ===
    frames, fps, human_height = step_bridge(pt_path, args)

    # === Step 4: Retarget ===
    qpos_array = step_retarget(frames, human_height, fps, args)

    # === Step 5: Playback ===
    if not args.no_playback:
        step_playback(qpos_array, fps, args)
    else:
        print("\n  Skipping playback (--no_playback)")

    elapsed_total = time.perf_counter() - pipeline_start
    print()
    print("=" * 60)
    print(f"Pipeline complete in {elapsed_total:.1f}s")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Real-time Pipeline: Camera → GVHMR → GMR → MuJoCo (live)

Captures camera frames, runs per-frame preprocessing, triggers GVHMR
inference on a sliding window, and retargets to Unitree G1 in real time.

Usage:
    # Basic real-time mode (static camera assumed)
    python scripts/realtime_pipeline.py

    # With pose overlay and comparison view
    python scripts/realtime_pipeline.py --vis-pose --vis-compare

    # Custom window/stride
    python scripts/realtime_pipeline.py --window 90 --stride 15

    # Headless mode (no GUI, print metrics only)
    python scripts/realtime_pipeline.py --headless

    # All visualization
    python scripts/realtime_pipeline.py --vis-all
"""

import argparse
import sys
from pathlib import Path

# Project setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Pipeline: Camera → GVHMR → GMR → MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Camera
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")

    # GVHMR
    parser.add_argument("-s", "--static-cam", action="store_true", default=True,
                        help="Assume static camera (default: True)")
    parser.add_argument("--window", type=int, default=90,
                        help="GVHMR sliding window size in frames (default: 90)")
    parser.add_argument("--stride", type=int, default=15,
                        help="GVHMR inference stride in frames (default: 15)")

    # Smoothing
    parser.add_argument("--ema-alpha", type=float, default=0.3,
                        help="EMA smoothing factor for qpos (default: 0.3)")

    # Display
    parser.add_argument("--headless", action="store_true",
                        help="Headless mode (no GUI)")

    # Visualization flags
    parser.add_argument("--vis-pose", action="store_true",
                        help="Show 2D keypoint overlay")
    parser.add_argument("--vis-dashboard", action="store_true", default=True,
                        help="Show metrics HUD (default: on)")
    parser.add_argument("--vis-compare", action="store_true",
                        help="Show side-by-side human vs robot")
    parser.add_argument("--vis-all", action="store_true",
                        help="Enable all visualization")

    args = parser.parse_args()

    if args.vis_all:
        args.vis_pose = True
        args.vis_dashboard = True
        args.vis_compare = True

    from src.realtime.pipeline import RealtimePipeline

    print()
    print("=" * 60)
    print("  Real-time Pipeline: Camera → GVHMR → GMR → MuJoCo")
    print("=" * 60)
    print(f"  Camera: index={args.camera}, {args.width}x{args.height} @ {args.fps}fps")
    print(f"  GVHMR: window={args.window}, stride={args.stride}, static_cam={args.static_cam}")
    print(f"  Smoothing: EMA alpha={args.ema_alpha}")
    print(f"  Display: {'headless' if args.headless else 'GUI'}")
    print(f"  Vis: pose={args.vis_pose} dashboard={args.vis_dashboard} compare={args.vis_compare}")
    print()
    print("  Press 'q' in the preview window or Ctrl+C to quit.")
    print()

    pipeline = RealtimePipeline(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        window_size=args.window,
        stride=args.stride,
        ema_alpha=args.ema_alpha,
        static_cam=args.static_cam,
        headless=args.headless,
        vis_pose=args.vis_pose,
        vis_dashboard=args.vis_dashboard,
        vis_compare=args.vis_compare,
    )

    with pipeline:
        pipeline.run_display_loop()

    print("\nPipeline stopped.")


if __name__ == "__main__":
    main()

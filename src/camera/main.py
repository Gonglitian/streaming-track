"""CLI entry point for camera capture module.

Usage:
    # List available cameras
    python -m src.camera.main detect

    # Record mode: save to MP4
    python -m src.camera.main record -o output.mp4

    # Stream mode: real-time preview
    python -m src.camera.main stream

    # Custom resolution and fps
    python -m src.camera.main stream --width 1280 --height 720 --fps 30
"""

import argparse
import sys
import time

import cv2

from .detector import CameraDetector
from .recorder import CameraRecorder
from .streamer import CameraStreamer


def cmd_detect(args: argparse.Namespace) -> None:
    cameras = CameraDetector.list_cameras()
    if not cameras:
        print("No cameras detected.")
        return
    print(f"Found {len(cameras)} camera(s):")
    for cam in cameras:
        print(f"  [{cam.index}] {cam.name}  {cam.width}x{cam.height} @ {cam.fps:.0f}fps")


def cmd_record(args: argparse.Namespace) -> None:
    print(f"Recording from camera {args.camera} -> {args.output}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps}fps")
    print("Press Ctrl+C to stop recording.\n")

    recorder = CameraRecorder(
        camera_index=args.camera,
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    recorder.start()

    try:
        while True:
            time.sleep(1.0)
            print(f"\r  FPS: {recorder.current_fps:.1f}", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        recorder.stop()
        print(f"Saved to {args.output}")


def cmd_stream(args: argparse.Namespace) -> None:
    print(f"Streaming from camera {args.camera}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps}fps")
    print("Press 'q' in the preview window to quit.\n")

    streamer = CameraStreamer(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        buffer_size=1,
    )
    streamer.start()

    try:
        while True:
            fd = streamer.buffer.get(timeout=1.0)
            if fd is None:
                continue

            display = fd.frame.copy()
            fps_text = f"FPS: {streamer.current_fps:.1f}"
            cv2.putText(
                display, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
            )
            cv2.imshow("Camera Stream", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()
        cv2.destroyAllWindows()
        print("Stream stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Camera capture module - recording & streaming"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # detect
    sub.add_parser("detect", help="List available cameras")

    # record
    p_rec = sub.add_parser("record", help="Record camera to MP4")
    p_rec.add_argument("-c", "--camera", type=int, default=0, help="Camera index")
    p_rec.add_argument("-o", "--output", default="output.mp4", help="Output file path")
    p_rec.add_argument("--width", type=int, default=640)
    p_rec.add_argument("--height", type=int, default=480)
    p_rec.add_argument("--fps", type=int, default=30)

    # stream
    p_str = sub.add_parser("stream", help="Real-time camera preview")
    p_str.add_argument("-c", "--camera", type=int, default=0, help="Camera index")
    p_str.add_argument("--width", type=int, default=640)
    p_str.add_argument("--height", type=int, default=480)
    p_str.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()

    handlers = {
        "detect": cmd_detect,
        "record": cmd_record,
        "stream": cmd_stream,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()

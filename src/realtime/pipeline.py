"""4-thread real-time GVHMR + GMR pipeline orchestrator.

Architecture:
    Thread 1: CameraStreamer → FrameBuffer (producer)
    Thread 2: Preprocessor → per-frame features → SlidingWindowBuffer
    Thread 3: GVHMR inference (triggered by SlidingWindowBuffer)
    Main thread: GMR retarget + display/render
"""

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GMR_ROOT = PROJECT_ROOT / "third_party" / "GMR"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(GMR_ROOT))

from src.camera.streamer import CameraStreamer
from src.camera.fps_counter import FPSCounter
from src.vis.pose_overlay import draw_keypoints, draw_bbox
from src.vis.dashboard import PipelineMetrics, MetricsDashboard

from .preprocessor import RealtimePreprocessor
from .sliding_window import SlidingWindowBuffer


class RealtimePipeline:
    """Orchestrates the real-time motion capture → retarget pipeline."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        window_size: int = 90,
        stride: int = 15,
        ema_alpha: float = 0.3,
        static_cam: bool = True,
        headless: bool = False,
        vis_pose: bool = False,
        vis_dashboard: bool = True,
        vis_compare: bool = False,
    ):
        self._width = width
        self._height = height
        self._fps = fps
        self._ema_alpha = ema_alpha
        self._static_cam = static_cam
        self._headless = headless
        self._vis_pose = vis_pose
        self._vis_dashboard = vis_dashboard
        self._vis_compare = vis_compare

        # Components
        self._streamer = CameraStreamer(
            camera_index=camera_index,
            width=width, height=height, fps=fps,
            buffer_size=1,
        )
        self._preprocessor = RealtimePreprocessor(
            width=width, height=height,
        )
        self._window = SlidingWindowBuffer(
            window_size=window_size, stride=stride,
        )

        # GMR retargeter (lazy-loaded)
        self._retarget = None
        self._body_model = None

        # Shared state
        self._latest_qpos = None
        self._smoothed_qpos = None
        self._latest_features = None
        self._latest_smpl_params = None
        self._qpos_lock = threading.Lock()

        # Metrics
        self._preprocess_fps = FPSCounter()
        self._inference_fps = FPSCounter()
        self._retarget_fps = FPSCounter()
        self._dashboard = MetricsDashboard() if vis_dashboard else None

        # Control
        self._running = threading.Event()
        self._preprocess_thread = None
        self._inference_thread = None

    def _ensure_retarget(self):
        """Lazy-load GMR retargeter."""
        if self._retarget is not None:
            return

        from general_motion_retargeting import GeneralMotionRetargeting as GMR
        self._retarget = GMR(
            actual_human_height=1.7,
            src_human="smplx",
            tgt_robot="unitree_g1",
            verbose=False,
        )
        print("[Pipeline] GMR retargeter loaded")

    def _ensure_body_model(self):
        """Lazy-load SMPLX body model for bridge conversion."""
        if self._body_model is not None:
            return

        import smplx
        body_models_dir = GMR_ROOT / "assets" / "body_models"
        self._body_model = smplx.create(
            str(body_models_dir), model_type="smplx",
            gender="neutral", num_betas=10,
            use_pca=False, flat_hand_mean=True,
        )
        print("[Pipeline] SMPLX body model loaded")

    def start(self):
        """Start all pipeline threads."""
        self._running.set()

        # Thread 1: Camera
        self._streamer.start()
        print("[Pipeline] Camera streamer started")

        # Thread 2: Preprocessor
        self._preprocess_thread = threading.Thread(
            target=self._preprocess_loop, daemon=True
        )
        self._preprocess_thread.start()

        # Thread 3: GVHMR inference
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._inference_thread.start()

    def stop(self):
        """Stop all threads."""
        self._running.clear()
        if self._preprocess_thread:
            self._preprocess_thread.join(timeout=5.0)
        if self._inference_thread:
            self._inference_thread.join(timeout=10.0)
        self._streamer.stop()

    def _preprocess_loop(self):
        """Thread 2: Grab frames from camera, extract features."""
        while self._running.is_set():
            fd = self._streamer.buffer.get(timeout=0.5)
            if fd is None:
                continue

            features = self._preprocessor.process_frame(fd.frame)
            self._latest_features = features
            self._window.add_features(features)
            self._preprocess_fps.tick()

    def _inference_loop(self):
        """Thread 3: Run GVHMR inference when window is ready."""
        while self._running.is_set():
            if not self._window.should_infer():
                time.sleep(0.01)
                continue

            t0 = time.monotonic()
            result = self._window.run_inference()
            if result is None:
                continue

            self._inference_fps.tick()

            # Bridge: SMPL params → per-joint dicts → GMR retarget
            self._process_inference_result(result)

    def _process_inference_result(self, result: dict):
        """Convert GVHMR output to robot qpos via bridge + GMR."""
        from src.bridge.gvhmr_to_gmr import gvhmr_to_smplx_frames

        smpl_params = result["smpl_params_global"]
        self._latest_smpl_params = smpl_params

        # Convert to the format gvhmr_to_smplx_frames expects
        gvhmr_result = {
            "body_pose": smpl_params["body_pose"],
            "betas": smpl_params["betas"],
            "global_orient": smpl_params["global_orient"],
            "transl": smpl_params["transl"],
        }

        self._ensure_body_model()
        body_models_dir = GMR_ROOT / "assets" / "body_models"

        try:
            frames, fps, human_height = gvhmr_to_smplx_frames(
                gvhmr_result,
                smplx_model_path=str(body_models_dir),
                tgt_fps=30,
            )
        except Exception as e:
            print(f"[Pipeline] Bridge error: {e}")
            return

        # Update human height for GMR
        self._ensure_retarget()
        if hasattr(self._retarget, '_actual_human_height'):
            self._retarget._actual_human_height = human_height

        # Retarget each frame
        for frame_data in frames:
            try:
                qpos = self._retarget.retarget(frame_data)
                self._retarget_fps.tick()

                with self._qpos_lock:
                    self._latest_qpos = qpos.copy()
                    # EMA smoothing
                    if self._smoothed_qpos is None:
                        self._smoothed_qpos = qpos.copy()
                    else:
                        self._smoothed_qpos = (
                            self._ema_alpha * qpos
                            + (1 - self._ema_alpha) * self._smoothed_qpos
                        )
            except Exception as e:
                print(f"[Pipeline] Retarget error: {e}")

    def get_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        try:
            import torch
            gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
        except Exception:
            gpu_mem = 0

        return PipelineMetrics(
            camera_fps=self._streamer.current_fps,
            preprocess_fps=self._preprocess_fps.fps,
            inference_fps=self._inference_fps.fps,
            retarget_fps=self._retarget_fps.fps,
            latency_ms=0,  # TODO: compute end-to-end latency
            gpu_mem_mb=gpu_mem,
            frame_seq=self._window.total_frames,
            window_count=self._window.window_count,
        )

    def run_display_loop(self):
        """Main thread: display camera feed with overlays + robot view."""
        if self._headless:
            self._run_headless()
            return

        comparison_view = None
        if self._vis_compare:
            from src.vis.comparison_view import ComparisonView
            comparison_view = ComparisonView()

        try:
            while self._running.is_set():
                fd = self._streamer.buffer.get(timeout=1.0)
                if fd is None:
                    continue

                display = fd.frame.copy()

                # Pose overlay
                if self._vis_pose and self._latest_features is not None:
                    features = self._latest_features
                    if features.valid:
                        draw_keypoints(display, features.kp2d.numpy())
                        draw_bbox(display, features.bbx_xyxy.numpy())

                # Dashboard overlay
                if self._dashboard is not None:
                    metrics = self.get_metrics()
                    self._dashboard.draw(display, metrics)

                cv2.imshow("Real-time Pipeline", display)

                # Comparison view
                if comparison_view is not None:
                    with self._qpos_lock:
                        qpos = self._smoothed_qpos
                    comp = comparison_view.render(
                        joint_dict=None,  # TODO: pass latest joint dict
                        robot_qpos=qpos,
                    )
                    cv2.imshow("Human vs Robot", comp)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

    def _run_headless(self):
        """Headless mode: just print metrics periodically."""
        try:
            while self._running.is_set():
                time.sleep(2.0)
                m = self.get_metrics()
                print(
                    f"  cam={m.camera_fps:.1f} preproc={m.preprocess_fps:.1f} "
                    f"infer={m.inference_fps:.1f} retarget={m.retarget_fps:.1f} "
                    f"gpu={m.gpu_mem_mb:.0f}MB frame={m.frame_seq} win={m.window_count}"
                )
        except KeyboardInterrupt:
            pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

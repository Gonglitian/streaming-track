"""4-thread real-time GVHMR + GMR pipeline orchestrator.

Architecture:
    Thread 1: CameraStreamer → FrameBuffer (producer)
    Thread 2: Preprocessor → per-frame features → SlidingWindowBuffer
    Thread 3: GVHMR inference (triggered by SlidingWindowBuffer)
    Main thread: GMR retarget + display/render
"""

import collections
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GMR_ROOT = PROJECT_ROOT / "third_party" / "GMR"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(GMR_ROOT))

from src.camera.streamer import CameraStreamer
from src.camera.fps_counter import FPSCounter
from src.vis.pose_overlay import draw_keypoints, draw_bbox
from src.vis.dashboard import PipelineMetrics, MetricsDashboard

from .motion_predictor import MotionPredictor
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
        window_size: int = 60,
        stride: int = 8,
        ema_alpha: float = 0.3,
        static_cam: bool = True,
        headless: bool = False,
        vis_pose: bool = False,
        vis_dashboard: bool = True,
        vis_compare: bool = False,
        debug: bool = False,
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
        self._debug = debug

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
        self._latest_joint_dict = None
        self._qpos_lock = threading.Lock()
        # Queue for smooth frame-by-frame playback (inference thread → display thread)
        self._qpos_queue = collections.deque(maxlen=60)
        # Motion predictor: extrapolates when queue is empty
        self._predictor = MotionPredictor(
            velocity_alpha=0.6, max_predict_dt=0.5, decay_rate=5.0,
        )

        # Metrics
        self._preprocess_fps = FPSCounter()
        self._inference_fps = FPSCounter()
        self._retarget_fps = FPSCounter()
        self._dashboard = MetricsDashboard() if vis_dashboard else None

        # Latency tracking (timestamp of the frame that triggered inference)
        self._infer_trigger_time = 0.0
        self._latest_latency_ms = 0.0

        # Debug timing accumulators
        self._debug_timings = {
            "preprocess": [],
            "gvhmr_infer": [],
            "bridge": [],
            "retarget": [],
            "total_infer_to_qpos": [],
        }
        self._debug_display_stats = {"consumed": 0, "predicted": 0, "idle": 0}
        self._debug_last_print = time.monotonic()

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
            gender="neutral", num_betas=16,
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

        # Print final debug summary
        if self._debug:
            self._print_debug_summary()

    def _preprocess_loop(self):
        """Thread 2: Grab frames from camera, extract features."""
        while self._running.is_set():
            fd = self._streamer.buffer.get(timeout=0.5)
            if fd is None:
                continue

            t0 = time.monotonic()
            features = self._preprocessor.process_frame(fd.frame)
            dt = time.monotonic() - t0

            self._latest_features = features
            self._window.add_features(features)
            self._preprocess_fps.tick()

            if self._debug:
                self._debug_timings["preprocess"].append(dt * 1000)

    def _inference_loop(self):
        """Thread 3: Run GVHMR inference when window is ready."""
        while self._running.is_set():
            if not self._window.should_infer():
                time.sleep(0.01)
                continue

            t_total = time.monotonic()
            self._infer_trigger_time = t_total

            # GVHMR inference
            t0 = time.monotonic()
            result = self._window.run_inference()
            dt_infer = time.monotonic() - t0

            if result is None:
                continue

            self._inference_fps.tick()

            if self._debug:
                self._debug_timings["gvhmr_infer"].append(dt_infer * 1000)

            # Bridge: SMPL params → per-joint dicts → GMR retarget
            self._process_inference_result(result)

            dt_total = time.monotonic() - t_total
            if self._debug:
                self._debug_timings["total_infer_to_qpos"].append(dt_total * 1000)

            # Update latency
            self._latest_latency_ms = dt_total * 1000

    def _process_inference_result(self, result: dict):
        """Convert GVHMR output to robot qpos via bridge + GMR."""
        from src.bridge.gvhmr_to_gmr import gvhmr_to_smplx_frames

        smpl_params = result["smpl_params_global"]
        self._latest_smpl_params = smpl_params

        # Convert to the format gvhmr_to_smplx_frames expects (numpy)
        def _to_np(t):
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().numpy()
            return t

        gvhmr_result = {
            "body_pose": _to_np(smpl_params["body_pose"]),
            "betas": _to_np(smpl_params["betas"]),
            "global_orient": _to_np(smpl_params["global_orient"]),
            "transl": _to_np(smpl_params["transl"]),
        }

        self._ensure_body_model()
        body_models_dir = GMR_ROOT / "assets" / "body_models"

        t0 = time.monotonic()
        try:
            frames, fps, human_height = gvhmr_to_smplx_frames(
                gvhmr_result,
                smplx_model_path=str(body_models_dir),
                tgt_fps=30,
                body_model=self._body_model,
            )
        except Exception as e:
            print(f"[Pipeline] Bridge error: {e}")
            return
        dt_bridge = time.monotonic() - t0

        if self._debug:
            self._debug_timings["bridge"].append(dt_bridge * 1000)

        # Update human height for GMR
        self._ensure_retarget()
        if hasattr(self._retarget, '_actual_human_height'):
            self._retarget._actual_human_height = human_height

        # Retarget each frame and queue for smooth playback
        t0 = time.monotonic()
        for frame_data in frames:
            try:
                qpos = self._retarget.retarget(frame_data)
                self._retarget_fps.tick()

                with self._qpos_lock:
                    self._qpos_queue.append((qpos.copy(), frame_data))
            except Exception as e:
                print(f"[Pipeline] Retarget error: {e}")
        dt_retarget = time.monotonic() - t0

        if self._debug:
            self._debug_timings["retarget"].append(dt_retarget * 1000)

    def _consume_one_frame(self) -> bool:
        """Pop one retargeted frame from the queue and apply EMA smoothing.

        Called by the display loop once per camera frame to achieve smooth playback
        instead of pose-teleporting when inference produces a batch of frames.

        Returns True if a frame was consumed, False if queue was empty.
        """
        with self._qpos_lock:
            if not self._qpos_queue:
                return False
            qpos, joint_dict = self._qpos_queue.popleft()
            self._latest_qpos = qpos
            self._latest_joint_dict = joint_dict
            if self._smoothed_qpos is None:
                self._smoothed_qpos = qpos.copy()
            else:
                self._smoothed_qpos = (
                    self._ema_alpha * qpos
                    + (1 - self._ema_alpha) * self._smoothed_qpos
                )
        # Update predictor with observed pose (outside lock)
        self._predictor.update(self._smoothed_qpos)
        return True

    def _drain_queue(self):
        """Drain all queued frames (for headless mode)."""
        while self._consume_one_frame():
            pass

    def _print_debug_timings(self):
        """Print debug timing stats periodically."""
        now = time.monotonic()
        if now - self._debug_last_print < 5.0:
            return
        self._debug_last_print = now

        print()
        print("=" * 60)
        print("[DEBUG] Pipeline Timing (ms) — last 5s")
        print("-" * 60)
        for name, times in self._debug_timings.items():
            if not times:
                print(f"  {name:25s}: (no data)")
                continue
            arr = np.array(times)
            print(
                f"  {name:25s}: "
                f"avg={arr.mean():7.1f}  "
                f"min={arr.min():7.1f}  "
                f"max={arr.max():7.1f}  "
                f"n={len(arr)}"
            )
            times.clear()
        print(f"  {'end-to-end latency':25s}: {self._latest_latency_ms:7.1f}")
        ds = self._debug_display_stats
        total = ds["consumed"] + ds["predicted"] + ds["idle"]
        if total > 0:
            print(f"  {'display frames':25s}: "
                  f"real={ds['consumed']}  predicted={ds['predicted']}  idle={ds['idle']}  "
                  f"({ds['predicted']/total*100:.0f}% predicted)")
        self._debug_display_stats = {"consumed": 0, "predicted": 0, "idle": 0}
        print("=" * 60)

    def _print_debug_summary(self):
        """Print final debug summary."""
        print()
        print("=" * 60)
        print("[DEBUG] Final Pipeline Summary")
        print("-" * 60)
        for name, times in self._debug_timings.items():
            if not times:
                continue
            arr = np.array(times)
            print(
                f"  {name:25s}: "
                f"avg={arr.mean():7.1f}ms  "
                f"p50={np.percentile(arr, 50):7.1f}ms  "
                f"p95={np.percentile(arr, 95):7.1f}ms  "
                f"n={len(arr)}"
            )
        print("=" * 60)

    def get_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        try:
            gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
        except Exception:
            gpu_mem = 0

        return PipelineMetrics(
            camera_fps=self._streamer.current_fps,
            preprocess_fps=self._preprocess_fps.fps,
            inference_fps=self._inference_fps.fps,
            retarget_fps=self._retarget_fps.fps,
            latency_ms=self._latest_latency_ms,
            gpu_mem_mb=gpu_mem,
            frame_seq=self._window.total_frames,
            window_count=self._window.window_count,
        )

    def run_display_loop(self):
        """Main thread: display camera feed with overlays + robot view."""
        if self._headless:
            self._run_headless()
            return

        import mujoco
        import mujoco.viewer

        # Native MuJoCo viewer for G1 robot (interactive 3D window)
        mj_viewer = None
        mj_model = None
        mj_data = None
        skeleton_viewer = None

        if self._vis_compare:
            from src.vis.skeleton_viewer import SkeletonViewer

            g1_xml = GMR_ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
            mj_model = mujoco.MjModel.from_xml_path(str(g1_xml))
            mj_data = mujoco.MjData(mj_model)
            mj_viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
            print("[Pipeline] MuJoCo G1 viewer launched")

            skeleton_viewer = SkeletonViewer(width=480, height=480)

        try:
            while self._running.is_set():
                # Check if MuJoCo viewer was closed
                if mj_viewer is not None and not mj_viewer.is_running():
                    break

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

                # Consume one queued frame, or predict if queue is empty
                consumed = self._consume_one_frame()

                if consumed:
                    # Fresh data from queue
                    display_qpos = self._smoothed_qpos.copy()
                    if self._debug:
                        self._debug_display_stats["consumed"] += 1
                else:
                    # No new data — extrapolate with motion predictor
                    predicted = self._predictor.predict()
                    if predicted is not None:
                        display_qpos = predicted
                        if self._debug:
                            self._debug_display_stats["predicted"] += 1
                    elif self._smoothed_qpos is not None:
                        display_qpos = self._smoothed_qpos.copy()
                        if self._debug:
                            self._debug_display_stats["idle"] += 1
                    else:
                        display_qpos = None
                        if self._debug:
                            self._debug_display_stats["idle"] += 1

                joint_dict = self._latest_joint_dict

                # G1 robot: native MuJoCo viewer (interactive 3D)
                if mj_viewer is not None and display_qpos is not None:
                    mj_data.qpos[:] = display_qpos
                    mujoco.mj_forward(mj_model, mj_data)
                    mj_viewer.sync()

                # Human skeleton: offscreen render → OpenCV window
                if skeleton_viewer is not None and joint_dict is not None:
                    skel_img = skeleton_viewer.render_from_dict(joint_dict)
                    skel_bgr = cv2.cvtColor(skel_img, cv2.COLOR_RGB2BGR)
                    cv2.putText(skel_bgr, "Human (SMPLX)", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                                cv2.LINE_AA)
                    cv2.imshow("Human Skeleton", skel_bgr)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # Debug timing print
                if self._debug:
                    self._print_debug_timings()

        except KeyboardInterrupt:
            pass
        finally:
            if mj_viewer is not None:
                mj_viewer.close()
            cv2.destroyAllWindows()

    def _run_headless(self):
        """Headless mode: just print metrics periodically."""
        try:
            while self._running.is_set():
                time.sleep(2.0)
                self._drain_queue()  # consume all queued frames
                m = self.get_metrics()
                print(
                    f"  cam={m.camera_fps:.1f} preproc={m.preprocess_fps:.1f} "
                    f"infer={m.inference_fps:.1f} retarget={m.retarget_fps:.1f} "
                    f"latency={m.latency_ms:.0f}ms "
                    f"gpu={m.gpu_mem_mb:.0f}MB frame={m.frame_seq} win={m.window_count}"
                )

                if self._debug:
                    self._print_debug_timings()

        except KeyboardInterrupt:
            pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

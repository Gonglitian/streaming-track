"""Recording mode: capture USB camera to MP4 (H.264)."""

import threading
import time
from pathlib import Path

import cv2

from .detector import CameraDetector
from .fps_counter import FPSCounter


class CameraRecorder:
    """Record camera feed to an MP4 file.

    Usage::

        recorder = CameraRecorder(camera_index=0, output_path="output.mp4")
        recorder.start()   # begins recording
        # ... do other work or wait for user input ...
        recorder.stop()    # stops and saves
    """

    def __init__(
        self,
        camera_index: int = 0,
        output_path: str = "output.mp4",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.camera_index = camera_index
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._fps_counter = FPSCounter()

    @property
    def current_fps(self) -> float:
        return self._fps_counter.fps

    def start(self) -> None:
        """Start recording in a background thread."""
        if self._running.is_set():
            return

        self._cap = CameraDetector.open_camera(
            self.camera_index, self.width, self.height, self.fps
        )

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, (actual_w, actual_h)
        )

        self._running.set()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop recording and release resources."""
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._release()

    def _capture_loop(self) -> None:
        frame_interval = 1.0 / self.fps
        while self._running.is_set():
            t0 = time.monotonic()
            ret, frame = self._cap.read()
            if not ret:
                break
            self._writer.write(frame)
            self._fps_counter.tick()

            elapsed = time.monotonic() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _release(self) -> None:
        if self._writer:
            self._writer.release()
            self._writer = None
        if self._cap:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

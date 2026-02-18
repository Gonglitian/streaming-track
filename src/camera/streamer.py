"""Streaming mode: camera frames into a thread-safe queue (producer-consumer)."""

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import CameraDetector
from .fps_counter import FPSCounter


@dataclass
class FrameData:
    """A single captured frame with metadata."""

    frame: np.ndarray
    timestamp: float = field(default_factory=time.monotonic)
    seq: int = 0


class FrameBuffer:
    """Thread-safe frame buffer that keeps only the latest frame (drop strategy).

    Consumers call ``get()`` to retrieve the most recent frame.
    If ``max_size`` > 1 a short history is kept, but old frames are dropped
    when the buffer is full (only-keep-newest policy).
    """

    def __init__(self, max_size: int = 1):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._max_size = max(1, max_size)
        self._buffer: list[FrameData] = []
        self._seq = 0

    def put(self, frame: np.ndarray) -> None:
        with self._condition:
            self._seq += 1
            fd = FrameData(frame=frame, seq=self._seq)
            if len(self._buffer) >= self._max_size:
                self._buffer.pop(0)
            self._buffer.append(fd)
            self._condition.notify_all()

    def get(self, timeout: float | None = None) -> FrameData | None:
        """Return the newest frame, blocking up to *timeout* seconds if empty."""
        with self._condition:
            if not self._buffer:
                self._condition.wait(timeout=timeout)
            if not self._buffer:
                return None
            return self._buffer[-1]

    @property
    def latest(self) -> FrameData | None:
        """Non-blocking peek at the newest frame."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None


class CameraStreamer:
    """Capture camera frames into a shared FrameBuffer.

    Usage::

        streamer = CameraStreamer(camera_index=0)
        streamer.start()

        while True:
            fd = streamer.buffer.get(timeout=1.0)
            if fd is not None:
                cv2.imshow("preview", fd.frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        streamer.stop()
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 1,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self.buffer = FrameBuffer(max_size=buffer_size)
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._fps_counter = FPSCounter()

    @property
    def current_fps(self) -> float:
        return self._fps_counter.fps

    def start(self) -> None:
        if self._running.is_set():
            return

        self._cap = CameraDetector.open_camera(
            self.camera_index, self.width, self.height, self.fps
        )
        self._running.set()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None

    def _capture_loop(self) -> None:
        frame_interval = 1.0 / self.fps
        while self._running.is_set():
            t0 = time.monotonic()
            ret, frame = self._cap.read()
            if not ret:
                break
            self.buffer.put(frame)
            self._fps_counter.tick()

            elapsed = time.monotonic() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

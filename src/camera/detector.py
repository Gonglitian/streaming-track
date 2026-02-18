"""Camera detection utility for enumerating available capture devices."""

import contextlib
import os
from dataclasses import dataclass

import cv2


@dataclass
class CameraInfo:
    index: int
    name: str
    width: int
    height: int
    fps: float


@contextlib.contextmanager
def _suppress_stderr():
    """Temporarily redirect stderr to /dev/null to suppress OpenCV C++ warnings."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


class CameraDetector:
    """Detect and list available cameras."""

    @staticmethod
    def list_cameras(max_index: int = 8) -> list[CameraInfo]:
        """Probe camera indices 0..max_index-1 and return info for those that open."""
        cameras: list[CameraInfo] = []
        with _suppress_stderr():
            for idx in range(max_index):
                cap = cv2.VideoCapture(idx)
                if not cap.isOpened():
                    cap.release()
                    continue
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
                cameras.append(
                    CameraInfo(
                        index=idx,
                        name=f"Camera {idx}",
                        width=w,
                        height=h,
                        fps=fps,
                    )
                )
        return cameras

    @staticmethod
    def open_camera(
        index: int = 0,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ) -> cv2.VideoCapture:
        """Open a camera and optionally negotiate resolution/fps.

        Raises RuntimeError if the camera cannot be opened.
        """
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")

        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)

        return cap

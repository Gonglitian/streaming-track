"""HUD metrics overlay for pipeline monitoring (pure OpenCV)."""

from dataclasses import dataclass, field
import time

import cv2
import numpy as np


@dataclass
class PipelineMetrics:
    """Metrics collected from the real-time pipeline."""
    camera_fps: float = 0.0
    preprocess_fps: float = 0.0
    inference_fps: float = 0.0
    retarget_fps: float = 0.0
    latency_ms: float = 0.0
    gpu_mem_mb: float = 0.0
    frame_seq: int = 0
    window_count: int = 0


class MetricsDashboard:
    """Semi-transparent HUD overlay showing pipeline metrics."""

    def __init__(self, position: str = "top-right", alpha: float = 0.7):
        """
        Args:
            position: One of 'top-left', 'top-right', 'bottom-left', 'bottom-right'.
            alpha: Background transparency (0=transparent, 1=opaque).
        """
        self._position = position
        self._alpha = alpha
        self._last_update = time.monotonic()

    def draw(self, frame: np.ndarray, metrics: PipelineMetrics) -> np.ndarray:
        """Draw metrics overlay on frame.

        Args:
            frame: BGR image (H, W, 3).
            metrics: Current pipeline metrics.

        Returns:
            Frame with HUD drawn (modified in-place).
        """
        lines = [
            f"Camera:     {metrics.camera_fps:5.1f} FPS",
            f"Preprocess: {metrics.preprocess_fps:5.1f} FPS",
            f"GVHMR:      {metrics.inference_fps:5.1f} FPS",
            f"Retarget:   {metrics.retarget_fps:5.1f} FPS",
            f"Latency:    {metrics.latency_ms:5.0f} ms",
            f"GPU Mem:    {metrics.gpu_mem_mb:5.0f} MB",
            f"Frame: {metrics.frame_seq}  Win: {metrics.window_count}",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        line_height = 20
        padding = 8

        # Measure text block size
        max_width = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            max_width = max(max_width, tw)

        panel_w = max_width + padding * 2
        panel_h = len(lines) * line_height + padding * 2
        h, w = frame.shape[:2]

        # Position
        if "right" in self._position:
            x0 = w - panel_w - 10
        else:
            x0 = 10
        if "bottom" in self._position:
            y0 = h - panel_h - 10
        else:
            y0 = 10

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self._alpha, frame, 1 - self._alpha, 0, frame)

        # Draw border
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (100, 100, 100), 1)

        # Draw text
        for i, line in enumerate(lines):
            y = y0 + padding + (i + 1) * line_height - 4
            cv2.putText(frame, line, (x0 + padding, y), font, font_scale,
                        (0, 255, 0), font_thickness, cv2.LINE_AA)

        return frame

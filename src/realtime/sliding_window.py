"""Thread-safe sliding window buffer with GVHMR inference trigger.

Accumulates per-frame features and triggers GVHMR inference every `stride` frames.
"""

import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GVHMR_DIR = PROJECT_ROOT / "third_party" / "GVHMR"
sys.path.insert(0, str(GVHMR_DIR))

from .preprocessor import FrameFeatures


class SlidingWindowBuffer:
    """Accumulates per-frame features and triggers GVHMR inference.

    Uses a sliding window of `window_size` frames, triggering inference
    every `stride` frames on the latest window.
    """

    def __init__(self, window_size: int = 90, stride: int = 15):
        self._window_size = window_size
        self._stride = stride

        self._lock = threading.Lock()
        self._features: list[FrameFeatures] = []
        self._frames_since_infer = 0
        self._total_frames = 0

        # GVHMR model (lazy-loaded)
        self._model = None
        self._infer_count = 0

    @property
    def window_count(self) -> int:
        return self._infer_count

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def add_features(self, features: FrameFeatures) -> bool:
        """Add preprocessed features for one frame.

        Returns True if inference should be triggered.
        """
        with self._lock:
            if features.valid:
                self._features.append(features)
                # Keep only latest window_size frames
                if len(self._features) > self._window_size:
                    self._features = self._features[-self._window_size:]
            self._total_frames += 1
            self._frames_since_infer += 1

            return self.should_infer()

    def should_infer(self) -> bool:
        """Check if we have enough frames and stride elapsed."""
        return (
            len(self._features) >= self._window_size
            and self._frames_since_infer >= self._stride
        )

    def _ensure_model(self):
        """Lazy-load GVHMR model on first inference."""
        if self._model is not None:
            return

        import contextlib

        @contextlib.contextmanager
        def _chdir(path):
            old = os.getcwd()
            os.chdir(path)
            try:
                yield
            finally:
                os.chdir(old)

        with _chdir(GVHMR_DIR):
            from hydra import initialize_config_module, compose
            from hmr4d.configs import register_store_gvhmr
            from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
            import hydra

            with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
                register_store_gvhmr()
                cfg = compose(config_name="demo", overrides=[
                    "video_name=realtime",
                    "static_cam=True",
                    "verbose=False",
                    "use_dpvo=False",
                ])

            self._model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
            self._model.load_pretrained_model(cfg.ckpt_path)
            self._model = self._model.eval().cuda()
            print("[SlidingWindow] GVHMR model loaded")

    @torch.no_grad()
    def run_inference(self) -> dict | None:
        """Run GVHMR inference on the current window.

        Returns:
            dict with smpl_params_global for the window, or None if not enough frames.
        """
        with self._lock:
            if not self.should_infer():
                return None
            # Snapshot the current window
            window = list(self._features[-self._window_size:])
            self._frames_since_infer = 0

        self._ensure_model()

        F = len(window)

        # Stack per-frame features into batch tensors
        bbx_xys = torch.stack([f.bbx_xys for f in window])        # (F, 3)
        kp2d = torch.stack([f.kp2d for f in window])              # (F, 17, 3)
        K_fullimg = torch.stack([f.K_fullimg for f in window])    # (F, 3, 3)
        f_imgseq = torch.stack([f.vit_features for f in window])  # (F, 1024)

        # Static camera: cam_angvel is zero
        cam_angvel = torch.zeros(F, 6)

        # Prepare data dict for GVHMR
        from hmr4d.utils.geo_transform import compute_cam_angvel

        # For static cam, R_w2c is identity
        R_w2c = torch.eye(3).unsqueeze(0).repeat(F, 1, 1)
        cam_angvel = compute_cam_angvel(R_w2c)

        data = {
            "length": torch.tensor(F),
            "bbx_xys": bbx_xys,
            "kp2d": kp2d,
            "K_fullimg": K_fullimg,
            "cam_angvel": cam_angvel,
            "f_imgseq": f_imgseq,
        }

        # Run GVHMR inference with FP16
        with torch.cuda.amp.autocast():
            pred = self._model.predict(data, static_cam=True)

        from hmr4d.utils.net_utils import detach_to_cpu
        pred = detach_to_cpu(pred)

        self._infer_count += 1

        # Return only the latest stride-worth of frames
        result = {
            "smpl_params_global": {
                k: v[-self._stride:] if isinstance(v, torch.Tensor) else v
                for k, v in pred["smpl_params_global"].items()
            },
            "smpl_params_global_full": pred["smpl_params_global"],
            "window_size": F,
            "stride": self._stride,
            "infer_count": self._infer_count,
        }
        return result

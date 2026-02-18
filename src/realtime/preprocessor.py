"""Per-frame feature extraction for real-time GVHMR pipeline.

Adapts GVHMR's batch extractors (YOLO, ViTPose, HMR2) to single-frame mode.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GVHMR_DIR = PROJECT_ROOT / "third_party" / "GVHMR"
sys.path.insert(0, str(GVHMR_DIR))


@dataclass
class FrameFeatures:
    """Preprocessed features for a single frame."""
    bbx_xys: torch.Tensor       # (3,) bounding box [cx, cy, scale]
    bbx_xyxy: torch.Tensor      # (4,) bounding box [x1, y1, x2, y2]
    kp2d: torch.Tensor          # (17, 3) COCO keypoints [x, y, conf]
    vit_features: torch.Tensor  # (1024,) HMR2 visual features
    K_fullimg: torch.Tensor     # (3, 3) camera intrinsics
    valid: bool = True


class RealtimePreprocessor:
    """Per-frame feature extraction using GVHMR's models.

    Loads YOLOv8s, ViTPose, and HMR2 once, then processes individual frames.
    Uses smaller models where possible to fit within 8GB VRAM.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        device: str = "cuda",
        yolo_model: str = "yolov8s.pt",
    ):
        self._width = width
        self._height = height
        self._device = torch.device(device)

        # Lazy-load models
        self._yolo = None
        self._vitpose = None
        self._hmr2 = None
        self._yolo_model_name = yolo_model

        # Pre-compute camera intrinsics (constant for fixed resolution)
        from hmr4d.utils.geo.hmr_cam import estimate_K
        self._K_fullimg = estimate_K(width, height)  # (1, 3, 3)

    def _ensure_models(self):
        """Lazy-load all models on first use."""
        if self._yolo is None:
            self._load_yolo()
        if self._vitpose is None:
            self._load_vitpose()
        if self._hmr2 is None:
            self._load_hmr2()

    def _load_yolo(self):
        """Load YOLOv8 for person detection."""
        from ultralytics import YOLO
        # Use YOLOv8s (smaller) instead of v8x for real-time
        ckpt = GVHMR_DIR / "inputs" / "checkpoints" / "yolo" / self._yolo_model_name
        if not ckpt.exists():
            # Fall back to ultralytics auto-download
            self._yolo = YOLO(self._yolo_model_name)
        else:
            self._yolo = YOLO(str(ckpt))
        print(f"[Preproc] YOLO loaded: {self._yolo_model_name}")

    def _load_vitpose(self):
        """Load ViTPose for 2D keypoint estimation."""
        from hmr4d.utils.preproc.vitpose import VitPoseExtractor
        self._vitpose = VitPoseExtractor(tqdm_leave=False)
        print("[Preproc] ViTPose loaded")

    def _load_hmr2(self):
        """Load HMR2 for visual feature extraction."""
        from hmr4d.utils.preproc.vitfeat_extractor import Extractor
        self._hmr2 = Extractor(tqdm_leave=False)
        print("[Preproc] HMR2 loaded")

    @torch.no_grad()
    def process_frame(self, frame: np.ndarray) -> FrameFeatures:
        """Extract features from a single BGR frame.

        Args:
            frame: BGR image (H, W, 3) uint8.

        Returns:
            FrameFeatures with all preprocessed data.
        """
        self._ensure_models()

        h, w = frame.shape[:2]

        # 1. YOLO person detection
        results = self._yolo(frame, classes=[0], conf=0.5, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return FrameFeatures(
                bbx_xys=torch.zeros(3),
                bbx_xyxy=torch.zeros(4),
                kp2d=torch.zeros(17, 3),
                vit_features=torch.zeros(1024),
                K_fullimg=self._K_fullimg.squeeze(0),
                valid=False,
            )

        # Pick the largest person detection
        areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
        best_idx = areas.argmax().item()
        bbx_xyxy = boxes.xyxy[best_idx].cpu().float()  # (4,)

        # Convert to bbx_xys format [cx, cy, scale]
        from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
        bbx_xys = get_bbx_xys_from_xyxy(
            bbx_xyxy.unsqueeze(0), base_enlarge=1.2
        ).float().squeeze(0)  # (3,)

        # 2. ViTPose: extract 2D keypoints
        # Wrap single frame as (1, 3) for bbx_xys input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
        kp2d = self._vitpose.extract(frame_tensor, bbx_xys.unsqueeze(0))  # (1, 17, 3)
        kp2d = kp2d.squeeze(0)  # (17, 3)

        # 3. HMR2: extract visual features
        vit_features = self._hmr2.extract_video_features(
            frame_tensor, bbx_xys.unsqueeze(0)
        )  # (1, 1024)
        vit_features = vit_features.squeeze(0)  # (1024,)

        return FrameFeatures(
            bbx_xys=bbx_xys,
            bbx_xyxy=bbx_xyxy,
            kp2d=kp2d,
            vit_features=vit_features,
            K_fullimg=self._K_fullimg.squeeze(0),
            valid=True,
        )

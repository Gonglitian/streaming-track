"""2D skeleton overlay on camera frames (pure OpenCV)."""

import cv2
import numpy as np

# COCO-17 skeleton connections: (joint_a, joint_b)
COCO_SKELETON = [
    (0, 1), (0, 2),      # nose → eyes
    (1, 3), (2, 4),      # eyes → ears
    (5, 6),               # left shoulder → right shoulder
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 11), (6, 12),    # shoulders → hips
    (11, 12),             # left hip → right hip
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Color coding: left=blue, right=red, center=green
_BONE_COLORS = {
    (0, 1): (0, 255, 0), (0, 2): (0, 255, 0),
    (1, 3): (255, 100, 0), (2, 4): (0, 100, 255),
    (5, 6): (0, 255, 0),
    (5, 7): (255, 100, 0), (7, 9): (255, 100, 0),
    (6, 8): (0, 100, 255), (8, 10): (0, 100, 255),
    (5, 11): (255, 100, 0), (6, 12): (0, 100, 255),
    (11, 12): (0, 255, 0),
    (11, 13): (255, 100, 0), (13, 15): (255, 100, 0),
    (12, 14): (0, 100, 255), (14, 16): (0, 100, 255),
}

JOINT_COLOR = (0, 255, 255)  # cyan


def draw_keypoints(
    frame: np.ndarray,
    kp2d: np.ndarray,
    confidence_threshold: float = 0.3,
    radius: int = 4,
    thickness: int = 2,
) -> np.ndarray:
    """Draw 2D keypoints and skeleton on a frame.

    Args:
        frame: BGR image (H, W, 3).
        kp2d: Keypoints array, shape (17, 2) or (17, 3) where last dim is confidence.
        confidence_threshold: Minimum confidence to draw a joint.
        radius: Joint circle radius.
        thickness: Bone line thickness.

    Returns:
        Frame with skeleton drawn (modified in-place).
    """
    n_joints = kp2d.shape[0]
    has_conf = kp2d.shape[1] >= 3

    # Draw bones first (under joints)
    for a, b in COCO_SKELETON:
        if a >= n_joints or b >= n_joints:
            continue
        if has_conf and (kp2d[a, 2] < confidence_threshold or kp2d[b, 2] < confidence_threshold):
            continue
        pt_a = (int(kp2d[a, 0]), int(kp2d[a, 1]))
        pt_b = (int(kp2d[b, 0]), int(kp2d[b, 1]))
        color = _BONE_COLORS.get((a, b), (0, 255, 0))
        cv2.line(frame, pt_a, pt_b, color, thickness, cv2.LINE_AA)

    # Draw joints
    for i in range(min(n_joints, 17)):
        if has_conf and kp2d[i, 2] < confidence_threshold:
            continue
        pt = (int(kp2d[i, 0]), int(kp2d[i, 1]))
        cv2.circle(frame, pt, radius, JOINT_COLOR, -1, cv2.LINE_AA)

    return frame


def draw_bbox(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    label: str | None = None,
) -> np.ndarray:
    """Draw bounding box on a frame.

    Args:
        frame: BGR image (H, W, 3).
        bbox_xyxy: Bounding box as [x1, y1, x2, y2].
        color: BGR color tuple.
        thickness: Line thickness.
        label: Optional text label above the box.

    Returns:
        Frame with bbox drawn (modified in-place).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy[:4]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    return frame

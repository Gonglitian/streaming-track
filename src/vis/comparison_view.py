"""Side-by-side human skeleton vs robot visualization."""

import numpy as np
import cv2

from .skeleton_viewer import SkeletonViewer


class ComparisonView:
    """Render side-by-side: human SMPL skeleton (left) + G1 robot (right)."""

    def __init__(self, width: int = 480, height: int = 480):
        self._width = width
        self._height = height
        self._skeleton_viewer = SkeletonViewer(width=width, height=height)
        self._robot_renderer = None  # lazy init

    def _get_robot_renderer(self):
        if self._robot_renderer is None:
            from src.bridge.mujoco_playback import MujocoPlayback
            self._robot_renderer = MujocoPlayback(headless=True)
        return self._robot_renderer

    def render(
        self,
        joint_dict: dict | None = None,
        robot_qpos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render side-by-side comparison.

        Args:
            joint_dict: {joint_name: (pos_3d, quat_wxyz)} for human skeleton.
            robot_qpos: (36,) qpos array for G1 robot.

        Returns:
            BGR image (H, W*2, 3).
        """
        # Left: human skeleton
        if joint_dict is not None:
            human_img = self._skeleton_viewer.render_from_dict(joint_dict)
            human_img = cv2.cvtColor(human_img, cv2.COLOR_RGB2BGR)
        else:
            human_img = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            cv2.putText(human_img, "No human data", (10, self._height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1)

        # Right: robot
        if robot_qpos is not None:
            renderer = self._get_robot_renderer()
            robot_img = renderer.render_frame(robot_qpos)
            if robot_img.shape[:2] != (self._height, self._width):
                robot_img = cv2.resize(robot_img, (self._width, self._height))
            if len(robot_img.shape) == 3 and robot_img.shape[2] == 3:
                # Already BGR from MuJoCo renderer
                pass
        else:
            robot_img = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            cv2.putText(robot_img, "No robot data", (10, self._height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1)

        # Add labels
        cv2.putText(human_img, "Human (SMPLX)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(robot_img, "Robot (G1)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        # Divider line
        combined = np.hstack([human_img, robot_img])
        cv2.line(combined, (self._width, 0), (self._width, self._height), (255, 255, 255), 1)

        return combined

"""
MuJoCo G1 robot playback with headless rendering and optional video export.

Provides both interactive (viewer) and headless (offscreen) playback modes,
independent of GMR's RobotMotionViewer for cases where we want simpler control.
"""

import time
import pathlib

import numpy as np
import mujoco as mj

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
GMR_ROOT = PROJECT_ROOT / "third_party" / "GMR"
G1_XML = GMR_ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"


class MujocoPlayback:
    """Play back retargeted robot motion in MuJoCo.

    Supports three modes:
    - headless=True: Offscreen rendering, optionally saves video
    - headless=False: Interactive MuJoCo viewer window
    """

    def __init__(
        self,
        xml_path: str = None,
        width: int = 1280,
        height: int = 720,
        headless: bool = False,
    ):
        self.xml_path = str(xml_path or G1_XML)
        self.width = width
        self.height = height
        self.headless = headless

        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.data = mj.MjData(self.model)
        self._renderer = None
        self._viewer = None

    def play(
        self,
        qpos_sequence: np.ndarray,
        fps: float = 30.0,
        video_path: str = None,
        loop: bool = False,
    ) -> None:
        """Play a qpos sequence.

        Args:
            qpos_sequence: (N, 36) array — [root_pos(3) + root_quat_wxyz(4) + dof(29)]
            fps: Playback framerate
            video_path: If set, save rendered video to this path (headless only)
            loop: If True, loop playback until interrupted
        """
        if self.headless:
            self._play_headless(qpos_sequence, fps, video_path, loop)
        else:
            self._play_viewer(qpos_sequence, fps, loop)

    def _play_headless(
        self,
        qpos_seq: np.ndarray,
        fps: float,
        video_path: str = None,
        loop: bool = False,
    ) -> None:
        renderer = mj.Renderer(self.model, height=self.height, width=self.width)
        writer = None

        if video_path:
            import imageio
            pathlib.Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(video_path, fps=fps)

        n_frames = len(qpos_seq)
        frame_idx = 0
        try:
            while True:
                self.data.qpos[:] = qpos_seq[frame_idx]
                mj.mj_forward(self.model, self.data)
                renderer.update_scene(self.data)
                pixels = renderer.render()

                if writer is not None:
                    writer.append_data(pixels)

                frame_idx += 1
                if frame_idx >= n_frames:
                    if loop:
                        frame_idx = 0
                    else:
                        break
        finally:
            renderer.close()
            if writer is not None:
                writer.close()

    def _play_viewer(
        self,
        qpos_seq: np.ndarray,
        fps: float,
        loop: bool = False,
    ) -> None:
        import mujoco.viewer as mjv

        viewer = mjv.launch_passive(
            model=self.model, data=self.data,
            show_left_ui=False, show_right_ui=False,
        )
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -10

        n_frames = len(qpos_seq)
        frame_interval = 1.0 / fps
        frame_idx = 0

        try:
            while viewer.is_running():
                t0 = time.monotonic()

                self.data.qpos[:] = qpos_seq[frame_idx]
                mj.mj_forward(self.model, self.data)

                # Follow the robot pelvis
                pelvis_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "pelvis")
                if pelvis_id > 0:
                    viewer.cam.lookat[:] = self.data.xpos[pelvis_id]

                viewer.sync()

                frame_idx += 1
                if frame_idx >= n_frames:
                    if loop:
                        frame_idx = 0
                    else:
                        break

                elapsed = time.monotonic() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            viewer.close()

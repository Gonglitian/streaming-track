"""3D joint visualization using MuJoCo offscreen renderer."""

import numpy as np
import mujoco


# SMPLX 22-joint skeleton connections (parent → child)
SMPLX_BONES = [
    (0, 1), (0, 2), (0, 3),    # pelvis → left_hip, right_hip, spine1
    (1, 4), (2, 5), (3, 6),    # → left_knee, right_knee, spine2
    (4, 7), (5, 8), (6, 9),    # → left_ankle, right_ankle, spine3
    (7, 10), (8, 11), (9, 12), # → left_foot, right_foot, neck
    (9, 13), (9, 14),          # spine3 → left_collar, right_collar
    (12, 15),                   # neck → head
    (13, 16), (14, 17),        # collars → shoulders
    (16, 18), (17, 19),        # shoulders → elbows
    (18, 20), (19, 21),        # elbows → wrists
]

SMPLX_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]


def _build_skeleton_xml(n_joints: int = 22) -> str:
    """Build a MuJoCo XML with sphere geoms for joints."""
    joints_xml = ""
    for i in range(n_joints):
        joints_xml += f'    <body name="joint_{i}" pos="0 0 0"><geom type="sphere" size="0.02" rgba="0 1 1 1"/></body>\n'

    return f"""<mujoco>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8"/>
  </visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom type="plane" size="2 2 0.01" rgba="0.2 0.2 0.2 1"/>
{joints_xml}  </worldbody>
</mujoco>"""


class SkeletonViewer:
    """Render 3D skeleton from joint positions using MuJoCo offscreen.

    Joints are rendered as sphere geoms. Bones are injected as connector
    geoms into the MjvScene at render time.
    """

    def __init__(self, width: int = 480, height: int = 480):
        self._width = width
        self._height = height
        xml = _build_skeleton_xml(n_joints=22)
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=height, width=width)
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.distance = 3.0
        self._camera.elevation = -20
        self._camera.azimuth = 135

    def render(self, joint_positions: np.ndarray) -> np.ndarray:
        """Render skeleton from joint positions.

        Args:
            joint_positions: (22, 3) array of joint positions in world coordinates (Z-up).

        Returns:
            RGB image as (H, W, 3) uint8 array.
        """
        assert joint_positions.shape == (22, 3), f"Expected (22, 3), got {joint_positions.shape}"

        # Update joint sphere positions
        for i in range(22):
            body_id = self._model.body(f"joint_{i}").id
            self._model.body_pos[body_id] = joint_positions[i]

        # Center camera on skeleton
        center = joint_positions.mean(axis=0)
        self._camera.lookat[:] = center

        mujoco.mj_forward(self._model, self._data)

        # Build scene with joint spheres
        self._renderer.update_scene(self._data, self._camera)

        # Inject bone connector geoms into the scene
        scene = self._renderer.scene
        bone_color = np.array([0.3, 0.8, 0.3, 0.8], dtype=np.float32)
        for a, b in SMPLX_BONES:
            if scene.ngeom >= scene.maxgeom:
                break
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3), np.zeros(3), np.zeros(9), bone_color,
            )
            mujoco.mjv_connector(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.008,
                joint_positions[a].astype(np.float64),
                joint_positions[b].astype(np.float64),
            )
            scene.ngeom += 1

        return self._renderer.render()

    def render_from_dict(self, joint_dict: dict) -> np.ndarray:
        """Render skeleton from a joint dictionary (as produced by gvhmr_to_gmr).

        Args:
            joint_dict: {joint_name: (pos_3d, quat_wxyz)} dict.

        Returns:
            RGB image as (H, W, 3) uint8 array.
        """
        positions = np.zeros((22, 3))
        for i, name in enumerate(SMPLX_JOINT_NAMES):
            if name in joint_dict:
                pos, _ = joint_dict[name]
                positions[i] = pos
        return self.render(positions)

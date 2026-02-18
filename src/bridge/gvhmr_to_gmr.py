"""
GVHMR → GMR data bridge.

Converts GVHMR .pt output (SMPL axis-angle params) to the per-frame
{body_name: (pos_3d, quat_wxyz)} dict format that GMR.retarget() expects.

This reimplements the logic from GMR's utils/smpl.py (load_gvhmr_pred_file +
get_gvhmr_data_offline_fast) so we don't depend on GMR's internal utils
which require smplx body models at import time.
"""

import pathlib
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R, Slerp

# SMPL-X joint names used by GMR's smplx_to_g1.json IK config
SMPLX_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]

# SMPL-X kinematic tree parent indices (for accumulating global rotations)
SMPLX_PARENTS = [
    -1,  # 0: pelvis
    0,   # 1: left_hip
    0,   # 2: right_hip
    0,   # 3: spine1
    1,   # 4: left_knee
    2,   # 5: right_knee
    3,   # 6: spine2
    4,   # 7: left_ankle
    5,   # 8: right_ankle
    6,   # 9: spine3
    7,   # 10: left_foot
    8,   # 11: right_foot
    9,   # 12: neck
    9,   # 13: left_collar
    9,   # 14: right_collar
    12,  # 15: head
    13,  # 16: left_shoulder
    14,  # 17: right_shoulder
    16,  # 18: left_elbow
    17,  # 19: right_elbow
    18,  # 20: left_wrist
    19,  # 21: right_wrist
]

# Coordinate correction: GVHMR Y-up → MuJoCo Z-up
_Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
_Y_UP_TO_Z_UP_QUAT = R.from_matrix(_Y_UP_TO_Z_UP).as_quat(scalar_first=True)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def load_gvhmr_result(pt_path: str) -> dict:
    """Load a GVHMR .pt result file.

    Returns:
        dict with keys: body_pose (N,63), betas (1,10), global_orient (N,3), transl (N,3)
    """
    pred = torch.load(pt_path, weights_only=False, map_location="cpu")
    params = pred["smpl_params_global"]
    return {
        "body_pose": params["body_pose"].numpy(),
        "betas": params["betas"].numpy(),
        "global_orient": params["global_orient"].numpy(),
        "transl": params["transl"].numpy(),
    }


def gvhmr_to_smplx_frames(
    gvhmr_result: dict,
    smplx_model_path: str,
    tgt_fps: int = 30,
) -> tuple[list[dict], float, float]:
    """Convert GVHMR output to per-frame SMPL-X joint dicts for GMR.

    Uses the smplx body model to do a forward pass, then extracts per-joint
    positions and global orientations, applies coordinate correction.

    Args:
        gvhmr_result: Output of load_gvhmr_result()
        smplx_model_path: Path to body_models/ directory containing smplx/
        tgt_fps: Target FPS for output (resamples if needed)

    Returns:
        (frames, aligned_fps, human_height) where frames is a list of
        {joint_name: (pos_xyz, quat_wxyz)} dicts
    """
    import smplx as smplx_lib

    body_pose = gvhmr_result["body_pose"]       # (N, 63)
    betas_raw = gvhmr_result["betas"]            # (1, 10)
    global_orient = gvhmr_result["global_orient"]  # (N, 3)
    transl = gvhmr_result["transl"]              # (N, 3)
    num_frames = body_pose.shape[0]

    # Pad betas from 10 → 16 for SMPL-X
    betas_16 = np.pad(betas_raw[0], (0, 6))  # (16,)

    # Estimate human height from shape parameter
    human_height = 1.66 + 0.1 * betas_16[0]

    # Create SMPL-X body model and run forward pass
    body_model = smplx_lib.create(
        smplx_model_path, "smplx",
        gender="neutral", use_pca=False,
        num_betas=16, batch_size=num_frames,
    )

    betas_t = torch.from_numpy(betas_16).unsqueeze(0).expand(num_frames, -1).float()
    body_pose_t = torch.from_numpy(body_pose).float()
    global_orient_t = torch.from_numpy(global_orient).float()
    transl_t = torch.from_numpy(transl).float()
    zeros_hand = torch.zeros(num_frames, 45)
    zeros_3 = torch.zeros(num_frames, 3)

    with torch.no_grad():
        smplx_out = body_model(
            betas=betas_t,
            global_orient=global_orient_t,
            body_pose=body_pose_t,
            transl=transl_t,
            left_hand_pose=zeros_hand,
            right_hand_pose=zeros_hand,
            jaw_pose=zeros_3,
            leye_pose=zeros_3,
            reye_pose=zeros_3,
            return_full_pose=True,
        )

    # joints: (N, num_joints, 3), full_pose: (N, num_joints*3) axis-angle
    joints = smplx_out.joints.numpy()        # (N, J, 3)
    full_pose = smplx_out.full_pose.numpy()  # (N, J*3)

    # Build per-frame dicts with global rotations
    num_joints = min(len(SMPLX_JOINT_NAMES), full_pose.shape[1] // 3)
    frames = []
    for f in range(num_frames):
        # Compute global rotations by accumulating through parent chain
        local_rots = []
        for j in range(num_joints):
            aa = full_pose[f, j*3:(j+1)*3]
            local_rots.append(R.from_rotvec(aa))

        global_rots = [None] * num_joints
        for j in range(num_joints):
            parent = SMPLX_PARENTS[j]
            if parent < 0:
                global_rots[j] = local_rots[j]
            else:
                global_rots[j] = global_rots[parent] * local_rots[j]

        frame = {}
        for j in range(num_joints):
            pos = joints[f, j]
            quat = global_rots[j].as_quat(scalar_first=True)
            name = SMPLX_JOINT_NAMES[j]
            frame[name] = (pos, quat)
        frames.append(frame)

    # Apply coordinate correction: Y-up → Z-up
    for frame in frames:
        for name in list(frame.keys()):
            pos, quat = frame[name]
            pos_corrected = pos @ _Y_UP_TO_Z_UP.T
            quat_corrected = _quat_mul(_Y_UP_TO_Z_UP_QUAT, quat)
            frame[name] = (pos_corrected, quat_corrected)

    # FPS resampling if needed (GVHMR outputs at 30 FPS)
    src_fps = 30.0
    if tgt_fps != src_fps and num_frames > 1:
        frames = _resample_frames(frames, src_fps, tgt_fps)

    return frames, float(tgt_fps), float(human_height)


def _resample_frames(
    frames: list[dict],
    src_fps: float,
    tgt_fps: float,
) -> list[dict]:
    """Resample motion frames to target FPS using linear/SLERP interpolation."""
    src_n = len(frames)
    src_duration = (src_n - 1) / src_fps
    tgt_n = int(src_duration * tgt_fps) + 1
    joint_names = list(frames[0].keys())

    new_frames = []
    for i in range(tgt_n):
        t = i / tgt_fps
        src_idx = t * src_fps
        idx0 = int(src_idx)
        idx1 = min(idx0 + 1, src_n - 1)
        alpha = src_idx - idx0

        frame = {}
        for name in joint_names:
            pos0, quat0 = frames[idx0][name]
            pos1, quat1 = frames[idx1][name]
            pos = pos0 * (1 - alpha) + pos1 * alpha
            r0 = R.from_quat(quat0, scalar_first=True)
            r1 = R.from_quat(quat1, scalar_first=True)
            slerp = Slerp([0, 1], R.concatenate([r0, r1]))
            quat = slerp(alpha).as_quat(scalar_first=True)
            frame[name] = (pos, quat)
        new_frames.append(frame)

    return new_frames

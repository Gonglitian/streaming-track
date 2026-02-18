#!/usr/bin/env python3
"""
GMR Retargeting 性能基准测试

测试不同场景下的 retargeting FPS:
- 基础 IK 求解 (无渲染)
- IK + headless 渲染
- 不同动作复杂度
"""

import sys
import time
import pathlib
import numpy as np
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
GMR_ROOT = PROJECT_ROOT / "third_party" / "GMR"
sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
import mujoco as mj


def generate_standing_frames(n=300):
    """静止站立姿态"""
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
    base = {
        "pelvis": (np.array([0.0, 0.0, 0.9]), identity_quat),
        "spine3": (np.array([0.0, 0.0, 1.3]), identity_quat),
        "left_hip": (np.array([0.1, 0.0, 0.85]), identity_quat),
        "right_hip": (np.array([-0.1, 0.0, 0.85]), identity_quat),
        "left_knee": (np.array([0.1, 0.0, 0.5]), identity_quat),
        "right_knee": (np.array([-0.1, 0.0, 0.5]), identity_quat),
        "left_foot": (np.array([0.1, 0.0, 0.05]), identity_quat),
        "right_foot": (np.array([-0.1, 0.0, 0.05]), identity_quat),
        "left_shoulder": (np.array([0.2, 0.0, 1.35]), identity_quat),
        "right_shoulder": (np.array([-0.2, 0.0, 1.35]), identity_quat),
        "left_elbow": (np.array([0.35, 0.0, 1.1]), identity_quat),
        "right_elbow": (np.array([-0.35, 0.0, 1.1]), identity_quat),
        "left_wrist": (np.array([0.45, 0.0, 0.9]), identity_quat),
        "right_wrist": (np.array([-0.45, 0.0, 0.9]), identity_quat),
    }
    return [base] * n


def generate_walking_frames(n=300):
    """模拟行走动作 (周期性)"""
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
    frames = []
    for i in range(n):
        t = i / 30.0  # 30 fps
        phase = t * 2 * np.pi  # ~1 step/sec

        # Oscillating legs
        leg_swing = np.sin(phase) * 0.15
        knee_bend = abs(np.sin(phase)) * 0.1

        frame = {
            "pelvis": (np.array([0.0, t * 0.3, 0.9 + np.sin(phase * 2) * 0.02]), identity_quat),
            "spine3": (np.array([0.0, t * 0.3, 1.3 + np.sin(phase * 2) * 0.01]), identity_quat),
            "left_hip": (np.array([0.1, t * 0.3 + leg_swing, 0.85]), identity_quat),
            "right_hip": (np.array([-0.1, t * 0.3 - leg_swing, 0.85]), identity_quat),
            "left_knee": (np.array([0.1, t * 0.3 + leg_swing * 0.5, 0.5 - knee_bend]),
                         R.from_euler("x", knee_bend * 2).as_quat(scalar_first=True)),
            "right_knee": (np.array([-0.1, t * 0.3 - leg_swing * 0.5, 0.5 - knee_bend]),
                          R.from_euler("x", knee_bend * 2).as_quat(scalar_first=True)),
            "left_foot": (np.array([0.1, t * 0.3 + leg_swing, 0.05 + max(0, np.sin(phase)) * 0.05]), identity_quat),
            "right_foot": (np.array([-0.1, t * 0.3 - leg_swing, 0.05 + max(0, -np.sin(phase)) * 0.05]), identity_quat),
            "left_shoulder": (np.array([0.2, t * 0.3, 1.35]), identity_quat),
            "right_shoulder": (np.array([-0.2, t * 0.3, 1.35]), identity_quat),
            "left_elbow": (np.array([0.35, t * 0.3 - leg_swing * 0.3, 1.1]), identity_quat),
            "right_elbow": (np.array([-0.35, t * 0.3 + leg_swing * 0.3, 1.1]), identity_quat),
            "left_wrist": (np.array([0.45, t * 0.3 - leg_swing * 0.5, 0.9]), identity_quat),
            "right_wrist": (np.array([-0.45, t * 0.3 + leg_swing * 0.5, 0.9]), identity_quat),
        }
        frames.append(frame)
    return frames


def benchmark_ik_only(retarget, frames, name=""):
    """Benchmark IK solving only (no rendering)"""
    # Warmup
    for i in range(min(20, len(frames))):
        retarget.retarget(frames[i])

    n = min(300, len(frames))
    start = time.perf_counter()
    for i in range(n):
        retarget.retarget(frames[i % len(frames)])
    elapsed = time.perf_counter() - start

    fps = n / elapsed
    ms = elapsed / n * 1000
    print(f"  {name:20s} | {fps:7.1f} FPS | {ms:6.2f} ms/frame | {n} frames")
    return fps


def benchmark_ik_with_render(retarget, frames, model, name=""):
    """Benchmark IK + headless rendering"""
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=480, width=640)

    # Warmup
    for i in range(min(10, len(frames))):
        qpos = retarget.retarget(frames[i])
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        renderer.update_scene(data)
        renderer.render()

    n = min(200, len(frames))
    start = time.perf_counter()
    for i in range(n):
        qpos = retarget.retarget(frames[i % len(frames)])
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        renderer.update_scene(data)
        renderer.render()
    elapsed = time.perf_counter() - start

    fps = n / elapsed
    ms = elapsed / n * 1000
    print(f"  {name:20s} | {fps:7.1f} FPS | {ms:6.2f} ms/frame | {n} frames")
    renderer.close()
    return fps


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     GMR + Unitree G1 性能基准测试                      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    xml_path = str(GMR_ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml")
    model = mj.MjModel.from_xml_path(xml_path)

    retarget = GMR(
        actual_human_height=1.8,
        src_human="smplx",
        tgt_robot="unitree_g1",
        verbose=False,
    )

    standing = generate_standing_frames(300)
    walking = generate_walking_frames(300)

    print("IK Only (无渲染):")
    print(f"  {'Scenario':20s} | {'FPS':>7s}     | {'Latency':>6s}      | Frames")
    print("  " + "-" * 60)
    fps_stand = benchmark_ik_only(retarget, standing, "Standing (static)")
    fps_walk = benchmark_ik_only(retarget, walking, "Walking (dynamic)")

    print()
    print("IK + Headless 渲染 (640x480):")
    print(f"  {'Scenario':20s} | {'FPS':>7s}     | {'Latency':>6s}      | Frames")
    print("  " + "-" * 60)
    fps_render_stand = benchmark_ik_with_render(retarget, standing, model, "Standing + render")
    fps_render_walk = benchmark_ik_with_render(retarget, walking, model, "Walking + render")

    print()
    print("总结:")
    print(f"  IK-only 平均:      {(fps_stand + fps_walk) / 2:.1f} FPS")
    print(f"  IK+Render 平均:    {(fps_render_stand + fps_render_walk) / 2:.1f} FPS")
    print(f"  目标:              60 FPS")
    print(f"  设备:              RTX 4060 8GB (CPU-based IK)")

    ok = min(fps_stand, fps_walk) >= 60
    if ok:
        print(f"  结果:              PASS")
    else:
        print(f"  结果:              WARN (IK-only < 60 FPS)")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

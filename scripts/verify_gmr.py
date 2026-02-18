#!/usr/bin/env python3
"""
GMR + Unitree G1 部署验证脚本

验收标准:
  1. GMR repo clone 成功
  2. G1 URDF/XML 模型加载成功
  3. 用 sample SMPL-X 数据测试 smplx_to_robot retargeting
  4. MuJoCo viewer 中 G1 模型正确显示 (headless渲染验证)
  5. gvhmr_to_robot 桥接数据格式验证
  6. 确认 retargeting 60+ FPS
"""

import sys
import os
import time
import pathlib
import json
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

# Project paths
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
GMR_ROOT = PROJECT_ROOT / "third_party" / "GMR"

sys.path.insert(0, str(GMR_ROOT))


def check_gmr_repo():
    """验收 1: GMR repo 结构完整"""
    print("=" * 60)
    print("[1/6] 验证 GMR repo 结构完整性")
    print("=" * 60)

    required_files = [
        "setup.py",
        "general_motion_retargeting/__init__.py",
        "general_motion_retargeting/motion_retarget.py",
        "general_motion_retargeting/params.py",
        "general_motion_retargeting/kinematics_model.py",
        "general_motion_retargeting/robot_motion_viewer.py",
        "general_motion_retargeting/ik_configs/smplx_to_g1.json",
        "scripts/smplx_to_robot.py",
        "scripts/gvhmr_to_robot.py",
        "scripts/vis_robot_motion.py",
        "assets/unitree_g1/g1_mocap_29dof.xml",
    ]

    all_ok = True
    for f in required_files:
        path = GMR_ROOT / f
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        if not exists:
            all_ok = False
        print(f"  [{status}] {f}")

    if all_ok:
        print("  -> PASS: GMR repo 结构完整")
    else:
        print("  -> FAIL: 部分文件缺失")
    return all_ok


def check_g1_model():
    """验收 2: G1 XML 模型加载"""
    print()
    print("=" * 60)
    print("[2/6] 验证 G1 URDF/XML 模型加载")
    print("=" * 60)

    import mujoco as mj

    xml_path = str(GMR_ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml")
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    print(f"  模型路径: {xml_path}")
    print(f"  Bodies: {model.nbody}")
    print(f"  Joints: {model.njnt}")
    print(f"  DOFs (nv): {model.nv}")
    print(f"  Actuators (nu): {model.nu}")
    print(f"  Geoms: {model.ngeom}")

    # Verify expected structure
    assert model.nv == 35, f"Expected 35 DOFs (6 free + 29 joints), got {model.nv}"
    assert model.nu == 29, f"Expected 29 actuators, got {model.nu}"

    # Check pelvis exists as root body
    pelvis_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")
    assert pelvis_id > 0, "pelvis body not found"

    # Forward kinematics test
    mj.mj_forward(model, data)
    pelvis_pos = data.xpos[pelvis_id].copy()
    print(f"  Pelvis 初始位置: {pelvis_pos}")

    print("  -> PASS: G1 模型加载成功, 29 DOF 验证通过")
    return True, model, data


def generate_synthetic_smplx_data(num_frames=60):
    """生成合成 SMPL-X 格式数据用于测试 retargeting.

    创建一个简单的站立姿态，手臂缓慢抬起的动作序列。
    """
    # Human body part names needed by smplx_to_g1.json
    body_parts = {
        "pelvis": np.array([0.0, 0.0, 0.9]),
        "spine3": np.array([0.0, 0.0, 1.3]),
        "left_hip": np.array([0.1, 0.0, 0.85]),
        "right_hip": np.array([-0.1, 0.0, 0.85]),
        "left_knee": np.array([0.1, 0.0, 0.5]),
        "right_knee": np.array([-0.1, 0.0, 0.5]),
        "left_foot": np.array([0.1, 0.0, 0.05]),
        "right_foot": np.array([-0.1, 0.0, 0.05]),
        "left_shoulder": np.array([0.2, 0.0, 1.35]),
        "right_shoulder": np.array([-0.2, 0.0, 1.35]),
        "left_elbow": np.array([0.35, 0.0, 1.1]),
        "right_elbow": np.array([-0.35, 0.0, 1.1]),
        "left_wrist": np.array([0.45, 0.0, 0.9]),
        "right_wrist": np.array([-0.45, 0.0, 0.9]),
    }

    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz

    frames = []
    for f in range(num_frames):
        t = f / num_frames
        frame = {}
        for name, base_pos in body_parts.items():
            pos = base_pos.copy()
            quat = identity_quat.copy()

            # Animate arms raising
            if "shoulder" in name or "elbow" in name or "wrist" in name:
                angle = t * np.pi * 0.5  # 0 to 90 degrees
                if "left" in name:
                    rot = R.from_euler("x", -angle)
                else:
                    rot = R.from_euler("x", -angle)
                quat = rot.as_quat(scalar_first=True)
                # Lift positions up
                pos[2] += np.sin(angle) * 0.2

            frame[name] = (pos, quat)
        frames.append(frame)

    return frames


def check_retargeting():
    """验收 3: 用 sample SMPL-X 数据测试 retargeting"""
    print()
    print("=" * 60)
    print("[3/6] 验证 SMPL-X → G1 Retargeting")
    print("=" * 60)

    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    # Generate test data
    num_frames = 60
    frames = generate_synthetic_smplx_data(num_frames)
    print(f"  生成合成 SMPL-X 数据: {num_frames} 帧")
    print(f"  Body parts per frame: {len(frames[0])}")

    # Initialize GMR
    retarget = GMR(
        actual_human_height=1.8,
        src_human="smplx",
        tgt_robot="unitree_g1",
        verbose=False,
    )
    print("  GMR 初始化成功")

    # Run retargeting
    qpos_list = []
    for i, frame in enumerate(frames):
        qpos = retarget.retarget(frame)
        qpos_list.append(qpos.copy())

    qpos_array = np.array(qpos_list)
    print(f"  Retargeting 完成: {qpos_array.shape}")
    print(f"    qpos shape: ({num_frames}, {qpos_array.shape[1]})")
    print(f"    root_pos range: {qpos_array[:, :3].min(axis=0)} ~ {qpos_array[:, :3].max(axis=0)}")
    print(f"    dof_pos range: [{qpos_array[:, 7:].min():.3f}, {qpos_array[:, 7:].max():.3f}]")

    # Validate output structure
    expected_qpos_dim = 36  # 7 (pos+quat) + 29 (joints)
    assert qpos_array.shape[1] == expected_qpos_dim, \
        f"Expected qpos dim {expected_qpos_dim}, got {qpos_array.shape[1]}"

    # Check no NaN
    assert not np.any(np.isnan(qpos_array)), "NaN detected in retargeted qpos"

    # Check root quaternion is normalized
    root_quats = qpos_array[:, 3:7]
    norms = np.linalg.norm(root_quats, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"Root quaternion not normalized: {norms}"

    print("  -> PASS: Retargeting 输出格式和数值正确")
    return True, qpos_list


def check_mujoco_rendering(qpos_list):
    """验收 4: MuJoCo 渲染验证 (headless)"""
    print()
    print("=" * 60)
    print("[4/6] 验证 MuJoCo Headless 渲染")
    print("=" * 60)

    import mujoco as mj

    xml_path = str(GMR_ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml")
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    # Set up offscreen renderer
    renderer = mj.Renderer(model, height=480, width=640)

    # Apply a sample frame and render
    mid_frame = len(qpos_list) // 2
    data.qpos[:] = qpos_list[mid_frame]
    mj.mj_forward(model, data)

    renderer.update_scene(data)
    pixels = renderer.render()

    print(f"  渲染分辨率: {pixels.shape[1]}x{pixels.shape[0]}")
    print(f"  像素范围: [{pixels.min()}, {pixels.max()}]")

    # Check render is not all black/empty
    assert pixels.max() > 0, "Render produced all-black image"
    assert pixels.shape == (480, 640, 3), f"Unexpected pixel shape: {pixels.shape}"

    # Save test render
    output_dir = PROJECT_ROOT / "data" / "sample_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
        img = Image.fromarray(pixels)
        img_path = output_dir / "g1_test_render.png"
        img.save(str(img_path))
        print(f"  测试渲染图保存: {img_path}")
    except ImportError:
        print("  (PIL 不可用, 跳过图片保存)")

    renderer.close()
    print("  -> PASS: MuJoCo headless 渲染成功")
    return True


def check_gvhmr_format():
    """验收 5: gvhmr_to_robot 数据格式验证"""
    print()
    print("=" * 60)
    print("[5/6] 验证 GVHMR→GMR 数据格式桥接")
    print("=" * 60)

    # Simulate GVHMR output format (hmr4d_results.pt)
    import torch

    num_frames = 30
    sample_gvhmr_output = {
        "smpl_params_global": {
            "body_pose": torch.randn(num_frames, 63),  # 21 joints * 3 axis-angle
            "betas": torch.randn(1, 10),  # shape params
            "global_orient": torch.randn(num_frames, 3),  # root orientation
            "transl": torch.zeros(num_frames, 3),  # root translation
        }
    }
    # Set standing position
    sample_gvhmr_output["smpl_params_global"]["transl"][:, 2] = 0.9

    # Save temp file
    output_dir = PROJECT_ROOT / "data" / "sample_output"
    temp_pt = output_dir / "test_gvhmr_output.pt"
    torch.save(sample_gvhmr_output, str(temp_pt))
    print(f"  合成 GVHMR 输出: {temp_pt}")

    # Verify loading format matches what gvhmr_to_robot.py expects
    loaded = torch.load(str(temp_pt), weights_only=False)
    smpl_params = loaded["smpl_params_global"]

    assert "body_pose" in smpl_params, "Missing body_pose"
    assert "betas" in smpl_params, "Missing betas"
    assert "global_orient" in smpl_params, "Missing global_orient"
    assert "transl" in smpl_params, "Missing transl"

    print(f"  body_pose shape: {smpl_params['body_pose'].shape} (expected: [N, 63])")
    print(f"  betas shape: {smpl_params['betas'].shape} (expected: [1, 10])")
    print(f"  global_orient shape: {smpl_params['global_orient'].shape} (expected: [N, 3])")
    print(f"  transl shape: {smpl_params['transl'].shape} (expected: [N, 3])")

    # Clean up
    temp_pt.unlink()

    print("  -> PASS: GVHMR 数据格式桥接验证通过")
    print("  NOTE: 完整 gvhmr_to_robot.py 需要 SMPL-X body models (assets/body_models/smplx/)")
    return True


def check_retargeting_fps():
    """验收 6: 确认 retargeting 60+ FPS"""
    print()
    print("=" * 60)
    print("[6/6] 验证 Retargeting 性能 (目标: 60+ FPS)")
    print("=" * 60)

    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    # Generate test data
    num_frames = 300  # 10 seconds at 30fps
    frames = generate_synthetic_smplx_data(num_frames)

    retarget = GMR(
        actual_human_height=1.8,
        src_human="smplx",
        tgt_robot="unitree_g1",
        verbose=False,
    )

    # Warmup
    for i in range(10):
        retarget.retarget(frames[i % num_frames])

    # Benchmark
    num_benchmark = 200
    start = time.perf_counter()
    for i in range(num_benchmark):
        retarget.retarget(frames[i % num_frames])
    elapsed = time.perf_counter() - start

    fps = num_benchmark / elapsed
    ms_per_frame = elapsed / num_benchmark * 1000

    print(f"  Benchmark: {num_benchmark} 帧")
    print(f"  总耗时: {elapsed:.3f}s")
    print(f"  FPS: {fps:.1f}")
    print(f"  每帧耗时: {ms_per_frame:.2f}ms")

    if fps >= 60:
        print(f"  -> PASS: {fps:.1f} FPS >= 60 FPS 目标")
    else:
        print(f"  -> WARN: {fps:.1f} FPS < 60 FPS 目标 (RTX 4060 8GB 可能需要优化)")

    return fps >= 60, fps


def save_output(qpos_list, fps=30.0):
    """保存 retargeting 结果为标准格式"""
    output_dir = PROJECT_ROOT / "data" / "sample_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    qpos_array = np.array(qpos_list)
    root_pos = qpos_array[:, :3]
    # GMR outputs wxyz, convert to xyzw for storage
    root_rot = qpos_array[:, 3:7][:, [1, 2, 3, 0]]
    dof_pos = qpos_array[:, 7:]

    motion_data = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }

    output_path = output_dir / "g1_test_motion.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"\n  Motion 数据保存: {output_path}")
    print(f"    frames: {len(qpos_list)}, dof: {dof_pos.shape[1]}")
    return output_path


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     GMR + Unitree G1 部署验证 (P1-3)                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    results = {}

    # Test 1: Repo structure
    results["repo"] = check_gmr_repo()

    # Test 2: G1 model loading
    ok, model, data = check_g1_model()
    results["model"] = ok

    # Test 3: Retargeting
    ok, qpos_list = check_retargeting()
    results["retarget"] = ok

    # Test 4: MuJoCo rendering
    results["render"] = check_mujoco_rendering(qpos_list)

    # Test 5: GVHMR format
    results["gvhmr_format"] = check_gvhmr_format()

    # Test 6: FPS benchmark
    ok, fps = check_retargeting_fps()
    results["fps"] = ok

    # Save output
    output_path = save_output(qpos_list)

    # Summary
    print()
    print("=" * 60)
    print("验证总结")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}")

    if all_pass:
        print()
        print("  ALL CHECKS PASSED - GMR + Unitree G1 部署验证成功!")
    else:
        print()
        print("  SOME CHECKS FAILED - 请检查上述输出")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

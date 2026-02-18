#!/bin/bash
# SMPL-X Body Models 下载指南
#
# GMR 的 smplx_to_robot.py 和 gvhmr_to_robot.py 需要 SMPL-X body models.
# 这些模型需要手动从官网下载 (需要注册).
#
# 步骤:
#   1. 访问 https://smpl-x.is.tue.mpg.de/ 注册并下载
#   2. 下载 SMPL-X v1.1 (models_smplx_v1_1.zip)
#   3. 解压到 GMR 的 assets/body_models/ 目录
#
# 目标目录结构:
#   third_party/GMR/assets/body_models/
#   └── smplx/
#       ├── SMPLX_FEMALE.npz  (或 .pkl)
#       ├── SMPLX_MALE.npz    (或 .pkl)
#       └── SMPLX_NEUTRAL.npz (或 .pkl)
#
# 注意: 如果下载的是 .pkl 格式, 需要修改 smplx 库:
#   smplx/body_models.py: ext='npz' 改为 ext='pkl'

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GMR_ROOT="$PROJECT_ROOT/third_party/GMR"
TARGET_DIR="$GMR_ROOT/assets/body_models/smplx"

echo "SMPL-X Body Models Setup"
echo "========================"
echo ""

if [ -f "$TARGET_DIR/SMPLX_NEUTRAL.npz" ] || [ -f "$TARGET_DIR/SMPLX_NEUTRAL.pkl" ]; then
    echo "Body models already exist at: $TARGET_DIR"
    ls -la "$TARGET_DIR/"
    echo ""
    echo "Setup complete!"
    exit 0
fi

echo "Body models NOT found at: $TARGET_DIR"
echo ""
echo "Please download SMPL-X models manually:"
echo "  1. Register at: https://smpl-x.is.tue.mpg.de/"
echo "  2. Download SMPL-X v1.1 models"
echo "  3. Extract to: $TARGET_DIR/"
echo ""

# Create target directory
mkdir -p "$TARGET_DIR"

# If user provides a zip file path as argument
if [ -n "$1" ]; then
    echo "Extracting from: $1"
    if [[ "$1" == *.zip ]]; then
        unzip -o "$1" -d "$TARGET_DIR/../"
        echo "Extraction complete!"
    else
        echo "Expected a .zip file"
        exit 1
    fi
else
    echo "Usage: $0 [path_to_models_smplx_v1_1.zip]"
    echo ""
    echo "Or manually copy the model files to: $TARGET_DIR/"
fi

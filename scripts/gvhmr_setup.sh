#!/bin/bash
# GVHMR Setup Script
# Sets up the GVHMR environment, dependencies, and model weights
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GVHMR_DIR="$PROJECT_ROOT/gvhmr_repo"

echo "=== GVHMR Setup ==="
echo "Project root: $PROJECT_ROOT"

# 1. Clone GVHMR if not present
if [ ! -d "$GVHMR_DIR" ]; then
    echo "[1/5] Cloning GVHMR..."
    git clone https://github.com/zju3dv/GVHMR.git "$GVHMR_DIR"
else
    echo "[1/5] GVHMR repo already exists, skipping clone"
fi

cd "$GVHMR_DIR"

# 2. Create conda environment
if ! conda env list | grep -q "^gvhmr "; then
    echo "[2/5] Creating conda environment (Python 3.10)..."
    conda create -y -n gvhmr python=3.10
else
    echo "[2/5] Conda env 'gvhmr' already exists, skipping"
fi

# 3. Install dependencies
echo "[3/5] Installing dependencies..."
eval "$(conda shell.bash hook)"
conda activate gvhmr

pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.0+cu121 torchvision==0.18.0+cu121

pip install timm==0.9.12 lightning==2.3.0 hydra-core==1.3 hydra-zen hydra_colorlog \
    rich 'numpy==1.23.5' matplotlib ipdb 'setuptools>=68.0' tensorboardX \
    opencv-python ffmpeg-python scikit-image termcolor einops 'imageio==2.34.1' \
    'av==13.0.0' joblib trimesh smplx pycolmap 'ultralytics==8.2.42' cython_bbox lapx wis3d

pip install 'pytorch3d @ https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl'
pip install --no-build-isolation chumpy
pip install -e .

# 4. Download model weights
echo "[4/5] Downloading model weights..."
mkdir -p inputs/checkpoints/{gvhmr,hmr2,vitpose,yolo,dpvo,body_models/smplx,body_models/smpl}
mkdir -p inputs outputs

# Download from Google Drive (pretrained models)
pip install gdown
if [ ! -f inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt ]; then
    gdown --folder https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD -O inputs/checkpoints/
fi

# Download SMPL/SMPLX body models from HuggingFace
for f in SMPLX_NEUTRAL.npz SMPLX_MALE.npz SMPLX_FEMALE.npz; do
    if [ ! -f "inputs/checkpoints/body_models/smplx/$f" ]; then
        wget -q -O "inputs/checkpoints/body_models/smplx/$f" \
            "https://huggingface.co/camenduru/SMPLer-X/resolve/main/$f"
    fi
done
for f in SMPL_NEUTRAL.pkl SMPL_MALE.pkl SMPL_FEMALE.pkl; do
    if [ ! -f "inputs/checkpoints/body_models/smpl/$f" ]; then
        wget -q -O "inputs/checkpoints/body_models/smpl/$f" \
            "https://huggingface.co/camenduru/SMPLer-X/resolve/main/$f"
    fi
done

# 5. Verify
echo "[5/5] Verifying installation..."
python -c "
import torch, pytorch3d, smplx, ultralytics
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'pytorch3d: {pytorch3d.__version__}')
print('All dependencies OK')
"

echo ""
echo "=== Setup Complete ==="
echo "Activate with: conda activate gvhmr"
echo "Run demo with: cd $GVHMR_DIR && python tools/demo/demo.py --video=<path> -s"

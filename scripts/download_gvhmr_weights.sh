#!/bin/bash
# Download GVHMR model weights and body models
# Requires: pip install gdown (already in requirements.txt deps)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GVHMR_DIR="$PROJECT_ROOT/third_party/GVHMR"

echo "=== GVHMR Weight Download ==="
echo "GVHMR dir: $GVHMR_DIR"

if [ ! -d "$GVHMR_DIR" ]; then
    echo "ERROR: GVHMR not found at $GVHMR_DIR"
    echo "Run: git clone https://github.com/zju3dv/GVHMR.git $GVHMR_DIR"
    exit 1
fi

cd "$GVHMR_DIR"

# Create checkpoint directories
mkdir -p inputs/checkpoints/{gvhmr,hmr2,vitpose,yolo,dpvo,body_models/smplx,body_models/smpl}
mkdir -p inputs outputs

# Download pretrained models from Google Drive
echo "[1/3] Downloading GVHMR pretrained models..."
pip install -q gdown
if [ ! -f inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt ]; then
    gdown --folder https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD -O inputs/checkpoints/
else
    echo "  Already downloaded, skipping"
fi

# Download SMPL-X body models from HuggingFace
echo "[2/3] Downloading SMPL-X body models..."
for f in SMPLX_NEUTRAL.npz SMPLX_MALE.npz SMPLX_FEMALE.npz; do
    if [ ! -f "inputs/checkpoints/body_models/smplx/$f" ]; then
        echo "  Downloading $f..."
        wget -q -O "inputs/checkpoints/body_models/smplx/$f" \
            "https://huggingface.co/camenduru/SMPLer-X/resolve/main/$f"
    fi
done

echo "[3/3] Downloading SMPL body models..."
for f in SMPL_NEUTRAL.pkl SMPL_MALE.pkl SMPL_FEMALE.pkl; do
    if [ ! -f "inputs/checkpoints/body_models/smpl/$f" ]; then
        echo "  Downloading $f..."
        wget -q -O "inputs/checkpoints/body_models/smpl/$f" \
            "https://huggingface.co/camenduru/SMPLer-X/resolve/main/$f"
    fi
done

# Verify
echo ""
echo "=== Download Complete ==="
echo "Checkpoints: $GVHMR_DIR/inputs/checkpoints/"
ls -la inputs/checkpoints/gvhmr/ 2>/dev/null || echo "  (gvhmr weights not found)"
ls -la inputs/checkpoints/body_models/smplx/ 2>/dev/null || echo "  (smplx models not found)"

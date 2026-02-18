#!/usr/bin/env bash
# Unified conda environment setup for GVHMR + GMR (streaming-track)
# Python 3.10 | PyTorch 2.3.0 + CUDA 12.1 | RTX 4060 8GB
set -euo pipefail

ENV_NAME="streaming-track"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_EXE:-$(which conda 2>/dev/null || echo "$HOME/miniconda3/bin/conda")}"

echo "=== Creating conda environment: $ENV_NAME ==="
$CONDA_BIN create -y -n "$ENV_NAME" python=3.10

ENV_PREFIX="$($CONDA_BIN info --base)/envs/$ENV_NAME"
PIP="$ENV_PREFIX/bin/pip"

echo "=== Installing all dependencies from requirements.txt ==="
$PIP install -r "$SCRIPT_DIR/requirements.txt"

echo "=== Fixing libstdc++ for Linux rendering ==="
$CONDA_BIN install -y -n "$ENV_NAME" -c conda-forge libstdcxx-ng

echo "=== Verifying installation ==="
$ENV_PREFIX/bin/python -c "
import torch; assert torch.cuda.is_available(), 'CUDA not available'
import mujoco
import smplx
import ultralytics
import mink
print('All checks passed!')
"

echo "=== Environment setup complete ==="
echo "Activate with: conda activate $ENV_NAME"

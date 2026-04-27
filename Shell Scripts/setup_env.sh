#!/bin/bash
# ============================================================
# One-time environment setup — run from the OSCAR login node
# before submitting slurm_train.sh for the first time.
#
# Usage:
#   bash setup_env.sh
# ============================================================

set -e

VENV_DIR=~/envs/cs1470
PYTHON_MODULE=python/3.11.11-5e66
CUDA_MODULE=cuda/12.9.0-cinr

echo "Loading modules: $PYTHON_MODULE  $CUDA_MODULE"
module load "$PYTHON_MODULE" "$CUDA_MODULE"
module load cudnn 2>/dev/null && echo "cuDNN loaded" || echo "cuDNN not available as separate module, continuing..."

# Recreate venv if wrong Python version
if [ -d "$VENV_DIR" ]; then
    VENV_PY=$("$VENV_DIR/bin/python" --version 2>&1 | grep -oP '3\.\d+')
    if [[ "$VENV_PY" != "3.11"* ]]; then
        echo "Wrong Python version in venv ($VENV_PY), recreating..."
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Upgrade pip/setuptools (pin setuptools<82 for torch compatibility)
pip install --quiet --upgrade pip wheel "setuptools<82"

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install PyTorch
if ! python -c "import torch" &>/dev/null; then
    echo "Installing PyTorch (cu126 wheels)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
fi

# Install other dependencies
if ! python -c "import timm, tensorboardX, einops, transformers, PIL" &>/dev/null; then
    echo "Installing dependencies..."
    pip install \
        timm==1.0.15 \
        tensorboardX==2.6.2.2 \
        einops==0.8.1 \
        transformers==4.50.0 \
        Pillow==11.1.0 \
        requests==2.32.3
fi

# Install mamba-ssm from GitHub (PyPI sdist missing C++ source files)
if ! python -c "import mamba_ssm" &>/dev/null; then
    echo "Installing mamba-ssm from GitHub source (this takes 10-20 min)..."
    pip install --no-build-isolation "git+https://github.com/state-spaces/mamba.git@v2.2.4"
fi

# Patch torch._utils to restore _accumulate removed in PyTorch 2.0
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
PATCHFILE="$SITE_PACKAGES/torch_accumulate_compat.py"
PTHFILE="$SITE_PACKAGES/torch_accumulate_compat.pth"
if [ ! -f "$PTHFILE" ]; then
    echo "Applying torch._utils._accumulate compatibility patch..."
    cat > "$PATCHFILE" << 'EOF'
try:
    import itertools
    import torch._utils
    if not hasattr(torch._utils, '_accumulate'):
        torch._utils._accumulate = itertools.accumulate
except Exception:
    pass
EOF
    echo "import torch_accumulate_compat" > "$PTHFILE"
fi

echo ""
echo "============================================"
echo "Setup complete. You can now submit the job:"
echo "  sbatch slurm_train.sh train"
echo "============================================"

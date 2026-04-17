#!/bin/bash

# ============================================================
# STL-10 Training - Slurm Job Script
# Compatible with Oscar/Slurm clusters
#
# Usage:
#   sbatch slurm_train.sh train         # runs training
#   sbatch slurm_train.sh validate      # runs validation
#
# Monitor your job:
#   myq                      # check job status
#   cat slurm-<jobid>.out    # view stdout
#   cat slurm-<jobid>.err    # view stderr
# ============================================================

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J stl10_train
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Default task if none provided as argument
TASK=${1:-train}

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Task:      $TASK"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

# Run from the code directory
cd "$SLURM_SUBMIT_DIR"

# ============================================================
# Environment setup — auto-detects modules and installs deps
# on first run. Safe to re-run; skips if already installed.
# ============================================================

VENV_DIR=~/envs/cs1470

# Auto-detect Python module (prefer 3.11, fall back to available)
PYTHON_MODULE=python/3.11.11-5e66
CUDA_MODULE=cuda/12.9.0-cinr

echo "Loading modules: $PYTHON_MODULE  $CUDA_MODULE"
module load "$PYTHON_MODULE" "$CUDA_MODULE"

# Create venv on first run
if [ ! -d "$VENV_DIR" ]; then
    echo "First-time setup: creating virtual environment at $VENV_DIR"
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install packages if torch is missing
if ! python -c "import torch" &>/dev/null; then
    echo "First-time setup: installing PyTorch..."
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu126
fi

# Install remaining packages if any are missing
if ! python -c "import timm, tensorboardX, einops, transformers, PIL, mamba_ssm" &>/dev/null; then
    echo "First-time setup: installing remaining dependencies..."
    pip install --quiet \
        mamba-ssm==2.2.4 \
        timm==1.0.15 \
        tensorboardX==2.6.2.2 \
        einops==0.8.1 \
        transformers==4.50.0 \
        Pillow==11.1.0 \
        requests==2.32.3
fi

echo "Environment ready."

# Run training or validation
if [ "$TASK" = "train" ]; then
    python MambaVision/train.py
elif [ "$TASK" = "validate" ]; then
    python MambaVision/validate.py
else
    echo "Unknown task: $TASK"
    exit 1
fi

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"

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
#SBATCH -t 04:00:00
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
# Environment setup — activate pre-built venv.
# Run setup_env.sh on the login node once before submitting.
# ============================================================

VENV_DIR=~/envs/cs1470
PYTHON_MODULE=python/3.11.11-5e66
CUDA_MODULE=cuda/12.9.0-cinr

module load "$PYTHON_MODULE" "$CUDA_MODULE"
module load cudnn 2>/dev/null || true

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: venv not found. Run setup_env.sh from the login node first."
    exit 1
fi

source "$VENV_DIR/bin/activate"
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "Environment ready."

# Run training or validation
if [ "$TASK" = "train" ]; then
    python MambaVision/train.py \
        --config MambaVision/configs/mambavision_tiny_1k.yaml \
        --data_dir "$SLURM_SUBMIT_DIR/STL-10/imagefolder" \
        --num-classes 10 \
        --data_len 25000 \
        --epochs 200 \
        --warmup-epochs 10 \
        --cooldown-epochs 5 \
        --min-lr 1e-5 \
        --workers 4 \
        --model-ema-decay 0.999 2>&1
elif [ "$TASK" = "validate" ]; then
    python MambaVision/validate.py \
        --config MambaVision/configs/mambavision_tiny_1k.yaml \
        --data_dir "$SLURM_SUBMIT_DIR/STL-10/imagefolder" \
        --num-classes 10 2>&1
else
    echo "Unknown task: $TASK"
    exit 1
fi

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"

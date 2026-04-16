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

# Activate your environment if needed
# source ~/.local/bin/env
# Or conda/pip/venv activation here

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

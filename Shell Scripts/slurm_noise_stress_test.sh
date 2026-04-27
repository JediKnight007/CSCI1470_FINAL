#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J noise_stress_test
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

cd "$SLURM_SUBMIT_DIR"

VENV_DIR=~/envs/cs1470
PYTHON_MODULE=python/3.11.11-5e66
CUDA_MODULE=cuda/12.9.0-cinr

module load "$PYTHON_MODULE" "$CUDA_MODULE"
module load cudnn 2>/dev/null || true

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: venv not found."
    exit 1
fi

source "$VENV_DIR/bin/activate"
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "Environment ready."
echo "Running Gaussian noise stress test..."

python stress_test_noise.py

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"

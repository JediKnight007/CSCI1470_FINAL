#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 05:00:00
#SBATCH -J stl10_nobypass
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

TASK=${1:-train}

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Task:      $TASK"
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

if [ "$TASK" = "train" ]; then
    python MambaVision/train.py \
        --model mamba_vision_T_nobypass \
        --config MambaVision/configs/mambavision_tiny_1k.yaml \
        --data_dir "$SLURM_SUBMIT_DIR/STL-10/imagefolder" \
        --num-classes 10 \
        --data_len 25000 \
        --epochs 300 \
        --warmup-epochs 10 \
        --cooldown-epochs 5 \
        --min-lr 1e-6 \
        --lr 0.001 \
        --weight-decay 0.05 \
        --mixup 0.4 \
        --cutmix 0.5 \
        --workers 4 \
        --model-ema-decay 0.999 \
        --clip-grad 1.0
elif [ "$TASK" = "validate" ]; then
    python MambaVision/validate.py \
        --model mamba_vision_T_nobypass \
        --config MambaVision/configs/mambavision_tiny_1k.yaml \
        --data_dir "$SLURM_SUBMIT_DIR/STL-10/imagefolder" \
        --num-classes 10 \
        --tta 3
fi

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
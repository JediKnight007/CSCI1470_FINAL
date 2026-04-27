#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J stress_test
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
source "$VENV_DIR/bin/activate"
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

CHECKPOINT="$SLURM_SUBMIT_DIR/output/20260427-094600-mamba_vision_T-224/checkpoint-225.pth.tar"
DATA="$SLURM_SUBMIT_DIR/STL-10/imagefolder"

echo "============================================"
echo "STRESS TEST: MambaVision-T"
echo "Checkpoint: $CHECKPOINT"
echo "============================================"

echo "--- Baseline: 224x224 ---"
python MambaVision/validate.py \
    --model mamba_vision_T \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$DATA" \
    --num-classes 10 \
    --img-size 224

echo "--- Small: 112x112 ---"
python MambaVision/validate.py \
    --model mamba_vision_T \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$DATA" \
    --num-classes 10 \
    --img-size 112

echo "--- Large: 336x336 ---"
python MambaVision/validate.py \
    --model mamba_vision_T \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$DATA" \
    --num-classes 10 \
    --img-size 336

echo "--- Very large: 448x448 ---"
python MambaVision/validate.py \
    --model mamba_vision_T \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$DATA" \
    --num-classes 10 \
    --img-size 448

echo "--- Tiny: 64x64 ---"
python MambaVision/validate.py \
    --model mamba_vision_T \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$DATA" \
    --num-classes 10 \
    --img-size 64

echo "--- Huge: 512x512 ---"
python MambaVision/validate.py \
    --model mamba_vision_T \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$DATA" \
    --num-classes 10 \
    --img-size 512

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
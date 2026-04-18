#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -J vit_train
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

module load python/3.11.11-5e66
module load cuda/12.9.0-cinr
module load cudnn 2>/dev/null || true

source ~/envs/cs1470/bin/activate
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

python ViT/train_vit_baseline.py \
  --data-dir "$SLURM_SUBMIT_DIR/STL-10/imagefolder" \
  --epochs 500 \
  --batch-size 128 \
  --lr 0.005 \
  --num-classes 10 \
  --output ./output/vit_baseline 2>&1

echo "Finished: $(date)"

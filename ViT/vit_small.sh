#!/bin/bash
#SBATCH --job-name=vit_small
#SBATCH --output=vit_small.out
#SBATCH --error=vit_small.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate cs1470
python ViT/train_vit_baseline.py --model vit_small_patch16_224

#!/bin/bash
#SBATCH --job-name=vit_tiny
#SBATCH --output=vit_tiny.out
#SBATCH --error=vit_tiny.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate cs1470
python ViT/train_vit_baseline.py --model vit_tiny_patch16_224

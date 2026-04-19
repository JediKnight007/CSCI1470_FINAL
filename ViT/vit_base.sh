#!/bin/bash
#SBATCH --job-name=vit_base
#SBATCH --output=vit_base.out
#SBATCH --error=vit_base.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate cs1470
python ViT/train_vit_baseline.py --model vit_base_patch16_224

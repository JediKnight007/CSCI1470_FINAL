#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J vit_small
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
	echo "ERROR: venv not found. Run setup_env.sh from the login node first."
	exit 1
fi

source "$VENV_DIR/bin/activate"
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

echo "Environment ready."

python ViT/train_vit_baseline.py --model vit_small_patch16_224

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
#!/bin/bash
#SBATCH --job-name=vit_small
#SBATCH --output=vit_small.out
#SBATCH --error=vit_small.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate cs1470
python ViT/train_vit_baseline.py --model vit_small_patch16_224

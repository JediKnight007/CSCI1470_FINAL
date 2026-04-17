# Ube Macchiatos: MambaVision Replication Project

## Project Overview
This project is a critical replication and analysis of MambaVision, the first hybrid Mamba-Transformer vision backbone. We focus on the MambaVision-T (Tiny) variant, aiming to verify its accuracy and efficiency claims, and to perform hypothesis-driven ablations on its hybrid mixer block. We compare it against transformer baselines (e.g., DeiT-Tiny / ViT-Tiny) and evaluate on the STL-10 dataset.

---

## Results Summary

All runs use MambaVision-T trained from scratch on STL-10 on Brown University's OSCAR cluster (NVIDIA RTX 3090).

| Run | Training Set | Epochs | Best Top-1 Acc | Notes |
|-----|-------------|--------|---------------|-------|
| 1 | 5k original | 110 | ~56% | Baseline, default config |
| 2 | 25k augmented | 110 | ~52% | Underfit — too few epochs for larger set |
| 3 | 25k augmented | 155 | 63.1% | Added LR schedule tuning |
| 4 | 25k augmented | 205 | 68.2% | Model still improving at end |
| 5 | 25k augmented | 250+ | In progress | — |

---

## Directory Structure

```
CS1470FINAL/
├── MambaVision/              # MambaVision model code (from official repo)
│   ├── train.py              # Main training script
│   ├── validate.py           # Validation script
│   ├── configs/              # YAML configs per model variant
│   └── models/mamba_vision.py
├── STL-10/                   # Dataset (not tracked in git)
│   ├── stl10_binary/         # Raw binary files from download
│   └── imagefolder/          # Converted ImageFolder structure (generated)
├── download_stl10.py         # Downloads STL-10 from Stanford
├── prepare_stl10_imagefolder.py  # Converts STL-10 to ImageFolder + augmentation
├── setup_env.sh              # One-time environment setup (run on login node)
├── slurm_train.sh            # SLURM job submission script
└── stl10_dataset.py          # Custom STL-10 dataset wrapper (reference)
```

---

## Setup & Running on OSCAR (Brown University HPC)

### Step 1 — Clone and download data (login node)
```bash
git clone git@github.com:JediKnight007/CSCI1470_FINAL.git
cd CSCI1470_FINAL
python download_stl10.py
```

### Step 2 — Build the augmented ImageFolder dataset (login node)
Converts the 5,000 binary STL-10 images into 25,000 PNG images organized by class, applying 4 offline augmentations (horizontal flip, 90°/180°/270° rotation) per image.
```bash
python prepare_stl10_imagefolder.py
```

### Step 3 — Set up the Python environment (login node, one-time)
Installs Python 3.11, CUDA 12.9, PyTorch, and all dependencies including `mamba-ssm` compiled from source.
```bash
bash setup_env.sh
```
This takes ~15-20 minutes on first run due to CUDA kernel compilation for `mamba-ssm`.

### Step 4 — Submit the training job
```bash
sbatch slurm_train.sh train
```

### Step 5 — Monitor progress
```bash
myq                          # check job status
tail -f slurm-<jobid>.out    # live training output
tail -f slurm-<jobid>.err    # errors (timm also logs here)
```

### Validation only
```bash
sbatch slurm_train.sh validate
```

---

## Key Configuration

Training is controlled entirely via command-line args in `slurm_train.sh` — no source files need editing:

| Argument | Value | Purpose |
|----------|-------|---------|
| `--config` | `mambavision_tiny_1k.yaml` | Model architecture and base hyperparameters |
| `--data_dir` | `STL-10/imagefolder` | ImageFolder-format dataset path |
| `--num-classes` | 10 | STL-10 has 10 classes |
| `--data_len` | 25000 | Total training images (5k × 5 augmentations) |
| `--epochs` | 250 | Training epochs |
| `--warmup-epochs` | 10 | LR warmup (reduced from 20 for small dataset) |
| `--cooldown-epochs` | 5 | LR cooldown (reduced from 10) |
| `--min-lr` | 1e-5 | Minimum LR floor |
| `--model-ema-decay` | 0.999 | EMA decay (reduced from 0.9998 for small dataset) |
| `--workers` | 4 | DataLoader workers (matches SLURM CPU allocation) |

---

## Struggles & Limitations

### Environment issues
- **Wrong Python version in venv**: OSCAR's system Python is 3.9 but modules provide 3.11. The venv must be created *after* loading the Python module. `setup_env.sh` handles this automatically.
- **`mamba-ssm` PyPI sdist missing C++ source files**: The PyPI package for `mamba-ssm==2.2.4` is missing `csrc/selective_scan/selective_scan.cpp`. Fixed by installing directly from GitHub source: `pip install git+https://github.com/state-spaces/mamba.git@v2.2.4`.
- **`libcusparseLt.so` not found during build**: `mamba-ssm` requires CUDA libraries visible at compile time. Fixed by using `--no-build-isolation` so pip reuses the already-installed torch instead of a sandboxed build environment.
- **Module names with hash suffixes**: OSCAR uses non-standard module names like `python/3.11.11-5e66` and `cuda/12.9.0-cinr`. Hardcoded in both `setup_env.sh` and `slurm_train.sh`.
- **Compute nodes have no internet access**: All pip installs must be done from the login node via `setup_env.sh` before submitting jobs.

### Dataset issues
- **`torch._utils._accumulate` removed in PyTorch 2.x**: MambaVision's `utils/datasets.py` imports this removed function. Fixed via a `sitecustomize.py` compatibility patch injected into the venv by `setup_env.sh` — no source modifications needed.
- **timm `torch/STL10` dataset not recognized**: timm's torchvision wrapper does not support STL-10 by name. Resolved by converting the binary STL-10 files to ImageFolder format using `prepare_stl10_imagefolder.py`.
- **STL-10 is a small dataset (5,000 training images)**: MambaVision is designed for ImageNet (1.28M images). Training from scratch on 5k images causes underfitting. Addressed with 5× offline augmentation to 25,000 images.

### Training issues
- **EMA model stuck at ~10% accuracy**: The default EMA decay (0.9998) is tuned for 1.28M ImageNet images. With 5k images, EMA updates too slowly to track training. Fixed by lowering `--model-ema-decay` to `0.999`.
- **LR schedule mismatch**: The cosine schedule in the config targets 310 epochs on ImageNet. Running 100-150 epochs caused the LR to decay too fast before convergence. Fixed by reducing `--warmup-epochs`, `--cooldown-epochs`, and raising `--min-lr`.
- **Training logs go to `.err` not `.out`**: timm uses Python's `logging` module which writes to `stderr`. Added `2>&1` redirect in `slurm_train.sh` to unify output into `.out`.
- **SLURM captures script at submission time**: Edits to `slurm_train.sh` only take effect on *new* job submissions after `git pull`. Running jobs always use the script as it was when `sbatch` was called.

### Model limitations
- **Training from scratch vs fine-tuning**: MambaVision's published results use ImageNet pretrained weights. Training from scratch significantly limits achievable accuracy on STL-10. Pretrained weights would require downloading ~100MB checkpoint files.
- **96×96 STL-10 images upscaled to 224×224**: MambaVision expects 224×224 input. STL-10 images are 96×96 and are upscaled, which limits the quality of learned features.

---

## Datasets

Large datasets are not tracked in this repository. STL-10 is downloaded automatically.

- **STL-10**: Downloaded via `python download_stl10.py` (Stanford hosted)
- **ObjectNet / MedMNIST**: Planned for future experiments

---

## Contact
Team: Ube Macchiatos

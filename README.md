## Check-in 2 Reflection

### Introduction
MambaVision is a recently proposed hybrid architecture that combines state space models (SSMs) with transformer-like components to improve the tradeoff between accuracy and efficiency. In particular, it introduces a mixer block that integrates an SSM branch for efficient sequence modeling and a symmetric non-SSM branch for global context aggregation.
In this project, we perform a critical replication of MambaVision-T, focusing on verifying its claimed advantages over standard vision transformers. We evaluate both MambaVision-T and a ViT baseline on STL-10 and compare their performance in terms of accuracy and computational efficiency.
Beyond standard evaluation, we design hypothesis-driven ablation studies to understand the contribution of the mixer block and conduct stress tests under challenging settings such as extreme aspect ratios, low-data regimes, and distribution shifts. Our goal is to examine not only whether MambaVision performs well, but also when its advantages hold and where they may break down.

### Challenges
1. **Fixing NaN Errors**: A major challenge was dealing with NaN errors caused by gradient explosion and numerical instability in the SSM components, where small errors accumulate to cause floating-point overflow. We had to carefully tune hyperparameters to stabilize the model.
2. **Untangling the Mixer Block**: For ablation studies, we needed to isolate specific model components. The original "mixer block" splits data into an SSM branch and a non-SSM branch, squeezing both to half their size before merging. Safely isolating just the SSM piece without breaking the model's structural integrity or test fairness was our hardest coding task.

### Insights
Our results are highly promising and validate the original paper's claims. Our replicated MambaVision-T baseline achieved a Best Top-1 accuracy of 89.225%, vastly outperforming ViT-Tiny (71.31%) and ViT-Small (68.39%). Beyond baseline comparisons, our ablation studies yielded two concrete insights:
  1. **Attention Placement Matters**: Moving self-attention to the first half of the network stages caused a 1.275% accuracy drop (to 87.95%), proving self-attention is most effective in the final layers to capture global context. Removing attention entirely caused a massive 3.85% drop. 
  2. **The Bypass Branch is Crucial**: Removing the symmetric non-SSM branch resulted in a substantial 1.6% accuracy drop (to 87.625%), proving this parallel path is vital for making up for content lost in the SSM.

## Plan
- **On Track?**: We’re on track with our project. Now that we've confirmed the model's high accuracy, we need to show why it works so well by testing it with certain components turned off. We need to spend more time isolating the effects of the MambaVision token mixer (by testing the model without its specific State Space Model branch) and experimenting with the hybrid layout (to see how performance suffers when we remove the self-attention blocks from the model's final layers).
-	**Possible changes**: Based on the final data, we are adjusting our narrative focus. Initially, preliminary tests made us think the bypass branch had a negligible effect. However, our final results show the "no bypass" ablation actually caused a steeper penalty (-1.6%) than shifting the attention blocks (-1.275%). We will now change our conclusion to emphasize that MambaVision’s success relies just as heavily on its internal block redesign (the bypass branch) as it does on its macro layout (final-layer attention).

<br></br>

## ViT Baseline: Data Preparation & Fair Comparison

### STL-10 ImageFolder Augmentation (No Duplicates)

To ensure a fair comparison between MambaVision and ViT baselines, we:

1. **Convert STL-10 to ImageFolder:**
  - Run `python prepare_stl10_imagefolder.py` to convert the STL-10 binary files to ImageFolder format.
  - The script applies 4 augmentations (horizontal flip, 90°/180°/270° rotation) to each original image, producing 5 images per original (5,000 × 5 = 25,000 images).
  - **Duplicate Prevention:** The script now clears the output train directory before saving, ensuring no duplicate images are present, even if rerun.

2. **Verify Dataset Size:**
  - After running the script, each class folder in `STL-10/imagefolder/train/` should contain exactly 2,500 images (for 25,000 total).
  - Check with:
    ```bash
    cd STL-10/imagefolder/train
    for d in */; do echo "$d: $(ls "$d" | wc -l)"; done
    ```

3. **Set Data Path in Training Scripts:**
  - Both MambaVision and ViT training scripts must use the same train directory: `/absolute/path/to/STL-10/imagefolder/train`.
  - This guarantees both models see the same data and augmentation.

4. **Retrain for Fair Comparison:**
  - After confirming the dataset, retrain both models. Results are only comparable if the data pipeline is identical.

**Note:** If you see more than 2,500 images per class (e.g., 3,000), clear the train directory and rerun the augmentation script to avoid duplicates.

---
---


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
| 5 | 25k augmented | 254 | 74.1% | Best at epoch 254; EMA top-1 74.09% |
| 6 | 25k augmented | 304 | 76.0% | Best at epoch 304; EMA top-1 76.03%; LR still decaying |
| 7 | 25k augmented | 404 | **79.7%** | Resumed from ep 304 with fixed data_len; LR reached min-lr at ep 400; fully converged |

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

### Resuming from a checkpoint
To continue training from a previous run's best checkpoint, add `--resume` to the `train.py` call in `slurm_train.sh`:
```bash
        --resume "$SLURM_SUBMIT_DIR/../output/train/mambavision_tiny_1k/<run-dir>/checkpoint-<epoch>.pth.tar" \
```
Checkpoints are saved to `../output/train/mambavision_tiny_1k/` relative to where `train.py` is run (i.e., one level above `CS1470FINAL/`). Each run creates a timestamped subdirectory like `20260417-165016-mamba_vision_T-224/`. You can list available runs with:
```bash
ls ~/output/train/mambavision_tiny_1k/
```
The best checkpoint filename and its accuracy are printed at the end of each epoch as `Current checkpoints:` in the training log.

Timm reads the saved epoch number from the checkpoint and resumes from the next epoch automatically — no need to set `--start-epoch`. The optimizer state and EMA weights are also restored.

To start a **fresh run** instead, remove the `--resume` line entirely.

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

## Recent Upgrades & Fixes (Spring 2026)

- **Learning Rate Schedule Bug Fixed:**
  - `--data_len` was set to 25000, but only 5000 images were used per epoch. Fixed by setting `--data_len 5000`, so the cosine LR schedule decays at the correct rate.
- **Longer Training:**
  - Increased `--epochs` to 500 for full convergence. Model now reaches LR floor and accuracy plateaus.
- **Augmentation Tuning:**
  - Reduced `--mixup` to 0.4 and `--cutmix` to 0.5 for better convergence on small data.
- **Weight Decay Stability:**
  - Kept `--weight-decay 0.05` (LAMB optimizer unstable with lower values).
- **Test-Time Augmentation:**
  - Added `--tta 3` to validation for a small accuracy boost.
- **Per-Epoch Timing:**
  - Each epoch now logs its runtime in seconds and minutes.
- **Accuracy Summary:**
  - At the end of training, the script prints top-1 accuracy every 25 epochs for easy progress tracking.
- **Resume Logic:**
  - Documented how to resume from checkpoints and how to start a clean run.
- **Best Result:**
  - With all fixes, MambaVision-T achieves **80.7% top-1** on STL-10 from scratch.

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

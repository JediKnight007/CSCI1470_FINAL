#!/usr/bin/env python3
"""
Stress test: evaluate MambaVision-T under Gaussian noise corruption.
Tests robustness to input noise at varying severity levels.
Run from CSCI1470_FINAL/ root directory.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import sys
import os
import json

# ── Setup paths ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MambaVision'))

from models.mamba_vision import mamba_vision_T


# ── Gaussian noise transform ───────────────────────────────────────────────
class AddGaussianNoise:
    """Add Gaussian noise to a tensor after normalization."""
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std

    def __repr__(self):
        return f"AddGaussianNoise(std={self.std})"


# ── Load checkpoint ────────────────────────────────────────────────────────
def load_model(checkpoint_path, num_classes=10):
    print(f"Loading checkpoint: {checkpoint_path}")
    model = mamba_vision_T(num_classes=num_classes)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Strip EMA prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded. Missing keys: {len(msg.missing_keys)}, "
          f"Unexpected: {len(msg.unexpected_keys)}")

    model = model.cuda().eval()
    return model


# ── Evaluate with a given transform ───────────────────────────────────────
def evaluate(model, data_dir, transform, batch_size=128):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    correct_top1 = correct_top5 = total = 0

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for images, labels in loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)

                # Top-1
                _, predicted = outputs.max(1)
                correct_top1 += predicted.eq(labels).sum().item()

                # Top-5
                _, top5_pred = outputs.topk(5, dim=1)
                correct_top5 += top5_pred.eq(
                    labels.view(-1, 1).expand_as(top5_pred)
                ).any(dim=1).sum().item()

                total += labels.size(0)

    return 100.0 * correct_top1 / total, 100.0 * correct_top5 / total


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CHECKPOINT = "output/20260427-094600-mamba_vision_T-224/checkpoint-225.pth.tar"
    DATA_DIR = "STL-10/imagefolder/val"

    # Noise levels to test
    # std=0.0 is clean baseline, then increasing severity
    noise_levels = [
        ("Clean (no noise)",      0.00),
        ("Very mild (std=0.05)",  0.05),
        ("Mild (std=0.10)",       0.10),
        ("Moderate (std=0.20)",   0.20),
        ("Severe (std=0.30)",     0.30),
        ("Very severe (std=0.50)", 0.50),
        ("Extreme (std=1.00)",    1.00),
    ]

    # Base transform without noise
    base_transform = [
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]

    print("=" * 65)
    print("MambaVision-T Gaussian Noise Stress Test")
    print("=" * 65)
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Data: {DATA_DIR}")
    print("=" * 65)

    model = load_model(CHECKPOINT)
    results = []
    baseline_acc = None

    for name, std in noise_levels:
        # Build transform for this noise level
        if std == 0.0:
            transform = T.Compose(base_transform)
        else:
            transform = T.Compose(base_transform + [AddGaussianNoise(std=std)])

        acc1, acc5 = evaluate(model, DATA_DIR, transform)
        drop = (acc1 - baseline_acc) if baseline_acc is not None else 0.0

        if baseline_acc is None:
            baseline_acc = acc1

        results.append({
            "condition": name,
            "noise_std": std,
            "top1": round(acc1, 3),
            "top5": round(acc5, 3),
            "drop": round(drop, 3),
        })

        print(f"{name:<28}  Acc@1: {acc1:6.3f}%  "
              f"Acc@5: {acc5:6.3f}%  Drop: {drop:+7.3f}%")

    # ── Summary table ──────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"{'Condition':<28} {'Acc@1':>8} {'Acc@5':>8} {'Drop':>10}")
    print("=" * 65)
    for r in results:
        print(f"{r['condition']:<28} {r['top1']:>7.3f}% "
              f"{r['top5']:>7.3f}% {r['drop']:>+10.3f}%")
    print("=" * 65)

    # ── Save results ───────────────────────────────────────────────────────
    with open("stress_test_noise_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to stress_test_noise_results.json")

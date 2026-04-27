#!/usr/bin/env python3
"""
Stress test: evaluate MambaVision-T on extreme aspect ratios.
Compares accuracy at standard resolution vs distorted inputs.
"""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import sys
import os

sys.path.insert(0, 'MambaVision')
from models.mamba_vision import mamba_vision_T

# ── Load your trained checkpoint ──────────────────────────────────────────
def load_model(checkpoint_path, num_classes=10):
    model = mamba_vision_T(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # handle different checkpoint formats
    state_dict = checkpoint.get('state_dict',
                  checkpoint.get('model',
                  checkpoint))
    # strip 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()
    return model

# ── Evaluate at a given resolution/aspect ratio ───────────────────────────
def evaluate(model, data_dir, transform, batch_size=64):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
    correct = total = 0
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for images, labels in loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
    return 100.0 * correct / total

# ── Define stress test conditions ─────────────────────────────────────────
def make_transform(height, width):
    return transforms.Compose([
        transforms.Resize((height, width)),  # force exact size, ignoring aspect ratio
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

stress_conditions = [
    # name,               height, width
    ("Standard (224x224)",      224,  224),   # baseline
    ("Small (112x112)",         112,  112),   # half resolution
    ("Large (336x336)",         336,  336),   # 1.5x resolution
    ("Very large (448x448)",    448,  448),   # 2x resolution
    ("Wide (112x336)",          112,  336),   # 1:3 aspect ratio
    ("Tall (336x112)",          336,  112),   # 3:1 aspect ratio
    ("Very wide (56x224)",       56,  224),   # 1:4 aspect ratio
    ("Very tall (224x56)",      224,   56),   # 4:1 aspect ratio
    ("Tiny (64x64)",             64,   64),   # very small
    ("Huge (512x512)",          512,  512),   # very large
]

# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Point to your best checkpoint
    CHECKPOINT = "output/train/mambavision_tiny_1k/20260420-215025-mamba_vision_T_nobypass-224/checkpoint-290.pth.tar"
    DATA_DIR = "STL-10/imagefolder/val"

    print("Loading model...")
    model = load_model(CHECKPOINT)
    print("Model loaded.")

    results = []
    for name, h, w in stress_conditions:
        transform = make_transform(h, w)
        acc = evaluate(model, DATA_DIR, transform)
        drop = acc - results[0][1] if results else 0.0
        results.append((name, acc, drop))
        print(f"{name:<30} Acc@1: {acc:.3f}%  Drop: {drop:+.3f}%")

    # Summary table
    print("\n" + "="*65)
    print(f"{'Condition':<30} {'Acc@1':>8} {'Drop':>10}")
    print("="*65)
    baseline = results[0][1]
    for name, acc, _ in results:
        drop = acc - baseline
        print(f"{name:<30} {acc:>7.3f}%  {drop:>+10.3f}%")
    print("="*65)

    # Save CSV
    import csv
    with open("stress_test_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "height", "width", "accuracy", "drop_from_baseline"])
        for (name, h, w), (_, acc, drop) in zip(stress_conditions, results):
            writer.writerow([name, h, w, acc, acc - baseline])
    print("Saved to stress_test_results.csv")
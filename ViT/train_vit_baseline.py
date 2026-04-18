import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import argparse
import time
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data/stl10')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', default='./output/vit_baseline')
    parser.add_argument('--num-classes', type=int, default=10)
    return parser.parse_args()

def main():
    args = get_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # --- Model ---
    # vit_tiny_patch16_224 is the standard ViT-T
    # pretrained=False to match your MambaVision from-scratch setup
    model = timm.create_model(
        'vit_tiny_patch16_224',
        pretrained=False,
        num_classes=args.num_classes
    )
    model = model.cuda()

    # --- Data (identical transforms to your MambaVision run) ---
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Use ImageFolder to match MambaVision pipeline
    train_dataset = ImageFolder(
        root=os.path.join(args.data_dir, 'train'),
        transform=train_transform
    )
    val_dataset = ImageFolder(
        root=os.path.join(args.data_dir, 'val'),
        transform=val_transform
    )

    # Print dataset sizes and class balance
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")
    from collections import Counter
    train_labels = [label for _, label in train_dataset.samples]
    val_labels = [label for _, label in val_dataset.samples]
    print("Train class counts:", Counter(train_labels))
    print("Val class counts:", Counter(val_labels))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # --- Optimizer (match MambaVision setup) ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()  # fp16 training

    best_acc = 0.0

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # --- Validate ---
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.cuda(), labels.cuda()
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{args.epochs} — Val Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{args.output}_best.pth")
            print(f"  New best: {best_acc:.2f}%")

    print(f"\nFinal best accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
"""
Convert CIFAR-10 pickle files into ImageFolder structure:
  cifar-10/imagefolder/train/<class_name>/<idx>.png
  cifar-10/imagefolder/val/<class_name>/<idx>.png

Augments the training set 5x (50000 → 250000 images) using:
  - horizontal flip
  - 90/180/270 degree rotations

Run once after the CIFAR-10 data is downloaded:
  python cifar10_dataset.py
"""
import os
import pickle
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def resolve_cifar10_dir():
    candidates = [
        os.path.join(REPO_ROOT, "cifar-10-batches-py"),
        os.path.join(os.getcwd(), "cifar-10-batches-py"),
        os.path.join(SCRIPT_DIR, "cifar-10-batches-py"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "Could not find 'cifar-10-batches-py'. Looked in:\n  " + "\n  ".join(candidates)
    )


CIFAR10_DIR = resolve_cifar10_dir()
OUT_DIR = os.path.join(REPO_ROOT, "cifar-10", "imagefolder")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

TRAIN_BATCHES = [
    "data_batch_1", "data_batch_2", "data_batch_3",
    "data_batch_4", "data_batch_5"
]
TEST_BATCH = "test_batch"


def unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_split(batch_files):
    all_data   = []
    all_labels = []
    for fname in batch_files:
        batch_path = os.path.join(CIFAR10_DIR, fname)
        d = unpickle(batch_path)
        # data is (N, 3072) uint8, labels is a list of ints
        all_data.append(d[b"data"])
        all_labels.extend(d[b"labels"])
    data = np.concatenate(all_data, axis=0)          # (N, 3072)
    data = np.transpose(data.reshape(-1, 3, 32, 32), (0, 2, 3, 1))  # (N, 32, 32, 3)
    return data, all_labels


def augment(img):
    """Return 4 additional augmented versions of a PIL image."""
    return [
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.rotate(90),
        img.rotate(180),
        img.rotate(270),
    ]


def save_split(images, labels, split, augment_train=False):
    split_dir = os.path.join(OUT_DIR, split)
    if os.path.exists(split_dir):
        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            if os.path.isdir(class_path):
                for fname in os.listdir(class_path):
                    os.remove(os.path.join(class_path, fname))
    else:
        os.makedirs(split_dir, exist_ok=True)

    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(OUT_DIR, split, class_name), exist_ok=True)

    count = 0
    for idx, (img_arr, label) in enumerate(zip(images, labels)):
        class_name = CLASS_NAMES[label]  # CIFAR-10 labels are 0-indexed
        img = Image.fromarray(img_arr).resize((256, 256), Image.BICUBIC)

        img.save(os.path.join(OUT_DIR, split, class_name, f"{idx:05d}_orig.png"))
        count += 1

        if augment_train:
            for aug_idx, aug_img in enumerate(augment(img)):
                aug_img.save(os.path.join(OUT_DIR, split, class_name, f"{idx:05d}_aug{aug_idx}.png"))
                count += 1

    print(f"  Saved {count} images to {OUT_DIR}/{split}/")


print("Converting CIFAR-10 to ImageFolder format...")
print(f"Using source: {CIFAR10_DIR}")
print(f"Saving to:   {OUT_DIR}")

print("Processing train split (5x augmentation → 250,000 images)...")
train_images, train_labels = load_split(TRAIN_BATCHES)
save_split(train_images, train_labels, "train", augment_train=True)

print("Processing test split (no augmentation)...")
test_images, test_labels = load_split([TEST_BATCH])
save_split(test_images, test_labels, "val", augment_train=False)

print(f"Done. ImageFolder dataset ready at: {OUT_DIR}")

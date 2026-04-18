"""
Convert STL-10 binary files into ImageFolder structure:
  STL-10/imagefolder/train/<class_name>/<idx>.png
  STL-10/imagefolder/test/<class_name>/<idx>.png

Augments the training set 5x (5000 → 25000 images) using:
  - horizontal flip
  - 90/180/270 degree rotations

Run once on the login node after download_stl10.py:
  python prepare_stl10_imagefolder.py
"""
import os
import numpy as np
from PIL import Image

STL10_DIR = "STL-10/stl10_binary"
OUT_DIR = "STL-10/imagefolder"

CLASS_NAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]


def load_images(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    # Each image: 3 * 96 * 96 = 27648 bytes
    data = data.reshape(-1, 3, 96, 96)
    # Convert from CHW to HWC
    return data.transpose(0, 2, 3, 1)


def load_labels(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)


def augment(img):
    """Return 4 additional augmented versions of a PIL image."""
    return [
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.rotate(90),
        img.rotate(180),
        img.rotate(270),
    ]


def save_split(images, labels, split, augment_train=False):

    # Clear split directory before saving (prevents duplicates)
    split_dir = os.path.join(OUT_DIR, split)
    if os.path.exists(split_dir):
        # Remove all files and subdirectories
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
        class_name = CLASS_NAMES[label - 1]  # labels are 1-indexed
        img = Image.fromarray(img_arr).resize((256, 256), Image.BICUBIC)

        # Save original
        img.save(os.path.join(OUT_DIR, split, class_name, f"{idx:05d}_orig.png"))
        count += 1

        # Save augmented copies for training set only
        if augment_train:
            for aug_idx, aug_img in enumerate(augment(img)):
                aug_img.save(os.path.join(OUT_DIR, split, class_name, f"{idx:05d}_aug{aug_idx}.png"))
                count += 1

    print(f"  Saved {count} images to {OUT_DIR}/{split}/")


print("Converting STL-10 to ImageFolder format...")

print("Processing train split (5x augmentation → 25,000 images)...")
train_images = load_images(os.path.join(STL10_DIR, "train_X.bin"))
train_labels = load_labels(os.path.join(STL10_DIR, "train_y.bin"))
save_split(train_images, train_labels, "train", augment_train=True)

print("Processing test split (no augmentation)...")
test_images = load_images(os.path.join(STL10_DIR, "test_X.bin"))
test_labels = load_labels(os.path.join(STL10_DIR, "test_y.bin"))
save_split(test_images, test_labels, "val", augment_train=False)

print(f"Done. ImageFolder dataset ready at: {OUT_DIR}")

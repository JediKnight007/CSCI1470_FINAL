"""
Convert STL-10 binary files into ImageFolder structure:
  STL-10/imagefolder/train/<class_name>/<idx>.png
  STL-10/imagefolder/test/<class_name>/<idx>.png

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


def save_split(images, labels, split):
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(OUT_DIR, split, class_name), exist_ok=True)

    for idx, (img_arr, label) in enumerate(zip(images, labels)):
        class_name = CLASS_NAMES[label - 1]  # labels are 1-indexed
        out_path = os.path.join(OUT_DIR, split, class_name, f"{idx:05d}.png")
        Image.fromarray(img_arr).save(out_path)

    print(f"  Saved {len(images)} images to {OUT_DIR}/{split}/")


print("Converting STL-10 to ImageFolder format...")

print("Processing train split...")
train_images = load_images(os.path.join(STL10_DIR, "train_X.bin"))
train_labels = load_labels(os.path.join(STL10_DIR, "train_y.bin"))
save_split(train_images, train_labels, "train")

print("Processing test split...")
test_images = load_images(os.path.join(STL10_DIR, "test_X.bin"))
test_labels = load_labels(os.path.join(STL10_DIR, "test_y.bin"))
save_split(test_images, test_labels, "val")

print(f"Done. ImageFolder dataset ready at: {OUT_DIR}")

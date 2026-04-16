import os
import requests
import tarfile

STL10_URL = "https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
DATA_DIR = "STL-10"
TAR_PATH = os.path.join(DATA_DIR, "stl10_binary.tar.gz")

os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading STL-10 dataset...")
with requests.get(STL10_URL, stream=True) as r:
    r.raise_for_status()
    with open(TAR_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
print("Download complete.")

print("Extracting dataset...")
with tarfile.open(TAR_PATH, "r:gz") as tar:
    tar.extractall(DATA_DIR)
print("Extraction complete.")

# Remove the tar.gz file to save space
os.remove(TAR_PATH)
print("Cleanup complete. Dataset is ready in 'STL-10'.")

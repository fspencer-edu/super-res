import os
import random
import shutil

DATASET_ROOT = "/mnt/ml/superres/datasets"
RAW_DIR = os.path.join(DATASET_ROOT, "raw")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

images = [
    f for f in os.listdir(RAW_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
]

if not images:
    raise FileNotFoundError(f"No images found in {RAW_DIR}")

random.seed(42)
random.shuffle(images)

total = len(images)
train_n = int(total * 0.8)
val_n = int(total * 0.1)
test_n = total - train_n - val_n

train_images = images[:train_n]
val_images = images[train_n:train_n + val_n]
test_images = images[train_n + val_n:]

def copy_files(file_list, target_dir):
    for name in file_list:
        src = os.path.join(RAW_DIR, name)
        dst = os.path.join(target_dir, name)
        shutil.copy2(src, dst)

copy_files(train_images, TRAIN_DIR)
copy_files(val_images, VAL_DIR)
copy_files(test_images, TEST_DIR)

print("Split complete")
print("raw :", total)
print("train:", len(train_images))
print("val  :", len(val_images))
print("test :", len(test_images))
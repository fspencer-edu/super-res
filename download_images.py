import os
import json
from datetime import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.utils import save_img

# --------------------------------------------------
# Project paths
# --------------------------------------------------
NAS_ROOT = "/mnt/ml"
PROJECT_NAME = "superres"
PROJECT_ROOT = os.path.join(NAS_ROOT, PROJECT_NAME)

DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")
RAW_DIR = os.path.join(DATASETS_ROOT, "raw")
TRAIN_DIR = os.path.join(DATASETS_ROOT, "train")
VAL_DIR = os.path.join(DATASETS_ROOT, "val")
TEST_DIR = os.path.join(DATASETS_ROOT, "test")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
IMAGES_DIR = os.path.join(OUTPUTS_DIR, "images")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")

for path in [
    PROJECT_ROOT,
    DATASETS_ROOT,
    RAW_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    MODELS_DIR,
    LOGS_DIR,
    OUTPUTS_DIR,
    IMAGES_DIR,
    PREDICTIONS_DIR,
]:
    os.makedirs(path, exist_ok=True)

# --------------------------------------------------
# Config
# --------------------------------------------------
DATASET_NAME = "tf_flowers"   # try: "cats_vs_dogs", "food101"
MAX_IMAGES = 3000
IMAGE_SIZE = (256, 256)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# --------------------------------------------------
# Download dataset
# --------------------------------------------------
print(f"Loading dataset: {DATASET_NAME}")

ds = tfds.load(DATASET_NAME, split="train", as_supervised=True)

saved = 0
for i, (image, label) in enumerate(ds):
    if saved >= MAX_IMAGES:
        break

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)

    filename = os.path.join(RAW_DIR, f"img_{i:06d}.png")
    save_img(filename, image.numpy())
    saved += 1

    if saved % 250 == 0:
        print(f"Saved {saved} images")

print(f"Saved {saved} raw images to {RAW_DIR}")

# --------------------------------------------------
# Split into train/val/test
# --------------------------------------------------
all_files = sorted([
    os.path.join(RAW_DIR, f)
    for f in os.listdir(RAW_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
])

total = len(all_files)
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_files = all_files[:train_end]
val_files = all_files[train_end:val_end]
test_files = all_files[val_end:]

def copy_files(file_list, target_dir):
    for src in file_list:
        dst = os.path.join(target_dir, os.path.basename(src))
        tf.io.gfile.copy(src, dst, overwrite=True)

copy_files(train_files, TRAIN_DIR)
copy_files(val_files, VAL_DIR)
copy_files(test_files, TEST_DIR)

print(f"Train: {len(train_files)}")
print(f"Val:   {len(val_files)}")
print(f"Test:  {len(test_files)}")

# --------------------------------------------------
# Save metadata
# --------------------------------------------------
metadata = {
    "run_id": RUN_ID,
    "dataset_name": DATASET_NAME,
    "max_images": MAX_IMAGES,
    "image_size": IMAGE_SIZE,
    "project_root": PROJECT_ROOT,
    "raw_dir": RAW_DIR,
    "train_dir": TRAIN_DIR,
    "val_dir": VAL_DIR,
    "test_dir": TEST_DIR,
    "train_count": len(train_files),
    "val_count": len(val_files),
    "test_count": len(test_files),
}

with open(os.path.join(LOGS_DIR, f"dataset_info_{RUN_ID}.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("Done.")
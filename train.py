import os
import json
import time
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------------------------------
# Project paths
# --------------------------------------------------
NAS_ROOT = "/mnt/ml"
PROJECT_NAME = "superres"
PROJECT_ROOT = os.path.join(NAS_ROOT, PROJECT_NAME)

TRAIN_DIR = os.path.join(PROJECT_ROOT, "datasets", "train")
VAL_DIR = os.path.join(PROJECT_ROOT, "datasets", "val")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
IMAGES_DIR = os.path.join(OUTPUTS_DIR, "images")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")

for path in [MODELS_DIR, LOGS_DIR, OUTPUTS_DIR, IMAGES_DIR, PREDICTIONS_DIR]:
    os.makedirs(path, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(LOGS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "superres_model.keras")
BEST_MODEL_PATH = os.path.join(RUN_DIR, "best_superres_model.keras")

# --------------------------------------------------
# Config
# --------------------------------------------------
LR_SIZE = 64
HR_SIZE = 128
CHANNELS = 3
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
SEED = 42

# --------------------------------------------------
# File loading
# --------------------------------------------------
def list_image_files(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ])

train_files = list_image_files(TRAIN_DIR)
val_files = list_image_files(VAL_DIR)

if not train_files:
    raise FileNotFoundError(f"No training images found in {TRAIN_DIR}")
if not val_files:
    raise FileNotFoundError(f"No validation images found in {VAL_DIR}")

print(f"Train images: {len(train_files)}")
print(f"Val images:   {len(val_files)}")

# --------------------------------------------------
# Augmentation
# --------------------------------------------------
def augment_image(img):
    # img should already be float32 in [0, 1]

    img = tf.image.random_flip_left_right(img)

    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.clip_by_value(img, 0.0, 1.0)

    # random crop, then resize back to HR size
    crop_size = tf.random.uniform([], minval=HR_SIZE - 24, maxval=HR_SIZE + 1, dtype=tf.int32)
    img = tf.image.random_crop(img, size=[crop_size, crop_size, CHANNELS])
    img = tf.image.resize(img, [HR_SIZE, HR_SIZE], method="bicubic")

    # jpeg quality works best on uint8
    img_uint8 = tf.image.convert_image_dtype(img, tf.uint8, saturate=True)
    img_uint8 = tf.image.random_jpeg_quality(img_uint8, 70, 100)
    img = tf.image.convert_image_dtype(img_uint8, tf.float32)

    return img

# --------------------------------------------------
# Dataset preprocessing
# --------------------------------------------------
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [HR_SIZE, HR_SIZE], method="bicubic")
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def make_pair(path, training=False):
    hr = load_image(path)

    if training:
        hr = augment_image(hr)

    lr = tf.image.resize(hr, [LR_SIZE, LR_SIZE], method="area")
    return lr, hr

def make_dataset(files, training=False):
    ds = tf.data.Dataset.from_tensor_slices(files)

    if training:
        ds = ds.shuffle(len(files), seed=SEED)

    ds = ds.map(
        lambda path: make_pair(path, training=training),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_files, training=True)
val_ds = make_dataset(val_files, training=False)

# --------------------------------------------------
# Model
# --------------------------------------------------
def residual_block(x, filters):
    skip = x
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.Add()([x, skip])
    return x

def upsample_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.ReLU()(x)
    return x

def build_superres_model():
    inputs = keras.Input(shape=(LR_SIZE, LR_SIZE, CHANNELS))

    x = layers.Conv2D(64, 5, padding="same")(inputs)
    x = layers.ReLU()(x)

    for _ in range(4):
        x = residual_block(x, 64)

    x = upsample_block(x, 64)
    outputs = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)

    return keras.Model(inputs, outputs, name="superres_model")

model = build_superres_model()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mae",
    metrics=[keras.metrics.MeanSquaredError(name="mse")]
)

model.summary()

# --------------------------------------------------
# Sample image callback
# --------------------------------------------------
class SaveSampleImagesCallback(keras.callbacks.Callback):
    def __init__(self, val_ds, output_dir):
        super().__init__()
        self.val_ds = val_ds
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        batch = next(iter(self.val_ds.take(1)))
        self.sample_lr, self.sample_hr = batch

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.sample_lr[:3], verbose=0)

        for i in range(min(3, preds.shape[0])):
            keras.utils.save_img(
                os.path.join(self.output_dir, f"epoch_{epoch+1:03d}_sample_{i}_lr.png"),
                self.sample_lr[i]
            )
            keras.utils.save_img(
                os.path.join(self.output_dir, f"epoch_{epoch+1:03d}_sample_{i}_pred.png"),
                preds[i]
            )
            keras.utils.save_img(
                os.path.join(self.output_dir, f"epoch_{epoch+1:03d}_sample_{i}_hr.png"),
                self.sample_hr[i]
            )

# --------------------------------------------------
# Callbacks
# --------------------------------------------------
csv_logger = keras.callbacks.CSVLogger(
    os.path.join(RUN_DIR, "history.csv"),
    append=False
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

tensorboard = keras.callbacks.TensorBoard(
    log_dir=os.path.join(RUN_DIR, "tensorboard"),
    histogram_freq=1
)

sample_callback = SaveSampleImagesCallback(
    val_ds=val_ds,
    output_dir=os.path.join(IMAGES_DIR, RUN_ID)
)

# --------------------------------------------------
# Train
# --------------------------------------------------
start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        csv_logger,
        checkpoint,
        early_stopping,
        reduce_lr,
        tensorboard,
        sample_callback,
    ],
    verbose=1
)

training_seconds = time.time() - start_time

model.save(MODEL_PATH)

# --------------------------------------------------
# Save history and run info
# --------------------------------------------------
with open(os.path.join(RUN_DIR, "history.json"), "w") as f:
    json.dump(history.history, f, indent=2)

run_info = {
    "run_id": RUN_ID,
    "project_root": PROJECT_ROOT,
    "train_dir": TRAIN_DIR,
    "val_dir": VAL_DIR,
    "model_path": MODEL_PATH,
    "best_model_path": BEST_MODEL_PATH,
    "lr_size": LR_SIZE,
    "hr_size": HR_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs_requested": EPOCHS,
    "epochs_completed": len(history.history["loss"]),
    "learning_rate": LEARNING_RATE,
    "train_images": len(train_files),
    "val_images": len(val_files),
    "training_seconds": training_seconds,
    "final_train_loss": float(history.history["loss"][-1]),
    "final_val_loss": float(history.history["val_loss"][-1]),
    "final_train_mse": float(history.history["mse"][-1]),
    "final_val_mse": float(history.history["val_mse"][-1]),
    "augmentations": [
        "tf.image.random_flip_left_right",
        "tf.image.random_brightness",
        "tf.image.random_crop",
        "tf.image.random_jpeg_quality",
    ],
}

with open(os.path.join(RUN_DIR, "run_info.json"), "w") as f:
    json.dump(run_info, f, indent=2)

print(f"\nSaved final model to: {MODEL_PATH}")
print(f"Saved best model to: {BEST_MODEL_PATH}")
print(f"Run logs saved to: {RUN_DIR}")
print(f"Sample images saved to: {os.path.join(IMAGES_DIR, RUN_ID)}")
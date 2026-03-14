import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array, save_img

NAS_ROOT = "/mnt/ml"
PROJECT_NAME = "superres"
PROJECT_ROOT = os.path.join(NAS_ROOT, PROJECT_NAME)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "superres_model.keras")
INPUT_IMAGE = os.path.join(PROJECT_ROOT, "datasets", "test", "test.jpg")
OUTPUT_IMAGE = os.path.join(PROJECT_ROOT, "outputs", "predictions", "test_upscaled.png")

LR_SIZE = 64

model = keras.models.load_model(MODEL_PATH)

img = load_img(INPUT_IMAGE, target_size=(LR_SIZE, LR_SIZE))
x = img_to_array(img).astype("float32") / 255.0
x = np.expand_dims(x, axis=0)

pred = model.predict(x, verbose=0)[0]
pred = np.clip(pred, 0.0, 1.0)

save_img(OUTPUT_IMAGE, pred)
print("Saved:", OUTPUT_IMAGE)
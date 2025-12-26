from __future__ import annotations
import os, numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

DEFAULT_MODEL_PATH = os.path.join("models", "waste_binary.keras")
IMG_SIZE = (224, 224)
_model = None

def load_model(model_path: str = DEFAULT_MODEL_PATH):
    global _model
    if _model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Train dulu: python backend/train.py --data_dir <DATASET_DIR>"
            )
        _model = tf.keras.models.load_model(model_path)
    return _model

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, 0)

def predict_binary(pil_img: Image.Image, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    model = load_model(model_path)
    x = preprocess_image(pil_img)
    prob = float(model.predict(x, verbose=0).reshape(-1)[0])
    label = "organik" if prob >= 0.5 else "non-organik"
    conf = prob if label == "organik" else (1.0 - prob)
    return {"label": label, "prob_organik": prob, "confidence": float(conf)}

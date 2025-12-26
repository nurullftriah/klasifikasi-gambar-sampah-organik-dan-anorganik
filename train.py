from __future__ import annotations
import os, argparse
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from .labels import map_to_binary

IMG_SIZE = (224, 224)

def list_images_with_labels(data_dir: str):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP"]
    fps = []
    labs = []
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not classes:
        raise ValueError(f"No class folders found in: {data_dir}")
    for cls in sorted(classes):
        cls_dir = os.path.join(data_dir, cls)
        files = []
        for e in exts:
            files += glob(os.path.join(cls_dir, e))
        for fp in files:
            fps.append(fp)
            labs.append(map_to_binary(cls))
    return np.array(fps), np.array(labs, dtype=np.int32)

def decode_and_resize(path, label):
    b = tf.io.read_file(path)
    img = tf.image.decode_image(b, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, tf.cast(label, tf.float32)

def build_dataset(fps, labs, batch, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((fps, labs))
    if training:
        ds = ds.shuffle(min(5000, len(fps)), reshuffle_each_iteration=True)
    ds = ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

def build_model(dropout=0.4):
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    base.trainable = False
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs), base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default=os.path.join("models", "waste_binary.keras"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--epochs_ft", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    X, y = list_images_with_labels(args.data_dir)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=args.seed, stratify=y_tmp
    )

    train_ds = build_dataset(X_train, y_train, args.batch, True)
    val_ds = build_dataset(X_val, y_val, args.batch, False)
    test_ds = build_dataset(X_test, y_test, args.batch, False)

    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    model, base = build_model(0.4)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
        ],
    )

    cbs = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=cbs,
    )

    if args.epochs_ft > 0:
        base.trainable = True
        for layer in base.layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC(),
            ],
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs_ft,
            class_weight=class_weight,
            callbacks=cbs,
        )

    model.evaluate(test_ds)
    model.save(args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()

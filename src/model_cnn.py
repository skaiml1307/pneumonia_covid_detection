"""
Baseline CNN for chest X-rays.
Trains from scratch (no pretrained weights).
Gave lower accuracy than expected, so I switched to ResNet50 transfer learning.
"""

import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
import random
import os


# stability: fix random seeds
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "data"

def preprocess_image(img_path):
    img = cv2.imread(img_path)  # RGB read
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return img

def build_cnn(input_shape=(224,224,3), num_classes=3):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    cnn_model = build_cnn()
    cnn_model.summary()

    # load datasets (RGB mode to avoid grayscale BMP errors)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/train", image_size=IMG_SIZE, color_mode="rgb", batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/val", image_size=IMG_SIZE, color_mode="rgb", batch_size=BATCH_SIZE
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/test", image_size=IMG_SIZE, color_mode="rgb", batch_size=BATCH_SIZE, shuffle=False
    )

    # normalize pixel values
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # train CNN
    history = cnn_model.fit(train_ds, validation_data=val_ds, epochs=15)

    # evaluate
    test_loss, test_acc = cnn_model.evaluate(test_ds)
    print(f"Baseline CNN test accuracy: {test_acc:.4f}")

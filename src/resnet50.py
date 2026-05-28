import tensorflow as tf
from keras import layers, models, applications
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import random


# stability: fix random seeds
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

DATA_DIR = "data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# data augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


# load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/train", image_size=IMG_SIZE, color_mode="rgb", batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/val", image_size=IMG_SIZE, color_mode="rgb", batch_size=BATCH_SIZE
)
test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/test", image_size=IMG_SIZE, color_mode="rgb", batch_size=BATCH_SIZE, shuffle=False
)
class_names = test_ds_raw.class_names  

# normalize
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds_raw.map(lambda x, y: (normalization_layer(x), y))


# base model: ResNet50
base_model = applications.ResNet50(include_top=False, input_shape=(224,224,3), weights="imagenet")
base_model.trainable = False

# build model
model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),  # tuned dropout
    layers.Dense(len(class_names), activation="softmax")
])

# compile
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# train frozen base
history = model.fit(train_ds, validation_data=val_ds, epochs=15)


# gradual unfreezing last 150 layers
for layer in base_model.layers[-150:]:
    layer.trainable = True


# callbacks for stability & accuracy
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "results/best_resnet50.keras",   # switched to .keras format
    save_best_only=True,
    monitor="val_accuracy"
)

# cosine learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-4,
    first_decay_steps=1000,
    t_mul=2.0,
    m_mul=1.0,
    alpha=1e-6
)

# recompile with cosine LR
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# fine tune
history_ft = model.fit(train_ds, validation_data=val_ds, epochs=30,
                       callbacks=[early_stop, checkpoint])


# load best weights 
model = tf.keras.models.load_model("results/best_resnet50.keras")

# Evaluate
test_loss, test_acc = model.evaluate(test_ds)
print(f"ResNet50 transfer learning final test accuracy: {test_acc:.4f}")


# ensure results folder exists
os.makedirs("results", exist_ok=True)


# accuracy & loss figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))

# accuracy curves
ax1.plot(history.history['accuracy'], label='Train Accuracy (Frozen)')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy (Frozen)')
ax1.plot(history_ft.history['accuracy'], label='Train Accuracy (FT)')
ax1.plot(history_ft.history['val_accuracy'], label='Val Accuracy (FT)')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()

# loss curves
ax2.plot(history.history['loss'], label='Train Loss (Frozen)')
ax2.plot(history.history['val_loss'], label='Val Loss (Frozen)')
ax2.plot(history_ft.history['loss'], label='Train Loss (FT)')
ax2.plot(history_ft.history['val_loss'], label='Val Loss (FT)')
ax2.set_title('Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.savefig("results/training_curves.png")
plt.close()
sys.exit(0)

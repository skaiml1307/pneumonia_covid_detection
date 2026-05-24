import tensorflow as tf
from keras import layers, models, applications
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DATA_DIR = "data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(factor=0.1)
])

# load datasets as RGB
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


normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds_raw.map(lambda x, y: (normalization_layer(x), y))


base_model = applications.ResNet50(include_top=False, input_shape=(224,224,3), weights="imagenet")
base_model.trainable = False


model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


class_weights = {0: 1.0, 1: 2.0, 2: 2.0}


history = model.fit(train_ds, validation_data=val_ds, epochs=10, class_weight=class_weights)


base_model.trainable = True
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history_ft = model.fit(train_ds, validation_data=val_ds, epochs=15,
                       class_weight=class_weights, callbacks=[lr_schedule])


test_loss, test_acc = model.evaluate(test_ds)
print(f"ResNet50 transfer learning test accuracy: {test_acc:.4f}")


y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_pred = np.argmax(model.predict(test_ds), axis=1)
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# accuracy curves
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history_ft.history['accuracy'], label='Train Accuracy (FT)')
plt.plot(history_ft.history['val_accuracy'], label='Val Accuracy (FT)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()

# loss curves
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.plot(history_ft.history['loss'], label='Train Loss (FT)')
plt.plot(history_ft.history['val_loss'], label='Val Loss (FT)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

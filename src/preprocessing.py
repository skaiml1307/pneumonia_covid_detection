import cv2
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator

DATA_DIR = "data"
IMG_SIZE = (224, 224)  

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
    img = cv2.resize(img, IMG_SIZE)                   
    img = img.astype("float32") / 255.0               
    img = np.expand_dims(img, axis=-1)                
    return img

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator()   
test_datagen = ImageDataGenerator()  

print("Preprocessing pipeline ready: augmentation generators created.")

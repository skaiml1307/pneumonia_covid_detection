from keras import layers, models

def build_cnn(input_shape=(224,224,1), num_classes=3):
    """
    Custom CNN for grayscale chest X-rays.
    Trains from scratch instead of using pretrained backbones.
    """
    model = models.Sequential([
        
        layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
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
    #example
    cnn_model = build_cnn()
    cnn_model.summary()

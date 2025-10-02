import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping

DATASET_PATH = "data_spectrograms"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    color_mode="grayscale"
)

val_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    color_mode="grayscale"
)

# MobileNetV2 expects 3 channels, convert grayscale to 3 channels
def preprocess_input_grayscale(x):
    return tf.image.grayscale_to_rgb(x)

# Base model
base_model = MobileNetV2(input_shape=(128,128,3),
                         include_top=False,
                         weights="imagenet")
base_model.trainable = False

# Full model
inputs = layers.Input(shape=(128,128,1))
x = layers.Lambda(preprocess_input_grayscale)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(train_gen.num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Early stopping
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/music_genre_cnn.keras")
print("✅ Training complete! Model saved at models/music_genre_cnn.keras")

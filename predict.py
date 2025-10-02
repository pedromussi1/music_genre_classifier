import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

MODEL_PATH = "models/music_genre_cnn.keras"
SEGMENT_DURATION = 5  # seconds
IMG_SIZE = (128, 128)

# Recreate the same function used in train.py
def preprocess_input_grayscale(x):
    return tf.image.grayscale_to_rgb(x)

# Load model with custom_objects
model = load_model(MODEL_PATH,
                   custom_objects={"preprocess_input_grayscale": preprocess_input_grayscale})
print("✅ Model loaded!")

# Genre labels (must match folder names in data_spectrograms)
GENRES = sorted(os.listdir("data_spectrograms"))
print(f"Genres: {GENRES}")

# Convert audio segment to image array
def segment_to_image_array(segment, sr):
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3,3))
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("temp.png", bbox_inches='tight', pad_inches=0, dpi=128)
    plt.close()

    img = image.load_img("temp.png", target_size=IMG_SIZE, color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict genre for an audio file
def predict_genre(audio_path, show_plot=True):
    y, sr = librosa.load(audio_path)
    total_samples = len(y)
    samples_per_segment = SEGMENT_DURATION * sr
    num_segments = max(1, total_samples // samples_per_segment)

    predictions = []

    for i in range(num_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = y[start:end]
        if len(segment) < 1000:  # skip too short segments
            continue

        img_array = segment_to_image_array(segment, sr)
        pred = model.predict(img_array, verbose=0)
        predictions.append(pred[0])

    if not predictions:
        return None, 0.0

    avg_pred = np.mean(predictions, axis=0)
    genre_idx = np.argmax(avg_pred)
    genre = GENRES[genre_idx]
    confidence = float(avg_pred[genre_idx])

    # Optional: Show confidence distribution plot
    if show_plot:
        plt.figure(figsize=(10,5))
        plt.bar(GENRES, avg_pred)
        plt.xticks(rotation=45)
        plt.title(f"Predicted Genre: {genre} ({confidence*100:.2f}% confidence)")
        plt.ylabel("Probability")
        plt.tight_layout()
        plt.show()

    return genre, confidence

# Example usage
audio_file = r"C:\Users\Pedro\Downloads\GTZAN\genres\rock\rock.00000.au"  # change to your file
genre, confidence = predict_genre(audio_file)
if genre:
    print(f"Predicted Genre: {genre} ({confidence*100:.2f}% confidence)")
else:
    print("❌ Could not predict genre (file too short or invalid).")

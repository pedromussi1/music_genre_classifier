import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tempfile
import os
import soundfile as sf

st.set_page_config(page_title="🎵 Music Genre Classifier", layout="wide")

MODEL_PATH = "models/music_genre_cnn.keras"
SEGMENT_DURATION = 5  # seconds
IMG_SIZE = (128, 128)

# Function from train.py
def preprocess_input_grayscale(x):
    return tf.image.grayscale_to_rgb(x)

# Load model
model = load_model(MODEL_PATH,
                   custom_objects={"preprocess_input_grayscale": preprocess_input_grayscale})

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_spectrograms")
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]


# Assign custom colors to genres (used for visualization)
GENRE_COLORS = {
    "blues": "#1f77b4",
    "classical": "#ff7f0e",
    "country": "#2ca02c",
    "disco": "#d62728",
    "hiphop": "#9467bd",
    "jazz": "#8c564b",
    "metal": "#e377c2",
    "pop": "#7f7f7f",
    "reggae": "#bcbd22",
    "rock": "#17becf"
}

st.title("🎵 Music Genre Classifier")
st.write("Upload a music file (AU or WAV) to predict its genre!")

# ------------------- Audio Upload -------------------
uploaded_file = st.file_uploader("Choose an audio file", type=["au", "wav"])

if uploaded_file is not None:
    # Save uploaded file to temp
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    temp_input_path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.read())

    # Convert AU to WAV if needed
    y, sr = librosa.load(temp_input_path, sr=None)
    temp_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    sf.write(temp_wav_path, y, sr)

    # ------------------- Layout Columns -------------------
    col1, col2 = st.columns([1, 2])

    # Column 1: Audio player and waveform
    with col1:
        st.subheader("🎧 Audio Player")
        st.audio(temp_wav_path)

        st.subheader("🔊 Waveform")
        fig_wf, ax_wf = plt.subplots(figsize=(4,2))
        ax_wf.plot(y, color="dodgerblue")
        ax_wf.set_xlabel("Samples")
        ax_wf.set_ylabel("Amplitude")
        ax_wf.set_yticks([])
        ax_wf.set_title("Waveform", fontsize=12)
        st.pyplot(fig_wf)

    # ------------------- Spectrogram Conversion -------------------
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

    # ------------------- Prediction -------------------
    def predict_genre(audio_path):
        y, sr = librosa.load(audio_path)
        total_samples = len(y)
        samples_per_segment = SEGMENT_DURATION * sr
        num_segments = max(1, total_samples // samples_per_segment)

        predictions = []

        for i in range(num_segments):
            start = i * samples_per_segment
            end = start + samples_per_segment
            segment = y[start:end]
            if len(segment) < 1000:
                continue
            img_array = segment_to_image_array(segment, sr)
            pred = model.predict(img_array, verbose=0)
            predictions.append(pred[0])

        if not predictions:
            return None, 0.0, []

        avg_pred = np.mean(predictions, axis=0)
        genre_idx = np.argmax(avg_pred)
        genre = GENRES[genre_idx]
        confidence = float(avg_pred[genre_idx])
        return genre, confidence, avg_pred

    # ------------------- Run Prediction -------------------
    genre, confidence, probs = predict_genre(temp_wav_path)

    # Column 2: Prediction and Probability Bars
    with col2:
        if genre:
            st.markdown(f"## 🎶 Predicted Genre: **{genre}** ({confidence*100:.2f}% confidence)")

            # Sort genres by probability
            sorted_idx = np.argsort(probs)[::-1]
            sorted_genres = [GENRES[i] for i in sorted_idx]
            sorted_probs = probs[sorted_idx]

            st.subheader("Genre Probability Distribution")
            for g, p in zip(sorted_genres, sorted_probs):
                color = GENRE_COLORS.get(g, "#17becf")
                st.markdown(f"**{g.capitalize()}**")
                st.progress(int(p * 100))  # Fix: Convert 0–1 to 0–100
        else:
            st.error("❌ Could not predict genre (file too short or invalid).")

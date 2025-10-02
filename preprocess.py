import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH = r"C:\Users\Pedro\Downloads\GTZAN\genres"   # change this to your extracted path
OUTPUT_PATH = "data_spectrograms"
os.makedirs(OUTPUT_PATH, exist_ok=True)

SEGMENT_DURATION = 5  # seconds per clip

def save_mel_spectrogram(y, sr, output_path):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(3,3))
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=128)
    plt.close()

for genre in os.listdir(DATASET_PATH):
    genre_path = os.path.join(DATASET_PATH, genre)
    if not os.path.isdir(genre_path):
        continue

    output_genre_path = os.path.join(OUTPUT_PATH, genre)
    os.makedirs(output_genre_path, exist_ok=True)

    for file in os.listdir(genre_path):
        if file.endswith(".au"):
            audio_path = os.path.join(genre_path, file)
            y, sr = librosa.load(audio_path)
            total_samples = len(y)
            samples_per_segment = SEGMENT_DURATION * sr
            num_segments = total_samples // samples_per_segment

            for i in range(num_segments):
                start = i * samples_per_segment
                end = start + samples_per_segment
                segment = y[start:end]
                output_file = os.path.join(output_genre_path,
                                           f"{file.replace('.au','')}_{i}.png")
                save_mel_spectrogram(segment, sr, output_file)
                print(f"Saved: {output_file}")

print("🎶 Spectrogram preprocessing complete!")
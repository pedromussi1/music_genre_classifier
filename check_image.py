import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Pick one spectrogram to check
IMG_PATH = "data_spectrograms/rock/rock.00000.png"  # change if needed

if os.path.exists(IMG_PATH):
    img = mpimg.imread(IMG_PATH)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
else:
    print(f"❌ File not found: {IMG_PATH}")

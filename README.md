# 🎵 Music Genre Classifier

A web-based music genre classification application built with **Python, TensorFlow/Keras, and Streamlit**. This project demonstrates end-to-end machine learning, from audio preprocessing and model training to deployment of a fully interactive web app.

Live Demo: [Streamlit App Link](YOUR_STREAMLIT_CLOUD_URL_HERE)

---

## **Project Overview**

This application allows users to upload a music clip and predicts its genre. The model is trained on the **GTZAN dataset**, which contains 10 genres:

- Blues  
- Classical  
- Country  
- Disco  
- Hip Hop  
- Jazz  
- Metal  
- Pop  
- Reggae  
- Rock  

The prediction is displayed alongside a **confidence score** for each genre, giving users insight into how confident the model is in its classification.

---

## **Features**

- **Audio Upload:** Users can upload `.wav` or `.au` audio files for analysis.  
- **Genre Prediction:** Outputs predicted genre and confidence scores.  
- **Spectrogram Visualization:** (Optional) Visualizes audio as spectrograms for deeper understanding.  
- **Interactive Web App:** Built with Streamlit for live online use.  
- **Deployed Online:** Anyone can access and use the application without installing Python or libraries.

---

## **Technology Stack**

- **Python 3.x**  
- **TensorFlow/Keras:** Convolutional Neural Network for genre classification  
- **Librosa:** Audio preprocessing and spectrogram creation  
- **Matplotlib / Pillow:** Visualizations  
- **Streamlit:** Web interface and live deployment  

---

## **Repository Structure**

music_genre_classifier/
├── app.py # Main Streamlit web application
├── models/
│ └── music_genre_cnn.keras # Trained Keras model
├── utils/
│ ├── audio_utils.py # Functions for loading and preprocessing audio
│ └── model_utils.py # Functions for model inference
├── assets/ # Optional: images or UI assets
├── requirements.txt # Python dependencies
├── README.md


---

## **Getting Started (Local Run)**

1. **Clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/music_genre_classifier.git
cd music_genre_classifier
```

2. **Create a virtual environment:**
   
```
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**

```
pip install -r requirements.txt
```

4. **Run the app locally:**

```
streamlit run app.py
```

5. **Open the URL displayed in your browser to use the app.**

## **How It Works**

1. **Audio Preprocessing**  
   - Uploaded audio files (`.wav` or `.au`) are loaded using `librosa`.  
   - The audio is converted into a **mel-spectrogram**, which is a visual representation of the frequency spectrum over time.  

2. **Model Prediction**  
   - The preprocessed spectrogram is fed into the trained **CNN model** (`music_genre_cnn.keras`).  
   - The model outputs probabilities for each genre.  

3. **Display in Web App**  
   - Streamlit displays the **predicted genre** with the **highest confidence**.  
   - Confidence scores for all genres can also be shown as a **bar chart** or **percentage list**.

4. **Optional Visualization**  
   - Spectrogram images can be displayed in the app to help users understand **what the model “sees”** in the audio data.

## **Results**

The model achieves strong accuracy on the GTZAN dataset and provides reliable genre predictions for short audio clips.  

**Example output in the app:**

Predicted Genre: Jazz (87.4% confidence)

---

## **Future Improvements**

- Integrate **multiple audio file uploads** for batch predictions.  
- Add **real-time audio analysis** for streaming music.  
- Experiment with **pretrained audio embeddings** or hybrid **CNN + LSTM models**.  
- Enhance the UI with **confidence bar charts** and **spectrogram visualizations**.  
- Include **unit tests** for preprocessing functions to ensure robustness.

---

## **License**

This project is open-source and available under the **MIT License**.  
Feel free to use, modify, or contribute.

---

## **Contact**

- **GitHub:** [YOUR_GITHUB_PROFILE](https://github.com/YOUR_USERNAME)  
- **LinkedIn:** [YOUR_LINKEDIN_PROFILE](https://www.linkedin.com/in/YOUR_PROFILE)  
- **Email:** YOUR_EMAIL_ADDRESS

---


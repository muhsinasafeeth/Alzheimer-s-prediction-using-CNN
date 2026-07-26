# 🧠 Early-Stage Alzheimer's Detection using CNN

A Streamlit web application that classifies brain MRI scans as **Non-Demented** or **Very Mild Demented** (early-stage Alzheimer's) using a custom-trained Convolutional Neural Network (CNN), with Grad-CAM explainability built in.

**🔗 Live demo:** https://alzheimer-s-prediction-using-cnn-vuxuaaesyae32njnhvthrq.streamlit.app/

> ⚠️ **Disclaimer:** This project is for **educational/research purposes only**. It is **not** a certified medical device and must not be used for real clinical diagnosis.

---

## 📌 Overview

| | |
|---|---|
| **Input** | Brain MRI image (JPG/PNG) |
| **Output** | Binary prediction — `0 = Non-Demented`, `1 = Very Mild Demented` |
| **Model** | Custom CNN: 3× (Conv2D → MaxPooling) → GlobalAveragePooling2D → Dense(128) → Dropout(0.5) → Dense(1, sigmoid) |
| **Input size** | 224 × 224 × 3 (RGB), pixels scaled to `[0, 1]` |
| **Explainability** | Grad-CAM heatmaps highlight the MRI regions that most influenced the prediction |

---

## 📁 Repository Structure

```
Alzheimer-s-prediction-using-CNN/
├── app.py                                          # Main Streamlit application (entry point)
├── config.py                                       # App settings, model path, class labels
├── model.py                                        # Cached model loading
├── preprocess.py                                   # Image loading & preprocessing (matches training)
├── predict.py                                       # Inference + confidence score calculation
├── gradcam.py                                      # Grad-CAM heatmap generation
├── alzheimer_model.keras                          # Trained model weights (~470 KB)
├── requirements.txt                                # Python dependencies
├── runtime.txt                                     # Python version (for buildpack-based hosts)
├── Early_Stage_Alzheimer_s_Detection_.ipynb        # Model training & evaluation notebook
├── Early-Stage Alzheimers Detection - Presentation.pdf   # Project presentation
├── .devcontainer/                                  # Dev container configuration
├── LICENSE                                         # MIT License
└── README.md
```

---

## ⚙️ How It Works (Procedure)

The application follows this end-to-end pipeline:

1. **Upload** — The user uploads a brain MRI image (JPG/PNG) via the Streamlit UI (`app.py`).
2. **Preprocess** (`preprocess.py`) — The image is resized to 224×224, converted to RGB, and pixel values are scaled to `[0, 1]` — matching the exact preprocessing used during training.
3. **Load model** (`model.py`) — The trained `alzheimer_model.keras` file is loaded once and cached for fast repeated inference.
4. **Predict** (`predict.py`) — The preprocessed image is passed through the CNN, producing a sigmoid output between 0 and 1. This is converted into a class label (`Non-Demented` / `Very Mild Demented`) and a confidence score.
5. **Explain** (`gradcam.py`) — A Grad-CAM heatmap is generated to visualize which regions of the MRI most influenced the model's decision, and is overlaid on the original scan.
6. **Display results** (`app.py`) — The app shows the predicted class, confidence score (metric + progress bar), the raw sigmoid output, and the Grad-CAM overlay.

---

## 🛠️ Model Training Procedure

Training and evaluation are done in `Early_Stage_Alzheimer_s_Detection_.ipynb`:

1. **Dataset** — A subset (`NonDemented`, `VeryMildDemented` classes) of a brain MRI dataset from Kaggle.
2. **Data split** — A **patient-level** train/validation/test split is used to prevent data leakage (ensuring scans from the same patient don't appear in multiple splits).
3. **Preprocessing** — Images resized to 224×224×3 and normalized to `[0, 1]`.
4. **Model architecture** — A simple custom CNN:
   - 3× `Conv2D` + `MaxPooling2D` blocks (feature extraction)
   - `GlobalAveragePooling2D` (dimensionality reduction)
   - `Dense(128)` fully connected layer
   - `Dropout(0.5)` for regularization
   - `Dense(1, activation="sigmoid")` output layer for binary classification
5. **Training** — The model is trained/fit on the training split, validated on the validation split.
6. **Evaluation** — Performance is measured using accuracy, precision, recall, F1-score, and ROC curve/AUC on the held-out test set.
7. **Explainability development** — Grad-CAM is implemented and tested to visualize model attention on MRI scans.
8. **Export** — The trained model is saved as `alzheimer_model.keras` for use in the Streamlit app.

See the notebook for the full code, metrics, and plots.

---

## 🚀 Running Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/muhsinasafeeth/Alzheimer-s-prediction-using-CNN.git
   cd Alzheimer-s-prediction-using-CNN
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. **Open the app** — Navigate to the URL shown in the terminal (usually `http://localhost:8501`).

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push the repository to GitHub (public, or private on a paid plan). Ensure `alzheimer_model.keras` is committed — it's small enough (~470 KB) to commit directly, no Git LFS needed.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** → select this repository and branch.
4. Set the **main file path** to `app.py`.
5. Click **Deploy**. Streamlit Cloud will automatically install dependencies from `requirements.txt`.

No secrets or external services are required for this app.

---

## 🖥️ App Features

- 📤 MRI image upload (JPG/PNG)
- 🎯 Prediction with class label and confidence score
- 📊 Confidence shown as a metric + progress bar, with the raw sigmoid output displayed for transparency
- 🔥 Grad-CAM heatmap overlay for visual explainability
- ℹ️ Sidebar with model architecture details and a medical disclaimer

---

## 📸 Screenshots

*Add screenshots of the running app here, e.g.:*

```
screenshots/
├── upload_screen.png
├── prediction_result.png
└── gradcam_overlay.png
```

---

## 🧪 Tech Stack

- **Python**
- **TensorFlow / Keras** — model architecture, training, inference
- **Streamlit** — web application / UI
- **NumPy / Pillow (or OpenCV)** — image preprocessing
- **Matplotlib** — Grad-CAM visualization overlay

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

**Muhsina Safeeth** — [GitHub](https://github.com/muhsinasafeeth)l training, evaluation (accuracy/precision/recall/F1/ROC),
and Grad-CAM development details.

# 🧠 Early-Stage Alzheimer's Detection

A Streamlit web app that classifies brain MRI scans as **Non-Demented** or
**Very Mild Demented** (early-stage Alzheimer's) using a custom-trained CNN.

> ⚠️ **Disclaimer:** This project is for educational/research purposes only.
> It is **not** a certified medical device and must not be used for real
> clinical diagnosis.

---

## 📌 Overview

- **Input:** Brain MRI image (JPG/PNG)
- **Output:** Binary prediction — `0 = Non-Demented`, `1 = Very Mild Demented`
- **Model:** Custom CNN (3× Conv2D/MaxPooling → GlobalAveragePooling2D →
  Dense(128) → Dropout(0.5) → Dense(1, sigmoid))
- **Input size:** 224 × 224 × 3 (RGB), pixels scaled to `[0, 1]`
- **Explainability:** Grad-CAM heatmaps highlight the regions of the scan
  that most influenced the prediction

---

## 📁 Repository Structure

```
alzheimer-app/
├── app.py                     # Main Streamlit application
├── config.py                  # App settings, model path, class labels
├── model.py                   # Cached model loading
├── preprocess.py              # Image loading & preprocessing (matches training)
├── predict.py                 # Inference + confidence calculation
├── alzheimer_model.keras      # Trained model weights
├── requirements.txt           # Python dependencies
├── runtime.txt                # Python version (for buildpack-based hosts)
└── README.md
```

---

## 🚀 Running Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/alzheimer-app.git
   cd alzheimer-app
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

5. Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push this repository to GitHub (public, or private on a paid plan).
   Make sure `alzheimer_model.keras` is committed — it's small enough
   (~470 KB) to commit directly, no Git LFS required.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** → select this repository and branch.
4. Set the **main file path** to `app.py`.
5. Click **Deploy**. Streamlit Cloud will install everything from
   `requirements.txt` automatically.

That's it — no secrets or external services are required for this app.

---

## 🖥️ App Features

- 📤 MRI image upload (JPG/PNG)
- 🎯 Prediction with class label and confidence score
- 📊 Confidence displayed as a metric + progress bar, with the raw sigmoid
  output shown for transparency
- ℹ️ Sidebar with model architecture details and a medical disclaimer

---

## 📸 Screenshots

_Add screenshots of the running app here after deployment, e.g.:_

```
screenshots/
├── upload_screen.png
├── prediction_result.png
└── gradcam_overlay.png
```

---

## 🏷️ Model Training

The model was trained in `Early_Stage_Alzheimer_s_Detection_.ipynb` on a
subset (NonDemented, VeryMildDemented) of an MRI dataset from Kaggle, with
a patient-level train/val/test split to avoid data leakage. See the
notebook for full training, evaluation (accuracy/precision/recall/F1/ROC),
and Grad-CAM development details.

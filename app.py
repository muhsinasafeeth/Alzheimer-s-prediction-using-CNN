import streamlit as st

from config import APP_TITLE, APP_ICON, MODEL_INFO
from model import load_model
from preprocess import preprocess_image
from predict import predict_image
from gradcam import generate_gradcam_images


# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide"
)

# -------------------------------------------------------
# Custom Styling — Dark Theme
# -------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App background — deep navy/charcoal */
.stApp {
    background: linear-gradient(180deg, #0F1620 0%, #121B29 45%, #0C131E 100%);
}

/* Animated gradient hero header */
.hero {
    text-align: center;
    padding: 34px 20px 26px 20px;
    border-radius: 20px;
    margin-bottom: 28px;
    background: linear-gradient(120deg, #0B4F6C, #1565C0, #7b2ff7, #0B4F6C);
    background-size: 300% 300%;
    animation: gradientShift 12s ease infinite;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.main-title {
    font-family: 'Poppins', sans-serif;
    font-size: 42px;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    text-shadow: 0 2px 10px rgba(0,0,0,0.35);
}

.sub-title {
    font-size: 16px;
    color: rgba(255,255,255,0.92);
    margin-top: 8px;
    font-weight: 400;
}

/* Section headers */
.section-header {
    font-family: 'Poppins', sans-serif;
    font-size: 20px;
    font-weight: 600;
    color: #7DD3FC;
    margin: 6px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Card container used around images / results */
.fade-card {
    background: #1B2434;
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(125, 211, 252, 0.14);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeInUp 0.5s ease both;
    margin-bottom: 18px;
}

.fade-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 26px rgba(0, 0, 0, 0.5);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

.card-caption {
    text-align: center;
    font-weight: 600;
    color: #C7D2E0;
    margin-top: 10px;
    font-size: 14px;
}

/* Result badge (pill) */
.result-pill {
    display: inline-block;
    padding: 10px 22px;
    border-radius: 999px;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 17px;
    color: white;
    margin-bottom: 14px;
    animation: popIn 0.4s ease both;
}

.pill-safe {
    background: linear-gradient(120deg, #11998e, #38ef7d);
    box-shadow: 0 6px 16px rgba(17, 153, 142, 0.4);
}

.pill-warn {
    background: linear-gradient(120deg, #f7971e, #ff5858);
    box-shadow: 0 6px 16px rgba(255, 88, 88, 0.4);
}

@keyframes popIn {
    0%   { transform: scale(0.85); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}

/* Confidence gauge bar */
.gauge-track {
    background: #2A3548;
    border-radius: 999px;
    height: 16px;
    width: 100%;
    overflow: hidden;
    margin: 6px 0 4px 0;
}

.gauge-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    animation: growBar 1s ease both;
}

@keyframes growBar {
    from { width: 0%; }
}

.gauge-label {
    font-size: 13px;
    color: #9AA8BD;
    margin-bottom: 2px;
}

.raw-output {
    font-size: 13px;
    color: #9AA8BD;
    margin-top: 10px;
}

/* Footer */
.footer {
    font-size: 13px;
    color: #8291A8;
    text-align: center;
    margin-top: 10px;
}

.footer-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2A3548, transparent);
    border: none;
    margin: 30px 0 16px 0;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.title("🧠 Model Information")

for key, value in MODEL_INFO.items():
    st.sidebar.write(f"**{key}:** {value}")

st.sidebar.markdown("---")

st.sidebar.info(
    """
This application predicts whether an uploaded MRI image is:

• Non Demented

or

• Very Mild Demented

using a CNN model trained with TensorFlow, and explains *why* using
**Grad-CAM** — a heatmap of the image regions that most influenced
the prediction.
"""
)

# -------------------------------------------------------
# Hero / Title
# -------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <p class="main-title">{APP_ICON} {APP_TITLE}</p>
        <p class="sub-title">Upload a brain MRI scan to get a prediction — and see exactly which regions the model focused on.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------
model = load_model()

if model is None:
    st.stop()

# -------------------------------------------------------
# Upload Section
# -------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image_array, display_image = preprocess_image(uploaded_file)

    with st.spinner("Analyzing image..."):
        result = predict_image(model, image_array)

        try:
            gradcam_images = generate_gradcam_images(
                model, image_array, display_image
            )
            gradcam_error = None
        except Exception as e:
            gradcam_images = None
            gradcam_error = str(e)

    prediction = result["prediction"]
    confidence = result["confidence"]
    probability = result["probability"]

    # ---------------- Prediction summary ----------------
    st.markdown('<p class="section-header">🎯 Prediction Result</p>', unsafe_allow_html=True)

    result_col, gauge_col = st.columns([1, 1.3])

    with result_col:
        pill_class = "pill-safe" if prediction == "Non Demented" else "pill-warn"
        icon = "✅" if prediction == "Non Demented" else "⚠️"

        st.markdown(
            f"""
            <div class="fade-card">
                <span class="result-pill {pill_class}">{icon} {prediction}</span>
                <p style="color:#B8C4D6; margin-top:6px;">
                    The uploaded MRI image was classified as
                    <strong style="color:#E8EDF4;">{prediction}</strong> by the CNN model.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with gauge_col:
        confidence_pct = confidence * 100
        st.markdown(
            f"""
            <div class="fade-card">
                <p class="gauge-label">Model Confidence</p>
                <div class="gauge-track">
                    <div class="gauge-fill" style="width:{confidence_pct:.1f}%;"></div>
                </div>
                <p style="text-align:right; font-weight:600; color:#4FC3F7; margin-top:4px;">
                    {confidence_pct:.2f}%
                </p>
                <p class="raw-output">Raw sigmoid output: <strong style="color:#C7D2E0;">{probability:.4f}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------- Image comparison ----------------
    st.markdown('<p class="section-header">🔍 Grad-CAM Explainability</p>', unsafe_allow_html=True)
    st.caption(
        "Grad-CAM highlights the regions of the scan the model relied on most. "
        "Warmer colors (red/yellow) indicate stronger influence on the prediction."
    )

    if gradcam_images is not None:
        img_col1, img_col2, img_col3 = st.columns(3)

        with img_col1:
            st.markdown('<div class="fade-card">', unsafe_allow_html=True)
            st.image(display_image, use_container_width=True)
            st.markdown('<p class="card-caption">🧠 Original MRI</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with img_col2:
            st.markdown('<div class="fade-card">', unsafe_allow_html=True)
            st.image(gradcam_images["heatmap_image"], use_container_width=True)
            st.markdown('<p class="card-caption">🔥 Grad-CAM Heatmap</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with img_col3:
            st.markdown('<div class="fade-card">', unsafe_allow_html=True)
            st.image(gradcam_images["overlay_image"], use_container_width=True)
            st.markdown('<p class="card-caption">🧩 Overlay (Original + Heatmap)</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error(
            f"Grad-CAM could not be generated for this image: {gradcam_error}"
        )

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown('<hr class="footer-divider">', unsafe_allow_html=True)

st.warning(
"""
### Medical Disclaimer

This application is intended **only for educational and research purposes**.

It is **not a medical diagnostic tool** and should **not** be used as a substitute for professional clinical assessment or medical advice.

Always consult a qualified healthcare professional for diagnosis and treatment decisions.
"""
)

st.markdown(
    "<p class='footer'>🧠 Built with TensorFlow &amp; Streamlit • Grad-CAM explainability</p>",
    unsafe_allow_html=True
)

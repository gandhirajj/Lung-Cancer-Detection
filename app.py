import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, LSTM, Dense
from PIL import Image
import cv2
import time
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ======================
# 🔹 App Title & Sidebar
# ======================
st.set_page_config(page_title="Lung Cancer Detector", page_icon="🫁", layout="wide")
st.title("Lung Cancer Detection Dashboard")
st.sidebar.header("📁 Patient Metadata")

# ------------------
# Patient Information
# ------------------
age = st.sidebar.number_input("Age", min_value=1, max_value=120, step=1)
gender = st.sidebar.radio("Gender", ["Male", "Female", "Other"])
symptoms = st.sidebar.multiselect("Symptoms", ["Cough", "Fever", "Chest Pain", "Fatigue"])
smoking_history = st.sidebar.selectbox("Smoking History", ["Non-smoker", "Former smoker", "Current smoker"])

# ------------------
# Rebuild Model Architecture & Load Weights
# ------------------
def build_model():
    inputs = Input(shape=(224, 224, 3))
    x = Conv2D(32, 3, activation="relu", kernel_initializer="he_normal")(inputs)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(32, 3, activation="relu", kernel_initializer="he_normal")(x)
    x = MaxPooling2D(2, 2)(x)
    x = Reshape((-1, 32))(x)
    x = LSTM(64)(x)
    x = Dense(32, activation="elu", kernel_initializer="he_uniform")(x)
    outputs = Dense(3, activation="softmax")(x)
    return Model(inputs, outputs)

@st.cache_resource
def load_main_model():
    try:
        model = build_model()
        model.load_weights("lung_weights.h5")  # load weights only
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model weights: {e}")
        return None

model = load_main_model()
model_version = "v1.0"

# ------------------
# Helper Functions
# ------------------
def preprocess(img):
    img = img.convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=0)

def export_pdf(report_text, img, filename="report.pdf"):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)

    text_obj = c.beginText(40, height - 40)
    for line in report_text.split("\n"):
        text_obj.textLine(line)
    c.drawText(text_obj)

    if img is not None:
        img_rgb = img.convert("RGB").resize((200, 200))
        img_rgb.save("xray_temp.png")
        c.drawImage("xray_temp.png", 40, height//2 - 100, width=200, preserveAspectRatio=True)

    c.save()
    buffer.seek(0)
    return buffer

# ------------------
# File Upload
# ------------------
uploaded_file = st.file_uploader("📤 Upload Chest CT Scan", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Scan", use_column_width=True)

    # Prediction
    start_time = time.time()
    img_array = preprocess(img)
    prediction = model.predict(img_array)[0]
    inference_time = time.time() - start_time

    classes = ["Benign", "Malignant", "Normal"]
    pred_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)

    with col2:
        st.metric("Prediction", pred_class)
        st.metric("Confidence", f"{confidence*100:.2f}%")
        st.metric("Inference Time", f"{inference_time:.2f} sec")
        st.metric("Model Version", model_version)

    # Recommendations
    st.subheader("🧪 Explainability & Recommendations")
    if pred_class == "Malignant":
        st.error("⚠️ Malignant case detected. Please consult an oncologist immediately.")
    elif pred_class == "Benign":
        st.warning("🟠 Benign case detected. Follow up with your doctor for monitoring.")
    else:
        st.success("✅ Normal scan. No immediate concerns detected.")

    if symptoms:
        st.info(f"📌 Reported symptoms: {', '.join(symptoms)}")

    # PDF Export
    st.subheader("📤 Export Report")
    report_text = f"""
    Patient Report
    -------------------------
    Age: {age}
    Gender: {gender}
    Smoking History: {smoking_history}
    Symptoms: {', '.join(symptoms) if symptoms else 'None'}

    Prediction: {pred_class}
    Confidence: {confidence*100:.2f}%
    Inference Time: {inference_time:.2f} sec
    Model Version: {model_version}
    """
    pdf_buffer = export_pdf(report_text, img)
    st.download_button(
        label="📥 Download Report (PDF)",
        data=pdf_buffer,
        file_name="lung_report.pdf",
        mime="application/pdf"
    )

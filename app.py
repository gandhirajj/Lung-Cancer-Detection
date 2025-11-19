# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image
# import cv2
# import time
# import io
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas

# # ======================
# # 🔹 App Title & Sidebar
# # ======================
# st.set_page_config(page_title="Lung Cancer Detector", page_icon="🫁", layout="wide")
# st.title("Lung Cancer Detection Dashboard")
# st.sidebar.header("📁 Patient Metadata")

# # ------------------
# # Patient Information
# # ------------------
# age = st.sidebar.number_input("Age", min_value=1, max_value=120, step=1)
# gender = st.sidebar.radio("Gender", ["Male", "Female", "Other"])
# smoking_history = st.sidebar.selectbox("Smoking History", ["Non-smoker", "Former smoker", "Current smoker"])
# symptoms = st.sidebar.multiselect("Symptoms", ["Cough", "Chest Pain", "Shortness of Breath", "Weight Loss", "Fatigue"])

# # ------------------
# # Load Model
# # ------------------
# @st.cache_resource
# def load_main_model():
#     return load_model("lung_model.h5")  # replace with your lung model path

# model = load_main_model()
# model_version = "v1.0"

# # ------------------
# # Helper Functions
# # ------------------
# def is_valid_xray(pil_img):
#     """Basic validation for image brightness and contrast (no grayscale restriction)."""
#     try:
#         img = np.array(pil_img)
#         if len(img.shape) == 2:  # grayscale
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#         elif len(img.shape) == 3 and img.shape[2] == 3:
#             img_rgb = img
#         else:
#             return False

#         mean_intensity = np.mean(img_rgb)
#         if mean_intensity > 240 or mean_intensity < 20:
#             return False

#         img_resized = cv2.resize(img_rgb, (224, 224))
#         edges = cv2.Canny(cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY), 50, 150)
#         edge_density = np.sum(edges) / edges.size
#         if edge_density < 0.005:
#             return False

#         return True
#     except:
#         return False

# def preprocess(img):
#     """Preprocess image for model input (RGB only)."""
#     img = np.array(img)
#     if len(img.shape) == 2:  # convert grayscale to RGB
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     elif img.shape[2] == 4:  # RGBA → RGB
#         img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

#     img_resized = cv2.resize(img, (224, 224))
#     img_normalized = img_resized / 255.0
#     return np.expand_dims(img_normalized, axis=0)

# def generate_gradcam(model, img_array, last_conv_layer_name=None):
#     """Generate Grad-CAM heatmap for last conv layer."""
#     if last_conv_layer_name is None:
#         for layer in reversed(model.layers):
#             if isinstance(layer, tf.keras.layers.Conv2D):
#                 last_conv_layer_name = layer.name
#                 break

#     grad_model = tf.keras.models.Model(
#         [model.inputs],
#         [model.get_layer(last_conv_layer_name).output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         class_idx = tf.argmax(predictions[0])
#         loss = predictions[:, class_idx]

#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     conv_outputs = conv_outputs[0]
#     heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
#     heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
#     heatmap = cv2.resize(heatmap, (224, 224))
#     return heatmap, int(class_idx)

# def overlay_heatmap(original_img, heatmap, alpha=0.4):
#     """Overlay heatmap on original image."""
#     img = np.array(original_img.convert("RGB").resize((224, 224)))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
#     return overlay

# def export_pdf(report_text, img, heatmap_img, filename="report.pdf"):
#     """Generate PDF report with findings and images."""
#     buffer = io.BytesIO()
#     c = canvas.Canvas(buffer, pagesize=letter)
#     width, height = letter
#     c.setFont("Helvetica", 12)

#     text_obj = c.beginText(40, height - 40)
#     for line in report_text.split("\n"):
#         text_obj.textLine(line)
#     c.drawText(text_obj)

#     if img is not None:
#         img_rgb = img.convert("RGB").resize((200, 200))
#         img_rgb.save("xray_temp.png")
#         c.drawImage("xray_temp.png", 40, height // 2 - 100, width=200, preserveAspectRatio=True)
#     if heatmap_img is not None:
#         heatmap_pil = Image.fromarray(heatmap_img)
#         heatmap_pil.save("heatmap_temp.png")
#         c.drawImage("heatmap_temp.png", 280, height // 2 - 100, width=200, preserveAspectRatio=True)

#     c.save()
#     buffer.seek(0)
#     return buffer

# # ------------------
# # File Upload
# # ------------------
# uploaded_file = st.file_uploader("📤 Upload Lung CT Scan / X-ray", type=["png", "jpg", "jpeg"])

# if uploaded_file:
#     img = Image.open(uploaded_file)

#     if not is_valid_xray(img):
#         st.error("❌ Invalid or poor-quality medical image. Please upload a clear X-ray/CT image.")
#     else:
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(img, caption="Original Lung Scan", use_column_width=True)

#         start_time = time.time()
#         img_array = preprocess(img)
#         prediction = model.predict(img_array)[0]
#         inference_time = time.time() - start_time

#         class_names = ["Benign", "Malignant", "Normal"]
#         class_idx = np.argmax(prediction)
#         label = class_names[class_idx]
#         confidence = prediction[class_idx]

#         with col2:
#             st.metric("Prediction", label)
#             st.metric("Confidence", f"{confidence*100:.2f}%")
#             st.metric("Inference Time", f"{inference_time:.2f} sec")
#             st.metric("Model Version", model_version)

#         heatmap, _ = generate_gradcam(model, img_array)
#         overlay = overlay_heatmap(img, heatmap)

#         st.subheader("🔍 Visual Explanation (Grad-CAM)")
#         col3, col4 = st.columns(2)
#         with col3:
#             st.image(img, caption="Original Scan", use_column_width=True)
#         with col4:
#             st.image(overlay, caption="Highlighted Regions", use_column_width=True)

#         st.subheader("🧪 Explainability & Recommendations")
#         if label == "Malignant":
#             st.error("⚠ Malignant nodule detected. Please consult an oncologist immediately.")
#         elif label == "Benign":
#             st.warning("⚠ Benign nodule detected. Regular monitoring and doctor follow-up recommended.")
#         else:
#             st.success("✅ Normal lungs detected. Continue regular health check-ups.")

#         if symptoms:
#             st.info(f"📌 Reported symptoms: {', '.join(symptoms)}")

#         st.subheader("📤 Export Report")
#         report_text = f"""
#         Patient Report
#         -------------------------
#         Age: {age}
#         Gender: {gender}
#         Smoking History: {smoking_history}
#         Symptoms: {', '.join(symptoms) if symptoms else 'None'}

#         Prediction: {label}
#         Confidence: {confidence*100:.2f}%
#         Inference Time: {inference_time:.2f} sec
#         Model Version: {model_version}
#         """

#         pdf_buffer = export_pdf(report_text, img, overlay)
#         st.download_button(
#             label="📥 Download Report (PDF)",
#             data=pdf_buffer,
#             file_name="lung_report.pdf",
#             mime="application/pdf"
#         )


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
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
smoking_history = st.sidebar.selectbox("Smoking History", ["Non-smoker", "Former smoker", "Current smoker"])
symptoms = st.sidebar.multiselect(
    "Symptoms", ["Cough", "Chest Pain", "Shortness of Breath", "Weight Loss", "Fatigue"]
)

# ------------------
# Load Model
# ------------------
@st.cache_resource
def load_main_model():
    return load_model("lung_model.h5")  # replace with your lung model path

model = load_main_model()
model_version = "v1.0"


# ------------------
# Helper Functions
# ------------------
def is_valid_xray(pil_img):
    """Basic validation for brightness, noise, edges."""
    try:
        img = np.array(pil_img)
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = img
        else:
            return False

        mean_intensity = np.mean(img_rgb)
        if mean_intensity > 240 or mean_intensity < 20:
            return False

        img_resized = cv2.resize(img_rgb, (224, 224))
        edges = cv2.Canny(cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY), 50, 150)
        edge_density = np.sum(edges) / edges.size
        if edge_density < 0.005:
            return False

        return True
    except:
        return False


def preprocess(img):
    """Preprocess image for model input."""
    img = np.array(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    return np.expand_dims(img_normalized, axis=0)


# ==========================
# ✅ FIXED SAFE GRAD-CAM
# ==========================
def generate_gradcam(model, img_array, last_conv_layer_name=None):
    """Generate Grad-CAM heatmap (Safe on all TF models)."""

    # Find last conv layer
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Ensure predictions is a Tensor
        predictions = tf.convert_to_tensor(predictions)

        class_idx = tf.argmax(predictions[0])

        # SAFE: avoid Tensor slicing errors
        loss = tf.gather(predictions[0], class_idx)

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    return heatmap, int(class_idx)


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    img = np.array(original_img.convert("RGB").resize((224, 224)))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay


def export_pdf(report_text, img, heatmap_img, filename="report.pdf"):
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
        c.drawImage("xray_temp.png", 40, height // 2 - 100, width=200)

    if heatmap_img is not None:
        heatmap_pil = Image.fromarray(heatmap_img)
        heatmap_pil.save("heatmap_temp.png")
        c.drawImage("heatmap_temp.png", 280, height // 2 - 100, width=200)

    c.save()
    buffer.seek(0)
    return buffer


# ------------------
# File Upload
# ------------------
uploaded_file = st.file_uploader("📤 Upload Lung CT Scan / X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)

    if not is_valid_xray(img):
        st.error("❌ Invalid or poor-quality medical image. Upload a clear X-ray/CT.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Original Lung Scan", use_column_width=True)

        start_time = time.time()
        img_array = preprocess(img)
        prediction = model.predict(img_array)[0]
        inference_time = time.time() - start_time

        class_names = ["Benign", "Malignant", "Normal"]
        class_idx = np.argmax(prediction)
        label = class_names[class_idx]
        confidence = prediction[class_idx]

        with col2:
            st.metric("Prediction", label)
            st.metric("Confidence", f"{confidence*100:.2f}%")
            st.metric("Inference Time", f"{inference_time:.2f} sec")
            st.metric("Model Version", model_version)

        # ---- Grad-CAM ----
        heatmap, _ = generate_gradcam(model, img_array)
        overlay = overlay_heatmap(img, heatmap)

        st.subheader("🔍 Visual Explanation (Grad-CAM)")
        col3, col4 = st.columns(2)
        with col3:
            st.image(img, caption="Original Scan", use_column_width=True)
        with col4:
            st.image(overlay, caption="Highlighted Regions", use_column_width=True)

        # ---- Interpretation ----
        st.subheader("🧪 Explainability & Recommendations")
        if label == "Malignant":
            st.error("⚠ Malignant nodule detected. Consult an oncologist immediately.")
        elif label == "Benign":
            st.warning("⚠ Benign nodule detected. Follow-up check recommended.")
        else:
            st.success("✅ Normal lungs. Continue routine check-ups.")

        if symptoms:
            st.info(f"📌 Reported symptoms: {', '.join(symptoms)}")

        # ---- PDF Report ----
        st.subheader("📤 Export Report")
        report_text = f"""
        Patient Report
        -------------------------
        Age: {age}
        Gender: {gender}
        Smoking History: {smoking_history}
        Symptoms: {', '.join(symptoms) if symptoms else 'None'}

        Prediction: {label}
        Confidence: {confidence*100:.2f}%
        Inference Time: {inference_time:.2f} sec
        Model Version: {model_version}
        """

        pdf_buffer = export_pdf(report_text, img, overlay)
        st.download_button(
            label="📥 Download Report (PDF)",
            data=pdf_buffer,
            file_name="lung_report.pdf",
            mime="application/pdf"
        )

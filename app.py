import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import LSTM, Reshape

# ---------------------------
# Model Loader
# ---------------------------
@st.cache_resource
def load_main_model():
    try:
        # 🔹 Preferred: Load SavedModel folder
        return tf.keras.models.load_model("lung_model")

    except Exception as e1:
        st.warning(f"SavedModel not found, trying .h5... ({e1})")
        try:
            # 🔹 Load H5 model with fallback
            return load_model(
                "lung_model.h5",
                compile=False,
                custom_objects={"LSTM": LSTM, "Reshape": Reshape},
                safe_mode=False  # disables strict checks
            )
        except Exception as e2:
            st.error(f"❌ Failed to load model: {e2}")
            return None


# ---------------------------
# Prediction Function
# ---------------------------
def predict_image(model, img_file):
    # Load and preprocess image
    img = image.load_img(img_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    return class_idx, confidence


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Lung Cancer Detection", layout="centered")

st.title("🫁 Lung Cancer Detection (CNN + LSTM)")
st.write("Upload a chest CT scan image to screen for lung disease.")

# Load model
model = load_main_model()

if model is not None:
    uploaded_file = st.file_uploader("Upload CT Scan Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Predict"):
            with st.spinner("Analyzing..."):
                class_idx, confidence = predict_image(model, uploaded_file)

            # Map class index to label (adjust to your dataset)
            classes = ["Normal", "Benign Nodule", "Malignant Nodule"]

            st.success(f"Prediction: **{classes[class_idx]}** ({confidence:.2%} confidence)")
else:
    st.error("Model could not be loaded. Please check your saved format (SavedModel or H5).")

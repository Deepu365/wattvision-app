import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image
IMAGE_SIZE = (224, 224)
MODEL_FILE = "leaf_model.keras"
class_names = ["Apple Scab", "Apple Black Rot", "Healthy"]
st.set_page_config(page_title="Crop Disease Detector", layout="wide")
st.title("🌿 Crop Disease Detection System")
if not os.path.exists(MODEL_FILE):
    st.error(f"Error: '{MODEL_FILE}' not found in this folder.")
else:
    @st.cache_resource
    def get_model():
        return load_model(MODEL_FILE)
    model = get_model()
    uploaded_files = st.file_uploader(
        "Upload leaf images...", 
        type=["jpg", "png", "jpeg"], 
        accept_multiple_files=True
    )
    if uploaded_files:
        cols = st.columns(3)
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx % 3]:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                image_resized = image.resize(IMAGE_SIZE)
                image_array = np.array(image_resized) / 255.0
                image_array = np.expand_dims(image_array, axis=0)
                prediction = model.predict(image_array, verbose=0)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                st.success(f"**{predicted_class}**")
                st.caption(f"Confidence: {confidence:.2f}%")

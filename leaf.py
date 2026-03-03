import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load the model (Make sure 'plant_model.h5' is in the SAME folder)
@st.cache_resource
def load_my_model():
    try:
        return tf.keras.models.load_model('plant_model.h5')
    except Exception as e:
        st.error(f"Error: Could not find 'plant_model.h5'. {e}")
        return None

model = load_my_model()

# 2. Define your disease classes (Update these to match your model's training)
CLASS_NAMES = ['Healthy', 'Powdery Mildew', 'Rust', 'Leaf Spot']

st.title("🌿 Smart Leaf Disease Detection")

# 3. File Uploader
uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf.', use_container_width=True)
    
    if model is not None:
        st.write("Analyzing...")
        
        # Preprocessing (Resizing image to match model input, e.g., 224x224)
        img = image.resize((224, 224)) 
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        # Prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        result = CLASS_NAMES[np.argmax(score)]
        confidence = 100 * np.max(score)

        st.success(f"Prediction: {result} ({confidence:.2f}% confidence)")
    else:
        st.warning("Please upload the 'plant_model.h5' file to the project directory.")
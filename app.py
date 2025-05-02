import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import gdown  # Used to download from Google Drive

# Google Drive file ID for the updated model
FILE_ID = "1MYgSZdZJOe-JzRjxyFD_B31p7crIA20K"
MODEL_PATH = "my1_boat_model.h5"

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# Load the model
model = load_model(MODEL_PATH)

# Define class names
class_names = [
    'Lanciafino10m', 'Ambulanza', 'Lanciafino10mBianca', 'Cacciapesca', 'Caorlina',
    'Alilaguna', 'Gondola', 'Barchino', 'Motobarca', 'Lanciamaggioredi10mBianca',
    'MotoscafoACTV', 'Lanciafino10mMarrone', 'Motopontonerettangolare',
    'Lanciamaggioredi10mMarrone', 'Raccoltarifiuti', 'Topa', 'Mototopo', 'Sandoloaremi',
    'Patanella', 'Sanpierota', 'Polizia', 'VigilidelFuoco', 'VaporettoACTV', 'water'
]

# App title
st.title("AI BOAT Classifier Model")

# Upload image
uploaded_file = st.file_uploader("Upload a boat image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((180, 180))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)[0]
    top_index = np.argmax(prediction)
    predicted_class = class_names[top_index]
    confidence = float(prediction[top_index]) * 100

    # Display result
    st.markdown(f"### Prediction: **{predicted_class}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    # Optional: Display top 3 predictions
    st.markdown("#### Top 3 Predictions:")
    top_3_indices = prediction.argsort()[-3:][::-1]
    for idx in top_3_indices:
        st.write(f"{class_names[idx]}: {prediction[idx]*100:.2f}%")

    if st.button("Like"):
        st.success("Thanks for your support!")

# Footer
st.markdown("---")
st.markdown("This tool is for educational purposes only and not a substitute for professional decision-making.")

# Import necessary libraries
import streamlit as st
import pickle
import numpy as np

# Define the CropRecommendationModel class
class CropRecommendationModel:
    def __init__(self, model, crop_names):
        self.model = model
        self.crop_names = crop_names
    
    def predict(self, X):
        y_pred = self.model.predict(X)
        predicted_crops = []
        for row in y_pred:
            index = np.argmax(row)
            predicted_crops.append(self.crop_names[index] if row[index] == 1 else "None")
        return predicted_crops

# Load the model
import pickle
with open("CropRecommendationModel.pkl", "rb") as file:
    model = pickle.load(file)

# Set up the page title and color
st.title("AI-Powered Crop Recommendation System")
st.markdown(
    "<style>body{ background-color: #e0ffe0; }</style>", 
    unsafe_allow_html=True
)

# Input fields for the seven features
N = st.number_input("Nitrogen Content (N)", min_value=0.0, max_value=200.0)
P = st.number_input("Phosphorous Content (P)", min_value=0.0, max_value=200.0)
K = st.number_input("Potassium Content (K)", min_value=0.0, max_value=200.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
ph = st.number_input("pH Level", min_value=4.0, max_value=14.0)
rainfall = st.number_input("Rainfall (mm)", min_value=50.0, max_value=1500.0)

# Submit button to make predictions
if st.button("Predict"):
    # Prepare input features for the model
    input_features = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = model.predict(input_features)
    
    # Display the result
    st.write(f"Recommended Crop: {prediction[0]}")

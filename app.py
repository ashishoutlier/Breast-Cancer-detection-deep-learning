import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model
model = load_model("breast_cancer_model.h5")

# Load scaler and feature details
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")
feature_ranges = joblib.load("feature_ranges.pkl")

st.title("Breast Cancer Detection App")
st.write("Adjust the sliders to input feature values and predict whether the tumor is **Malignant** or **Benign**")

inputs = []
st.subheader("Input Feature Values:")

cols = st.columns(3)

for i, feature in enumerate(feature_names):
    min_val, max_val = feature_ranges[feature]
    with cols[i % 3]:
        value = st.slider(feature, min_val, max_val, (min_val + max_val) / 2)
        inputs.append(value)

# Convert input to numpy and scale
input_data = np.array([inputs])
input_data_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    result = "Malignant" if prediction[0][0] > 0.5 else "Benign"
    st.success(f"Prediction: **{result}** (Probability: {prediction[0][0]:.2f})")

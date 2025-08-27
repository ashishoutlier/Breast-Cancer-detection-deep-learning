import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import re

# Load model, scaler, and feature info
model = load_model("breast_cancer_model.h5")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features.pkl")
feature_ranges = joblib.load("feature_ranges.pkl")

st.title("Breast Cancer Detection App")
st.write("### Choose input method: Sliders OR Paste Array")

# Tabs for two input methods
tab1, tab2 = st.tabs(["📊 Use Sliders", "📝 Paste Array"])

inputs = None

with tab1:
    st.subheader("Adjust the sliders to input feature values")
    inputs_slider = []
    cols = st.columns(3)

    for i, feature in enumerate(feature_names):
        min_val, max_val = feature_ranges[feature]
        with cols[i % 3]:
            value = st.slider(feature, min_val, max_val, (min_val + max_val) / 2)
            inputs_slider.append(value)

    if st.button("Predict (Using Sliders)", key="slider_predict"):
        inputs = np.array([inputs_slider])
        inputs_scaled = scaler.transform(inputs)
        prediction = model.predict(inputs_scaled)
        result = "Malignant" if prediction[0][0] > 0.5 else "Benign"
        st.success(f"Prediction: **{result}** (Probability: {prediction[0][0]:.4f})")

with tab2:
    st.subheader("Paste array values (comma separated or numpy format)")
    array_text = st.text_area("Example: [7.760e+00, 2.454e+01, 4.792e+01, ...]", height=150)

    if st.button("Predict (Using Pasted Array)", key="text_predict"):
        try:
            # Extract numbers using regex
            values = re.findall(r"[-+]?\d*\.\d+|\d+", array_text)
            values = [float(v) for v in values]

            if len(values) != len(feature_names):
                st.error(f"Expected {len(feature_names)} values, got {len(values)}")
            else:
                inputs = np.array([values])
                inputs_scaled = scaler.transform(inputs)
                prediction = model.predict(inputs_scaled)
                result = "Malignant" if prediction[0][0] > 0.5 else "Benign"
                st.success(f"Prediction: **{result}** (Probability: {prediction[0][0]:.4f})")
        except Exception as e:
            st.error(f"Error processing input: {e}")

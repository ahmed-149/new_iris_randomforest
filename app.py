import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_iris

# =========================
# Load trained model
# =========================
model = joblib.load("iris_random_forest_model.pkl")

# Load iris dataset (for class names)
iris = load_iris()

# =========================
# App Title
# =========================
st.title("🌸 Iris Flower Prediction App")
st.write("Predict Iris species using Random Forest model")

# =========================
# User Inputs
# =========================
st.subheader("Enter Flower Measurements")

sepal_length = st.number_input(
    "Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2
)

# =========================
# Prediction
# =========================
if st.button("Predict"):
    
    # Convert inputs to numpy array
    input_data = np.array([
        [sepal_length, sepal_width, petal_length, petal_width]
    ])

    # Make prediction
    prediction = model.predict(input_data)
    predicted_class = iris.target_names[prediction][0]

    # Show result
    st.success(f"🌼 Predicted Iris Species: **{predicted_class}**")

    # Optional: show input values
    st.write("### Input Features")
    st.write(input_data)

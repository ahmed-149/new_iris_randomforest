import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Iris Prediction")

st.title("🌸 Iris Flower Classification")
st.write("Predict the species of Iris flower using Decision Tree Model")

# Load model
model = joblib.load("iris_decision_tree_model.pkl")

st.header("Input Features")

sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8)
sepal_width  = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.0)
petal_width  = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.2)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted Species: 🌼 {species[prediction[0]]}")

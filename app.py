import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.datasets import load_iris

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="centered"
)

# =========================
# Load Model (Cached)
# =========================
@st.cache_resource
def load_model():
    return joblib.load("iris_random_forest_model.pkl")

model = load_model()

# Load iris dataset
iris = load_iris()

# =========================
# Sidebar - User Inputs
# =========================
st.sidebar.title("🌼 Input Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width  = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width  = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# =========================
# Main Title
# =========================
st.title("🌸 Iris Flower Prediction App")
st.write("Predict Iris species using a **Random Forest Classifier**")

# =========================
# Display Input Summary
# =========================
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

st.subheader("📊 Input Measurements")
input_df = pd.DataFrame(
    input_data,
    columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
)
st.dataframe(input_df, use_container_width=True)

# =========================
# Prediction
# =========================
if st.button("🔍 Predict Species"):

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    predicted_class = iris.target_names[prediction][0]

    # =========================
    # Result
    # =========================
    st.success(f"🌼 **Predicted Species:** {predicted_class.upper()}")

    # =========================
    # Confidence Bars
    # =========================
    st.subheader("📈 Prediction Confidence")

    proba_df = pd.DataFrame(
        prediction_proba,
        columns=iris.target_names
    ).T

    proba_df.columns = ["Confidence"]

    st.bar_chart(proba_df)

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("📌 Built with Streamlit & Scikit-learn | Random Forest Model")

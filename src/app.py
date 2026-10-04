
from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "best_model.pkl")
    imputer = joblib.load(MODEL_DIR / "imputer.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    threshold = joblib.load(MODEL_DIR / "threshold.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

    return model, imputer, scaler, threshold, feature_names

model, imputer, scaler, threshold, feature_names = load_model()

# Page setup
st.set_page_config(
    page_title="AquaShield AI",
    page_icon="💧",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #f0f8ff;
}
h1, h2, h3, p, label {
    color: #123b5d;
}
div.stButton > button {
    background-color: #087e8b;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.5rem;
}
div.stButton > button:hover {
    background-color: #05636d;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("💧 AquaShield AI")
st.markdown("### Intelligent Water Potability Prediction")
st.write(
    "Analyze water-quality parameters using a machine-learning model "
    "trained on the Kaggle Water Potability dataset."
)

st.divider()

# Input section
st.subheader("Enter Water Quality Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
    hardness = st.number_input("Hardness", min_value=0.0, value=200.0)
    solids = st.number_input("Solids", min_value=0.0, value=20000.0)

with col2:
    chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
    sulfate = st.number_input("Sulfate", min_value=0.0, value=333.0)
    conductivity = st.number_input("Conductivity", min_value=0.0, value=400.0)

with col3:
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=14.0)
    trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=66.0)
    turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0)

st.divider()

# Prediction
if st.button("Analyze Water", use_container_width=True):
    input_data = pd.DataFrame([{
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }])

    input_data = input_data[feature_names]

    input_imputed = imputer.transform(input_data)
    input_scaled = scaler.transform(input_imputed)

    probability = model.predict_proba(input_scaled)[0][1]
    prediction = int(probability >= threshold)

    st.subheader("Analysis Result")

    result_col, probability_col = st.columns(2)

    with result_col:
        if prediction == 1:
            st.success("Prediction: Potable")
        else:
            st.error("Prediction: Non-potable")

    with probability_col:
        st.metric("Potability Probability", f"{probability:.1%}")

    st.progress(float(probability))

    st.caption(
        "This is an experimental machine-learning prediction, "
        "not a laboratory test or drinking-water safety certification."
    )

st.divider()
st.caption("AquaShield AI | Water Quality Analysis Project")

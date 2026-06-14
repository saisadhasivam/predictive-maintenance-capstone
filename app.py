import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🔧",
    layout="centered"
)

# ── Load Model from Hugging Face Model Hub ───────────────────────────────────
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="SaiSadhasivam/predictive-maintenance-best-model",
        filename="best_model.pkl"
    )
    model = joblib.load(model_path)
    return model

model = load_model()

# ── App Header ───────────────────────────────────────────────────────────────
st.title("🔧 Predictive Maintenance")
st.subheader("Engine Failure Classification")
st.markdown("Enter the engine sensor readings below to predict whether the engine requires maintenance or is operating normally.")
st.markdown("---")

# ── Input Fields ─────────────────────────────────────────────────────────────
st.markdown("### Engine Sensor Inputs")

col1, col2 = st.columns(2)

with col1:
    engine_rpm       = st.number_input("Engine RPM",          min_value=0.0,  max_value=3000.0, value=800.0,  step=1.0)
    fuel_pressure    = st.number_input("Fuel Pressure (bar)", min_value=0.0,  max_value=30.0,   value=6.5,    step=0.1)
    lub_oil_temp     = st.number_input("Lub Oil Temp (°C)",   min_value=0.0,  max_value=200.0,  value=77.0,   step=0.1)

with col2:
    lub_oil_pressure = st.number_input("Lub Oil Pressure (bar)", min_value=0.0, max_value=10.0, value=3.3,  step=0.1)
    coolant_pressure = st.number_input("Coolant Pressure (bar)", min_value=0.0, max_value=10.0, value=2.3,  step=0.1)
    coolant_temp     = st.number_input("Coolant Temp (°C)",      min_value=0.0, max_value=200.0, value=80.0, step=0.1)

st.markdown("---")

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("Predict Engine Condition", use_container_width=True):
    input_data = pd.DataFrame([[
        engine_rpm,
        lub_oil_pressure,
        fuel_pressure,
        coolant_pressure,
        lub_oil_temp,
        coolant_temp
    ]], columns=[
        "Engine_RPM",
        "Lub_Oil_Pressure",
        "Fuel_Pressure",
        "Coolant_Pressure",
        "Lub_Oil_Temp",
        "Coolant_Temp"
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.markdown("### Prediction Result")

    if prediction == 1:
        st.error("Engine Requires Maintenance (Faulty)")
        st.metric("Confidence", f"{probability[1]*100:.2f}%")
    else:
        st.success("Engine is Operating Normally")
        st.metric("Confidence", f"{probability[0]*100:.2f}%")

    st.markdown("#### Input Summary")
    st.dataframe(input_data, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Model:** XGBoost Classifier | **Best F1-Score:** 0.7605 | **Recall (Faulty):** 84.37%")
st.markdown("**Project:** Predictive Maintenance | PGP-AIML Capstone | Great Learning")

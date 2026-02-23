import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ------------------------------
# Page configuration (must be first)
# ------------------------------
st.set_page_config(
    page_title="ET₀ Predictor – Smart Irrigation",
    page_icon="💧",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Custom CSS for a professional look
# ------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        text-align: center;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-label {
        font-size: 1.2rem;
        color: #4a5568;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2d3748;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Header
# ------------------------------
st.markdown(
    '<div class="main-header"><h1>💧 Evapotranspiration (ET₀) Predictor</h1><p>AI‑powered irrigation scheduling for Sri Lankan farmers</p></div>',
    unsafe_allow_html=True)


# ------------------------------
# Load model and explainer (cached)
# ------------------------------
@st.cache_resource
def load_models():
    try:
        model = joblib.load('et0_xgboost_model.pkl')
        explainer = joblib.load('shap_explainer.pkl')
    except FileNotFoundError:
        st.error(
            "Model files not found. Please ensure `et0_xgboost_model.pkl` and `shap_explainer.pkl` are in the same directory.")
        st.stop()
    return model, explainer


model, explainer = load_models()

feature_names = [
    'mean_temp', 'diurnal_range', 'shortwave_radiation_sum',
    'windspeed_10m_max', 'precipitation_sum', 'wet_hours',
    'latitude', 'longitude', 'elevation'
]

# ------------------------------
# Sidebar for inputs (professional layout)
# ------------------------------
st.sidebar.header("🌦️ Weather Parameters")
st.sidebar.markdown("Enter the daily weather measurements below:")

with st.sidebar.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        mean_temp = st.number_input(
            "🌡️ Mean Temp (°C)",
            min_value=0.0, max_value=40.0, value=25.0,
            step=0.1, format="%.1f"
        )
        diurnal_range = st.number_input(
            "📊 Diurnal Range (°C)",
            min_value=0.0, max_value=20.0, value=8.0,
            step=0.1, format="%.1f"
        )
        solar_rad = st.number_input(
            "☀️ Solar Radiation (MJ/m²)",
            min_value=0.0, max_value=30.0, value=18.0,
            step=0.1, format="%.1f"
        )
        wind_speed = st.number_input(
            "💨 Max Wind Speed (m/s)",
            min_value=0.0, max_value=15.0, value=4.0,
            step=0.1, format="%.1f"
        )
    with col2:
        precip = st.number_input(
            "🌧️ Precipitation (mm)",
            min_value=0.0, max_value=100.0, value=0.0,
            step=0.1, format="%.1f"
        )
        wet_hours = st.number_input(
            "⏱️ Precipitation Hours",
            min_value=0.0, max_value=24.0, value=0.0,
            step=0.5, format="%.1f"
        )
        lat = st.number_input(
            "🌍 Latitude",
            value=7.0,
            step=0.01, format="%.2f"
        )
        lon = st.number_input(
            "🌍 Longitude",
            value=80.0,
            step=0.01, format="%.2f"
        )
        elev = st.number_input(
            "⛰️ Elevation (m)",
            value=100.0,
            step=1.0, format="%.1f"
        )

    # Submit button – MUST be inside the form
    submitted = st.form_submit_button("🚀 Predict ET₀", use_container_width=True)

# ------------------------------
# Main panel – show prediction and explanation
# ------------------------------
if submitted:
    # Prepare input dataframe
    input_dict = {
        'mean_temp': mean_temp,
        'diurnal_range': diurnal_range,
        'shortwave_radiation_sum': solar_rad,
        'windspeed_10m_max': wind_speed,
        'precipitation_sum': precip,
        'wet_hours': wet_hours,
        'latitude': lat,
        'longitude': lon,
        'elevation': elev
    }
    input_df = pd.DataFrame([input_dict])

    # Predict
    prediction = model.predict(input_df)[0]

    # Display prediction in a nice card with explicit colors
    st.markdown(f"""
    <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 1rem 0; text-align: center;">
        <p style="color: #4a5568; font-size: 1.2rem; margin-bottom: 0.5rem;">Predicted ET₀</p>
        <p style="color: #2d3748; font-size: 2.5rem; font-weight: bold; margin: 0;">{prediction:.2f} mm/day</p>
    </div>
    """, unsafe_allow_html=True)

    # SHAP explanation
    st.subheader("🔍 Why this prediction?")
    shap_values = explainer.shap_values(input_df)

    # Create waterfall plot
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0].values,
            feature_names=feature_names
        ),
        show=False,
        max_display=9
    )
    st.pyplot(fig)
    plt.close()

    # Add interpretation
    # Custom styled info box
    st.markdown(f"""
    <div style="background-color: #e8f4fd; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #3182ce; margin: 1rem 0; color: #1a202c;">
        <h4 style="color: #2c5282; margin-top: 0;">📊 How to read this chart:</h4>
        <ul style="color: #2d3748; font-size: 1rem; line-height: 1.6; margin-bottom: 0;">
            <li><strong>Starting point (E[f(x)] = {explainer.expected_value:.3f})</strong> – average ET₀ across all locations</li>
            <li><span style="color: #e53e3e; font-weight: bold;">🔴 Red bars</span> → features that increase ET₀ (push prediction higher)</li>
            <li><span style="color: #3182ce; font-weight: bold;">🔵 Blue bars</span> → features that decrease ET₀ (push prediction lower)</li>
            <li><strong>Final value (f(x) = {prediction:.3f})</strong> – your predicted ET₀</li>
        </ul>
        <p style="color: #4a5568; margin-top: 0.5rem; margin-bottom: 0; font-style: italic;">The longer the bar, the greater the impact of that feature.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Placeholder instructions
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: #333333;">
        <h3 style="color: #2d3748;">Enter weather data in the sidebar and click <strong>Predict ET₀</strong></h3>
        <p style="color: #4a5568;">This tool uses a machine learning model (XGBoost) trained on 13 years of daily weather data from 30 cities in Sri Lanka.</p>
        <p style="color: #4a5568;">The model estimates reference evapotranspiration (ET₀) – the amount of water lost from soil and plants – helping farmers schedule irrigation efficiently.</p>
        <p style="color: #4a5568;">After prediction, you'll see a SHAP explanation showing which weather factors most influenced the result.</p>
    </div>
    """, unsafe_allow_html=True)
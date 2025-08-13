import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import numpy as np
import re

# --- CONFIG & LOADING ---
st.set_page_config(page_title="Growth Analyzer Pro", layout="wide")

@st.cache_resource
def load_models():
    """Load all pre-trained models from disk."""
    models = {
        'wfa': joblib.load('models/wfa_interpolators.joblib'),
        'lhfa': joblib.load('models/lhfa_interpolators.joblib'),
        'wfl': joblib.load('models/wfl_interpolators.joblib'),
        'wfh': joblib.load('models/wfh_interpolators.joblib'),
        'predictor': joblib.load('models/growth_predictor.joblib')
    }
    return models

# --- HELPER FUNCTIONS ---
def get_percentile(metric_value, measurement, gender, model_key):
    """Calculates the percentile for a given measurement using the appropriate model."""
    model = models[model_key][gender]
    p_cols_names = list(model.keys())
    
    # Dynamically create the numeric percentile values from the column names
    p_labels = [float(re.findall(r'(\d+\.?\d*)', p_col)[0]) for p_col in p_cols_names]
    
    # Get the percentile values from the interpolation functions for the given metric
    p_values = [func(metric_value) for func in model.values()]

    # Sort by p_values to ensure correct interpolation
    sorted_pairs = sorted(zip(p_values, p_labels))
    p_values_sorted, p_labels_sorted = zip(*sorted_pairs)
    
    # Interpolate the child's measurement against the standard percentiles
    percentile = np.interp(measurement, p_values_sorted, p_labels_sorted)
    return round(percentile, 2)

def get_classification(percentile):
    if percentile < 3: return "At Risk / Low"
    if 3 <= percentile <= 97: return "Normal"
    return "At Risk / High"

# --- UI LAYOUT ---
st.title("🩺 Comprehensive Neonatal & Infant Growth Analyzer")
st.markdown("Based on WHO Child Growth Standards. Enter details in the sidebar to begin.")

# Load models
models = load_models()

# Sidebar for Inputs
st.sidebar.header("Child's Information")
gender = st.sidebar.radio("Gender", ("Male", "Female"), key="gender")
age_days = st.sidebar.number_input("Age (in days)", min_value=0, max_value=1856, value=365, step=1, key="age_days")
weight = st.sidebar.number_input("Weight (kg)", 1.0, 40.0, 9.6, 0.1, key="weight")
height = st.sidebar.number_input("Length/Height (cm)", 40.0, 130.0, 75.0, 0.5, key="height")

age_months = age_days / 30.4375

# --- ANALYSIS & DISPLAY ---
st.header("Growth Analysis Results")

tab1, tab2, tab3, tab4 = st.tabs(["Weight for Age", "Length/Height for Age", "Weight for Length/Height", "Future Prediction"])

with tab1:
    st.subheader("Weight-for-Age (WFA)")
    wfa_perc = get_percentile(age_months, weight, gender, 'wfa')
    wfa_class = get_classification(wfa_perc)
    st.metric(label="WFA Percentile", value=f"{wfa_perc}%", delta=wfa_class)
    st.write("This indicates if a child's weight is appropriate for their age. Low values may suggest the child is underweight.")

with tab2:
    st.subheader("Length/Height-for-Age (LHFA)")
    lhfa_perc = get_percentile(age_months, height, gender, 'lhfa')
    lhfa_class = get_classification(lhfa_perc)
    st.metric(label="LHFA Percentile", value=f"{lhfa_perc}%", delta=lhfa_class)
    st.write("This is an indicator of nutritional status over time. Low values can indicate stunting.")
    
with tab3:
    st.subheader("Weight-for-Length/Height (WFLH)")
    # Decide which model to use based on age (under 2 years vs 2+ years)
    if age_days < 730:
        wflh_model_key = 'wfl'
        wflh_metric_val = height
        st.info("Using Weight-for-Length data (child is under 2 years old).")
    else:
        wflh_model_key = 'wfh'
        wflh_metric_val = height
        st.info("Using Weight-for-Height data (child is 2 years or older).")

    wflh_perc = get_percentile(wflh_metric_val, weight, gender, wflh_model_key)
    wflh_class = get_classification(wflh_perc)
    st.metric(label="WFLH Percentile", value=f"{wflh_perc}%", delta=wflh_class)
    st.write("This is a key indicator of acute malnutrition (wasting) or being overweight, as it assesses weight relative to body length/height.")

with tab4:
    st.subheader("Illustrative Future Growth Prediction")
    predicted_weight = models['predictor'].predict([[age_months, weight]])[0]
    st.metric(label=f"Predicted Weight at {age_months + 6:.1f} Months", value=f"{predicted_weight:.2f} kg")
    st.caption("Note: This prediction is for illustrative purposes only and is based on a simplified model.")
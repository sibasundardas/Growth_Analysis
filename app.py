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

@st.cache_data
def load_charting_data():
    """Load the raw data needed for plotting the background curves."""
    data_frames = {
        'wfa': pd.concat([pd.read_excel('data/wfa-boys-percentiles-expanded-tables.xlsx', header=0).assign(Sex='Male'), pd.read_excel('data/wfa-girls-percentiles-expanded-tables.xlsx', header=0).assign(Sex='Female')]),
        'lhfa': pd.concat([pd.read_excel('data/lhfa-boys-percentiles-expanded-tables.xlsx', header=0).assign(Sex='Male'), pd.read_excel('data/lhfa-girls-percentiles-expanded-tables.xlsx', header=0).assign(Sex='Female')]),
        'wfl': pd.concat([pd.read_excel('data/wfl-boys-percentiles-expanded-tables.xlsx', header=0).assign(Sex='Male'), pd.read_excel('data/wfl-girls-percentiles-expanded-tables.xlsx', header=0).assign(Sex='Female')]),
        'wfh': pd.concat([pd.read_excel('data/wfh-boys-percentiles-expanded-tables.xlsx', header=0).assign(Sex='Male'), pd.read_excel('data/wfh-girls-percentiles-expanded-tables.xlsx', header=0).assign(Sex='Female')])
    }
    # Add Agemos column for age-based charts
    for key in ['wfa', 'lhfa']:
        primary_col = data_frames[key].columns[0]
        data_frames[key]['Agemos'] = data_frames[key][primary_col] / 30.4375
    return data_frames

# --- HELPER FUNCTIONS ---
def get_percentile(metric_value, measurement, gender, model_key):
    model = models[model_key][gender]
    p_cols_names = list(model.keys())
    p_labels = [float(re.findall(r'(\d+\.?\d*)', p_col)[0]) for p_col in p_cols_names]
    p_values = [func(metric_value) for func in model.values()]
    sorted_pairs = sorted(zip(p_values, p_labels))
    p_values_sorted, p_labels_sorted = zip(*sorted_pairs)
    percentile = np.interp(measurement, p_values_sorted, p_labels_sorted)
    return round(percentile, 2)

# --- UPDATED CLASSIFICATION FUNCTIONS ---
def get_classification_details(percentile):
    if percentile < 3: return "High Risk / Severely Underweight or Stunted", "error"
    if 3 <= percentile < 15: return "At Risk / Underweight", "warning"
    if 15 <= percentile <= 85: return "Normal Growth", "success"
    if 85 < percentile <= 97: return "At Risk / Overweight", "warning"
    return "High Risk / Obese", "error"

def display_classification(percentile):
    message, type = get_classification_details(percentile)
    if type == "success": st.success(message)
    elif type == "warning": st.warning(message)
    elif type == "error": st.error(message)

def plot_gauge_chart(percentile_value, title_text):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=percentile_value,
        title={'text': title_text, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"}, 'bgcolor': "white",
            'steps': [
                {'range': [0, 15], 'color': 'rgba(255, 165, 0, 0.7)'},
                {'range': [15, 85], 'color': 'rgba(0, 128, 0, 0.7)'},
                {'range': [85, 100], 'color': 'rgba(255, 165, 0, 0.7)'},
            ],
             'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.9, 'value': 97},
        }))
    fig.update_layout(height=250, margin={'t':0, 'b':0, 'l':0, 'r':0})
    return fig

def plot_growth_curve(df, x_metric, x_val, y_val, gender, title):
    df_gender = df[df['Sex'] == gender].sort_values(by=x_metric)
    fig = go.Figure()
    p_cols_all = [col for col in df.columns if isinstance(col, str) and col.startswith('P')]
    p_vals_all = [float(re.findall(r'(\d+\.?\d*)', p_col)[0]) for p_col in p_cols_all]
    
    p3_col = p_cols_all[(np.abs(np.array(p_vals_all) - 3)).argmin()]
    p50_col = p_cols_all[(np.abs(np.array(p_vals_all) - 50)).argmin()]
    p97_col = p_cols_all[(np.abs(np.array(p_vals_all) - 97)).argmin()]

    fig.add_trace(go.Scatter(x=df_gender[x_metric], y=df_gender[p97_col], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df_gender[x_metric], y=df_gender[p3_col], mode='lines', line=dict(width=0), name='Normal Range', fill='tonexty', fillcolor='rgba(0,128,0,0.2)'))
    fig.add_trace(go.Scatter(x=df_gender[x_metric], y=df_gender[p50_col], mode='lines', name=f'{p50_col} (Median)', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=[x_val], y=[y_val], mode='markers', name='Child', marker=dict(color='red', size=12, symbol='star')))
    fig.update_layout(title=f'<b>{title} ({gender})</b>', xaxis_title=x_metric.replace("Agemos", "Age (Months)"), yaxis_title="Measurement", template='plotly_white')
    return fig

# --- NEW PREDICTION PLOT FUNCTION ---
def plot_prediction_curve(df, gender, current_age, current_weight, future_age, future_weight):
    df_gender = df[df['Sex'] == gender].sort_values(by='Agemos')
    fig = go.Figure()

    p3_col = df.columns[df.columns.str.contains('P3', na=False)][0]
    p97_col = df.columns[df.columns.str.contains('P97', na=False)][0]

    fig.add_trace(go.Scatter(x=df_gender['Agemos'], y=df_gender[p97_col], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df_gender['Agemos'], y=df_gender[p3_col], mode='lines', line=dict(width=0), name='Normal Range', fill='tonexty', fillcolor='rgba(0,128,0,0.2)'))
    
    # Plot current and predicted points
    fig.add_trace(go.Scatter(x=[current_age, future_age], y=[current_weight, future_weight], mode='lines+markers', name='Predicted Path',
                             line=dict(color='purple', dash='dash'), marker=dict(color='red', size=10, symbol='star')))
    
    fig.update_layout(title=f'<b>Predicted Growth Trajectory ({gender})</b>', xaxis_title="Age (Months)", yaxis_title="Weight (kg)", template='plotly_white')
    return fig

# --- UI LAYOUT ---
st.title("🩺 Comprehensive Neonatal & Infant Growth Analyzer")
models = load_models()
chart_data = load_charting_data()

st.sidebar.header("Child's Information")
gender = st.sidebar.radio("Gender", ("Male", "Female"))
age_days = st.sidebar.number_input("Age (in days)", min_value=0, max_value=1856, value=365, step=1)
weight = st.sidebar.number_input("Weight (kg)", 1.0, 40.0, 9.6, 0.1)
height = st.sidebar.number_input("Length/Height (cm)", 40.0, 130.0, 75.0, 0.5)

age_months = age_days / 30.4375

st.header("Growth Analysis Results")
tab1, tab2, tab3, tab4 = st.tabs(["Weight for Age", "Length/Height for Age", "Weight for Length/Height", "Future Prediction"])

with tab1:
    st.subheader("Weight-for-Age (WFA)")
    wfa_perc = get_percentile(age_months, weight, gender, 'wfa')
    display_classification(wfa_perc)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="WFA Percentile", value=f"{wfa_perc}%")
        st.plotly_chart(plot_gauge_chart(wfa_perc, "WFA Percentile"), use_container_width=True)
    with col2:
        st.plotly_chart(plot_growth_curve(chart_data['wfa'], 'Agemos', age_months, weight, gender, "Weight-for-Age Growth Curve"), use_container_width=True)

with tab2:
    st.subheader("Length/Height-for-Age (LHFA)")
    lhfa_perc = get_percentile(age_months, height, gender, 'lhfa')
    display_classification(lhfa_perc)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="LHFA Percentile", value=f"{lhfa_perc}%")
        st.plotly_chart(plot_gauge_chart(lhfa_perc, "LHFA Percentile"), use_container_width=True)
    with col2:
        st.plotly_chart(plot_growth_curve(chart_data['lhfa'], 'Agemos', age_months, height, gender, "Length/Height-for-Age Curve"), use_container_width=True)

with tab3:
    st.subheader("Weight-for-Length/Height (WFLH)")
    if age_days < 730:
        model_key, metric_val, df_key = 'wfl', height, 'wfl'
    else:
        model_key, metric_val, df_key = 'wfh', height, 'wfh'

    wflh_perc = get_percentile(metric_val, weight, gender, model_key)
    display_classification(wflh_perc)
    x_metric_col = chart_data[df_key].columns[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="WFLH Percentile", value=f"{wflh_perc}%")
        st.plotly_chart(plot_gauge_chart(wflh_perc, "WFLH Percentile"), use_container_width=True)
    with col2:
        st.plotly_chart(plot_growth_curve(chart_data[df_key], x_metric_col, height, weight, gender, "Weight-for-Length/Height Curve"), use_container_width=True)

with tab4:
    st.subheader("Future Growth Prediction")
    predicted_weight = models['predictor'].predict([[age_months, weight]])[0]
    future_age_months = age_months + 6
    
    col1, col2 = st.columns([1,2])
    with col1:
        st.metric(label=f"Predicted Weight at {future_age_months:.1f} Months", value=f"{predicted_weight:.2f} kg")
        st.caption("This prediction shows an estimated growth trajectory based on the child's current percentile.")
    with col2:
        st.plotly_chart(plot_prediction_curve(chart_data['wfa'], gender, age_months, weight, future_age_months, predicted_weight), use_container_width=True)